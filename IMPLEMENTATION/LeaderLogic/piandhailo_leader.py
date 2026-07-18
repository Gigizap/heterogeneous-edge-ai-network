"""
LeaderLogic/piandhailo_leader.py

Leader for a Pi + Hailo NPU device.

Owns ALL leader/network management:
  * builds the single shared LeaderNetwork (tool registry + dispatcher)
  * wires it into the leader transport + discovery
  * constructs the inference workflows ONCE (sharing the network + bot)
  * handles /changehost mode cycling and /status
  * keeps per-chat history for main.py's BackupManager

The Workflow classes in complete_workflow.py are PURE INFERENCE strategies:
they dispatch a tool / craft an answer using the shared network, and never
create transports or networks of their own.

Modes (cycled in-chat with /changehost):
    workflow1 — Hailo NPU native .hef (qwen3:1.7b): dispatch + answer
    workflow2 — CPU Ollama (qwen3:1.7b, OpenAI HTTP): dispatch + answer
    workflow3 — FunctionGemma (CPU Ollama) dispatch + Hailo (.hef) answer

Exposed for main.py:
    MODEL_NAME, MODEL_PARAMS_B
    boot(cfg, agent_id, transport, discovery, bot) -> GenericWorkflow
"""

import logging
import json
import threading
from pathlib import Path
from typing import Dict, List

from LeaderLogic.complete_workflow import LLM, Workflow1, Workflow2, Workflow3
from LeaderLogic.leader_network import make_network
from LeaderLogic.ollama_loader import start_and_wait

logger = logging.getLogger(__name__)

_QWEN_LABEL    = "Qwen3-1.7B"
_QWEN_PARAMS_B = 1.7

MODEL_NAME     = "unknown"
MODEL_PARAMS_B = 0.0

_CHAT_PATH   = "/v1/chat/completions"
# device-specific inference config (Ollama backends + hailo hef), next to this file
_DEVICE_CONFIG = Path(__file__).parent / "raspberry_config.json"


# ── boot ──────────────────────────────────────────────────────────────────────

def boot(cfg: dict, agent_id: str, transport, discovery, bot, profile: dict = None) -> "GenericWorkflow":
    global MODEL_NAME, MODEL_PARAMS_B
    MODEL_NAME, MODEL_PARAMS_B = _QWEN_LABEL, _QWEN_PARAMS_B

    # single shared network object — keeps the tool registry current
    # (pull-on-join over discovery) and dispatches picked tools to their owners.
    net = make_network(transport, discovery,
                       reply_timeout=cfg.get("timeouts", {}).get("replies", 3.0))

    # device-specific inference config lives in raspberry_config.json (next to
    # this file); cfg is used only for timeouts.
    dev_cfg = json.loads(_DEVICE_CONFIG.read_text())

    # device-specific LLM endpoints (config stores the base host + model)
    _cpu_cfg  = dev_cfg.get("llm_cpu", {})
    _tool_cfg = dev_cfg.get("llm_cpu_tool_calling", {})

    _cpu_host   = _cpu_cfg.get("host",  "http://127.0.0.1:11434")
    _tool_host  = _tool_cfg.get("host", "http://127.0.0.1:11434")
    _cpu_model  = _cpu_cfg.get("model",  "qwen3:1.7b")
    _tool_model = _tool_cfg.get("model", "functiongemma")

    # Ensure Ollama is up and both CPU models are pulled. start_and_wait only
    # touches disk (lists/pulls); it doesn't load models into RAM — that happens
    # lazily at inference time. Idempotent: no-ops if the server already runs.
    start_and_wait(_cpu_model,  host=_cpu_host)
    start_and_wait(_tool_model, host=_tool_host)

    llm_cpu = LLM(
        endpoint = _cpu_host + _CHAT_PATH,
        api_key  = _cpu_cfg.get("api_key", ""),
        model    = _cpu_model,
    )
    llm_tool = LLM(
        endpoint = _tool_host + _CHAT_PATH,
        api_key  = _tool_cfg.get("api_key", ""),
        model    = _tool_model,
    )

    hef_path = dev_cfg.get("hailo", {}).get("hef_path")

    # build workflows ONCE, sharing net + bot. workflow1 owns the Hailo
    # LLM; workflow3 reuses it for its answer step so the NPU loads only once.
    wf1 = Workflow1(llm_hef_path=hef_path, net=net, bot=bot)
    wf2 = Workflow2(llm=llm_cpu,           net=net, bot=bot)
    wf3 = Workflow3(llm_dispatch=llm_tool, hailo_answerer=wf1,
                    net=net, bot=bot)

    logger.info("ready — modes: workflow1 (Hailo), "
                "workflow2 (CPU qwen3:1.7b), workflow3 (FunctionGemma+Hailo)")
    return GenericWorkflow(
        workflows=[("workflow1", wf1), ("workflow2", wf2), ("workflow3", wf3)],
        bot=bot,
    )


# ── GenericWorkflow (management layer) ─────────────────────────────────────────

class GenericWorkflow:
    """Management wrapper main.py talks to: routes messages to the active
    workflow, cycles modes on /changehost, and keeps history for backup."""

    def __init__(self, workflows, bot):
        self._workflows = workflows          # list[(name, BaseWorkflow)]
        self._bot       = bot

        self._mode_index = 0
        self._mode_lock  = threading.Lock()

        self._histories: Dict[int, List[dict]] = {}
        self._history_lock = threading.Lock()

        self._hailo_active = False           # activate the NPU once, on first use

    # ── mode helpers ───────────────────────────────────────────────────────────

    def _active(self):
        with self._mode_lock:
            return self._workflows[self._mode_index]

    def _cycle_mode(self) -> str:
        with self._mode_lock:
            self._mode_index = (self._mode_index + 1) % len(self._workflows)
            return self._workflows[self._mode_index][0]

    def _ensure_hailo(self):
        """Activate the shared Hailo NPU LLM (owned by workflow1) once."""
        if self._hailo_active:
            return
        wf1 = self._workflows[0][1]          # workflow1 owns the Hailo LLM
        wf1.activate()
        self._hailo_active = True

    # ── main.py interface ───────────────────────────────────────────────────────

    def handle(self, chat_id: int, text: str) -> None:
        if text == "/changehost":
            self._bot.send_message(chat_id, f"Switched to {self._cycle_mode()}")
            return

        name, wf = self._active()
        if name in ("workflow1", "workflow3"):
            self._ensure_hailo()

        # Run the pipeline here so the leader gets the answer string for history
        # and delivery — same control flow as the original self-contained leader.
        self._bot.start_typing_indicator(chat_id)
        try:
            reply = wf._run_pipeline(text)
        except Exception as e:
            logger.exception("pipeline failed: %s", e)
            reply = "Sorry, something went wrong."
        finally:
            self._bot.stop_typing_indicator(chat_id)

        self._append_history(chat_id, text, reply)
        self._bot.send_message(chat_id, reply)

    def status_extra(self) -> str:
        """Extra /status line: which of the cycled workflows is currently active."""
        name, _ = self._active()
        return f"\n🔀 Active workflow: {name}"

    # ── history (used by BackupManager) ─────────────────────────────────────────

    def get_history(self, chat_id: int) -> List[dict]:
        with self._history_lock:
            return list(self._histories.get(chat_id, []))

    def load_history(self, chat_id: int, messages: List[dict]) -> None:
        with self._history_lock:
            self._histories[chat_id] = list(messages)

    def _append_history(self, chat_id: int, user_text: str, reply: str) -> None:
        with self._history_lock:
            h = self._histories.setdefault(chat_id, [])
            h.append({"role": "user",      "content": user_text})
            h.append({"role": "assistant", "content": reply})