"""
UNUSED AS OF RN

Generic leader - any device without a specific hardware preset.
Uses llama.cpp (via llama-cpp-python) and automatically selects
(and downloads if missing) the largest GGUF that fits in ~60% of RAM.

Exposed module-level constants (read by main.py for capability broadcast):
    MODEL_NAME     : str
    MODEL_PARAMS_B : float

Exposed function:
    boot(cfg, agent_id, transport, discovery, bot) → GenericWorkflow
"""

import os
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from LeaderLogic.complete_workflow import BaseWorkflow
from LeaderLogic.network_collector import NetworkCollector
from utils import available_ram_gb

log = logging.getLogger(__name__)

# ── model catalogue ───────────────────────────────────────────────────────────
# (min_ram_gb, params_b, hf_repo, filename, label) - largest first

_MODELS = [
    ( 1.5,  1.5, "unsloth/granite-4.0-350m-GGUF",
            "granite-4.0-350m-Q4_K_M.gguf", "granite:350m"),
]

_RAM_FRACTION        = 0.60
_MODELS_DIR          = Path(__file__).parent.parent / "models"
_HISTORY_THRESHOLD_B = 7.0

_DISPATCH_SYSTEM = (
    "You are a smart-home agent dispatcher. "
    "Call exactly ONE tool that best matches the user's intent. "
    "Extract any names or object targets precisely from the user's message. "
    "If nothing matches, reply normally without calling any tool."
)
_ANSWER_SYSTEM = (
    "You are a helpful assistant for a smart home agent network. "
    "The network has already executed a command and returned real data. "
    "Translate that raw result into a short, friendly reply for the user. "
    "Always reply in the same language the user wrote in."
)

# Filled by boot(), read by main.py
MODEL_NAME     = "unknown"
MODEL_PARAMS_B = 0.0


# ── boot ──────────────────────────────────────────────────────────────────────

def boot(cfg: dict, agent_id: str, transport, discovery, bot, profile=None) -> "GenericWorkflow":
    global MODEL_NAME, MODEL_PARAMS_B

    entry                  = _select_model(available_ram_gb())
    min_ram, params_b, repo, filename, label = entry
    MODEL_NAME, MODEL_PARAMS_B = label, params_b

    model_path = _MODELS_DIR / filename
    if not model_path.exists():
        log.info("downloading %s …", filename)
        _download_model(repo, filename, model_path)
    else:
        log.info("using cached %s", filename)

    log.info("loading %s (%sB params)", label, params_b)
    from llama_cpp import Llama
    llm = Llama(model_path=str(model_path), n_ctx=4096,
                n_threads=os.cpu_count() or 4, verbose=False)

    collector = NetworkCollector(
        broadcaster    = transport.broadcast_sync,
        skills_timeout = cfg.get("timeouts", {}).get("fetchskills", 1.5),
        reply_timeout  = cfg.get("timeouts", {}).get("replies",     3.0),
    )

    # Wire collector into the leader transport so it sees skill
    # replies and command responses from sensing agents
    prev_handler = transport.on_message
    def _handler(msg: dict):
        collector.handle_incoming(msg)
        prev_handler(msg)
    transport.on_message = _handler

    log.info("ready - %s", label)
    return GenericWorkflow(llm=llm, params_b=params_b, collector=collector, bot=bot)


# ── GenericWorkflow ───────────────────────────────────────────────────────────

class GenericWorkflow(BaseWorkflow):

    fetch_skills_from_network = True
    broadcast_format          = "json"

    def __init__(self, llm, params_b: float, collector: NetworkCollector, bot=None):
        super().__init__(collector=collector, bot=bot)
        self._llm           = llm
        self._params_b      = params_b
        self._histories:    Dict[int, List[dict]] = {}
        self._history_lock  = threading.Lock()
        self._current_chat_id: Optional[int] = None

    # ── main.py interface ─────────────────────────────────────────────────────

    def handle(self, chat_id: int, text: str) -> None:
        self._current_chat_id = chat_id
        self.run(chat_id, text)

    # ── BaseWorkflow hooks ────────────────────────────────────────────────────

    def _dispatch(self, user_text: str, tool_defs: List[Dict]) -> Tuple[Optional[str], Optional[dict]]:
        import json, re as _re
        chat_id  = self._current_chat_id
        history  = self._get_history_for_dispatch(chat_id)
        messages = [{"role": "system", "content": _DISPATCH_SYSTEM},
                    *history,
                    {"role": "user", "content": user_text}]

        lc_tools = []
        for td in tool_defs:
            fn = td.get("function", td)  # handle both nested and flat format
            lc_tools.append({"type": "function", "function": {
                "name":        fn["name"],
                "description": fn.get("description", ""),
                "parameters":  fn.get("parameters", {}),
            }})

        try:
            resp = self._llm.create_chat_completion(
                messages=messages, tools=lc_tools, tool_choice="auto",
                max_tokens=256, temperature=0.0, repeat_penalty=1.1,
            )
            log.debug("chat completion created with tools")
            log.debug("available tools: %s", lc_tools)
        except Exception as e:
            log.error("dispatch error: %s", e)
            return None, None

        msg       = resp["choices"][0]["message"]
        raw_text  = msg.get("content", "") or ""
        log.debug("raw dispatch: %r", raw_text)

        # path 1: structured tool_calls
        if msg.get("tool_calls"):
            log.debug("tool call: %s", msg.get("tool_calls"))
            call = msg["tool_calls"][0]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}
            return call["function"]["name"], args

        # path 2: <tool_call>{...}</tool_call>
        m = _re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw_text, _re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if obj.get("name"):
                    return obj["name"], obj.get("arguments") or {}
            except Exception:
                pass

        # path 3: bare JSON with "name" key
        m = _re.search(r"\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\}", raw_text, _re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if obj.get("name"):
                    return obj["name"], obj.get("arguments") or {}
            except Exception:
                pass

        self._append_history(chat_id, user_text, raw_text)
        return None, None

    def _answer(self, user_text: str, command: str, network_reply: str) -> str:
        chat_id  = self._current_chat_id
        messages = [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user",   "content": (
                f"User request: {user_text}\n"
                f"Command executed: {command}\n\n"
                f"RESULT (you MUST include the following in your reply) \n"
                f"{network_reply}\n"
            )},
        ]
        try:
            resp  = self._llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.0, repeat_penalty=1.1)
            reply = resp["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"[error generating answer] {e}"
        self._append_history(chat_id, user_text, reply)
        return reply

    def _no_tool_answer(self, user_text: str) -> str:
        if self._params_b < _HISTORY_THRESHOLD_B:
            return "I'm not sure how to help with that using the available commands."
        chat_id  = self._current_chat_id
        messages = [{"role": "system", "content": _ANSWER_SYSTEM},
                    *self._get_history_for_dispatch(chat_id),
                    {"role": "user", "content": user_text}]
        try:
            resp  = self._llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.7)
            reply = resp["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"[error] {e}"
        self._append_history(chat_id, user_text, reply)
        return reply

    # ── history (used by backup_manager) ─────────────────────────────────────

    def get_history(self, chat_id: int) -> List[dict]:
        with self._history_lock:
            return list(self._histories.get(chat_id, []))

    def load_history(self, chat_id: int, messages: List[dict]) -> None:
        with self._history_lock:
            self._histories[chat_id] = list(messages)

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_history_for_dispatch(self, chat_id: int) -> List[dict]:
        if self._params_b < _HISTORY_THRESHOLD_B:
            return []
        with self._history_lock:
            return list(self._histories.get(chat_id, []))

    def _append_history(self, chat_id: int, user_text: str, reply: str) -> None:
        if self._params_b < _HISTORY_THRESHOLD_B:
            return
        with self._history_lock:
            h = self._histories.setdefault(chat_id, [])
            h.append({"role": "user",      "content": user_text})
            h.append({"role": "assistant", "content": reply})


# ── helpers ───────────────────────────────────────────────────────────────────

def _select_model(ram_gb: float):
    usable = ram_gb * _RAM_FRACTION
    for entry in _MODELS:
        if usable >= entry[0]:
            log.info("RAM=%.1fGB → %s", ram_gb, entry[4])
            return entry
    log.info("low RAM - using smallest model")
    return _MODELS[-1]

def _download_model(repo: str, filename: str, dest: Path):
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(_MODELS_DIR))
        Path(tmp).rename(dest)
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename}: {e}") from e