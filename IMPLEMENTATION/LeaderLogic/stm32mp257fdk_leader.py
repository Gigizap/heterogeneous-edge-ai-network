"""
LeaderLogic/generic_leader.py

Generic leader — any device without a specific hardware preset.
Uses llama.cpp (via llama-cpp-python). The model is chosen from the device
SCORE (when LLM_detection_with_score is True):

    score >= 170          -> qwen3:1.7b
    score < 170           -> functiongemma:270m

When LLM_detection_with_score is False, always use functiongemma:270m.

FunctionGemma is loaded through the functiongemma_simple_handler chat format
and driven with the developer-role prompt; Qwen3 is loaded through the
qwen3_simple_handler chat format and uses the standard smart-home
dispatcher/answer prompts.

Exposed module-level constants (read by main.py for capability broadcast):
    MODEL_NAME     : str
    MODEL_PARAMS_B : float

Exposed function:
    boot(cfg, agent_id, transport, discovery, bot) -> GenericWorkflow
"""

import json
import os
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

log = logging.getLogger(__name__)

from LeaderLogic.complete_workflow import BaseWorkflow
from LeaderLogic.leader_network import make_network

# Optional FunctionGemma chat handler for llama.cpp.
try:
    import LeaderLogic.functiongemma_simple_handler  # noqa: F401
    HAS_FG = True
except ImportError:
    HAS_FG = False

# Optional Qwen3 chat handler for llama.cpp.
try:
    import LeaderLogic.qwen3_handler  # noqa: F401
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

# ── model selection ─────────────────────────────────────────────────────────
# When True, pick the model from the device score; when False, always functiongemma.
LLM_detection_with_score = False

# Conversation history is disabled: all models we run are < 7B and behave better
# stateless (matches the benchmark, which uses no history). Flip to True later if
# you switch to a larger model that benefits from multi-turn context.
ENABLE_HISTORY = False

# (params_b, hf_repo, filename, label, is_functiongemma)
_QWEN = (1.7,  "unsloth/Qwen3-1.7B-GGUF",
         "Qwen3-1.7B-Q4_K_M.gguf",          "qwen3:1.7b",        False)
_FUNCTIONGEMMA = (0.27, "unsloth/functiongemma-270m-it-GGUF",
                  "functiongemma-270m-it-Q4_K_M.gguf", "functiongemma:270m", True)

# Score thresholds (inclusive lower bounds).
_SCORE_QWEN    = 170

_MODELS_DIR = Path(__file__).parent.parent / "models"

# Standard prompts (used by qwen3).
_DISPATCH_SYSTEM = (
    "You are a smart-home agent dispatcher. "
    "Call exactly ONE tool that best matches the user's intent. "
    "Extract any names or object targets precisely from the user's message. "
    "If nothing matches, reply normally without calling any tool."
)
# FunctionGemma dispatch prompt (supplied via the developer role).
_DISPATCH_SYSTEM_FG = (
    "You are a smart-home agent dispatcher. "
    "You are a model that can do function calling with the following functions"
)
_ANSWER_SYSTEM = (
    "You are a smart home assistant. Turn the data into a short, friendly reply to the user. "
    "Reply in English. Summarize the reply, "
    "MENTION ALL THE IMPORTANT DATA RECEIVED FROM THE TOOL "
    "AND THE AVAILABLE SENSORS WHEN REPLYING."
)

# Filled by boot(), read by main.py
MODEL_NAME     = "unknown"
MODEL_PARAMS_B = 0.0


# ── boot ──────────────────────────────────────────────────────────────────────

def boot(cfg: dict, agent_id: str, transport, discovery, bot, profile: dict = None) -> "GenericWorkflow":
    global MODEL_NAME, MODEL_PARAMS_B

    score = (profile or {}).get("score", 0)
    params_b, repo, filename, label, is_fg = _select_model(score)
    MODEL_NAME, MODEL_PARAMS_B = label, params_b

    model_path = _MODELS_DIR / filename
    if not model_path.exists():
        log.info("downloading %s …", filename)
        _download_model(repo, filename, model_path)
    else:
        log.info("using cached %s", filename)

    log.info("loading %s (%sB params)", label, params_b)
    from llama_cpp import Llama
    if is_fg and HAS_FG:
        llm = Llama(model_path=str(model_path), n_ctx=4096,
                    chat_format="functiongemma",
                    n_threads=os.cpu_count() or 4, verbose=False)
    elif not is_fg and HAS_QWEN:
        llm = Llama(model_path=str(model_path), n_ctx=4096,
                    chat_format="qwen3",
                    n_threads=os.cpu_count() or 4, verbose=False)
    else:
        if is_fg and not HAS_FG:
            log.warning("functiongemma_simple_handler not installed — tool calling will likely fail")
        if not is_fg and not HAS_QWEN:
            log.warning("qwen3_simple_handler not installed — tool calling will likely fail")
        llm = Llama(model_path=str(model_path), n_ctx=4096,
                    n_threads=os.cpu_count() or 4, verbose=False)

    # LeaderNetwork keeps the tool registry current (pull-on-join over
    # discovery) and dispatches picked tools to the agents that own them.
    net = make_network(transport, discovery,
                       reply_timeout=cfg.get("timeouts", {}).get("replies", 3.0))

    log.info("ready — %s", label)
    return GenericWorkflow(llm=llm, params_b=params_b, is_fg=is_fg,
                           net=net, bot=bot)


# ── GenericWorkflow ───────────────────────────────────────────────────────────

class GenericWorkflow(BaseWorkflow):

    def __init__(self, llm, params_b: float, is_fg: bool,
                 net, bot=None):
        super().__init__(net=net, bot=bot)
        self._llm           = llm
        self._params_b      = params_b
        self._is_fg         = is_fg
        self._histories:    Dict[int, List[dict]] = {}
        self._history_lock  = threading.Lock()
        self._current_chat_id: Optional[int] = None
        # the following 3 could be implemented more cleanly...
        self._reply_if_no_tool = None
        self._last_tool_call = None
        self.last_tool_defs = {}

    # ── main.py interface ─────────────────────────────────────────────────────

    def handle(self, chat_id: int, text: str) -> None:
        self._current_chat_id = chat_id
        self.run(chat_id, text)

    # ── BaseWorkflow hooks ────────────────────────────────────────────────────

    def _dispatch(self, user_text: str, tool_defs: List[Dict]) -> Tuple[Optional[str], Optional[dict]]:

        #can be eliminated, used only in _answer to retrieve the description of the tool for further context in the answer generation
        self.last_tool_defs = {td.get("function", td)["name"]: td.get("function", td)
                for td in tool_defs}

        chat_id = self._current_chat_id
        history = self._get_history_for_dispatch(chat_id)

        # FunctionGemma uses the developer role + its own prompt; others use
        # the system role + the smart-home dispatcher prompt.
        if self._is_fg:
            messages = [{"role": "developer", "content": _DISPATCH_SYSTEM_FG},
                        *history,
                        {"role": "user", "content": user_text}]
        else:
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

        kw = dict(messages=messages, tools=lc_tools,
                  max_tokens=256, temperature=0.0,
                  seed=42, repeat_penalty=1.1)
        if not self._is_fg:
            kw["tool_choice"] = "auto"

        try:
            resp = self._llm.create_chat_completion(**kw)
        except Exception as e:
            log.error("dispatch error: %s", e)
            return None, None

        msg      = resp["choices"][0]["message"]
        self._reply_if_no_tool = msg
        log.debug("raw dispatch: %r", msg)

        # path 1: structured tool_calls
        if msg.get("tool_calls"):
            log.debug("tool_calls: %s", msg["tool_calls"])
            call = msg["tool_calls"][0]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}
            log.info("dispatching %s with args %s", call["function"]["name"], args)
            self._last_tool_call = call
            return call["function"]["name"], args

        return None, None

    def _answer(self, user_text: str, command: str, network_reply: List[Dict]) -> str:
        chat_id     = self._current_chat_id
        role_system = "developer" if self._is_fg else "system"

        messages = [
            {"role": role_system, "content": _ANSWER_SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "",
            "tool_calls": [self._last_tool_call]},
            {"role": "tool",
            "name": self._last_tool_call["function"]["name"],
            "content": json.dumps(network_reply, ensure_ascii=False)},
        ]
        kw = dict(messages=messages, max_tokens=512, temperature=0.0, repeat_penalty=1.1)

        td = self.last_tool_defs.get(self._last_tool_call["function"]["name"])
        if td:
            kw["tools"] = [{"type": "function", "function": {
                "name":        td["name"],
                "description": td.get("description", ""),
                "parameters":  td.get("parameters", {})}}]

        try:
            resp  = self._llm.create_chat_completion(**kw)
            reply = resp["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"[error generating answer] {e}"

        self._append_history(chat_id, user_text, reply)
        return reply

    def _no_tool_answer(self, user_text: str) -> str:
        return self._reply_if_no_tool.get("content", "ERROR: Sorry, I couldn't understand your request, retry.")

    # ── history (used by backup_manager) ─────────────────────────────────────

    def get_history(self, chat_id: int) -> List[dict]:
        with self._history_lock:
            return list(self._histories.get(chat_id, []))

    def load_history(self, chat_id: int, messages: List[dict]) -> None:
        with self._history_lock:
            self._histories[chat_id] = list(messages)

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_history_for_dispatch(self, chat_id: int) -> List[dict]:
        if not ENABLE_HISTORY:
            return []
        with self._history_lock:
            return list(self._histories.get(chat_id, []))

    def _append_history(self, chat_id: int, user_text: str, reply: str) -> None:
        if not ENABLE_HISTORY:
            return
        with self._history_lock:
            h = self._histories.setdefault(chat_id, [])
            h.append({"role": "user",      "content": user_text})
            h.append({"role": "assistant", "content": reply})


# ── helpers ───────────────────────────────────────────────────────────────────

def _select_model(score: float):
    """Pick (params_b, repo, filename, label, is_fg) from the device score."""
    if not LLM_detection_with_score:
        log.info("LLM_detection_with_score=False -> functiongemma:270m")
        return _FUNCTIONGEMMA

    if score >= _SCORE_QWEN:
        chosen = _QWEN
    else:
        chosen = _FUNCTIONGEMMA
    log.info("score=%s -> %s", score, chosen[3])
    return chosen

def _download_model(repo: str, filename: str, dest: Path):
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(_MODELS_DIR))
        Path(tmp).rename(dest)
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename}: {e}") from e