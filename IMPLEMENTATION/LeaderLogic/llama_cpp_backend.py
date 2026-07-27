"""
LeaderLogic/llama_cpp_backend.py

Shared llama-cpp-python (GGUF, CPU-only) backend, used by the `generic` and
`stm32mp257fdk` leader presets - they differ only in which model they load and
how they phrase their prompts, not in how they call it.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional chat-format handlers (registered globally via decorator on import).
try:
    import LeaderLogic.functiongemma_simple_handler  # noqa: F401
    HAS_FG = True
except ImportError:
    HAS_FG = False

try:
    import LeaderLogic.qwen3_handler  # noqa: F401
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

_MODELS_DIR = Path(__file__).parent.parent / "models"


def load_llm(repo: str, filename: str, is_fg: bool):
    """Download (if missing) and load a GGUF via llama-cpp-python."""
    from llama_cpp import Llama   # before the download, so a missing wheel fails first

    model_path = _MODELS_DIR / filename
    if not model_path.exists():
        logger.info("downloading %s ...", filename)
        _download_model(repo, filename, model_path)
    else:
        logger.info("using cached %s", filename)

    if is_fg and HAS_FG:
        return Llama(model_path=str(model_path), n_ctx=4096, chat_format="functiongemma",
                     n_threads=os.cpu_count() or 4, verbose=False)
    if not is_fg and HAS_QWEN:
        return Llama(model_path=str(model_path), n_ctx=4096, chat_format="qwen3",
                     n_threads=os.cpu_count() or 4, verbose=False)
    if is_fg and not HAS_FG:
        logger.warning("functiongemma_simple_handler not installed - tool calling will likely fail")
    if not is_fg and not HAS_QWEN:
        logger.warning("qwen3_handler not installed - tool calling will likely fail")
    return Llama(model_path=str(model_path), n_ctx=4096,
                 n_threads=os.cpu_count() or 4, verbose=False)


def _download_model(repo: str, filename: str, dest: Path):
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    tmp = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(_MODELS_DIR))
    Path(tmp).rename(dest)


class LlamaCppBackend:
    """Dispatch + answer over a local llama-cpp-python chat model.

    FunctionGemma is driven via the developer role (its own chat template);
    everything else uses the system role.

    dispatch_system and answer_system are independent - a preset that wants
    them to be the SAME string (so the leading system-prompt tokens stay
    byte-identical across the dispatch -> answer switch, letting
    llama-cpp-python's Llama.generate() reuse that prefix's KV cache instead of
    recomputing it from scratch) can just pass the same constant for both; a
    preset that wants them different (e.g. to keep each step's instructions
    focused) can pass two.
    """

    def __init__(self, llm, is_fg: bool, dispatch_system: str, answer_system: str,
                 model_name: str, model_params_b: float):
        self._llm             = llm
        self._is_fg           = is_fg
        self._dispatch_system = dispatch_system
        self._answer_system   = answer_system
        self.model_name       = model_name
        self.model_params_b   = model_params_b
        self._last_tool_call: Optional[dict] = None
        self._last_tool_defs: Dict[str, dict] = {}

    def dispatch(self, user_text: str,
                 tool_defs: List[Dict]) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
        self._last_tool_defs = {td.get("function", td)["name"]: td.get("function", td)
                                 for td in tool_defs}

        role     = "developer" if self._is_fg else "system"
        messages = [{"role": role, "content": self._dispatch_system},
                    {"role": "user", "content": user_text}]

        lc_tools = [{"type": "function", "function": {
            "name":        td.get("function", td)["name"],
            "description": td.get("function", td).get("description", ""),
            "parameters":  td.get("function", td).get("parameters", {}),
        }} for td in tool_defs]

        kw = dict(messages=messages, tools=lc_tools, max_tokens=256,
                  temperature=0.0, seed=42, repeat_penalty=1.1)
        if not self._is_fg:
            kw["tool_choice"] = "auto"

        try:
            resp = self._llm.create_chat_completion(**kw)
        except Exception as e:
            logger.error("dispatch error: %s", e)
            return None, None, "Sorry, I couldn't reach the model."

        msg = resp["choices"][0]["message"]
        logger.info("raw dispatch: %r", msg)

        if msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}
            self._last_tool_call = call
            logger.info("dispatching %s with args %s", call["function"]["name"], args)
            return call["function"]["name"], args, None

        return None, None, msg.get("content") or "ERROR: Sorry, I couldn't understand your request, retry."

    def answer(self, user_text: str, command: str, network_reply: List[Dict]) -> str:
        role     = "developer" if self._is_fg else "system"
        messages = [
            {"role": role, "content": self._answer_system},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "", "tool_calls": [self._last_tool_call]},
            {"role": "tool",
             "name": self._last_tool_call["function"]["name"],
             "content": json.dumps(network_reply, ensure_ascii=False)},
        ]
        kw = dict(messages=messages, max_tokens=512, temperature=0.0, repeat_penalty=1.1)

        td = self._last_tool_defs.get(self._last_tool_call["function"]["name"])
        if td:
            kw["tools"] = [{"type": "function", "function": {
                "name": td["name"], "description": td.get("description", ""),
                "parameters": td.get("parameters", {})}}]

        try:
            resp = self._llm.create_chat_completion(**kw)
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[error generating answer] {e}"
