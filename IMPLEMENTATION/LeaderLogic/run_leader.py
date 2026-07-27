"""
LeaderLogic/run_leader.py

Leader entry point - the leader-side counterpart to SensingLogic/sensing_agent.py.

Owns everything every leader preset needs in common: the LeaderNetwork (tool
registry + dispatcher), conversation history for backup/failover, and the
two-step pipeline (query -> tool -> devices, tool result -> reply). A preset
under LeaderLogic/<preset>/leader.py only supplies the two model-specific
pieces of that pipeline, via:

    load(cfg, profile) -> Backend
        Backend.model_name       : str
        Backend.model_params_b   : float
        Backend.dispatch(user_text, tool_defs) -> (fn_name, args, no_tool_reply)
        Backend.answer(user_text, command, network_reply) -> str

(The `mock` preset predates this split and is not a pipeline at all - it
still exposes the old boot(cfg, agent_id, transport, discovery, bot, profile)
contract directly; a preset module with no load() is booted that way instead.)

Exposed for main.py:
    MODEL_NAME, MODEL_PARAMS_B   (filled in by boot())
    boot(cfg, agent_id, transport, discovery, bot, profile) -> Leader
"""

import importlib
import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

from ElectionLogic.identity import get_leader_module
from LeaderLogic.leader_network import make_network

logger = logging.getLogger(__name__)

# Filled in by boot(), read by main.py for the capability broadcast / /status.
MODEL_NAME     = "unknown"
MODEL_PARAMS_B = 0.0


# ── boot ──────────────────────────────────────────────────────────────────────

def boot(cfg: dict, agent_id: str, transport, discovery, bot, profile: dict = None):
    global MODEL_NAME, MODEL_PARAMS_B

    module_path = get_leader_module(profile or {})
    try:
        preset_mod = importlib.import_module(module_path)
    except ImportError:
        logger.warning("%s not found - falling back to generic", module_path)
        preset_mod = importlib.import_module("LeaderLogic.generic.leader")

    if not hasattr(preset_mod, "load"):
        # legacy preset (the mock leader): owns its whole boot, no shared pipeline.
        leader = preset_mod.boot(cfg=cfg, agent_id=agent_id, transport=transport,
                                  discovery=discovery, bot=bot, profile=profile)
        MODEL_NAME     = getattr(preset_mod, "MODEL_NAME", "unknown")
        MODEL_PARAMS_B = getattr(preset_mod, "MODEL_PARAMS_B", 0.0)
        return leader

    net     = make_network(transport, discovery,
                           reply_timeout=cfg.get("timeouts", {}).get("replies", 3.0))
    backend = preset_mod.load(cfg, profile or {})
    MODEL_NAME     = backend.model_name
    MODEL_PARAMS_B = backend.model_params_b

    logger.info("ready - %s", backend.model_name)
    return Leader(net=net, bot=bot, backend=backend)


# ── Leader (shared pipeline for every preset that provides a Backend) ───────

class Leader:
    """
    Shared control flow:
        1. look up the tools currently available on the network
        2. backend.dispatch() picks ONE tool (or answers directly, no tool)
        3. tell the user which tool is running on which device(s)
        4. net.dispatch() fans the tool out to its owners and gathers replies
        5. backend.answer() turns the replies into a reply for the user
    """

    def __init__(self, net, bot, backend):
        self.net     = net
        self.bot     = bot
        self.backend = backend

        self._histories:    Dict[int, List[dict]] = {}
        self._history_lock = threading.Lock()

    # ── main.py interface ─────────────────────────────────────────────────────

    def handle(self, chat_id: int, text: str) -> None:
        if self.bot:
            self.bot.start_typing_indicator(chat_id)
        try:
            reply = self._run_pipeline(chat_id, text)
        except Exception:
            logger.exception("pipeline failed")
            reply = "Sorry, something went wrong."
        finally:
            if self.bot:
                self.bot.stop_typing_indicator(chat_id)

        self._append_history(chat_id, text, reply)
        if self.bot:
            self.bot.send_message(chat_id, reply)

    def get_history(self, chat_id: int) -> List[dict]:
        with self._history_lock:
            return list(self._histories.get(chat_id, []))

    def load_history(self, chat_id: int, messages: List[dict]) -> None:
        with self._history_lock:
            self._histories[chat_id] = list(messages)

    # ── pipeline ──────────────────────────────────────────────────────────────

    def _run_pipeline(self, chat_id: int, user_text: str) -> str:
        tool_defs = self.net.available_tools()
        if not tool_defs:
            return "No skills are available on the network right now."

        fn_name, args, no_tool_reply = self.backend.dispatch(user_text, tool_defs)
        if fn_name is None:
            return no_tool_reply or "I wasn't sure which action to take for your request."

        owners = sorted(self.net.owners(fn_name))
        if self.bot and owners:
            self.bot.send_message(
                chat_id, f'using "{fn_name}" on those devices: {", ".join(owners)}')

        command = json.dumps({"name": fn_name, "arguments": args or {}})
        replies = self.net.dispatch(fn_name, args or {})
        if not replies:
            return f"No peer responded to '{fn_name}'"
        return self.backend.answer(user_text, command, replies)

    # ── internal ──────────────────────────────────────────────────────────────

    def _append_history(self, chat_id: int, user_text: str, reply: str) -> None:
        with self._history_lock:
            h = self._histories.setdefault(chat_id, [])
            h.append({"role": "user",      "content": user_text})
            h.append({"role": "assistant", "content": reply})


# ── shared tool-call parsing helpers (used by preset backends) ──────────────

def to_openai_tools(tool_defs: List[Dict]) -> List[Dict]:
    """Normalise tool defs (nested or flat) into OpenAI function-calling shape."""
    out = []
    for td in tool_defs:
        fn = td.get("function", td)
        out.append({"type": "function", "function": {
            "name":        fn["name"],
            "description": fn.get("description", ""),
            "parameters":  fn.get("parameters", {}),
        }})
    return out


def parse_openai_dispatch(resp: dict) -> tuple:
    """Parse (fn_name, args, no_tool_reply) from an OpenAI-style chat-completion
    response (Ollama's /v1/chat/completions).

    Checks the structured tool_calls field first; if it's empty, falls back to
    parse_tool_call_from_text() on the plain content, since local models served
    over Ollama very often emit the call as <tool_call>{...}</tool_call> text
    instead of populating tool_calls.
    """
    msg        = (resp.get("choices") or [{}])[0].get("message", {}) or {}
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        call = tool_calls[0]
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception:
            args = {}
        return call["function"]["name"], args, None

    content = (msg.get("content") or "").strip()
    parsed  = parse_tool_call_from_text(content)
    if parsed:
        return parsed["name"], parsed.get("arguments") or {}, None
    return None, None, content


def parse_tool_call_from_text(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call out of free text. Handles:
        <tool_call>{...}</tool_call>  (JSON or XML body)
        bare {"name": ..., "arguments": ...}
    Returns {"name", "arguments"} or None.
    """
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    if m:
        content = m.group(1).strip()
        try:
            obj = json.loads(content)
            if obj.get("name"):
                return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
        except json.JSONDecodeError:
            pass
        name_match = re.search(
            r"<name>(.*?)</name>|<function-name>(.*?)</function-name>",
            content, re.DOTALL)
        if name_match:
            args_match = re.search(r"<arguments>(.*?)</arguments>", content, re.DOTALL)
            try:
                arguments = json.loads(args_match.group(1).strip()) if args_match else {}
            except json.JSONDecodeError:
                arguments = {}
            return {"name": (name_match.group(1) or name_match.group(2)).strip(),
                    "arguments": arguments}

    m = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if obj.get("name"):
                return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
        except json.JSONDecodeError:
            pass

    # Final fallback: a valid tool-call JSON object is present but not in a clean
    # <tool_call>{...}</tool_call> wrapper. The Hailo qwen3 NPU, for instance,
    # emits an EMPTY <tool_call></tool_call> pair (which the first regex matches,
    # empty) followed by the real object, whose nested "arguments": {} the flat
    # bare-JSON regex above cannot span. Scan for the first brace-balanced {...}
    # that parses and carries a "name". Only reached once the paths above miss,
    # so every previously-handled shape still returns exactly as before.
    for candidate in _iter_balanced_braces(raw):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
    return None


def _iter_balanced_braces(text: str):
    """Yield each top-level brace-balanced {...} substring of `text`, in order."""
    depth, start = 0, None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None
