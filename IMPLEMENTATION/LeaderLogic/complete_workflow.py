"""
complete_workflow.py

All workflow variants unified in one file, sharing common infrastructure
via LeaderNetwork (leader_network.py).

Workflows are PURE INFERENCE STRATEGIES. They do not own the transport,
discovery, or network — those are created and managed by the leader module
(LeaderLogic/generic_leader.py) and a single shared LeaderNetwork is passed
in. A workflow only knows how to:
    _dispatch(user_text, tool_defs) -> (fn_name, args) | (None, None)
    _answer(user_text, command, network_reply) -> str
    _no_tool_answer(user_text) -> str
…and uses the shared network for tool lookup (available_tools) + dispatch.

Workflows:
  Workflow1 — Hailo NPU native tool-calling (hailo_platform .hef)
  Workflow2 — OpenAI tool-calling on a single LLM (CPU Ollama, OpenAI HTTP)
  Workflow3 — Split-LLM: dispatch LLM (FunctionGemma, CPU) + answer LLM (Hailo)

The leader dispatches a picked tool to the sensing agents that own it as a
JSON-RPC tools/call; see leader_network.py / tool_dispatcher.py.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

import requests

# Tool defs + dispatch come from LeaderNetwork (registry + dispatcher),
# injected by the leader module — see leader_network.py.

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT (OpenAI-compatible endpoint)
# ══════════════════════════════════════════════════════════════════════════════

class LLM:
    """Thin wrapper around an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint
        self.model    = model
        self.headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def ask(self, system: str, user: str, temperature: float = 0.1) -> str:
        payload = {
            "model":    self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        logger.debug("LLM.ask POST %s", self.endpoint)
        logger.debug("LLM.ask payload: %s", json.dumps(payload))
        resp = requests.post(self.endpoint, headers=self.headers,
                            json=payload, timeout=30)
        logger.debug("LLM.ask status: %s", resp.status_code)
        logger.debug("LLM.ask response body: %s", resp.text)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def chat(self, messages: list, tools: list = None,
             temperature: float = 0, seed: int = 42,
             tool_choice: str = "required") -> dict:
        """Full chat completion with optional tool calling. Returns raw API response.

        tool_choice "required" forces a tool call; "auto" lets the model answer
        in plain text when nothing matches (so dispatch can short-circuit).
        """
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = tool_choice
        resp = requests.post(self.endpoint, headers=self.headers,
                             json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED PROMPTS  (from generic_leader.py)
# ══════════════════════════════════════════════════════════════════════════════

# Qwen3 is a hybrid reasoning model: by default it prepends a <think>…</think>
# block before its real output, and on edge CPU/NPU that reasoning dominates
# generation time. Dispatch only needs to pick one tool and the answer is a short
# summary, so we disable reasoning via Qwen's native `/no_think` soft switch.
# This reaches every path (Workflow1/3 Hailo answer, Workflow2 Ollama dispatch+
# answer); FunctionGemma (Workflow3 dispatch) has no thinking mode and ignores it.
# NOTE: `/no_think` on DISPATCH trades a little tool-selection accuracy for speed
# (see BENCHMARK CODE/ACCURACY, think vs no_think). Remove it from DISPATCH_SYSTEM
# alone to restore dispatch reasoning while keeping the fast answer step.
DISPATCH_SYSTEM = (
    "You are a smart-home agent dispatcher. "
    "Call exactly ONE tool that best matches the user's intent. "
    "Extract any names or object targets precisely from the user's message. "
    "If nothing matches, reply normally without calling any tool. "
    "/no_think"
)
ANSWER_SYSTEM = (
    "You are a helpful assistant for a smart home agent network. "
    "The network has already executed a command and returned real data. "
    "Translate that raw result into a short, friendly reply for the user. "
    "DO NOT OUTPUT JSON, field names, brackets, or raw data structures. "
    "ALWAYS MENTION IN THE REPLY WHERE THE RESULTS ARE COMING FROM: the 'from' value is the "
    "device that executed the tool, and the user MUST be told which device(s) reported it. "
    "Always reply in the same language the user wrote in. "
    "/no_think"
)


# ══════════════════════════════════════════════════════════════════════════════
# BASE WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

class BaseWorkflow:
    """
    Common skeleton for all workflows. PURE INFERENCE — owns no transport or
    network; the shared LeaderNetwork is injected by the leader module.

    Two ways a leader can drive a workflow:
      * call run(chat_id, text) — handles typing indicator + send to bot
        (used by the GGUF leaders, which record history inside their hooks);
      * call _run_pipeline(text) directly and handle send/history itself
        (used by the Pi leader, whose workflow hooks don't touch history).

    Subclasses implement:
      _dispatch(user_text, tool_defs) -> (fn_name, args) or (None, None)
      _answer(user_text, command, network_reply) -> str
      _no_tool_answer(user_text) -> str   (optional override)
    """

    def __init__(self, net, bot=None):
        self.net = net          # LeaderNetwork: available_tools() + dispatch()
        self.bot = bot
        # When dispatch decides no tool is needed it already produces a
        # plain-text reply. Subclasses stash it here so _no_tool_answer can
        # forward it directly instead of spending a second LLM pass
        # regenerating an answer (cf. generic_leader._reply_if_no_tool).
        self._reply_if_no_tool: Optional[str] = None

    def run(self, chat_id: int, user_text: str) -> None:
        """Full pipeline for one user message, delivering the reply to the bot."""
        if self.bot:
            self.bot.start_typing_indicator(chat_id)
        try:
            answer = self._run_pipeline(user_text)
        except Exception as exc:
            logger.exception("Workflow pipeline failed: %s", exc)
            answer = "Sorry, something went wrong."
        finally:
            if self.bot:
                self.bot.stop_typing_indicator(chat_id)

        if self.bot:
            self.bot.send_message(chat_id, answer)
        else:
            logger.info("answer: %s", answer)

    def _run_pipeline(self, user_text: str) -> str:
        # Step 1: tool defs are a local lookup now (the registry is kept current
        # by LeaderNetwork as peers join/leave) — no per-call network round-trip.
        tool_defs = self.net.available_tools()
        if not tool_defs:
            return "No skills are available on the network right now."

        # Step 2: dispatch — subclass picks a tool
        fn_name, args = self._dispatch(user_text, tool_defs)
        if fn_name is None:
            return self._no_tool_answer(user_text)

        # Step 3: send the picked tool to every agent that owns it, gather replies
        command = json.dumps({"name": fn_name, "arguments": args or {}})
        logger.info("%s tool -> command: %r", self.__class__.__name__, command)

        replies = self.net.dispatch(fn_name, args or {})
        if not replies:
            return f"No peer responded to '{fn_name}'"
        return self._answer(user_text, command, replies)   # list, not joined string

    # ── Subclass hooks ─────────────────────────────────────────────────────────

    def _dispatch(self, user_text: str, tool_defs: List[Dict]) -> tuple:
        raise NotImplementedError

    def _answer(self, user_text: str, command: str, network_reply: List[Dict]) -> str:
        raise NotImplementedError

    def _no_tool_answer(self, user_text: str) -> str:
        # Dispatch already generated a natural reply (no tool call); forward it
        # instead of regenerating. Subclasses may override for a smarter fallback.
        if self._reply_if_no_tool:
            return self._reply_if_no_tool
        return "I wasn't sure which action to take for your request."


# ══════════════════════════════════════════════════════════════════════════════
# Tool-def normalisation + dispatch parsing helpers (shared)
# ══════════════════════════════════════════════════════════════════════════════

def _to_openai_tools(tool_defs: List[Dict]) -> List[Dict]:
    """Normalise tool defs into OpenAI function-calling shape."""
    out = []
    for td in tool_defs:
        fn = td.get("function", td)
        out.append({"type": "function", "function": {
            "name":        fn["name"],
            "description": fn.get("description", ""),
            "parameters":  fn.get("parameters", {}),
        }})
    return out


def _parse_tool_call_from_text(raw: str) -> Optional[Dict[str, Any]]:
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
        # XML variants inside <tool_call>
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

    # bare JSON with a "name" key
    m = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if obj.get("name"):
                return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
        except json.JSONDecodeError:
            pass
    return None


def _content_from_openai(resp: dict) -> str:
    """Extract the assistant's plain-text content from an OpenAI-style response."""
    msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
    return (msg.get("content") or "").strip()


def _parse_openai_dispatch(resp: dict) -> tuple:
    """Parse (fn_name, args) from an OpenAI-style chat-completion dict."""
    msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        call = tool_calls[0]
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception:
            args = {}
        return call["function"]["name"], args
    # fall back to parsing the text body
    raw = msg.get("content", "") or ""
    parsed = _parse_tool_call_from_text(raw)
    if parsed:
        return parsed["name"], parsed["arguments"]
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW 1 — Hailo NPU native tool-calling (hailo_platform .hef)
# ══════════════════════════════════════════════════════════════════════════════

class Workflow1(BaseWorkflow):
    """
    Hailo NPU handles both dispatch (native tool calling) and answer.
    Inference only — collector is shared/injected.
    """

    def __init__(self, llm_hef_path: str, net, bot=None,
                 shared_vdevice=None):
        super().__init__(net, bot)
        self.llm_hef_path = str(llm_hef_path)
        self._shared_vdevice = shared_vdevice
        self._own_vdevice = False
        self.vdevice = None
        self.llm = None

        self._hailo_temperature = 0.1
        self._hailo_seed = 42
        self._hailo_max_tokens = 512

    # ── Hailo lifecycle ────────────────────────────────────────────────────────

    def activate(self):
        """Acquire the Hailo device + load the HEF. Call before first use."""
        if self.llm is not None:
            return
        from hailo_platform import VDevice
        from hailo_platform.genai import LLM as HailoLLM

        if self._shared_vdevice is None:
            params = VDevice.create_params()
            params.group_id = "1"
            self.vdevice = VDevice(params)
            self._own_vdevice = True
        else:
            self.vdevice = self._shared_vdevice
            self._own_vdevice = False

        self.llm = HailoLLM(self.vdevice, self.llm_hef_path)
        logger.info("Workflow1 Hailo LLM loaded")

    def deactivate(self):
        """Release the Hailo LLM + VDevice."""
        if self.llm is not None:
            try:
                self.llm.release()
            except Exception as e:
                logger.warning("Workflow1 llm.release() failed: %s", e)
            self.llm = None
        if self._own_vdevice and self.vdevice is not None:
            try:
                self.vdevice.release()
            except Exception as e:
                logger.warning("Workflow1 vdevice.release() failed: %s", e)
        self.vdevice = None
        logger.info("Workflow1 Hailo LLM released")

    def close(self):
        self.deactivate()

    def _hailo_generate(self, messages: list, tools=None) -> str:
        parts = []
        with self.llm.generate(
            prompt=messages,
            tools=tools,
            temperature=self._hailo_temperature,
            seed=self._hailo_seed,
            max_generated_tokens=self._hailo_max_tokens,
        ) as gen:
            for token in gen:
                parts.append(token)
        # Hailo's genai LLM emits the qwen3 turn-end token as a literal string;
        # strip it. NOTE: .replace() removes EVERY occurrence, not just a
        # trailing one — fine here since <|im_end|> shouldn't appear mid-text.
        return "".join(parts).replace("<|im_end|>", "")

    # ── hooks ──────────────────────────────────────────────────────────────────

    def _dispatch(self, user_text: str, tool_defs: List[Dict]) -> tuple:
        messages = [
            {"role": "system", "content": DISPATCH_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        self.llm.clear_context()
        raw = self._hailo_generate(messages, tools=tool_defs)
        logger.info("Workflow1 dispatch raw: %r", raw[:400])
        parsed = _parse_tool_call_from_text(raw)
        if parsed is None:
            # No tool call: the model replied normally — forward that text.
            self._reply_if_no_tool = raw.strip()
            return None, None
        return parsed.get("name"), parsed.get("arguments", {})

    def _answer(self, user_text: str, command: str, network_reply: List[Dict]) -> str:
        reply_json = json.dumps(network_reply, ensure_ascii=False)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",   "content": (
                f"User request: {user_text}\n"
                f"Command executed: {command}\n\n"
                f"RESULT DATA (write your reply from this; the 'from' field is the device "
                f"that ran the tool, so name it in your answer)\n"
                f"{reply_json}\n"
            )},
        ]
        self.llm.clear_context()
        raw = self._hailo_generate(messages)
        logger.debug("Workflow1 answer: %s", raw[:400])
        return raw.strip()

    # _no_tool_answer is inherited: it forwards the dispatch text stashed in
    # self._reply_if_no_tool, avoiding a second Hailo generation pass.


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW 2 — Single-LLM tool-calling (CPU Ollama, OpenAI HTTP)
# ══════════════════════════════════════════════════════════════════════════════

class Workflow2(BaseWorkflow):
    """Single CPU LLM (qwen3:1.7b) for both dispatch and answer."""

    def __init__(self, llm: LLM, net, bot=None):
        super().__init__(net, bot)
        self.llm = llm

    def _dispatch(self, user_text: str, tool_defs: List[Dict]) -> tuple:
        messages = [
            {"role": "system", "content": DISPATCH_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        # tool_choice="auto" so the model can reply in plain text when no tool
        # matches; that reply is then forwarded as-is (see _no_tool_answer).
        resp = self.llm.chat(messages, tools=_to_openai_tools(tool_defs),
                             tool_choice="auto")
        logger.info("Workflow2 dispatch response: %s",
                    json.dumps(resp.get("choices", [{}])[0].get("message", {})))
        fn_name, args = _parse_openai_dispatch(resp)
        if fn_name is None:
            # No tool call: stash the model's own reply for _no_tool_answer.
            self._reply_if_no_tool = _content_from_openai(resp)
        return fn_name, args

    def _answer(self, user_text: str, command: str, network_reply: List[Dict]) -> str:
        reply_json = json.dumps(network_reply, ensure_ascii=False)
        prompt = (
            f"User request: {user_text}\n"
            f"Command executed: {command}\n\n"
            f"RESULT DATA (write your reply from this; the 'from' field is the device "
            f"that ran the tool, so name it in your answer)\n"
            f"{reply_json}\n"
        )
        return self.llm.ask(ANSWER_SYSTEM, prompt, temperature=0.7)

    # _no_tool_answer is inherited: it forwards the dispatch content stashed in
    # self._reply_if_no_tool, avoiding a second CPU LLM call.


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW 3 — Split: FunctionGemma (CPU) dispatch + Hailo (.hef) answer
# ══════════════════════════════════════════════════════════════════════════════

class Workflow3(BaseWorkflow):
    """
    Dispatch via FunctionGemma on CPU Ollama (OpenAI function calling);
    answer via the Hailo NPU (.hef native).

    A Hailo-backed Workflow1 instance is injected as `hailo_answerer` so the
    NPU device/HEF is shared and managed in one place (the leader module),
    rather than re-acquired here.
    """

    def __init__(self, llm_dispatch: LLM, hailo_answerer: "Workflow1",
                 net, bot=None):
        super().__init__(net, bot)
        self.llm_dispatch   = llm_dispatch
        self.hailo_answerer = hailo_answerer   # Workflow1 owning the Hailo LLM

    def _dispatch(self, user_text: str, tool_defs: List[Dict]) -> tuple:
        messages = [
            {"role": "system", "content": DISPATCH_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        resp = self.llm_dispatch.chat(messages, tools=_to_openai_tools(tool_defs))
        logger.info("Workflow3 dispatch response: %s",
                    json.dumps(resp.get("choices", [{}])[0].get("message", {})))
        fn_name, args = _parse_openai_dispatch(resp)
        if fn_name is None:
            # No tool call: stash the dispatch model's own reply.
            self._reply_if_no_tool = _content_from_openai(resp)
        return fn_name, args

    def _answer(self, user_text: str, command: str, network_reply: List[Dict]) -> str:
        # Reuse the Hailo answerer's generation path.
        return self.hailo_answerer._answer(user_text, command, network_reply)

    def _no_tool_answer(self, user_text: str) -> str:
        # Forward FunctionGemma's own dispatch text; only fall back to a fresh
        # Hailo pass if it produced nothing to forward.
        if self._reply_if_no_tool:
            return self._reply_if_no_tool
        return self.hailo_answerer._no_tool_answer(user_text)