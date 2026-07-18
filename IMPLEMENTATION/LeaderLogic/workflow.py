"""
workflow.py

Telegram message → LLM → broadcast command → wait for reply → LLM final answer.

One LLM class, two workflows, no external dependencies.

Workflow1: text-based dispatch (original simple version)
Workflow2: LLM tool-calling, broadcast as plain text, feed reply back as tool result
"""

import json
import threading
import time
import logging
from unittest.mock import call
import requests

log = logging.getLogger(__name__)


SKILLS_TIMEOUT = 1.5   # seconds to collect fetchskills replies
REPLY_TIMEOUT  = 3.0   # seconds to wait for a network command reply
ENABLE_HISTORY = False  # set to True to enable chat history for Workflow2
MAX_HISTORY    = 20


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLM:
    """Thin wrapper around an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint
        self.model    = model
        self.headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    
    def ask(self, system: str, user: str) -> str:
        """Ask does not use tools, returns the raw text reply from the LLM."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
        }
        resp = requests.post(self.endpoint, headers=self.headers,
                             json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def chat(self, messages: list, tools: list = None, temperature: float = 0, seed: int = 42) -> dict:
        """Chat does use tools, returns the full LLM response (including tool_calls) as a dict."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        resp = requests.post(self.endpoint, headers=self.headers,
                             json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()


# ── Workflow1 (original simple version + typing indicator) ─────────────────────

class Workflow1:
    """
    Two-step pipeline:
      1. LLM(system=DISPATCH_PROMPT) reads user message + available skills
         → outputs exactly one network command to broadcast
      2. Wait up to REPLY_TIMEOUT for a network reply
      3. LLM(system=ANSWER_PROMPT) reads user message + network reply
         → outputs a human-friendly answer for Telegram
    """

    _DISPATCH_PROMPT = """You are a command dispatcher. Output ONLY ONE command to execute, nothing else.
    Rules:
    - Copy the command EXACTLY as listed, replace <object> or <n> with the actual value
    - No extra words, no punctuation, no explanation
    - If nothing fits output: none
    - If more than one command seems to fit, pick the best one: only one!

    Examples:
    User: who is in front of the camera -> recognize
    User: remember this person as Mario -> add face Mario
    User: tell me when you see a dog -> run till detect dog
    User: what objects are visible -> detect objects now"""

    _ANSWER_PROMPT = """You are a helpful assistant for a smart home agent network.
    The network has already executed a command and returned real data.
    Your job: translate that raw result into a short, friendly reply for the user.
    Always reply in the same language the user wrote in.
    Only if no data are provided say you cannot answer — otherwise the data is right there in the network reply, use it."""

    def __init__(self, llm: LLM, broadcaster, bot):
        self.llm         = llm
        self.broadcaster = broadcaster
        self.bot         = bot

        self._lock              = threading.Lock()
        self._skills_buf:       list[str] = []
        self._reply_buf:        list[str] = []
        self._collecting_skills = False
        self._collecting_reply  = False
        self._reply_event       = threading.Event()

    def run(self, chat_id: int, user_text: str):
        """Run the full pipeline for one Telegram message (call in a thread)."""

        self.bot.start_typing_indicator(chat_id)

        # step 1: fetch available skills
        skills_text = self._fetch_skills()

        # step 2: LLM picks a command
        user_prompt = (
            f"User request: {user_text}\n\n"
            f"Available commands:\n{skills_text or '(none)'}"
        )
        command = self.llm.ask(self._DISPATCH_PROMPT, user_prompt)
        log.info("workflow1 LLM command: %s", command)

        # step 3: broadcast (unless LLM said none)
        network_reply = ""
        if command.strip().lower() != "none":
            network_reply = self._broadcast_and_wait(command)
        log.debug("workflow1 network reply: %s", network_reply)

        # step 4: LLM crafts final answer
        summary = (
            f"User request: {user_text}\n"
            f"Command executed: {command}\n"
            f"Network reply:\n{network_reply or '(no reply)'}\n"
            "Craft a short answer to the user based on the request and the network reply."
        )
        log.debug("workflow1 system prompt: %s", self._ANSWER_PROMPT)
        log.debug("workflow1 prompt: %s", summary)
        answer = self.llm.ask(self._ANSWER_PROMPT, summary)
        log.debug("workflow1 answer: %s", answer)

        self.bot.stop_typing_indicator(chat_id)
        self.bot.send_message(chat_id, answer)

    def handle_incoming(self, msg: dict):
        """Called from run_leader when a P2P message arrives.
        Feeds the skills and reply collection buffers."""
        text   = msg.get("text", "")
        sender = msg.get("from", "?")

        with self._lock:
            if self._collecting_skills:
                self._skills_buf.append(f"{sender}: {text}")
            if self._collecting_reply:
                self._reply_buf.append(f"{sender}: {text}")
                self._reply_event.set()

    def _fetch_skills(self) -> str:
        """Broadcast fetchskills to all peers, collect replies for SKILLS_TIMEOUT seconds."""
        with self._lock:
            self._skills_buf.clear()
            self._collecting_skills = True

        self.broadcaster({"text": "fetchskills"})
        time.sleep(SKILLS_TIMEOUT)

        with self._lock:
            self._collecting_skills = False
            result = list(self._skills_buf)

        return "\n".join(result)

    def _broadcast_and_wait(self, command: str) -> str:
        with self._lock:
            self._reply_buf.clear()
            self._collecting_reply = True
        self._reply_event.clear()

        self.broadcaster({"text": command})
        self._reply_event.wait(timeout=REPLY_TIMEOUT)

        with self._lock:
            self._collecting_reply = False
            result = list(self._reply_buf)

        return "\n".join(result)


# ── Workflow2 (tool-calling, plain text broadcast) ─────────────────────────────
# this will have to be sent by the boards if requested
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "recognize",
            "description": "Open camera and identify who is in front of it",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_face",
            "description": "Register the person in frame under a given name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to register the face under"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_face",
            "description": "Delete a registered face by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the face to delete"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_faces",
            "description": "Return the list of all registered face names",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects_now",
            "description": "Open camera and detect objects currently in frame",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_till_detect",
            "description": "Loop object recognition until a specific target is spotted",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {"type": "string", "description": "Object name to watch for"},
                },
                "required": ["object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetchskills",
            "description": "Return the full list of available agent commands",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


TOOL_SYSTEM = """You are a smart-home agent dispatcher.
Given the user's request, call exactly ONE tool that best matches their intent.
Extract any names or object targets precisely from the user's message.
Do not call multiple tools. Do not reply with text — always use a tool call."""


class Workflow2:

    def __init__(self, llm: LLM, broadcaster, bot=None, tool_timeout: float = 30.0):
        self.llm          = llm
        self.broadcaster  = broadcaster
        self.bot          = bot
        self.tool_timeout = tool_timeout

        self._lock             = threading.Lock()
        self._reply_buf        = []
        self._collecting_reply = False
        self._reply_event      = threading.Event()

        self._history      = {}
        self._history_lock = threading.Lock()

        self._openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": td["function"]["name"],
                    "description": td["function"]["description"],
                    "parameters": td["function"]["parameters"],
                },
            }
            for td in TOOL_DEFS
        ]

    def run(self, chat_id: int, user_text: str):
        if self.bot:
            self.bot.start_typing_indicator(chat_id)
        try:
            self._run_inner(chat_id, user_text)
        finally:
            self.bot.stop_typing_indicator(chat_id)

    def _run_inner(self, chat_id: int, user_text: str):
        if ENABLE_HISTORY:
            with self._history_lock:
                if chat_id not in self._history:
                    self._history[chat_id] = []
                history = list(self._history[chat_id])
        else:
            history = []

        messages = [{"role": "system", "content": TOOL_SYSTEM}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        # step 1: LLM decides — tool call or plain text
        debug_payload = {"model": self.llm.model, "messages": messages, "tools": self._openai_tools, "temperature": 0}
        log.debug("workflow2 full payload: %s", json.dumps(debug_payload, indent=2))

        resp    = self.llm.chat(messages, tools=self._openai_tools)
        message = resp["choices"][0]["message"]
        log.debug("workflow2 raw LLM response: %s", json.dumps(message, indent=2))
        tool_calls = message.get("tool_calls")

        # if no tool calls, just a normal text reply: send it and update history, done
        if not tool_calls:
            text = message.get("content") or "(no response)"
            self.bot.send_message(chat_id, text)
            if ENABLE_HISTORY:
                history.append({"role": "user",     "content": user_text})
                history.append({"role": "assistant", "content": text})
                with self._history_lock:
                    self._history[chat_id] = history[-MAX_HISTORY:]
            return

        # step 2: convert tool call to plain text command
        call     = tool_calls[0]
        fn_name  = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])

        command_name = fn_name.replace("_", " ")
        if args:
            arg_str = " ".join(str(v) for v in args.values())
            command = f"{command_name} {arg_str}"
        else:
            command = command_name

        log.info("workflow2 tool → command: '%s'", command)

        # step 3: broadcast and wait for peer reply
        network_reply = self._broadcast_and_wait(command)

        if not network_reply:
            self.bot.send_message(chat_id, f"⏱ No peer responded to '{command}'")
            return

        log.debug("workflow2 network reply: %s", network_reply)

        # step 4: feed reply back as tool result, get final answer
        messages.append(message)
        messages.append({
            "role":         "tool",
            "tool_call_id": call["id"], # necessary for OpenAI endpoint
            "content":      network_reply,
        })

        final = self.llm.chat(messages, tools=self._openai_tools)
        reply = final["choices"][0]["message"].get("content") or "(no response)"

        log.debug("workflow2 final answer: %s", reply)
        self.bot.send_message(chat_id, reply)

        if ENABLE_HISTORY:
            history.append({"role": "user",     "content": user_text})
            history.append({"role": "assistant", "content": reply})
            with self._history_lock:
                self._history[chat_id] = history[-MAX_HISTORY:]

    def handle_incoming(self, msg: dict):
        text   = msg.get("text", "")
        sender = msg.get("from", "?")

        with self._lock:
            if self._collecting_reply:
                self._reply_buf.append(f"{sender}: {text}")
                self._reply_event.set()

    def _broadcast_and_wait(self, command: str) -> str:
        with self._lock:
            self._reply_buf.clear()
            self._collecting_reply = True
        self._reply_event.clear()

        self.broadcaster({"text": command})
        self._reply_event.wait(timeout=self.tool_timeout)

        with self._lock:
            self._collecting_reply = False
            result = list(self._reply_buf)

        return "\n".join(result)

# ── Workflow3 (split LLMs: functiongemma for dispatch, qwen2.5 for answer) ────

class Workflow3:
    """
    Same two-step tool-calling pipeline as Workflow2, but using two separate LLMs:
      - llm_dispatch  (functiongemma)  : tool selection via chat() + tool_calls
      - llm_answer    (qwen2.5:1.5b)   : final human-friendly answer via ask()

    Step 1  — llm_dispatch decides which tool to call.
    Step 2  — The tool call is converted to a plain-text broadcast command.
    Step 3  — Wait for a peer network reply.
    Step 4  — llm_answer generates the final Telegram reply from the raw result.
    """

    _ANSWER_PROMPT = """You are a helpful assistant for a smart home agent network.
    The network has already executed a command and returned real data.
    Your job: translate that raw result into a short, friendly reply for the user.
    Always reply in the same language the user wrote in.
    Only if no data are provided say you cannot answer — otherwise the data is right there in the network reply, use it."""

    def __init__(self, llm_dispatch: LLM, llm_answer: LLM, broadcaster, bot=None,
                 tool_timeout: float = 30.0):
        self.llm_dispatch  = llm_dispatch   # functiongemma  — handles tool_calls
        self.llm_answer    = llm_answer     # qwen2.5:1.5b   — handles final text
        self.broadcaster   = broadcaster
        self.bot           = bot
        self.tool_timeout  = tool_timeout

        self._lock             = threading.Lock()
        self._reply_buf        = []
        self._collecting_reply = False
        self._reply_event      = threading.Event()

        self._openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": td["function"]["name"],
                    "description": td["function"]["description"],
                    "parameters": td["function"]["parameters"],
                },
            }
            for td in TOOL_DEFS
        ]

    def run(self, chat_id: int, user_text: str):
        if self.bot:
            self.bot.start_typing_indicator(chat_id)
        try:
            self._run_inner(chat_id, user_text)
        finally:
            if self.bot:
                self.bot.stop_typing_indicator(chat_id)

    def _run_inner(self, chat_id: int, user_text: str):
        messages = [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user",   "content": user_text},
        ]

        # ── Step 1: llm_dispatch (functiongemma) selects a tool ──────────────
        debug_payload = {
            "model":       self.llm_dispatch.model,
            "messages":    messages,
            "tools":       self._openai_tools,
            "temperature": 0,
        }
        log.debug("workflow3 dispatch payload: %s", json.dumps(debug_payload, indent=2))

        resp       = self.llm_dispatch.chat(messages, tools=self._openai_tools)
        message    = resp["choices"][0]["message"]
        log.debug("workflow3 dispatch response: %s", json.dumps(message, indent=2))

        tool_calls = message.get("tool_calls")

        # ── No tool selected: llm_answer explains it gracefully ──────────────
        if not tool_calls:
            log.info("workflow3 no tool call from dispatch — delegating to llm_answer")
            answer_prompt = (
                f"User request: {user_text}\n"
                "No matching command was found for this request.\n"
                "Inform the user politely that you cannot fulfill this request with the available smart home commands."
            )
            reply = self.llm_answer.ask(self._ANSWER_PROMPT, answer_prompt)
            log.debug("workflow3 no-tool answer: %s", reply)
            if self.bot:
                self.bot.send_message(chat_id, reply)
            return

        # ── Step 2: convert tool call → plain-text broadcast command ─────────
        call     = tool_calls[0]
        fn_name  = call["function"]["name"]
        args     = json.loads(call["function"]["arguments"])

        command_name = fn_name.replace("_", " ")
        if args:
            arg_str = " ".join(str(v) for v in args.values())
            command = f"{command_name} {arg_str}"
        else:
            command = command_name

        log.info("workflow3 tool → command: '%s'", command)

        # ── Step 3: broadcast and wait for a peer reply ───────────────────────
        network_reply = self._broadcast_and_wait(command)

        if not network_reply:
            if self.bot:
                self.bot.send_message(chat_id, f"⏱ No peer responded to '{command}'")
            return

        log.debug("workflow3 network reply: %s", network_reply)

        # ── Step 4: llm_answer (qwen2.5:1.5b) generates the final reply ──────
        answer_prompt = (
            f"User request: {user_text}\n"
            f"Command executed: {command}\n"
            f"Network reply:\n{network_reply}\n"
            "Craft a short answer to the user based on the request and the network reply."
        )
        log.debug("workflow3 answer prompt: %s", answer_prompt)

        reply = self.llm_answer.ask(self._ANSWER_PROMPT, answer_prompt)
        log.debug("workflow3 final answer: %s", reply)

        if self.bot:
            self.bot.send_message(chat_id, reply)

    def handle_incoming(self, msg: dict):
        """Called from run_leader when a P2P message arrives."""
        text   = msg.get("text", "")
        sender = msg.get("from", "?")

        with self._lock:
            if self._collecting_reply:
                self._reply_buf.append(f"{sender}: {text}")
                self._reply_event.set()

    def _broadcast_and_wait(self, command: str) -> str:
        with self._lock:
            self._reply_buf.clear()
            self._collecting_reply = True
        self._reply_event.clear()

        self.broadcaster({"text": command})
        self._reply_event.wait(timeout=self.tool_timeout)

        with self._lock:
            self._collecting_reply = False
            result = list(self._reply_buf)

        return "\n".join(result)