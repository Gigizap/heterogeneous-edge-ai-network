"""
network_collector.py

Shared infrastructure for all workflows:
  - Incoming P2P message routing (skills buffer, reply buffer)
  - broadcast_and_collect(): send a command, gather replies for N seconds
  - fetch_and_merge_skills(): broadcast "fetchskills", merge deduplicated tool defs
  - tool_call_to_payload(): convert (fn_name, args) → JSON the sensing agents understand

Every workflow delegates network I/O here instead of reimplementing it.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def merge_tool_lists(raw_buffers: List[str]) -> List[Dict[str, Any]]:
    """
    Parse and deduplicate tool definitions received from the network.

    Each buffer entry may be prefixed with "sender: " (added by handle_incoming).
    The payload is expected to be a JSON array of OpenAI-style tool objects
    (with "type": "function", "function": {...}) or a single such object.
    Deduplication is by function name — last definition wins.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for entry in raw_buffers:
        # Strip optional "sender: " prefix
        text = entry.split(": ", 1)[-1].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse tool JSON from peer: %r", text[:120])
            continue

        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            # Handle OpenAI-style {"type":"function","function":{"name":...}}
            fn_block = item.get("function", {})
            fn_name = fn_block.get("name")
            if fn_name:
                merged[fn_name] = item

    return list(merged.values())


def tool_call_to_payload(fn_name: str, args: Dict[str, Any]) -> str:
    """
    Convert a parsed tool call into the JSON command payload that sensing agents expect.

    Sensing agents parse incoming messages as JSON with "name" and "arguments" keys:
        {"name": "run_till_detect", "arguments": {"object": "dog"}}
    """
    return json.dumps({"name": fn_name, "arguments": args})


def tool_call_to_plain_text(fn_name: str, args: Dict[str, Any]) -> str:
    """
    Legacy helper: convert a tool call to a plain-text command string.
    e.g. ("add_face", {"name": "Mario"}) → "add face Mario"

    Only used by Workflow1 which broadcasts plain text.
    """
    command_name = fn_name.replace("_", " ")
    if args:
        arg_str = " ".join(str(v) for v in args.values())
        return f"{command_name} {arg_str}"
    return command_name


# ── NetworkCollector ───────────────────────────────────────────────────────────

class NetworkCollector:
    """
    Centralised P2P message router and collector.

    Manages two independent collection windows:
      1. Skills collection  — activated by fetch_and_merge_skills()
      2. Reply collection   — activated by broadcast_and_collect()

    Wire handle_incoming() into the P2P transport's on_message callback.
    """

    def __init__(
        self,
        broadcaster: Callable[[Dict], None],
        skills_timeout: float = 1.5,
        reply_timeout: float = 3.0,
    ):
        self.broadcaster    = broadcaster
        self.skills_timeout = skills_timeout
        self.reply_timeout  = reply_timeout

        self._lock = threading.Lock()

        # skills collection state
        self._skills_buf: List[str] = []
        self._collecting_skills = False

        # reply collection state
        self._reply_buf: List[Dict] = []
        self._collecting_reply = False
        self._reply_event = threading.Event() #unused but could be used to unblock early when a reply is received

    def handle_incoming(self, msg: Dict) -> None:
        text = msg.get("text", "")
        sender = msg.get("from", "?")

        with self._lock:
            if self._collecting_skills:
                self._skills_buf.append(f"{sender}: {text}")   # KEEP — merge_tool_lists parses "sender: "
            if self._collecting_reply:
                self._reply_buf.append(msg)                     # store whole payload, any shape

    def fetch_and_merge_skills(self) -> List[Dict[str, Any]]:
        """
        Broadcast 'fetchskills' to all peers, wait skills_timeout seconds,
        then merge and deduplicate all received tool definitions.

        Returns a list of OpenAI-style tool defs.
        """
        with self._lock:
            self._skills_buf.clear()
            self._collecting_skills = True

        self.broadcaster({"text": "fetchskills"})
        time.sleep(self.skills_timeout)

        with self._lock:
            self._collecting_skills = False
            raw = list(self._skills_buf)

        merged = merge_tool_lists(raw)
        logger.info("[collector] %d tools discovered from network", len(merged))
        return merged

    def broadcast_and_collect(self, command: str, timeout: Optional[float] = None) -> List[Dict]:
        """
        Broadcast a command string, wait up to timeout seconds, return all replies.

        The command is sent as-is in {"text": command}.
        Returns a list of "sender: reply_text" strings.
        """
        timeout = timeout if timeout is not None else self.reply_timeout

        with self._lock:
            self._reply_buf.clear()
            self._collecting_reply = True
        # self._reply_event.clear()  ← delete this line

        self.broadcaster({"text": command})
        time.sleep(timeout)  # replaces self._reply_event.wait(timeout=timeout)

        with self._lock:
            self._collecting_reply = False
            result = list(self._reply_buf)

        return result

    def broadcast_and_collect_joined(self, command, timeout=None):
        replies = self.broadcast_and_collect(command, timeout)
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in replies)