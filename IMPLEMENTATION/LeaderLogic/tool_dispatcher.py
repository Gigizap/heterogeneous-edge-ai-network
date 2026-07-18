"""
LeaderLogic/tool_dispatcher.py

Tool dispatch for the leader (the MCP *client* side).

The LLM picks ONE tool for a user turn. The dispatcher:

  1. resolves that tool to the set of agents that own it (ToolRegistry);
  2. sends the call to ALL of them at once, as JSON-RPC `tools/call` requests,
     each tagged with its own correlation id;
  3. gathers replies BY ID — never by a time window — with a flat timeout cap
     (default 3s), finishing as soon as every owner has replied;
  4. returns one {from, text} per owner. An owner that did not answer within the
     cap is recorded as {from, text: "did not reply"}, so the answer LLM always
     sees one concatenated result set.

It owns no transport: `send` is injected, and `on_reply` is wired into the
transport's message handler. Because replies are matched purely by id,
concurrent / late / stray messages can never cross requests.

Wire shape sent to a sensing agent (one per owner):
    {"jsonrpc":"2.0","id":"rpc-7","method":"tools/call",
     "params":{"name":"person_detection","arguments":{}}}

Reply shape expected back (`from` auto-stamped by the transport):
    {"jsonrpc":"2.0","id":"rpc-7","result":{"content":[{"type":"text","text":"3"}]},"from":"pi-1"}
    {"jsonrpc":"2.0","id":"rpc-7","error":{"code":-32601,"message":"..."},"from":"pi-1"}
"""

import itertools
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class _Batch:
    """Mutable state for one dispatch, guarded by `cond`."""
    __slots__ = ("cond", "expected", "results", "deadline")

    def __init__(self, deadline: float):
        self.cond     = threading.Condition()
        self.expected: set                       = set()   # rpc_ids still awaited
        self.results:  Dict[str, Dict[str, Any]] = {}      # rpc_id -> {from, text}
        self.deadline = deadline                           # monotonic; shared by all owners


class ToolDispatcher:
    def __init__(
        self,
        registry,
        send: Callable[[str, dict], None],
        timeout: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._registry = registry
        self._send     = send
        self._timeout  = timeout
        self._clock    = clock

        self._lock     = threading.Lock()
        self._inflight: Dict[str, _Batch] = {}    # rpc_id -> batch (routes replies)
        self._ids      = itertools.count(1)       # sequential, unique correlation ids

    def _next_id(self) -> str:
        return f"rpc-{next(self._ids)}"

    # ── dispatch (blocking, one tool call) ───────────────────────────────────

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Send `name` to every agent that owns it and block until all reply or
        the timeout cap elapses.

        Returns [{from, text}, ...] — one per owner, with silent owners marked
        "did not reply". Returns [] when no agent owns the tool.
        """
        owners = sorted(self._registry.owners(name))
        if not owners:
            return []

        arguments = arguments or {}
        routing: Dict[str, str] = {}              # rpc_id -> agent_id
        for agent_id in owners:
            routing[self._next_id()] = agent_id

        batch = _Batch(deadline=self._clock() + self._timeout)
        with batch.cond:
            batch.expected.update(routing)

        # Register for reply routing BEFORE sending, so a fast reply can't race
        # in before we are ready to record it.
        with self._lock:
            for rpc_id in routing:
                self._inflight[rpc_id] = batch

        for rpc_id, agent_id in routing.items():
            message = {"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call",
                       "params": {"name": name, "arguments": arguments}}
            try:
                self._send(agent_id, message)
            except Exception as e:
                logger.warning("send to %s failed: %s", agent_id, e)
                with batch.cond:
                    batch.results[rpc_id] = {"from": agent_id, "text": "send failed"}
                    batch.expected.discard(rpc_id)
                    batch.cond.notify_all()

        with batch.cond:
            while batch.expected:
                remaining = batch.deadline - self._clock()
                if remaining <= 0:
                    break
                batch.cond.wait(timeout=remaining)

        replies = []
        with batch.cond:
            for rpc_id, agent_id in routing.items():
                payload = batch.results.get(rpc_id)
                replies.append(payload if payload is not None
                               else {"from": agent_id, "text": "did not reply"})

        with self._lock:
            for rpc_id in routing:
                self._inflight.pop(rpc_id, None)

        return replies

    # ── reply intake (wire into transport.on_message) ────────────────────────

    def on_reply(self, msg: dict) -> bool:
        """Record a tools/call reply, matched by id.

        Returns True if the message was a reply this dispatcher was waiting for
        (so the caller can stop processing it), False otherwise.
        """
        rpc_id = msg.get("id")
        if rpc_id is None:
            return False
        with self._lock:
            batch = self._inflight.get(rpc_id)
        if batch is None:
            return False                          # not ours / already finalised / stray
        with batch.cond:
            if rpc_id not in batch.expected:
                return False
            if self._clock() > batch.deadline:    # arrived after the cap: count as silent
                batch.expected.discard(rpc_id)
                batch.cond.notify_all()
                return True
            reply = self._extract_reply(msg)
            batch.results[rpc_id] = reply
            logger.info("tool reply from %s: %r", reply.get("from"), reply.get("text"))
            batch.expected.discard(rpc_id)
            batch.cond.notify_all()
        return True

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_reply(msg: dict) -> Dict[str, Any]:
        sender = msg.get("from")
        err    = msg.get("error")
        if err:
            detail = err.get("message", err) if isinstance(err, dict) else err
            return {"from": sender, "text": f"error: {detail}"}
        return {"from": sender, "text": ToolDispatcher._text_from_result(msg.get("result"))}

    @staticmethod
    def _text_from_result(result: Any) -> str:
        """Pull plain text out of an MCP tools/call result (or tolerate a bare
        string / {"text": ...} object)."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                return "\n".join(p for p in parts if p)
            if "text" in result:
                return str(result["text"])
            return json.dumps(result, ensure_ascii=False)
        return str(result)
