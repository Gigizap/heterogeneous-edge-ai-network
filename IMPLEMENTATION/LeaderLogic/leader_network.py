"""
LeaderLogic/leader_network.py

The leader's network-facing object — the replacement for NetworkCollector.

It bundles the two pieces the pipeline needs and wires them into the leader's
transport + discovery:

  * ToolRegistry    — the live {tool_name: {agent_ids}} map.
  * ToolDispatcher  — sends a picked tool to its owners and gathers replies
                      (id-correlated, flat timeout cap).

Skill discovery is pull-on-join, not per-call: when the leader's discovery sees
a peer, the leader asks it for its tools (`tools/list`, retrying until it
answers) and registers the reply; when a peer is lost, its tools are dropped
(and a tool whose last owner left disappears). So the registry is always current
and NO fetch happens on the hot path (`available_tools()` is a local lookup).

Transport message routing (wired ahead of the leader's own handler):
  * a `tools/call` reply (has an id we issued)  -> ToolDispatcher.on_reply
  * a `tools/list` reply (result has "tools")   -> ToolRegistry.register
  * everything else                             -> passed through unchanged
"""

import itertools
import logging
import threading

from LeaderLogic.tool_registry import ToolRegistry
from LeaderLogic.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

# tools/list retry offsets (seconds). A freshly joined peer may not have
# discovered the leader back yet, so its first reply can fail silently: replies
# open a NEW TCP connection, which needs the leader in the peer's routing table.
# Retrying until the peer answers self-heals that bidirectional-discovery race.
_PULL_RETRIES = (0.0, 2.0, 5.0)


class LeaderNetwork:
    def __init__(self, transport, discovery, reply_timeout: float = 3.0):
        self.registry   = ToolRegistry()
        self.dispatcher = ToolDispatcher(self.registry,
                                         send=transport.send_sync,
                                         timeout=reply_timeout)
        self._transport = transport
        self._list_ids  = itertools.count(1)
        self._pulled    = set()      # agents whose tools we've registered

        self._wire_transport(transport)
        self._wire_discovery(discovery)

        # Pull tools from peers the leader already knows about at boot time
        # (discovery may have populated the table before we got here).
        for pid in list(getattr(discovery, "_peers", {})):
            self._pull_tools(pid)

    # ── pipeline interface (what BaseWorkflow calls) ─────────────────────────

    def available_tools(self):
        """Deduplicated tool defs for the LLM — a local lookup, no network."""
        return self.registry.available_tools()

    def dispatch(self, name, arguments):
        """Send the picked tool to its owners; return [{from, text}, ...]."""
        return self.dispatcher.dispatch(name, arguments)

    # ── skill discovery ──────────────────────────────────────────────────────

    def _request_tools(self, peer_id: str):
        # tools/list ids use a distinct prefix so the dispatcher (which only
        # claims ids it issued) never mistakes a tools/list reply for a
        # tools/call reply.
        self._transport.send_sync(peer_id, {
            "jsonrpc": "2.0",
            "id":      f"list-{next(self._list_ids)}",
            "method":  "tools/list",
        })

    def _pull_tools(self, peer_id: str):
        """Ask a peer for its tools, retrying on the _PULL_RETRIES schedule until
        it answers. Stops early once its tools are registered (see _pulled), so a
        peer that replies on the first try costs exactly one request."""
        def attempt(i: int):
            if peer_id in self._pulled or i >= len(_PULL_RETRIES):
                return
            self._request_tools(peer_id)
            if i + 1 < len(_PULL_RETRIES):
                timer = threading.Timer(_PULL_RETRIES[i + 1] - _PULL_RETRIES[i],
                                        attempt, args=(i + 1,))
                timer.daemon = True
                timer.start()
        attempt(0)

    def _wire_transport(self, transport):
        prev = transport.on_message

        def on_message(msg: dict):
            if self.dispatcher.on_reply(msg):          # tools/call reply (by id)
                return
            result = msg.get("result")
            if isinstance(result, dict) and "tools" in result:   # tools/list reply
                sender = msg.get("from")
                self.registry.register(sender, result.get("tools") or [])
                self._pulled.add(sender)                          # stop retrying this peer
                logger.info("registered %d tool(s) from %s",
                            len(result.get("tools") or []), sender)
                return
            if prev:
                prev(msg)

        transport.on_message = on_message

    def _wire_discovery(self, discovery):
        prev_found = discovery.on_peer_found
        prev_lost  = discovery.on_peer_lost

        def on_found(pid, ip, port):
            if prev_found:
                prev_found(pid, ip, port)
            self._pull_tools(pid)                      # welcome its tools (with retries)

        def on_lost(pid):
            if prev_lost:
                prev_lost(pid)
            self._pulled.discard(pid)                  # so a rejoin re-pulls
            self.registry.unregister(pid)              # drop its tools (last owner -> tool gone)

        discovery.on_peer_found = on_found
        discovery.on_peer_lost  = on_lost


def make_network(transport, discovery, reply_timeout: float = 3.0) -> LeaderNetwork:
    """Build the leader's LeaderNetwork and wire it into transport + discovery."""
    return LeaderNetwork(transport, discovery, reply_timeout=reply_timeout)
