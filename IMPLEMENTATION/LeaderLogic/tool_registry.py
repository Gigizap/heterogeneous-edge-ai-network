"""
LeaderLogic/tool_registry.py

Network tool registry, maintained by the leader (the MCP *client* side).

The leader keeps, at all times, a map of which sensing agents expose which
tools. It is fed by the session lifecycle instead of a per-request broadcast:

    agent connects / pushes its skills  -> register(agent_id, tool_defs)
    agent is lost (discovery watchdog)  -> unregister(agent_id)

From that it derives the two things the dispatch path needs:

    owners(tool_name)   -> the set of agent_ids that can run a tool
    available_tools()   -> deduplicated tool defs handed to the LLM
                           (an equally-named tool is shown only ONCE)

This replaces the old `fetchskills` broadcast in network_collector: the tool
list is always current, with no round-trip on the hot path. `fetchskills`
(now `tools/list`) stays available as a fallback but is no longer on the
critical path.

Tool defs are accepted in OpenAI function-calling shape - either nested
({"type":"function","function":{"name":...}}) or flat ({"name":...}). The
registry keeps one representative def per tool name for the LLM.
"""

import threading
from typing import Any, Dict, List, Set


class ToolRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._owners:   Dict[str, Set[str]]        = {}   # tool_name -> {agent_id}
        self._defs:     Dict[str, Dict[str, Any]]  = {}   # tool_name -> representative def
        self._by_agent: Dict[str, Set[str]]        = {}   # agent_id  -> {tool_name}

    # ── feed from the session lifecycle ──────────────────────────────────────

    def register(self, agent_id: str, tool_defs: List[Dict[str, Any]]) -> None:
        """(Re)register the full tool list for one agent.

        Idempotent: replaces whatever this agent exposed before, so a device
        that changes its skills (or re-announces) is handled cleanly.
        """
        names: Set[str] = set()
        defs:  Dict[str, Dict[str, Any]] = {}
        for td in tool_defs or []:
            fn   = td.get("function", td)        # accept nested or flat
            name = fn.get("name")
            if not name:
                continue
            names.add(name)
            defs[name] = td

        with self._lock:
            self._remove_agent_locked(agent_id)  # drop prior tools first
            self._by_agent[agent_id] = names
            for name in names:
                self._owners.setdefault(name, set()).add(agent_id)
                self._defs.setdefault(name, defs[name])   # first owner's def represents the tool

    def unregister(self, agent_id: str) -> None:
        """Drop an agent (peer lost). A tool whose last owner just left is
        removed entirely, so available_tools() never lists a dead tool."""
        with self._lock:
            self._remove_agent_locked(agent_id)

    def _remove_agent_locked(self, agent_id: str) -> None:
        for name in self._by_agent.pop(agent_id, set()):
            owners = self._owners.get(name)
            if owners is None:
                continue
            owners.discard(agent_id)
            if not owners:                        # last owner gone -> drop the key
                self._owners.pop(name, None)
                self._defs.pop(name, None)

    # ── read side (dispatch + LLM) ───────────────────────────────────────────

    def owners(self, tool_name: str) -> Set[str]:
        with self._lock:
            return set(self._owners.get(tool_name, ()))

    def available_tools(self) -> List[Dict[str, Any]]:
        """Deduplicated tool defs for the LLM (one entry per tool name)."""
        with self._lock:
            return list(self._defs.values())

    def tool_names(self) -> Set[str]:
        with self._lock:
            return set(self._owners.keys())
