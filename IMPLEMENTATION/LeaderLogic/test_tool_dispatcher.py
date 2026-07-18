"""
Stdlib-only tests (no third-party deps) for ToolRegistry, ToolDispatcher, and
the LeaderNetwork wiring.

Run from the IMPLEMENTATION/ directory:
    python -m unittest LeaderLogic.test_tool_dispatcher
"""

import threading
import time
import unittest

from LeaderLogic.tool_registry import ToolRegistry
from LeaderLogic.tool_dispatcher import ToolDispatcher
from LeaderLogic.leader_network import make_network


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": name,
                                             "parameters": {"type": "object", "properties": {}}}}


def _result(text):
    return {"content": [{"type": "text", "text": text}]}


class FakeRegistry:
    """Minimal registry: tool_name -> set(agent_ids)."""
    def __init__(self, mapping):
        self._m = {k: set(v) for k, v in mapping.items()}
    def owners(self, name):
        return set(self._m.get(name, ()))


class _Harness:
    """Captures sends and lets the test feed replies back by rpc id."""
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.sends = []
        self._lock = threading.Lock()

    def send(self, agent_id, message):
        with self._lock:
            self.sends.append((agent_id, message))

    def wait_for_sends(self, n, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if len(self.sends) >= n:
                    return list(self.sends)
            time.sleep(0.005)
        raise AssertionError(f"only {len(self.sends)} sends, expected {n}")

    def reply(self, rpc_id, text, sender):
        self.dispatcher.on_reply({"jsonrpc": "2.0", "id": rpc_id,
                                  "result": _result(text), "from": sender})


def _run(fn):
    box = {}
    t = threading.Thread(target=lambda: box.update(r=fn()))
    t.start()
    return t, box


# ── ToolRegistry ────────────────────────────────────────────────────────────

class TestToolRegistry(unittest.TestCase):
    def test_register_owners_and_dedup(self):
        r = ToolRegistry()
        r.register("pi", [_tool("person_detection"), _tool("add_face")])
        r.register("stm", [_tool("person_detection")])
        self.assertEqual(r.owners("person_detection"), {"pi", "stm"})
        self.assertEqual(r.owners("add_face"), {"pi"})
        names = sorted(t["function"]["name"] for t in r.available_tools())
        self.assertEqual(names, ["add_face", "person_detection"])   # dedup: once each

    def test_unregister_drops_empty_tool(self):
        r = ToolRegistry()
        r.register("pi", [_tool("person_detection")])
        r.register("stm", [_tool("person_detection"), _tool("add_face")])
        r.unregister("stm")
        self.assertEqual(r.owners("person_detection"), {"pi"})
        self.assertEqual(r.owners("add_face"), set())               # last owner gone
        self.assertNotIn("add_face", r.tool_names())                # key removed entirely

    def test_reregister_replaces(self):
        r = ToolRegistry()
        r.register("pi", [_tool("a"), _tool("b")])
        r.register("pi", [_tool("a")])                              # b dropped
        self.assertEqual(r.owners("a"), {"pi"})
        self.assertEqual(r.owners("b"), set())


# ── ToolDispatcher ──────────────────────────────────────────────────────────

class TestToolDispatcher(unittest.TestCase):
    def test_fan_out_to_all_owners_concatenated(self):
        d = ToolDispatcher(FakeRegistry({"person_detection": ["pi", "stm"]}),
                           send=None, timeout=1.0)
        h = _Harness(d); d._send = h.send
        t, box = _run(lambda: d.dispatch("person_detection", {}))
        sends = h.wait_for_sends(2)
        self.assertEqual({a for a, _ in sends}, {"pi", "stm"})
        for agent, msg in sends:
            self.assertEqual(msg["method"], "tools/call")
            h.reply(msg["id"], "seen by " + agent, agent)
        t.join(2.0)
        self.assertEqual(sorted(r["text"] for r in box["r"]), ["seen by pi", "seen by stm"])

    def test_missing_owner_marked_did_not_reply(self):
        d = ToolDispatcher(FakeRegistry({"x": ["pi", "stm"]}), send=None, timeout=0.2)
        h = _Harness(d); d._send = h.send
        t, box = _run(lambda: d.dispatch("x", {}))
        sends = h.wait_for_sends(2)
        pi_msg = next(m for a, m in sends if a == "pi")
        h.reply(pi_msg["id"], "3 people", "pi")
        t.join(2.0)
        got = {r["from"]: r["text"] for r in box["r"]}
        self.assertEqual(got["pi"], "3 people")
        self.assertEqual(got["stm"], "did not reply")

    def test_finishes_early_when_all_reply(self):
        d = ToolDispatcher(FakeRegistry({"x": ["pi", "stm"]}), send=None, timeout=5.0)
        h = _Harness(d); d._send = h.send
        start = time.monotonic()
        t, box = _run(lambda: d.dispatch("x", {}))
        for a, msg in h.wait_for_sends(2):
            h.reply(msg["id"], "fast", a)
        t.join(3.0)
        self.assertLess(time.monotonic() - start, 2.0)              # nowhere near the 5s cap
        self.assertFalse(t.is_alive())

    def test_no_owner_returns_empty(self):
        d = ToolDispatcher(FakeRegistry({}), send=lambda *_: None, timeout=0.2)
        self.assertEqual(d.dispatch("ghost", {}), [])

    def test_late_reply_after_cap_is_ignored(self):
        d = ToolDispatcher(FakeRegistry({"x": ["pi"]}), send=None, timeout=0.15)
        h = _Harness(d); d._send = h.send
        t, box = _run(lambda: d.dispatch("x", {}))
        sends = h.wait_for_sends(1)
        t.join(2.0)                                                 # cap elapses, pi silent
        self.assertEqual(box["r"][0]["text"], "did not reply")
        h.reply(sends[0][1]["id"], "too late", "pi")               # must be safely ignored


# ── LeaderNetwork (pull-on-join registry + dispatch wired together) ─────────

class FakeTransport:
    def __init__(self):
        self.on_message = None
        self.sends = []
        self._lock = threading.Lock()
    def send_sync(self, peer_id, msg):
        with self._lock:
            self.sends.append((peer_id, msg))
    def deliver(self, msg):
        self.on_message(msg)
    def wait_for(self, pred, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                hit = next((s for s in self.sends if pred(s)), None)
            if hit:
                return hit
            time.sleep(0.005)
        raise AssertionError("expected send not seen")


class FakeDiscovery:
    def __init__(self):
        self.on_peer_found = None
        self.on_peer_lost = None
        self._peers = {}


class TestLeaderNetwork(unittest.TestCase):
    def test_pull_register_dispatch_unregister(self):
        t = FakeTransport()
        d = FakeDiscovery()
        net = make_network(t, d, reply_timeout=0.5)

        # peer joins -> leader pulls its tools with tools/list
        d.on_peer_found("pi", "1.2.3.4", 5555)
        peer, req = t.wait_for(lambda s: s[1].get("method") == "tools/list")
        self.assertEqual(peer, "pi")

        # tools/list reply registers the tool
        t.deliver({"jsonrpc": "2.0", "id": req["id"],
                   "result": {"tools": [_tool("person_detection")]}, "from": "pi"})
        self.assertEqual({tt["function"]["name"] for tt in net.available_tools()},
                         {"person_detection"})

        # dispatch fans out to the owner and gathers its reply
        th, box = _run(lambda: net.dispatch("person_detection", {}))
        peer2, call = t.wait_for(lambda s: s[1].get("method") == "tools/call")
        self.assertEqual(peer2, "pi")
        t.deliver({"jsonrpc": "2.0", "id": call["id"],
                   "result": _result("3 people"), "from": "pi"})
        th.join(2.0)
        self.assertEqual(box["r"][0]["text"], "3 people")

        # peer lost -> its tool disappears
        d.on_peer_lost("pi")
        self.assertEqual(net.available_tools(), [])


if __name__ == "__main__":
    unittest.main()
