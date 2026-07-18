#!/usr/bin/env python3
"""
LatencyTest3/mock_leader_stm32.py - run on the STM32MP257F-DK (or both roles on
one machine for a local dry-run).

SECOND latency test, LEADER side: a MOCK leader that runs the LIVE network
overlay (Discovery + LeaderNetwork: tool registry + pull-on-join) but NO LLM,
NO election / polling, NO Telegram, and NO tool dispatch. Its only job is to
discover the fake sensing fleet (fake_sensing_fleet.py, on a laptop) and
aggregate their tools, so an EXTERNAL power meter on the board can capture how
the network overlay's power draw scales as the fleet grows:

    1 -> 10 -> 20 -> 50 -> 100 sensing agents

Because the whole point is to isolate the network-overlay cost, the leader does
as little else as possible:
  - every fleet agent exposes ONE shared placeholder tool, so there is no
    per-tool read/write work and no dispatch is ever issued;
  - the timeline JSON is written only when the observed fleet size CHANGES (a
    handful of writes), never on a hot loop, so disk I/O does not pollute the
    power trace;
  - a light heartbeat is logged every few seconds only to time-align the log
    against the external power capture.

LatencyTest3 addition: the leader logs EVERY JSON message it receives on its TCP
port (LEADER_PORT = 5700) at INFO - the fleet's tools/list replies and anything
else that arrives - so the application-level traffic pushed through the leader
port is visible message-by-message as the fleet scales. Pair this with an
external packet monitor (Wireshark/TShark filter tcp.port==5700) for the on-wire
byte totals; this log is the app-level ground truth.

This deliberately does NOT reuse LeaderLogic/run_mock_leader.py: that mock builds
NO LeaderNetwork and echoes an LLM stub, which is the opposite of what we need
here (we want the live aggregation path, and no LLM).

Run from the IMPLEMENTATION folder (nothing to type during the run):
    python LatencyTest3/mock_leader_stm32.py

Output: LatencyTest3/results/leader_stm32_<ts>.json
  - found_at_s / registered_at_s   per-agent discovery + tool-registration times
  - reg_latency_stats              found -> tool-registered latency (count/mean/std/min/max)
  - timeline                       (elapsed, peers, tool_owners) at each fleet-size change
"""

import sys
import pathlib

# OS-agnostic imports: put the IMPLEMENTATION dir (parent) and this dir on the
# path so "from ConnectionLogic..." resolves no matter the working directory.
_HERE = pathlib.Path(__file__).resolve().parent
_IMPL = _HERE.parent
for _p in (str(_IMPL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import statistics
import threading
import time
from pathlib import Path

from ConnectionLogic.transport import P2PTransport
from ConnectionLogic.discovery import Discovery
from LeaderLogic.leader_network import make_network
from utils import start_async_loop, schedule

log = logging.getLogger("latencytest3")

LEADER_ID       = "stm32-mock-leader"
LEADER_PORT     = 5700    # distinct from the fleet's 5001..5100; discovery is UDP 9999 for all
SAMPLE_INTERVAL = 5.0     # heartbeat log cadence (s) - only to time-align the power trace


def _stats(values, ndigits: int) -> dict:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": round(statistics.fmean(vals), ndigits),
        "std": round(statistics.pstdev(vals), ndigits) if len(vals) > 1 else 0.0,
        "min": round(min(vals), ndigits),
        "max": round(max(vals), ndigits),
    }


def _results_dir() -> Path:
    d = _HERE / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s]  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- live network overlay: transport + discovery on one background loop ------
    loop = start_async_loop()
    transport = P2PTransport(LEADER_ID, LEADER_PORT, lambda msg: None)
    transport._event_loop = loop
    discovery = Discovery(
        LEADER_ID, LEADER_PORT,
        on_peer_found=lambda pid, ip, p: schedule(loop, transport.register_peer(pid, ip, p)),
        on_peer_lost=lambda pid: schedule(loop, transport.unregister_peer(pid)),
    )
    schedule(loop, transport.start())
    schedule(loop, discovery.start())

    # -- live leader tool-aggregation path (pull-on-join, never dispatches) ------
    net = make_network(transport, discovery)

    # -- LatencyTest3: log EVERY JSON the leader receives on its TCP port ---------
    # make_network set transport.on_message to the LeaderNetwork handler; wrap it
    # so every message that arrives on LEADER_PORT (the fleet's tools/list replies,
    # and anything else) is logged before being handled. This is the whole point
    # of this variant: watch the application-level traffic through the leader port
    # grow message-by-message as the fleet scales 1 -> 10 -> 20 -> 50 -> 100.
    _downstream = transport.on_message

    def _log_and_handle(msg: dict):
        raw = json.dumps(msg, ensure_ascii=False)
        log.info("RX port %d  ~%d B  from=%s  %s",
                 LEADER_PORT, len(raw.encode("utf-8")) + 1, msg.get("from"), raw)
        _downstream(msg)

    transport.on_message = _log_and_handle

    # -- lightweight instrumentation: per-agent discovery -> tool-registered -----
    lock = threading.Lock()
    t0 = time.perf_counter()
    found_at:      dict[str, float] = {}   # agent_id -> elapsed s at first HELLO seen
    registered_at: dict[str, float] = {}   # agent_id -> elapsed s at first tool registered

    # make_network already wrapped discovery.on_peer_found (to fire the tools/list
    # pull); wrap that wrapper so we stamp the moment the peer is discovered.
    prev_found = discovery.on_peer_found

    def on_found(pid, ip, port):
        prev_found(pid, ip, port)                 # keeps the LeaderNetwork pull-on-join
        with lock:
            found_at.setdefault(pid, round(time.perf_counter() - t0, 3))

    discovery.on_peer_found = on_found

    # registry.register is called by LeaderNetwork on each tools/list reply; wrap
    # the instance method so we stamp the moment a peer's tool is aggregated.
    _register = net.registry.register

    def register(agent_id, tool_defs):
        _register(agent_id, tool_defs)
        with lock:
            registered_at.setdefault(agent_id, round(time.perf_counter() - t0, 3))

    net.registry.register = register

    # -- output (written only when the fleet size changes, plus at exit) ---------
    out = _results_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"leader_stm32_{stamp}.json"
    timeline: list[dict] = []

    def snapshot(reason: str) -> None:
        with lock:
            fa = dict(found_at)
            ra = dict(registered_at)
        names = net.registry.tool_names()
        owners = {n: len(net.registry.owners(n)) for n in names}
        timeline.append({
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "reason": reason,
            "peers": len(getattr(transport, "_peers", {})),
            "distinct_tools": len(names),
            "tool_owners": owners,
        })
        latencies = [round(ra[a] - fa[a], 3) for a in fa if a in ra]
        data = {
            "role": "mock_leader",
            "device": "stm32",
            "leader_id": LEADER_ID,
            "leader_port": LEADER_PORT,
            "note": ("network-overlay scaling / power test; single shared placeholder "
                     "tool; no LLM, no dispatch, no election"),
            "timeline": timeline,
            "found_at_s": fa,
            "registered_at_s": ra,
            "reg_latency_unit": "seconds",
            "reg_latency_stats": _stats(latencies, 3),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("mock leader up on port %d as %s - waiting for the fake sensing fleet ...",
             LEADER_PORT, LEADER_ID)
    log.info("results -> %s", path)

    last_size = -1
    try:
        while True:
            time.sleep(SAMPLE_INTERVAL)
            peers = len(getattr(transport, "_peers", {}))
            owners_total = sum(len(net.registry.owners(n)) for n in net.registry.tool_names())
            log.info("t=+%6.0fs  peers=%-3d  tool_owners=%-3d  distinct_tools=%d",
                     time.perf_counter() - t0, peers, owners_total,
                     len(net.registry.tool_names()))
            if peers != last_size:                 # size change -> snapshot (rare write)
                snapshot(reason=f"fleet size -> {peers}")
                last_size = peers
    except KeyboardInterrupt:
        log.info("interrupted - writing final snapshot")
        snapshot(reason="final")
        log.info("done -> %s", path)


if __name__ == "__main__":
    main()
