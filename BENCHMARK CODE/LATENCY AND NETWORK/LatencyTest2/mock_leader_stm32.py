#!/usr/bin/env python3
import sys
import pathlib

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

log = logging.getLogger("latencytest2")

LEADER_ID       = "stm32-mock-leader"
LEADER_PORT     = 5700
SAMPLE_INTERVAL = 5.0

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

    net = make_network(transport, discovery)

    lock = threading.Lock()
    t0 = time.perf_counter()
    found_at:      dict[str, float] = {}
    registered_at: dict[str, float] = {}

    prev_found = discovery.on_peer_found

    def on_found(pid, ip, port):
        prev_found(pid, ip, port)
        with lock:
            found_at.setdefault(pid, round(time.perf_counter() - t0, 3))

    discovery.on_peer_found = on_found

    _register = net.registry.register

    def register(agent_id, tool_defs):
        _register(agent_id, tool_defs)
        with lock:
            registered_at.setdefault(agent_id, round(time.perf_counter() - t0, 3))

    net.registry.register = register

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
            if peers != last_size:
                snapshot(reason=f"fleet size -> {peers}")
                last_size = peers
    except KeyboardInterrupt:
        log.info("interrupted - writing final snapshot")
        snapshot(reason="final")
        log.info("done -> %s", path)

if __name__ == "__main__":
    main()
