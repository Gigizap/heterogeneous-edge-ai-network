#!/usr/bin/env python3
"""
LatencyTest3/fake_sensing_fleet.py - run on this laptop.

SECOND latency test, SENSING side: spawns a GROWING fleet of FAKE sensing agents
that only announce their presence (live Discovery HELLO) and serve tools/list
with ONE shared placeholder tool. They run NO real skill and do NO tool work, so
the only thing the mock STM32 leader (mock_leader_stm32.py) does in response is
the network overlay - discover + aggregate. That is exactly what the board-side
power measurement isolates: no read/write ops from tool execution interfere.

Each agent gets its own TCP port and a unique agent-id, but ALL of them expose
the SAME single tool ("noop"). Agent i listens on port 5000+i, so the full fleet
spans ports 5001..5100.

Schedule (CUMULATIVE - agent #1 stays up the whole run; each step adds the delta):

    1 agent  -> wait 1 min -> 10 -> wait 1 min -> 20 -> wait 1 min
             -> 50 -> wait 1 min -> 100 -> wait 1 min -> STOP ALL

On stop, every agent's TCP server is closed and announcing halts, so the leader's
discovery watchdog (PEER_TIMEOUT = 15 s) drops them shortly after - a clean
end-of-run signal on the board side.

Run from the IMPLEMENTATION folder (start the STM32 leader FIRST):
    python LatencyTest3/fake_sensing_fleet.py

The laptop is not power-measured, so this side can be as busy as it needs to be
(100 co-located Discovery instances all announcing + listening on UDP 9999).
"""

import sys
import pathlib

# OS-agnostic imports: put the IMPLEMENTATION dir (parent) on the path so
# "from ConnectionLogic..." resolves no matter the working directory.
_HERE = pathlib.Path(__file__).resolve().parent
_IMPL = _HERE.parent
for _p in (str(_IMPL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
import logging
import time

from ConnectionLogic.transport import P2PTransport
from ConnectionLogic.discovery import Discovery
from utils import start_async_loop, schedule

log = logging.getLogger("latencytest3")

PORT_BASE     = 5000               # agent i -> port PORT_BASE + i (5001..5100)
FLEET_SIZES   = (1, 10, 20, 50, 100)
WAVE_INTERVAL = 60.0               # seconds to hold each size (1 minute)

# The one tool every agent advertises. A no-op with empty params: it is never
# dispatched, it only has to make each agent a tool OWNER in the leader's registry.
SHARED_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "Placeholder tool for the network-overlay scaling test. Never dispatched.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def _make_handler(transport: P2PTransport):
    """A tools/list responder bound to one agent's transport. Fake agents only
    advertise, so every other message (discovery/election/backup, tools/call) is
    ignored - the leader in this test never calls a tool."""
    def on_message(msg: dict):
        if msg.get("type") or msg.get("method") != "tools/list":
            return
        sender = msg.get("from")
        rpc_id = msg.get("id")
        if sender is None or rpc_id is None:
            return
        try:
            transport.send_sync(sender, {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {"tools": [SHARED_TOOL]},
            })
        except Exception as e:
            log.warning("%s could not reply tools/list to %s: %r",
                        transport.agent_id, sender, e)
    return on_message


def add_agent(loop, port: int, agents: dict) -> None:
    """Bring up one fake sensing agent: a TCP transport that serves tools/list and
    a Discovery that announces the agent + finds the leader (so replies can route)."""
    aid = f"fake-sensing-{port}"
    transport = P2PTransport(aid, port, lambda m: None)
    transport._event_loop = loop
    transport.on_message = _make_handler(transport)
    discovery = Discovery(
        aid, port,
        on_peer_found=lambda pid, ip, p: schedule(loop, transport.register_peer(pid, ip, p)),
        on_peer_lost=lambda pid: schedule(loop, transport.unregister_peer(pid)),
    )
    schedule(loop, transport.start())
    schedule(loop, discovery.start())
    agents[port] = (aid, transport, discovery)


def stop_all(loop, agents: dict) -> None:
    """Cancel every agent's tasks (TCP servers + announce/listen/watchdog loops)
    and stop the event loop. Halting the announcements is what makes the leader
    drop the peers on its watchdog timeout. Tasks are awaited after cancellation
    so the loop stops cleanly (no 'task destroyed while pending' noise)."""
    log.info("stopping all %d agents (halting servers + announcements) ...", len(agents))

    async def _shutdown():
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    try:
        asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=10)
    except Exception as e:
        log.warning("shutdown wait: %r", e)
    loop.call_soon_threadsafe(loop.stop)
    time.sleep(0.3)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s]  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    loop = start_async_loop()
    agents: dict = {}   # port -> (agent_id, transport, discovery)

    try:
        current = 0
        for target in FLEET_SIZES:
            for port in range(PORT_BASE + current + 1, PORT_BASE + target + 1):
                add_agent(loop, port, agents)
            current = target
            log.info("===== fleet now at %d agent(s) (ports %d..%d) - holding %.0f min =====",
                     current, PORT_BASE + 1, PORT_BASE + current, WAVE_INTERVAL / 60)
            time.sleep(WAVE_INTERVAL)     # hold this size (incl. after the final 100)
        stop_all(loop, agents)
        log.info("run complete - all agents stopped")
    except KeyboardInterrupt:
        log.info("interrupted - stopping all agents")
        stop_all(loop, agents)


if __name__ == "__main__":
    main()
