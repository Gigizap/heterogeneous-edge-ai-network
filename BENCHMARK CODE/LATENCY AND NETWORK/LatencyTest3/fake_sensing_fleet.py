#!/usr/bin/env python3
import sys
import pathlib

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

PORT_BASE     = 5000
FLEET_SIZES   = (1, 10, 20, 50, 100)
WAVE_INTERVAL = 60.0

SHARED_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "Placeholder tool for the network-overlay scaling test. Never dispatched.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

def _make_handler(transport: P2PTransport):
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
    agents: dict = {}

    try:
        current = 0
        for target in FLEET_SIZES:
            for port in range(PORT_BASE + current + 1, PORT_BASE + target + 1):
                add_agent(loop, port, agents)
            current = target
            log.info("===== fleet now at %d agent(s) (ports %d..%d) - holding %.0f min =====",
                     current, PORT_BASE + 1, PORT_BASE + current, WAVE_INTERVAL / 60)
            time.sleep(WAVE_INTERVAL)
        stop_all(loop, agents)
        log.info("run complete - all agents stopped")
    except KeyboardInterrupt:
        log.info("interrupted - stopping all agents")
        stop_all(loop, agents)

if __name__ == "__main__":
    main()
