"""
run_agents.py — fully self-contained. One file, no project imports.

Spawns N agents in one process. Each agent:
  * runs its own P2PTransport + Discovery on its own asyncio loop/thread
  * prints every message it receives
  * every 5s sends {"text": "HI"} to a peer named "leader"
    - if no peer "leader" is known, prints "waiting for leader"

    python run_agents.py 3     # 3 agents (Agent-test-1 .. Agent-test-3)
    python run_agents.py       # defaults to 3

Ctrl-C stops everything.
"""

import asyncio
import json
import socket
import sys
import threading
import time

# ── Discovery (UDP presence broadcast) ───────────────────────────────────────

DISCOVERY_PORT = 9999
BROADCAST_ADDR = "192.168.1.255"
ANNOUNCE_INTERVAL = 5
PEER_TIMEOUT = 15


class Discovery:
    def __init__(self, agent_id, tcp_port, on_peer_found, on_peer_lost):
        self.agent_id = agent_id
        self.tcp_port = tcp_port
        self.on_peer_found = on_peer_found
        self.on_peer_lost = on_peer_lost
        self._peers = {}
        self._lock = asyncio.Lock()

    async def start(self):
        try:
            await asyncio.gather(
                self._announce_loop(),
                self._listen_loop(),
                self._watchdog_loop(),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            pass

    async def _announce_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)
        msg = json.dumps({"type": "HELLO", "id": self.agent_id, "port": self.tcp_port}).encode()
        loop = asyncio.get_event_loop()
        while True:
            try:
                await loop.sock_sendto(sock, msg, (BROADCAST_ADDR, DISCOVERY_PORT))
            except Exception as e:
                print(f"announce error: {e}")
            await asyncio.sleep(ANNOUNCE_INTERVAL)

    async def _listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", DISCOVERY_PORT))
        sock.setblocking(False)
        loop = asyncio.get_event_loop()
        while True:
            try:
                data, addr = await loop.sock_recvfrom(sock, 1024)
                try:
                    msg = json.loads(data.decode())
                    if msg["type"] == "HELLO" and msg["id"] != self.agent_id:
                        await self._register(msg["id"], addr[0], msg["port"])
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                print(f"listen error: {e}")
                await asyncio.sleep(0.1)

    async def _register(self, peer_id, ip, port):
        async with self._lock:
            is_new = peer_id not in self._peers
            self._peers[peer_id] = {"ip": ip, "port": port, "last_seen": asyncio.get_event_loop().time()}
        if is_new:
            print(f"[discovery] new peer: {peer_id}@{ip}:{port}")
            self.on_peer_found(peer_id, ip, port)

    async def _watchdog_loop(self):
        while True:
            await asyncio.sleep(PEER_TIMEOUT)
            now = asyncio.get_event_loop().time()
            async with self._lock:
                deads = [p for p, i in self._peers.items() if now - i["last_seen"] > PEER_TIMEOUT]
                for p in deads:
                    del self._peers[p]
            for p in deads:
                print(f"[discovery] lost peer: {p}")
                self.on_peer_lost(p)


# ── P2PTransport (TCP send/recv) ──────────────────────────────────────────────

class P2PTransport:
    def __init__(self, agent_id, port, on_message):
        self.agent_id = agent_id
        self.port = port
        self.on_message = on_message
        self._peers = {}
        self._lock = asyncio.Lock()
        self._server = None
        self._event_loop = None

    async def register_peer(self, agent_id, ip, port):
        async with self._lock:
            self._peers[agent_id] = (ip, port)

    async def unregister_peer(self, peer_id):
        async with self._lock:
            self._peers.pop(peer_id, None)

    async def send(self, peer_id, msg):
        async with self._lock:
            if peer_id not in self._peers:
                return
            ip, port = self._peers[peer_id]
        msg["from"] = self.agent_id
        data = json.dumps(msg).encode() + b"\n"
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=3.0)
            writer.write(data)
            await writer.drain()
        except Exception as e:
            print(f"failed to send to {peer_id}: {e}")
            return
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    def send_sync(self, peer_id, msg):
        if self._event_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.send(peer_id, msg), self._event_loop)

    async def start(self):
        self._server = await asyncio.start_server(self._handle, "0.0.0.0", self.port)
        async with self._server:
            await self._server.serve_forever()

    async def _handle(self, reader, writer):
        try:
            data = b""
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            try:
                msg = json.loads(data.strip())
                await asyncio.get_event_loop().run_in_executor(None, self.on_message, msg)
            except Exception:
                import traceback
                traceback.print_exc()
        finally:
            writer.close()
            await writer.wait_closed()


# ── launcher ──────────────────────────────────────────────────────────────────

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
BASE_PORT = 5000


def start_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def schedule(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop)


def make_agent(agent_id, port):
    loop = start_loop()

    def on_message(msg):
        print(f"[{agent_id}] <- {msg.get('from', '?')}: {msg.get('text', msg)}")

    transport = P2PTransport(agent_id, port, on_message)
    transport._event_loop = loop
    discovery = Discovery(
        agent_id, port,
        on_peer_found=lambda pid, ip, p: schedule(loop, transport.register_peer(pid, ip, p)),
        on_peer_lost=lambda pid:         schedule(loop, transport.unregister_peer(pid)),
    )
    schedule(loop, transport.start())
    schedule(loop, discovery.start())
    return {"id": agent_id, "transport": transport, "discovery": discovery}


def main():
    agents = []
    for i in range(N):
        agent_id = f"Agent-test-{i + 1}"
        port = BASE_PORT + i * 2
        agents.append(make_agent(agent_id, port))
        print(f"[launcher] started {agent_id} on port {port}")

    print(f"\n[launcher] {N} agents running. Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(5)
            for a in agents:
                peers = a["discovery"]._peers   # source of truth
                if "leader" in peers:
                    info = peers["leader"]
                    # ensure transport has the leader before sending
                    schedule(a["transport"]._event_loop,
                             a["transport"].register_peer("leader", info["ip"], info["port"]))
                    a["transport"].send_sync("leader", {"text": "HI"})
                else:
                    print(f"[{a['id']}] waiting for leader")
    except KeyboardInterrupt:
        print("\n[launcher] stopping.")


if __name__ == "__main__":
    main()