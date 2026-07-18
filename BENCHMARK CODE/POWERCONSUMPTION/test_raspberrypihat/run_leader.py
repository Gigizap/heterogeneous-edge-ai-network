"""
run_leader.py — fully self-contained. One file, no project imports.

A single agent with id "leader" that:
  * runs P2PTransport + Discovery (registers peers as they appear)
  * prints every message it receives
  * every 5s sends {"text": "HI FROM LEADER"} to all known peers

    python run_leader.py

Run alongside run_agents.py: the Agent-test-N agents will find "leader",
send it HI, and receive HI FROM LEADER back.

Ctrl-C stops it.
"""

import asyncio
import json
import socket
import threading
import time

# ── Discovery (UDP presence broadcast) ───────────────────────────────────────

DISCOVERY_PORT = 9999
BROADCAST_ADDR = "255.255.255.255"
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
        # SO_REUSEPORT: lets several processes on ONE machine all receive the
        # UDP broadcasts. With only SO_REUSEADDR the kernel delivers each
        # datagram to just one socket, so peers get missed intermittently.
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
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
        # Closing races with the receiver tearing down after it reads our \n.
        # The data is already delivered at this point, so swallow teardown errors.
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    async def broadcast(self, msg):
        async with self._lock:
            peers = dict(self._peers)
        await asyncio.gather(*(self._send_safe(p, msg) for p in peers), return_exceptions=True)

    async def _send_safe(self, peer_id, msg):
        try:
            await self.send(peer_id, msg)
        except Exception as e:
            print(f"node not reachable: {peer_id}: {e}")

    def broadcast_sync(self, msg):
        if self._event_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(msg), self._event_loop)

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
    
    def send_sync(self, peer_id: str, msg: dict):
        if self._event_loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.send(peer_id, msg),
            self._event_loop
        )

    


# ── leader ────────────────────────────────────────────────────────────────────

AGENT_ID = "leader"
PORT = 6000


def start_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def schedule(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop)


def main():
    loop = start_loop()

    def on_message(msg):
        print(f"[leader] <- {msg.get('from', '?')}: {msg.get('text', msg)}")

    transport = P2PTransport(AGENT_ID, PORT, on_message)
    transport._event_loop = loop
    discovery = Discovery(
        AGENT_ID, PORT,
        on_peer_found=lambda pid, ip, p: schedule(loop, transport.register_peer(pid, ip, p)),
        on_peer_lost=lambda pid:         schedule(loop, transport.unregister_peer(pid)),
    )
    schedule(loop, transport.start())
    schedule(loop, discovery.start())

    print(f"[leader] running as '{AGENT_ID}' on port {PORT}. Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(5)
            # discovery._peers is the source of truth (filled by UDP directly).
            # We broadcast straight from it so there's no dependence on the
            # async register_peer having landed in transport._peers yet.
            peers = dict(discovery._peers)
            if not peers:
                print("[leader] no peers yet")
                continue
            for pid, info in peers.items():
                # make sure transport knows this peer, then send
                schedule(loop, transport.register_peer(pid, info["ip"], info["port"]))
                transport.send_sync(pid, {"text": "HI FROM LEADER"})
    except KeyboardInterrupt:
        print("\n[leader] stopping.")


if __name__ == "__main__":
    main()