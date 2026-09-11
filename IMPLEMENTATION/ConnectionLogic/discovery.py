# ConnectionLogic/discovery.py

import json
import os
import asyncio
import socket
import time
import logging

from utils import own_ip

DISCOVERY_PORT = 9999


def _derive_broadcast_addr() -> str:
    """
    Best-effort subnet broadcast address for THIS host's primary interface,
    so discovery works on any network (and so two `--port` instances on one
    machine find each other) without hardcoding a subnet.

    Override with the AGENT_BROADCAST_ADDR env var if auto-detection is wrong.
    Falls back to the limited-broadcast address 255.255.255.255.
    """
    override = os.environ.get("AGENT_BROADCAST_ADDR")
    if override:
        return override
    ip = own_ip()
    if ip and ip.count(".") == 3 and not ip.startswith("127."):
        return ip.rsplit(".", 1)[0] + ".255"   # assume /24
    return "255.255.255.255"


BROADCAST_ADDR = _derive_broadcast_addr()
ANNOUNCE_INTERVAL = 2   # seconds - how often we re-announce presence
PEER_TIMEOUT      = 15  # seconds of silence before a peer is considered gone
WATCHDOG_INTERVAL = 2   # seconds - how often we check for dead peers

class Discovery:
    def __init__(self, agent_id: str, tcp_port: int, on_peer_found, on_peer_lost, meter=None):
        self.agent_id = agent_id
        self.tcp_port = tcp_port
        self.on_peer_found = on_peer_found
        self.on_peer_lost = on_peer_lost
        self._meter = meter      # optional TrafficMeter (--traffic); None = off
        self._peers = {} #{agent_id: {"ip": ..., "port": ..., "last_seen": ...}}
        self._lock = asyncio.Lock() # to protect access to _peers, since it's shared across async tasks
        # Tag every line with this discovery's agent_id (agent vs its -leader twin).
        self.log = logging.LoggerAdapter(logging.getLogger(__name__), {"agent": agent_id})

    async def start(self):
        # Launch all three loops concurrently
        try:
            await asyncio.gather(
                self._announce_loop(),
                self._listen_loop(),
                self._watchdog_loop(),
                return_exceptions=True
            )
        except asyncio.CancelledError:
            pass

    async def get_peers(self) -> dict:
        async with self._lock:
            return dict(self._peers) #Returns a copy
        
    async def _announce_loop(self):
        """Periodically broadcast our presence via UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)
        
        msg = json.dumps(
            {
                "type": "HELLO", 
                "id": self.agent_id,
                "port": self.tcp_port,
            }
        ).encode()

        while True:
            try:
                # Use asyncio to handle non-blocking socket
                loop = asyncio.get_event_loop()
                await loop.sock_sendto(sock, msg, (BROADCAST_ADDR, DISCOVERY_PORT))
                if self._meter:
                    self._meter.record(self.agent_id, "out", len(msg),
                                       {"type": "HELLO", "id": self.agent_id, "port": self.tcp_port},
                                       peer="broadcast")
            except Exception as e:
                self.log.warning("error while announcing: %s", e)

            await asyncio.sleep(ANNOUNCE_INTERVAL)

    async def _listen_loop(self):
        """Listen for peer announcements via UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", DISCOVERY_PORT))
        sock.setblocking(False)
        
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                data, addr = await loop.sock_recvfrom(sock, 1024)
                ip = addr[0]
                try:
                    msg = json.loads(data.decode())
                    if self._meter:
                        self._meter.record(self.agent_id, "in", len(data), msg, peer=msg.get("id"))
                    if msg["type"] == "HELLO" and msg["id"] != self.agent_id:
                        await self._register(msg["id"], ip, msg["port"])
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                self.log.warning("error while listening: %s", e)
                await asyncio.sleep(0.1)
    
    async def _register(self, peer_id, ip, port):
        """Register a discovered peer."""
        async with self._lock:
            is_new = peer_id not in self._peers
            loop = asyncio.get_event_loop()
            self._peers[peer_id] = {
                "ip": ip,
                "port": port,
                "last_seen": loop.time()
            }
        if is_new:
            self.log.info("new peer: %s@%s:%s", peer_id, ip, port)
            self.on_peer_found(peer_id, ip, port)

    async def _watchdog_loop(self):
        """Periodically check for dead peers and remove them."""
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL)
            loop = asyncio.get_event_loop()
            current_time = loop.time()
            deads = []
            
            async with self._lock:
                deads = [
                    pid for pid, info in self._peers.items()
                    if current_time - info["last_seen"] > PEER_TIMEOUT
                ]
                for pid in deads:
                    del self._peers[pid]
            
            for pid in deads:
                self.log.info("peer lost: %s", pid)
                self.on_peer_lost(pid)


#function outside the class, used by identity.py during setup before any transport exists
def collect_peer_ids(window: float = 6.0) -> set[str]:
    """
    Blocking one-shot scan of the UDP discovery port.
 
    Opens a raw UDP socket on DISCOVERY_PORT (9999), listens for
    HELLO broadcasts from running agents, and returns the set of
    agent IDs seen.
 
    Used by identity._prompt_agent_id() during first-run setup to
    detect name conflicts before any transport or event loop exists.
 
    The window defaults to 6 seconds - one full ANNOUNCE_INTERVAL
    cycle (5s) plus a buffer - to guarantee we catch at least one
    HELLO from every running agent.
 
    Returns:
        set[str] - agent IDs currently on the network, e.g.
                   {"laptop-huawei", "camera-corridor"}
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", DISCOVERY_PORT))
    sock.settimeout(0.5)
 
    seen = set()
    deadline = time.time() + window
    while time.time() < deadline:
        try:
            data, _ = sock.recvfrom(1024)
            msg = json.loads(data.decode())
            if msg.get("type") == "HELLO":
                seen.add(msg["id"])
        except (socket.timeout, json.JSONDecodeError):
            pass
    sock.close()
    return seen
 