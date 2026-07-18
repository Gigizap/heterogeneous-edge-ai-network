# ConnectionLogic/transport.py

import json
import asyncio
import logging

class P2PTransport:

    def __init__(self, agent_id: str, port: int, on_message, meter=None):
        self.agent_id = agent_id
        self.port = port
        self.on_message = on_message
        self._peers = {} #{agent_id: (ip, port)}
        self._lock = asyncio.Lock() # prevents race conditions
        self._server = None
        self._event_loop = None  # Set by main.py after loop starts
        self._meter = meter      # optional TrafficMeter (--traffic); None = off
        # Tag every line with this transport's agent_id (agent vs its -leader twin).
        self.log = logging.LoggerAdapter(logging.getLogger(__name__), {"agent": agent_id})

    async def register_peer(self, agent_id, ip, port):
        # meant to be wired directly to Discovery's on_peer_found
        async with self._lock:
            self._peers[agent_id] = (ip, port)

    async def unregister_peer(self, peer_id):
        # meant to be wired directly to Discovery's on_peer_lost
        async with self._lock:
            self._peers.pop(peer_id, None)

    async def send(self, peer_id: str, msg: dict):
        async with self._lock:
            ip, port = self._peers[peer_id]
        msg['from'] = self.agent_id
        data = json.dumps(msg).encode() + b"\n"
        if self._meter:
            self._meter.record(self.agent_id, "out", len(data), msg, peer=peer_id)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=3.0
            )
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            self.log.warning("failed to send to %s: %s", peer_id, e)

    def send_sync(self, peer_id: str, msg: dict):
        # Returns the concurrent.futures.Future for the scheduled send so a caller
        # can block on it (e.g. to time a completed send); fire-and-forget callers
        # can ignore it. None when the event loop isn't running yet.
        if self._event_loop is None:
            return None
        return asyncio.run_coroutine_threadsafe(
            self.send(peer_id, msg),
            self._event_loop
        )
        
    async def broadcast(self, msg: dict):
        async with self._lock:
            peers = dict(self._peers)  # snapshot
        tasks = []
        for peer_id in peers:
            tasks.append(self._send_safe(peer_id, msg))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_safe(self, peer_id: str, msg: dict):
        try:
            await self.send(peer_id, msg)
        except Exception as e:
            self.log.warning("node not reachable: %s (%s)", peer_id, e)

    def broadcast_sync(self, msg: dict):
        """Synchronous broadcast for use from threads.
        Call this instead of broadcast() from non-async code.
        """
        if self._event_loop is None:
            self.log.warning("event loop not set, cannot broadcast")
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast(msg),
                self._event_loop
            )
            # Optionally wait for completion with timeout
            future.result(timeout=5.0)
        except Exception as e:
            self.log.warning("sync broadcast failed: %s", e)
    
    async def start(self):
        self._server = await asyncio.start_server(
            self._handle,
            "0.0.0.0",
            self.port
        )
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass  # stopped via stop() / task cancellation

    async def stop(self):
        """Close the listening server so the port is freed (used on leader teardown)."""
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None

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
                if self._meter:
                    self._meter.record(self.agent_id, "in", len(data), msg, peer=msg.get("from"))
                await asyncio.get_event_loop().run_in_executor(None, self.on_message, msg)
            except Exception:
                self.log.exception("error handling incoming message")
        finally:
            writer.close()
            await writer.wait_closed()