"""
LeaderLogic/backup_manager.py

Conversation history backup
────────────────────────────
The active leader calls `push(chat_id, messages)` after every user message.
BackupManager serialises the full conversation and replicates it to the
top-N strongest peers (N = replication_factor), so a leader failure — and
even a simultaneous leader + backup-peer failure — can be recovered from.

When a new leader is elected it calls `restore(chat_id)` to retrieve the
last known conversation so it can resume without losing context.

Protocol messages
─────────────────
  CONV_BACKUP  {type, from, chat_id, messages, leader_id, ts}
               Leader → backup peer.  Contains the full message list.

  CONV_RESTORE_REQ {type, from, chat_id}
               New leader → old backup peer (or broadcast) asking for history.

  CONV_RESTORE_RESP {type, from, chat_id, messages, ts}
               Backup peer → new leader with the stored conversation.
"""

import json
import time
import logging
import threading
from typing import Callable


class BackupManager:
    """
    Parameters
    ----------
    agent_id      : this device's ID (used as sender tag)
    transport     : P2PTransport — used to send targeted messages
    get_backup_peers   : callable (n) → list[str]  — returns the N strongest
                         other peers (best first) that should hold a backup.
                         Returns fewer than N on a small network.
    replication_factor : how many peers each conversation is replicated to.
                         A double failure (leader + one backup) still leaves a
                         copy when this is >= 2.
    """

    def __init__(
        self,
        agent_id: str,
        transport,
        get_backup_peers: Callable[[int], list[str]],
        replication_factor: int = 2,
    ):
        self._agent_id           = agent_id
        self._transport          = transport
        self._get_backup_peers   = get_backup_peers
        self._replication_factor = replication_factor
        self.log = logging.LoggerAdapter(logging.getLogger(__name__), {"agent": agent_id})

        # Local cache: chat_id → list of message dicts
        self._store: dict[str, list] = {}
        self._lock = threading.Lock()

        # Restore requests waiting for a reply: chat_id → threading.Event
        self._pending_restores: dict[str, threading.Event] = {}
        self._restore_results:  dict[str, list]            = {}

        # Hook into transport
        self._prev_handler = transport.on_message
        transport.on_message = self._handle_message

    # ── leader-side API ───────────────────────────────────────────────────────

    def push(self, chat_id: int | str, messages: list[dict]):
        """
        Call this every time the leader receives or sends a message.
        Stores locally and ships to the top-N backup peers (N =
        replication_factor), so the conversation survives a leader failure and
        even a simultaneous leader + backup-peer failure.
        """
        cid = str(chat_id)
        with self._lock:
            self._store[cid] = messages

        peers = self._get_backup_peers(self._replication_factor)
        if not peers:
            self.log.debug("no backup peers yet — chat %s cached locally only", cid)
            return  # no backup peers yet — data is at least cached locally

        payload = {
            "type":      "CONV_BACKUP",
            "from":      self._agent_id,
            "chat_id":   cid,
            "messages":  messages,
            "leader_id": self._agent_id,
            "ts":        time.time(),
        }
        self.log.debug("backing up chat %s (%d msg) -> %s", cid, len(messages), ", ".join(peers))
        # Schedule all sends first so they run concurrently, then wait for each to
        # finish. Blocking here makes the caller's timing cover the real TCP send
        # (connect + write + drain + close), not just the scheduling of it.
        futures = [(peer, self._transport.send_sync(peer, payload)) for peer in peers]
        for peer, fut in futures:
            if fut is None:
                continue
            try:
                fut.result(timeout=5.0)
            except Exception as e:
                self.log.warning("failed to push to %s: %s", peer, e)

    # ── new-leader-side API ───────────────────────────────────────────────────

    def restore(self, chat_id: int | str, timeout: float = 5.0) -> list[dict]:
        """
        Called by a newly elected leader.  Checks local cache first, then
        asks peers for the stored conversation.

        Returns the message list, or [] if nothing is found.
        """
        cid = str(chat_id)

        # Fast path: we were the backup peer ourselves
        with self._lock:
            if cid in self._store:
                local = list(self._store[cid])
                self.log.info("restore chat %s: found locally (%d msg)", cid, len(local))
                return local

        # Ask the network
        self.log.info("restore chat %s: not local — asking network", cid)
        evt = threading.Event()
        self._pending_restores[cid] = evt
        self._restore_results.pop(cid, None)

        self._transport.broadcast_sync({
            "type":    "CONV_RESTORE_REQ",
            "from":    self._agent_id,
            "chat_id": cid,
        })

        evt.wait(timeout=timeout)
        got = self._restore_results.pop(cid, [])
        self.log.info("restore chat %s: %s", cid,
                      f"{len(got)} msg from network" if got else "no reply")
        return got

    def restore_all(self) -> dict[str, list]:
        """
        Return every conversation this device is holding — no network request.

        The just-elected leader IS the previous leader's backup peer: a live
        leader pushes CONV_BACKUP to the 2nd-strongest device, which is exactly
        the device election promotes when the leader dies. So the history is
        already in our local _store; broadcasting a restore request and sleeping
        for replies was pure boot latency for data we already have. (A
        simultaneous double failure that also takes down the backup peer loses
        history — an acceptable edge for a no-cloud fleet.)
        """
        with self._lock:
            result = dict(self._store)
        self.log.info("restore: %d conversation(s) held locally", len(result))
        return result

    # ── backup-peer-side handler ──────────────────────────────────────────────

    def _handle_message(self, msg: dict):
        mtype  = msg.get("type")
        sender = msg.get("from", "?")

        if mtype == "CONV_BACKUP":
            cid      = str(msg.get("chat_id", ""))
            messages = msg.get("messages", [])
            with self._lock:
                self._store[cid] = messages
            self.log.debug("stored backup for chat %s (%d msg) from %s", cid, len(messages), sender)

            # Confirm receipt (optional, helps debugging)
            try:
                self._transport.send_sync(sender, {
                    "type":    "CONV_BACKUP_ACK",
                    "from":    self._agent_id,
                    "chat_id": cid,
                })
            except Exception:
                pass

        elif mtype == "CONV_RESTORE_REQ":
            req_cid = msg.get("chat_id", "")
            with self._lock:
                if req_cid == "*":
                    # Return everything we have
                    for cid, messages in self._store.items():
                        self._transport.send_sync(sender, {
                            "type":     "CONV_RESTORE_RESP",
                            "from":     self._agent_id,
                            "chat_id":  cid,
                            "messages": messages,
                            "ts":       time.time(),
                        })
                    self.log.info("restore-all request from %s — sending %d conversation(s)",
                                  sender, len(self._store))
                elif req_cid in self._store:
                    self._transport.send_sync(sender, {
                        "type":     "CONV_RESTORE_RESP",
                        "from":     self._agent_id,
                        "chat_id":  req_cid,
                        "messages": self._store[req_cid],
                        "ts":       time.time(),
                    })
                    self.log.info("restore request from %s for chat %s — sending it (%d msg)",
                                  sender, req_cid, len(self._store[req_cid]))
                else:
                    self.log.info("restore request from %s for chat %s — nothing stored",
                                  sender, req_cid)

        elif mtype == "CONV_RESTORE_RESP":
            cid      = str(msg.get("chat_id", ""))
            messages = msg.get("messages", [])
            with self._lock:
                self._store[cid] = messages
            self.log.info("received restored history for chat %s from %s (%d msg)",
                          cid, sender, len(messages))
            if cid in self._pending_restores:
                self._restore_results[cid] = messages
                self._pending_restores[cid].set()

        if self._prev_handler:
            self._prev_handler(msg)