"""
ConnectionLogic/traffic.py

Optional per-agent traffic recorder, toggled by `python main.py --traffic`.

When enabled it is handed to P2PTransport and Discovery, which call record()
for EVERY message they send or receive. It does two things:

  * appends a JSON-Lines record of every message to a file (a full capture the
    reviewer can inspect or load into pandas), and
  * keeps in-memory aggregates {agent, direction, kind} -> (count, bytes) for a
    summary table (the `traffic` console command, and an auto-dump at exit).

`kind` is the message's JSON-RPC `method`, its discovery/election/backup `type`,
or a coarse label (result / error / text) for anything else.

`nbytes` is the APPLICATION payload size (the newline-terminated JSON on the
wire), not the raw NIC bytes: it excludes TCP/IP/Ethernet headers, ACKs, and the
per-message TCP handshake this transport pays (connection-per-message). For true
bandwidth, pair this with a one-off `tcpdump`/Wireshark capture.

Disabled by default (the meter is None), so there is zero overhead when not asked
for: the transport/discovery simply skip the record() call.
"""

import json
import threading
import time
from collections import defaultdict


def classify(msg) -> str:
    """Coarse message category for the summary table."""
    if not isinstance(msg, dict):
        return "raw"
    if "method" in msg:
        return msg["method"]          # tools/list, tools/call, notifications/*
    if "type" in msg:
        return msg["type"]            # HELLO, ELECTION, LEADER_CLAIM, CONV_*, CAPABILITY_*
    if "result" in msg:
        return "result"               # JSON-RPC response
    if "error" in msg:
        return "error"                # JSON-RPC error
    if "text" in msg:
        return "text"                 # console broadcast
    return "other"


class TrafficMeter:
    def __init__(self, path: str | None = None):
        self._lock  = threading.Lock()
        self._path  = path
        self._file  = open(path, "a", encoding="utf-8") if path else None
        self._agg   = defaultdict(lambda: [0, 0])   # (agent, dir, kind) -> [count, bytes]
        self._start = time.time()

    def record(self, agent: str, direction: str, nbytes: int, msg, peer=None):
        """direction is 'in' or 'out'. Called from transport/discovery threads."""
        kind = classify(msg)
        with self._lock:
            slot = self._agg[(agent, direction, kind)]
            slot[0] += 1
            slot[1] += nbytes
            if self._file:
                self._file.write(json.dumps({
                    "ts":    round(time.time(), 3),
                    "dir":   direction,
                    "agent": agent,
                    "peer":  peer,
                    "kind":  kind,
                    "bytes": nbytes,
                    "msg":   msg,
                }, ensure_ascii=False) + "\n")
                self._file.flush()

    def report(self) -> str:
        with self._lock:
            rows    = sorted(self._agg.items())
            elapsed = time.time() - self._start
        out = ["=== traffic summary (%.0fs) ===" % elapsed,
               f"{'agent':<20} {'dir':<3} {'kind':<24} {'count':>7} {'bytes':>10}"]
        tot_in = tot_out = 0
        for (agent, direction, kind), (count, nbytes) in rows:
            out.append(f"{agent:<20} {direction:<3} {kind:<24} {count:>7} {nbytes:>10}")
            if direction == "in":
                tot_in += nbytes
            else:
                tot_out += nbytes
        out.append("-" * 68)
        rate = (tot_in + tot_out) / max(elapsed, 1)
        out.append(f"TOTAL   in={tot_in} B   out={tot_out} B   "
                   f"({rate:.0f} B/s app-level payload, excl. TCP/IP overhead)")
        if self._path:
            out.append(f"full per-message capture: {self._path}")
        return "\n".join(out)

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
