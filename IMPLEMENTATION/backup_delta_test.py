"""
backup_delta_test.py

Standalone micro-benchmark for ONE thing: how long BackupManager.push() takes to
replicate a conversation to the strongest peer, now that push() blocks until the
CONV_BACKUP TCP send actually completes (transport.send_sync returns the future,
push waits on it).

It stands up a real transport + discovery + election node that claims a huge
score (200000) so it wins leadership and does the pushing, waits until discovery
and election hand it a backup peer, then for each of the 10 person queries:

    t0 = perf_counter()
    backup.push(chat_id, [user query])     # measured -> delta_backup
    sleep 0.5 s
    backup.push(chat_id, [])               # eliminate the backup, exactly as
                                           # main.py does after a reply is sent

The 10 delta_backup values are collected; mean and sample std are reported and
written to backup_delta_results.json next to this file.

Run it on the machine you want to measure FROM, with at least one other node
(e.g. the STM32 running main.py) alive on the same LAN so there is a peer to
back up to.

    python backup_delta_test.py
    python backup_delta_test.py --port 5730

Stdlib only, plus the repo's own ConnectionLogic / ElectionLogic / LeaderLogic
modules. Changes no live preset and speaks no test-only wire messages: it uses
the exact production CONV_BACKUP path.
"""

import sys
import pathlib

# Put IMPLEMENTATION (this file's dir) on the path so "from ConnectionLogic..."
# resolves no matter the working directory.
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import argparse
import json
import logging
import statistics
import time

from ConnectionLogic.transport import P2PTransport
from ConnectionLogic.discovery import Discovery
from ElectionLogic.election    import ElectionManager
from LeaderLogic.backup_manager import BackupManager
from utils import start_async_loop, schedule


# ── config ────────────────────────────────────────────────────────────────────

AGENT_ID           = "backup-timer"
DEFAULT_PORT       = 5730
SCORE              = 200000     # huge -> we win the election and do the pushing
REPLICATION_FACTOR = 2          # same as main.py; on a 2-device net this still
                                # resolves to the single other peer
CHAT_ID            = 424242     # fixed test conversation id
SETTLE_S           = 0.5        # gap between backing up and clearing (per spec)
INTER_QUERY_S      = 0.2        # small gap so each measurement is clean
PEER_WAIT_TIMEOUT  = 60         # give up if no peer appears within this many seconds

# The 10 person queries under test.
PERSON_QUERIES = [
    "look for a person you can see",
    "detect all the people you see now",
    "is there a person in front of the camera",
    "how many people are there",
    "do you see anyone",
    "check if a person is in the room",
    "are there any people visible",
    "detect the people in the frame",
    "is any person there right now",
    "tell me how many people you can see",
]

log = logging.getLogger("backup_delta_test")

# Assigned in _build_node(); referenced by the discovery callbacks and by the
# backup manager's get_backup_peers lambda (both only run after it is set).
election = None


# ── node wiring (mirrors main.py, minus the leader brain) ─────────────────────

def _build_node(port):
    global election

    loop = start_async_loop()

    def _on_peer_found(pid, ip, peer_port):
        # Register the peer for routing, then re-enter the election plane so its
        # score reaches us (that is what backup_peers ranks on).
        log.info("RECV discovery: peer FOUND %s @ %s:%s", pid, ip, peer_port)
        schedule(loop, transport.register_peer(pid, ip, peer_port))
        if election is not None:
            schedule(loop, election.announce())

    def _on_peer_lost(pid):
        log.info("RECV discovery: peer LOST %s", pid)
        schedule(loop, transport.unregister_peer(pid))
        if election is not None:
            election.notify_peer_lost(pid)

    def _log_incoming(msg):
        # Bottom of the transport.on_message chain (election -> backup -> here), so
        # this fires for EVERY message this node receives over TCP.
        log.info("RECV tcp: %s from %s | %s",
                 msg.get("type") or msg.get("method") or "?",
                 msg.get("from", "?"), msg)

    transport = P2PTransport(AGENT_ID, port, _log_incoming)
    transport._event_loop = loop

    discovery = Discovery(AGENT_ID, port,
                          on_peer_found=_on_peer_found,
                          on_peer_lost=_on_peer_lost)

    # BackupManager and ElectionManager each chain themselves into
    # transport.on_message. Build backup first so election wraps it: incoming
    # messages then flow election -> backup -> base (same chaining main.py uses).
    backup = BackupManager(
        agent_id=AGENT_ID,
        transport=transport,
        get_backup_peers=lambda n: election.backup_peers(n) if election is not None else [],
        replication_factor=REPLICATION_FACTOR,
    )

    profile = {"agent_id": AGENT_ID, "leader_preset": None,
               "sensing_preset": None, "score": SCORE}

    election = ElectionManager(
        agent_id=AGENT_ID,
        score=SCORE,
        profile=profile,
        transport=transport,
        on_became_leader=lambda p: log.info("became leader (score=%d)", SCORE),
        on_became_follower=lambda lid: log.info("following %s", lid),
        on_demoted=lambda: log.info("demoted"),
    )

    schedule(loop, transport.start())
    schedule(loop, discovery.start())
    schedule(loop, election.run_startup_election())

    return transport, discovery, backup


def _safe_keys(d):
    """Snapshot a dict's keys from this thread while the loop mutates it."""
    try:
        return list(d)
    except RuntimeError:
        return []


def _election_scores():
    try:
        return {p: info.get("score") for p, info in list(election._peers.items())}
    except RuntimeError:
        return {}


def _wait_for_backup_peer(transport, discovery, timeout):
    """Block until election ranks a peer AND transport can route to it - exactly
    what a real leader's backup.push() needs (get_backup_peers ranks from the
    election plane; transport routes from the discovery-fed peer table).

    Logs the discovery / election / transport peer sets every few seconds so a
    stall is diagnosable instead of silent.
    """
    deadline     = time.time() + timeout
    last_report  = 0.0
    while time.time() < deadline:
        peers    = election.backup_peers(REPLICATION_FACTOR)
        routable = [p for p in peers if p in transport._peers]
        if peers and len(routable) == len(peers):
            return peers

        now = time.time()
        if now - last_report >= 3.0:
            last_report = now
            log.info("still waiting: discovery=%s  election(score)=%s  transport=%s",
                     _safe_keys(getattr(discovery, "_peers", {})),
                     _election_scores(),
                     _safe_keys(getattr(transport, "_peers", {})))
        time.sleep(0.2)
    return []


# ── the measurement ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    # Hush the module DEBUG/INFO dumps so only this test's own lines show; keep a
    # window open on WARNING so a failed send is still visible.
    for noisy in ("ConnectionLogic", "ElectionLogic", "LeaderLogic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log.info("starting backup-timing node %s on port %d (score=%d)",
             AGENT_ID, args.port, SCORE)
    transport, discovery, backup = _build_node(args.port)

    log.info("waiting for a backup peer (max %ds) ...", PEER_WAIT_TIMEOUT)
    peers = _wait_for_backup_peer(transport, discovery, PEER_WAIT_TIMEOUT)
    if not peers:
        disc = _safe_keys(getattr(discovery, "_peers", {}))
        elec = _election_scores()
        tran = _safe_keys(getattr(transport, "_peers", {}))
        if not disc:
            log.error("no backup peer in %ds: OUR discovery never found any peer "
                      "(discovery=%s). If you are running this on the same machine "
                      "as another agent, they share UDP discovery port 9999 and this "
                      "listener can be starved - run it on its own device, like a real "
                      "leader.", PEER_WAIT_TIMEOUT, disc)
        elif not elec:
            log.error("no backup peer in %ds: discovery sees %s but no peer ever "
                      "put a score on our election plane (no HELLO/ELECTION/"
                      "LEADER_CLAIM received). transport=%s", PEER_WAIT_TIMEOUT, disc, tran)
        else:
            log.error("no backup peer in %ds: election ranks %s but transport can't "
                      "route to it yet (transport=%s).", PEER_WAIT_TIMEOUT, elec, tran)
        sys.exit(1)
    log.info("backup peer(s): %s - running %d queries", peers, len(PERSON_QUERIES))

    deltas = []
    for i, query in enumerate(PERSON_QUERIES, 1):
        t0 = time.perf_counter()
        backup.push(CHAT_ID, [{"role": "user", "content": query}])
        dt_ms = (time.perf_counter() - t0) * 1000
        deltas.append(dt_ms)
        log.info("[%2d/%d] backed up in %8.3f ms  |  %s",
                 i, len(PERSON_QUERIES), dt_ms, query)

        time.sleep(SETTLE_S)
        backup.push(CHAT_ID, [])        # eliminate the backup, as main.py does after a reply
        time.sleep(INTER_QUERY_S)

    mean = statistics.mean(deltas)
    std  = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    log.info("=" * 60)
    log.info("mean = %.3f ms   std = %.3f ms   (n=%d)", mean, std, len(deltas))

    results = {
        "agent_id": AGENT_ID,
        "port": args.port,
        "score": SCORE,
        "replication_factor": REPLICATION_FACTOR,
        "backup_peers": peers,
        "queries": [{"query": q, "delta_backup_ms": d}
                    for q, d in zip(PERSON_QUERIES, deltas)],
        "mean_ms": mean,
        "std_ms": std,
    }
    out = _HERE / "backup_delta_results.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
