"""
ElectionLogic/election.py

Leader election protocol (deterministic, single-leader)
───────────────────────────────────────────────────────
Every leader-capable device runs an ElectionManager attached to its agent
transport.  The rightful leader is, at all times, the best **alive** candidate
under one total order shared by everyone:

    A is a better leader than B  ⇔  A.score > B.score
                                    or (A.score == B.score and A.id < B.id)

Because the order is total and every node computes it from the same set of
HELLO/ELECTION/LEADER_CLAIM facts, all nodes converge on the *same* winner -
ties no longer cause a split-brain, and there is no separate "lex fallback".

Messages (all carry "type" and "from"):

  HELLO         {type, from, score, has_leader_preset, sensing_preset, leader_preset}
  ELECTION      {type, from, score}
  LEADER_CLAIM  {type, from, score}   - "I am the leader" (carries the claimant's score)

How it stays single-leader
──────────────────────────
  • Startup: announce repeatedly while discovery fills the peer table, then run
    one election window and `_resolve()` - claim if we are the best, else follow.
  • Any LEADER_CLAIM is reconciled by `_resolve()`: a *better* claimant is
    accepted (we demote if we were leader); an *inferior* claim received while
    we are leader makes us re-assert (re-broadcast our claim).  Two crossed
    claims therefore always collapse onto the single best node.
  • A leader re-asserts whenever it hears a peer's HELLO/ELECTION, so a freshly
    joined device learns there is already a leader (and only takes over if it is
    genuinely stronger).
  • When discovery links a NEW peer after startup, `announce()` re-enters the
    plane (a leader sends LEADER_CLAIM, a follower HELLO): this registers late
    joiners for backup ranking and collapses a split-brain, since two
    independently elected leaders then reconcile through `_resolve()` instead of
    coexisting.
  • When a peer goes silent, `notify_peer_lost()` drops it from the table and,
    if it was the leader, re-resolves - so the next-best device takes over.
"""

import asyncio
import logging
from typing import Callable, Optional


DISCOVERY_WINDOW = 4.0   # total time spent announcing while discovery populates peers
ELECTION_WINDOW  = 2.0   # time spent collecting bids before resolving
HELLO_BURSTS     = 4     # number of HELLOs spread across the discovery window


class ElectionManager:
    """
    Parameters
    ----------
    agent_id           : this agent's ID
    score              : this agent's numeric rank score
    profile            : full device_profile dict
    transport          : P2PTransport (already started)
    on_became_leader   : callable(profile)    - called when this agent wins
    on_became_follower : callable(leader_id)  - called when another agent wins
    on_demoted         : callable()           - optional; called when this agent
                          stops being leader (hand off / out-ranked) so main.py
                          can tear the leader process down.
    """

    def __init__(
        self,
        agent_id: str,
        score: int,
        profile: dict,
        transport,
        on_became_leader: Callable[[dict], None],
        on_became_follower: Callable[[str], None],
        on_demoted: Optional[Callable[[], None]] = None,
    ):
        self.agent_id  = agent_id
        self.score     = score
        self.profile   = profile
        self.transport = transport

        self._on_became_leader   = on_became_leader
        self._on_became_follower = on_became_follower
        self._on_demoted         = on_demoted

        # peer_id → their last HELLO/ELECTION/LEADER_CLAIM payload (incl. "score")
        self._peers: dict[str, dict] = {}
        self._leader_id: str | None  = None
        self._i_am_leader            = False
        self._lock                   = asyncio.Lock()
        self.log = logging.LoggerAdapter(logging.getLogger(__name__), {"agent": agent_id})

        # Inject into the transport handler chain
        self._prev_handler   = transport.on_message
        transport.on_message = self._handle_message

    # ── ranking ───────────────────────────────────────────────────────────────

    @staticmethod
    def _better(id_a: str, score_a: int, id_b: str, score_b: int) -> bool:
        """True iff candidate A is a better leader than candidate B."""
        if score_a != score_b:
            return score_a > score_b
        return id_a < id_b

    def _compute_leader_locked(self) -> str:
        """Best candidate among {self} ∪ known peers.  Call under self._lock."""
        best_id, best_score = self.agent_id, self.score
        for pid, info in self._peers.items():
            s = info.get("score")
            if s is None:
                continue
            if self._better(pid, s, best_id, best_score):
                best_id, best_score = pid, s
        return best_id

    # ── public API ────────────────────────────────────────────────────────────

    async def run_startup_election(self):
        """Announce presence while discovery fills the table, then resolve role."""
        self.log.info("discovery window %ss …", DISCOVERY_WINDOW)
        for _ in range(HELLO_BURSTS):
            await self._broadcast_hello()
            await asyncio.sleep(DISCOVERY_WINDOW / HELLO_BURSTS)

        async with self._lock:
            already_following = self._leader_id is not None and not self._i_am_leader
        if already_following:
            self.log.info("existing leader: %s", self._leader_id)
            return

        await self._run_election()

    async def announce(self):
        """
        Called when discovery links a NEW peer (see main.py `on_peer_found`), so
        the election plane learns about nodes that joined *after* our startup
        window. Discovery keeps such a peer alive in the transport table, but only
        a HELLO/ELECTION/LEADER_CLAIM ever puts its score into `_peers`.

        A leader announces with a LEADER_CLAIM: this hands the newcomer our score
        (used by `backup_peers()` for replication ranking) and, if the newcomer is
        itself a stale leader, drives it through `_resolve()` so the weaker of the
        two steps down, collapsing a split-brain in one direction without waiting
        for the reverse-side discovery to fire. A non-leader just shares its score
        via HELLO, which makes any existing leader re-assert and reconcile.
        """
        async with self._lock:
            leader = self._i_am_leader
        kind    = "LEADER_CLAIM" if leader else "HELLO"
        targets = list(getattr(self.transport, "_peers", {}))
        self.log.debug("announce: sending %s to %d peer(s): %s", kind, len(targets), targets)
        if leader:
            await self._broadcast_claim()
        else:
            await self._broadcast_hello()

    def notify_peer_lost(self, peer_id: str):
        """
        Called by main.py when a peer's discovery heartbeat goes silent (it was
        killed / powered off / left the network).  Drops it from the table and,
        if it was the current leader, triggers a fresh election.
        """
        loop = self.transport._event_loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._handle_peer_lost(peer_id), loop)

    async def _handle_peer_lost(self, peer_id: str):
        async with self._lock:
            self._peers.pop(peer_id, None)
            was_leader = (peer_id == self._leader_id)
            if was_leader:
                self._leader_id = None
        if was_leader:
            self.log.info("leader %s lost - re-electing", peer_id)
            await self._run_election()

    def backup_peers(self, n: int = 1) -> list[str]:
        """
        The N strongest *other* peers, best first - the devices that should
        hold conversation backups so a failover (or even a simultaneous
        leader + backup failure) can still recover history. Returns fewer than
        N on a smaller network. Best-effort snapshot - safe to call from the
        leader thread.
        """
        try:
            items = list(self._peers.items())
        except RuntimeError:
            items = []
        ranked = sorted(
            (
                (pid, info["score"])
                for pid, info in items
                if info.get("score") is not None
            ),
            key=lambda kv: (-kv[1], kv[0]),   # higher score first, ties by lower id
        )
        result = [pid for pid, _ in ranked[:max(0, n)]]
        self.log.debug("backup_peers(n=%d) -> %s (known: %s)",
                       n, result, [pid for pid, _ in items])
        return result

    def backup_peer(self) -> str | None:
        """
        The single strongest *other* peer (the second-best overall when we
        lead). Kept for callers that only need one; see backup_peers().
        """
        peers = self.backup_peers(1)
        return peers[0] if peers else None

    @property
    def current_leader(self) -> str | None:
        return self._leader_id

    @property
    def i_am_leader(self) -> bool:
        return self._i_am_leader

    # ── core election logic ───────────────────────────────────────────────────

    async def _run_election(self):
        """Broadcast candidacy, collect bids for one window, then resolve."""
        async with self._lock:
            if self._i_am_leader:
                return  # already leading - nothing to contest
        await self._broadcast({"type": "ELECTION", "from": self.agent_id, "score": self.score})
        await self._broadcast_hello()  # make sure peers have our score
        self.log.info("election window %ss …", ELECTION_WINDOW)
        await asyncio.sleep(ELECTION_WINDOW)
        await self._resolve()

    async def _resolve(self, reassert: bool = False):
        """
        Recompute the rightful leader from current knowledge and act:
          - if we are the best and not yet leader → claim;
          - if we are leader but someone better exists → demote and follow;
          - otherwise follow the best (it will claim on its own).
        `reassert` re-broadcasts our claim when we are (still) the best leader -
        used to inform a newcomer or rebuff an inferior claim.
        """
        async with self._lock:
            best        = self._compute_leader_locked()
            i_am_best   = (best == self.agent_id)
            was_leader  = self._i_am_leader
            prev_leader = self._leader_id

        if i_am_best:
            if not was_leader:
                await self._claim_leadership()
            elif reassert:
                await self._broadcast_claim()
        else:
            if was_leader:
                await self._demote(best)
            elif best != prev_leader:
                async with self._lock:
                    self._leader_id = best
                self.log.info("leader is %s", best)
                self._on_became_follower(best)

    async def _claim_leadership(self):
        async with self._lock:
            if self._i_am_leader:
                return
            self._i_am_leader = True
            self._leader_id   = self.agent_id
        self.log.info("claiming leadership (score=%s)", self.score)
        await self._broadcast_claim()
        self._on_became_leader(self.profile)

    async def _demote(self, new_leader: str):
        async with self._lock:
            self._i_am_leader = False
            self._leader_id   = new_leader
        self.log.info("stepping down -> %s", new_leader)
        if self._on_demoted:
            try:
                self._on_demoted()
            except Exception as e:
                self.log.error("on_demoted error: %s", e)
        self._on_became_follower(new_leader)

    async def _broadcast_claim(self):
        await self._broadcast({"type": "LEADER_CLAIM", "from": self.agent_id, "score": self.score})

    # ── message handler (injected into transport chain) ───────────────────────
    #
    # NOTE: transport dispatches on a thread-pool thread, so we hop back onto the
    # event loop with run_coroutine_threadsafe.

    def _handle_message(self, msg: dict):
        mtype  = msg.get("type")
        sender = msg.get("from", "?")
        loop   = self.transport._event_loop

        if loop is None:
            if self._prev_handler:
                self._prev_handler(msg)
            return

        if mtype == "HELLO":
            asyncio.run_coroutine_threadsafe(self._handle_hello(sender, msg), loop)
        elif mtype == "ELECTION":
            asyncio.run_coroutine_threadsafe(self._handle_election(sender, msg), loop)
        elif mtype == "LEADER_CLAIM":
            asyncio.run_coroutine_threadsafe(self._handle_leader_claim(sender, msg), loop)

        # Always pass through to the previous handler (backup manager, sensing agent, etc.)
        if self._prev_handler:
            self._prev_handler(msg)

    async def _handle_hello(self, sender: str, msg: dict):
        if sender == self.agent_id:
            return
        async with self._lock:
            self._peers[sender] = msg
            leader = self._i_am_leader
            known  = list(self._peers)
        self.log.debug("recorded peer %s (score=%s) via HELLO; election peers now: %s",
                       sender, msg.get("score"), known)
        # As leader, re-announce our claim so a newcomer learns there is already
        # a leader. If the newcomer is genuinely stronger it will still claim,
        # and we step down when that LEADER_CLAIM arrives (handled below) - so we
        # never tear the leader down on a mere HELLO.
        if leader:
            await self._broadcast_claim()

    async def _handle_election(self, sender: str, msg: dict):
        if sender == self.agent_id:
            return
        async with self._lock:
            self._peers[sender] = msg
            leader = self._i_am_leader
            known  = list(self._peers)
        self.log.debug("recorded peer %s (score=%s) via ELECTION; election peers now: %s",
                       sender, msg.get("score"), known)
        if leader:
            await self._broadcast_claim()

    async def _handle_leader_claim(self, sender: str, msg: dict):
        if sender == self.agent_id:
            return
        async with self._lock:
            self._peers[sender] = msg            # remember claimant's score
            known = list(self._peers)
        self.log.debug("recorded peer %s (score=%s) via LEADER_CLAIM; election peers now: %s",
                       sender, msg.get("score"), known)
        # Reconcile: a better claimant wins (we demote if we were leader);
        # an inferior claim makes us re-assert our own.
        await self._resolve(reassert=True)

    # ── broadcast helpers ─────────────────────────────────────────────────────

    async def _broadcast_hello(self):
        await self._broadcast({
            "type":              "HELLO",
            "from":              self.agent_id,
            "score":             self.score,
            "has_leader_preset": self.profile.get("leader_preset") not in (None, "none"),
            "sensing_preset":    self.profile.get("sensing_preset"),
            "leader_preset":     self.profile.get("leader_preset"),
        })

    async def _broadcast(self, payload: dict):
        payload.setdefault("from", self.agent_id)
        await self.transport.broadcast(payload)
