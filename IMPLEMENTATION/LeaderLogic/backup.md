# Conversation Backup & Failover

How Telegram conversation history survives a leader dying, so a newly elected
leader can resume chats without losing context or dropping an unanswered
question.

Code:
- [`LeaderLogic/backup_manager.py`](backup_manager.py) - the `BackupManager`
- [`ElectionLogic/election.py`](../ElectionLogic/election.py) - peer ranking / `backup_peers()`
- [`main.py`](../main.py) - wiring, push points, restore-on-boot

---

## Design in one line

Every device runs a backup **receiver**; the **leader** replicates each
conversation to the **`REPLICATION_FACTOR` strongest other peers**. On
failover the new leader restores from whichever replica survived.

## Who holds a backup

The leader picks its backup targets from `election.backup_peers(n)`
([election.py](../ElectionLogic/election.py)), which returns the `n` strongest
*other* peers, best first, under the same total order used for leader election
(`score`, ties broken by lower ID). It returns fewer than `n` when the network
is smaller.

`REPLICATION_FACTOR` is set in [main.py](../main.py) (currently **2**):

| Factor | Devices that hold a copy        | Survives leader death | Survives leader + 1 backup |
|:------:|---------------------------------|:---------------------:|:--------------------------:|
| 1      | 2nd-strongest only              | ✅                    | ❌ (history lost)          |
| 2      | 2nd + 3rd strongest             | ✅                    | ✅                         |
| N      | 2nd … (N+1)th strongest         | ✅                    | survives N−1 backup losses |

The factor is automatically capped by how many peers actually exist (only 2
devices total → only the one other device is used).

## Sending (`push`)

The leader calls `backup.push(chat_id, messages)` on every Telegram turn, and
it happens **twice per turn** in [main.py](../main.py):

1. **Before inference** - backs up `prior history + new user message`
   immediately, so an unanswered question survives even on leader types that
   keep no history.
2. **After inference** - pushes the updated transcript so the slot no longer
   looks "pending" (prevents a double-answer on a later failover).

`push()` caches locally, then sends a `CONV_BACKUP` message to **each** peer in
`get_backup_peers(replication_factor)`. With no peers yet, data is at least
cached locally. Each receiver stores it and returns a `CONV_BACKUP_ACK`.

## Restoring on a new leader

When a node boots as leader it calls `backup.restore_all(timeout=3.0)`:

1. Broadcast `CONV_RESTORE_REQ` with `chat_id="*"`.
2. `time.sleep(3.0)` - passive wait while peers reply with one
   `CONV_RESTORE_RESP` per stored conversation, merged into the local store.
3. Return a copy of the **entire local store**.

Then `_plan_restore()` ([main.py](../main.py)) splits each restored chat:
- ends in a `user` turn → the old leader died before replying; that text is
  replayed as a *pending* turn (and stripped from the history loaded as context).
- otherwise → the full transcript is loaded as context.

Prior history is loaded via `leader.load_history()` (if the leader type
supports it); pending turns are finished in a background thread by
`_resume_pending_requests()`, after telling the affected user "leader changed,
X is resuming".

## How failover is triggered

From the election layer: when a peer's discovery heartbeat goes silent,
`notify_peer_lost()` drops it and, if it was the leader, re-runs the election
([election.py](../ElectionLogic/election.py)). Peer-loss detection currently
takes ~15 s.

---

## Failure analysis (with `REPLICATION_FACTOR = 2`)

**Single failure - leader dies.** New leader = strongest survivor = the
2nd-strongest = a device that was already a backup target. The history is
**already in its local store**; restore succeeds immediately. The broadcast +
3 s sleep in `restore_all` does no real work in this case - it's a fallback /
safety net for inconsistent peer-table views, not the primary path.

**Double failure - leader + one backup die.** New leader = 3rd-strongest,
which was the *second* backup target, so it also holds a replica → history
**survives**. (Under the old factor-1 design this case lost the conversation
entirely, because only the single 2nd-strongest device ever held a copy.)

**Triple failure - leader + both backups die.** History is lost. Raise
`REPLICATION_FACTOR` to tolerate more simultaneous losses, at the cost of more
per-message network traffic.

## Known characteristics / further work

- **Redundant network round-trip on single failure.** `restore_all` always
  broadcasts and sleeps 3 s even when the data is already local. Could
  short-circuit when `_store` already has the chats.
- **Store is never pruned.** `_store` accumulates conversations from prior
  leadership eras; `restore_all` returns all of them.
- **Replication cost scales with the factor.** Each `push` (twice per turn)
  sends the full transcript to every target peer.

## Protocol messages

| Type                | Direction              | Payload                                   |
|---------------------|------------------------|-------------------------------------------|
| `CONV_BACKUP`       | leader → backup peers  | `chat_id`, full `messages`, `leader_id`, `ts` |
| `CONV_BACKUP_ACK`   | backup peer → leader   | `chat_id`                                 |
| `CONV_RESTORE_REQ`  | new leader → all peers | `chat_id` (or `"*"` wildcard)             |
| `CONV_RESTORE_RESP` | peer → new leader      | `chat_id`, `messages`, `ts`               |
