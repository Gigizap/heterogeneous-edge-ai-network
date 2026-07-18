# LatencyTest2 - network-overlay power/scaling test (mock STM32 leader + fake fleet)

> <- Back to [IMPLEMENTATION README](../README.md)

Measures how the **network overlay** on the STM32MP257F-DK behaves as the number
of sensing agents grows: `1 -> 10 -> 20 -> 50 -> 100`. The STM32 runs a **mock
leader** (live discovery + tool aggregation, **no LLM / no election / no
dispatch**); a laptop runs a growing fleet of **fake sensing agents** that only
announce themselves and expose **one shared placeholder tool**.

The design goal is to **isolate the overlay** so an external power meter on the
board reads the cost of *discovery + tool aggregation only* - nothing else. That
is why:

- **All agents advertise the SAME single tool** (`noop`, empty params). No real
  skill runs, no tool is ever dispatched, so no read/write work from tool
  execution interferes with the power trace. In the leader's registry this shows
  as **one distinct tool whose owner count climbs 1 -> 100**.
- The leader writes its results JSON **only when the fleet size changes** (a
  handful of writes), never on a hot loop, so disk I/O does not pollute the
  measurement. A light heartbeat is logged every few seconds only to
  **time-align** the log with the external power capture.

Stdlib only. Both scripts reuse the **live** `ConnectionLogic.discovery`,
`ConnectionLogic.transport`, and (leader side) `LeaderLogic.leader_network` - no
live module is modified and no new preset is added.

## Roles

| Script | Runs on | What it does |
|---|---|---|
| [`mock_leader_stm32.py`](mock_leader_stm32.py) | STM32MP257F-DK | Live `Discovery` + `LeaderNetwork` (registry + pull-on-join). Discovers the fleet, aggregates the shared tool, records per-agent discovery->registration latency and a size-change timeline. No LLM, no election, no polling, no dispatch. |
| [`fake_sensing_fleet.py`](fake_sensing_fleet.py) | this laptop | Spawns the growing fleet. Each agent = live `Discovery` (announces + finds the leader) + `P2PTransport` serving `tools/list` with the one shared tool. Drives the 5-minute schedule and stops everything at the end. |

This intentionally does **not** reuse `LeaderLogic/run_mock_leader.py`: that mock
builds **no** `LeaderNetwork` and echoes an LLM stub - the opposite of what this
test needs (we want the live aggregation path, and no LLM).

## Ports

- Discovery is UDP **9999** for everyone (the live `DISCOVERY_PORT`).
- Fake agent *i* listens on TCP **5000 + i**, so the fleet spans **5001..5100**.
- The mock leader listens on TCP **5700** (kept clear of the fleet range).

TCP ports are self-advertised over discovery HELLOs, so no script needs to know
another's port ahead of time.

## How to run

Run from the `IMPLEMENTATION/` folder. **Start the leader first**, then the fleet.
You type nothing during a run.

```
# on the STM32
python LatencyTest2/mock_leader_stm32.py
# then on the laptop
python LatencyTest2/fake_sensing_fleet.py
```

Sequence: the fleet starts **1** agent; the leader discovers it; the fleet holds
**5 minutes**, then grows to **10 -> 20 -> 50 -> 100** (5 minutes each, cumulative),
holds 5 minutes at 100, then **stops all agents**. Total ~25 minutes. Once the
fleet stops announcing, the leader's watchdog (`PEER_TIMEOUT = 15 s`) drops the
peers - a clean end-of-run marker in the leader's log. Ctrl-C on the leader
writes a final snapshot; Ctrl-C on the fleet stops all agents early.

Both machines must be on the **same /24** for subnet-broadcast discovery to reach
across. If auto-detection is wrong, set the `AGENT_BROADCAST_ADDR` env var on both
(see `ConnectionLogic/discovery.py`).

To change the cadence or the size steps, edit `WAVE_INTERVAL` / `FLEET_SIZES` in
`fake_sensing_fleet.py`.

## Output

Written to `LatencyTest2/results/`:

- `leader_stm32_<ts>.json`
  - `found_at_s` / `registered_at_s` - per-agent discovery + tool-registration times (s from start)
  - `reg_latency_stats` - found->tool-registered latency (count / mean / std / min / max)
  - `timeline` - `(elapsed_s, peers, distinct_tools, tool_owners)` sampled at each fleet-size change

The fleet side writes no results (it is unmeasured); the board-side power capture
is external and correlated against the leader log's heartbeat timestamps.
