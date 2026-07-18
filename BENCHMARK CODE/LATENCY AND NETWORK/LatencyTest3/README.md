# LatencyTest3 - network-overlay test with full per-message JSON logging (1-min waves)

> Back to [Latency and network README](../README.md)

Same design as [LatencyTest2](../LatencyTest2/README.md) - a **mock STM32 leader**
(live discovery + tool aggregation, **no LLM / no election / no dispatch**) plus a
laptop fleet of **fake sensing agents** that each expose **one shared placeholder
tool** - with two differences:

1. **1-minute waves** instead of 5 (`WAVE_INTERVAL = 60.0` in
   [`fake_sensing_fleet.py`](fake_sensing_fleet.py)), so a full `1 -> 10 -> 20 -> 50 -> 100`
   run takes ~5 minutes instead of ~25.
2. **The leader logs every JSON message it receives on its TCP port**
   (`LEADER_PORT = 5700`) at INFO. In this test that traffic is the fleet's
   `tools/list` replies; the log line shows the sender, an approximate on-wire
   byte size, and the raw JSON. This is the **application-level** view of the
   traffic through the leader port - pair it with an external packet monitor for
   true on-wire byte totals (see below).

All agents still advertise the SAME single tool (`noop`, empty params, never
dispatched), so no skill runs and the only work the leader does is the network
overlay: discover + aggregate. Stdlib only; both scripts reuse the **live**
`ConnectionLogic.discovery`, `ConnectionLogic.transport`, and (leader side)
`LeaderLogic.leader_network` unchanged.

## Roles

| Script | Runs on | What it does |
|---|---|---|
| [`mock_leader_stm32.py`](mock_leader_stm32.py) | STM32MP257F-DK (or this machine) | Live `Discovery` + `LeaderNetwork` (registry + pull-on-join). Discovers the fleet, aggregates the shared tool, **logs every JSON received on port 5700**, records per-agent discovery->registration latency and a size-change timeline. No LLM, no election, no polling, no dispatch. |
| [`fake_sensing_fleet.py`](fake_sensing_fleet.py) | this laptop | Spawns the growing fleet. Each agent = live `Discovery` (announces + finds the leader) + `P2PTransport` serving `tools/list` with the one shared tool. Drives the 1-minute schedule and stops everything at the end. |

## Ports

- Discovery is UDP **9999** for everyone (the live `DISCOVERY_PORT`).
- Fake agent *i* listens on TCP **5000 + i**, so the fleet spans **5001..5100**.
- The mock leader listens on TCP **5700** (kept clear of the fleet range). This is
  the port whose incoming JSON the leader logs.

TCP ports are self-advertised over discovery HELLOs, so no script needs to know
another's port ahead of time.

## How to run

Run from the `IMPLEMENTATION/` folder. **Start the leader first**, then the fleet.
You type nothing during a run. Both roles can run on **one machine** for a local
dry-run (subnet-broadcast discovery loops back over the host's own LAN IP).

```
# leader (on the STM32, or in one terminal on this machine)
python LatencyTest3/mock_leader_stm32.py
# then the fleet (on the laptop, or a second terminal on this machine)
python LatencyTest3/fake_sensing_fleet.py
```

Sequence: the fleet starts **1** agent; the leader discovers it; the fleet holds
**1 minute**, then grows to **10 -> 20 -> 50 -> 100** (1 minute each, cumulative),
holds 1 minute at 100, then **stops all agents**. Total ~5 minutes. Once the fleet
stops announcing, the leader's watchdog (`PEER_TIMEOUT = 15 s`) drops the peers - a
clean end-of-run marker in the leader's log. Ctrl-C on the leader writes a final
snapshot; Ctrl-C on the fleet stops all agents early.

Both machines must be on the **same /24** for subnet-broadcast discovery to reach
across. If auto-detection is wrong, set the `AGENT_BROADCAST_ADDR` env var on both
(see `ConnectionLogic/discovery.py`).

To change the cadence or the size steps, edit `WAVE_INTERVAL` / `FLEET_SIZES` in
`fake_sensing_fleet.py`.

## Output

Written to `LatencyTest3/results/`:

- `leader_stm32_<ts>.json`
  - `found_at_s` / `registered_at_s` - per-agent discovery + tool-registration times (s from start)
  - `reg_latency_stats` - found->tool-registered latency (count / mean / std / min / max)
  - `timeline` - `(elapsed_s, peers, distinct_tools, tool_owners)` sampled at each fleet-size change

Plus, on stdout, one `RX port 5700 ...` line per JSON message the leader receives -
redirect the leader's output to a file to keep the full app-level capture.

The fleet side writes no results (it is unmeasured); any board-side power capture
is external and correlated against the leader log's heartbeat timestamps.
