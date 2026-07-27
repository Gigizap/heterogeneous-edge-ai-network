# Latency and network

> Back to [Benchmarks README](../README.md)

Latency and network-overlay benchmarks for the fleet: how fast a leader answers,
and how the discovery + tool-aggregation overlay scales with the number of agents.

Several scripts here import the live implementation (`ConnectionLogic`,
`LeaderLogic`, `SensingLogic`, `utils`, `models/`). Those were written to run
**from the `IMPLEMENTATION/` folder** on their target board, so their imports and
model paths resolve there, not from inside this folder. Each subfolder README says
where its scripts run.

## Latency tests

| Folder | Refers to | Devices |
|---|---|---|
| [LatencyTest0](LatencyTest0/README.md) | Leader latency as the number of **replies** returned by a single tool grows (1, 5, 10, 15, 20). Single device, no network. | Pi+Hailo, STM32 |
| [LatencyTest1](LatencyTest1/README.md) | Leader + sensing latency as the number of **tools** exposed grows (1, 6, 12, 18, 24). Two boards, real network dispatch. | Pi+Hailo, STM32 |
| [LatencyTest2](LatencyTest2/README.md) | Network-overlay scaling: a mock STM32 leader (discovery + tool aggregation, no LLM) against a fake fleet of 1 -> 100 agents, 5-minute waves, for an external power meter. | STM32 + laptop |
| [LatencyTest3](LatencyTest3/README.md) | Same as LatencyTest2 but 1-minute waves and the leader logs every JSON message it receives on its port. | STM32 + laptop |

## TTFT measurements

[TTFT MEASUREMENTS](TTFT%20MEASUREMENTS/README.md) - time-to-first-token for the
two leader backends (qwen3 on Hailo, functiongemma on the STM32 CPU, with and
without KV-cache reuse).

## Network workload simulator

[`run_empty_agents.py`](run_empty_agents.py) - run on a computer to simulate the
network workload of N agents. Each spawns its own discovery + transport, announces
itself, and every 5s sends a message to a peer named "leader". Use it to put a
fleet's worth of load on a real leader without N physical boards:
`python run_empty_agents.py 20`.

## Analysis and data

| Folder | Contents |
|---|---|
| [SCRIPTS](SCRIPTS) | `aggregate_network.py` turns the leader-side traffic capture in `network_data/` into per-fleet-size metrics. |
| [REPORT SCRIPTS](REPORT%20SCRIPTS) | `explore.py`, `analysis.py`, `gen_tex.py`, `validate_tex.py`: read the logs in `NEW RESULTS/`, compute the statistics, and generate the LaTeX tables and figures. Run from this folder. |
| [NEW RESULTS](NEW%20RESULTS) | The leader-log JSON result sets the report is built from: `VARYING TOOLS 1 reply/` and `VARYING REPLIES 1 tool/`. |
