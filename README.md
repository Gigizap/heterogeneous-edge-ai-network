# Heterogeneous Edge AI Network

> This project has been implemented during my internship in STMICROELECTRONICS Italy in Agrate Brianza.

A distributed **agentic system** in which a fleet of heterogeneous embedded devices collaborate over a zero-configuration peer-to-peer (P2P) network, with **no cloud dependency**. The system answers natural-language queries by running a generative LLM on edge to dispatch hardware-accelerated skills - object detection, person detection, face recognition, camera capture - across the network and summarize the results back to the user.

---

## What's in here

The repository has two top-level parts: the working **implementation** of the network, and the **benchmark code** used to characterize the models and the edge hardware.

| Folder | Contents |
|---|---|
| [`IMPLEMENTATION/`](IMPLEMENTATION/README.md) | The full agentic P2P system: discovery, leader election, leader/sensing logic, LLM dispatch pipeline |
| [`BENCHMARK CODE/`](BENCHMARK%20CODE/README.md) | Accuracy, latency, and power-consumption benchmarks for the models and edge boards |

## Documentation map

Docs live next to the code they describe.

| | |
|---|---|
| [IMPLEMENTATION/README.md](IMPLEMENTATION/README.md) | System overview: LLM pipeline, skills, election, presets, configuration, first-run setup |
| [ConnectionLogic/README.md](IMPLEMENTATION/ConnectionLogic/README.md) | P2P layer: UDP discovery, TCP transport, connection troubleshooting |
| [LeaderLogic/leader.md](IMPLEMENTATION/LeaderLogic/leader.md) | Leader internals and inference backends |
| [LeaderLogic/backup.md](IMPLEMENTATION/LeaderLogic/backup.md) | Conversation backup and leader failover |
| [LeaderLogic/stm32mp257fdk/README.md](IMPLEMENTATION/LeaderLogic/stm32mp257fdk/README.md) | STM32MP257F-DK bring-up, from the box to a running agent |
| [BENCHMARK CODE/README.md](BENCHMARK%20CODE/README.md) | Index of the benchmark suites |

Benchmarks run in different places: **accuracy** and **FunctionGemma handlers** need an external CUDA workstation, **power consumption** runs on the edge boards, and **latency and network** mixes the two, with the network-scaling tests driving a real board from a laptop that simulates a fleet of up to 100 agents. Each suite's README says where its scripts run.

---

# How does it work?

## Definition of agents

The unit of the network is the **agent**, identified by a unique agent-ID and bound to its **own dedicated TCP port**. A TCP connection is therefore established per *agent*, not per device. There are two kinds:

- **Sensing agent** - executes skills (tools) on local hardware (NPU / accelerator / CPU) and returns the results.
- **Leader agent** - manages user interaction over Telegram, runs an on-device LLM, and formats the collected replies into a natural-language answer. There is exactly **one** leader agent in the network at any time.

Sensing agents and leader agents can **co-live on the same physical device** - each keeps its own agent-ID and its own port, so they act as two independent agents (two separate TCP endpoints) that happen to share an IP.

Leadership is decided by **an election process**. The election score is computed **per device** from its hardware (accelerator, RAM, CPU, memory bandwidth); the highest-scoring device runs the leader agent. If that agent goes down, the next-best device is promoted automatically - conversation history included (continuously backed up on the N strongest other devices, N = `replication_factor` in [`software_config.json`](IMPLEMENTATION/software_config.json), 2 by default).

2 approaches shown in the following pictures are proposed: they differ with respect to how tool calling is performed.
Given a user query through telegram:
1) approach 1 uses an LLM on the leader agent to perform distributed tool calling, then gathers the results and gives the user a natural language reply

![Approach 1: full-LLM dispatch](figures/approach1.png)

2) approach 2 uses a classifier or cross encoder to perform the tool call directly on the sensing agent, the LLM on the leader agent gathers the results and gives the user a natural language reply as in approach 1.

![Approach 2: classifier + LLM dispatch](figures/approach2.png)

NOTE: Only approach 1 is fully implemented!
---

## Runtime flow

The seven flowcharts below trace a device from power-on, through peer discovery, leader election and failover, to answering a user query.

Shape legend:

| Symbol | Meaning |
|---|---|
| Ellipse / oval | Terminator (entry point or "Done") |
| Rounded rectangle | Process / action step |
| Diamond | Decision / branch |
| Parallelogram | Input/Output: data sent to or received from another node |
| Coloured box with "(see X diagram)" text | Call into another flowchart (a subroutine) |
| Green box, red dashed border | The shared tool-list data store, and reads/writes on it |

**1. Device startup** - load agent presets, then enter the discovery loop.

![Device startup](figures/1_startup.png)

**2. Peer discovery** - the continuous announce/listen loop that finds other agents.

![Peer discovery](figures/2_discovery.png)

**3. Peer joined** - what happens when a new peer is discovered.

![Peer joined](figures/3_peer_joined.png)

**4. Peer lost** - handling a peer that stopped announcing.

![Peer lost](figures/4_peer_lost.png)

**5. Leader election** - score-based selection of the single leader.

![Leader election](figures/5_leader_electionpng.png)

**6. Query polling** - the leader's loop that receives user queries.

![Query polling](figures/6_polling_query.png)

**7. Query reply** - dispatch to sensing agents, aggregate, and answer the user.

![Query reply](figures/7_query_reply.png)

---

## Hardware targets

| Device | Role | Accelerator | Model |
|---|---|---|---|
| Raspberry Pi 5 + Hailo AI HAT+ 2 | Leader / Sensing | Hailo H10 (40 TOPS) | `qwen3:1.7b` |
| STM32MP257F-DK | Leader / Sensing | On-chip NPU | `functiongemma:270m` |
| Generic device (Linux or Windows PC) | Leader / Sensing | llama.cpp (CPU or GPU) | `functiongemma:270m`* |

\* new leader and sensing presets can be added as subfolders in `/LeaderLogic` and in `/SensingLogic`

---

## Quick start

[`IMPLEMENTATION/`](IMPLEMENTATION/README.md) is the folder you **copy onto every device** in the network. Every node runs the *same* codebase. On each device, install the requirements and run `main.py`. Some of them are device-specific and installed by hand (see the per-board setup guides in the documentation map). The first run is interactive: it asks for the agent-ID, the presets, the hardware figures used for scoring, and (on leader-capable devices) the Telegram bot token and allowed user IDs, then writes `device_profile.json` and installs the requirements for the roles you picked. Every run after that is automatic: it reads the profile, scores the hardware, discovers peers, and joins leader election.

```bash
# on each device, inside the copied IMPLEMENTATION/ folder
pip install -r requirements/base.txt   # plus generic.txt or hailo.txt per device
python main.py
```

To simulate a second device on the same machine (for testing), run another copy on a different port:

```bash
python main.py --port 5002
```

Per-device setup (Hailo packages, Ollama models, OpenSTLinux AI packages, Telegram token, etc.) is documented in **[IMPLEMENTATION/README.md](IMPLEMENTATION/README.md)**.

> **Note:** the mesh-networking and CLASSIFIER + LLM architecture sections in the sub-READMEs are described as theoretical / not yet implemented - see those documents for the current status.

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). See [LICENSE](LICENSE) for details, or read the full text at <https://creativecommons.org/licenses/by-nc-sa/4.0/>.
