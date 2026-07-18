"""
BenchmarkLeader/bench_people.py

Bench B — RPi + qwen3-on-Hailo leader, PEOPLE detection.

Measures the qwen3 (native Hailo .hef) leader's end-to-end latency vs the number
of tools. The correct tool is `person_detection`, served for real by the STM32
sensing agent (preset `stm32mp257_yolo_CPU` or `stm32mp257_yolo_NPU`) running
`python main.py` on the OTHER device.

Usage (run on the RPi/Hailo, with the STM32 sensing agent already up):
    python -m BenchmarkLeader.bench_people            # runs BOTH think + no_think
    python -m BenchmarkLeader.bench_people think      # only thinking
    python -m BenchmarkLeader.bench_people no_think    # only no-thinking
    python -m BenchmarkLeader.bench_people probe       # just check if Hailo qwen3 reasons
                                                       # (prints full raw, no network needed)
"""

import json
import sys
from pathlib import Path

from BenchmarkLeader.engine import (
    setup_network, wait_for_peer, run_benchmark, has_thinking,
)
from LeaderLogic.network_collector import NetworkCollector
from LeaderLogic.complete_workflow import Workflow1

PEOPLE_DETECTION_QUERIES = [
    "how many people are there",
    "is there a person in front of the camera",
    "do you see anyone",
    "count the people in the scene",
    "are there any people visible",
    "detect the people in the frame",
    "is anybody there",
    "tell me how many persons you can see",
    "check if there is someone in view",
    "are there humans in the camera frame",
]

AGENT_ID     = "bench-people-leader"
PORT         = 5602
CORRECT_TOOL = "person_detection"

# A throwaway tool used only by `probe` (so it needs no live sensing agent).
_PROBE_TOOL = [{"type": "function", "function": {
    "name": "person_detection",
    "description": "detect people in the camera view and return how many are seen",
    "parameters": {"type": "object", "properties": {}, "required": []},
}}]


def _load_hef_path(root: Path):
    p = root / "LeaderLogic" / "raspberry_config.json"
    if p.exists():
        return json.loads(p.read_text()).get("hailo", {}).get("hef_path")
    return None


def _wrap_capture(wf):
    """Wrap Workflow1._hailo_generate so we can read the last raw dispatch text.
    Returns a get_last_raw() callable."""
    last = {"raw": ""}
    orig = wf._hailo_generate

    def _cap(messages, tools=None):
        r = orig(messages, tools=tools)
        last["raw"] = r
        return r

    wf._hailo_generate = _cap
    return lambda: last["raw"]


def _build_leader(root):
    """Bring up the Hailo qwen3 workflow (no network needed yet)."""
    hef_path = _load_hef_path(root)
    if not hef_path:
        print("[bench] WARNING: no hailo.hef_path in LeaderLogic/raspberry_config.json "
              "— Workflow1.activate() will fail without it.")
    cfg = json.loads((root / "software_config.json").read_text())

    loop, transport, discovery = setup_network(AGENT_ID, PORT)
    collector = NetworkCollector(
        broadcaster   = transport.broadcast_sync,
        skills_timeout= cfg.get("timeouts", {}).get("fetchskills", 1.5),
        reply_timeout = cfg.get("timeouts", {}).get("replies",     3.0),
    )
    _prev = transport.on_message
    def _handler(msg: dict):
        collector.handle_incoming(msg)
        _prev(msg)
    transport.on_message = _handler

    wf = Workflow1(llm_hef_path=hef_path, collector=collector, bot=None)
    get_last_raw = _wrap_capture(wf)
    print("[bench] activating qwen3 on the Hailo NPU …")
    wf.activate()
    return transport, collector, wf, get_last_raw


def _probe(wf, get_last_raw):
    """One dispatch with default formatting and one with /no_think, printing the
    FULL raw each time, so you can SEE whether the Hailo qwen3 reasons and whether
    the soft switch suppresses it. No sensing agent required."""
    q = PEOPLE_DETECTION_QUERIES[0]
    print("\n" + "=" * 70)
    print("THINKING PROBE — Hailo qwen3 dispatch (full raw output below)")
    print("=" * 70)
    for label, query in [("DEFAULT (no switch)", q), ("/no_think", q + " /no_think")]:
        fn, args = wf._dispatch(query, _PROBE_TOOL)
        raw = get_last_raw()
        print(f"\n--- {label} ---")
        print(f"picked tool        : {fn}")
        print(f"thinking_detected  : {int(has_thinking(raw))}")
        print(f"raw ({len(raw)} chars):\n{raw}")
    print("\n" + "=" * 70)
    print("If DEFAULT shows a filled <think>…</think> and /no_think shows an empty\n"
          "or absent one, the soft switch works. If both look the same, the .hef\n"
          "bakes the template and /no_think is ignored — rely on dispatch_ms instead.")
    print("=" * 70)


def main():
    arg = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    root = Path(__file__).parent.parent

    transport, collector, wf, get_last_raw = _build_leader(root)

    if arg == "probe":
        _probe(wf, get_last_raw)
        return

    modes = ["think", "no_think"] if arg in ("both", "") else [arg]
    if any(m not in ("think", "no_think") for m in modes):
        print(f"[bench] unknown mode '{arg}' — use: think | no_think | both | probe")
        return

    print("[bench] waiting for the STM32 sensing agent on the network …")
    if not wait_for_peer(transport, timeout=30):
        print("[bench] no peer found — is the STM32 `person_detection` sensing agent running? aborting.")
        return

    for mode in modes:
        print(f"\n[bench] ===== running mode: {mode} =====")
        run_benchmark(
            name=f"people_qwen3_hailo_{mode}",
            collector=collector,
            workflow=wf,
            correct_tool=CORRECT_TOOL,
            queries=PEOPLE_DETECTION_QUERIES,
            think_mode=mode,
            get_last_raw=get_last_raw,
        )


if __name__ == "__main__":
    main()
