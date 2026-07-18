"""
BenchmarkLeader/bench_objects.py

Bench A — STM32 + FunctionGemma leader, OBJECT detection.

Measures the FunctionGemma (CPU, llama.cpp) leader's end-to-end latency vs the
number of tools. The correct tool is `object_detection`, served for real by the
RPi + Hailo sensing agent (preset `raspberrypi5_yolo_NPU`) running
`python main.py` on the OTHER device.

Run on the STM32 device (with the RPi sensing agent already up):
    python -m BenchmarkLeader.bench_objects
"""

import json
from pathlib import Path

from BenchmarkLeader.engine import setup_network, wait_for_peer, run_benchmark

OBJECT_DETECTION_QUERIES = [
    "detect the objects",
    "what objects are visible",
    "what do you see from the camera?",
    "tell me what objects you see",
    "what are the objects",
    "list everything you can see",
    "identify the objects in front of the camera",
    "what items are in view",
    "scan the scene and report the objects",
    "what's in the camera frame right now",
]

AGENT_ID     = "bench-objects-leader"
PORT         = 5600
CORRECT_TOOL = "object_detection"


def main():
    root = Path(__file__).parent.parent
    cfg = json.loads((root / "software_config.json").read_text())

    loop, transport, discovery = setup_network(AGENT_ID, PORT)

    # Boot the FunctionGemma leader (loads the GGUF; downloads it if missing).
    # It builds the shared NetworkCollector and wires it into our transport.
    import LeaderLogic.stm32mp257fdk_leader as leadermod
    print("[bench] booting FunctionGemma leader (this loads the model) …")
    wf = leadermod.boot(cfg=cfg, agent_id=AGENT_ID, transport=transport,
                        discovery=discovery, bot=None, profile={"score": 0})
    collector = wf.collector

    print("[bench] waiting for the RPi sensing agent on the network …")
    if not wait_for_peer(transport, timeout=30):
        print("[bench] no peer found — is the RPi `object_detection` sensing agent running? aborting.")
        return

    def reset():
        # keep each query independent (no carried-over chat history)
        if hasattr(wf, "_histories"):
            wf._histories.clear()
        if hasattr(wf, "_current_chat_id"):
            wf._current_chat_id = 0

    run_benchmark(
        name="objects_functiongemma_stm32",
        collector=collector,
        workflow=wf,
        correct_tool=CORRECT_TOOL,
        queries=OBJECT_DETECTION_QUERIES,
        reset=reset,
    )


if __name__ == "__main__":
    main()
