#!/usr/bin/env python3
"""
LatencyTest1/test_stm32_sensing.py

FIRST TEST - couple A, SENSING side. Run this on the STM32MP257F-DK.

Exposes the LIVE STM32 person-detection skill (stm32mp257_yolo_CPU preset,
YOLOv8n on the CPU via TFLite) as the network tool "detect_people" (name +
description come from bench_core, shared with the other couple). The other
device (Raspberry Pi) runs test_raspberry_leader.py as the qwen3/Hailo leader,
which drives the tool-count configs and the queries.

Run (from the IMPLEMENTATION folder). Start this BEFORE the Pi leader:
    python LatencyTest1/test_stm32_sensing.py

It serves tools/list + tools/call, times every tools/call (time_to_reply, the
tool-received to result-sent interval, no network lag), switches how many tools
it exposes when the leader says so, and on the leader's done signal writes
avg +- std of time_to_reply per config and overall.
Results: LatencyTest1/results/sensing_stm32_*.json
"""

import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_IMPL = _HERE.parent
for _p in (str(_IMPL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging

import bench_core

log = logging.getLogger("latencytest")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s]  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # LIVE skill: single-frame YOLOv8n person detection on the STM32 CPU (TFLite).
    # NPU variant (uncomment to test _NPU instead of _CPU):
    #   from SensingLogic.stm32mp257_yolo_NPU.person_detection import detect_people
    from SensingLogic.stm32mp257_yolo_CPU.person_detection import detect_people

    loop, transport, discovery = bench_core.start_network(
        "stm32-sensing-bench", bench_core.SENSING_PORT)

    bench = bench_core.SensingBench(
        device="stm32",
        preset_label="stm32mp257_yolo_CPU",
        skill_fn=detect_people,
        transport=transport,
    )
    bench.serve()


if __name__ == "__main__":
    main()
