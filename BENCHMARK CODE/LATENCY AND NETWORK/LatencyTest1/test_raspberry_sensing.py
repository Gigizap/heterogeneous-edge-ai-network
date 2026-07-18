#!/usr/bin/env python3
"""
LatencyTest1/test_raspberry_sensing.py

FIRST TEST - couple B, SENSING side. Run this on the RASPBERRY PI 5 + Hailo.

The Pi's live preset (raspberrypi5_yolo_NPU) exposes OBJECT detection, not person
detection. As you specified, person detection here reuses the SAME live Hailo
model and applies the "if object == person then person detected" rule: it calls
the live detect_object() and counts the "person" class in its result. That skill
is exposed to the network as the tool "detect_people" (name + description come
from bench_core, shared with the other couple).

The other device (STM32) runs test_stm32_leader.py as the functiongemma leader,
which drives the tool-count configs and the queries.

Run (from the IMPLEMENTATION folder). Start this BEFORE the STM32 leader:
    python LatencyTest1/test_raspberry_sensing.py

Results: LatencyTest1/results/sensing_raspberry_*.json
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

    # LIVE skill: single-frame YOLOv8n object detection on the Hailo NPU. It
    # returns a human-readable string listing each detected object, e.g.
    # "raspberry detection says: detected 2 object(s): person (0.90), chair (0.60)".
    from SensingLogic.raspberrypi5_yolo_NPU.yolo_object_detection import detect_object

    def detect_people(**kwargs):
        """Run the live object-detection model, then keep only the 'person' class
        (the 'if object == person then person detected' rule). The heavy work - the
        Hailo inference inside detect_object() - is identical to the live skill, so
        time_to_reply reflects the real model."""
        raw = str(detect_object())
        # Each detected person appears as a "person (score)" entry; count them.
        count = raw.lower().count("person")
        if count > 0:
            return f"raspberry detection says: detected {count} person(s) [via object detection]"
        return "raspberry detection says: no people detected"

    loop, transport, discovery = bench_core.start_network(
        "raspberry-sensing-bench", bench_core.SENSING_PORT)

    bench = bench_core.SensingBench(
        device="raspberry",
        preset_label="raspberrypi5_yolo_NPU",
        skill_fn=detect_people,
        transport=transport,
    )
    bench.serve()


if __name__ == "__main__":
    main()
