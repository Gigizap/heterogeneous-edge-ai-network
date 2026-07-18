#!/usr/bin/env python3
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

    from SensingLogic.raspberrypi5_yolo_NPU.yolo_object_detection import detect_object

    def detect_people(**kwargs):
        raw = str(detect_object())
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
