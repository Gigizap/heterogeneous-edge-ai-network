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
