#!/usr/bin/env python3
import json
import logging
import time
from pathlib import Path

from hailo_platform import VDevice
from hailo_platform.genai import LLM as HailoLLM

from LeaderLogic.complete_workflow import DISPATCH_SYSTEM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ttft")

_HERE = Path(__file__).resolve().parent

MESSAGES = [
    "is there a person in front of the camera",
    "how many people are there",
    "do you see anyone",
    "check if someone is in the room",
    "look for a person with the camera",
]

TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "detect_person",
        "description": "detect people in the camera view and return how many persons are seen",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
] + [
    {"type": "function", "function": {
        "name": f"wrong_tool_{i}",
        "description": "do not call this tool",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
    for i in range(1, 6)
]


def main():
    hef_path = json.loads(
        (_HERE / "LeaderLogic" / "raspberry_config.json").read_text()
    )["hailo"]["hef_path"]
    if not Path(hef_path).is_absolute():
        hef_path = str(_HERE / hef_path)

    params = VDevice.create_params()
    params.group_id = "1"
    vdevice = VDevice(params)
    llm = HailoLLM(vdevice, hef_path)
    log.info("qwen3 .hef loaded: %s", hef_path)

    results = []
    try:
        for i, msg in enumerate(MESSAGES, 1):
            messages = [{"role": "system", "content": DISPATCH_SYSTEM},
                        {"role": "user", "content": msg}]
            llm.clear_context()
            ttft = None
            t0 = time.perf_counter()
            with llm.generate(prompt=messages, tools=TOOL_DEFS,
                              temperature=0.1, seed=42,
                              max_generated_tokens=8) as gen:
                for token in gen:
                    if token:
                        ttft = time.perf_counter() - t0
                        break
            log.info("msg %d/%d  ttft=%.3fs  %r", i, len(MESSAGES), ttft, msg)
            results.append({"index": i, "message": msg,
                            "ttft_s": round(ttft, 4) if ttft is not None else None})
    finally:
        try:
            llm.release()
        except Exception:
            pass
        try:
            vdevice.release()
        except Exception:
            pass

    out = _HERE / "ttft_qwen3_hailo.json"
    out.write_text(json.dumps({"model": "qwen3-1.7b-hailo", "results": results},
                              indent=2), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
