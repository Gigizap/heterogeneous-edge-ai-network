#!/usr/bin/env python3
"""
test_hailo_inference_LLM.py

Stupidly simple, self-contained smoke test for LLM inference on a Hailo NPU.
Loads the .hef and runs one prompt, streaming tokens to stdout.

Lives at the IMPLEMENTATION/ root (next to LeaderLogic/ and models/). The HEF
path is read from LeaderLogic/raspberry_hailo/config.json and resolved against
this folder, so it works no matter where you run from.

    python3 test_hailo_inference_LLM.py
"""

import json
from pathlib import Path

from hailo_platform import VDevice
from hailo_platform.genai import LLM as HailoLLM

IMPL_ROOT = Path(__file__).resolve().parent
CONFIG = IMPL_ROOT / "LeaderLogic" / "raspberry_hailo" / "config.json"
HEF = IMPL_ROOT / json.loads(CONFIG.read_text())["hailo"]["hef_path"]

MESSAGES = [
    {"role": "system", "content": "You are a helpful, concise assistant."},
    {"role": "user", "content": "In two short sentences, explain what an edge AI accelerator does."},
]


def main():
    if not HEF.is_file():
        raise SystemExit(f"HEF not found: {HEF}")

    params = VDevice.create_params()
    params.group_id = "1"
    vdevice = VDevice(params)
    llm = HailoLLM(vdevice, str(HEF))
    print(f"[test] loaded {HEF}\n")

    try:
        llm.clear_context()
        with llm.generate(prompt=MESSAGES, temperature=0.7, seed=42,
                          max_generated_tokens=256) as gen:
            for token in gen:
                print(token, end="", flush=True)
        print()
    finally:
        llm.release()
        vdevice.release()


if __name__ == "__main__":
    main()
