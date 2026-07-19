#!/usr/bin/env python3
import os
import subprocess
import sys

PYTHON = sys.executable

CMDS = {
    "cpu_qwen":   ["qwen3_cpu.py"],
    "hailo_qwen": ["qwen3_hailo.py"],
    "cpu_yolo":   ["detect_cpu.py"],
    "hailo_yolo": ["detect_hailo.py"],
    "run_leader": ["run_leader.py"],
}

MENU = [
    ("cpu qwen",               ["cpu_qwen"]),
    ("hailo qwen",             ["hailo_qwen"]),
    ("cpu yolo",               ["cpu_yolo"]),
    ("hailo yolo",             ["hailo_yolo"]),
    ("rasp qwen + hailo yolo", ["cpu_qwen", "hailo_yolo"]),
    ("rasp yolo + hailo qwen", ["cpu_yolo", "hailo_qwen"]),
    ("cpu run_leader.py",      ["run_leader"]),
]

def make_env(tag):
    e = os.environ.copy()
    e["BENCH_TAG"] = tag
    return e

def main():
    print("\nChoose what to run:\n")
    for i, (label, _) in enumerate(MENU, 1):
        print(f"  {i}. {label}")
    print()
    try:
        choice = int(input("Enter number: ").strip())
    except (ValueError, KeyboardInterrupt):
        print("Invalid choice.")
        sys.exit(1)
    if not (1 <= choice <= len(MENU)):
        print("Out of range.")
        sys.exit(1)

    label, keys = MENU[choice - 1]
    tag = "solo" if len(keys) == 1 else "with_other"

    print(f"\n>>> Running: {label}  (tag={tag})\n")
    procs = []
    for key in keys:
        script = CMDS[key]
        print(f"    [{key}] {PYTHON} {' '.join(script)}")
        p = subprocess.Popen([PYTHON] + script, env=make_env(tag))
        procs.append((key, p))

    print("\n(Ctrl+C to stop everything)\n")
    try:
        for key, p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        for key, p in procs:
            p.terminate()
        for key, p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("Done.")

if __name__ == "__main__":
    main()
