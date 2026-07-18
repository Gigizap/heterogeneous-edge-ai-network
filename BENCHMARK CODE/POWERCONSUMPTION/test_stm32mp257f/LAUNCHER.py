#!/usr/bin/env python3
"""Launcher for STM32MP257F-DK benchmarks.
Runs each config for 21 min then auto-advances to the next."""

import os, sys, signal, subprocess, time

SCRIPTS = {
    "gemma":      "functiongemma_cpu.py",
    "detect_npu": "detect_person_npu.py",
    "detect_cpu": "detect_person_cpu.py",
    "leader":     "run_leader.py",
}

CONFIGS = [
    ("functiongemma CPU only",         ["gemma"],              "solo"),
    ("detect NPU only",               ["detect_npu"],         "solo"),
    ("detect CPU only",                ["detect_cpu"],         "solo"),
    ("detect NPU + functiongemma CPU", ["detect_npu", "gemma"], "with_other_npu"),
    ("detect CPU + functiongemma CPU", ["detect_cpu", "gemma"], "with_other_cpu"),
]

RUN_SECONDS = None  # set at startup


def run_config(label, keys, tag):
    print(f"\n{'='*60}")
    mins = RUN_SECONDS // 60
    print(f"  {label}   ({mins} min)")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["BENCH_TAG"] = tag

    procs = []
    for key in keys:
        script = SCRIPTS[key]
        print(f"    starting: python3 {script}  (tag={tag})")
        procs.append(subprocess.Popen(["python3", script], env=env))

    time.sleep(RUN_SECONDS)

    print(f"\n--- {mins} min elapsed, stopping {label} ---")
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired: p.kill()

    print("--- idle 20s before next config ---")
    time.sleep(20)


def main():
    try:
        mins = int(input("Minutes per config (e.g. 21): ").strip())
    except (ValueError, KeyboardInterrupt):
        sys.exit("Invalid.")
    global RUN_SECONDS
    RUN_SECONDS = mins * 60

    print(f"\n  1. Run ALL configs sequentially ({mins} min each)")
    for i, (label, _, _) in enumerate(CONFIGS, 2):
        print(f"  {i}. {label} only")
    print(f"  {len(CONFIGS)+2}. run_leader.py")
    print()

    try:
        choice = int(input("Enter number: ").strip())
    except (ValueError, KeyboardInterrupt):
        sys.exit("Invalid.")

    if choice == 1:
        print(f"\nRunning all {len(CONFIGS)} configs, {mins} min each.")
        print(f"Total estimated time: {len(CONFIGS)*mins} min.\n")
        try:
            for label, keys, tag in CONFIGS:
                run_config(label, keys, tag)
        except KeyboardInterrupt:
            print("\nAborted.")
        print("\n>>> ALL DONE <<<")

    elif 2 <= choice <= len(CONFIGS) + 1:
        label, keys, tag = CONFIGS[choice - 2]
        try:
            run_config(label, keys, tag)
        except KeyboardInterrupt:
            print("\nAborted.")

    elif choice == len(CONFIGS) + 2:
        subprocess.run(["python3", SCRIPTS["leader"]])

    else:
        sys.exit("Out of range.")


if __name__ == "__main__":
    main()