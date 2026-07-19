#!/usr/bin/env python3
import os
import time
import signal
import itertools
from hailo_platform import VDevice
from hailo_platform.genai import LLM as HailoLLM

HEF_PATH = "Qwen3-1.7B-Instruct.hef"

MESSAGES = [
    {"role": "system", "content": "You are a helpful, detailed assistant."},
    {"role": "user", "content": (
        "Write a detailed step-by-step explanation of how a four-stroke internal "
        "combustion engine works, covering intake, compression, power, and exhaust."
    )},
]
TEMPERATURE = 0.8
SEED        = 42
MAX_TOKENS  = 512

WINDOW_SECONDS = 20 * 60
_TAG        = os.environ.get("BENCH_TAG", "solo")
RESULT_FILE = f"avg_tps_hailo_qwen_{_TAG}.txt"

_STOP = False
def _handle_sigint(signum, frame):
    global _STOP
    _STOP = True
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print("\n[qwen_hailo] stopping after current token... (Ctrl+C again to force)",
          flush=True)

def save_stats(tokens, gen_time, gap_time, n_gaps):
    total_time = gen_time + gap_time
    gen_tps   = tokens / gen_time   if gen_time   > 0 else 0.0
    total_tps = tokens / total_time if total_time > 0 else 0.0
    with open(RESULT_FILE, "w") as f:
        f.write(f"added_gap_load_prompt_time={gap_time:.3f}\n")
        f.write(f"number_of_gaps={n_gaps}\n")
        f.write(f"total_time={total_time:.3f}\n")
        f.write(f"generation_only_tok_per_s={gen_tps:.3f}\n")
        f.write(f"total_tok_per_s={total_tps:.3f}\n")
        f.write(f"total_tokens={tokens}\n")
    print(f"\n[qwen_hailo] >>> saved to {RESULT_FILE}: "
          f"gen-only {gen_tps:.2f} tok/s, total {total_tps:.2f} tok/s, "
          f"gap {gap_time:.1f}s over {n_gaps} gaps", flush=True)

S = {"start": 0.0, "tokens": 0, "gen_time": 0.0, "gap_time": 0.0,
     "n_gaps": 0, "saved": False}

def generate_once(llm, gap_start) -> int:
    n = 0
    gen_start = None
    last_tick = None
    try:
        with llm.generate(prompt=MESSAGES, temperature=TEMPERATURE, seed=SEED,
                          max_generated_tokens=MAX_TOKENS) as gen:
            for _token in gen:
                if gen_start is None:
                    gen_start = time.time()
                    S["gap_time"] += gen_start - gap_start
                    S["n_gaps"] += 1
                    last_tick = gen_start

                now = time.time()
                S["gen_time"] += now - last_tick
                last_tick = now

                n += 1
                S["tokens"] += 1
                print(_token, end="", flush=True)

                if not S["saved"] and (now - S["start"]) >= WINDOW_SECONDS:
                    save_stats(S["tokens"], S["gen_time"], S["gap_time"], S["n_gaps"])
                    S["saved"] = True

                if _STOP:
                    break
    except Exception as e:
        print(f"\n[qwen_hailo] generator stopped ({e})", flush=True)
    return n

def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    params = VDevice.create_params()
    params.group_id = "1"
    vdevice = VDevice(params)
    llm = HailoLLM(vdevice, HEF_PATH)
    print(f"[qwen_hailo] LLM loaded from {HEF_PATH}. Continuous load - Ctrl+C to stop.",
          flush=True)

    S["start"] = time.time()

    try:
        prev_gen_end = S["start"]
        for cycle in itertools.count(1):
            try:
                llm.clear_context()
            except Exception:
                pass
            n = generate_once(llm, prev_gen_end)
            prev_gen_end = time.time()
            print(f"\n[cycle {cycle:4d}] {n:4d} tokens", flush=True)

            if _STOP:
                if not S["saved"]:
                    save_stats(S["tokens"], S["gen_time"], S["gap_time"], S["n_gaps"])
                    S["saved"] = True
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not S["saved"]:
            save_stats(S["tokens"], S["gen_time"], S["gap_time"], S["n_gaps"])
            S["saved"] = True
        try:
            llm.release()
        except Exception as e:
            print(f"[qwen_hailo] llm.release() failed: {e}")
        try:
            vdevice.release()
        except Exception as e:
            print(f"[qwen_hailo] vdevice.release() failed: {e}")
        print("[qwen_hailo] stopped.", flush=True)

if __name__ == "__main__":
    main()
