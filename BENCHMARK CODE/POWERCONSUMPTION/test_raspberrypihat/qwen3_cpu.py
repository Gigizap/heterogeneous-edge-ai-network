#!/usr/bin/env python3
"""
qwen_cpu.py  --  continuous LLM inference load on the Raspberry Pi 5 CPU
                 (for power measurement)

Loads qwen3:1.7b (.gguf) once via llama.cpp, then loops forever.
Saves stats either when 20 minutes of wall-clock elapse OR when you press
Ctrl+C (whichever first), using everything up to that moment. On Ctrl+C the
current cycle is treated as finished right then.

Per cycle we measure two phases:
  gap  = time from end of previous generation to the FIRST token of this cycle
         (i.e. clear_context + prompt prefill / load)
  gen  = time spent streaming tokens

Saved stats:
  added_gap_load_prompt_time  = total gap seconds
  number_of_gaps              = how many gaps occurred
  total_time                  = gap + gen seconds
  generation_only_tok_per_s   = tokens / gen seconds
  total_tok_per_s             = tokens / total seconds

Run:   python3 qwen_cpu.py        Stop: Ctrl+C
"""
import os
import sys
import time
import signal
import itertools
from llama_cpp import Llama

GGUF_PATH = "Qwen3-1.7B-Q4_K_M.gguf"

N_THREADS = os.cpu_count() or 4
N_CTX     = 2048
PROMPT = (
    "Write a detailed step-by-step explanation of how a four-stroke internal "
    "combustion engine works, covering intake, compression, power, and exhaust."
)
MAX_TOKENS = 512

WINDOW_SECONDS = 20 * 60
_TAG        = os.environ.get("BENCH_TAG", "solo")
RESULT_FILE = f"avg_tps_cpu_qwen_{_TAG}.txt"

_STOP = False
def _handle_sigint(signum, frame):
    global _STOP
    _STOP = True
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print("\n[qwen_cpu] stopping after current token... (Ctrl+C again to force)")


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
    print(f"\n[qwen_cpu] >>> saved to {RESULT_FILE}: "
          f"gen-only {gen_tps:.2f} tok/s, total {total_tps:.2f} tok/s, "
          f"gap {gap_time:.1f}s over {n_gaps} gaps")


def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"[qwen_cpu] loading {GGUF_PATH} on {N_THREADS} CPU threads...")
    llm = Llama(model_path=GGUF_PATH, n_ctx=N_CTX, n_threads=N_THREADS,
                n_gpu_layers=0, verbose=False)
    print("[qwen_cpu] loaded. Starting continuous load. Ctrl+C to stop.")

    start = time.time()
    tokens   = 0
    gen_time = 0.0
    gap_time = 0.0
    n_gaps   = 0
    saved = False

    def maybe_save():
        nonlocal saved
        if not saved and (time.time() - start) >= WINDOW_SECONDS:
            save_stats(tokens, gen_time, gap_time, n_gaps)
            saved = True

    try:
        prev_gen_end = start
        for cycle in itertools.count(1):
            # gap phase begins here (clear_context + building/prefilling the prompt)
            gap_start = prev_gen_end
            stream = llm.create_completion(prompt=PROMPT, max_tokens=MAX_TOKENS,
                                           temperature=0.8, stream=True)

            n_tokens = 0
            gen_start = None
            for chunk in stream:
                if gen_start is None:                 # first token of this cycle
                    gen_start = time.time()
                    gap_time += gen_start - gap_start
                    n_gaps += 1
                    last_tick = gen_start

                now = time.time()
                gen_time += now - last_tick            # accrue gen time per token
                last_tick = now

                n_tokens += 1
                tokens   += 1
                sys.stdout.write(chunk["choices"][0]["text"]); sys.stdout.flush()

                maybe_save()
                if _STOP:
                    stream.close()
                    break

            prev_gen_end = time.time()
            gen_dt = (prev_gen_end - gen_start) if gen_start else 0.0
            tps = n_tokens / gen_dt if gen_dt > 0 else 0.0
            print(f"\n[cycle {cycle:4d}] {n_tokens:4d} tokens "
                  f"gen {gen_dt:6.2f}s = {tps:5.1f} tok/s")

            if _STOP:
                if not saved:
                    save_stats(tokens, gen_time, gap_time, n_gaps)
                    saved = True
                break
    except KeyboardInterrupt:
        pass

    print("\n[qwen_cpu] stopped.")


if __name__ == "__main__":
    main()
