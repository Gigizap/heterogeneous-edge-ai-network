#!/usr/bin/env python3
"""functiongemma CPU — continuous LLM inference for benchmarking."""

import os, sys, time, signal, itertools
from llama_cpp import Llama

GGUF_PATH  = "functiongemma-270m-it-Q4_K_M.gguf"
N_THREADS  = os.cpu_count() or 4
N_CTX      = 2048
PROMPT     = (
    "Write a detailed step-by-step explanation of how a four-stroke internal "
    "combustion engine works, covering intake, compression, power, and exhaust."
)
MAX_TOKENS = 512

WINDOW_SECONDS = 20 * 60
TAG = os.environ.get("BENCH_TAG", "")
RESULT_FILE = f"avg_tps_cpu_gemma_{TAG}.txt" if TAG else "avg_tps_cpu_gemma.txt"

_STOP = False
def _handle_sigint(signum, frame):
    global _STOP
    _STOP = True
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print("\n[gemma] stopping after current token...")


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
    print(f"\n[gemma] >>> saved to {RESULT_FILE}: "
          f"gen {gen_tps:.2f} tok/s, total {total_tps:.2f} tok/s")


def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    print(f"[gemma] loading {GGUF_PATH} on {N_THREADS} threads...")
    llm = Llama(model_path=GGUF_PATH, n_ctx=N_CTX, n_threads=N_THREADS,
                n_gpu_layers=0, verbose=False)
    print(f"[gemma] loaded. Results → {RESULT_FILE}. Ctrl+C to stop.")

    start = time.time()
    tokens, gen_time, gap_time, n_gaps = 0, 0.0, 0.0, 0
    saved = False

    def maybe_save():
        nonlocal saved
        if not saved and (time.time() - start) >= WINDOW_SECONDS:
            save_stats(tokens, gen_time, gap_time, n_gaps); saved = True

    try:
        prev_gen_end = start
        for cycle in itertools.count(1):
            gap_start = prev_gen_end
            stream = llm.create_completion(prompt=PROMPT, max_tokens=MAX_TOKENS,
                                           temperature=0.8, stream=True)
            n_tok, gen_start = 0, None
            for chunk in stream:
                if gen_start is None:
                    gen_start = time.time()
                    gap_time += gen_start - gap_start
                    n_gaps += 1
                    last_tick = gen_start
                now = time.time()
                gen_time += now - last_tick
                last_tick = now
                n_tok += 1; tokens += 1
                sys.stdout.write(chunk["choices"][0]["text"]); sys.stdout.flush()
                maybe_save()
                if _STOP:
                    stream.close(); break

            prev_gen_end = time.time()
            gen_dt = (prev_gen_end - gen_start) if gen_start else 0.0
            tps = n_tok / gen_dt if gen_dt > 0 else 0.0
            print(f"\n[cycle {cycle:4d}] {n_tok:4d} tok  {gen_dt:5.2f}s = {tps:5.1f} tok/s")

            if _STOP:
                if not saved: save_stats(tokens, gen_time, gap_time, n_gaps)
                break
    except KeyboardInterrupt:
        pass
    print("\n[gemma] stopped.")

if __name__ == "__main__":
    main()
