#!/usr/bin/env python3
"""
LatencyTest0/test_leader_stm32_replies_reset.py

ZEROTH TEST - STM32 leader, RESET variant. Run this on the STM32MP257F-DK.

Same live FunctionGemma-on-CPU leader path as LatencyTest1's STM32 leader (same
GGUF, same prompts, same _build_prompt/_parse_calls dispatch/answer), but:
  - the tool is ALWAYS exactly one (detect_people);
  - what varies is how many replies come back (1, 5, 10, 15, 20), built
    in-process as if that many sensing agents had answered (no network);
  - llm.reset() is called before EVERY generation, so each dispatch and each
    answer is an independent COLD prefill (no KV-cache carry-over). This is the
    baseline for the KV-reuse variant (test_leader_stm32_replies_kvreuse.py).

Everything is in this one file (plus the shared bench_core helper): because
network latency does not matter here, there is no sensing process to run.

Run (from the IMPLEMENTATION folder, nothing to type during the run):
    python LatencyTest0/test_leader_stm32_replies_reset.py

For each reply-count config, drive the 10 person queries through the LIVE
FunctionGemma prompts, STREAMING so we time query->first-token (TTFT),
query->tool call, result->reply, decode tk/s, and end-to-end.
Results: LatencyTest0/results/leader_stm32-reset_*.json
"""

import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_IMPL = _HERE.parent
for _p in (str(_IMPL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import os

from llama_cpp import Llama

import bench_core
import LeaderLogic.stm32mp257fdk_leader as leadermod
from LeaderLogic.functiongemma_simple_handler import _build_prompt, _parse_calls

log = logging.getLogger("latencytest0")

# Match the deployed FunctionGemma handler's generation stops so a tool call halts
# naturally (the handler appends these itself; we stream one level below it).
_STOP = ["<end_of_turn>", "<end_function_call>", "<start_function_response>"]
_DISPATCH_MAX_TOKENS = 256    # mirrors stm32mp257fdk_leader._dispatch
_ANSWER_MAX_TOKENS = 512      # mirrors stm32mp257fdk_leader._answer

# This variant resets the model between generations (cold prefill per generation).
RESET_KV = True


def _lc_tools(tool_defs):
    """Normalise tool defs to the flat OpenAI shape _build_prompt expects."""
    out = []
    for td in tool_defs:
        fn = td.get("function", td)
        out.append({"type": "function", "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {})}})
    return out


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s]  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Resolve + ensure the model on disk (as the live leader does) ──────────
    _pb, repo, filename, _label, _is_fg = leadermod._select_model(0)
    model_path = leadermod._MODELS_DIR / filename
    if not model_path.exists():
        log.info("downloading %s ...", filename)
        leadermod._download_model(repo, filename, model_path)

    # ── Load the LIVE FunctionGemma model for the query phase ─────────────────
    log.info("loading FunctionGemma for the query phase (reset variant) ...")
    llm = Llama(model_path=str(model_path), n_ctx=4096,
                chat_format="functiongemma",
                n_threads=os.cpu_count() or 4, verbose=False)

    # ── LIVE prompts, streamed so we can time TTFT (mirrors _dispatch/_answer) ─
    def dispatch_stream(query, tool_defs):
        lc = _lc_tools(tool_defs)
        messages = [{"role": "developer", "content": leadermod._DISPATCH_SYSTEM_FG},
                    {"role": "user", "content": query}]
        prompt = _build_prompt(messages, lc)
        tokens_in = len(llm.tokenize(prompt.encode("utf-8"), special=True))
        if RESET_KV:
            llm.reset()   # cold prefill per query -> comparable, independent timings
        text, ttft_s, gen_s, _ = bench_core.timed_stream(
            ch["choices"][0]["text"] for ch in llm.create_completion(
                prompt=prompt, stream=True, max_tokens=_DISPATCH_MAX_TOKENS,
                temperature=0.0, seed=42, repeat_penalty=1.1,
                top_p=0.95, top_k=64, stop=_STOP))
        tokens_out = max(llm.n_tokens - tokens_in, 0)   # exact (== completion_tokens)
        tps = bench_core.decode_tps(tokens_out, ttft_s, gen_s)

        calls = _parse_calls(text) if lc else []
        if calls:
            try:
                args = json.loads(calls[0]["arguments"])
            except Exception:
                args = {}
            return {"fn": calls[0]["name"], "args": args, "reply_text": "",
                    "ttft_s": ttft_s, "gen_s": gen_s,
                    "tokens_in": tokens_in, "tokens_out": tokens_out, "decode_tps": tps,
                    "prompt": prompt}   # exact FunctionGemma-rendered dispatch prompt
        return {"fn": None, "args": None, "reply_text": text.strip(),
                "ttft_s": ttft_s, "gen_s": gen_s,
                "tokens_in": tokens_in, "tokens_out": tokens_out, "decode_tps": tps,
                "prompt": prompt}

    def answer_stream(query, fn, args, replies, tool_defs):
        tool_call = {"function": {"name": fn, "arguments": args or {}}}
        messages = [
            {"role": "developer", "content": leadermod._ANSWER_SYSTEM},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "name": fn,
             "content": json.dumps(replies, ensure_ascii=False)},
        ]
        # Pass the called tool's def for context, as _answer does via last_tool_defs.
        tdmap = {td.get("function", td)["name"]: td.get("function", td) for td in tool_defs}
        called = tdmap.get(fn)
        tools = _lc_tools([called]) if called else None
        prompt = _build_prompt(messages, tools)
        tokens_in = len(llm.tokenize(prompt.encode("utf-8"), special=True))
        if RESET_KV:
            llm.reset()
        text, ttft_s, gen_s, _ = bench_core.timed_stream(
            ch["choices"][0]["text"] for ch in llm.create_completion(
                prompt=prompt, stream=True, max_tokens=_ANSWER_MAX_TOKENS,
                temperature=0.0, seed=42, repeat_penalty=1.1,
                top_p=0.95, top_k=64, stop=_STOP))
        tokens_out = max(llm.n_tokens - tokens_in, 0)
        tps = bench_core.decode_tps(tokens_out, ttft_s, gen_s)
        return {"text": text.strip(), "ttft_s": ttft_s, "gen_s": gen_s,
                "tokens_in": tokens_in, "tokens_out": tokens_out, "decode_tps": tps,
                "prompt": prompt}   # exact FunctionGemma-rendered answer prompt

    bench_core.run_reply_benchmark(
        device="stm32-reset",
        preset_label="functiongemma_cpu_reset",
        dispatch_stream=dispatch_stream,
        answer_stream=answer_stream,
    )


if __name__ == "__main__":
    main()
