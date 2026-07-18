#!/usr/bin/env python3
"""
LatencyTest0/test_leader_stm32_replies_kvreuse.py

ZEROTH TEST - STM32 leader, KV-REUSE variant. Run this on the STM32MP257F-DK.

Like test_leader_stm32_replies_reset.py, but with two changes that together let
the KV cache actually be re-used:
  1. llm.reset() is NEVER called between generations, so the context is left
     intact. llama.cpp's create_completion() then detects the longest common
     prefix between the new prompt and the tokens already in the KV cache and
     re-uses it instead of re-evaluating it, so prefill (and therefore TTFT /
     query->tool time) is faster whenever consecutive prompts share a prefix.
  2. The prompts use the bare FunctionGemma TRIGGER developer block ONLY (no
     custom smart-home system prompt). That makes the developer block (TRIGGER +
     the single tool declaration) IDENTICAL for the dispatch and answer stages
     and across every query, which is what gives the re-used prefix its length -
     with divergent system prompts the shared prefix would be just a few tokens.
This is the KV-cache-reuse counterpart to the RESET baseline; comparing the two
isolates how much cold prefill costs on this board.

Note: dispatch_tokens_in / answer_tokens_in still report the FULL prompt size
(what was fed); the reuse shows up as lower dispatch_ttft_ms / query_to_tool_ms,
not as fewer input tokens. tokens_out stays exact (llm.n_tokens - tokens_in).

The tool is ALWAYS exactly one (detect_people); what varies is how many replies
come back (1, 5, 10, 15, 20), built in-process (no network, no sensing process).

Run (from the IMPLEMENTATION folder, nothing to type during the run):
    python LatencyTest0/test_leader_stm32_replies_kvreuse.py

Results: LatencyTest0/results/leader_stm32-kvreuse_*.json
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

# This variant leaves the context intact so llama.cpp re-uses the cached prefix.
RESET_KV = False


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
    log.info("loading FunctionGemma for the query phase (KV-reuse variant) ...")
    llm = Llama(model_path=str(model_path), n_ctx=4096,
                chat_format="functiongemma",
                n_threads=os.cpu_count() or 4, verbose=False)

    # ── LIVE prompts, streamed so we can time TTFT (mirrors _dispatch/_answer) ─
    def dispatch_stream(query, tool_defs):
        lc = _lc_tools(tool_defs)
        # Trigger-only: omit the custom system prompt so _build_prompt renders the
        # bare FunctionGemma TRIGGER developer block. That block (TRIGGER + the one
        # tool declaration) is then IDENTICAL for dispatch and answer and across
        # every query, so llama.cpp re-uses its cached prefix maximally.
        messages = [{"role": "user", "content": query}]
        prompt = _build_prompt(messages, lc)
        tokens_in = len(llm.tokenize(prompt.encode("utf-8"), special=True))
        if RESET_KV:
            llm.reset()   # KV-reuse variant: skipped, so the cached prefix is re-used
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
        # Trigger-only (see dispatch_stream): no custom system prompt, so the
        # developer block matches the dispatch stage for maximal KV-cache reuse.
        messages = [
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
        device="stm32-kvreuse",
        preset_label="functiongemma_cpu_kvreuse",
        dispatch_stream=dispatch_stream,
        answer_stream=answer_stream,
    )


if __name__ == "__main__":
    main()
