#!/usr/bin/env python3
"""
LatencyTest1/test_stm32_leader_shorter.py

FIRST TEST - couple B, LEADER side, SHORTER-PROMPT + IN-PLACE-TOOL variant.
Run on the STM32MP257F-DK. NO sensing agent, NO discovery, NO network.

Two differences from test_stm32_leader.py:

  1. PROMPTS: the LLM flow uses the base FunctionGemma developer TRIGGER only -
     "You are a model that can do function calling with the following functions" -
     for the dispatch (tool-call) stage, and no _ANSWER_SYSTEM for the answer
     (tool-reply) stage. Both stages attach the SAME full N-tool block (so the KV
     prefix stays warm across queries - see point 3).
  2. TOOL: instead of dispatching over the network to a sensing agent, the tool
     runs IN PLACE - it just sleeps a fixed 0.59s (the person-detection skill's
     measured, reliable runtime) and returns a canned detection reply. So
     tool_to_result_ms is a stable constant and no second device is needed.
  3. KV CACHE: WITHIN a config the closures do NOT call llm.reset(), so llama.cpp's
     longest-common-prefix reuse keeps the shared [TRIGGER + tool declarations]
     block warm across every query AND both stages (this works only because both
     stages share the same base developer prompt - the custom-prompt leader's
     _ANSWER_SYSTEM would evict the dispatch prefix). The cache IS reset once at the
     START of each config (when the tool count changes), so q0 measures the TRUE
     cold prefill of N tools, while q1..q9 only prefill the new user text and their
     dispatch TTFT drops sharply. That q0-vs-rest split is the whole point.

Everything else is the same: same GGUF, same streaming, and the SAME per-query
metrics - TTFT, tokens in/out, decode tk/s, all stage latencies, end-to-end,
prompt_specs - so its JSON is directly comparable to the networked tests. (The
10-cold-load time_to_boot measurement is skipped for this variant.)

Run (from the IMPLEMENTATION folder; nothing else needs to be running):
    python LatencyTest1/test_stm32_leader_shorter.py
Results: LatencyTest1/results/leader_stm32_shorter_*.json
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
import time

from llama_cpp import Llama

import bench_core
import LeaderLogic.stm32mp257fdk_leader as leadermod
# TRIGGER is FunctionGemma's base developer prompt (single source of truth).
from LeaderLogic.functiongemma_simple_handler import _build_prompt, _parse_calls, TRIGGER

log = logging.getLogger("latencytest")

_STOP = ["<end_of_turn>", "<end_function_call>", "<start_function_response>"]
_DISPATCH_MAX_TOKENS = 256
_ANSWER_MAX_TOKENS = 512

# In-place tool: no sensing agent, no network. A fixed 0.59s stands in for the
# real person-detection skill (measured, reliable runtime), so tool_to_result_ms
# is a stable constant instead of a network round trip.
_TOOL_EXEC_S = 0.59
_MOCK_REPLY = [{"from": "in-place-mock", "text": "detection says: no people detected"}]


def _run_tool_in_place(fn, args):
    """Execute the picked tool locally: sleep the fixed skill runtime, return a
    canned reply shaped like a real sensing reply ([{from, text}, ...])."""
    time.sleep(_TOOL_EXEC_S)
    return [dict(r) for r in _MOCK_REPLY]


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

    # ── Load the GGUF once (no boot timing, no leader, no network) ────────────
    _pb, repo, filename, label, _is_fg = leadermod._select_model(0)
    model_path = leadermod._MODELS_DIR / filename
    if not model_path.exists():
        log.info("downloading %s ...", filename)
        leadermod._download_model(repo, filename, model_path)

    log.info("loading %s for the query phase ...", label)
    llm = Llama(model_path=str(model_path), n_ctx=4096,
                chat_format="functiongemma",
                n_threads=os.cpu_count() or 4, verbose=False)

    # ── SHORTER flow: base TRIGGER only, streamed for TTFT ────────────────────
    # Reset the KV cache only when the tool count changes (first query of a new
    # config): q0 then measures the true cold prefill of N tools, q1.. run warm.
    # TOOL_COUNTS are distinct, so len(tool_defs) uniquely marks the config.
    _cache_state = {"last_n": None}

    def dispatch_stream(query, tool_defs):
        lc = _lc_tools(tool_defs)
        messages = [{"role": "developer", "content": TRIGGER},   # base prompt only
                    {"role": "user", "content": query}]
        prompt = _build_prompt(messages, lc)
        tokens_in = len(llm.tokenize(prompt.encode("utf-8"), special=True))
        # Reset ONLY on the first query of a new config (tool count changed) so q0
        # measures the true cold prefill of N tools; within a config keep the KV
        # cache warm so q1..q9 reuse the [TRIGGER + tools] prefix.
        if len(lc) != _cache_state["last_n"]:
            llm.reset()
            _cache_state["last_n"] = len(lc)
        text, ttft_s, gen_s, _ = bench_core.timed_stream(
            ch["choices"][0]["text"] for ch in llm.create_completion(
                prompt=prompt, stream=True, max_tokens=_DISPATCH_MAX_TOKENS,
                temperature=0.0, seed=42, repeat_penalty=1.1,
                top_p=0.95, top_k=64, stop=_STOP))
        tokens_out = max(llm.n_tokens - tokens_in, 0)
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
                    "prompt": prompt}
        return {"fn": None, "args": None, "reply_text": text.strip(),
                "ttft_s": ttft_s, "gen_s": gen_s,
                "tokens_in": tokens_in, "tokens_out": tokens_out, "decode_tps": tps,
                "prompt": prompt}

    def answer_stream(query, fn, args, replies, tool_defs):
        # Base TRIGGER developer prompt (no _ANSWER_SYSTEM). Attach ALL N tools -
        # the SAME block as dispatch - so the [TRIGGER + tools] prefix is NOT
        # evicted from the KV cache; the next query's dispatch then reuses it warm.
        # (Attaching only the called tool shrinks the tool block and forces every
        # later dispatch to re-prefill the other N-1 tools -> no speedup at n>1.)
        tool_call = {"function": {"name": fn, "arguments": args or {}}}
        messages = [
            {"role": "developer", "content": TRIGGER},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "name": fn,
             "content": json.dumps(replies, ensure_ascii=False)},
        ]
        lc = _lc_tools(tool_defs)
        prompt = _build_prompt(messages, lc)
        tokens_in = len(llm.tokenize(prompt.encode("utf-8"), special=True))
        # NO llm.reset(): reuse the warm prefix (same base prompt as dispatch); the
        # answer only prefills the tool-call + result turns on top of it.
        text, ttft_s, gen_s, _ = bench_core.timed_stream(
            ch["choices"][0]["text"] for ch in llm.create_completion(
                prompt=prompt, stream=True, max_tokens=_ANSWER_MAX_TOKENS,
                temperature=0.0, seed=42, repeat_penalty=1.1,
                top_p=0.95, top_k=64, stop=_STOP))
        tokens_out = max(llm.n_tokens - tokens_in, 0)
        tps = bench_core.decode_tps(tokens_out, ttft_s, gen_s)
        return {"text": text.strip(), "ttft_s": ttft_s, "gen_s": gen_s,
                "tokens_in": tokens_in, "tokens_out": tokens_out, "decode_tps": tps,
                "prompt": prompt}

    # LOCAL mode: no transport/net. tool_fn runs the 0.59s in-place tool, tools_for
    # supplies the same per-config tool sets the sensing agent would have exposed.
    bench_core.run_leader_benchmark(
        device="stm32_shorter",
        preset_label="functiongemma_cpu_base_prompt_inplace",
        dispatch_stream=dispatch_stream,
        answer_stream=answer_stream,
        reset_fn=None,
        tool_fn=_run_tool_in_place,
        tools_for=bench_core.build_tool_defs,
    )


if __name__ == "__main__":
    main()
