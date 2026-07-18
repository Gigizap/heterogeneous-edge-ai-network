#!/usr/bin/env python3
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
from LeaderLogic.functiongemma_simple_handler import _build_prompt, _parse_calls, TRIGGER

log = logging.getLogger("latencytest")

_STOP = ["<end_of_turn>", "<end_function_call>", "<start_function_response>"]
_DISPATCH_MAX_TOKENS = 256
_ANSWER_MAX_TOKENS = 512

_TOOL_EXEC_S = 0.59
_MOCK_REPLY = [{"from": "in-place-mock", "text": "detection says: no people detected"}]

def _run_tool_in_place(fn, args):
    time.sleep(_TOOL_EXEC_S)
    return [dict(r) for r in _MOCK_REPLY]

def _lc_tools(tool_defs):
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

    _pb, repo, filename, label, _is_fg = leadermod._select_model(0)
    model_path = leadermod._MODELS_DIR / filename
    if not model_path.exists():
        log.info("downloading %s ...", filename)
        leadermod._download_model(repo, filename, model_path)

    log.info("loading %s for the query phase ...", label)
    llm = Llama(model_path=str(model_path), n_ctx=4096,
                chat_format="functiongemma",
                n_threads=os.cpu_count() or 4, verbose=False)

    _cache_state = {"last_n": None}

    def dispatch_stream(query, tool_defs):
        lc = _lc_tools(tool_defs)
        messages = [{"role": "developer", "content": TRIGGER},
                    {"role": "user", "content": query}]
        prompt = _build_prompt(messages, lc)
        tokens_in = len(llm.tokenize(prompt.encode("utf-8"), special=True))
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
