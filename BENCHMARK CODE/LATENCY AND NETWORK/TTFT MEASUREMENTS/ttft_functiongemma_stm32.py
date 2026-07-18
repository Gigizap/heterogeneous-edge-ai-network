#!/usr/bin/env python3
import json
import logging
import os
import time
from pathlib import Path

from llama_cpp import Llama

from LeaderLogic.functiongemma_simple_handler import _build_prompt, _parse_calls

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ttft")

_HERE = Path(__file__).resolve().parent
_MODEL = _HERE / "models" / "functiongemma-270m-it-Q4_K_M.gguf"

DISPATCH_SYSTEM_FG = (
    "You are a smart-home agent dispatcher. "
    "You are a model that can do function calling with the following functions"
)
ANSWER_SYSTEM = (
    "You are a smart home assistant. Turn the data into a short, friendly reply to the user. "
    "Reply in English. Summarize the reply, "
    "MENTION ALL THE IMPORTANT DATA RECEIVED FROM THE TOOL "
    "AND THE AVAILABLE SENSORS WHEN REPLYING."
)

PERSON_QUERIES = [
    "look for a person you can see",
    "detect all the people you see now",
    "is there a person in front of the camera",
    "how many people are there",
    "do you see anyone",
    "check if a person is in the room",
    "are there any people visible",
    "detect the people in the frame",
    "is any person there right now",
    "tell me how many people you can see",
]

TOOL_COUNTS = [1, 6, 12, 18, 24]

MOCK_TOOL_REPLY = [{"from": "mock-sensing", "text": "no people detected"}]

STOP = ["<end_of_turn>", "<end_function_call>", "<start_function_response>"]
DISPATCH_MAX_TOKENS = 256
ANSWER_MAX_TOKENS = 512


def make_tools(n):
    if n <= 0:
        return []
    tools = [{"type": "function", "function": {
        "name": "detect_people",
        "description": ("call this tool when the user asks about people or a person, "
                        "how many people can you see? who is there? this is the right tool to call"),
        "parameters": {"type": "object", "properties": {}, "required": []}}}]
    for i in range(1, n):
        tools.append({"type": "function", "function": {
            "name": f"Wrong_tool{i}",
            "description": "do NOT call this tool, this tool is WRONG",
            "parameters": {"type": "object", "properties": {}, "required": []}}})
    return tools


def _avg(vals):
    xs = [v for v in vals if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _generate(llm, prompt, max_tokens):
    tokens_in = len(llm.tokenize(prompt.encode("utf-8"), special=True))
    llm.reset()
    ttft = None
    out_text = ""
    t0 = time.perf_counter()
    for chunk in llm.create_completion(prompt=prompt, stream=True, max_tokens=max_tokens,
                                       temperature=0.0, seed=42, repeat_penalty=1.1,
                                       top_p=0.95, top_k=64, stop=STOP):
        piece = chunk["choices"][0]["text"]
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - t0
            out_text += piece
    gen_s = time.perf_counter() - t0
    tokens_out = max(llm.n_tokens - tokens_in, 0)
    decode_s = gen_s - ttft if ttft is not None else None
    decode_tps = (round((tokens_out - 1) / decode_s, 2)
                  if tokens_out > 1 and decode_s and decode_s > 0 else None)
    return {"ttft_s": round(ttft, 4) if ttft is not None else None,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "gen_s": round(gen_s, 4), "decode_tps": decode_tps,
            "text": out_text}


def run_query(llm, tools, msg):
    disp_msgs = [{"role": "developer", "content": DISPATCH_SYSTEM_FG},
                 {"role": "user", "content": msg}]
    d = _generate(llm, _build_prompt(disp_msgs, tools), DISPATCH_MAX_TOKENS)

    calls = _parse_calls(d["text"]) if tools else []
    tool_called = calls[0]["name"] if calls else None
    correct = 1 if tool_called == "detect_people" else 0

    row = {
        "query": msg,
        "tool_called": tool_called,
        "correct": correct,
        "ttft_s": d["ttft_s"],
        "tokens_in": d["tokens_in"],
        "tokens_out": d["tokens_out"],
        "decode_tps": d["decode_tps"],
        "time_to_tool_call_s": d["gen_s"],
        "dispatch_output": d["text"].strip(),
        "time_result_to_reply_s": None,
        "reply_tokens_out": None,
        "reply": None,
        "end_to_end_s": d["gen_s"],
    }

    if correct:
        tool_call = {"function": {"name": tool_called, "arguments": calls[0]["arguments"]}}
        ans_msgs = [
            {"role": "developer", "content": ANSWER_SYSTEM},
            {"role": "user", "content": msg},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "name": tool_called,
             "content": json.dumps(MOCK_TOOL_REPLY, ensure_ascii=False)},
        ]
        a = _generate(llm, _build_prompt(ans_msgs, [tools[0]]), ANSWER_MAX_TOKENS)
        row["time_result_to_reply_s"] = a["gen_s"]
        row["reply_tokens_out"] = a["tokens_out"]
        row["reply"] = a["text"].strip()
        row["end_to_end_s"] = round(d["gen_s"] + a["gen_s"], 4)

    return row


def main():
    llm = Llama(model_path=str(_MODEL), n_ctx=4096,
                n_threads=os.cpu_count() or 4, verbose=False)
    log.info("functiongemma loaded: %s", _MODEL)

    configs = []
    for n in TOOL_COUNTS:
        tools = make_tools(n)
        results = []
        for i, msg in enumerate(PERSON_QUERIES, 1):
            row = {"index": i, **run_query(llm, tools, msg)}
            results.append(row)
            log.info("tools=%-2d  q%2d/%d  ttft=%.3fs  in=%d out=%d  tps=%s  "
                     "ttc=%.3fs  r2r=%s  e2e=%.3fs  called=%s",
                     n, i, len(PERSON_QUERIES), row["ttft_s"], row["tokens_in"],
                     row["tokens_out"], row["decode_tps"], row["time_to_tool_call_s"],
                     row["time_result_to_reply_s"], row["end_to_end_s"], row["tool_called"])
        summary = {
            "accuracy": _avg([r["correct"] for r in results]),
            "ttft_s": _avg([r["ttft_s"] for r in results]),
            "tokens_in": _avg([r["tokens_in"] for r in results]),
            "tokens_out": _avg([r["tokens_out"] for r in results]),
            "decode_tps": _avg([r["decode_tps"] for r in results]),
            "time_to_tool_call_s": _avg([r["time_to_tool_call_s"] for r in results]),
            "time_result_to_reply_s": _avg([r["time_result_to_reply_s"] for r in results]),
            "end_to_end_s": _avg([r["end_to_end_s"] for r in results]),
        }
        log.info("tools=%-2d  AVG acc=%s ttft=%.3fs in=%.1f out=%.1f tps=%s "
                 "ttc=%.3fs r2r=%s e2e=%.3fs",
                 n, summary["accuracy"], summary["ttft_s"], summary["tokens_in"],
                 summary["tokens_out"], summary["decode_tps"],
                 summary["time_to_tool_call_s"], summary["time_result_to_reply_s"],
                 summary["end_to_end_s"])
        configs.append({"n_tools": n, "results": results, "avg": summary})

    out = _HERE / "ttft_functiongemma_stm32.json"
    out.write_text(json.dumps({"model": "functiongemma-270m-cpu",
                               "dispatch_max_tokens": DISPATCH_MAX_TOKENS,
                               "answer_max_tokens": ANSWER_MAX_TOKENS,
                               "dispatch_system": DISPATCH_SYSTEM_FG,
                               "answer_system": ANSWER_SYSTEM,
                               "configs": configs},
                              indent=2), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
