#!/usr/bin/env python3
"""
LatencyTest0/test_leader_raspberry_replies.py

ZEROTH TEST - Raspberry Pi leader. Run this on the RASPBERRY PI 5 + Hailo.

Same live qwen3-on-Hailo leader path as LatencyTest1's Raspberry leader (Workflow1
from complete_workflow.py, same prompts, same streaming, same clear_context each
generation), but:
  - the tool is ALWAYS exactly one (detect_people);
  - what varies is how many replies come back (1, 5, 10, 15, 20), built
    in-process as if that many sensing agents had answered (no network);
  - everything is in this one file (plus the shared bench_core helper): because
    network latency does not matter here, there is no sensing process to run.

The Hailo path already clears its context before each generation (as Workflow1
does live), so there is no reset/KV-reuse split here - only the STM32 has the two
variants.

Run (from the IMPLEMENTATION folder, nothing to type during the run):
    python LatencyTest0/test_leader_raspberry_replies.py

For each reply-count config, drive the 10 person queries through the LIVE
Workflow1 prompts, STREAMING the Hailo token stream so we time query->first-token
(TTFT), query->tool call, result->reply, decode tk/s, and end-to-end.
Results: LatencyTest0/results/leader_raspberry_*.json
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

import bench_core
from LeaderLogic.complete_workflow import (
    Workflow1, DISPATCH_SYSTEM, ANSWER_SYSTEM, _parse_tool_call_from_text)

log = logging.getLogger("latencytest0")

# The Hailo genai LLM applies its own chat template inside generate(), so (unlike
# the STM32/llama.cpp path) we cannot capture the exact rendered prompt string.
# We store the messages + attached tools we passed instead, for the JSON record.
_HAILO_PROMPT_NOTE = ("Hailo genai renders its own chat template internally; the "
                      "exact rendered string is not exposed. 'messages' is what was "
                      "passed to generate(); tool descriptions are in this config's 'tools'.")


def _hailo_pieces(wf, messages, tools=None):
    """Stream the Hailo token generator, yielding each token with the qwen3
    turn-end marker stripped (mirrors Workflow1._hailo_generate, one token at a
    time so bench_core.timed_stream can clock the first one)."""
    with wf.llm.generate(
        prompt=messages,
        tools=tools,
        temperature=wf._hailo_temperature,
        seed=wf._hailo_seed,
        max_generated_tokens=wf._hailo_max_tokens,
    ) as gen:
        for token in gen:
            yield token.replace("<|im_end|>", "")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s]  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    dev_cfg = json.loads((_IMPL / "LeaderLogic" / "raspberry_config.json").read_text())

    hef_path = dev_cfg.get("hailo", {}).get("hef_path")
    if not hef_path:
        log.error("no hailo.hef_path in LeaderLogic/raspberry_config.json - cannot run")
        return
    # Resolve relative hef paths against IMPLEMENTATION so cwd does not matter.
    if not pathlib.Path(hef_path).is_absolute():
        hef_path = str(_IMPL / hef_path)

    # No network in this test: the tool result is built in-process, so Workflow1's
    # net (available_tools/dispatch) is never used. We drive wf.llm directly.
    wf = Workflow1(llm_hef_path=hef_path, net=None, bot=None)

    log.info("activating qwen3 on the Hailo NPU for the query phase ...")
    wf.activate()

    # ── LIVE prompts, streamed so we can time TTFT (mirrors Workflow1 hooks) ───
    def dispatch_stream(query, tool_defs):
        messages = [{"role": "system", "content": DISPATCH_SYSTEM},
                    {"role": "user", "content": query}]
        wf.llm.clear_context()             # Workflow1 clears context each dispatch
        text, ttft_s, gen_s, n = bench_core.timed_stream(
            _hailo_pieces(wf, messages, tools=tool_defs))
        tps = bench_core.decode_tps(n, ttft_s, gen_s)
        parsed = _parse_tool_call_from_text(text)
        prompt = {"messages": messages,
                  "tools_attached": [t.get("function", t)["name"] for t in tool_defs],
                  "note": _HAILO_PROMPT_NOTE}
        # tokens_in: the Hailo genai streaming API does not expose the prompt
        # token count (unlike llama.cpp on the STM32), so it stays None here.
        if parsed is None:
            return {"fn": None, "args": None, "reply_text": text.strip(),
                    "ttft_s": ttft_s, "gen_s": gen_s,
                    "tokens_in": None, "tokens_out": n, "decode_tps": tps, "prompt": prompt}
        return {"fn": parsed.get("name"), "args": parsed.get("arguments", {}),
                "reply_text": "", "ttft_s": ttft_s, "gen_s": gen_s,
                "tokens_in": None, "tokens_out": n, "decode_tps": tps, "prompt": prompt}

    def answer_stream(query, fn, args, replies, tool_defs):
        command = json.dumps({"name": fn, "arguments": args or {}})
        reply_json = json.dumps(replies, ensure_ascii=False)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": (
                f"User request: {query}\n"
                f"Command executed: {command}\n\n"
                f"RESULT DATA (write your reply from this; the 'from' field is the device "
                f"that ran the tool, so name it in your answer)\n"
                f"{reply_json}\n"
            )},
        ]
        wf.llm.clear_context()
        text, ttft_s, gen_s, n = bench_core.timed_stream(_hailo_pieces(wf, messages))
        tps = bench_core.decode_tps(n, ttft_s, gen_s)
        prompt = {"messages": messages, "note": _HAILO_PROMPT_NOTE}
        return {"text": text.strip(), "ttft_s": ttft_s, "gen_s": gen_s,
                "tokens_in": None, "tokens_out": n, "decode_tps": tps, "prompt": prompt}

    try:
        bench_core.run_reply_benchmark(
            device="raspberry",
            preset_label="workflow1_qwen3_hailo",
            dispatch_stream=dispatch_stream,
            answer_stream=answer_stream,
        )
    finally:
        try:
            wf.deactivate()                  # release the Hailo device
        except Exception:
            pass


if __name__ == "__main__":
    main()
