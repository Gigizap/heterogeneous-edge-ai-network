#!/usr/bin/env python3
"""
count_leader_tokens_in.py

The Raspberry+Hailo leader latency runs (IMPLEMENTATION/LatencyTest1/
test_raspberry_leader.py, varying the tool count, and LatencyTest0/
test_leader_raspberry_replies.py, varying the reply count) record everything
except tokens_in: the Hailo genai streaming API never exposes the prompt-token
count, so every dispatch_tokens_in / answer_tokens_in in the result JSON is null.
The run's varying dimension (n_tools or n_replies) is auto-detected.

This script recovers those counts OFFLINE, purely from the tokenizer. The CPU
Qwen3 path (qwen3_handler.py) uses the exact same tokenizer + ChatML/Hermes
prompt rendering the Hailo build renders internally, so we can rebuild the exact
prompt string each generation saw and tokenize it with the real Qwen3 GGUF
vocabulary to get the prompt-token count llama.cpp would report.

What it fills:
  * dispatch_tokens_in  - every sample (a dispatch always ran).
  * answer_tokens_in    - only samples that got a tool result (got_result == 1);
                          failed-dispatch samples had no answer step, so their
                          answer_tokens_in stays null.

What it leaves untouched:
  * dispatch_tokens_out / answer_tokens_out - Hailo streams the real generated
    tokens, so those counts are already the ground truth.
  * everything else (timings, replies, prompt_specs, ...) is copied verbatim.

Faithfulness: the prompt is rendered by qwen3_handler._build_prompt (the CPU
handler), including the `<think></think>` no-think scaffold, and tokenized with
`Llama.tokenize(prompt, add_bos=True, special=True)` - exactly how llama-cpp's
create_completion counts prompt_tokens. The GGUF is loaded vocab_only, so no
weights are read; it is fast and light.

Usage (from BENCHMARK CODE/ACCURACY, or anywhere):
    python count_leader_tokens_in.py \
        --source ../../new_results/raspberry_board/leader_raspberry_20260707_012636.json \
        --model  ../../IMPLEMENTATION/models/Qwen3-1.7B-Q4_K_M.gguf
Defaults point at the committed run + GGUF, so plain `python count_leader_tokens_in.py`
works from this folder.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # BENCHMARK CODE/ACCURACY -> BENCHMARK CODE -> repo
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))  # so `import qwen3_handler` works from any cwd

from llama_cpp import Llama          # noqa: E402
import qwen3_handler                 # noqa: E402  (_build_prompt, _thinking_from_messages)

log = logging.getLogger("count_tokens_in")

_DEFAULT_SOURCE = (_REPO / "new_results" / "raspberry_board"
                   / "leader_raspberry_20260707_012636.json")
_DEFAULT_MODEL = _REPO / "IMPLEMENTATION" / "models" / "Qwen3-1.7B-Q4_K_M.gguf"


def _prompt_tokens(llm, messages, tools):
    """Render `messages` (+ optional `tools`) with the CPU qwen3 handler and
    return the prompt-token count llama.cpp's create_completion would report.

    Mirrors qwen3_handler(): enable_thinking is derived from the /no_think or
    /think markers in the messages, and tokenize() uses the same add_bos/special
    flags create_completion uses.
    """
    enable_thinking = qwen3_handler._thinking_from_messages(messages)
    prompt = qwen3_handler._build_prompt(messages, tools, enable_thinking=enable_thinking)
    return len(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))


def _dispatch_messages(dispatch_system, query):
    return [{"role": "system", "content": dispatch_system},
            {"role": "user", "content": query}]


def _answer_messages(answer_system, query, fn, replies):
    """Rebuild the answer-step user turn exactly as test_raspberry_leader.py's
    answer_stream() built it (detect_people carries no arguments, so args={})."""
    command = json.dumps({"name": fn, "arguments": {}})
    reply_json = json.dumps(replies, ensure_ascii=False)
    user = (
        f"User request: {query}\n"
        f"Command executed: {command}\n\n"
        f"RESULT DATA (write your reply from this; the 'from' field is the device "
        f"that ran the tool, so name it in your answer)\n"
        f"{reply_json}\n"
    )
    return [{"role": "system", "content": answer_system},
            {"role": "user", "content": user}]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s]  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=_DEFAULT_SOURCE,
                    help="leader_raspberry_*.json produced by the Hailo latency run")
    ap.add_argument("--model", type=Path, default=_DEFAULT_MODEL,
                    help="Qwen3 GGUF whose tokenizer to use (loaded vocab_only)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <source dir>/leader_raspberry_definitive.json)")
    args = ap.parse_args()

    if not args.source.exists():
        log.error("source JSON not found: %s", args.source)
        sys.exit(1)
    if not args.model.exists():
        log.error("Qwen3 GGUF not found: %s", args.model)
        sys.exit(1)

    out_path = args.out or (args.source.parent / "leader_raspberry_definitive.json")

    data = json.loads(args.source.read_text(encoding="utf-8"))

    # System prompts + per-n_tools tool lists come straight from the recorded
    # prompt_specs, so we never import the Hailo-dependent complete_workflow.
    specs = data["prompt_specs"]
    dispatch_system = specs[0]["dispatch_prompt"]["messages"][0]["content"]
    answer_system = specs[0]["answer_prompt"]["messages"][0]["content"]
    # The run varies by tool count (LatencyTest1) or reply count (LatencyTest0);
    # detect which discriminator the prompt_specs use and key the tool lists by it.
    cfg_key = "n_tools" if "n_tools" in specs[0] else "n_replies"
    tools_by_cfg = {spec[cfg_key]: spec["tools"] for spec in specs}
    log.info("varying '%s' over %s | dispatch_sys=%dch answer_sys=%dch",
             cfg_key, sorted(tools_by_cfg), len(dispatch_system), len(answer_system))

    log.info("loading Qwen3 tokenizer (vocab_only) from %s", args.model)
    llm = Llama(model_path=str(args.model), vocab_only=True, verbose=False)

    n_dispatch = n_answer = 0
    for sample in data["samples"]:
        query = sample["query"]
        tools = tools_by_cfg[sample[cfg_key]]

        sample["dispatch_tokens_in"] = _prompt_tokens(
            llm, _dispatch_messages(dispatch_system, query), tools)
        n_dispatch += 1

        # An answer generation happened only when a tool result came back.
        if sample.get("got_result") == 1:
            msgs = _answer_messages(answer_system, query,
                                    sample["tool_called"], sample["tool_result_raw"])
            sample["answer_tokens_in"] = _prompt_tokens(llm, msgs, None)
            n_answer += 1
        # else: no answer step -> leave answer_tokens_in as recorded (null)

    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("filled %d dispatch + %d answer tokens_in over %d samples",
             n_dispatch, n_answer, len(data["samples"]))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
