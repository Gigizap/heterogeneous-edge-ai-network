#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from llama_cpp import Llama
import qwen3_handler

log = logging.getLogger("count_tokens_in")

_DEFAULT_SOURCE = (_REPO / "new_results" / "raspberry_board"
                   / "leader_raspberry_20260707_012636.json")
_DEFAULT_MODEL = _REPO / "IMPLEMENTATION" / "models" / "Qwen3-1.7B-Q4_K_M.gguf"

def _prompt_tokens(llm, messages, tools):
    enable_thinking = qwen3_handler._thinking_from_messages(messages)
    prompt = qwen3_handler._build_prompt(messages, tools, enable_thinking=enable_thinking)
    return len(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))

def _dispatch_messages(dispatch_system, query):
    return [{"role": "system", "content": dispatch_system},
            {"role": "user", "content": query}]

def _answer_messages(answer_system, query, fn, replies):
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

    specs = data["prompt_specs"]
    dispatch_system = specs[0]["dispatch_prompt"]["messages"][0]["content"]
    answer_system = specs[0]["answer_prompt"]["messages"][0]["content"]
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

        if sample.get("got_result") == 1:
            msgs = _answer_messages(answer_system, query,
                                    sample["tool_called"], sample["tool_result_raw"])
            sample["answer_tokens_in"] = _prompt_tokens(llm, msgs, None)
            n_answer += 1

    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("filled %d dispatch + %d answer tokens_in over %d samples",
             n_dispatch, n_answer, len(data["samples"]))
    log.info("wrote %s", out_path)

if __name__ == "__main__":
    main()
