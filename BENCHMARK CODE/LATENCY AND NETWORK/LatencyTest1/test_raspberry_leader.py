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
import time

import bench_core
from LeaderLogic.complete_workflow import (
    Workflow1, DISPATCH_SYSTEM, ANSWER_SYSTEM, _parse_tool_call_from_text)
from LeaderLogic.leader_network import make_network

log = logging.getLogger("latencytest")

_HAILO_PROMPT_NOTE = ("Hailo genai renders its own chat template internally; the "
                      "exact rendered string is not exposed. 'messages' is what was "
                      "passed to generate(); tool descriptions are in this config's 'tools'.")

def _hailo_pieces(wf, messages, tools=None):
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

    cfg = json.loads((_IMPL / "software_config.json").read_text())
    dev_cfg = json.loads((_IMPL / "LeaderLogic" / "raspberry_config.json").read_text())

    hef_path = dev_cfg.get("hailo", {}).get("hef_path")
    if not hef_path:
        log.error("no hailo.hef_path in LeaderLogic/raspberry_config.json - cannot run")
        return
    if not pathlib.Path(hef_path).is_absolute():
        hef_path = str(_IMPL / hef_path)

    loop, transport, discovery = bench_core.start_network(
        "raspberry-leader-bench", bench_core.LEADER_PORT)

    net = make_network(transport, discovery,
                       reply_timeout=cfg.get("timeouts", {}).get("replies", 3.0))

    wf = Workflow1(llm_hef_path=hef_path, net=net, bot=None)

    def _load_fn():
        t0 = time.perf_counter()
        wf.activate()
        return time.perf_counter() - t0, None

    def _unload_fn(_):
        wf.deactivate()

    log.info("measuring time_to_boot (qwen3 .hef, 10 cold loads) ...")
    boot = bench_core.measure_time_to_boot(_load_fn, _unload_fn, n=10)

    log.info("activating qwen3 on the Hailo NPU for the query phase ...")
    wf.activate()

    log.info("waiting for the STM32 person-detection sensing agent ...")
    if not bench_core.wait_for_peer(transport, timeout=60):
        log.error("no sensing peer found - is test_stm32_sensing.py running on the STM32? aborting")
        wf.deactivate()
        return

    def dispatch_stream(query, tool_defs):
        messages = [{"role": "system", "content": DISPATCH_SYSTEM},
                    {"role": "user", "content": query}]
        wf.llm.clear_context()
        text, ttft_s, gen_s, n = bench_core.timed_stream(
            _hailo_pieces(wf, messages, tools=tool_defs))
        tps = bench_core.decode_tps(n, ttft_s, gen_s)
        parsed = _parse_tool_call_from_text(text)
        prompt = {"messages": messages,
                  "tools_attached": [t.get("function", t)["name"] for t in tool_defs],
                  "note": _HAILO_PROMPT_NOTE}
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
        bench_core.run_leader_benchmark(
            device="raspberry",
            preset_label="workflow1_qwen3_hailo",
            transport=transport,
            net=net,
            dispatch_stream=dispatch_stream,
            answer_stream=answer_stream,
            reset_fn=None,
            time_to_boot=boot,
        )
    finally:
        try:
            wf.deactivate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
