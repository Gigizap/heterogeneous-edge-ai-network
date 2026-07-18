"""Exploration helper: dumps the schema-level facts of every data file so the
latency analysis can be built on verified numbers, not assumptions.

Run from the WRITING_REPORT folder:  python scripts/explore.py
"""
import json
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "new_results")

FILES = {
    "qwen3_hailo_tools": "VARYING TOOLS 1 reply/leader_raspberry_20260707_170137_withtkincorrect.json",
    "fgemma_stm32_noKV_tools": "VARYING TOOLS 1 reply/leader_stm32_prompt_noKV.json",
    "fgemma_stm32_KV_tools": "VARYING TOOLS 1 reply/leader_stm32_shorter_prompt+KVcache-newpertoolbatch.json",
    "qwen3_hailo_replies": "VARYING REPLIES 1 tool/leader_raspberry_definitive.json",
    "fgemma_replies_reset": "VARYING REPLIES 1 tool/leader_stm32-reset_20260707_150325.json",
    "fgemma_replies_kvreuse": "VARYING REPLIES 1 tool/leader_stm32-kvreuse_20260707_154629.json",
}


def load(rel):
    with open(os.path.join(BASE, rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


for tag, rel in FILES.items():
    d = load(rel)
    print("=" * 80)
    print(tag, "->", rel)
    print("  role:", d.get("role"), "| device:", d.get("device"),
          "| preset:", d.get("leader_preset"))
    print("  correct_tool:", d.get("correct_tool"))
    print("  tool_counts:", d.get("tool_counts"))
    print("  reply_counts:", d.get("reply_counts"))
    print("  n queries:", len(d.get("queries", [])))
    print("  top-level keys:", list(d.keys()))
    ttb = d.get("time_to_boot", {})
    print("  time_to_boot: unit=%s count=%s mean=%s std=%s min=%s max=%s" % (
        ttb.get("unit"), ttb.get("count"), ttb.get("mean"),
        ttb.get("std"), ttb.get("min"), ttb.get("max")))
    samples = d.get("samples", [])
    print("  n samples:", len(samples))
    if samples:
        print("  sample keys:", list(samples[0].keys()))
    # per-bin availability of each numeric field
    binkey = "n_tools" if "n_tools" in samples[0] else "n_replies"
    if binkey not in samples[0]:
        # figure out the binning key
        for cand in ("n_tools", "n_replies", "n_reply", "reply_count"):
            if cand in samples[0]:
                binkey = cand
                break
    print("  bin key:", binkey)
    bins = {}
    for s in samples:
        b = s.get(binkey)
        bins.setdefault(b, []).append(s)
    fields = ["dispatch_ttft_ms", "dispatch_tokens_in", "dispatch_tokens_out",
              "query_to_tool_ms", "tool_to_result_ms", "result_to_reply_ms",
              "answer_ttft_ms", "answer_tokens_in", "answer_tokens_out",
              "end_to_end_ms", "correct", "got_result"]
    for b in sorted(bins, key=lambda x: (x is None, x)):
        rows = bins[b]
        n = len(rows)
        gotres = sum(1 for r in rows if r.get("got_result"))
        e2e_present = sum(1 for r in rows if r.get("end_to_end_ms") is not None)
        disp_ttft_present = sum(1 for r in rows if r.get("dispatch_ttft_ms") is not None)
        disp_tin_present = sum(1 for r in rows if r.get("dispatch_tokens_in") is not None)
        ans_tin_present = sum(1 for r in rows if r.get("answer_tokens_in") is not None)
        print("    %s=%s: n=%d got_result=%d e2e_present=%d disp_ttft=%d disp_tin=%d ans_tin=%d"
              % (binkey, b, n, gotres, e2e_present, disp_ttft_present, disp_tin_present, ans_tin_present))
