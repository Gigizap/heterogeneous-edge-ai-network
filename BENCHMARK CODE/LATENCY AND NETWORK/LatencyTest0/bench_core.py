"""
LatencyTest0/bench_core.py

Shared engine for the ZEROTH latency test (single device, no network, no sensing
process). This is a trimmed copy of LatencyTest1/bench_core.py: the LEADER-side streaming
machinery is kept verbatim, but the whole SENSING side (the live TCP fleet,
discovery, tools/list handshake) and the boot-timing phase are dropped, because
here the tool result never travels over a wire and boot is not measured.

The difference from LatencyTest1 is the swept axis:

  * LatencyTest1 varies the NUMBER OF TOOLS the sensing agent exposes
    (1, 6, 12, 18, 24) and always gets one reply back.
  * LatencyTest0 exposes exactly ONE tool (detect_people) and varies the
    NUMBER OF REPLIES the tool returns (1, 5, 10, 15, 20), as if that many
    sensing agents had all answered the same call.

Because "network latency does not matter" here, the replies are built in-process
(build_replies) with the EXACT shape ToolDispatcher.dispatch() hands the leader:
a list of {"from": <agent-id>, "text": <str>}, one per answering agent. So the
answer LLM is fed the same result set a real multi-agent leader would aggregate;
only the (irrelevant) network hop is skipped. Everything measured is identical to
LatencyTest1's leader metrics, so the two tests' leader JSONs are comparable
field-for-field (the only renamed axis key is n_replies instead of n_tools).

Stdlib only. The heavy runtimes (Hailo, llama.cpp) come in only through the LIVE
preset modules the thin device scripts import - never from this file.
"""

import sys
import pathlib

# OS-agnostic imports: put the IMPLEMENTATION dir (parent) and this dir on the
# path so "import bench_core" and the device scripts' "from LeaderLogic..." both
# resolve no matter the current working directory or platform.
_HERE = pathlib.Path(__file__).resolve().parent
_IMPL = _HERE.parent
for _p in (str(_IMPL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import statistics
import time
from pathlib import Path

log = logging.getLogger("latencytest0")


# ── test configuration ───────────────────────────────────────────────────────

# 5 configurations: how many replies the single tool returns (as if this many
# sensing agents all answered the same detect_people call).
REPLY_COUNTS = (1, 5, 10, 15, 20)

# The one and only tool exposed in this test. Same name AND description as the
# LatencyTest1 correct tool, so the dispatch stage is measured on an identical
# prompt (and is expected to be flat across configs: the tool never changes).
CORRECT_TOOL = "detect_people"

DETECT_PEOPLE_DEF = {
    "type": "function",
    "function": {
        "name": CORRECT_TOOL,
        "description": (
            "Detects the people that are currently visible. Use this when the user "
            "says 'do you see a person?', 'how many people?', or 'is there anyone?'."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# 10 queries that each require the person-detection tool (same set as LatencyTest1).
PERSON_QUERIES = [
    "can you use the detect person tool",
    "is there a person in front of the camera",
    "how many people are there",
    "do you see anyone",
    "check if someone is in the room",
    "are there any people visible",
    "detect the people in the frame",
    "is anybody there right now",
    "tell me how many persons you can see",
    "look for a person with the camera",
]

# The text every agent returns. The user asked for a fixed reply ("no people
# detected"): the SIZE and SHAPE of the reply set is what varies, not its wording,
# so the answer LLM's growing summarization cost is isolated.
DEFAULT_REPLY_TEXT = "no people detected"

# Naming for the synthetic answering agents (build_replies stamps from-1..from-n).
SENSING_PREFIX = "sensing-agent"


# ── generic helpers ──────────────────────────────────────────────────────────

def build_replies(n: int, text: str = DEFAULT_REPLY_TEXT,
                  prefix: str = SENSING_PREFIX) -> list:
    """The tool result the leader 'receives', shaped EXACTLY like
    ToolDispatcher.dispatch() returns from n owner agents: one {from, text} per
    answering agent. Network latency is not modelled (this runs in-process); only
    the shape + size of the reply set the answer LLM must aggregate is real."""
    return [{"from": f"{prefix}-{i}", "text": text} for i in range(1, n + 1)]


def _stats(values, ndigits: int) -> dict:
    """count / mean / std (population) / min / max for a list of numbers."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": round(statistics.fmean(vals), ndigits),
        "std": round(statistics.pstdev(vals), ndigits) if len(vals) > 1 else 0.0,
        "min": round(min(vals), ndigits),
        "max": round(max(vals), ndigits),
    }


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _results_dir(out_dir) -> Path:
    d = Path(out_dir) if out_dir else (_HERE / "results")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── streaming helpers (leader side, verbatim from LatencyTest1) ──

def timed_stream(pieces):
    """Consume an iterator of text pieces (one per generated token), timing the
    FIRST token (TTFT) and the whole generation, while accumulating the text.

    Returns (text, ttft_s, gen_s, n_pieces). Empty pieces are ignored (they do
    not count as the first token nor toward n_pieces), so a model that emits a
    stop token as "" does not skew TTFT. n_pieces is a token count only when the
    backend yields one token per piece; callers that can get an exact count
    (e.g. llama.cpp's n_tokens) should override it.
    """
    t0 = time.perf_counter()
    ttft = None
    text = ""
    n = 0
    for p in pieces:
        if p:
            if ttft is None:
                ttft = time.perf_counter() - t0
            text += p
            n += 1
    gen_s = time.perf_counter() - t0
    return text, (ttft if ttft is not None else gen_s), gen_s, n


def decode_tps(tokens_out: int, ttft_s: float, gen_s: float):
    """Decode throughput = (tokens_out - 1) / (gen_s - ttft_s). None if undefined
    (the first token is attributed to prefill, so it is excluded)."""
    decode_s = gen_s - ttft_s
    if tokens_out > 1 and decode_s > 0:
        return round((tokens_out - 1) / decode_s, 2)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LEADER side (reply-count sweep)
# ══════════════════════════════════════════════════════════════════════════════

def _leader_aggregate(samples: list) -> list:
    """Per-config (per reply count) avg/std of the stage latencies plus the TTFT /
    token-count / decode-tps metrics for both LLM stages. Stats are over samples
    that produced a result; accuracy is over all samples of the config."""
    metrics = [
        "query_to_tool_ms", "dispatch_ttft_ms",
        "dispatch_tokens_in", "dispatch_tokens_out", "dispatch_tps",
        "tool_to_result_ms",
        "result_to_reply_ms", "answer_ttft_ms",
        "answer_tokens_in", "answer_tokens_out", "answer_tps",
        "end_to_end_ms",
    ]
    out = []
    for r in REPLY_COUNTS:
        rows = [s for s in samples if s["n_replies"] == r and not s["error"]]
        ok = [s for s in rows if s["got_result"] == 1]
        agg = {
            "n_replies": r,
            "n_samples": len(rows),
            "n_correct": sum(s["correct"] for s in rows),
            "accuracy": round(sum(s["correct"] for s in rows) / len(rows), 3) if rows else None,
            "n_got_result": len(ok),
        }
        base = ok or rows
        for m in metrics:
            st = _stats([s[m] for s in base], 2)
            for k, v in st.items():
                agg[f"{m}_{k}"] = v
        out.append(agg)
    return out


def run_reply_benchmark(*, device, preset_label, dispatch_stream, answer_stream,
                        reply_text=DEFAULT_REPLY_TEXT, out_dir=None) -> dict:
    """Drive every (reply-count config x query), STREAMING both LLM stages so TTFT
    + decode tk/s are measured alongside the stage latencies, and save.

    The tool is ALWAYS exactly one (detect_people). What changes per config is how
    many replies the tool returns: after the dispatch picks the tool, the result
    is built in-process as `reply_count` agent replies (build_replies), so the
    answer stage is fed a growing multi-agent result set with NO network hop.

    dispatch_stream(query, tool_defs) -> dict with:
        fn           picked tool name, or None when the model answered in prose
        args         parsed arguments (dict) or None
        reply_text   the prose reply when fn is None (else "")
        ttft_s, gen_s, tokens_in, tokens_out, decode_tps   dispatch metrics
        prompt       the exact dispatch prompt (for prompt_specs)
    answer_stream(query, fn, args, replies, tool_defs) -> dict with:
        text                                              the final reply
        ttft_s, gen_s, tokens_in, tokens_out, decode_tps  answer metrics
        prompt                                            the exact answer prompt

    The JSON is rewritten after EVERY query so nothing is lost on a crash.
    """
    out = _results_dir(out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"leader_{device}_{stamp}.json"

    tool_defs = [DETECT_PEOPLE_DEF]     # the tool is ALWAYS exactly one

    result = {
        "role": "leader",
        "device": device,
        "leader_preset": preset_label,
        "correct_tool": CORRECT_TOOL,
        "reply_counts": list(REPLY_COUNTS),
        "queries": list(PERSON_QUERIES),
        "reply_text": reply_text,
        "prompt_specs": [],
        "samples": [],
        "aggregate": [],
    }

    for r in REPLY_COUNTS:
        log.info("===== config: tool returns %d repl%s =====",
                 r, "y" if r == 1 else "ies")

        # Per-config record of exactly what the LLM was given. The tool defs and
        # the dispatch prompt are identical across configs (the tool never
        # changes); the answer prompt GROWS with the reply count, which is the
        # whole point of this test.
        spec = {
            "n_replies": r,
            "tools": tool_defs,
            "dispatch_prompt": None,   # tool-call stage prompt (system + 1 tool + query)
            "answer_prompt": None,     # tool-reply stage prompt (system + tool + r replies)
        }
        result["prompt_specs"].append(spec)

        for qi, query in enumerate(PERSON_QUERIES):
            sample = {
                "n_replies": r, "query_idx": qi, "query": query,
                "tool_called": "", "tool_result": "", "tool_result_raw": [],
                "final_reply": "",
                "query_to_tool_ms": None, "dispatch_ttft_ms": None,
                "dispatch_tokens_in": None, "dispatch_tokens_out": None,
                "dispatch_tps": None,
                "tool_to_result_ms": None,
                "result_to_reply_ms": None, "answer_ttft_ms": None,
                "answer_tokens_in": None, "answer_tokens_out": None,
                "answer_tps": None,
                "end_to_end_ms": None,
                "correct": 0, "got_result": 0, "error": "",
            }
            try:
                t0 = time.perf_counter()
                d = dispatch_stream(query, tool_defs)
                t1 = time.perf_counter()

                if spec["dispatch_prompt"] is None:
                    spec["dispatch_prompt"] = d.get("prompt")

                fn, args = d.get("fn"), d.get("args")
                # The tool "runs" in-process: build r agent replies (no network).
                replies = build_replies(r, reply_text) if fn else []
                t2 = time.perf_counter()

                a = None
                if replies:
                    a = answer_stream(query, fn, args, replies, tool_defs)
                    final = a.get("text", "")
                elif fn is None:
                    final = d.get("reply_text", "")
                else:
                    final = f"No result built for '{fn}'"
                t3 = time.perf_counter()

                if a is not None and spec["answer_prompt"] is None:
                    spec["answer_prompt"] = a.get("prompt")

                sample.update({
                    "tool_called": fn or "",
                    "tool_result": "\n".join(str(rp.get("text", "")) for rp in replies),
                    "tool_result_raw": replies,
                    "final_reply": final,
                    "query_to_tool_ms": round((t1 - t0) * 1000, 2),
                    "dispatch_ttft_ms": round(d.get("ttft_s", 0.0) * 1000, 2),
                    "dispatch_tokens_in": d.get("tokens_in"),
                    "dispatch_tokens_out": d.get("tokens_out"),
                    "dispatch_tps": d.get("decode_tps"),
                    "tool_to_result_ms": round((t2 - t1) * 1000, 2),
                    "result_to_reply_ms": round((t3 - t2) * 1000, 2),
                    "answer_ttft_ms": round(a.get("ttft_s", 0.0) * 1000, 2) if a else None,
                    "answer_tokens_in": a.get("tokens_in") if a else None,
                    "answer_tokens_out": a.get("tokens_out") if a else None,
                    "answer_tps": a.get("decode_tps") if a else None,
                    "end_to_end_ms": round((t3 - t0) * 1000, 2),
                    "correct": int(fn == CORRECT_TOOL),
                    "got_result": int(bool(replies)),
                })
                log.info("r=%-2d q=%d picked=%-12s ok=%d  "
                         "D[in=%s out=%s ttft=%.0f qt=%.0f tps=%s]  tr=%.0f  "
                         "A[in=%s out=%s ttft=%s rr=%.0f tps=%s]  e2e=%.0f ms",
                         r, qi, sample["tool_called"] or "-", sample["correct"],
                         sample["dispatch_tokens_in"], sample["dispatch_tokens_out"],
                         sample["dispatch_ttft_ms"], sample["query_to_tool_ms"],
                         sample["dispatch_tps"], sample["tool_to_result_ms"],
                         sample["answer_tokens_in"], sample["answer_tokens_out"],
                         sample["answer_ttft_ms"], sample["result_to_reply_ms"],
                         sample["answer_tps"], sample["end_to_end_ms"])
            except Exception as e:
                sample["error"] = repr(e)
                log.exception("query failed (r=%d q=%d)", r, qi)

            result["samples"].append(sample)
            _save_json(path, result)  # save after every query

    result["aggregate"] = _leader_aggregate(result["samples"])
    _save_json(path, result)
    log.info("reply benchmark complete -> %s", path)
    return result
