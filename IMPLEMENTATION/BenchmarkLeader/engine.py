"""
BenchmarkLeader/engine.py

Shared latency-benchmark engine for the two leader benchmarks.

It measures a leader's FULL network pipeline for every (query x tool_count):

    query received                                          [t0]
      -> fetch_and_merge_skills()  (network round-trip)     [t_fetch]
      -> pad with fake wrong_tool_i to reach n tools
      -> LLM dispatch picks a tool                          [t_dispatch]  ("tool executed")
      -> broadcast the tool call to the other device
      -> sensing result received                            [t_result]
      -> LLM turns the result into a NL reply               [t_answer]

Recorded per sample (ms):
    fetch_ms                  = t_fetch     - t0
    dispatch_ms               = t_dispatch  - t_fetch
    delta1_query_to_tool_ms   = t_dispatch  - t0        (fetch + dispatch)
    delta2_tool_to_result_ms  = t_result    - t_dispatch
    delta3_result_to_reply_ms = t_answer    - t_result
    end_to_end_ms             = t_answer    - t0
plus picked_tool / correct (did it pick `correct_tool`) / got_result.

The "more tools" axis is SIMULATED: fetch returns the real tool(s) from the
network, then wrong_tool_1..wrong_tool_(n-1) (description "do not call this
tool") are appended so the dispatcher sees exactly n tools.

This module is LLM/hardware-agnostic: it only needs a `collector`
(NetworkCollector) and a `workflow` exposing `_dispatch(query, tool_defs)` and
`_answer(query, command, replies)`. The two bench_*.py scripts supply those.
"""

import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Callable, List, Optional

from LeaderLogic.network_collector import tool_call_to_payload

WRONG_TOOL_DESC = "do not call this tool"
TOOL_COUNTS     = (1, 5, 10, 15, 20)

# Qwen3 soft switches appended to the query to force reasoning on/off.
_THINK_SUFFIX = {"think": " /think", "no_think": " /no_think"}

_SAMPLE_FIELDS = [
    "think_mode", "n_tools", "query_idx", "query", "picked_tool", "correct",
    "thinking_detected", "got_result", "n_real_tools", "fetch_ms", "dispatch_ms",
    "delta1_query_to_tool_ms", "delta2_tool_to_result_ms",
    "delta3_result_to_reply_ms", "end_to_end_ms", "error",
]


def has_thinking(raw: str) -> bool:
    """True if the raw model output contains a NON-EMPTY <think> … </think> block
    (Qwen3 emits <think>, not <thinking>; an empty block means thinking was off)."""
    if not raw:
        return False
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        return bool(m.group(1).strip())
    m = re.search(r"<think>(.*)$", raw, re.DOTALL)   # opened but truncated/unclosed
    return bool(m and m.group(1).strip())
_METRICS = [
    "fetch_ms", "dispatch_ms", "delta1_query_to_tool_ms",
    "delta2_tool_to_result_ms", "delta3_result_to_reply_ms", "end_to_end_ms",
]


# ── tool padding ────────────────────────────────────────────────────────────────

def _wrong_tool(i: int) -> dict:
    return {"type": "function", "function": {
        "name": f"wrong_tool_{i}",
        "description": WRONG_TOOL_DESC,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }}


def pad_tools(real_tools: List[dict], n: int) -> List[dict]:
    """real_tools + wrong_tool_i until the list has exactly n entries.
    If there are already >= n real tools, all real tools are kept (can't go below)."""
    tools = list(real_tools)
    i = 1
    while len(tools) < n:
        tools.append(_wrong_tool(i))
        i += 1
    return tools


# ── network setup helpers (used by the bench scripts) ───────────────────────────

def setup_network(agent_id: str, port: int):
    """Start a leader-side transport + discovery on a background event loop.
    Returns (loop, transport, discovery). The benchmark talks to whatever sensing
    agent is already on the network (run `python main.py` on the other device)."""
    from ConnectionLogic.transport import P2PTransport
    from ConnectionLogic.discovery import Discovery
    from utils import start_async_loop, schedule

    loop = start_async_loop()
    transport = P2PTransport(agent_id, port, lambda msg: None)
    transport._event_loop = loop
    discovery = Discovery(
        agent_id, port,
        on_peer_found=lambda pid, ip, p: schedule(loop, transport.register_peer(pid, ip, p)),
        on_peer_lost=lambda pid:         schedule(loop, transport.unregister_peer(pid)),
    )
    schedule(loop, transport.start())
    schedule(loop, discovery.start())
    return loop, transport, discovery


def wait_for_peer(transport, timeout: float = 30.0) -> bool:
    """Block until at least one peer (the sensing agent) is registered."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if getattr(transport, "_peers", {}):
            return True
        time.sleep(0.5)
    return bool(getattr(transport, "_peers", {}))


# ── the benchmark ───────────────────────────────────────────────────────────────

def run_benchmark(
    *,
    name: str,
    collector,
    workflow,
    correct_tool: str,
    queries: List[str],
    tool_counts=TOOL_COUNTS,
    out_dir: Optional[Path] = None,
    reset: Optional[Callable[[], None]] = None,
    think_mode: Optional[str] = None,
    get_last_raw: Optional[Callable[[], str]] = None,
) -> dict:
    """
    Run every (query x tool_count), collect per-sample latencies, write a samples
    CSV + an aggregate (CSV + JSON), and return the aggregate dict.

    think_mode    : None | "think" | "no_think" — appends the Qwen3 /think or
                    /no_think soft switch to each query.
    get_last_raw  : optional () -> str returning the model's last raw dispatch
                    text, so each sample records whether it actually reasoned.
    """
    out_dir = Path(out_dir) if out_dir else (Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = _THINK_SUFFIX.get(think_mode, "")

    samples: List[dict] = []
    total = len(tool_counts) * len(queries)
    done = 0

    for n in tool_counts:
        for qi, q in enumerate(queries):
            done += 1
            if reset:
                try:
                    reset()
                except Exception:
                    pass

            row = {k: "" for k in _SAMPLE_FIELDS}
            row.update({"think_mode": think_mode or "default", "n_tools": n,
                        "query_idx": qi, "query": q})
            try:
                dispatch_query = q + suffix
                t0 = time.perf_counter()
                real = collector.fetch_and_merge_skills() or []
                t_fetch = time.perf_counter()

                tools = pad_tools(real, n)
                fn, args = workflow._dispatch(dispatch_query, tools)
                t_dispatch = time.perf_counter()
                thinking = has_thinking(get_last_raw()) if get_last_raw else None

                picked = fn or ""
                if fn:
                    payload = tool_call_to_payload(fn, args or {})
                    replies = collector.broadcast_and_collect(payload) or []
                else:
                    payload, replies = "", []
                t_result = time.perf_counter()

                answer = workflow._answer(dispatch_query, payload, replies) if replies else ""
                t_answer = time.perf_counter()

                row.update({
                    "picked_tool": picked,
                    "correct": int(picked == correct_tool),
                    "thinking_detected": "" if thinking is None else int(thinking),
                    "got_result": int(bool(replies)),
                    "n_real_tools": len(real),
                    "fetch_ms": round((t_fetch - t0) * 1000, 2),
                    "dispatch_ms": round((t_dispatch - t_fetch) * 1000, 2),
                    "delta1_query_to_tool_ms": round((t_dispatch - t0) * 1000, 2),
                    "delta2_tool_to_result_ms": round((t_result - t_dispatch) * 1000, 2),
                    "delta3_result_to_reply_ms": round((t_answer - t_result) * 1000, 2),
                    "end_to_end_ms": round((t_answer - t0) * 1000, 2),
                })
                think_str = "" if thinking is None else f" think={int(thinking)}"
                print(f"[bench {name}] {done}/{total}  n={n:<2} q={qi}  "
                      f"picked={picked or '-'} correct={row['correct']}{think_str} "
                      f"e2e={row['end_to_end_ms']}ms")
            except Exception as e:
                row["error"] = repr(e)
                print(f"[bench {name}] {done}/{total}  n={n} q={qi}  ERROR: {e!r}")

            samples.append(row)

    aggregate = _aggregate(samples, tool_counts)
    _save(out_dir, name, stamp, samples, aggregate)
    return aggregate


# ── aggregation + saving ────────────────────────────────────────────────────────

def _stats(values: List[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean":   round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "std":    round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
        "min":    round(min(values), 2),
        "max":    round(max(values), 2),
    }


def _aggregate(samples: List[dict], tool_counts) -> List[dict]:
    """Aggregate per n_tools. Latency stats are over successful samples
    (got_result == 1) so timeouts from a wrong dispatch don't skew them;
    accuracy is over all samples."""
    out = []
    for n in tool_counts:
        rows = [s for s in samples if s["n_tools"] == n and not s["error"]]
        ok   = [s for s in rows if s.get("got_result") == 1]
        measured = [s for s in rows if s.get("thinking_detected") in (0, 1)]
        agg = {
            "n_tools": n,
            "n_samples": len(rows),
            "n_correct": sum(s.get("correct", 0) for s in rows),
            "accuracy": round(sum(s.get("correct", 0) for s in rows) / len(rows), 3) if rows else None,
            "n_got_result": len(ok),
            "thinking_rate": round(sum(s["thinking_detected"] for s in measured) / len(measured), 3) if measured else None,
        }
        base = ok or rows  # fall back to all rows if nothing produced a result
        for m in _METRICS:
            st = _stats([s[m] for s in base if isinstance(s.get(m), (int, float))])
            for k, v in st.items():
                agg[f"{m}_{k}"] = v
        out.append(agg)
    return out


def _save(out_dir: Path, name: str, stamp: str, samples: List[dict], aggregate: List[dict]):
    samples_path = out_dir / f"{name}_{stamp}_samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SAMPLE_FIELDS)
        w.writeheader()
        w.writerows(samples)

    agg_fields = ["n_tools", "n_samples", "n_correct", "accuracy", "n_got_result", "thinking_rate"]
    for m in _METRICS:
        agg_fields += [f"{m}_mean", f"{m}_median", f"{m}_std", f"{m}_min", f"{m}_max"]
    agg_csv = out_dir / f"{name}_{stamp}_aggregate.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(aggregate)

    agg_json = out_dir / f"{name}_{stamp}_aggregate.json"
    agg_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(f"\n[bench {name}] saved:\n  {samples_path}\n  {agg_csv}\n  {agg_json}")
