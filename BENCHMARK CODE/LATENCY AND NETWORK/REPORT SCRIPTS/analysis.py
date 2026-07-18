"""Latency analysis for the tool-scaling benchmark.

Reads the three "VARYING TOOLS 1 reply" leader logs plus the boot-time fields,
computes every statistic used in latency.tex, writes them to
scripts/computed_stats.json, and renders the TTFT-vs-tokens figure to
figures/ttft_vs_tokens.png.

Run from the WRITING_REPORT folder:  python scripts/analysis.py
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "..", "new_results")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Tool-scaling logs (VARYING TOOLS 1 reply/).
FILES = {
    "qwen3_noKV": "VARYING TOOLS 1 reply/leader_raspberry_20260707_170137_withtkincorrect.json",
    "fgemma_noKV": "VARYING TOOLS 1 reply/leader_stm32_prompt_noKV.json",
    "fgemma_KV": "VARYING TOOLS 1 reply/leader_stm32_shorter_prompt+KVcache-newpertoolbatch.json",
}

# Reply-scaling logs (VARYING REPLIES 1 tool/): one tool, growing number of
# aggregated results. reset = no KV reuse, kvreuse = KV reuse (analogous to the
# no-KV / KV tool-scaling pair).
REPLY_FILES = {
    "qwen3": "VARYING REPLIES 1 tool/leader_raspberry_definitive.json",
    "fgemma_reset": "VARYING REPLIES 1 tool/leader_stm32-reset_20260707_150325.json",
    "fgemma_kvreuse": "VARYING REPLIES 1 tool/leader_stm32-kvreuse_20260707_154629.json",
}

MODEL_LABEL = {
    "qwen3_noKV": "Qwen3:1.7b (Hailo), full prompt, no KV reuse",
    "fgemma_noKV": "functiongemma:270m (STM32 CPU), full prompt, no KV reuse",
    "fgemma_KV": "functiongemma:270m (STM32 CPU), short prompt, KV reuse",
}


def load(rel):
    with open(os.path.join(BASE, rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


def ms(mean_std, n=2):
    return None if mean_std is None else round(mean_std, n)


def mstd(vals):
    """Return (mean, sample-std, n). std is 0.0 for a single value."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def by_tools(samples):
    g = {}
    for s in samples:
        g.setdefault(s["n_tools"], []).append(s)
    return {k: g[k] for k in sorted(g)}


out = {"boot": {}, "tools": {}, "kv": {}}

# ---------------------------------------------------------------- boot times
# Recompute from the raw samples rather than trusting the stored mean/std/min/max,
# which can go stale if a sample is edited by hand. The maximum is the cold start,
# the minimum a warm load.
for tag, rel in FILES.items():
    d = load(rel)
    ttb = d.get("time_to_boot") or {}
    raw = [v for v in ttb.get("samples", []) if v is not None]
    if raw:
        m, sd, n = mstd(raw)
        out["boot"][tag] = {
            "unit": ttb.get("unit"),
            "mean": round(m, 2), "std": round(sd, 2),
            "min": round(min(raw), 2), "max": round(max(raw), 2), "count": n,
        }

# ------------------------------------------------ per-tool-count statistics
TOOL_FILES = ["qwen3_noKV", "fgemma_noKV", "fgemma_KV"]
for tag in TOOL_FILES:
    d = load(FILES[tag])
    groups = by_tools(d["samples"])
    rows = {}
    for nt, rs in groups.items():
        got = [r for r in rs if r["got_result"]]
        d_ttft_m, d_ttft_s, _ = mstd([r["dispatch_ttft_ms"] for r in rs])
        d_tin_m, _, _ = mstd([r["dispatch_tokens_in"] for r in rs])
        # stage columns are all averaged over the completed queries so that
        # query->tool + tool->result + result->reply == E2E exactly.
        q2t_m, q2t_s, _ = mstd([r["query_to_tool_ms"] for r in got])
        a_ttft_m, a_ttft_s, _ = mstd([r["answer_ttft_ms"] for r in got])
        a_tin_m, _, _ = mstd([r["answer_tokens_in"] for r in got])
        r2r_m, r2r_s, _ = mstd([r["result_to_reply_ms"] for r in got])
        t2r_m, t2r_s, _ = mstd([r["tool_to_result_ms"] for r in got])
        e2e_m, e2e_s, e2e_n = mstd([r["end_to_end_ms"] for r in got])
        rows[nt] = {
            "n": len(rs), "n_got_result": len(got),
            "dispatch_ttft_ms": [ms(d_ttft_m), ms(d_ttft_s)],
            "dispatch_tokens_in": ms(d_tin_m, 0),
            "query_to_tool_ms": [ms(q2t_m), ms(q2t_s)],
            "answer_ttft_ms": [ms(a_ttft_m), ms(a_ttft_s)],
            "answer_tokens_in": ms(a_tin_m, 0),
            "tool_to_result_ms": [ms(t2r_m), ms(t2r_s)],
            "result_to_reply_ms": [ms(r2r_m), ms(r2r_s)],
            "e2e_ms": [ms(e2e_m), ms(e2e_s)], "e2e_n": e2e_n,
        }
    out["tools"][tag] = rows

# ------------------------------------- KV cache: cold first vs warm cached
d = load(FILES["fgemma_KV"])
groups = by_tools(d["samples"])
kv = {}
for nt, rs in groups.items():
    rs_sorted = sorted(rs, key=lambda r: r["query_idx"])
    cold = rs_sorted[0]
    warm = rs_sorted[1:]
    w_ttft_m, w_ttft_s, w_n = mstd([r["dispatch_ttft_ms"] for r in warm])
    # steady-state (warm) stage breakdown excludes the one cold prefill per pool
    got_warm = [r for r in warm if r["got_result"]]
    wtin_m, _, _ = mstd([r["dispatch_tokens_in"] for r in warm])
    wq2t_m, wq2t_s, _ = mstd([r["query_to_tool_ms"] for r in got_warm])
    wt2r_m, wt2r_s, _ = mstd([r["tool_to_result_ms"] for r in got_warm])
    wr2r_m, wr2r_s, _ = mstd([r["result_to_reply_ms"] for r in got_warm])
    we2e_m, we2e_s, we2e_n = mstd([r["end_to_end_ms"] for r in got_warm])
    kv[nt] = {
        "cold_dispatch_ttft_ms": round(cold["dispatch_ttft_ms"], 2),
        "cold_dispatch_tokens_in": cold["dispatch_tokens_in"],
        "warm_dispatch_ttft_ms": [ms(w_ttft_m), ms(w_ttft_s)], "warm_n": w_n,
        "warm_tokens_in": ms(wtin_m, 0),
        "warm_query_to_tool_ms": [ms(wq2t_m), ms(wq2t_s)],
        "warm_tool_to_result_ms": [ms(wt2r_m), ms(wt2r_s)],
        "warm_result_to_reply_ms": [ms(wr2r_m), ms(wr2r_s)],
        "warm_e2e_ms": [ms(we2e_m), ms(we2e_s)], "warm_e2e_n": we2e_n,
    }
out["kv"] = kv

# --------------------------------------------- replies-scaling statistics
# One tool is exposed and a growing number of sensing agents return a result
# (1, 5, 10, 15, 20). The dispatch stage is invariant (single tool); the answer
# stage grows because the aggregation prompt lengthens with the number of results.
out["replies"] = {}
for cfg, rel in REPLY_FILES.items():
    drep = load(rel)
    rgroups = {}
    for s in drep["samples"]:
        rgroups.setdefault(s["n_replies"], []).append(s)
    rep = {}
    for nr in sorted(rgroups):
        rs = rgroups[nr]
        got = [r for r in rs if r["got_result"]]
        d_ttft_m, d_ttft_s, _ = mstd([r["dispatch_ttft_ms"] for r in rs])
        q2t_m, q2t_s, _ = mstd([r["query_to_tool_ms"] for r in got])
        t2r_m, t2r_s, _ = mstd([r["tool_to_result_ms"] for r in got])
        a_ttft_m, a_ttft_s, _ = mstd([r["answer_ttft_ms"] for r in got])
        a_tin_m, _, _ = mstd([r["answer_tokens_in"] for r in got])
        a_tout_m, _, _ = mstd([r["answer_tokens_out"] for r in got])
        r2r_m, r2r_s, _ = mstd([r["result_to_reply_ms"] for r in got])
        e2e_m, e2e_s, e2e_n = mstd([r["end_to_end_ms"] for r in got])
        rep[nr] = {
            "n": len(rs), "n_got_result": len(got),
            "dispatch_ttft_ms": [ms(d_ttft_m), ms(d_ttft_s)],
            "query_to_tool_ms": [ms(q2t_m), ms(q2t_s)],
            "tool_to_result_ms": [ms(t2r_m), ms(t2r_s)],
            "answer_ttft_ms": [ms(a_ttft_m), ms(a_ttft_s)],
            "answer_tokens_in": ms(a_tin_m, 0),
            "answer_tokens_out": ms(a_tout_m, 1),
            "result_to_reply_ms": [ms(r2r_m), ms(r2r_s)],
            "e2e_ms": [ms(e2e_m), ms(e2e_s)], "e2e_n": e2e_n,
        }
    out["replies"][cfg] = rep

with open(os.path.join(HERE, "computed_stats.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)

# --------------------------------------------------- pretty console report
def sfmt(pair):
    if pair[0] is None:
        return "   -   "
    return "%8.1f +/- %6.1f" % (pair[0], pair[1])

print("\n================ BOOT TIME (ms) ================")
for tag, b in out["boot"].items():
    print("  %-12s mean=%8.1f std=%8.1f min=%8.1f max=%8.1f (n=%d)"
          % (tag, b["mean"], b["std"], b["min"], b["max"], b["count"]))

for tag in TOOL_FILES:
    print("\n================ %s ================" % tag)
    print("  nt | n gr | disp_ttft(ms)        tin | q2tool(ms)           | "
          "ans_ttft(ms)         tin | res2reply(ms)        | e2e(ms)              n")
    for nt, r in out["tools"][tag].items():
        print("  %2d | %d %2d | %s %4s | %s | %s %4s | %s | %s %d"
              % (nt, r["n"], r["n_got_result"],
                 sfmt(r["dispatch_ttft_ms"]), r["dispatch_tokens_in"],
                 sfmt(r["query_to_tool_ms"]),
                 sfmt(r["answer_ttft_ms"]), r["answer_tokens_in"],
                 sfmt(r["result_to_reply_ms"]),
                 sfmt(r["e2e_ms"]), r["e2e_n"]))

print("\n================ KV CACHE (functiongemma) ================")
print("  nt | cold_ttft(ms) tin | warm_ttft(ms)        n | warm_e2e(ms)         n")
for nt, r in out["kv"].items():
    print("  %2d | %10.1f %4d | %s %2d | %s %2d"
          % (nt, r["cold_dispatch_ttft_ms"], r["cold_dispatch_tokens_in"],
             sfmt(r["warm_dispatch_ttft_ms"]), r["warm_n"],
             sfmt(r["warm_e2e_ms"]), r["warm_e2e_n"]))

for cfg, rep in out["replies"].items():
    print("\n================ REPLIES: %s ================" % cfg)
    print("  nrep | disp_ttft(ms)        | ans_ttft(ms)        | ans_tin | ans_tok_out | e2e(ms)")
    for nr, r in rep.items():
        print("  %4d | %s | %s | %5s | %8s | %s"
              % (nr, sfmt(r["dispatch_ttft_ms"]), sfmt(r["answer_ttft_ms"]),
                 r["answer_tokens_in"], r["answer_tokens_out"], sfmt(r["e2e_ms"])))

# --------------------------------------------------------------- TTFT plot
# Two panels (one per device/model) because the STM32 CPU and the Hailo operate
# on very different time scales. Each panel shows per-sample TTFT against prompt
# length (tokens in) for the tool-dispatch stage and for the final-reply stage,
# so the prefill-dominated growth is visible. The STM32 panel also overlays the
# short-prompt + KV-cache configuration (average over the cached queries) to show
# how much of that growth the cache removes.
import numpy as np

COL_DISP = "#2f6db5"   # tool-dispatch stage (full prompt, no KV)
COL_ANS = "#e08214"    # final-reply stage
COL_KV = "#2ca25f"     # tool-dispatch stage (short prompt, KV cached)


def stage_points(tag):
    d = load(FILES[tag])
    dx, dy, ax_x, ax_y = [], [], [], []
    for s in d["samples"]:
        if s["dispatch_ttft_ms"] is not None and s["dispatch_tokens_in"] is not None:
            dx.append(s["dispatch_tokens_in"])
            dy.append(s["dispatch_ttft_ms"] / 1000.0)
        if s["got_result"] and s["answer_ttft_ms"] is not None and s["answer_tokens_in"] is not None:
            ax_x.append(s["answer_tokens_in"])
            ax_y.append(s["answer_ttft_ms"] / 1000.0)
    return dx, dy, ax_x, ax_y


def kv_points():
    """Per pool return the cold first-load point (query_idx 0) and the warm
    average point (mean/std over the cached queries, query_idx >= 1)."""
    d = load(FILES["fgemma_KV"])
    cold_x, cold_y = [], []
    warm_x, warm_y, warm_e = [], [], []
    for nt, rs in by_tools(d["samples"]).items():
        rs_sorted = sorted(rs, key=lambda r: r["query_idx"])
        cold, warm = rs_sorted[0], rs_sorted[1:]
        cold_x.append(cold["dispatch_tokens_in"])
        cold_y.append(cold["dispatch_ttft_ms"] / 1000.0)
        tin = [r["dispatch_tokens_in"] for r in warm if r["dispatch_tokens_in"] is not None]
        ttft = [r["dispatch_ttft_ms"] / 1000.0 for r in warm if r["dispatch_ttft_ms"] is not None]
        warm_x.append(st.mean(tin))
        warm_y.append(st.mean(ttft))
        warm_e.append(st.stdev(ttft) if len(ttft) > 1 else 0.0)
    return (cold_x, cold_y), (warm_x, warm_y, warm_e)


def fit_line(ax, dx, dy, color):
    if len(set(dx)) > 1:
        slope, intercept = np.polyfit(dx, dy, 1)
        xs = np.array([min(dx), max(dx)])
        ax.plot(xs, slope * xs + intercept, color=color, linestyle="--",
                linewidth=1.2, alpha=0.7, zorder=2,
                label="linear fit (%.1f ms/token)" % (slope * 1000.0))


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

# ---- left panel: functiongemma, no-KV full prompt vs short-prompt KV cache ----
axf = axes[0]
dx, dy, ax_x, ax_y = stage_points("fgemma_noKV")
axf.scatter(dx, dy, s=34, color=COL_DISP, edgecolor="white", linewidth=0.5,
            label="tool-dispatch, full prompt, no KV", zorder=3)
fit_line(axf, dx, dy, COL_DISP)
axf.scatter(ax_x, ax_y, s=34, color=COL_ANS, edgecolor="white", linewidth=0.5,
            marker="s", label="final-reply, full prompt, no KV", zorder=3)
(cx, cy), (kx, ky, ke) = kv_points()
# cold first-load of each pool: hollow triangles, sit on the no-KV dispatch curve
axf.scatter(cx, cy, s=52, facecolor="none", edgecolor=COL_KV, linewidth=1.6,
            marker="^", zorder=4,
            label="tool-dispatch, short prompt, KV first load (cold)")
# warm average: filled triangles with std error bars, only over the cached queries
axf.errorbar(kx, ky, yerr=ke, fmt="^", ms=8, color=COL_KV, mec="white", mew=0.5,
             capsize=3, linestyle="none", zorder=5,
             label="tool-dispatch, short prompt, KV cached avg (warm only)")
axf.set_title("functiongemma:270m on STM32MP257F-DK (CPU)", fontsize=10)
axf.set_xlabel("prompt length (tokens in)")
axf.set_ylabel("time to first token (s)")
axf.grid(True, alpha=0.3)
axf.legend(fontsize=7.5, loc="upper left")

# ---- right panel: qwen3 ----
axq = axes[1]
dx, dy, ax_x, ax_y = stage_points("qwen3_noKV")
axq.scatter(dx, dy, s=34, color=COL_DISP, edgecolor="white", linewidth=0.5,
            label="tool-dispatch prefill", zorder=3)
fit_line(axq, dx, dy, COL_DISP)
axq.scatter(ax_x, ax_y, s=34, color=COL_ANS, edgecolor="white", linewidth=0.5,
            marker="s", label="final-reply prefill", zorder=3)
axq.set_xlim(left=100)
axq.set_title("Qwen3:1.7b on Raspberry Pi 5 (Hailo)", fontsize=10)
axq.set_xlabel("prompt length (tokens in)")
axq.set_ylabel("time to first token (s)")
axq.grid(True, alpha=0.3)
axq.legend(fontsize=8, loc="upper left")

fig.tight_layout()
outpng = os.path.join(FIGDIR, "ttft_vs_tokens.png")
fig.savefig(outpng, dpi=150)
print("\nWrote figure:", os.path.relpath(outpng, HERE))

# ------------------------------------------------------ replies-scaling plot
# Two panels, mirroring the tool-scaling figure. The answer-stage TTFT (prefill of
# the aggregation prompt) is plotted against the answer prompt length (tokens in).
# STM32 panel compares KV reset vs KV reuse; the Hailo panel shows Qwen3. The
# number of aggregated replies is annotated on each point.
def rep_series(cfg):
    d = load(REPLY_FILES[cfg])
    g = {}
    for s in d["samples"]:
        if s["got_result"] and s["answer_ttft_ms"] is not None and s["answer_tokens_in"] is not None:
            g.setdefault(s["n_replies"], {"x": [], "y": []})
            g[s["n_replies"]]["x"].append(s["answer_tokens_in"])
            g[s["n_replies"]]["y"].append(s["answer_ttft_ms"] / 1000.0)
    nrs = sorted(g)
    xs = [st.mean(g[n]["x"]) for n in nrs]
    ys = [st.mean(g[n]["y"]) for n in nrs]
    return nrs, xs, ys


figr, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.5))

# ---- left: functiongemma on STM32, reset vs kvreuse ----
axl = axes2[0]
for cfg, color, marker, lab in [
        ("fgemma_reset", COL_DISP, "o", "no KV reuse (reset)"),
        ("fgemma_kvreuse", COL_KV, "^", "KV reuse")]:
    nrs, xs, ys = rep_series(cfg)
    axl.scatter(xs, ys, s=40, color=color, edgecolor="white", linewidth=0.5,
                marker=marker, zorder=3, label="answer TTFT, " + lab)
    sl, ic = np.polyfit(xs, ys, 1)
    xf = np.array([min(xs), max(xs)])
    axl.plot(xf, sl * xf + ic, color=color, linestyle="--", linewidth=1.2, alpha=0.7,
             zorder=2, label="fit (%.0f ms/token)" % (sl * 1000.0))
    for n, x, y in zip(nrs, xs, ys):
        axl.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, -9),
                     fontsize=7, color=color)
axl.set_title("functiongemma:270m on STM32MP257F-DK (CPU), one tool", fontsize=10)
axl.set_xlabel("answer prompt length (tokens in)")
axl.set_ylabel("answer-stage TTFT (s)")
axl.grid(True, alpha=0.3)
axl.legend(fontsize=7.5, loc="upper left")

# ---- right: qwen3 on Hailo ----
axrp = axes2[1]
nrs, xs, ys = rep_series("qwen3")
axrp.scatter(xs, ys, s=40, color=COL_ANS, edgecolor="white", linewidth=0.5,
             marker="s", zorder=3, label="answer TTFT")
sl, ic = np.polyfit(xs, ys, 1)
xf = np.array([min(xs), max(xs)])
axrp.plot(xf, sl * xf + ic, color=COL_ANS, linestyle="--", linewidth=1.2, alpha=0.7,
          zorder=2, label="fit (%.1f ms/token)" % (sl * 1000.0))
for n, x, y in zip(nrs, xs, ys):
    axrp.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, -9),
                  fontsize=7, color=COL_ANS)
axrp.set_title("Qwen3:1.7b on Raspberry Pi 5 (Hailo), one tool", fontsize=10)
axrp.set_xlabel("answer prompt length (tokens in)")
axrp.set_ylabel("answer-stage TTFT (s)")
axrp.grid(True, alpha=0.3)
axrp.legend(fontsize=8, loc="upper left")

figr.tight_layout()
outpng_r = os.path.join(FIGDIR, "latency_vs_replies.png")
figr.savefig(outpng_r, dpi=150)
print("Wrote figure:", os.path.relpath(outpng_r, HERE))
print("Wrote stats :", os.path.relpath(os.path.join(HERE, 'computed_stats.json'), HERE))
