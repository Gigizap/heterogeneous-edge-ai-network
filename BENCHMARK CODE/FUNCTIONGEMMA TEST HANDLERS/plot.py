#!/usr/bin/env python3
"""
fg_plot.py — Plot handler comparison from the summary.json produced by fg_compare.py.

For each handler it shows:
  * Accuracy (all four metrics computed by fg_compare):
      - selection_all
      - toolcall_all
      - selection_with_tool_only
      - toolcall_with_tool_only
  * Cold-start time (first_sentence_seconds) and total time (seconds)
  * Average tokens generated per sentence (mean_gen_tokens)

Just point SUMMARY at the summary.json and click Run.
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write to file, no display needed
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
SUMMARY = "fg_compare_out/summary.json"
OUTPUT  = "fg_compare_out/comparison.png"
TABLE   = "fg_compare_out/comparison.tex"
# ═══════════════════════════════════════════════════════════════

ACC_METRICS = [
    ("selection_all",            "Selection (all)"),
    ("toolcall_all",             "Tool-call (all)"),
    ("selection_with_tool_only", "Selection (tool only)"),
    ("toolcall_with_tool_only",  "Tool-call (tool only)"),
]


def build_latex_table(summaries):
    """Build a LaTeX table from the per-handler summary dicts.

    Columns (accuracy reported on the tool-bearing subset only):
      Handler  : handler name
      Sel      : selection accuracy on the tool-bearing subset
      TC       : tool-call (selection + params) accuracy on the tool-bearing subset
      Cold(s)  : cold-start latency (first sentence)
      Total(s) : total wall-clock time
      Tok      : mean generated tokens / sentence
    """
    rows = []
    for handler, s in summaries.items():
        name = handler.replace("_", r"\_")
        rows.append(
            f"{name} & {s.get('selection_with_tool_only', 0):.3f} & "
            f"{s.get('toolcall_with_tool_only', 0):.3f} & "
            f"{(s.get('first_sentence_seconds') or 0):.3f} & {(s.get('seconds') or 0):.1f} & "
            f"{s.get('mean_gen_tokens', 0):.1f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{FunctionGemma handler comparison on the full tool set. "
        "Selection (Sel) and tool-call (TC, selection+params) accuracy on the "
        "tool-bearing subset; cold-start latency on the first sentence (Cold), "
        "total wall-clock time (Total), and mean generated tokens per sentence (Tok).}\n"
        "\\label{tab:fg_compare}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Handler & Sel & TC & Cold(s) & Total(s) & Tok \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main():
    summaries = json.loads(Path(SUMMARY).read_text(encoding="utf-8"))
    handlers = list(summaries.keys())
    if not handlers:
        raise SystemExit("No handlers found in summary.json")

    fig, (ax_acc, ax_time, ax_tok) = plt.subplots(
        1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.7, 1, 0.9]}
    )
    fig.suptitle("FunctionGemma handler comparison", fontsize=15, fontweight="bold")

    x = np.arange(len(handlers))

    # ── Left: grouped accuracy bars ──────────────────────────────
    n_metrics = len(ACC_METRICS)
    width = 0.8 / n_metrics

    for j, (key, label) in enumerate(ACC_METRICS):
        vals = [summaries[h].get(key, 0) for h in handlers]
        offset = (j - (n_metrics - 1) / 2) * width
        bars = ax_acc.bar(x + offset, vals, width, label=label)
        for b, v in zip(bars, vals):
            ax_acc.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax_acc.set_title("Accuracy by handler")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(handlers, rotation=15, ha="right")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=8, loc="upper right")
    ax_acc.grid(axis="y", alpha=0.3)

    # ── Middle: cold-start vs total time ─────────────────────────
    cold = [summaries[h].get("first_sentence_seconds") or 0 for h in handlers]
    total = [summaries[h].get("seconds") or 0 for h in handlers]

    width2 = 0.38
    b_cold = ax_time.bar(x - width2 / 2, cold, width2, label="Cold start (1st sentence)", color="#d98c3f")
    b_total = ax_time.bar(x + width2 / 2, total, width2, label="Total time", color="#3f7fd9")
    for bars, vals in ((b_cold, cold), (b_total, total)):
        for b, v in zip(bars, vals):
            ax_time.text(b.get_x() + b.get_width() / 2, v,
                         f"{v:.1f}s", ha="center", va="bottom", fontsize=7)

    ax_time.set_title("Timing by handler")
    ax_time.set_ylabel("Seconds")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(handlers, rotation=15, ha="right")
    ax_time.legend(fontsize=8, loc="upper left")
    ax_time.grid(axis="y", alpha=0.3)

    # ── Right: average tokens generated per sentence ─────────────
    mean_tok = [summaries[h].get("mean_gen_tokens") or 0 for h in handlers]
    b_tok = ax_tok.bar(x, mean_tok, 0.5, color="#5fae6b")
    for b, v in zip(b_tok, mean_tok):
        ax_tok.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax_tok.set_title("Avg tokens generated / sentence")
    ax_tok.set_ylabel("Mean completion_tokens")
    ax_tok.set_xticks(x)
    ax_tok.set_xticklabels(handlers, rotation=15, ha="right")
    ax_tok.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"✓ Plot → {OUTPUT}")

    latex = build_latex_table(summaries)
    Path(TABLE).parent.mkdir(parents=True, exist_ok=True)
    Path(TABLE).write_text(latex, encoding="utf-8")
    print(f"✓ Table → {TABLE}")
    print("\n" + latex)


if __name__ == "__main__":
    main()