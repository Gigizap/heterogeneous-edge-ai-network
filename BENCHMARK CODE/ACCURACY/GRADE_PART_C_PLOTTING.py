#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]

def _color(i):
    return PALETTE[i % len(PALETTE)]

def load_grades(csv_path: Path):
    data = defaultdict(lambda: defaultdict(list))
    skipped = 0
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            g = str(row.get("grade", "")).strip()
            try:
                n = int(row["n_replies"])
            except (ValueError, KeyError):
                continue
            config = row.get("config", "unknown")
            if g.startswith("ID_") or g == "":
                skipped += 1
                continue
            try:
                gi = int(g)
            except ValueError:
                skipped += 1
                continue
            data[config][n].append(gi)
    return data, skipped

def aggregate(grades, how: str):
    if how == "median":
        return statistics.median(grades)
    return sum(grades) / len(grades)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grading", default="results/grading",
                    help="dir containing grades.csv")
    ap.add_argument("--csv", default=None, help="override path to grades.csv")
    ap.add_argument("--output", default=None,
                    help="output dir (default <grading>/plots)")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    ap.add_argument("--no-band", action="store_true",
                    help="don't draw the ±1 std shaded band")
    a = ap.parse_args()

    gdir = Path(a.grading)
    csv_path = Path(a.csv) if a.csv else gdir / "grades.csv"
    if not csv_path.exists():
        print(f"grades.csv not found: {csv_path}")
        return
    out = Path(a.output) if a.output else gdir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    data, skipped = load_grades(csv_path)
    if not data:
        print("No numeric grades found - annotate the pending rows first.")
        return
    if skipped:
        print(f"[note] skipped {skipped} un-annotated/invalid row(s) "
              "(still ID_* placeholders).")

    configs = sorted(data.keys())
    agg_rows = []
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, config in enumerate(configs):
        ns = sorted(data[config].keys())
        ys, los, his = [], [], []
        for n in ns:
            gs = data[config][n]
            mid = aggregate(gs, a.agg)
            sd = statistics.pstdev(gs) if len(gs) > 1 else 0.0
            ys.append(mid)
            los.append(mid - sd)
            his.append(mid + sd)
            agg_rows.append({
                "config": config, "n_replies": n,
                "n_samples": len(gs),
                a.agg: round(mid, 4),
                "std": round(sd, 4),
                "min": min(gs), "max": max(gs),
            })
        col = _color(i)
        ax.plot(ns, ys, "-o", lw=2, color=col, label=config)
        if not a.no_band and len(ns) > 0:
            ax.fill_between(ns, los, his, color=col, alpha=0.12)
        for n, y in zip(ns, ys):
            cnt = len(data[config][n])
            ax.annotate(f"n={cnt}", (n, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=col)

    ax.set_xlabel("Number of function replies shown (N)")
    ax.set_ylabel(f"{a.agg.capitalize()} judge grade (1-10)")
    ax.set_title("Part C - LLM-judge grade vs number of function replies")
    all_ns = sorted({n for c in data.values() for n in c})
    ax.set_xticks(all_ns)
    ax.set_ylim(0.5, 10.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, title="Configuration")
    fig.tight_layout()

    png = out / "grade_vs_replies.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Wrote {png}")

    agg_csv = out / "grade_vs_replies.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "config", "n_replies", "n_samples", a.agg, "std", "min", "max"])
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Wrote {agg_csv}")

if __name__ == "__main__":
    main()
