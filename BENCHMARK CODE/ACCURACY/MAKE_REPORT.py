#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, csv, re, statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RERANKER_LABEL = "Jina-Reranker"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]

def _load_json(path: Path):
    if not path.exists():
        print(f"  [warn] missing {path.name} - related outputs will be skipped")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] could not parse {path.name}: {e}")
        return None

def _sizes(res, key="sizes"):
    return [int(x) for x in res.get(key, [])]

def _series(d, sizes):
    return [d.get(str(s)) for s in sizes]

def load_grades(csv_path: Path):
    data = defaultdict(lambda: defaultdict(list))
    skipped = 0
    if not csv_path or not csv_path.exists():
        return data, skipped
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

def parse_c_tokens(token_summary):
    out = {}
    for lb, d in (token_summary or {}).items():
        per_prompt = {}
        for key, val in d.items():
            m = re.match(r"(.+)/N(\d+)$", key)
            if not m:
                continue
            per_prompt.setdefault(m.group(1), {})[int(m.group(2))] = val
        out[lb] = per_prompt
    return out

def collapse_c_tokens(parsed):
    flat = {}
    for lb, per_prompt in parsed.items():
        if not per_prompt:
            continue
        prompt = sorted(per_prompt)[0]
        flat[lb] = per_prompt[prompt]
    return flat

def build_color_map(res_a, res_b, c_tokens_flat, grades):
    labels = []

    def add(name):
        if name and name != RERANKER_LABEL and name not in labels:
            labels.append(name)

    if res_a:
        for k in ("selection_with_tool_only", "mean_gen_tokens"):
            for lb in (res_a.get(k) or {}):
                add(lb)
    if res_b:
        for lb in (res_b.get("llms") or {}):
            add(lb)
    for lb in (c_tokens_flat or {}):
        add(lb)
    for lb in (grades or {}):
        add(lb)

    cmap = {lb: _PALETTE[i % len(_PALETTE)] for i, lb in enumerate(labels)}
    cmap[RERANKER_LABEL] = "#000000"
    return cmap

def _f(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"

def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

def fig_a_accuracy(res_a, cmap, out: Path):
    if not res_a or not res_a.get("selection_with_tool_only"):
        print("  [skip] Part A accuracy - no data"); return
    sizes = _sizes(res_a)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lb, d in res_a["selection_with_tool_only"].items():
        col = cmap.get(lb)
        ax.plot(sizes, _series(d, sizes), "-o", lw=2, color=col, label=lb)
        td = (res_a.get("toolcall_with_tool_only") or {}).get(lb)
        if td:
            ax.plot(sizes, _series(td, sizes), "--s", lw=1.5, color=col, alpha=0.65)
    ax.set_xlabel("Available tools (pool size)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Part A - Tool-only accuracy (solid = selection, dashed = +params)")
    ax.set_xticks(sizes); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, out / "partA_accuracy.png")

def fig_a_tokens(res_a, cmap, out: Path):
    if not res_a or not res_a.get("mean_gen_tokens"):
        print("  [skip] Part A tokens - no data"); return
    sizes = _sizes(res_a)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lb, d in res_a["mean_gen_tokens"].items():
        ax.plot(sizes, _series(d, sizes), "-o", lw=2, color=cmap.get(lb), label=lb)
    ax.set_xlabel("Available tools (pool size)")
    ax.set_ylabel("Mean generated tokens / question")
    ax.set_title("Part A - Generation cost vs pool size")
    ax.set_xticks(sizes); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, out / "partA_tokens.png")

def fig_b_accuracy(res_b, cmap, out: Path):
    if not res_b:
        print("  [skip] Part B accuracy - no data"); return
    sizes = _sizes(res_b)
    fig, ax = plt.subplots(figsize=(8, 5))
    if res_b.get("reranker"):
        ax.plot(sizes, _series(res_b["reranker"], sizes), "--s", lw=2,
                color=cmap[RERANKER_LABEL], label=RERANKER_LABEL)
    for lb, d in (res_b.get("llms") or {}).items():
        ax.plot(sizes, _series(d, sizes), "-o", lw=2, color=cmap.get(lb), label=lb)
    ax.set_xlabel("Available tools (pool size)")
    ax.set_ylabel("Selection accuracy")
    n = res_b.get("n", 0)
    ax.set_title(f"Part B - Selection accuracy: LLMs vs reranker (n={n})")
    ax.set_xticks(sizes); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, out / "partB_accuracy.png")

def fig_b_tokens(res_b, cmap, out: Path):
    if not res_b or not res_b.get("mean_gen_tokens"):
        print("  [skip] Part B tokens - no data"); return
    sizes = _sizes(res_b)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lb, d in res_b["mean_gen_tokens"].items():
        ax.plot(sizes, _series(d, sizes), "-o", lw=2, color=cmap.get(lb), label=lb)
    ax.plot(sizes, [0] * len(sizes), "--s", lw=2,
            color=cmap[RERANKER_LABEL], label=f"{RERANKER_LABEL} (0)")
    ax.set_xlabel("Available tools (pool size)")
    ax.set_ylabel("Mean generated tokens / question")
    ax.set_title("Part B - Generation cost vs pool size")
    ax.set_xticks(sizes); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, out / "partB_tokens.png")

def fig_c_grade(grades, cmap, out: Path, replies_n=None):
    if not grades:
        print("  [skip] Part C grade - no grades.csv data"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    for lb in sorted(grades):
        ns = sorted(grades[lb])
        means = [statistics.mean(grades[lb][n]) for n in ns]
        stds  = [statistics.pstdev(grades[lb][n]) if len(grades[lb][n]) > 1 else 0.0
                 for n in ns]
        meds  = [statistics.median(grades[lb][n]) for n in ns]
        col = cmap.get(lb)
        ax.plot(ns, means, "-o", lw=2, color=col, label=f"{lb} (mean)")
        ax.plot(ns, meds, ":^", lw=1.4, color=col, alpha=0.7)
        ax.fill_between(ns, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=col, alpha=0.12)
    ax.set_xlabel("Number of function replies shown (N)")
    ax.set_ylabel("Judge grade (1-10)")
    ax.set_title("Part C - Answer quality vs N (solid = mean ±1 std, dotted = median)")
    xticks = sorted({n for d in grades.values() for n in d})
    ax.set_xticks(xticks); ax.set_ylim(0.5, 10.5); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, out / "partC_grade.png")

def fig_c_tokens(c_tokens_flat, cmap, out: Path, replies_n=None):
    if not c_tokens_flat:
        print("  [skip] Part C tokens - no data"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    for lb in sorted(c_tokens_flat):
        nmap = c_tokens_flat[lb]
        ns = sorted(nmap)
        ax.plot(ns, [nmap[n] for n in ns], "-o", lw=2, color=cmap.get(lb), label=lb)
    ax.set_xlabel("Number of function replies shown (N)")
    ax.set_ylabel("Mean generated tokens / question")
    ax.set_title("Part C - Generation cost vs N")
    if replies_n:
        ax.set_xticks([int(x) for x in replies_n])
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, out / "partC_tokens.png")

def _save(fig, path: Path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")

def _write_tex(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  saved {path.name}")

def table_a(res_a, out: Path):
    if not res_a or not res_a.get("selection_with_tool_only"):
        print("  [skip] tableA - no data"); return
    sizes = _sizes(res_a)
    smin, smax = sizes[0], sizes[-1]
    sel = res_a["selection_with_tool_only"]
    tc  = res_a.get("toolcall_with_tool_only", {})
    nul = res_a.get("null_tool_only", {})
    tok = res_a.get("mean_gen_tokens", {})

    cols = "l" + "r" * 6
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Part~A: tool-calling on the full 24-tool set. Selection and "
        r"selection+params accuracy on the tool-bearing subset at the smallest "
        rf"and largest pools ($n{{=}}{smin},{smax}$), abstention accuracy on the "
        rf"null-tool subset at $n{{=}}{smax}$, and mean generated tokens per "
        r"question (averaged over all pools).}",
        r"\label{tab:partA}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        rf"Configuration & Sel@{smin} & Sel@{smax} & SelP@{smin} & SelP@{smax} "
        rf"& Null@{smax} & Tok \\",
        r"\midrule",
    ]
    for lb in sel:
        sel_lo = sel[lb].get(str(smin)); sel_hi = sel[lb].get(str(smax))
        tcd = tc.get(lb, {})
        tc_lo = tcd.get(str(smin)); tc_hi = tcd.get(str(smax))
        null_hi = nul.get(lb, {}).get(str(smax))
        tvals = [v for v in (tok.get(lb, {}) or {}).values() if v is not None]
        tok_mean = sum(tvals) / len(tvals) if tvals else None
        lines.append(
            f"{_tex_escape(lb)} & {_f(sel_lo)} & {_f(sel_hi)} & {_f(tc_lo)} & "
            f"{_f(tc_hi)} & {_f(null_hi)} & {_f(tok_mean,1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex(out / "tableA.tex", lines)

def table_b(res_b, out: Path):
    if not res_b:
        print("  [skip] tableB - no data"); return
    sizes = _sizes(res_b)
    cols = "l" + "r" * len(sizes)
    head = " & ".join([f"$n{{=}}{s}$" for s in sizes])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Part~B: selection accuracy on the 16 zero-parameter tools "
        rf"($n{{=}}{res_b.get('n', 0)}$ queries), for each LLM configuration and "
        r"the Jina reranker, at every pool size.}",
        r"\label{tab:partB}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        rf"Method & {head} \\",
        r"\midrule",
    ]
    for lb, d in (res_b.get("llms") or {}).items():
        row = " & ".join(_f(d.get(str(s))) for s in sizes)
        lines.append(f"{_tex_escape(lb)} & {row} \\\\")
    if res_b.get("reranker"):
        lines.append(r"\midrule")
        row = " & ".join(_f(res_b["reranker"].get(str(s))) for s in sizes)
        lines.append(f"{_tex_escape(RERANKER_LABEL)} & {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex(out / "tableB.tex", lines)

def table_c(grades, c_tokens_flat, out: Path):
    if not grades and not c_tokens_flat:
        print("  [skip] tableC - no data"); return
    all_n = sorted({n for d in grades.values() for n in d}
                   | {n for d in c_tokens_flat.values() for n in d})
    all_lb = sorted(set(grades) | set(c_tokens_flat))
    have_grades = bool(grades)

    if have_grades:
        cols = "ll" + "r" * 3
        header = r"Configuration & $N$ & Median & Mean$\pm$Std & Tok \\"
        caption = (r"Part~C: answer-generation quality and cost vs number of "
                   r"function replies $N$. Grade is the LLM-judge score (1--10); "
                   r"tokens are mean generated tokens per question.")
    else:
        cols = "ll" + "r"
        header = r"Configuration & $N$ & Tok \\"
        caption = (r"Part~C: answer-generation cost vs number of function replies "
                   r"$N$ (mean generated tokens per question). Judge grades pending.")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:partC}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for li, lb in enumerate(all_lb):
        for j, n in enumerate(all_n):
            tok = c_tokens_flat.get(lb, {}).get(n)
            label_cell = _tex_escape(lb) if j == 0 else ""
            if have_grades:
                gs = grades.get(lb, {}).get(n, [])
                if gs:
                    med = _f(statistics.median(gs), 1)
                    mean = statistics.mean(gs)
                    std = statistics.pstdev(gs) if len(gs) > 1 else 0.0
                    ms = f"{mean:.2f}$\\pm${std:.2f}"
                else:
                    med, ms = "--", "--"
                lines.append(f"{label_cell} & {n} & {med} & {ms} & {_f(tok,1)} \\\\")
            else:
                lines.append(f"{label_cell} & {n} & {_f(tok,1)} \\\\")
        if li != len(all_lb) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex(out / "tableC.tex", lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results",
                    help="dir with results_a/b/_c_tokens.json")
    ap.add_argument("--grades", default=None,
                    help="path to grades.csv (default <results>/grading/grades.csv)")
    ap.add_argument("--no-grades", action="store_true",
                    help="skip Part C judge grades entirely (grade figure/table "
                         "columns omitted) - use until grades are computed")
    ap.add_argument("--output", default=None,
                    help="output dir (default <results>/report)")
    a = ap.parse_args()

    rdir = Path(a.results)
    out = Path(a.output) if a.output else rdir / "report"
    out.mkdir(parents=True, exist_ok=True)
    grades_csv = Path(a.grades) if a.grades else rdir / "grading" / "grades.csv"

    res_a = _load_json(rdir / "results_a.json")
    res_b = _load_json(rdir / "results_b.json")
    res_c = _load_json(rdir / "results_c_tokens.json")

    c_tokens_flat = collapse_c_tokens(
        parse_c_tokens(res_c.get("mean_gen_tokens") if res_c else None))
    if a.no_grades:
        print("  [note] --no-grades: Part C judge grades skipped")
        grades = {}
    else:
        grades, skipped = load_grades(grades_csv)
        if skipped:
            print(f"  [note] {skipped} un-annotated/invalid grade row(s) skipped")
    replies_n = res_c.get("replies_n") if res_c else None

    cmap = build_color_map(res_a, res_b, c_tokens_flat, grades)

    print("Figures:")
    fig_a_accuracy(res_a, cmap, out)
    fig_a_tokens(res_a, cmap, out)
    fig_b_accuracy(res_b, cmap, out)
    fig_b_tokens(res_b, cmap, out)
    fig_c_grade(grades, cmap, out, replies_n)
    fig_c_tokens(c_tokens_flat, cmap, out, replies_n)

    print("Tables:")
    table_a(res_a, out)
    table_b(res_b, out)
    table_c(grades, c_tokens_flat, out)

    print(f"\nDone - {out}")

if __name__ == "__main__":
    main()
