#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

def load_csv(path: Path):
    with path.open(encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        rows = list(r)
        fields = r.fieldnames
    return rows, fields

def save_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def load_pending(path: Path):
    items = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items

def save_annotation(annotations_dir: Path, item: dict, grade: str):
    annotations_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "row_id": item.get("row_id"),
        "config": item.get("config"),
        "prompt": item.get("prompt"),
        "user_query": item.get("user", ""),
        "tool_call": item.get("tool_call", {}),
        "function_reply": item.get("function_reply", []),
        "model_final_reply": item.get("model_final_reply", ""),
        "judge_raw_answer": item.get("judge_raw_answer", ""),
        "grade": int(grade),
    }
    raw_id = str(item.get("row_id", "unknown"))
    safe_id = re.sub(r'[<>:"/\\|?*]', "_", raw_id)
    safe_id = re.sub(r"[\x00-\x1f]", "_", safe_id).rstrip(". ") or "unknown"
    out_path = annotations_dir / f"{safe_id}.json"
    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path

def show(item: dict, idx: int, total: int):
    bar = "=" * 70
    print("\n" + bar)
    print(f"[{idx}/{total}]  row_id = {item['row_id']}")
    print(f"config={item['config']}  prompt={item['prompt']}  "
          f"N={item['n_replies']}  (placeholder {item['placeholder']})")
    print(bar)
    print("\nUSER QUERY:")
    print("  " + str(item.get("user", "")))
    print("\nTOOL CALL:")
    print("  " + json.dumps(item.get("tool_call", {}), ensure_ascii=False))
    print("\nFUNCTION REPLY (data returned to the model):")
    print(json.dumps(item.get("function_reply", []), ensure_ascii=False, indent=2))
    print("\nMODEL FINAL REPLY (what you are grading):")
    print("  " + str(item.get("model_final_reply", "")))
    print("\nJUDGE RAW ANSWER (could not be parsed):")
    raw = item.get("judge_raw_answer", "")
    print("  " + (raw if raw else "<empty>"))
    if item.get("judge_error"):
        print(f"\n[judge error: {item['judge_error']}]")
    print(bar)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grading", default="results/grading",
                    help="dir containing grades.csv and pending_annotation.jsonl")
    ap.add_argument("--csv", default=None, help="override path to grades.csv")
    ap.add_argument("--pending", default=None,
                    help="override path to pending_annotation.jsonl")
    ap.add_argument("--annotations", default=None,
                    help="override path to the annotations/ output dir "
                         "(default: <grading>/annotations)")
    a = ap.parse_args()

    gdir = Path(a.grading)
    csv_path = Path(a.csv) if a.csv else gdir / "grades.csv"
    pending_path = Path(a.pending) if a.pending else gdir / "pending_annotation.jsonl"
    annotations_dir = Path(a.annotations) if a.annotations else gdir / "annotations"

    if not csv_path.exists():
        print(f"grades.csv not found: {csv_path}")
        return
    pending = load_pending(pending_path)
    if not pending:
        print(f"No pending items in {pending_path} - nothing to annotate.")
        return

    rows, fields = load_csv(csv_path)
    by_id = {r["row_id"]: r for r in rows}
    pending_by_id = {it["row_id"]: it for it in pending}

    todo = []
    for it in pending:
        row = by_id.get(it["row_id"])
        if row is None:
            print(f"  [warn] {it['row_id']} not in grades.csv - skipping")
            continue
        if str(row["grade"]).startswith("ID_"):
            todo.append(it)

    if not todo:
        print("All previously-pending rows are already graded. Nothing to do.")
        return

    print(f"{len(todo)} item(s) need a grade. "
          "Enter 1-10, 's' to skip, 'q' to save & quit.")

    changed = 0
    for i, it in enumerate(todo, 1):
        show(it, i, len(todo))
        while True:
            ans = input("Grade (1-10 / s / q): ").strip().lower()
            if ans == "q":
                print("Saving and quitting.")
                save_csv(csv_path, rows, fields)
                print(f"Wrote {csv_path}  ({changed} new grade(s) this session)")
                return
            if ans == "s":
                print("Skipped.")
                break
            if ans.isdigit() and 1 <= int(ans) <= 10:
                by_id[it["row_id"]]["grade"] = ans
                changed += 1
                out_path = save_annotation(annotations_dir, it, ans)
                print(f"  -> set grade {ans} for {it['row_id']}")
                print(f"     annotation saved to {out_path}")
                break
            print("  Please enter an integer 1-10, or 's' / 'q'.")

    save_csv(csv_path, rows, fields)
    remaining = sum(1 for r in rows if str(r["grade"]).startswith("ID_"))
    print(f"\nWrote {csv_path}  ({changed} new grade(s) this session)")
    print(f"Annotations saved in {annotations_dir}")
    if remaining:
        print(f"{remaining} placeholder(s) still un-graded "
              " - run again to finish them.")
    else:
        print("All records now have a numeric grade.")

if __name__ == "__main__":
    main()
