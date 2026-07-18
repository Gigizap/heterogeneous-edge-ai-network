#!/usr/bin/env python3
"""
grade_partc.py — LLM-as-judge grading of Part C summaries.

Walks every partC_<config>__<prompt>__N<n>.json produced by benchmark.py and,
for each record, asks an OpenAI-compatible chat endpoint to grade how well
"model_final_reply" answers the user query given the tool call and the data the
tool returned. The judge is instructed to reply with ONLY {"grade": <1-10>}.

For each record we extract the integer grade from the model's answer:
  * If a valid grade JSON is found  -> the grade goes into the CSV.
  * If NOT (no parseable JSON)       -> the CSV gets a placeholder "ID_<k>"
                                        and the full datapoint + the judge's raw
                                        answer are appended to a pending file so
                                        you can annotate them by hand later.

Outputs (into --output):
  grades.csv                 one row per record (see columns below)
  raw_judge_answers.jsonl    raw model output for EVERY record, keyed by row_id
  pending_annotation.jsonl   un-parseable cases, keyed by the SAME row id

CSV columns:
  row_id, config, prompt, n_replies, source_file, record_index, grade
  - grade is an int 1-10 when parsed, otherwise the placeholder ID_<k>.
  - row_id is a stable, unique key: <config>|<prompt>|N<n>|<record_index>

Usage:
  export OPENAI_API_KEY=sk-...
  python grade_partc.py --results results/ \
      --base-url https://api.openai.com/v1 --model gpt-4o-mini
  # local / vLLM / Ollama / LM Studio:
  python grade_partc.py --results results/ \
      --base-url http://localhost:8000/v1 --model my-model --api-key dummy
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import requests
except ImportError:
    print("This script needs `requests`  ->  pip install requests")
    sys.exit(1)


SYSTEM_PROMPT = (
    "\nYou are an expert evaluator. Given a user query, the function/tool call "
    "that was made, and the data the function returned, grade from 1 to 10 how "
    'well "model_final_reply" answers the query. Functions may have been '
    "executed across different devices/sources, and the task varies by query. "
    "Judge the reply on: - Relevance: does it directly address what the user "
    "asked? - Faithfulness: is it grounded in the returned data, with no "
    "invented or contradicted facts? - Use of evidence: does it sensibly "
    "weigh/prioritize the data (e.g. confidences, conflicting sources) rather "
    "than dumping everything indiscriminately? - Clarity: is it a coherent, "
    "useful summary? Scoring guide: 1-3: ignores the query, misreads the data, "
    "or hallucinates. 4-6: partially addresses it; weak prioritization or "
    "notable omissions. 7-8: addresses the query and is grounded, minor gaps. "
    "9-10: directly and accurately answers, reasons well over the data, "
    'concise. Respond with ONLY valid JSON and nothing else: {"grade": '
    '<integer 1-10>}. Always state the grade in json.\n'
)

# partC_<config>__<prompt>__N<n>.json   (config/prompt may contain underscores)
FILE_RE = re.compile(r"^partC_(.+?)__(.+?)__N(\d+)\.json$")


# ---------------------------------------------------------------------------
# Grade extraction from a model answer
# ---------------------------------------------------------------------------
def extract_grade(text: str) -> Optional[int]:
    """Return an int grade 1-10 if a valid grade can be parsed, else None.

    Strategy (most-strict first):
      1. Whole answer parses as JSON with an int-ish 'grade' in range.
      2. A ```json ...``` / ``` ...``` fenced block parses likewise.
      3. Any {...} substring containing "grade" parses likewise.
      4. A loose  "grade": N  regex within range.
    """
    if not text:
        return None

    def _valid(obj) -> Optional[int]:
        if isinstance(obj, dict) and "grade" in obj:
            try:
                g = int(obj["grade"])
            except (ValueError, TypeError):
                return None
            if 1 <= g <= 10:
                return g
        return None

    # 1) whole thing
    try:
        g = _valid(json.loads(text.strip()))
        if g is not None:
            return g
    except Exception:
        pass

    # 2) fenced code blocks
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        try:
            g = _valid(json.loads(m.group(1).strip()))
            if g is not None:
                return g
        except Exception:
            continue

    # 3) any object literal mentioning "grade"
    for m in re.finditer(r"\{[^{}]*\"grade\"[^{}]*\}", text, re.DOTALL):
        try:
            g = _valid(json.loads(m.group(0)))
            if g is not None:
                return g
        except Exception:
            continue

    # 4) loose key:value
    m = re.search(r"\"?grade\"?\s*[:=]\s*(\d{1,2})", text, re.IGNORECASE)
    if m:
        try:
            g = int(m.group(1))
            if 1 <= g <= 10:
                return g
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------
def build_user_content(rec: dict) -> str:
    """Send the judge exactly the fields it grades, as pretty JSON."""
    payload = {
        "user": rec.get("user", ""),
        "tool_call": rec.get("tool_call", {}),
        "function_reply": rec.get("function_reply", []),
        "model_final_reply": rec.get("model_final_reply", ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_judge(base_url: str, api_key: str, model: str, user_content: str,
               temperature: float, timeout: int, max_retries: int) -> Tuple[Optional[str], Optional[str]]:
    """Return (answer_text, error). On success error is None."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "reasoning_effort": "none",
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"].get("content", "")
                return msg or "", None
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries:
            time.sleep(min(2 ** attempt, 15))
    return None, last_err


# ---------------------------------------------------------------------------
# Collect records
# ---------------------------------------------------------------------------
def collect_records(results: Path):
    """Yield (row_id, config, prompt, n, source_file, record_index, rec)."""
    files = sorted(results.glob("partC_*.json"))
    for f in files:
        m = FILE_RE.match(f.name)
        if not m:
            print(f"  [skip] filename not recognised: {f.name}")
            continue
        config, prompt, n = m.group(1), m.group(2), int(m.group(3))
        try:
            recs = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] could not parse {f.name}: {e}")
            continue
        if not isinstance(recs, list):
            continue
        for i, rec in enumerate(recs):
            row_id = f"{config}|{prompt}|N{n}|{i}"
            yield row_id, config, prompt, n, f.name, i, rec


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results",
                    help="dir containing partC_*.json")
    ap.add_argument("--output", default=None,
                    help="output dir (default <results>/grading)")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL",
                                                          "http://localhost:11434/v1"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "glm-4.7-flash"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to sleep between calls (rate limiting)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only grade the first N records (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't call the LLM; mark everything un-evaluated")
    a = ap.parse_args()

    results = Path(a.results)
    if not results.exists():
        print(f"Results dir not found: {results}")
        sys.exit(1)
    out = Path(a.output) if a.output else results / "grading"
    out.mkdir(parents=True, exist_ok=True)

    if not a.api_key and not a.dry_run:
        print("[warn] no API key set (use --api-key or OPENAI_API_KEY). "
              "Continuing in case the endpoint needs none.")

    csv_path = out / "grades.csv"
    pending_path = out / "pending_annotation.jsonl"
    raw_path = out / "raw_judge_answers.jsonl"

    records = list(collect_records(results))
    if a.limit:
        records = records[: a.limit]
    if not records:
        print("No Part C records found — nothing to grade.")
        return
    print(f"Found {len(records)} records to grade across "
          f"{len(set(r[1] for r in records))} configuration(s).\n")

    rows = []
    pending = []
    raw_log = []          # raw judge output for EVERY record
    pending_counter = 0

    for k, (row_id, config, prompt, n, src, idx, rec) in enumerate(records, 1):
        if a.dry_run:
            answer, err = None, "dry-run"
        else:
            answer, err = call_judge(
                a.base_url, a.api_key, a.model,
                build_user_content(rec),
                a.temperature, a.timeout, a.max_retries)

        grade = extract_grade(answer or "") if answer is not None else None

        if grade is not None:
            grade_cell = str(grade)
            status = f"grade={grade}"
        else:
            pending_counter += 1
            grade_cell = f"ID_{pending_counter}"
            status = "UN-PARSED -> " + grade_cell
            pending.append({
                "row_id": row_id,
                "placeholder": grade_cell,
                "config": config,
                "prompt": prompt,
                "n_replies": n,
                "source_file": src,
                "record_index": idx,
                "user": rec.get("user", ""),
                "tool_call": rec.get("tool_call", {}),
                "function_reply": rec.get("function_reply", []),
                "model_final_reply": rec.get("model_final_reply", ""),
                "judge_raw_answer": answer if answer is not None else "",
                "judge_error": err or "",
            })

        rows.append({
            "row_id": row_id,
            "config": config,
            "prompt": prompt,
            "n_replies": n,
            "source_file": src,
            "record_index": idx,
            "grade": grade_cell,
        })

        # raw judge output for EVERY datapoint (parsed or not)
        raw_log.append({
            "row_id": row_id,
            "config": config,
            "prompt": prompt,
            "n_replies": n,
            "source_file": src,
            "record_index": idx,
            "grade": grade if grade is not None else None,
            "grade_cell": grade_cell,
            "judge_raw_answer": answer if answer is not None else "",
            "judge_error": err or "",
        })

        print(f"  [{k}/{len(records)}] {row_id}  {status}"
              + (f"  ({err})" if err and err != 'dry-run' else ""))
        if a.sleep:
            time.sleep(a.sleep)

    # write CSV
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "row_id", "config", "prompt", "n_replies",
            "source_file", "record_index", "grade"])
        w.writeheader()
        w.writerows(rows)

    # write pending (only un-parsed)
    with pending_path.open("w", encoding="utf-8") as fh:
        for p in pending:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # write raw judge output for EVERY record
    with raw_path.open("w", encoding="utf-8") as fh:
        for r in raw_log:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in rows if not r["grade"].startswith("ID_"))
    print(f"\nWrote {csv_path}  ({n_ok}/{len(rows)} auto-graded)")
    print(f"Wrote {raw_path}  ({len(raw_log)} raw judge answers)")
    print(f"Wrote {pending_path}  ({len(pending)} need manual annotation)")
    if pending:
        print("Run annotate_pending.py next to fill in the ID_* placeholders.")


if __name__ == "__main__":
    main()