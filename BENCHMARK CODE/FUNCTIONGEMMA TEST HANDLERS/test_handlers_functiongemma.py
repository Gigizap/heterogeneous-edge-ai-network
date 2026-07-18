#!/usr/bin/env python3
"""
fg_compare.py — Compare FunctionGemma using three handlers on N sentences
                with the FULL available tool set. Saves per-sentence outputs.

Just edit the CONFIG paths below and click Run.

Handlers compared (each must register under a DISTINCT chat_format name to
avoid collisions):
  functiongemma_handler.py        -> "functiongemma"
  functiongemma_simple_handler.py -> "functiongemma_simple"
  functiongemma_cache_handler.py  -> "functiongemma_cache"
"""

from __future__ import annotations
import json, re, time, gc
from pathlib import Path
from llama_cpp import Llama

# ═══════════════════════════════════════════════════════════════
#  CONFIG — EDIT THESE THREE PATHS, THEN JUST CLICK RUN
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = "models/functiongemma-270m-it-Q4_K_M.gguf"
DATASET    = "dataset.json"
TOOLS      = "tools.json"

LIMIT      = 500
OUTPUT_DIR = "fg_compare_out"
# ═══════════════════════════════════════════════════════════════

SEED = 7
P_DISPATCH_FG = "You are a model that can do function calling with the following functions"

# ───────────────────────────────────────────────
# Loading
# ───────────────────────────────────────────────
def load_tools(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name = {t["function"]["name"]: t for t in raw}
    return raw, by_name


def load_dataset(path, limit=100):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(raw):
        if isinstance(r, dict):
            prompt = r["query"]
            expected = {"name": r["tool"], "arguments": r.get("parameters", {})} \
                       if r.get("tool") is not None else None
        else:
            prompt = r[0]
            expected = r[1]
        out.append({"id": i, "prompt": prompt, "expected": expected})
    return out[:limit]


# ───────────────────────────────────────────────
# Model loading — one per handler
# ───────────────────────────────────────────────
def load_llm(path, handler):
    """handler key -> imported module + chat_format name it registers under"""
    if handler == "functiongemma":
        import functiongemma_handler  # noqa
        chat_format = "functiongemma"
    elif handler == "functiongemma_simple":
        import functiongemma_simple_handler  # noqa
        chat_format = "functiongemma_simple"   # must match the @register name
    elif handler == "functiongemma_cache":
        import functiongemma_cache_handler  # noqa
        chat_format = "functiongemma_cache"    # must match the @register name
    else:
        chat_format = handler
    return Llama(model_path=path, n_ctx=4096,
                 chat_format=chat_format, n_gpu_layers=-1, verbose=False)


# ───────────────────────────────────────────────
# Dispatch — returns (name, args, raw_text, gen_tokens)
# ───────────────────────────────────────────────
def _lc(tdefs):
    return [{"type": "function", "function": {
        "name": t.get("function", t)["name"],
        "description": t.get("function", t).get("description", ""),
        "parameters": t.get("function", t).get("parameters", {})}} for t in tdefs]


def dispatch(llm, text, tdefs):
    tools = _lc(tdefs)
    msgs = [{"role": "developer", "content": P_DISPATCH_FG},
            {"role": "user", "content": text}]
    try:
        r = llm.create_chat_completion(messages=msgs, tools=tools,
                                       temperature=0.0, seed=SEED, repeat_penalty=1.1)
    except Exception as e:
        return None, None, f"[ERROR] {e}", 0

    msg = r["choices"][0]["message"]
    raw = msg.get("content", "") or ""
    # Faithful count of what the model actually generated. Per llama.cpp's
    # _create_completion this is len(completion_tokens): every sampled token is
    # counted (incl. trimmed stop-string tokens, incl. runaway repeats), except
    # the terminal EOS which breaks the loop before being appended.
    gen_tokens = (r.get("usage") or {}).get("completion_tokens", 0)

    if msg.get("tool_calls"):
        c = msg["tool_calls"][0]
        try:    args = json.loads(c["function"]["arguments"])
        except: args = {}
        return c["function"]["name"], args, raw, gen_tokens

    m = re.search(r"[◁<]\s*(\{.*?\})\s*[▷>]", raw, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(1))
            if o.get("name"): return o["name"], o.get("arguments") or {}, raw, gen_tokens
        except: pass

    m = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            if o.get("name"): return o["name"], o.get("arguments") or {}, raw, gen_tokens
        except: pass

    for n in [t.get("function", t)["name"] for t in tdefs]:
        if n in raw: return n, {}, raw, gen_tokens
    return None, None, raw, gen_tokens


# ───────────────────────────────────────────────
# Param matching
# ───────────────────────────────────────────────
def pmatch(exp, pred):
    exp, pred = exp or {}, pred or {}
    if exp.keys() != pred.keys():
        return False
    def norm(v):
        try: return float(v)
        except (ValueError, TypeError): pass
        return v.strip().lower() if isinstance(v, str) else v
    return all(norm(exp[k]) == norm(pred[k]) for k in exp)


# ───────────────────────────────────────────────
# Eval — returns (summary, per_sentence_list)
# ───────────────────────────────────────────────
def evaluate(handler, data, all_tools, model_path):
    print(f"\n{'='*55}\n  Handler: {handler}\n{'='*55}")
    llm = load_llm(model_path, handler)

    n_all = len(data)
    n_tool = sum(1 for r in data if r["expected"])
    sel_all = tc_all = 0
    sel_tool = tc_tool = 0
    per_sentence = []

    total_gen_tokens = 0
    first_sentence_seconds = None  # cold-start: first dispatch (incl. any one-time
                                   # grammar compile for the cached/grammar handlers)
    t0 = time.time()

    for idx, row in enumerate(data):
        t_call = time.time()
        pred_name, pred_args, raw, gen_tokens = dispatch(llm, row["prompt"], all_tools)
        call_dt = time.time() - t_call
        if idx == 0:
            first_sentence_seconds = call_dt
        total_gen_tokens += gen_tokens

        gt = row["expected"]
        gt_name = gt["name"] if gt else None
        gt_args = gt.get("arguments") if gt else None

        match_name = (pred_name == gt_name)
        match_args = match_name and pmatch(gt_args, pred_args)

        if match_name:
            sel_all += 1
            if gt_name is not None: sel_tool += 1
        if match_args:
            tc_all += 1
            if gt_name is not None: tc_tool += 1

        per_sentence.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "expected_name": gt_name,
            "expected_args": gt_args,
            "predicted_name": pred_name,
            "predicted_args": pred_args,
            "raw_output": raw,
            "name_match": match_name,
            "args_match": match_args,
            "seconds": round(call_dt, 4),
            "gen_tokens": gen_tokens,
        })

    dt = time.time() - t0
    # Mean over sentences after the first, to isolate the warm per-call cost
    # from the cold first call.
    rest_seconds = [p["seconds"] for p in per_sentence[1:]]
    mean_rest = (sum(rest_seconds) / len(rest_seconds)) if rest_seconds else 0.0

    summary = {
        "handler": handler,
        "n_all": n_all,
        "n_with_tool": n_tool,
        "selection_all": sel_all / n_all,
        "toolcall_all": tc_all / n_all,
        "selection_with_tool_only": sel_tool / n_tool if n_tool else 0,
        "toolcall_with_tool_only": tc_tool / n_tool if n_tool else 0,
        "seconds": round(dt, 1),
        "first_sentence_seconds": round(first_sentence_seconds, 4)
                                  if first_sentence_seconds is not None else None,
        "mean_seconds_after_first": round(mean_rest, 4),
        "total_gen_tokens": total_gen_tokens,
        "mean_gen_tokens": round(total_gen_tokens / n_all, 2) if n_all else 0,
    }
    print(f"  sel(all)={summary['selection_all']:.4f} tc(all)={summary['toolcall_all']:.4f} | "
          f"sel(tool)={summary['selection_with_tool_only']:.4f} "
          f"tc(tool)={summary['toolcall_with_tool_only']:.4f}  {dt:.1f}s")
    print(f"  first sentence={summary['first_sentence_seconds']:.4f}s | "
          f"mean after first={summary['mean_seconds_after_first']:.4f}s")
    print(f"  gen tokens: total={summary['total_gen_tokens']} "
          f"mean={summary['mean_gen_tokens']}")

    del llm
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except ImportError:
        pass
    return summary, per_sentence


def main():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)

    data = load_dataset(DATASET, LIMIT)
    all_tools, _ = load_tools(TOOLS)
    print(f"Loaded {len(data)} sentences, {len(all_tools)} tools (full set)")
    print(f"Model: {MODEL_PATH}")

    summaries = {}
    handler_map = {
        "functiongemma":        "outputs_handler.json",
        "functiongemma_simple": "outputs_simple_handler.json",
        "functiongemma_cache":  "outputs_cache_handler.json",
    }

    for handler, fname in handler_map.items():
        summary, per_sentence = evaluate(handler, data, all_tools, MODEL_PATH)
        summaries[handler] = summary
        (out / fname).write_text(
            json.dumps(per_sentence, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  per-sentence outputs → {out / fname}")

    (out / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\n✓ Summary → {out / 'summary.json'}")
    print(f"✓ Done — {out}")


if __name__ == "__main__":
    main()