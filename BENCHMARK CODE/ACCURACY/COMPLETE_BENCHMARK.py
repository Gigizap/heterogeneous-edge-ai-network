#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re, sys, time, gc
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
from llama_cpp import Llama
from transformers import AutoModelForSequenceClassification

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import functiongemma_simple_handler
    HAS_FG = True
except ImportError:
    HAS_FG = False

try:
    import qwen3_handler
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False

print("functiongemma has handler", HAS_FG)
print("qwen3 has handler", HAS_QWEN3)

SEED = 7

LIMIT = False

POOL_A = [4, 8, 12, 16, 20, 24]
POOL_B = [4, 8, 12, 16]

P_DISPATCH = (
    "You are a smart-home agent dispatcher. "
    "Call exactly ONE tool that best matches the user's intent. "
    "Extract any names or object targets precisely from the user's message. "
    "If nothing matches, reply normally without calling any tool."
)
P_DISPATCH_FG = (
    "You are a model that can do function calling with the following functions"
)

NO_THINK = " /no_think"

REPLIES_N = [1, 5, 10, 15, 20]

P_C_SUMMARY = (
    "You are a smart home assistant. Turn the data into a short friendly reply "
    "to the user. Reply in English. Summarize the reply, MENTION ALL THE IMPORTANT "
    "DATA RECEIVED FROM THE TOOL AND THE AVAILABLE SENSORS WHEN REPLYING."
)
P_C_PROMPTS = {"summary_prompt": P_C_SUMMARY}

def load_tools(path: str):
    raw_tools = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name = {t["function"]["name"]: t for t in raw_tools}

    no_param = sorted([
        t["function"]["name"] for t in raw_tools
        if not t["function"].get("parameters", {}).get("properties")
    ])

    return raw_tools, by_name, no_param

def load_dataset(path: str) -> List[Dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(raw):
        if isinstance(r, dict):
            prompt = r["query"]
            if r.get("tool") is not None:
                expected = {"name": r["tool"], "arguments": r.get("parameters", {})}
            else:
                expected = None
            replies = r.get("replies", [])
        else:
            prompt = r[0]
            expected = r[1]
            replies = r[2] if len(r) > 2 else []

        out.append({"id": i, "prompt": prompt, "expected": expected, "replies": replies})
    if LIMIT:
        out = out[:LIMIT]
    return out

def make_pools(universe, data, sizes, seed=SEED):
    rng = random.Random(seed)
    pools = {}
    for n in sizes:
        pools[str(n)] = {}
        for row in data:
            req = row["expected"]["name"] if row["expected"] else None
            if n >= len(universe):
                subset = list(universe)
            elif req is None:
                subset = rng.sample(universe, n)
            else:
                need = next(t for t in universe if t["function"]["name"] == req)
                rest = rng.sample([t for t in universe if t["function"]["name"] != req], n - 1)
                subset = rest + [need]
                rng.shuffle(subset)
            pools[str(n)][str(row["id"])] = [t["function"]["name"] for t in subset]
    return {"seed": seed, "sizes": sizes, "pools": pools}

def write_pools_txt(pools, path: Path):
    lines = [f"# seed={pools['seed']} sizes={','.join(map(str, pools['sizes']))}"]
    for n in pools["sizes"]:
        block = pools["pools"][str(n)]
        for sid in sorted(block, key=lambda x: int(x)):
            lines.append(f"{n}\t{sid}\t{','.join(block[sid])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def read_pools_txt(path: Path):
    pools = {}
    seed = SEED
    sizes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            for tok in line.lstrip("#").split():
                if tok.startswith("seed="):
                    seed = int(tok.split("=", 1)[1])
                elif tok.startswith("sizes="):
                    sizes = [int(x) for x in tok.split("=", 1)[1].split(",") if x]
            continue
        n, sid, names = line.split("\t")
        pools.setdefault(str(int(n)), {})[sid] = names.split(",") if names else []
    if not sizes:
        sizes = sorted(int(k) for k in pools)
    return {"seed": seed, "sizes": sizes, "pools": pools}

def get_or_make_pools(universe, data, sizes, path: Path, label=""):
    if path.exists():
        print(f"  Reusing cached pools{(' ' + label) if label else ''} from {path}")
        return read_pools_txt(path)
    pools = make_pools(universe, data, sizes)
    write_pools_txt(pools, path)
    print(f"  Saved pools{(' ' + label) if label else ''} to {path}")
    return pools

def grade(gt_name, gt_args, pred_name, pred_args):
    if pred_name != gt_name:
        return "incorrect"
    if gt_name is None:
        return "correct_selection"
    if pmatch(gt_args, pred_args):
        return "correct_tool_call"
    return "correct_selection"

def open_model_log(out: Path, label: str, part: str):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    path = out / f"log_{part}_{safe}.tsv"
    f = path.open("w", encoding="utf-8")
    f.write("sentence_id\tn_tools\tgt_name\tgt_args\tpred_name\tpred_args\t"
            "grade\tgen_tokens\traw_text\n")
    return f, path

def _tsv_clean(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")

def log_row(f, sid, n_tools, gt_name, gt_args, pred_name, pred_args, g,
            gen_tokens=0, raw_text=""):
    def js(x):
        return json.dumps(x, ensure_ascii=False) if x is not None else "null"
    f.write(f"{sid}\t{n_tools}\t{gt_name if gt_name is not None else 'null'}\t"
            f"{js(gt_args)}\t{pred_name if pred_name is not None else 'null'}\t"
            f"{js(pred_args)}\t{g}\t{gen_tokens}\t{_tsv_clean(raw_text)}\n")

_PAT = {
    "functiongemma": ("FunctionGemma-270M", 0.27, "fg"),
    "qwen3-1.7b":    ("Qwen3-1.7B",         1.7,  "qwen3"),
}

def find_models(d):
    raw = []
    for g in sorted(d.glob("**/*.gguf")):
        fn = g.name.lower()
        for pat, (lbl, pb, kind) in _PAT.items():
            if pat in fn:
                raw.append({"path": str(g), "label": lbl, "params_b": pb, "kind": kind})
                break
        else:
            raw.append({"path": str(g), "label": g.stem, "params_b": 0.0, "kind": ""})

    out = []
    for mi in raw:
        if mi["kind"] == "qwen3":
            out.append({**mi, "label": f"{mi['label']}-think",
                        "fg": False, "qwen3": True, "think": True})
            out.append({**mi, "label": f"{mi['label']}-nothink",
                        "fg": False, "qwen3": True, "think": False})
        else:
            out.append({**mi, "fg": (mi["kind"] == "fg"),
                        "qwen3": False, "think": False})
    return out

_LLM_CACHE: Dict[str, Llama] = {}

def load_llm(mi):
    path = mi["path"]
    if path in _LLM_CACHE:
        return _LLM_CACHE[path]
    if mi["fg"] and HAS_FG:
        llm = Llama(model_path=path, n_ctx=4096,
                    chat_format="functiongemma", n_gpu_layers=-1, verbose=False)
    elif mi["qwen3"] and HAS_QWEN3:
        llm = Llama(model_path=path, n_ctx=4096,
                    chat_format="qwen3", n_gpu_layers=-1, verbose=False)
    else:
        llm = Llama(model_path=path, n_ctx=4096,
                    n_gpu_layers=-1, verbose=False)
    _LLM_CACHE[path] = llm
    return llm

def free_llm(mi):
    path = mi["path"]
    if path in _LLM_CACHE:
        del _LLM_CACHE[path]
        gc.collect()
        torch.cuda.empty_cache()

def variants_share_path(models, idx):
    p = models[idx]["path"]
    return any(m["path"] == p for m in models[idx + 1:])

def sys_prompt_for(mi, base):
    if mi.get("qwen3") and not mi.get("think"):
        return base + NO_THINK
    return base

def _lc(tdefs):
    return [{"type": "function", "function": {
        "name": t.get("function", t)["name"],
        "description": t.get("function", t).get("description", ""),
        "parameters": t.get("function", t).get("parameters", {})}} for t in tdefs]

def dispatch(llm, text, tdefs, mi):
    fg = mi["fg"]
    tools = _lc(tdefs)
    if fg:
        msgs = [{"role": "developer", "content": P_DISPATCH_FG},
                {"role": "user", "content": text}]
        kw = dict(messages=msgs, tools=tools, temperature=0.0,
                  seed=SEED, repeat_penalty=1.1)
    else:
        msgs = [{"role": "system", "content": sys_prompt_for(mi, P_DISPATCH)},
                {"role": "user", "content": text}]
        kw = dict(messages=msgs, tools=tools, tool_choice="auto",
                  max_tokens=256, temperature=0.0, seed=SEED, repeat_penalty=1.1)
    try:
        r = llm.create_chat_completion(**kw)
    except Exception:
        return None, None, 0, ""

    msg = r["choices"][0]["message"]
    raw = msg.get("content", "") or ""
    raw_text = msg.get("_raw_text", "") or ""
    gen_tokens = (r.get("usage") or {}).get("completion_tokens", 0)

    if msg.get("tool_calls"):
        c = msg["tool_calls"][0]
        try:    args = json.loads(c["function"]["arguments"])
        except: args = {}
        return c["function"]["name"], args, gen_tokens, raw_text

    m = re.search(r"[◁<]\s*(\{.*?\})\s*[▷>]", raw, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(1))
            if o.get("name"): return o["name"], o.get("arguments") or {}, gen_tokens, raw_text
        except: pass

    m = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            if o.get("name"): return o["name"], o.get("arguments") or {}, gen_tokens, raw_text
        except: pass

    pool = [t.get("function", t)["name"] for t in tdefs]
    for n in pool:
        if n in raw: return n, {}, gen_tokens, raw_text

    return None, None, gen_tokens, raw_text

def pmatch(exp, pred):
    exp = exp or {}
    pred = pred or {}

    if exp.keys() != pred.keys():
        return False

    def norm(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
        if isinstance(v, str):
            return v.strip().lower()
        return v

    return all(norm(exp[k]) == norm(pred[k]) for k in exp)

class Reranker:
    def __init__(self):
        print("[reranker] loading …")
        self.m = AutoModelForSequenceClassification.from_pretrained(
            "jinaai/jina-reranker-v2-base-multilingual",
            torch_dtype="auto", trust_remote_code=True)
        self.m.to("cuda").eval()

    def top1(self, q, docs):
        s = self.m.compute_score([[q, d] for d in docs], max_length=1024)
        return 0 if isinstance(s, (int, float)) else int(max(range(len(s)), key=lambda i: s[i]))

    def done(self):
        del self.m; gc.collect(); torch.cuda.empty_cache()

def _doc(t):
    f = t.get("function", t)
    return f"{f['name']}: {f.get('description', '')}"

def run_a(data, models, out, all_tools, by_name):
    print("\n" + "=" * 60 + "\n  PART A - LLM Tool Calling (Full Tool Set)\n" + "=" * 60)

    sub_a = [r for r in data if r["expected"]]
    n_all = len(data)
    n_filt = len(sub_a)
    n_null = n_all - n_filt

    print(f"  Total questions: {n_all} (with tool: {n_filt}, null tool: {n_null})")

    pools = get_or_make_pools(all_tools, data, POOL_A, out / "pools_a.txt", "(Part A)")

    sel_acc_all  = defaultdict(dict)
    tc_acc_all   = defaultdict(dict)
    sel_acc_filt = defaultdict(dict)
    tc_acc_filt  = defaultdict(dict)
    null_acc     = defaultdict(dict)
    tok_mean     = defaultdict(dict)

    for idx, mi in enumerate(models):
        lb = mi["label"]
        print(f"\n  {lb}")
        llm = load_llm(mi)
        logf, logpath = open_model_log(out, lb, "A")

        for ps in POOL_A:
            s_ok_all = t_ok_all = 0
            s_ok_filt = t_ok_filt = 0
            n_ok_null = 0
            tok_sum = 0
            t0 = time.time()

            for row in data:
                qid = row["id"]
                pn = pools["pools"][str(ps)][str(qid)]
                pt = [by_name[n] for n in pn]
                pred_name, pred_args, gen_tokens, raw_text = dispatch(llm, row["prompt"], pt, mi)
                tok_sum += gen_tokens
                gt = row["expected"]
                gt_name = gt["name"] if gt else None
                gt_args = gt.get("arguments") if gt else None

                match_name = (pred_name == gt_name)
                match_args = match_name and pmatch(gt_args, pred_args)

                g = grade(gt_name, gt_args, pred_name, pred_args)
                log_row(logf, qid, ps, gt_name, gt_args, pred_name, pred_args,
                        g, gen_tokens, raw_text)

                if match_name:
                    s_ok_all += 1
                    if gt_name is not None:
                        s_ok_filt += 1
                    else:
                        n_ok_null += 1

                if match_args:
                    t_ok_all += 1
                    if gt_name is not None:
                        t_ok_filt += 1

            sel_acc_all[lb][str(ps)] = s_ok_all / n_all
            tc_acc_all[lb][str(ps)]  = t_ok_all / n_all
            sel_acc_filt[lb][str(ps)] = s_ok_filt / n_filt if n_filt else 0
            tc_acc_filt[lb][str(ps)]  = t_ok_filt / n_filt if n_filt else 0
            null_acc[lb][str(ps)]     = n_ok_null / n_null if n_null else 0
            tok_mean[lb][str(ps)]     = tok_sum / n_all if n_all else 0

            print(f"    pool={ps}: "
                  f"sel(all)={s_ok_all/n_all:.4f} tc(all)={t_ok_all/n_all:.4f} | "
                  f"sel(tool)={s_ok_filt/n_filt:.4f} tc(tool)={t_ok_filt/n_filt:.4f} | "
                  f"null={n_ok_null/n_null if n_null else 0:.4f} "
                  f"tok/q={tok_sum/n_all:.1f} "
                  f"{time.time()-t0:.1f}s")

        logf.close()
        print(f"    log -> {logpath}")
        if not variants_share_path(models, idx):
            free_llm(mi)

    res = {
        "part": "A",
        "n_all": n_all,
        "n_with_tool": n_filt,
        "n_null_tool": n_null,
        "sizes": POOL_A,
        "selection_all": dict(sel_acc_all),
        "toolcall_all": dict(tc_acc_all),
        "selection_with_tool_only": dict(sel_acc_filt),
        "toolcall_with_tool_only": dict(tc_acc_filt),
        "null_tool_only": dict(null_acc),
        "mean_gen_tokens": dict(tok_mean),
    }
    res_path = out / "results_a.json"
    res_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  Saved Part A results to {res_path}")

def run_b(data, models, out, all_tools, by_name, no_param):
    print("\n" + "=" * 60 + "\n  PART B - LLMs vs. Reranker (Zero-Param Tools)\n" + "=" * 60)

    sub = [r for r in data if r["expected"] and r["expected"]["name"] in no_param]
    uni = [t for t in all_tools if t["function"]["name"] in no_param]
    print(f"  {len(sub)} questions, {len(uni)}-tool universe")

    pools = get_or_make_pools(uni, sub, POOL_B, out / "pools_b.txt", "(Part B)")

    rr = Reranker()
    rr_acc = {}
    rrlogf, rrlogpath = open_model_log(out, "Jina-Reranker", "B")
    for ps in POOL_B:
        ok = 0
        for r in sub:
            pn = pools["pools"][str(ps)][str(r["id"])]
            pt = [by_name[n] for n in pn]
            docs = [_doc(t) for t in pt]
            top = rr.top1(r["prompt"], docs)
            pred = pt[top]["function"]["name"]
            gt_name = r["expected"]["name"]
            g = "correct_selection" if pred == gt_name else "incorrect"
            log_row(rrlogf, r["id"], ps, gt_name, None, pred, None, g,
                    0, docs[top])
            if pred == gt_name: ok += 1
        rr_acc[str(ps)] = ok / len(sub) if sub else 0
        print(f"  reranker pool={ps}: {rr_acc[str(ps)]:.4f}")
    rrlogf.close()
    print(f"  log -> {rrlogpath}")
    rr.done()

    llm_acc = {}
    tok_mean = defaultdict(dict)
    for idx, mi in enumerate(models):
        lb = mi["label"]
        print(f"\n  {lb}")
        llm = load_llm(mi)
        llm_acc[lb] = {}
        logf, logpath = open_model_log(out, lb, "B")
        for ps in POOL_B:
            t0 = time.time(); ok = 0; tok_sum = 0
            for r in sub:
                pn = pools["pools"][str(ps)][str(r["id"])]
                pt = [by_name[n] for n in pn]
                pred, pred_args, gen_tokens, raw_text = dispatch(llm, r["prompt"], pt, mi)
                tok_sum += gen_tokens
                gt_name = r["expected"]["name"]
                gt_args = r["expected"].get("arguments")
                g = grade(gt_name, gt_args, pred, pred_args)
                log_row(logf, r["id"], ps, gt_name, gt_args, pred, pred_args,
                        g, gen_tokens, raw_text)
                if pred == gt_name: ok += 1
            a = ok / len(sub) if sub else 0
            llm_acc[lb][str(ps)] = a
            tok_q = tok_sum / len(sub) if sub else 0
            tok_mean[lb][str(ps)] = tok_q
            print(f"    pool={ps}: {a:.4f}  tok/q={tok_q:.1f}  {time.time()-t0:.1f}s")
        logf.close()
        print(f"    log -> {logpath}")
        if not variants_share_path(models, idx):
            free_llm(mi)

    res = {"part": "B", "n": len(sub), "sizes": POOL_B,
           "reranker": rr_acc, "llms": llm_acc,
           "mean_gen_tokens": dict(tok_mean)}

    res_path = out / "results_b.json"
    res_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  Saved Part B results to {res_path}")

def load_dataset_replies(path: str) -> List[Dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(raw):
        if isinstance(r, dict):
            if r.get("tool") is None:
                continue
            prompt = r["query"]
            expected = {"name": r["tool"], "arguments": r.get("parameters", {})}
            replies = r.get("replies", [])
        else:
            if len(r) < 2 or r[1] is None:
                continue
            prompt = r[0]
            expected = r[1]
            replies = r[2] if len(r) > 2 else []
        out.append({"id": i, "prompt": prompt, "expected": expected, "replies": replies})
    if LIMIT:
        out = out[:LIMIT]
    return out

def make_reply_samples(data, sizes, seed=SEED):
    samples = {}
    for n in sizes:
        samples[str(n)] = {}
        for row in data:
            total = len(row["replies"])
            order = list(range(total))
            random.Random(seed + row["id"]).shuffle(order)
            k = min(n, total)
            samples[str(n)][str(row["id"])] = sorted(order[:k])
    return {"seed": seed, "sizes": sizes, "samples": samples}

def write_samples_txt(s, path: Path):
    lines = [f"# seed={s['seed']} sizes={','.join(map(str, s['sizes']))}"]
    for n in s["sizes"]:
        block = s["samples"][str(n)]
        for sid in sorted(block, key=lambda x: int(x)):
            lines.append(f"{n}\t{sid}\t{','.join(map(str, block[sid]))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def read_samples_txt(path: Path):
    samples = {}; seed = SEED; sizes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            for tok in line.lstrip("#").split():
                if tok.startswith("seed="):
                    seed = int(tok.split("=", 1)[1])
                elif tok.startswith("sizes="):
                    sizes = [int(x) for x in tok.split("=", 1)[1].split(",") if x]
            continue
        n, sid, idxs = line.split("\t")
        samples.setdefault(str(int(n)), {})[sid] = [int(x) for x in idxs.split(",") if x]
    if not sizes:
        sizes = sorted(int(k) for k in samples)
    return {"seed": seed, "sizes": sizes, "samples": samples}

def get_or_make_reply_samples(data, sizes, path: Path):
    if path.exists():
        print(f"  Reusing cached reply samples from {path}")
        return read_samples_txt(path)
    s = make_reply_samples(data, sizes)
    write_samples_txt(s, path)
    print(f"  Saved reply samples to {path}")
    return s

def summarize(llm, system_prompt, query, gt, replies, by_name, mi):
    fg = mi["fg"]
    tname = gt["name"]
    targs = gt.get("arguments", {}) or {}
    tdef = by_name.get(tname)
    tools = _lc([tdef]) if tdef else []
    role_sys = "developer" if fg else "system"
    sysp = system_prompt if fg else sys_prompt_for(mi, system_prompt)

    msgs = [
        {"role": role_sys, "content": sysp},
        {"role": "user", "content": query},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": tname, "arguments": json.dumps(targs)}}]},
        {"role": "tool", "name": tname, "content": json.dumps(replies, ensure_ascii=False)},
    ]
    kw = dict(messages=msgs, temperature=0.0, seed=SEED, repeat_penalty=1.1)
    if tools:
        kw["tools"] = tools
    if not fg:
        kw["max_tokens"] = 1024
    try:
        r = llm.create_chat_completion(**kw)
    except Exception as e:
        return f"[ERROR] {e}", "", 0
    msg = r["choices"][0]["message"]
    text = msg.get("content", "") or ""
    raw_text = msg.get("_raw_text", "") or ""
    gen_tokens = (r.get("usage") or {}).get("completion_tokens", 0)
    return text, raw_text, gen_tokens

def run_c(data_rep, models, out, by_name):
    print("\n" + "=" * 60 +
          "\n  PART C - Summarize from Tool Results (N replies)\n" +
          "=" * 60)

    if not data_rep:
        print("  No tool-bearing sentences with replies - skipping Part C")
        return

    print(f"  {len(data_rep)} sentences, N_replies={REPLIES_N}, "
          f"prompts={list(P_C_PROMPTS)}")

    samples = get_or_make_reply_samples(data_rep, REPLIES_N, out / "reply_samples_c.txt")

    token_summary = defaultdict(dict)

    for idx, mi in enumerate(models):
        lb = mi["label"]
        print(f"\n  {lb}")
        llm = load_llm(mi)

        for pkey, sysprompt in P_C_PROMPTS.items():
            for n in REPLIES_N:
                records = []
                tok_sum = 0
                t0 = time.time()
                for row in data_rep:
                    sid = str(row["id"])
                    idxs = samples["samples"][str(n)][sid]
                    seen = [row["replies"][i] for i in idxs]
                    text, raw_text, gen_tokens = summarize(
                        llm, sysprompt, row["prompt"], row["expected"],
                        seen, by_name, mi)
                    tok_sum += gen_tokens
                    records.append({
                        "user": row["prompt"],
                        "tool_call": row["expected"],
                        "function_reply": seen,
                        "raw_text": raw_text,
                        "model_final_reply": text,
                        "gen_tokens": gen_tokens,
                    })
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", lb)
                fname = out / f"partC_{safe}__{pkey}__N{n}.json"
                fname.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                mean_tok = tok_sum / len(data_rep) if data_rep else 0
                token_summary[lb][f"{pkey}/N{n}"] = round(mean_tok, 2)
                print(f"    {pkey} N={n}: mean_tok={mean_tok:.1f} "
                      f"{time.time()-t0:.1f}s -> {fname.name}")

        if not variants_share_path(models, idx):
            free_llm(mi)

    tok_path = out / "results_c_tokens.json"
    tok_path.write_text(json.dumps({
        "part": "C", "n": len(data_rep),
        "replies_n": REPLIES_N, "prompts": list(P_C_PROMPTS),
        "mean_gen_tokens": dict(token_summary),
    }, indent=2), encoding="utf-8")
    print(f"  Saved Part C token summary to {tok_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.json")
    ap.add_argument("--replies-dataset", default="dataset_tools_w_replies.json")
    ap.add_argument("--tools", default="tools.json")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--output", default="results")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    a = ap.parse_args()

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if LIMIT:
        print(f"[LIMIT] Active - testing only the first {LIMIT} sentences")

    data = load_dataset(a.dataset)
    print(f"Loaded {len(data)} questions from {a.dataset}")

    all_tools, by_name, no_param = load_tools(a.tools)
    print(f"Loaded {len(all_tools)} tools from {a.tools} ({len(no_param)} zero-param)")

    models = find_models(Path(a.models_dir))
    if not models:
        print("No GGUF models found."); sys.exit(1)
    for m in models:
        if m["fg"]:
            tag = " <- FG handler"
        elif m["qwen3"]:
            tag = f" <- qwen3 handler ({'think' if m['think'] else 'no_think'})"
        else:
            tag = ""
        print(f"  {m['label']} ({m['params_b']}B){tag}")
    if not HAS_FG and any(m["fg"] for m in models):
        print("[WARN] functiongemma_handler not installed")
    if not HAS_QWEN3 and any(m["qwen3"] for m in models):
        print("[WARN] qwen3_handler not installed")

    if not a.skip_a: run_a(data, models, out, all_tools, by_name)
    if not a.skip_b: run_b(data, models, out, all_tools, by_name, no_param)
    if not a.skip_c:
        if Path(a.replies_dataset).exists():
            data_rep = load_dataset_replies(a.replies_dataset)
            print(f"Loaded {len(data_rep)} tool+replies sentences "
                  f"from {a.replies_dataset}")
            run_c(data_rep, models, out, by_name)
        else:
            print(f"[skip C] {a.replies_dataset} not found")
    print("\nDone - ", out)

if __name__ == "__main__":
    main()
