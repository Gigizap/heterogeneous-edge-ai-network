import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_IMPL = _HERE.parent
for _p in (str(_IMPL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc
import itertools
import json
import logging
import statistics
import threading
import time
from pathlib import Path

from ConnectionLogic.transport import P2PTransport
from ConnectionLogic.discovery import Discovery
from utils import start_async_loop, schedule

log = logging.getLogger("latencytest")

TOOL_COUNTS = (1, 6, 12, 18, 24)

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

SENSING_PORT = 5720
LEADER_PORT = 5721

_list_ids = itertools.count(1)

def _wrong_tool(i: int) -> dict:
    return {
        "type": "function",
        "function": {
            "name": f"wrong_tool_{i}",
            "description": "do not call this tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

def build_tool_defs(n: int, correct_def: dict = DETECT_PEOPLE_DEF) -> list:
    defs = [correct_def]
    i = 1
    while len(defs) < n:
        defs.append(_wrong_tool(i))
        i += 1
    return defs

def _stats(values, ndigits: int) -> dict:
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

def start_network(agent_id: str, port: int):
    loop = start_async_loop()
    transport = P2PTransport(agent_id, port, lambda msg: None)
    transport._event_loop = loop
    discovery = Discovery(
        agent_id, port,
        on_peer_found=lambda pid, ip, p: schedule(loop, transport.register_peer(pid, ip, p)),
        on_peer_lost=lambda pid: schedule(loop, transport.unregister_peer(pid)),
    )
    schedule(loop, transport.start())
    schedule(loop, discovery.start())
    return loop, transport, discovery

def wait_for_peer(transport, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if getattr(transport, "_peers", {}):
            return True
        time.sleep(0.5)
    return bool(getattr(transport, "_peers", {}))

def timed_stream(pieces):
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
    decode_s = gen_s - ttft_s
    if tokens_out > 1 and decode_s > 0:
        return round((tokens_out - 1) / decode_s, 2)
    return None

def measure_time_to_boot(load_fn, unload_fn, n: int = 10) -> dict:
    samples = []
    for i in range(n):
        elapsed, handle = load_fn()
        samples.append(round(elapsed * 1000, 2))
        log.info("time_to_boot %d/%d = %.0f ms", i + 1, n, samples[-1])
        try:
            unload_fn(handle)
        except Exception as e:
            log.warning("time_to_boot unload failed: %r", e)
        gc.collect()
    st = _stats(samples, 2)
    log.info("time_to_boot: mean=%s std=%s min=%s max=%s ms (n=%d)",
             st["mean"], st["std"], st["min"], st["max"], st["count"])
    return {"unit": "ms", "samples": samples, **st}

def _sensing_peers(transport, net) -> list:
    owners = sorted(net.registry.owners(CORRECT_TOOL))
    if owners:
        return owners
    return list(getattr(transport, "_peers", {}).keys())

def _repull_and_wait(transport, net, peers, n: int, timeout: float = 6.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for pid in peers:
            transport.send_sync(pid, {
                "jsonrpc": "2.0",
                "id": f"list-bench-{next(_list_ids)}",
                "method": "tools/list",
            })
        time.sleep(0.4)
        if len(net.available_tools()) == n:
            return True
    return len(net.available_tools()) == n

def _set_remote_tools(transport, net, n: int) -> None:
    peers = _sensing_peers(transport, net)
    if not peers:
        log.warning("no sensing peer to configure for n=%d", n)
        return
    for pid in peers:
        transport.send_sync(pid, {
            "jsonrpc": "2.0",
            "id": f"bench-set-{n}",
            "method": "bench/set_tools",
            "params": {"n": n, "correct_tool": CORRECT_TOOL},
        })
    time.sleep(0.3)
    if not _repull_and_wait(transport, net, peers, n):
        log.warning("registry shows %d tool(s), expected %d (continuing)",
                    len(net.available_tools()), n)

def _signal_done(transport, net) -> None:
    for pid in _sensing_peers(transport, net):
        transport.send_sync(pid, {"jsonrpc": "2.0", "id": "bench-done", "method": "bench/done"})
    time.sleep(1.0)

def _leader_aggregate(samples: list) -> list:
    metrics = [
        "query_to_tool_ms", "dispatch_ttft_ms",
        "dispatch_tokens_in", "dispatch_tokens_out", "dispatch_tps",
        "tool_to_result_ms",
        "result_to_reply_ms", "answer_ttft_ms",
        "answer_tokens_in", "answer_tokens_out", "answer_tps",
        "end_to_end_ms",
    ]
    out = []
    for n in TOOL_COUNTS:
        rows = [s for s in samples if s["n_tools"] == n and not s["error"]]
        ok = [s for s in rows if s["got_result"] == 1]
        agg = {
            "n_tools": n,
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

def run_leader_benchmark(*, device, preset_label, transport=None, net=None,
                         dispatch_stream, answer_stream, reset_fn=None,
                         time_to_boot=None, out_dir=None,
                         tool_fn=None, tools_for=None) -> dict:
    local = tool_fn is not None
    out = _results_dir(out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"leader_{device}_{stamp}.json"

    result = {
        "role": "leader",
        "device": device,
        "leader_preset": preset_label,
        "correct_tool": CORRECT_TOOL,
        "tool_counts": list(TOOL_COUNTS),
        "queries": list(PERSON_QUERIES),
        "time_to_boot": time_to_boot or {},
        "prompt_specs": [],
        "samples": [],
        "aggregate": [],
    }

    for n in TOOL_COUNTS:
        log.info("===== config: %s exposes %d tool(s) =====",
                 "in-place tool" if local else "sensing", n)
        if local:
            tools_now = tools_for(n)
        else:
            _set_remote_tools(transport, net, n)
            tools_now = net.available_tools()
        log.info("leader sees %d tool(s) for this config", len(tools_now))

        spec = {
            "n_tools": n,
            "tools": tools_now,
            "dispatch_prompt": None,
            "answer_prompt": None,
        }
        result["prompt_specs"].append(spec)

        for qi, query in enumerate(PERSON_QUERIES):
            if reset_fn:
                try:
                    reset_fn()
                except Exception:
                    pass

            sample = {
                "n_tools": n, "query_idx": qi, "query": query,
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
                tool_defs = tools_now if local else net.available_tools()
                d = dispatch_stream(query, tool_defs)
                t1 = time.perf_counter()

                if spec["dispatch_prompt"] is None:
                    spec["dispatch_prompt"] = d.get("prompt")

                fn, args = d.get("fn"), d.get("args")
                if fn:
                    replies = tool_fn(fn, args or {}) if local else net.dispatch(fn, args or {})
                else:
                    replies = []
                t2 = time.perf_counter()

                a = None
                if replies:
                    a = answer_stream(query, fn, args, replies, tool_defs)
                    final = a.get("text", "")
                elif fn is None:
                    final = d.get("reply_text", "")
                else:
                    final = f"No peer responded to '{fn}'"
                t3 = time.perf_counter()

                if a is not None and spec["answer_prompt"] is None:
                    spec["answer_prompt"] = a.get("prompt")

                sample.update({
                    "tool_called": fn or "",
                    "tool_result": "\n".join(str(r.get("text", "")) for r in replies),
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
                log.info("n=%-2d q=%d picked=%-12s ok=%d  "
                         "D[in=%s out=%s ttft=%.0f qt=%.0f tps=%s]  tr=%.0f  "
                         "A[in=%s out=%s ttft=%s rr=%.0f tps=%s]  e2e=%.0f ms",
                         n, qi, sample["tool_called"] or "-", sample["correct"],
                         sample["dispatch_tokens_in"], sample["dispatch_tokens_out"],
                         sample["dispatch_ttft_ms"], sample["query_to_tool_ms"],
                         sample["dispatch_tps"], sample["tool_to_result_ms"],
                         sample["answer_tokens_in"], sample["answer_tokens_out"],
                         sample["answer_ttft_ms"], sample["result_to_reply_ms"],
                         sample["answer_tps"], sample["end_to_end_ms"])
            except Exception as e:
                sample["error"] = repr(e)
                log.exception("query failed (n=%d q=%d)", n, qi)

            result["samples"].append(sample)
            _save_json(path, result)

    result["aggregate"] = _leader_aggregate(result["samples"])
    _save_json(path, result)
    if not local:
        _signal_done(transport, net)
    log.info("leader benchmark complete -> %s", path)
    return result

class SensingBench:

    def __init__(self, *, device, preset_label, skill_fn, transport,
                 out_dir=None, warmup=True):
        self.device = device
        self.preset_label = preset_label
        self.skill_fn = skill_fn
        self.transport = transport

        out = _results_dir(out_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = out / f"sensing_{device}_{stamp}.json"

        self._lock = threading.Lock()
        self.current_n = 1
        self.current_defs = build_tool_defs(1)
        self.calls = []
        self.done = threading.Event()
        self._finalized = False

        if warmup:
            self._warmup()

    def _warmup(self):
        try:
            log.info("sensing warmup: one detection to load model/camera ...")
            r = self.skill_fn()
            log.info("sensing warmup done: %r", str(r)[:120])
        except Exception as e:
            log.warning("sensing warmup failed (continuing anyway): %r", e)

    def on_message(self, msg: dict):
        if msg.get("type"):
            return
        method = msg.get("method")
        if not method:
            return

        sender = msg.get("from")
        rpc_id = msg.get("id")

        def respond(payload: dict):
            if rpc_id is None:
                return
            try:
                self.transport.send_sync(sender, {"jsonrpc": "2.0", "id": rpc_id, **payload})
            except Exception as e:
                log.warning("could not reply to %s: %r", sender, e)

        if method == "tools/list":
            with self._lock:
                defs = list(self.current_defs)
            respond({"result": {"tools": defs}})
            return

        if method == "tools/call":
            t_recv = time.perf_counter()
            params = msg.get("params") or {}
            name = params.get("name")
            if name == CORRECT_TOOL:
                try:
                    text = str(self.skill_fn())
                except Exception as e:
                    text = f"detection error: {e}"
            else:
                text = f"{name}: wrong tool executed (placeholder)"
            dt = time.perf_counter() - t_recv
            with self._lock:
                n = self.current_n
                self.calls.append({
                    "n_tools": n, "tool": name,
                    "time_to_reply_s": round(dt, 4),
                    "correct": int(name == CORRECT_TOOL),
                    "result": text,
                })
            log.info("tools/call %s (n=%d) time_to_reply=%.4fs", name, n, dt)
            respond({"result": {"content": [{"type": "text", "text": text}]}})
            return

        if method == "bench/set_tools":
            params = msg.get("params") or {}
            n = int(params.get("n", 1))
            with self._lock:
                self.current_n = n
                self.current_defs = build_tool_defs(n)
            log.info("bench/set_tools -> now exposing %d tool(s)", n)
            respond({"result": {"n": n, "exposed": n}})
            return

        if method == "bench/done":
            self._finalize()
            respond({"result": {"written": str(self.path)}})
            self.done.set()
            return

        respond({"error": {"code": -32601, "message": f"method not found: {method}"}})

    def _finalize(self):
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            calls = list(self.calls)

        per_config = []
        for n in TOOL_COUNTS:
            vals = [c["time_to_reply_s"] for c in calls if c["n_tools"] == n and c["correct"]]
            per_config.append({"n_tools": n, **_stats(vals, 4)})
        overall = _stats([c["time_to_reply_s"] for c in calls if c["correct"]], 4)

        data = {
            "role": "sensing",
            "device": self.device,
            "sensing_preset": self.preset_label,
            "correct_tool": CORRECT_TOOL,
            "tool_counts": list(TOOL_COUNTS),
            "time_to_reply_unit": "seconds",
            "calls": calls,
            "per_config": per_config,
            "overall": overall,
        }
        _save_json(self.path, data)
        log.info("sensing results written -> %s", self.path)

    def serve(self):
        self.transport.on_message = self.on_message
        log.info("sensing agent ready (%s) - waiting for the leader to drive configs",
                 self.preset_label)
        try:
            while not self.done.wait(timeout=1.0):
                pass
        except KeyboardInterrupt:
            log.info("interrupted - finalizing partial results")
            self._finalize()
        time.sleep(0.5)
        log.info("sensing agent exiting")
