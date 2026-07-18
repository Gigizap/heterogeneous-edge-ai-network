"""
LatencyTest1/bench_core.py

Shared engine for the FIRST latency test (2 devices, no laptop).

Two roles, driven by the four thin scripts next to this file:

  * LEADER  (test_raspberry_leader.py / test_stm32_leader.py)
        Uses the LIVE leader inference path - Workflow1 (qwen3 on the Hailo NPU)
        on the Pi, functiongemma on CPU on the STM32 - and times, per query:
            query_to_tool   = available_tools() + _dispatch()   (query received -> tool sent)
            tool_to_result  = net.dispatch()                    (tool sent -> result received)
            result_to_reply = _answer()                         (result received -> reply ready)
        These are the exact hooks the live pipeline calls (BaseWorkflow._run_pipeline),
        so the numbers match main.py; only the Telegram/backup envelope (which is
        OUTSIDE those three intervals) is stripped.

  * SENSING (test_raspberry_sensing.py / test_stm32_sensing.py)
        Runs the LIVE detection skill and measures, per tools/call:
            time_to_reply   = tool received -> result sent      (no network lag)

The "number of tools" axis (1, 6, 12, 18, 24) is REAL, not simulated: the sensing
agent actually exposes N tools (detect_people + wrong_tool_1..wrong_tool_(N-1)),
and the leader picks from what it fetched over the network. The leader is the
orchestrator: before each config it tells the sensing agent how many tools to
expose (bench/set_tools), re-pulls tools/list so its registry matches, runs the
10 queries, then signals the end (bench/done).

bench/set_tools and bench/done are TEST-ONLY control messages spoken only between
these scripts. They are deliberately NOT added to the production sensing_agent /
protocol - running these scripts bypasses main.py and changes no live preset.

Stdlib only (json, time, statistics, threading, itertools, logging, pathlib).
The heavy runtimes (Hailo, llama.cpp, TFLite, camera) come in only through the
LIVE preset modules the thin scripts import - never from this file.
"""

import sys
import pathlib

# OS-agnostic imports: put the IMPLEMENTATION dir (parent) and this dir on the
# path so "from ConnectionLogic..." and "import bench_core" both resolve no
# matter the current working directory or platform.
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


# ── test configuration ───────────────────────────────────────────────────────

# 5 configurations: how many tools the sensing agent exposes (1 correct + rest wrong).
TOOL_COUNTS = (1, 6, 12, 18, 24)

# The single correct tool. Same name AND description exposed to both couples, so
# the two leaders are measured on an identical dispatch prompt (same token count);
# on the STM32 it maps to the live person_detection skill, on the Pi to
# object_detection filtered to "person" (see the sensing scripts).
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

# 10 queries that each require the person-detection tool.
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

# Test TCP ports (distinct from main.py's 5555 and BenchmarkLeader's 5600/5602).
# The two roles run on different devices, so these never collide; distinct values
# just keep things sane if someone runs both on one box.
SENSING_PORT = 5720
LEADER_PORT = 5721

# tools/list request ids for the leader's manual re-pulls (kept distinct from the
# dispatcher's "rpc-N" and LeaderNetwork's "list-N" so nothing is mis-routed).
_list_ids = itertools.count(1)


# ── generic helpers ──────────────────────────────────────────────────────────

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
    """[correct_def] padded with wrong_tool_i until the list has exactly n entries."""
    defs = [correct_def]
    i = 1
    while len(defs) < n:
        defs.append(_wrong_tool(i))
        i += 1
    return defs


def _stats(values, ndigits: int) -> dict:
    """count / mean / std (population) / min / max for a list of numbers."""
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


# ── network bringup (shared by both roles) ───────────────────────────────────

def start_network(agent_id: str, port: int):
    """Start a transport + discovery on a background event loop.

    Wires discovery -> transport peer table (so this agent can send to / reply to
    whoever it discovers). Leaders then hand transport+discovery to make_network /
    the leader boot(); sensing agents set transport.on_message to their handler.
    Returns (loop, transport, discovery).
    """
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
    """Block until at least one peer (the other role) is in the transport table."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if getattr(transport, "_peers", {}):
            return True
        time.sleep(0.5)
    return bool(getattr(transport, "_peers", {}))


# ── streaming + boot-timing helpers (leader side) ─────────────────────────────

def timed_stream(pieces):
    """Consume an iterator of text pieces (one per generated token), timing the
    FIRST token (TTFT) and the whole generation, while accumulating the text.

    Returns (text, ttft_s, gen_s, n_pieces). Empty pieces are ignored (they do
    not count as the first token nor toward n_pieces), so a model that emits a
    stop token as "" does not skew TTFT. n_pieces is a token count only when the
    backend yields one token per piece; callers that can get an exact count
    (e.g. llama.cpp's n_tokens) should override it.
    """
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
    """Decode throughput = (tokens_out - 1) / (gen_s - ttft_s). None if undefined
    (the first token is attributed to prefill, so it is excluded)."""
    decode_s = gen_s - ttft_s
    if tokens_out > 1 and decode_s > 0:
        return round((tokens_out - 1) / decode_s, 2)
    return None


def measure_time_to_boot(load_fn, unload_fn, n: int = 10) -> dict:
    """Measure the LLM cold-start cost: load the model n times, fully evicting it
    between loads. load_fn() -> (elapsed_s, handle) times ONE cold load and
    returns something unload_fn(handle) can release. Returns
    {unit, samples(ms), count, mean, std, min, max}.

    The model file must already be present (download OUTSIDE this loop) so the
    samples measure load time, not a one-off download on the first iteration.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# LEADER side
# ══════════════════════════════════════════════════════════════════════════════

def _sensing_peers(transport, net) -> list:
    """Which peer(s) to drive. Prefer the owners of the correct tool (the sensing
    agent) if the registry knows them yet, else every current peer (there is one)."""
    owners = sorted(net.registry.owners(CORRECT_TOOL))
    if owners:
        return owners
    return list(getattr(transport, "_peers", {}).keys())


def _repull_and_wait(transport, net, peers, n: int, timeout: float = 6.0) -> bool:
    """Re-send tools/list to the sensing peer(s) and wait until the leader's
    registry reflects exactly n tools (i.e. the sensing agent has switched)."""
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
    """Tell the sensing agent to expose n tools, then sync the leader's registry."""
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
    time.sleep(0.3)  # let the switch apply before we re-pull
    if not _repull_and_wait(transport, net, peers, n):
        log.warning("registry shows %d tool(s), expected %d (continuing)",
                    len(net.available_tools()), n)


def _signal_done(transport, net) -> None:
    """Tell the sensing agent the whole run is over so it writes its results."""
    for pid in _sensing_peers(transport, net):
        transport.send_sync(pid, {"jsonrpc": "2.0", "id": "bench-done", "method": "bench/done"})
    time.sleep(1.0)  # let the sensing agent finalize + flush its reply


def _leader_aggregate(samples: list) -> list:
    """Per-config (and the caller adds overall) avg/std of the stage latencies plus
    the TTFT / token-count / decode-tps metrics for both LLM stages. Stats are over
    samples that produced a result, so a wrong dispatch that timed out does not skew
    them; accuracy is over all samples."""
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
    """Drive every (config x query), STREAMING both LLM stages so TTFT + decode
    tk/s are measured alongside the stage latencies, and save.

    dispatch_stream(query, tool_defs) -> dict with:
        fn           picked tool name, or None when the model answered in prose
        args         parsed arguments (dict) or None
        reply_text   the prose reply when fn is None (else "")
        ttft_s, gen_s, tokens_out, decode_tps   dispatch generation metrics
    answer_stream(query, fn, args, replies, tool_defs) -> dict with:
        text                                    the final reply
        ttft_s, gen_s, tokens_out, decode_tps   answer generation metrics
    reset_fn()                (optional; per-query state reset)
    time_to_boot              (optional dict from measure_time_to_boot; saved as-is)

    NETWORK mode (default): the tool is dispatched over the network to the sensing
    agent that owns it, so the skill runs on the real sensing device
    (tool_to_result_ms is that round trip). Pass transport + net.

    LOCAL mode: pass tool_fn(fn, args) -> replies and tools_for(n) -> tool_defs.
    The tool then runs in-process (e.g. a fixed-latency mock) with NO discovery /
    peer / network, and the bench/set_tools + bench/done handshake is skipped;
    transport + net are ignored. Everything else (metrics, schema, saving) is
    identical, so a local run's JSON is directly comparable to a network run's.

    The JSON is rewritten after EVERY query so nothing is lost on a crash.
    """
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

        # Per-config record of exactly what the LLM was given: the tool defs
        # attached (names + descriptions + params) and the precise prompt
        # construction of each stage, captured live from the closures on the
        # first query that exercises that stage.
        spec = {
            "n_tools": n,
            "tools": tools_now,
            "dispatch_prompt": None,   # the tool-call stage prompt (system + N tools + query)
            "answer_prompt": None,     # the tool-reply stage prompt (system + called tool + result)
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
            _save_json(path, result)  # save after every query

    result["aggregate"] = _leader_aggregate(result["samples"])
    _save_json(path, result)
    if not local:
        _signal_done(transport, net)
    log.info("leader benchmark complete -> %s", path)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SENSING side
# ══════════════════════════════════════════════════════════════════════════════

class SensingBench:
    """Serves tools/list + tools/call for the live detection skill, switches how
    many tools it exposes on bench/set_tools, and times every tools/call
    (time_to_reply, seconds). On bench/done it writes per-config and overall
    avg/std and stops.
    """

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
        self.calls = []              # {n_tools, tool, time_to_reply_s, correct, result}
        self.done = threading.Event()
        self._finalized = False

        if warmup:
            self._warmup()

    def _warmup(self):
        """One detection at startup so the model + camera load OUTSIDE the measured
        calls (the first real tools/call is then steady-state)."""
        try:
            log.info("sensing warmup: one detection to load model/camera ...")
            r = self.skill_fn()
            log.info("sensing warmup done: %r", str(r)[:120])
        except Exception as e:
            log.warning("sensing warmup failed (continuing anyway): %r", e)

    # ── message handling ─────────────────────────────────────────────────────

    def on_message(self, msg: dict):
        if msg.get("type"):        # election / backup / capability - not our concern
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
                # A padding tool: the leader mis-dispatched. Answer so it does not
                # hang; this call is excluded from the correct-tool aggregates.
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
                self.current_n = n                       # set first so calls bucket right
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

    # ── finalize + serve ─────────────────────────────────────────────────────

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
            "per_config": per_config,   # avg +- std of time_to_reply per tool count
            "overall": overall,         # avg +- std across all configs
        }
        _save_json(self.path, data)
        log.info("sensing results written -> %s", self.path)

    def serve(self):
        """Attach the handler and block until the leader signals bench/done
        (or Ctrl-C, which finalizes whatever was collected)."""
        self.transport.on_message = self.on_message
        log.info("sensing agent ready (%s) - waiting for the leader to drive configs",
                 self.preset_label)
        try:
            while not self.done.wait(timeout=1.0):
                pass
        except KeyboardInterrupt:
            log.info("interrupted - finalizing partial results")
            self._finalize()
        time.sleep(0.5)  # let the bench/done reply flush
        log.info("sensing agent exiting")
