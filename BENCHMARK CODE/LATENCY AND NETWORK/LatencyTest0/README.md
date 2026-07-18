# LatencyTest0 - reply-count latency test (single device, no network)

> Back to [Latency and network README](../README.md)

Self-contained scripts that measure **leader** latency on each edge board as the
number of **replies returned by a single tool** grows. This is the mirror image
of [LatencyTest1](../LatencyTest1/README.md):

| | LatencyTest1 | **LatencyTest0** |
|---|---|---|
| Swept axis | number of **tools** exposed (1, 6, 12, 18, 24) | number of **replies** returned (1, 5, 10, 15, 20) |
| Tool count | grows | **always exactly 1** (`detect_people`) |
| Replies per call | always 1 (one sensing agent) | **grows** (as if N agents answered) |
| Sensing process / network | real, over TCP | **none** - built in-process |

The dispatch stage never changes here (always the same single tool on the same
query), so it is expected to be flat across configs. What grows is the **answer
stage**: the leader is fed a larger multi-agent result set to summarize, so
`answer_tokens_in`, `answer_ttft_ms` and `result_to_reply_ms` climb with the
reply count. That growth is exactly what this test isolates.

Everything for one device lives in **one file** (plus the shared `bench_core`
helper). Because "network latency does not matter" for this measurement, there is
no sensing script to start and no discovery/TCP: the tool result is built
in-process by `bench_core.build_replies(n)` with the **exact** shape a real leader
receives from its dispatcher - a list of `{"from": "sensing-agent-<i>", "text":
"no people detected"}`, one entry per answering agent (see
[`tool_dispatcher.py`](../../../IMPLEMENTATION/LeaderLogic/tool_dispatcher.py)). Only the (irrelevant)
network hop is skipped; the answer LLM sees the same result set it would
aggregate live.

The leader stages still use the **live preset models and the live prompts**
(FunctionGemma on the STM32, Workflow1 / qwen3-on-Hailo on the Pi) and are
**streamed** so time-to-first-token and decode tk/s can be measured. No live
module is modified and no new preset is added; this folder is self-contained.

## Files

| Script | Runs on | Variant |
|---|---|---|
| [`bench_core.py`](bench_core.py) | (imported) | Shared engine: reply-count sweep, streaming, stats, JSON. A trimmed copy of LatencyTest1's `bench_core` with the sensing/network half removed. |
| [`test_leader_stm32_replies_reset.py`](test_leader_stm32_replies_reset.py) | STM32MP257F-DK | FunctionGemma on CPU, **`llm.reset()` before every generation** (cold prefill each time). |
| [`test_leader_stm32_replies_kvreuse.py`](test_leader_stm32_replies_kvreuse.py) | STM32MP257F-DK | FunctionGemma on CPU, **no reset + trigger-only prompt** - llama.cpp re-uses the cached prefix (KV-cache reuse). |
| [`test_leader_raspberry_replies.py`](test_leader_raspberry_replies.py) | Raspberry Pi 5 + Hailo | Workflow1 / qwen3 on the Hailo NPU (clears context each generation, as it does live). |

### The two STM32 variants

The only difference between the two STM32 scripts is whether `llm.reset()` is
called between generations:

- **reset**: `llm.reset()` before each dispatch and each answer, so every
  generation is an independent **cold prefill** (no KV-cache carry-over). This is
  the baseline and matches LatencyTest1's STM32 leader.
- **kvreuse**: the context is left intact **and** the prompts use the bare
  FunctionGemma `TRIGGER` developer block only (no custom smart-home system
  prompt). Leaving the context intact lets llama.cpp's `create_completion()`
  detect the longest common prefix between the new prompt and the tokens already
  in the KV cache and re-use it; using the trigger-only prompt makes that shared
  prefix long, because the developer block (`TRIGGER` + the single tool
  declaration) is then IDENTICAL for dispatch and answer and across every query.
  With the divergent per-stage system prompts of the reset variant, the shared
  prefix would be only a few tokens, so prefill (and TTFT / `query_to_tool_ms`)
  gets faster here whenever consecutive prompts share that block.

`dispatch_tokens_in` / `answer_tokens_in` report the **full** prompt size in both
variants (what was fed); the reuse shows up as lower `dispatch_ttft_ms` /
`query_to_tool_ms`, not as fewer input tokens. `tokens_out` stays exact.

The Pi/Hailo path clears its context each generation (Workflow1's live behavior),
so it has no reset/reuse split - one script only.

## What is measured

**Leader**, per query (saved after every query, so a crash loses nothing):
- `query_to_tool_ms`   - query received to tool call ready (dispatch generation)
- `dispatch_ttft_ms`   - query received to **first** dispatch token (prefill included)
- `dispatch_tokens_in` / `dispatch_tokens_out` / `dispatch_tps` - dispatch tokens + decode tk/s
- `tool_to_result_ms`  - time to build the reply set in-process (no network, so ~0)
- `result_to_reply_ms` - result received to final reply ready (answer generation)
- `answer_ttft_ms` / `answer_tokens_in` / `answer_tokens_out` / `answer_tps` - the answer stage
- plus `{query, tool_called, tool_result, tool_result_raw, final_reply}`, `correct`, `end_to_end_ms`
- plus a per-config (per reply count) and overall aggregate.

`tokens_in` is exact on the STM32 (llama.cpp). On the Pi/Hailo the genai streaming
API does not expose prompt tokens, so `*_tokens_in` is `null` there.

**Leader**, `prompt_specs` (one entry per reply count) records what the LLM was
fed: the single tool def, the (constant) `dispatch_prompt`, and the
`answer_prompt` for that config - the last one **grows** with the reply count,
which is the point of the test.

## How to run

Run from the `IMPLEMENTATION/` folder. There is nothing else to start (no sensing
agent, no network). You type nothing during a run.

```
# on the STM32 - run whichever variant you want (or both, one after the other)
python LatencyTest0/test_leader_stm32_replies_reset.py
python LatencyTest0/test_leader_stm32_replies_kvreuse.py

# on the Raspberry Pi
python LatencyTest0/test_leader_raspberry_replies.py
```

Each script loads the model, then runs reply-count configs
`1 -> 5 -> 10 -> 15 -> 20` automatically, driving the same 10 person queries each
time, and writes its JSON.

## Output

Written to `LatencyTest0/results/`:
- `leader_stm32-reset_<ts>.json`
- `leader_stm32-kvreuse_<ts>.json`
- `leader_raspberry_<ts>.json`

Each holds `prompt_specs` (per reply count), `samples` (per query) and
`aggregate` (per reply count). The schema matches LatencyTest1's leader JSON
except the swept-axis key is `n_replies` / `reply_counts` instead of `n_tools` /
`tool_counts`, and there is no `time_to_boot` block (this test does not measure
boot), so the two tests' per-query numbers stay directly comparable
field-for-field.

## Notes on fidelity

The leader stages build the **same prompts** the live workflow hooks build
(`stm32mp257fdk_leader._dispatch`/`_answer`, `Workflow1` on the Pi) on the same
model objects. The only reason the bench streams the generation itself instead of
calling the hooks verbatim is that the hooks return finished text, which cannot
expose time-to-first-token; streaming yields identical output while clocking the
first token and the decode rate. The replies fed to the answer stage carry the
exact `{from, text}` shape the live dispatcher hands the leader, so the answer
prompt is byte-for-byte what a real N-agent aggregation would produce - only the
network round trip (which this test deliberately ignores) is absent.
