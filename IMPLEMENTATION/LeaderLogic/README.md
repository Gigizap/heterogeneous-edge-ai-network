# Leader preset contract

Every leader preset is a folder under `LeaderLogic/<preset>/` holding a
`leader.py` (+ `config.json` / `requirements.txt` as needed) - mirroring how a
sensing preset is a folder under `SensingLogic/<preset>/`. Presets are discovered automatically at first-run setup
(`ElectionLogic.identity.discover_leader_presets()`) - dropping in a new
folder makes it selectable with no code change elsewhere.

`LeaderLogic/run_leader.py` is the shared entry point (the leader-side
counterpart of `SensingLogic/sensing_agent.py`): it owns the LeaderNetwork
(tool registry + dispatcher), conversation history for backup/failover, and
the two-step pipeline (query -> tool -> devices, tool result -> reply). A
preset only supplies the two model-specific pieces of that pipeline.

It also owns `/loop` (see the root README): when enabled, `_run_pipeline`
hands the tool it just dispatched to a background thread that re-dispatches it
and calls `backend.answer()` again whenever the per-device results change. It
reuses `net.dispatch()` and the preset's `answer()` unchanged, so a preset
needs no code for it.

## `leader.py` contract

```python
def load(cfg: dict, profile: dict) -> Backend:
    ...
```

Returns a **Backend** object:

```python
class Backend:
    model_name: str        # human-readable model label, e.g. "qwen3:1.7b (Hailo)"
    model_params_b: float  # approximate parameter count in billions

    def dispatch(self, user_text: str, tool_defs: list) -> tuple:
        """Return (fn_name, args, no_tool_reply).
        fn_name is None when no tool matches - no_tool_reply is then the
        text to answer with directly (dispatch's own reply, or a fallback)."""

    def answer(self, user_text: str, command: str, network_reply: list) -> str:
        """Turn the sensing agents' replies into a reply for the user."""
```

`run_leader.boot()` reads `Backend.model_name` / `model_params_b` after
`load()` returns (for the capability broadcast / `/status`).

### Legacy contract (the `mock` preset only)

`mock` predates this split and isn't a pipeline at all (no LeaderNetwork, no
LLM) - its `leader.py` has no `load()`, so `run_leader.boot()` falls back to
calling its `boot(cfg, agent_id, transport, discovery, bot, profile) -> Workflow`
directly, where `Workflow` implements `handle(chat_id, text)`,
`get_history(chat_id)`, `load_history(chat_id, messages)`. Only add a preset
in this shape if it genuinely can't fit the Backend contract above.

## Existing presets

| Folder                | Dispatch                        | Answer                | Hardware              |
|------------------------|----------------------------------|------------------------|------------------------|
| `raspberry_cpu/`       | qwen3:1.7b (CPU Ollama)          | same (CPU Ollama)      | Raspberry Pi 5         |
| `raspberry_hailo/`     | qwen3:1.7b (Hailo native `.hef`) | same (Hailo)           | Raspberry Pi 5 + Hailo |
| `raspberry_cpuhailo/`  | functiongemma:270m (CPU Ollama)  | qwen3:1.7b (Hailo)     | Raspberry Pi 5 + Hailo |
| `generic/`             | llama.cpp GGUF (score-selected)  | same (llama.cpp)       | Any device             |
| `stm32mp257fdk/`       | functiongemma:270m (llama.cpp)   | same (llama.cpp)       | STM32MP257F-DK         |
| `mock/`                | -                                | -                       | Any device (test only) |

Shared infrastructure (used by more than one preset) lives at the top level
of `LeaderLogic/`, not inside a preset folder: `run_leader.py` (pipeline),
`leader_network.py` / `tool_registry.py` / `tool_dispatcher.py` (network),
`http_backend.py` (OpenAI-compatible HTTP client), `hailo_backend.py` (Hailo
NPU wrapper), `llama_cpp_backend.py` (shared GGUF backend for `generic` and
`stm32mp257fdk`), `ollama_loader.py` (starts/waits on an Ollama server).

## To add a new preset

1. Create `LeaderLogic/yourdevice/leader.py` implementing `load()` above
   (reuse `http_backend.py` / `hailo_backend.py` / `llama_cpp_backend.py`
   where they fit).
2. Add `LeaderLogic/yourdevice/requirements.txt` (and `config.json` if the
   preset needs device-specific settings) next to it.
3. Run `main.py` on the device - the new preset shows up in the first-run
   setup menu automatically.
