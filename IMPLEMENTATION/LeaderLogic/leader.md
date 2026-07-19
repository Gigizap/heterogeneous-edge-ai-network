# Leader module contract

Every file in `LeaderLogic/` that can be elected as leader must expose:

## Module-level constants

```python
MODEL_NAME: str        # human-readable model label, e.g. "Qwen2.5-7B"
MODEL_PARAMS_B: float  # approximate parameter count in billions, e.g. 7.0
```

These are read by `main.py` when broadcasting capabilities to Telegram users.

## `boot()` function

```python
def boot(cfg: dict, agent_id: str, transport, discovery, bot) -> Workflow:
    ...
```

| Parameter   | Type            | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| `cfg`       | `dict`          | Full `config.json` contents                      |
| `agent_id`  | `str`           | This device's resolved agent ID                  |
| `transport` | `P2PTransport`  | Already-started transport                        |
| `discovery` | `Discovery`     | Already-started discovery                        |
| `bot`       | `TelegramBot`   | Telegram bot (not yet started; `boot` must call `bot.start()` or return before main does) |

Returns a **Workflow** object.

## Workflow object interface

```python
class Workflow:
    def run(self, chat_id: int, text: str) -> None:
        """Process one user message and send reply via bot."""

    def get_history(self, chat_id: int) -> list[dict]:
        """Return full message history for this chat (for backup)."""

    def load_history(self, chat_id: int, messages: list[dict]) -> None:
        """Restore message history after a leader handoff."""
```

`get_history` and `load_history` may be no-ops for small models that do
not maintain history - `main.py` checks `hasattr` before calling them.

## Existing presets

| File                    | Preset tag  | Hardware           |
|-------------------------|-------------|--------------------|
| `run_leader.py`         | `hailo`     | Raspberry Pi+Hailo |
| `jetson_leader.py`      | `jetson`    | NVIDIA Jetson      |
| `nuc_leader.py`         | `nuc`       | Intel NUC          |
| `generic_leader.py`     | `null`      | Any (llama.cpp)    |

To add a new preset:
1. Create `LeaderLogic/yourdevice_leader.py` implementing the contract above.
2. Add `("yourdevice", "LeaderLogic.yourdevice_leader")` to `PRESET_OPTIONS`
   in `ElectionLogic/identity.py`.
3. Run `main.py` on the device and select the new preset at first-run setup.