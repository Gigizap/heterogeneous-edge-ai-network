# Sensing preset contract

Every sensing preset is a folder under `SensingLogic/<preset>/` holding a `tool_config.json` plus the Python files implementing its skills - the sensing-side mirror of a `LeaderLogic/<preset>/` folder holding a `leader.py`. Presets are discovered automatically at first-run setup (`ElectionLogic.identity.discover_sensing_presets()`), so dropping in a new folder makes it selectable with no code change elsewhere.

`SensingLogic/sensing_agent.py` is the shared entry point: it reads `tool_config.json`, imports the skill functions, and answers the leader's JSON-RPC `tools/list` and `tools/call` requests. A preset supplies only the tool definitions and the functions behind them.

## What a preset folder needs

- **`tool_config.json`** with two arrays:
  - `skills` - how to reach each function: `{"file": <module, no .py>, "function": <function name>, "command": <tool name>}`. `command` is what the leader calls; `file` is imported as `SensingLogic.<preset>.<file>`.
  - `tool_defs` - the OpenAI-style function definitions sent to the leader's LLM in the `tools/list` reply. The `name` in a definition must match a skill's `command`.
- **the skill modules** themselves, in the same folder.
- **`requirements.txt`** with the pip-installable dependencies, installed automatically at first-run setup. Runtimes that are not pip packages (`stai_mpu`, `tflite_runtime`, `picamera2`, the Hailo SDK) stay commented there with the real `apt` / `x-linux-ai` command.
- **the models**, next to the skill that loads them, resolved as `Path(__file__).parent / "<model>"`.

Skills are discovered dynamically through `tools/list`, so adding one needs no change on the leader.

## Skill functions

A skill function receives the tool call's `arguments` as keyword arguments and returns a value; whatever it returns is stringified into the `tools/call` result.

```json
{
  "skills": [
    { "file": "people_counter", "function": "detect_people_number", "command": "detect_people_number" }
  ],
  "tool_defs": [
    {
      "type": "function",
      "function": {
        "name": "detect_people_number",
        "description": "detects the people in the meeting room and returns how many people are present.",
        "parameters": { "type": "object", "properties": {}, "required": [] }
      }
    }
  ]
}
```

A skill marked `"threaded": true` is long-running: the `tools/call` result is an immediate acknowledgement, and the function is started on its own thread with `(notify, stop_event)` as its first two arguments plus the call's `arguments` as keywords. It reports later hits by calling `notify(text)`, which sends a `notifications/sensing/alert`. Starting a threaded skill stops the previous one. **Leader-side handling of those alerts is not wired yet**, so they currently go nowhere.

## Available presets

| Preset folder | Commands | Hardware |
|---|---|---|
| `meeting_room_stm` | `detect_people_number` | STM32MP257F-DK (NPU) |
| `meeting_room_raspberry` | `detect_people_number` | Raspberry Pi 5 (CPU/ONNX) |
| `stm32mp257_yolo_NPU` | `person_detection` | STM32MP257F-DK (NPU) |
| `stm32mp257_yolo_CPU` | `person_detection`, `await_person` | STM32MP257F-DK (CPU/TFLite) |
| `raspberrypi5_yolo_NPU` | `object_detection` | Raspberry Pi 5 + Hailo |
| `raspberrypi5_yolo_CPU` | `object_detection` | Raspberry Pi 5 (CPU/ONNX) |
| `mock_test` | `people_detection`, `measure_co2`, `identify_face` | Any device, no hardware |
| `Legacy_stm32mp257fdk` | `recognize`, `add_face`, `remove_face`, `list_faces`, `detect_objects_now`, `run_till_detect` | STM32MP257F-DK, superseded |

See [`../LeaderLogic/README.md`](../LeaderLogic/README.md) for the leader-side contract, and the protocol summary in [`../ConnectionLogic/README.md`](../ConnectionLogic/README.md) for the exact message shapes.
