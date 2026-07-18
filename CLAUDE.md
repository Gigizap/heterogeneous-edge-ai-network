# CLAUDE.md - Specifications to Always Follow

This file is loaded into context at the start of every session. It is the single source of truth for how to work in this repository. Read it, follow it, and keep it up to date. When a rule here conflicts with a habit, the rule wins.

> **Maintainer note:** add/adjust rules below as the project evolves. Keep each rule short and concrete. If a rule stops being true, fix it here in the same change.

---

## 1. What this project is

A distributed **agentic system**: a fleet of heterogeneous embedded devices (Raspberry Pi 5 + Hailo, STM32MP257F-DK, generic Linux or Windows) collaborate over a zero-config P2P network with **no cloud dependency**. A single elected **leader agent** runs an on-device LLM, handles the user messages over Telegram, and dispatches hardware-accelerated **skills** (object/person/face detection, camera capture) to **sensing agents**, then summarizes the results back to the user.

- Every device runs the **same code** (`IMPLEMENTATION/`). Behavior is decided by `device_profile.json` + presets, **not** by a different program per device.
- The preset chosen (both leader and sensing) decides the portion of the code that is executed on each device.
- There is exactly **one** leader at any time, chosen by score-based election. Leader and sensing agents can co-live on one device as two independent agents (own agent-ID, own TCP port).
- Only **Approach 1** (full-LLM dispatch + answer) is implemented. Approach 2 (classifier/cross-encoder dispatch on sensing agents) is **not** - don't assume it works.

The README files are the architecture spec. Start with [`README.md`](README.md) → [`IMPLEMENTATION/README.md`](IMPLEMENTATION/README.md) → the per-folder READMEs.

## 2. Golden rules

1. **Docs live next to the code they describe.** Each folder has its own `README.md` (or `*.md`). When you change behavior, update the README in the *same* folder in the *same* change. Do not let docs drift.
2. **Keep the wire protocol self-consistent.** Message shapes (discovery/election/backup/operational planes) are documented in [`IMPLEMENTATION/README.md`](IMPLEMENTATION/README.md#communication) and [`ConnectionLogic/README.md`](IMPLEMENTATION/ConnectionLogic/README.md). You may change these shapes freely: this repo is the single source of truth and every node is later redeployed from it, so there is never a live mix of old and new nodes. Do NOT preserve backward or version intercompatibility, it is not required at all. The only constraint: a shape change must land on every role at once (sender and receiver, all presets) and update the protocol docs in the same change, so the fleet still agrees with itself.
3. **One code path, many devices.** Never hard-code device-specific behavior in shared code. Branch on the device profile / preset / score, not on hostnames or ports.
4. **Verify before claiming done.** If you can't run it on hardware, say what you did and did not verify. Report test failures with their output.
5. **No long dashes.** Never use em dashes or en dashes (the long Unicode dashes) anywhere - in code, docs, comments, commit messages, or chat. Use a normal hyphen `-`, comma, colon, or parentheses instead.
6. **Plan and ask before touching code.** Before editing any code, give the user a short plan: which files and functions you will change, and what each change does. Wait for their go-ahead. Do not start editing until they approve.

## 3. Coding discipline

1. **Put code where it belongs.** Do not add functions to modules they don't belong in - respect the existing module boundaries.
2. **No useless or undesired fallbacks.** Don't add fallback paths unless they were asked for.
3. **Bare-minimum skeleton first.** Implement only what the architecture needs to work - nothing speculative.
4. **No useless redundancy.** Don't duplicate logic; reuse what already exists.
5. **Log new code.** Add logging for anything new you add (see §4).

## 4. Logging - never use `print()`

The codebase migrated from prints to **tagged logging** (see [`main.py`](IMPLEMENTATION/main.py): `_FMT`, the `[%(agent)s]` tag, and the per-module loggers `log` / `leaderlog`). Follow it:

- Use `logging.getLogger(__name__)` (or the existing module logger), **not** `print()`.
- Preserve the agent-context tag: leader-context messages use the leader logger so they show `[LEADER_ID]`.
- Pick the right level: `DEBUG` for raw model I/O / wire dumps, `INFO` for lifecycle events, `WARNING`/`ERROR` for problems. Default log level is `DEBUG`.

## 5. Configuration & models - where things go

- **`device_profile.json`** - per-device identity (holds the **agent-ID**) and hardware preset. This is what makes two copies behave differently.
- **`software_config.json`** - static config shared across roles (TCP port, Telegram token + allowed users, timeouts). The agent-ID is **not** here.
- **`LeaderLogic/raspberry_config.json`** - Pi+Hailo leader inference config (CPU Ollama hosts, `.hef` path).
- **`SensingLogic/<preset>/tool_config.json`** - a sensing device's exposed tools (`tool_defs` for the LLM, `skills` mapping tool→file/function).
- **Leader LLM models** → top-level `IMPLEMENTATION/models/` (GGUF / `.hef`).
- **Sensing models** → inside the sensing preset's own folder, resolved as `Path(__file__).parent / "<model>"`. Drop a new sensing model next to the skill that loads it.

Skills are discovered dynamically via `fetchskills` - adding a skill to a sensing preset's `tool_config.json` needs **no** leader change.

## 6. Dependencies & roles

- There is **no single requirements file**. Deps depend on role: `requirements/base.txt` for everyone; `generic.txt` / `hailo.txt` / `stm32mp257fdk.txt` for the leader backend; `SensingLogic/<preset>/requirements.txt` for sensing.
- Hardware runtimes (`hailo_platform`, `stai_mpu`, `tflite_runtime`, `picamera2`, STM32 `llama-cpp-python`) are **not pip packages** - they're installed via `apt` / `x-linux-ai` / Hailo SDK / cross-compile. Don't add them to a pip requirements file; document the real command in comments instead.

## 7. Environment & workflow

- The target devices are Linux and Windows. Keep paths portable; don't bake in Windows-only assumptions in code that runs on the boards.
- Python is run as `python main.py` (the `.pyc` files target CPython 3.11).
- `__pycache__/*.pyc` files should not be hand-edited and ideally shouldn't be tracked - don't treat their churn as meaningful changes.
- Git: default working branch is `developer`; PRs target `master`. Commit/push only when asked. End commit messages with the required Co-Authored-By trailer.