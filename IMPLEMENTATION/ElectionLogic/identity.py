"""
ElectionLogic/identity.py

Handles first-run identity setup:
  - Prompts user for an agent name (e.g. "camera-corridor")
  - Scans the network for existing agent IDs to avoid conflicts
  - Asks whether this device has sensing skills and a leader preset
  - Writes device_profile.json so it is never asked again

device_profile.json schema:
{
    "agent_id":          "camera-corridor",
    "sensing_preset":     "stm32mp257fdk",   // folder under SensingLogic/, or null (no sensing)
    "leader_preset":     "generic",         // folder under LeaderLogic/ ("raspberry_cpu" |
                                             // "raspberry_hailo" | "raspberry_cpuhailo" |
                                             // "stm32mp257fdk" | "generic"), or "mock" | "none"
    "cpu_cores":         8,
    "cpu_tflops":        0.5,               // user-declared peak CPU compute (TFLOPS)
    "ram_bandwidth_gbs": 25.6,              // user-declared memory bandwidth (GB/s)
    "accelerator":       "none",            // "none" | "gpu" | "hailo" | "npu"
    "score":             0,                 // filled in at runtime by scoring.py
    "first_run_done":    true
}
"""

import json
import socket
from pathlib import Path
from utils import available_cpu_cores

PROFILE_PATH  = Path(__file__).parent.parent / "device_profile.json"
SENSING_DIR   = Path(__file__).parent.parent / "SensingLogic"
LEADER_DIR    = Path(__file__).parent.parent / "LeaderLogic"

# A leader preset is any folder under LeaderLogic/ holding a leader.py - the
# tag IS the folder name, mirroring SensingLogic/<preset>/tool_config.json.
# Discovered dynamically (see discover_leader_presets()), same as sensing.
_DEFAULT_LEADER_PRESET = "generic"   # safe default: works on any device


# ── public API ────────────────────────────────────────────────────────────────

def load_or_create_profile(profile_path: Path = None) -> dict:
    """
    Return the device profile, running first-run setup if needed.

    On first run, calls each _prompt_* function to collect user choices,
    then writes device_profile.json. On subsequent runs, just reads and
    returns the saved profile.

    `profile_path` — override the default path (used when --port is set so
                     two agents on the same machine keep separate profiles).
    """
    path = profile_path or PROFILE_PATH
    if path.exists():
        profile = json.loads(path.read_text())
        if profile.get("first_run_done"):
            return profile

    print("\n" + "═" * 52)
    print("  First-run setup")
    print("═" * 52)

    agent_id        = _prompt_agent_id()
    sensing_preset   = _prompt_sensing_preset()
    preset          = _prompt_leader_preset()
    cpu_cores       = available_cpu_cores()
    cpu_tflops      = _prompt_cpu_tflops()
    ram_bandwidth   = _prompt_ram_bandwidth()
    accelerator     = _prompt_accelerator()

    profile = {
        "agent_id":          agent_id,
        "sensing_preset":     sensing_preset,
        "leader_preset":     preset,
        "cpu_cores":         cpu_cores,
        "cpu_tflops":        cpu_tflops,
        "ram_bandwidth_gbs": ram_bandwidth,
        "accelerator":       accelerator,
        "score":             0,
        "first_run_done":    True,
    }

    path.write_text(json.dumps(profile, indent=2))
    print(f"\n[identity] Profile saved → {path}")

    _install_dependencies(preset, sensing_preset)

    return profile


def resolve_id_conflict(desired_id: str, taken_ids: set[str]) -> str:
    """
    Check desired_id against a set of taken names.
    If taken, appends -2, -3, … until unique.

    Called by _prompt_agent_id() with the set returned from
    ConnectionLogic.discovery.collect_peer_ids().
    """
    if desired_id not in taken_ids:
        return desired_id

    suffix = 2
    while f"{desired_id}-{suffix}" in taken_ids:
        suffix += 1
    resolved = f"{desired_id}-{suffix}"
    print(f"[identity] '{desired_id}' already on network → using '{resolved}'")
    return resolved


def get_leader_module(profile: dict) -> str | None:
    """
    Return the dotted module path for this device's leader implementation,
    or None if the device has no leader capability (preset == "none").

    Called by main.py at startup to determine if this device can lead,
    and again in run_leader.boot() to import the correct leader module.
    The tag IS the LeaderLogic/ folder name (see discover_leader_presets()).
    """
    preset = profile.get("leader_preset")
    if not preset or preset == "none":
        return None
    return f"LeaderLogic.{preset}.leader"


def discover_leader_presets() -> list[str]:
    """
    Return the available leader presets: every sub-folder of LeaderLogic/
    that holds a leader.py. Each such folder *is* a leader preset, so dropping
    in a new one makes it selectable automatically with no code change - the
    leader-side mirror of discover_sensing_presets().
    """
    presets = []
    if not LEADER_DIR.is_dir():
        return presets
    for child in sorted(LEADER_DIR.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "leader.py").exists():
            continue
        presets.append(child.name)
    return presets


# ── prompts ───────────────────────────────────────────────────────────────────

def _prompt_agent_id() -> str:
    """
    Ask the user to name this device.

    Calls ConnectionLogic.discovery.collect_peer_ids() to scan the network
    for 6 seconds and find existing agent names. Then calls
    resolve_id_conflict() to auto-rename if the chosen name is taken.
    """
    from ConnectionLogic.discovery import collect_peer_ids

    print("\n  Listening for existing agents (6s) …")
    taken = collect_peer_ids(window=6.0)
    if taken:
        print(f"  Found on network: {', '.join(sorted(taken))}")
    else:
        print("  No agents found on network.")

    hostname_default = socket.gethostname().lower().replace(" ", "-")

    while True:
        raw = input(
            f"\nAgent name for this device (e.g. camera-corridor) [{hostname_default}]: "
        ).strip()
        if not raw:
            raw = hostname_default

        sanitised = "".join(
            c if c.isalnum() or c == "-" else "-" for c in raw.lower()
        ).strip("-")

        if not sanitised:
            print("  Name cannot be empty, try again.")
            continue

        agent_id = resolve_id_conflict(sanitised, taken)
        print(f"  Agent ID will be: {agent_id}")
        confirm = input("  Confirm? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            return agent_id


def discover_sensing_presets() -> list[str]:
    """
    Return the available sensing-agent presets: every sub-folder of
    SensingLogic/ that contains a valid tool_config.json. Each such folder
    *is* a sensing preset, so dropping in a new folder makes it selectable
    automatically with no code change.
    """
    presets = []
    if not SENSING_DIR.is_dir():
        return presets
    for child in sorted(SENSING_DIR.iterdir()):
        if not child.is_dir():
            continue
        config = child / "tool_config.json"
        if not config.exists():
            continue
        try:
            json.loads(config.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # broken config — don't offer it
        presets.append(child.name)
    return presets


def _prompt_sensing_preset() -> str | None:
    """
    Ask what kind of sensing agent this device is.

    The menu is built from the folders actually present in SensingLogic/
    (those with a valid tool_config.json), plus a "None" option for a
    leader-only / no-sensing device. Returns the folder name or None.
    """
    presets = discover_sensing_presets()

    print("\nSensing-agent preset on this device:")
    for i, name in enumerate(presets, start=1):
        print(f"  [{i}] {name}")
    none_choice = len(presets) + 1
    print(f"  [{none_choice}] None  (leader-only / no sensing skills)")

    if not presets:
        print("  (no valid sensing presets found under SensingLogic/ — defaulting to None)")
        return None

    default = str(none_choice)
    while True:
        choice = input(f"  Choice [{default}]: ").strip() or default
        if choice == str(none_choice):
            print("  No sensing skills will run on this device.")
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(presets):
            tag = presets[int(choice) - 1]
            print(f"  Sensing preset: {tag}")
            return tag
        print(f"  Please enter a number between 1 and {none_choice}.")


def _prompt_leader_preset() -> str:
    """
    Ask what leader capability this device has.

    The menu is built from the folders actually present in LeaderLogic/
    (those holding a leader.py), plus a "none" option for a sensing-only /
    no-leader device. Returns the folder name (tag) or "none".
    """
    presets = discover_leader_presets()

    print("\nLeader capability of this device:")
    for i, name in enumerate(presets, start=1):
        tag = "  <- safe default" if name == _DEFAULT_LEADER_PRESET else ""
        print(f"  [{i}] {name}{tag}")
    none_choice = len(presets) + 1
    print(f"  [{none_choice}] none  (follower only, no leader capability)")

    if not presets:
        print("  (no valid leader presets found under LeaderLogic/ — defaulting to none)")
        return "none"

    default = (str(presets.index(_DEFAULT_LEADER_PRESET) + 1)
               if _DEFAULT_LEADER_PRESET in presets else str(none_choice))
    while True:
        choice = input(f"  Choice [{default}]: ").strip() or default
        if choice == str(none_choice):
            print("  No leader capability on this device.")
            return "none"
        if choice.isdigit() and 1 <= int(choice) <= len(presets):
            preset = presets[int(choice) - 1]
            print(f"  Leader preset: {preset}")
            return preset
        print(f"  Please enter a number between 1 and {none_choice}.")
        print(f"  Please enter one of: {valid}.")


def _prompt_cpu_tflops() -> float:
    """
    Ask the user for this device's peak CPU compute in TFLOPS.
    Used by scoring.py (no benchmark is run). Returns 0.0 if unknown.
    """
    print("\nPeak CPU compute of this device, in TFLOPS (e.g. 0.5).")
    print("  Leave blank if unknown (counts as weakest for leadership).")
    while True:
        raw = input("  CPU TFLOPS [0]: ").strip()
        if not raw:
            return 0.0
        try:
            val = float(raw)
            if val < 0:
                raise ValueError
            return val
        except ValueError:
            print("  Enter a non-negative number (e.g. 0.5), or blank.")


def _prompt_ram_bandwidth() -> float:
    """
    Ask the user for this device's memory bandwidth in GB/s (e.g. 25.6).
    Used by scoring.py. Returns 0.0 if unknown.
    """
    print("\nMemory (RAM) bandwidth of this device, in GB/s (e.g. 25.6).")
    print("  Leave blank if unknown (counts as weakest for leadership).")
    while True:
        raw = input("  RAM bandwidth GB/s [0]: ").strip()
        if not raw:
            return 0.0
        try:
            val = float(raw)
            if val < 0:
                raise ValueError
            return val
        except ValueError:
            print("  Enter a non-negative number (e.g. 25.6), or blank.")


def _prompt_accelerator() -> str:
    """
    Ask if this device has a hardware accelerator.
    Returns one of: "none", "gpu", "hailo", "npu".
    """
    print("\nDoes this device have a hardware accelerator?")
    print("  [1] None")
    print("  [2] GPU  (NVIDIA / AMD)")
    print("  [3] Hailo NPU")
    print("  [4] Other NPU  (Coral, OpenVINO, …)")
    mapping = {"1": "none", "2": "gpu", "3": "hailo", "4": "npu"}
    while True:
        choice = input("  Choice [1]: ").strip() or "1"
        if choice in mapping:
            return mapping[choice]
        print("  Please enter 1, 2, 3, or 4.")


def _install_dependencies(leader_preset: str | None, sensing_preset: str | None):
    """
    Pip-install the requirements for this device's role(s):
      - requirements/base.txt                          (always)
      - LeaderLogic/<leader_preset>/requirements.txt    (if it can lead)
      - SensingLogic/<sensing_preset>/requirements.txt  (if it senses)

    Requirements files list only pip-installable packages; hardware runtimes
    that must come from apt / X-LINUX-AI / the Hailo SDK are left as comments,
    so pip skips them. A reminder is printed so those manual steps aren't missed.
    """
    import subprocess
    import sys

    root  = Path(__file__).parent.parent
    files = [root / "requirements" / "base.txt"]

    # "mock" is the no-LLM test leader: it needs nothing beyond base.txt.
    if leader_preset and leader_preset not in ("none", "mock"):
        files.append(LEADER_DIR / leader_preset / "requirements.txt")

    if sensing_preset:
        files.append(SENSING_DIR / sensing_preset / "requirements.txt")

    for req_path in files:
        if not req_path.exists():
            print(f"[setup] {req_path.name} not found — skipping")
            continue
        lines = [l.strip() for l in req_path.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        if not lines:
            print(f"[setup] {req_path.name}: no pip packages "
                  f"(see the file for manual install notes) — skipping")
            continue
        print(f"\n[setup] installing {req_path.name} …")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
            )
            print(f"[setup] {req_path.name} done")
        except subprocess.CalledProcessError as e:
            print(f"[setup] install failed for {req_path.name}: {e}")
            print("[setup] you may need to install some packages manually — see the file for notes")

    print("\n[setup] NOTE: some edge runtimes are NOT pip-installable "
          "(e.g. hailo_platform, stai_mpu, tflite_runtime, picamera2, llama-cpp-python on the STM32).\n"
          "        Check the commented lines in the requirements files above for the "
          "apt / x-linux-ai / Hailo SDK / cross-compile steps.")