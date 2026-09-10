"""
ElectionLogic/identity.py

Handles first-run identity setup:
  - Prompts user for an agent name (e.g. "camera-corridor")
  - Scans the network for existing agent IDs to avoid conflicts
  - Asks whether this device has sensing skills and a leader preset
  - Asks for the Telegram bot token + allowed user IDs, but ONLY when the
    device is leader-capable (a follower-only device never runs the bot).
    These land in software_config.json, not in device_profile.json, because
    that is where main.py reads them from.
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
from utils import (available_cpu_cores,
                   BOLD, DIM, CYAN, GREEN, YELLOW, RED, RESET)

PROFILE_PATH  = Path(__file__).parent.parent / "device_profile.json"
SENSING_DIR   = Path(__file__).parent.parent / "SensingLogic"
LEADER_DIR    = Path(__file__).parent.parent / "LeaderLogic"
SW_CONFIG_PATH = Path(__file__).parent.parent / "software_config.json"

# A leader preset is any folder under LeaderLogic/ holding a leader.py - the
# tag IS the folder name, mirroring SensingLogic/<preset>/tool_config.json.
# Discovered dynamically (see discover_leader_presets()), same as sensing.
_DEFAULT_LEADER_PRESET = "generic"   # safe default: works on any device


# -- public API ----------------------------------------------------------------

def _print_banner():
    """
    Startup splash for the first-run setup: a house over a camera (the fleet
    watches a building) next to the HOUSE AGENT wordmark.

    One fixed layout, plain ASCII inside 40 columns, so it renders identically
    on a dev laptop and on the STM32MP257F-DK's small panel. Only the colour
    varies: the constants collapse to "" when stdout is not a TTY, so log files
    stay free of escape codes (see utils).
    """
    art = [
        "    /\\    ",
        "   /  \   ",
        "  /____\  ",
        "  | [] |  ",
        "  |_||_|  ",
        "          ",
        "  ______  ",
        " |  __  | ",
        " | (__) | ",
        " |______| ",
        "          ",
    ]
    text = [
        "#  # #### #  # #### ####",
        "#  # #  # #  # #    #   ",
        "#### #  # #  # #### ### ",
        "#  # #  # #  #    # #   ",
        "#  # #### #### #### ####",
        "                        ",
        "#### #### #### #  # ####",
        "#  # #    #    ## #  #  ",
        "#### # ## ###  # ##  #  ",
        "#  # #  # #    #  #  #  ",
        "#  # #### #### #  #  #  ",
    ]
    print()
    for left, right in zip(art, text):
        print(f"  {CYAN}{left}{RESET}  {BOLD}{right}{RESET}")
    print()
    print(f"  {DIM}Heterogeneous edge AI network{RESET}")
    print(f"  {DIM}First-run setup{RESET}")
    print()
    print(f"  {DIM}No profile on this device yet.{RESET}")
    print(f"  {DIM}A few questions, then it joins the fleet.{RESET}")


def load_or_create_profile(profile_path: Path = None) -> dict:
    """
    Return the device profile, running first-run setup if needed.

    On first run, calls each _prompt_* function to collect user choices,
    then writes device_profile.json. On subsequent runs, just reads and
    returns the saved profile.

    `profile_path` - override the default path (used when --port is set so
                     two agents on the same machine keep separate profiles).
    """
    path = profile_path or PROFILE_PATH
    if path.exists():
        profile = json.loads(path.read_text())
        if profile.get("first_run_done"):
            return profile

    _print_banner()

    agent_id        = _prompt_agent_id()
    sensing_preset   = _prompt_sensing_preset()
    preset          = _prompt_leader_preset()
    cpu_cores       = available_cpu_cores()
    cpu_tflops      = _prompt_cpu_tflops()
    ram_bandwidth   = _prompt_ram_bandwidth()
    accelerator     = _prompt_accelerator()

    # Telegram credentials are only meaningful on a device that can be elected
    # leader - the leader is the only role that runs the bot.
    if preset != "none":
        token, allowed_users = _prompt_telegram()
        _save_telegram_credentials(token, allowed_users)

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
    print(f"\n  {GREEN}Profile saved -> {path}{RESET}")

    _install_dependencies(preset, sensing_preset)

    return profile


def resolve_id_conflict(desired_id: str, taken_ids: set[str]) -> str:
    """
    Check desired_id against a set of taken names.
    If taken, appends -2, -3, ... until unique.

    Called by _prompt_agent_id() with the set returned from
    ConnectionLogic.discovery.collect_peer_ids().
    """
    if desired_id not in taken_ids:
        return desired_id

    suffix = 2
    while f"{desired_id}-{suffix}" in taken_ids:
        suffix += 1
    resolved = f"{desired_id}-{suffix}"
    print(f"  {YELLOW}'{desired_id}' already on network -> using '{resolved}'{RESET}")
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


# -- prompts -------------------------------------------------------------------

def _prompt_agent_id() -> str:
    """
    Ask the user to name this device.

    Calls ConnectionLogic.discovery.collect_peer_ids() to scan the network
    for 6 seconds and find existing agent names. Then calls
    resolve_id_conflict() to auto-rename if the chosen name is taken.
    """
    from ConnectionLogic.discovery import collect_peer_ids

    print(f"\n  {DIM}Listening for existing agents (6s) ...{RESET}")
    taken = collect_peer_ids(window=6.0)
    if taken:
        print(f"  {DIM}Found on network: {', '.join(sorted(taken))}{RESET}")
    else:
        print(f"  {DIM}No agents found on network.{RESET}")

    hostname_default = socket.gethostname().lower().replace(" ", "-")

    while True:
        print(f"\n{BOLD}Agent name for this device{RESET} {DIM}(e.g. camera-corridor){RESET}")
        raw = input(
            f"  {GREEN}>{RESET} Name {DIM}[{hostname_default}]{RESET}: "
        ).strip()
        if not raw:
            raw = hostname_default

        sanitised = "".join(
            c if c.isalnum() or c == "-" else "-" for c in raw.lower()
        ).strip("-")

        if not sanitised:
            print(f"  {RED}Name cannot be empty, try again.{RESET}")
            continue

        agent_id = resolve_id_conflict(sanitised, taken)
        print(f"  {GREEN}Agent ID will be: {agent_id}{RESET}")
        confirm = input(f"  {GREEN}>{RESET} Confirm? {DIM}[Y/n]{RESET}: ").strip().lower()
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
            continue  # broken config - don't offer it
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

    print(f"\n{BOLD}Sensing-agent preset on this device:{RESET}")
    for i, name in enumerate(presets, start=1):
        print(f"  {CYAN}[{i}]{RESET} {name}")
    none_choice = len(presets) + 1
    print(f"  {CYAN}[{none_choice}]{RESET} None  {DIM}(leader-only / no sensing skills){RESET}")

    if not presets:
        print(f"  {YELLOW}(no valid sensing presets found under SensingLogic/ - defaulting to None){RESET}")
        return None

    default = str(none_choice)
    while True:
        choice = input(f"  {GREEN}>{RESET} Choice {DIM}[{default}]{RESET}: ").strip() or default
        if choice == str(none_choice):
            print(f"  {GREEN}No sensing skills will run on this device.{RESET}")
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(presets):
            tag = presets[int(choice) - 1]
            print(f"  {GREEN}Sensing preset: {tag}{RESET}")
            return tag
        print(f"  {RED}Please enter a number between 1 and {none_choice}.{RESET}")


def _prompt_leader_preset() -> str:
    """
    Ask what leader capability this device has.

    The menu is built from the folders actually present in LeaderLogic/
    (those holding a leader.py), plus a "none" option for a sensing-only /
    no-leader device. Returns the folder name (tag) or "none".
    """
    presets = discover_leader_presets()

    print(f"\n{BOLD}Leader capability of this device:{RESET}")
    for i, name in enumerate(presets, start=1):
        tag = f"{DIM}  <- safe default{RESET}" if name == _DEFAULT_LEADER_PRESET else ""
        print(f"  {CYAN}[{i}]{RESET} {name}{tag}")
    none_choice = len(presets) + 1
    print(f"  {CYAN}[{none_choice}]{RESET} none  {DIM}(follower only, no leader capability){RESET}")

    if not presets:
        print(f"  {YELLOW}(no valid leader presets found under LeaderLogic/ - defaulting to none){RESET}")
        return "none"

    default = (str(presets.index(_DEFAULT_LEADER_PRESET) + 1)
               if _DEFAULT_LEADER_PRESET in presets else str(none_choice))
    while True:
        choice = input(f"  {GREEN}>{RESET} Choice {DIM}[{default}]{RESET}: ").strip() or default
        if choice == str(none_choice):
            print(f"  {GREEN}No leader capability on this device.{RESET}")
            return "none"
        if choice.isdigit() and 1 <= int(choice) <= len(presets):
            preset = presets[int(choice) - 1]
            print(f"  {GREEN}Leader preset: {preset}{RESET}")
            return preset
        print(f"  {RED}Please enter a number between 1 and {none_choice}.{RESET}")


def _prompt_cpu_tflops() -> float:
    """
    Ask the user for this device's peak CPU compute in TFLOPS.
    Used by scoring.py (no benchmark is run). Returns 0.0 if unknown.
    """
    print(f"\n{BOLD}Peak CPU compute of this device, in TFLOPS (e.g. 0.5).{RESET}")
    print(f"  {DIM}Leave blank if unknown (counts as weakest for leadership).{RESET}")
    while True:
        raw = input(f"  {GREEN}>{RESET} CPU TFLOPS {DIM}[0]{RESET}: ").strip()
        if not raw:
            return 0.0
        try:
            val = float(raw)
            if val < 0:
                raise ValueError
            return val
        except ValueError:
            print(f"  {RED}Enter a non-negative number (e.g. 0.5), or blank.{RESET}")


def _prompt_ram_bandwidth() -> float:
    """
    Ask the user for this device's memory bandwidth in GB/s (e.g. 25.6).
    Used by scoring.py. Returns 0.0 if unknown.
    """
    print(f"\n{BOLD}Memory (RAM) bandwidth of this device, in GB/s (e.g. 25.6).{RESET}")
    print(f"  {DIM}Leave blank if unknown (counts as weakest for leadership).{RESET}")
    while True:
        raw = input(f"  {GREEN}>{RESET} RAM bandwidth GB/s {DIM}[0]{RESET}: ").strip()
        if not raw:
            return 0.0
        try:
            val = float(raw)
            if val < 0:
                raise ValueError
            return val
        except ValueError:
            print(f"  {RED}Enter a non-negative number (e.g. 25.6), or blank.{RESET}")


def _prompt_accelerator() -> str:
    """
    Ask if this device has a hardware accelerator.
    Returns one of: "none", "gpu", "hailo", "npu".
    """
    print(f"\n{BOLD}Does this device have a hardware accelerator?{RESET}")
    print(f"  {CYAN}[1]{RESET} None")
    print(f"  {CYAN}[2]{RESET} GPU  {DIM}(NVIDIA / AMD){RESET}")
    print(f"  {CYAN}[3]{RESET} Hailo NPU")
    print(f"  {CYAN}[4]{RESET} Other NPU  {DIM}(Coral, OpenVINO, ...){RESET}")
    mapping = {"1": "none", "2": "gpu", "3": "hailo", "4": "npu"}
    while True:
        choice = input(f"  {GREEN}>{RESET} Choice {DIM}[1]{RESET}: ").strip() or "1"
        if choice in mapping:
            return mapping[choice]
        print(f"  {RED}Please enter 1, 2, 3, or 4.{RESET}")


def _prompt_telegram() -> tuple[str, list[int]]:
    """
    Ask for the Telegram bot token and the user IDs allowed to talk to it.

    Only called for a leader-capable device: the elected leader is the only
    role that starts TelegramBot (see main.py, where these values are read
    back out of software_config.json).

    Returns (token, [user_id, ...]) - both are required, the prompts repeat
    until they are given.
    """
    print(f"\n{BOLD}Telegram bot for this device (it can be elected leader).{RESET}")
    print(f"  {DIM}The leader talks to you over Telegram, so it needs a bot token{RESET}")
    print(f"  {DIM}from @BotFather and the numeric user IDs allowed to use it.{RESET}")

    while True:
        token = input(f"  {GREEN}>{RESET} Bot token: ").strip()
        if token:
            break
        print(f"  {RED}The token cannot be empty - this device can be elected leader.{RESET}")

    print(f"\n  {BOLD}Allowed Telegram user IDs, comma-separated (e.g. 1234567890, 987654321).{RESET}")
    print(f"  {DIM}Get yours from @userinfobot. Anyone not listed is ignored by the bot.{RESET}")
    while True:
        raw = input(f"  {GREEN}>{RESET} Allowed user IDs: ").strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            print(f"  {RED}At least one user ID is required, try again.{RESET}")
            continue
        try:
            users = [int(p) for p in parts]
        except ValueError:
            print(f"  {RED}User IDs are numbers - enter digits separated by commas.{RESET}")
            continue
        print(f"  {GREEN}{len(users)} allowed user(s): {', '.join(str(u) for u in users)}{RESET}")
        return token, users


def _save_telegram_credentials(token: str, allowed_users: list[int]):
    """
    Write the token / allowed_users into the telegram section of
    software_config.json, leaving the rest of that file untouched.

    That file is where main.py reads them from, so it is where they belong -
    but it is also tracked by git, so restore the placeholders before any
    commit (see the repository hygiene notes in the README).
    """
    config = json.loads(SW_CONFIG_PATH.read_text())
    config.setdefault("telegram", {})
    config["telegram"]["token"]         = token
    config["telegram"]["allowed_users"] = allowed_users
    SW_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"  {GREEN}Telegram credentials saved -> {SW_CONFIG_PATH.name}{RESET}")
    print(f"  {YELLOW}NOTE: that file is tracked by git.{RESET}")
    print(f"  {YELLOW}Restore the placeholders before committing.{RESET}")


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
            print(f"  {DIM}{req_path.name} not found - skipping{RESET}")
            continue
        lines = [l.strip() for l in req_path.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        if not lines:
            print(f"  {DIM}{req_path.name}: no pip packages "
                  f"(see the file for manual install notes) - skipping{RESET}")
            continue
        print(f"\n  {CYAN}installing {req_path.name} ...{RESET}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
            )
            print(f"  {GREEN}{req_path.name} done{RESET}")
        except subprocess.CalledProcessError as e:
            print(f"  {RED}install failed for {req_path.name}: {e}{RESET}")
            print(f"  {YELLOW}you may need to install some packages manually - see the file for notes{RESET}")

    print(f"\n  {YELLOW}NOTE: some edge runtimes are NOT pip-installable "
          "(e.g. hailo_platform, stai_mpu, tflite_runtime, picamera2, llama-cpp-python on the STM32).\n"
          "        Check the commented lines in the requirements files above for the "
          f"apt / x-linux-ai / Hailo SDK / cross-compile steps.{RESET}")