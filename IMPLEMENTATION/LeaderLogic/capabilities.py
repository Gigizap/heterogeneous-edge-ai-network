"""
LeaderLogic/capabilities.py

Capability broadcast
─────────────────────
After a leader is elected (or after a handoff) the new leader calls
`announce(...)` which:
  1. Builds a human-readable capability summary.
  2. Sends it to all known Telegram chat_ids (active + new joiners).
  3. Broadcasts a structured CAPABILITY_ANNOUNCE on the P2P network so that in the future
     peer devices also can know what the leader can do (not implemented management from peer devices w.r.t. capabilities).

The TelegramBot keeps a set of known chat_ids;

CAPABILITY_ANNOUNCE P2P message schema
───────────────────────────────────────
{
  "type":            "CAPABILITY_ANNOUNCE",
  "from":            <agent_id>,
  "leader_id":       <agent_id>,
  "preset":          "generic" | "raspberry_cpu" | "raspberry_hailo" | "raspberry_cpuhailo" | "stm32mp257fdk" | null,
  "model":           "qwen2.5-coder:1.5b",
  "history_support": true,
  "hardware":        {ram_gb, cpu_cores, has_gpu, has_hailo, has_npu},
  "score":           215,
  "connected_peers": 3,
  "commands":        ["/status"],
  "ts":              1234567890.0
}
"""

import time
import logging

log = logging.getLogger(__name__)


# ── presets → known commands ──────────────────────────────────────────────────

_DEFAULT_COMMANDS = ["/status"]

# Models small enough to lack history (parameter count threshold, in billions)
_HISTORY_THRESHOLD_B = 7.0


# ── public API ────────────────────────────────────────────────────────────────

def announce(
    *,
    agent_id: str,
    profile: dict,
    model_name: str,
    model_params_b: float,          # estimated parameter count in billions
    hardware: dict,
    peer_count: int,
    transport,
    bot,                            # TelegramBot instance
    known_chat_ids: set[int],
):
    """
    Build and send a capability announcement to:
      - all known Telegram chats
      - all P2P peers (structured payload)
    """
    preset    = profile.get("leader_preset")
    commands  = _DEFAULT_COMMANDS
    has_hist  = model_params_b >= _HISTORY_THRESHOLD_B

    telegram_text = _build_telegram_text(
        agent_id    = agent_id,
        preset      = preset,
        model_name  = model_name,
        has_hist    = has_hist,
        hardware    = hardware,
        peer_count  = peer_count,
        commands    = commands,
        score       = profile.get("score", "?"),
    )

    # Send to every chat we know about
    for chat_id in known_chat_ids:
        try:
            bot.send_message(chat_id, telegram_text)
        except Exception as e:
            log.warning("could not notify chat %s: %s", chat_id, e)

    # P2P structured broadcast
    transport.broadcast_sync({
        "type":            "CAPABILITY_ANNOUNCE",
        "from":            agent_id,
        "leader_id":       agent_id,
        "preset":          preset,
        "model":           model_name,
        "history_support": has_hist,
        "hardware":        hardware,
        "score":           profile.get("score"),
        "connected_peers": peer_count,
        "commands":        commands,
        "ts":              time.time(),
    })

    log.info("announced as leader — model=%s hist=%s", model_name, has_hist)


def status_text(
    *,
    agent_id: str,
    profile: dict,
    model_name: str,
    model_params_b: float,
    hardware: dict,
    peer_count: int,
) -> str:
    """Build the reply for the `/status` command (current model + capabilities)."""
    preset   = profile.get("leader_preset")
    commands = _DEFAULT_COMMANDS
    has_hist = model_params_b >= _HISTORY_THRESHOLD_B
    return f"[SYSTEM MESSAGE]\n*Status* — `{agent_id}`\n\n" + _capability_lines(
        preset      = preset,
        model_name  = model_name,
        has_hist    = has_hist,
        hardware    = hardware,
        peer_count  = peer_count,
        commands    = commands,
        score       = profile.get("score", "?"),
    )


def resume_notice(
    *,
    agent_id: str,
    profile: dict,
    model_name: str,
    model_params_b: float,
    hardware: dict,
    peer_count: int,
) -> str:
    """Message for a user whose reply was interrupted by a leader failover:
    a heads-up that a new leader is resuming, followed by its capabilities."""
    preset   = profile.get("leader_preset")
    commands = _DEFAULT_COMMANDS
    has_hist = model_params_b >= _HISTORY_THRESHOLD_B
    return (
        "The previous leader disconnected. The new leader is resuming your "
        "reply — this may take a while.\n\n"
        "*Current leader capabilities:*\n"
        + _capability_lines(
            preset      = preset,
            model_name  = model_name,
            has_hist    = has_hist,
            hardware    = hardware,
            peer_count  = peer_count,
            commands    = commands,
            score       = profile.get("score", "?"),
        )
    )


def build_new_chat_greeting(cap: dict) -> str:
    """
    Given a stored CAPABILITY_ANNOUNCE dict, produce the greeting message
    sent to a user who opens a brand-new chat.
    """
    preset   = cap.get("preset") or "generic"
    model    = cap.get("model", "unknown")
    hw       = cap.get("hardware", {})
    peers    = cap.get("connected_peers", 0)
    hist     = cap.get("history_support", False)
    commands = cap.get("commands", [])

    return (
        f"Hello! I'm *{cap.get('leader_id', 'leader')}* "
        f"({preset} leader).\n\n"
        f"Model: `{model}`\n"
        f"RAM: {hw.get('ram_gb', '?')} GB  |  "
        f"Cores: {hw.get('cpu_cores', '?')}"
        + ("  |  GPU" if hw.get("has_gpu") else "")
        + ("  |  Hailo" if hw.get("has_hailo") else "")
        + f"\nPeers: {peers}\n"
        f"Conversation history: {'yes' if hist else 'no (single-turn only)'}\n"
        + (f"\nCommands: {', '.join(commands)}" if commands else "")
    )


# ── internal ──────────────────────────────────────────────────────────────────

def _build_telegram_text(
    *,
    agent_id: str,
    preset: str | None,
    model_name: str,
    has_hist: bool,
    hardware: dict,
    peer_count: int,
    commands: list[str],
    score,
) -> str:
    return f"*New leader elected*: `{agent_id}`\n\n" + _capability_lines(
        preset      = preset,
        model_name  = model_name,
        has_hist    = has_hist,
        hardware    = hardware,
        peer_count  = peer_count,
        commands    = commands,
        score       = score,
    )


def _capability_lines(
    *,
    preset: str | None,
    model_name: str,
    has_hist: bool,
    hardware: dict,
    peer_count: int,
    commands: list[str],
    score,
) -> str:
    preset_label = preset if preset else "generic"
    hw_line = (
        f"Hardware: {hardware.get('ram_gb', '?')} GB RAM  |  "
        f"{hardware.get('cpu_cores', '?')} cores"
        + ("  |  GPU" if hardware.get("has_gpu") else "")
        + ("  |  Hailo" if hardware.get("has_hailo") else "")
    )
    hist_line = "full history" if has_hist else "single-turn (model < 7B)"
    cmd_line  = ", ".join(commands) if commands else "none"

    return (
        f"Preset  : {preset_label}\n"
        f"Model   : `{model_name}`\n"
        f"{hw_line}\n"
        f"Score   : {score}\n"
        f"Peers   : {peer_count} connected\n"
        f"History : {hist_line}\n"
        f"Commands: {cmd_line}"
    )