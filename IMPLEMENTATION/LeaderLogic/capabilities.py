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

Every user-facing string built here is system-authored, so it is sent with
`bot.send_card()` (parse_mode=HTML) rather than `bot.send_message()`, which stays
reserved for model output. Dynamic values interpolated into a card must go
through `_esc()`. Layout is one bold-labelled fact per line: no space-padded
columns, since Telegram's proportional font makes column widths device-specific.

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

from LeaderLogic.bot_handler import MAIN_KEYBOARD

log = logging.getLogger(__name__)


# ── presets → known commands ──────────────────────────────────────────────────

_DEFAULT_COMMANDS = ["/status", "/capabilities"]

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
            bot.send_card(chat_id, telegram_text, reply_markup=MAIN_KEYBOARD)
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

    log.info("announced as leader - model=%s hist=%s", model_name, has_hist)


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
    return f"⚙️ <b>Status</b>\n<code>{_esc(agent_id)}</code>\n\n" + _capability_lines(
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
        "⚠️ <b>Leader changed</b>\n"
        "The previous leader disconnected. The new leader is resuming your "
        "reply, this may take a while.\n\n"
        "<b>New leader</b>\n"
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


def tools_block(tools: list[tuple[str, list[str]]] | None) -> str:
    """Render the live tool list: one tool per line with the devices that own it.

    `tools` is [(tool_name, [owner_agent_id, ...]), ...] as collected from the
    leader's tool registry; None when the leader preset exposes no registry.
    """
    if not tools:
        return "<i>none reachable right now</i>"
    return "\n".join(
        f"• <code>{_esc(name)}</code>"
        + (f" (on device: {', '.join(_esc(o) for o in owners)})" if owners else "")
        for name, owners in tools
    )


def capabilities_text(tools: list[tuple[str, list[str]]] | None) -> str:
    """The `/capabilities` card: what the fleet can do right now."""
    return (
        "I can reply in natural language to your questions! "
        "My current capabilities are the following:\n\n"
        "<b>Available tools</b>\n" + tools_block(tools)
    )


def welcome_text(
    *,
    agent_id: str,
    profile: dict,
    model_name: str,
    model_params_b: float,
    hardware: dict,
    peer_count: int,
    tools: list[tuple[str, list[str]]] | None = None,
) -> str:
    """The `/start` card: what this fleet is, what it can do right now, and how
    the command buttons behave. `tools` is [(tool_name, [owners]), ...] as
    reachable on the network (None when the preset exposes no tool registry)."""
    has_hist = model_params_b >= _HISTORY_THRESHOLD_B

    return (
        "👋 <b>Heterogeneous Edge AI Network</b>\n"
        "Ask me questions about your house in plain language. A local large "
        "language model is currently running on the leader device. Ask a "
        "question about the available capabilities, I'll pick the right tool "
        "and run it on whichever device in the network owns the hardware "
        "supporting it. Nothing leaves the local network.\n\n"
        "<b>Available tools</b>\n" + tools_block(tools) + "\n\n"
        "<b>This leader</b>\n"
        + _capability_lines(
            preset      = profile.get("leader_preset"),
            model_name  = model_name,
            has_hist    = has_hist,
            hardware    = hardware,
            peer_count  = peer_count,
            commands    = _DEFAULT_COMMANDS,
            score       = profile.get("score", "?"),
        )
        + "\n\n<b>Commands</b>\n"
        "/capabilities - the tools available right now, and where they run\n"
        "/status - live leader, model and peer count\n"
        "/loop - after each answer, keep re-running the tool it used and "
        "message you when the result changes\n"
        "/loop off - stop that\n\n"
        "<i>Use the buttons below the keyboard.</i>"
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
        f"👋 Hello! I'm <b>{_esc(cap.get('leader_id', 'leader'))}</b>, "
        f"the {_esc(preset)} leader of this fleet.\n\n"
        + _capability_lines(
            preset      = preset,
            model_name  = model,
            has_hist    = hist,
            hardware    = hw,
            peer_count  = peers,
            commands    = commands,
            score       = cap.get("score", "?"),
        )
    )


# ── internal ──────────────────────────────────────────────────────────────────

def _esc(v) -> str:
    """Escape a dynamic value (agent id, model name, ...) for the HTML cards.

    Only the three characters Telegram's HTML parse mode treats as markup: an
    unescaped one makes it reject the whole message, so the user would get
    nothing instead of a slightly odd line.
    """
    return (str(v).replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;"))


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
    return f"✅ <b>New leader elected</b>\n<code>{_esc(agent_id)}</code>\n\n" + _capability_lines(
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
    # One fact per line with a bold label, NOT space-padded columns: Telegram
    # renders in a proportional font, so padded columns only line up at one font
    # size on one screen width. `commands` is no longer printed here - the bot
    # exposes them as a keyboard and in the native "/" menu instead.
    accel = []
    if hardware.get("has_gpu"):
        accel.append("GPU")
    if hardware.get("has_hailo"):
        accel.append("Hailo")
    if hardware.get("has_npu"):
        accel.append("NPU")

    return (
        f"🧩 <b>Preset</b>: {_esc(preset if preset else 'generic')}\n"
        f"🧠 <b>Model</b>: <code>{_esc(model_name)}</code>\n"
        f"💾 <b>Memory</b>: "
        + ("full history" if has_hist else "single-turn (model under 7B)")
        + f"\n🖥 <b>Hardware</b>: {_esc(hardware.get('ram_gb', '?'))} GB RAM, "
        f"{_esc(hardware.get('cpu_cores', '?'))} cores"
        + (", " + ", ".join(accel) if accel else "")
        + f"\n🏅 <b>Score</b>: {_esc(score)}"
        f"\n🔗 <b>Peers</b>: {peer_count} connected"
    )