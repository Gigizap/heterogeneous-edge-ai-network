"""
main.py

Usage
─────
    python main.py
    python main.py --port 5001      # this is to simulate a second device on the same machine by running a second copy of the program in another terminal.

Architecture per device:
    agent transport  →  AGENT_ID          on TCP_PORT      (always: election + sensing-agent skills)
    leader transport →  AGENT_ID-leader   on TCP_PORT + 1  (created only when elected)
"""

import argparse
import json
import logging
import threading
import asyncio
import time
from pathlib import Path

from ConnectionLogic.transport import P2PTransport
from ConnectionLogic.discovery import Discovery
from ElectionLogic.identity    import load_or_create_profile, get_leader_module
from ElectionLogic.election    import ElectionManager
from ElectionLogic.scoring     import compute_score, hardware_info
from utils import own_ip, start_async_loop, schedule

# ── CLI + config ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--log_level", default="DEBUG",
                    help="logging verbosity: DEBUG (default), INFO, WARNING, ERROR")
parser.add_argument("--traffic", action="store_true",
                    help="record every message in/out of this device to "
                         "traffic_<agent>_<port>.jsonl (type 'traffic' for a live summary)")
args = parser.parse_args()

LOG_LEVEL = args.log_level.upper()

cfg      = json.loads((Path(__file__).parent / "software_config.json").read_text())
TCP_PORT = args.port or cfg["agent"]["tcp_port"]

# creates a device_profile.json if run normally
# creates a device_profile_TCP_PORT.json when for testing purposes you want different instances on the same machine
profile_path = Path(__file__).parent / (
    f"device_profile_{TCP_PORT}.json" if args.port else "device_profile.json"
)

election = None # will be set later if device can lead

# ── identity + scoring ────────────────────────────────────────────────────────

profile        = load_or_create_profile(profile_path=profile_path)
AGENT_ID       = profile["agent_id"]
LEADER_ID      = f"{AGENT_ID}-leader"
LEADER_PORT    = TCP_PORT + 1

# ── logging ───────────────────────────────────────────────────────────────────
# Every line is stamped with the agent it came from. Components that bound their own
# ID via a LoggerAdapter keep it; otherwise each handler fills in a default:
#   root handler          → AGENT_ID   (device/agent: main, scoring, transport, election)
#   "LeaderLogic" handler → LEADER_ID  (the leader brain: bot, LLM, announce, collector)
# so a device that both senses and leads keeps the two roles distinct. Level is set
# by --log_level (silent if omitted; pass DEBUG to also see the verbose dumps).
class _AgentDefault(logging.Filter):
    def __init__(self, default_agent):
        super().__init__()
        self._default = default_agent
    def filter(self, record):
        if not hasattr(record, "agent"):
            record.agent = self._default
        return True

_FMT = logging.Formatter("%(asctime)s  [%(agent)s]  %(levelname)-7s  %(message)s", "%H:%M:%S")

_root_handler = logging.StreamHandler()
_root_handler.setFormatter(_FMT)
_root_handler.addFilter(_AgentDefault(AGENT_ID))
logging.getLogger().addHandler(_root_handler)
logging.getLogger().setLevel(LOG_LEVEL)

_leader_handler = logging.StreamHandler()
_leader_handler.setFormatter(_FMT)
_leader_handler.addFilter(_AgentDefault(LEADER_ID))
_leader_logger = logging.getLogger("LeaderLogic")
_leader_logger.addHandler(_leader_handler)
_leader_logger.setLevel(LOG_LEVEL)
_leader_logger.propagate = False

log       = logging.getLogger("main")                                              # agent context  → AGENT_ID
leaderlog = logging.LoggerAdapter(logging.getLogger("main"), {"agent": LEADER_ID})  # leader context → LEADER_ID

# Hush chatty third-party loggers so --log_level DEBUG shows OUR debug, not stdlib /
# library internals (asyncio's "Using proactor", urllib3 connection spam, etc.).
for _noisy in ("asyncio", "urllib3", "huggingface_hub", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SCORE          = compute_score(profile)
HW_INFO        = hardware_info(profile)
profile["score"] = SCORE

can_lead       = get_leader_module(profile) is not None
sensing_preset = profile.get("sensing_preset")

log.info("agent  : %s  (port %s)", AGENT_ID, TCP_PORT)
if can_lead:
    log.info("leader : %s  (port %s, on election win)", LEADER_ID, LEADER_PORT)
log.info("ip     : %s", own_ip())
log.info("score  : %s", SCORE)
log.info("preset : %s", profile.get("leader_preset") or "none")
log.info("sensing: %s", sensing_preset or "none")

# ── async event loop (runs in background thread) ──────────────────────────────

_loop = start_async_loop()

# ── agent transport + discovery (always exists) ──────────────────────────────

def _on_peer_lost(pid):
    schedule(_loop, transport.unregister_peer(pid))
    if election is not None: #IF THE SENSING AGENT HAS ELECTION => THE DEVICE HAS LEADER CAPABILITIES
        #Notify the election manager that a peer has gone silent (killed / powered
        #off / left the network). It drops the peer and, if it was the current
        #leader, starts a fresh election so the next-best device takes over.
        election.notify_peer_lost(pid)   # re-elects only if pid == current leader


def _on_peer_found(pid, ip, port):
    schedule(_loop, transport.register_peer(pid, ip, port))
    if election is not None: #LEADER-CAPABLE DEVICE => RE-ENTER THE ELECTION PLANE
        #Discovery just linked a peer that may have joined AFTER our startup
        #window - it is alive in the transport table but its score is not yet on
        #the election plane. Announce so the score reaches us (backup ranking) and
        #any leadership conflict reconciles: a leader claims, a follower says
        #HELLO. See ElectionManager.announce().
        schedule(_loop, election.announce())


def _on_agent_message(msg: dict):
    """Fallback: plain-text messages not handled by sensing agent or election."""
    if not msg.get("type") and msg.get("text"):
        log.info("agent message from %s: %s", msg.get("from", "?"), msg["text"])

# Optional traffic recorder (--traffic): ONE meter for the whole device, shared
# by the agent transport, both discoveries, and the leader twin (on election),
# so its capture/summary covers every message in and out of this device.
TRAFFIC = None
if args.traffic:
    from ConnectionLogic.traffic import TrafficMeter
    _traffic_path = Path(__file__).parent / f"traffic_{AGENT_ID}_{TCP_PORT}.jsonl"
    TRAFFIC = TrafficMeter(str(_traffic_path))
    log.info("traffic recording ON -> %s", _traffic_path)

transport = P2PTransport(AGENT_ID, TCP_PORT, _on_agent_message, meter=TRAFFIC)
discovery = Discovery(
    AGENT_ID, TCP_PORT,
    on_peer_found = _on_peer_found,
    on_peer_lost  = _on_peer_lost,
    meter = TRAFFIC,
)
transport._event_loop = _loop

# ── conversation backup receiver (EVERY device, on the agent transport) ───────
# Attaching this here — not only on leaders — is what makes backups actually
# land: a follower can store the leader's history and hand it back to the next
# leader. When THIS device is the leader it also uses `backup` to push, to the
# REPLICATION_FACTOR strongest peers (election.backup_peers()).
#
# REPLICATION_FACTOR = how many devices hold a copy of each conversation.
#   1 → only the 2nd-strongest device (the leader and that device both dying
#       loses the history); 2 → 2nd and 3rd strongest, so a double failure still
#       leaves a replica. Capped automatically by how many peers actually exist.
from LeaderLogic.backup_manager import BackupManager
REPLICATION_FACTOR = 2
backup = BackupManager(
    agent_id           = AGENT_ID,
    transport          = transport,
    get_backup_peers   = lambda n: election.backup_peers(n) if election is not None else [],
    replication_factor = REPLICATION_FACTOR,
)

# ── sensing-agent skills (attach to agent transport) ─────────────────────────

if sensing_preset:
    try:
        import SensingLogic.sensing_agent as sensing_agent
        sensing_agent.start(
            agent_id       = AGENT_ID,
            transport      = transport,
            sensing_preset = sensing_preset,
            log_level      = LOG_LEVEL,
        )
    except Exception as e:
        log.error("sensing agent start failed: %s", e)

# ── election (only if device can lead) ────────────────────────────────────────

_current_leader   = None
_leader_resources = None          # {bot, transport, discovery, *_fut} while leading
_leader_lock      = threading.Lock()

def _on_became_leader(prof: dict):
    leaderlog.info("elected as leader")
    threading.Thread(target=_boot_leader, args=(prof,), daemon=True).start()

def _on_became_follower(leader_id: str):
    log.info("following: %s", leader_id)

def _stop_leader():
    """Tear the leader process down cleanly after a handoff / being out-ranked,
    so we never run two leaders (e.g. two Telegram pollers on the same token)."""
    global _current_leader, _leader_resources
    with _leader_lock:
        res = _leader_resources
        _leader_resources = None
        _current_leader   = None
    if not res:
        return
    leaderlog.info("stopping leader (handoff / demotion)")
    try:
        if res.get("bot"):
            res["bot"].stop()
    except Exception as e:
        leaderlog.error("bot stop error: %s", e)
    for key in ("transport_fut", "discovery_fut"):
        fut = res.get(key)
        if fut is not None:
            try:
                fut.cancel()
            except Exception:
                pass
    lt = res.get("transport")
    if lt is not None:
        try:
            schedule(_loop, lt.stop())
        except Exception:
            pass

if can_lead:
    election = ElectionManager(
        agent_id           = AGENT_ID,
        score              = SCORE,
        profile            = profile,
        transport          = transport,
        on_became_leader   = _on_became_leader,
        on_became_follower = _on_became_follower,
        on_demoted         = _stop_leader,
    )

# ── leader boot (creates its own transport + discovery) ──────────────────────

def _on_leader_message(msg: dict):
    """Fallback for leader transport."""
    if not msg.get("type") and msg.get("text"):
        leaderlog.info("leader message from %s: %s", msg.get("from", "?"), msg["text"])

def _plan_restore(restored: dict):
    """Split restored {chat_id: messages} into (chat_id, history_to_load, pending_text).

    If a chat ends in an unanswered user turn, the dead leader died before
    replying: pending_text is that turn's text and it is stripped from
    history_to_load (handle() re-adds it once on replay). Otherwise pending_text
    is None and the full history is loaded as context. Non-int chat ids are skipped.
    """
    plan = []
    for chat_id_str, messages in (restored or {}).items():
        try:
            chat_id = int(chat_id_str)
        except (TypeError, ValueError):
            continue
        if messages and messages[-1].get("role") == "user":
            plan.append((chat_id, messages[:-1], messages[-1].get("content", "")))
        else:
            plan.append((chat_id, messages, None))
    return plan

def _resume_pending_requests(leader, pending):
    """A newly elected leader finishes any request the dead leader received but
    never answered (its backup ended in an unanswered user turn).

    `pending` is [(chat_id, user_text), ...]. Each reply is delivered ONLY to its
    own chat_id (leader.handle sends to that chat — no broadcast). After replying
    we advance that chat's backup so a later failover won't answer it twice.
    Runs sequentially in one thread to avoid hammering the single LLM.
    """
    for chat_id, text in pending:
        try:
            leaderlog.info("resuming unanswered request for chat %s: %r", chat_id, text)
            leader.handle(chat_id, text)
            # advance/clear the slot so a further failover won't answer it twice
            updated = list(leader.get_history(chat_id)) if hasattr(leader, "get_history") else []
            backup.push(chat_id, updated)
        except Exception as e:
            leaderlog.error("could not resume request for chat %s: %s", chat_id, e)

def _boot_leader(prof: dict):
    global _current_leader, _leader_resources
    import LeaderLogic.run_leader as run_leader
    from LeaderLogic.capabilities import announce, resume_notice
    from LeaderLogic.bot_handler  import TelegramBot

    # ── leader gets its own transport + discovery (for tool dispatch) ─────
    leader_transport = P2PTransport(LEADER_ID, LEADER_PORT, _on_leader_message, meter=TRAFFIC)
    leader_discovery = Discovery(
        LEADER_ID, LEADER_PORT,
        on_peer_found = lambda pid, ip, port: schedule(_loop, leader_transport.register_peer(pid, ip, port)),
        on_peer_lost  = lambda pid:           schedule(_loop, leader_transport.unregister_peer(pid)),
        meter = TRAFFIC,
    )
    leader_transport._event_loop = _loop
    transport_fut = schedule(_loop, leader_transport.start())
    discovery_fut = schedule(_loop, leader_discovery.start())

    # Let leader discover peers before issuing fetchskills
    time.sleep(3.0)

    bot = TelegramBot(
        token           = cfg["telegram"]["token"],
        allowed_users   = cfg["telegram"]["allowed_users"],
        on_user_message = _on_telegram_message,
    )

    # Conversation history is pushed to / restored from followers over the AGENT
    # transport (every device runs a backup receiver), using the module-level
    # `backup`. Backups go to the REPLICATION_FACTOR strongest devices
    # (election.backup_peers); restore pulls from whichever of them survived.
    restored = backup.restore_all()
    if restored:
        leaderlog.info("restored %d conversation(s)", len(restored))

    # run_leader picks the leader implementation matching this device's
    # leader_preset (a LeaderLogic/<preset>/ folder) and boots it; leader_mod
    # (run_leader itself) exposes MODEL_NAME / MODEL_PARAMS_B for /status.
    leader = run_leader.boot(cfg=cfg, agent_id=LEADER_ID,
                             transport=leader_transport,
                             discovery=leader_discovery, bot=bot, profile=profile)
    leader_mod = run_leader

    pending = []   # chats whose backup ended in an unanswered user turn
    if restored:
        for chat_id, history, pending_text in _plan_restore(restored):
            # Restoring PRIOR context needs history support; replaying the single
            # UNANSWERED message does not — so we gate only the load_history call,
            # and always replay the pending turn (just needs leader.handle).
            if hasattr(leader, "load_history"):
                try:
                    leader.load_history(chat_id, history)
                except Exception:
                    pass
            if pending_text:
                pending.append((chat_id, pending_text))

    with _leader_lock:
        _current_leader   = (leader, backup)
        _leader_resources = {
            "bot":           bot,
            "transport":     leader_transport,
            "discovery":     leader_discovery,
            "transport_fut": transport_fut,
            "discovery_fut": discovery_fut,
            "leader_mod":    leader_mod,   # for /status (MODEL_NAME / MODEL_PARAMS_B)
            "profile":       prof,         # for /status (leader_preset)
        }

    announce(
        agent_id       = LEADER_ID,
        profile        = prof,
        model_name     = getattr(leader_mod, "MODEL_NAME", "unknown"),
        model_params_b = getattr(leader_mod, "MODEL_PARAMS_B", 0.0),
        hardware       = HW_INFO,
        peer_count     = len(getattr(leader_discovery, "_peers", {})),
        transport      = leader_transport,
        bot            = bot,
        known_chat_ids = set(),
    )
    bot.start()

    # Finish any request the dead leader never answered. Background thread so boot
    # and Telegram polling aren't blocked by LLM inference.
    if pending:
        # Mid-reply failover: tell the affected user(s) the leader changed and
        # who's resuming. chat_ids come straight from the restored backup, so we
        # notify only the chat(s) that were mid-reply (empty otherwise).
        notice = resume_notice(
            agent_id       = LEADER_ID,
            profile        = prof,
            model_name     = getattr(leader_mod, "MODEL_NAME", "unknown"),
            model_params_b = getattr(leader_mod, "MODEL_PARAMS_B", 0.0),
            hardware       = HW_INFO,
            peer_count     = len(getattr(leader_discovery, "_peers", {})),
        )
        for chat_id, _ in pending:
            bot.send_card(chat_id, notice)

        threading.Thread(target=_resume_pending_requests,
                         args=(leader, pending), daemon=True).start()

# ── telegram handler ──────────────────────────────────────────────────────────

def _live_tools(leader):
    """[(tool_name, [owner_agent_id, ...]), ...] currently reachable, or None.

    None means this leader has no tool registry at all (the mock preset), which
    the cards render differently from "a registry that is currently empty".
    """
    if not hasattr(leader, "net"):
        return None
    try:
        names = sorted({(td.get("function", td)).get("name")
                        for td in leader.net.available_tools()})
        return [(n, sorted(leader.net.owners(n))) for n in names]
    except Exception as e:
        leaderlog.warning("could not list tools: %s", e)
        return None


def _on_telegram_message(chat_id: int, text: str):
    t0 = time.perf_counter()   # query received — timed to the backup below
    with _leader_lock:
        pair = _current_leader
        res  = _leader_resources
    if pair is None:
        return
    leader, backup = pair

    # /loop keeps the tool a question picks running after the answer: the leader
    # re-dispatches it and messages the user again whenever the result changes.
    # It stays on until "/loop off"; each new question stops the previous loop.
    # /start greets a new chat: what the fleet does, which skills are reachable
    # right now, and the command keyboard.
    if text.strip() in ("/start", "/start@"):
        from LeaderLogic.capabilities import welcome_text
        from LeaderLogic.bot_handler import MAIN_KEYBOARD
        mod = res["leader_mod"]
        res["bot"].send_card(chat_id, welcome_text(
            agent_id       = LEADER_ID,
            profile        = res["profile"],
            model_name     = getattr(mod, "MODEL_NAME", "unknown"),
            model_params_b = getattr(mod, "MODEL_PARAMS_B", 0.0),
            hardware       = HW_INFO,
            peer_count     = len(getattr(res["discovery"], "_peers", {})),
            tools          = _live_tools(leader),
        ), reply_markup=MAIN_KEYBOARD)
        return

    # /capabilities lists the tools reachable right now and which devices own
    # each one. Read straight from the leader's live registry, so a sensing
    # device joining or leaving is reflected on the next call.
    if text.strip() == "/capabilities":
        from LeaderLogic.capabilities import capabilities_text
        res["bot"].send_card(chat_id, capabilities_text(_live_tools(leader)))
        return

    if text.strip() in ("/loop", "/loop off"):
        on = text.strip() == "/loop"
        if hasattr(leader, "set_loop"):        # the mock preset has no pipeline
            leader.set_loop(on)
            res["bot"].send_card(
                chat_id,
                "🔁 <b>Loop on</b>\nAfter each answer I'll keep re-running the "
                "skill it used, and message you when the result changes."
                if on else
                "⏹ <b>Loop off</b>\nI'll answer once per question.")
        else:
            res["bot"].send_card(chat_id, "⚠️ This leader does not support /loop.")
        return

    # /status reports the current model + leader capabilities. Built from the same
    # data as the capability broadcast; multi-workflow leaders append their active
    # workflow via status_extra().
    if text == "/status":
        from LeaderLogic.capabilities import status_text
        mod = res["leader_mod"]
        msg = status_text(
            agent_id       = LEADER_ID,
            profile        = res["profile"],
            model_name     = getattr(mod, "MODEL_NAME", "unknown"),
            model_params_b = getattr(mod, "MODEL_PARAMS_B", 0.0),
            hardware       = HW_INFO,
            peer_count     = len(getattr(res["discovery"], "_peers", {})),
        )
        if hasattr(leader, "status_extra"):
            msg += leader.status_extra()
        res["bot"].send_card(chat_id, msg)
        return

    # Back up the still-unanswered query immediately (prepended with prior history
    # if the leader keeps it). This one message must survive a mid-reply crash even
    # on leaders that don't track history at all.
    prior = list(leader.get_history(chat_id)) if hasattr(leader, "get_history") else []
    backup.push(chat_id, prior + [{"role": "user", "content": text}])
    leaderlog.info("query received -> backed up in %.3f ms", (time.perf_counter() - t0) * 1000)

    def _run():
        leader.handle(chat_id, text)
        # Advance the slot so it no longer looks "pending" (prevents a double-answer
        # on a later failover): history leaders push the updated transcript;
        # history-less leaders just clear it to [].
        updated = list(leader.get_history(chat_id)) if hasattr(leader, "get_history") else []
        backup.push(chat_id, updated)

    threading.Thread(target=_run, daemon=True).start()

# ── start async components + console loop ────────────────────────────────────

async def _async_main():
    tasks = [
        asyncio.create_task(transport.start()),
        asyncio.create_task(discovery.start()),
    ]
    if election:
        tasks.append(asyncio.create_task(election.run_startup_election()))
    await asyncio.gather(*tasks)

schedule(_loop, _async_main())

while True:
    try:
        text = input().strip()
    except EOFError:
        threading.Event().wait()      # no interactive console: keep the agent running
    except KeyboardInterrupt:
        log.info("bye!")
        break
    if not text or text.lower() in ("quit", "exit"):
        break
    if text.lower() == "traffic":
        if TRAFFIC:
            log.info("traffic report:\n%s", TRAFFIC.report())
        else:
            log.info("traffic recording is off (start with --traffic)")
        continue
    peers = getattr(discovery, "_peers", {})
    if not peers:
        log.info("no peers")
        continue
    schedule(_loop, transport.broadcast({"text": text}))
    log.info("broadcast to %d peer(s)", len(peers))

if TRAFFIC:
    log.info("final traffic report:\n%s", TRAFFIC.report())
    TRAFFIC.close()