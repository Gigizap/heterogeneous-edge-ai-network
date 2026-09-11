"""
SensingLogic/sensing_agent.py

Hooks sensing-agent skills into an existing transport.

Called by main.py via:
    sensing_agent.start(agent_id=AGENT_ID, transport=transport, sensing_preset=...)
"""

import json
import importlib
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


def start(agent_id: str, transport, sensing_preset: str, log_level="INFO"):
    """
    Attach sensing-agent skills to an existing transport.
    Chains into transport.on_message - typed messages (election, backup, etc.)
    pass through; JSON-RPC requests (tools/list, tools/call) are handled here
    and answered with a same-id JSON-RPC result/error.

    `log_level` - verbosity for this sensing agent's logger (set from main's
    --log_level; a level above CRITICAL keeps it silent).
    """
    # Every log line from this sensing agent - sensing_agent AND the skill modules it
    # loads - is stamped with the agent-ID, so it is always clear which agent on the
    # terminal is acting. The skills log via logging.getLogger(__name__) under the
    # shared "SensingLogic" parent, so they inherit this one handler/format.
    parent = logging.getLogger("SensingLogic")
    if not parent.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt=f"%(asctime)s  [{agent_id}]  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        parent.addHandler(handler)
        parent.propagate = False   # isolated from root / the leader's logging
    parent.setLevel(log_level)

    config_path = Path(__file__).parent / sensing_preset / "tool_config.json"
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError:
        log.warning("no tool_config.json for preset '%s' (%s) - starting with no sensing skills",
                    sensing_preset, config_path)
        return
    except (json.JSONDecodeError, OSError) as e:
        log.warning("invalid tool_config.json for '%s': %r - starting with no sensing skills",
                    sensing_preset, e)
        return

    if not isinstance(config, dict):
        log.warning("tool_config.json for '%s' is not a JSON object - starting with no sensing skills",
                    sensing_preset)
        return

    tool_defs   = config.get("tool_defs", [])
    skills      = _load_skills(config.get("skills", []), sensing_preset)
    current_job = {"id": None, "stop": None}

    prev_handler = transport.on_message

    def on_message(msg: dict):
        # Election / backup / capability messages carry a "type": pass through.
        if msg.get("type"):
            if prev_handler:
                prev_handler(msg)
            return

        method = msg.get("method")
        if not method:
            # Not a JSON-RPC request we handle (stray text, console broadcast…).
            if prev_handler:
                prev_handler(msg)
            return

        sender = msg.get("from", "?")
        rpc_id = msg.get("id")            # None => notification (no reply expected)

        def respond(payload: dict):
            if rpc_id is None:            # JSON-RPC: notifications get no response
                return
            try:
                transport.send_sync(sender, {"jsonrpc": "2.0", "id": rpc_id, **payload})
            except Exception as e:
                log.warning("could not reply to %s: %r", sender, e)

        def notify(content: str, tool: str):
            # Server -> client notification (no id) for a later alert from a
            # long-running skill. Leader-side delivery to the user is not wired
            # yet (that is the async-alert phase).
            try:
                transport.send_sync(sender, {
                    "jsonrpc": "2.0",
                    "method":  "notifications/sensing/alert",
                    "params":  {"tool": tool, "text": content},
                })
            except Exception as e:
                log.warning("could not notify %s: %r", sender, e)

        if method == "tools/list":
            respond({"result": {"tools": tool_defs}})
            return

        if method != "tools/call":
            log.info("unknown method from %s: %s", sender, method)
            respond({"error": {"code": -32601, "message": f"method not found: {method}"}})
            return

        params    = msg.get("params") or {}
        name      = params.get("name")
        arguments = params.get("arguments") or {}
        log.info("tools/call from %s: %s %s", sender, name, arguments)

        skill = next((s for s in skills if s["command"] == name), None)
        if skill is None:
            respond({"error": {"code": -32601, "message": f"unknown tool: {name}"}})
            return

        def ok(text):
            log.info("tools/call result -> %s: %r", sender, str(text))
            respond({"result": {"content": [{"type": "text", "text": str(text)}]}})

        if skill.get("threaded"):
            # Long-running skill: ack now as the tools/call result; later hits
            # are pushed as notifications/sensing/alert.
            stop = threading.Event()
            if current_job["stop"] is not None:
                current_job["stop"].set()
            current_job["id"]   = name + (f" {list(arguments.values())}" if arguments else "")
            current_job["stop"] = stop
            try:
                threading.Thread(
                    target=skill["fn"],
                    args=(lambda text: notify(text, name), stop),
                    kwargs=arguments, daemon=True,
                ).start()
            except TypeError as e:
                respond({"error": {"code": -32602, "message": f"invalid params for '{name}': {e}"}})
                return
            target = f"'{list(arguments.values())}'" if arguments else "a person"
            ok(f"watching for {target} - I'll message you when I spot it")
        else:
            try:
                ok(skill["fn"](**arguments))
            except TypeError as e:
                respond({"error": {"code": -32602, "message": f"invalid params for '{name}': {e}"}})
            except Exception as e:
                respond({"error": {"code": -32603, "message": f"internal error in '{name}': {e}"}})

    transport.on_message = on_message
    log.info("skills attached: %s", [s["command"] for s in skills])


# ── skill loader ───────────────────────────────────────────────────────────────

def _load_skills(skill_defs: list, sensing_preset: str) -> list:
    loaded = []
    for skill in skill_defs:
        try:
            module = importlib.import_module(f"SensingLogic.{sensing_preset}.{skill['file']}")
            skill["fn"] = getattr(module, skill["function"])
            if "stream_function" in skill:
                skill["stream_fn"] = getattr(module, skill["stream_function"])
            loaded.append(skill)
        except Exception as e:
            log.error("failed to load '%s': %s", skill["function"], e)
    return loaded