"""
LeaderLogic/raspberry_hailo/leader.py

Leader preset: Raspberry Pi 5 + Hailo AI HAT, fully on-chip.
Both dispatch (query -> tool) and answer (tool result -> reply) run natively
on the Hailo NPU, via the qwen3 .hef.

Exposed:
    load(cfg, profile) -> HailoBackend
"""

import json
import logging
from pathlib import Path

from LeaderLogic.hailo_backend import HailoLLM
from LeaderLogic.run_leader import parse_tool_call_from_text

logger = logging.getLogger(__name__)

_MODEL_NAME     = "qwen3:1.7b (Hailo)"
_MODEL_PARAMS_B = 1.7
_CONFIG_PATH    = Path(__file__).parent / "config.json"

# Qwen3 is a hybrid reasoning model: by default it prepends a <think>...</think>
# block before its real output, and on the NPU that reasoning dominates
# generation time. /no_think is Qwen's native soft switch to skip it.
DISPATCH_SYSTEM = (
    "You are a smart-home agent dispatcher. "
    "Call exactly ONE tool that best matches the user's intent. "
    "Extract any names or object targets precisely from the user's message. "
    "If nothing matches, reply normally without calling any tool. "
    "/no_think"
)
ANSWER_SYSTEM = (
    "You are a helpful assistant for a smart home agent network. "
    "The network has already executed a command and returned real data. "
    "Translate that raw result into a short, friendly reply for the user. "
    "DO NOT OUTPUT JSON, field names, brackets, or raw data structures. "
    "ALWAYS MENTION IN THE REPLY WHERE THE RESULTS ARE COMING FROM: the 'from' value is the "
    "device that executed the tool, and the user MUST be told which device(s) reported it. "
    "Always reply in the same language the user wrote in. "
    "/no_think"
)


def load(cfg: dict, profile: dict) -> "HailoBackend":
    dev_cfg  = json.loads(_CONFIG_PATH.read_text())
    hef_path = dev_cfg.get("hailo", {}).get("hef_path")

    hailo = HailoLLM(hef_path)
    hailo.activate()
    return HailoBackend(hailo)


class HailoBackend:
    model_name     = _MODEL_NAME
    model_params_b = _MODEL_PARAMS_B

    def __init__(self, hailo: HailoLLM):
        self._hailo = hailo

    def dispatch(self, user_text: str, tool_defs: list):
        messages = [
            {"role": "system", "content": DISPATCH_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        raw = self._hailo.generate(messages, tools=tool_defs)
        logger.info("dispatch raw: %r", raw[:400])

        parsed = parse_tool_call_from_text(raw)
        if parsed is None:
            return None, None, raw.strip()
        return parsed.get("name"), parsed.get("arguments", {}), None

    def answer(self, user_text: str, command: str, network_reply: list) -> str:
        reply_json = json.dumps(network_reply, ensure_ascii=False)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",   "content": (
                f"User request: {user_text}\n"
                f"Command executed: {command}\n\n"
                f"RESULT DATA (write your reply from this; the 'from' field is the device "
                f"that ran the tool, so name it in your answer)\n"
                f"{reply_json}\n"
            )},
        ]
        return self._hailo.generate(messages).strip()
