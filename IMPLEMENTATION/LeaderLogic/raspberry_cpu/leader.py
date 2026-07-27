"""
LeaderLogic/raspberry_cpu/leader.py

Leader preset: Raspberry Pi 5, CPU-only (no Hailo NPU).
Both dispatch (query -> tool) and answer (tool result -> reply) run on a
single CPU Ollama model (qwen3:1.7b) over its OpenAI-compatible HTTP endpoint.

Exposed:
    load(cfg, profile) -> CpuBackend
"""

import json
import logging
from pathlib import Path

from LeaderLogic.http_backend import LLM
from LeaderLogic.ollama_loader import start_and_wait
from LeaderLogic.run_leader import parse_openai_dispatch, to_openai_tools

logger = logging.getLogger(__name__)

_MODEL_NAME     = "qwen3:1.7b (CPU)"
_MODEL_PARAMS_B = 1.7
_CHAT_PATH      = "/v1/chat/completions"
_CONFIG_PATH    = Path(__file__).parent / "config.json"

# Qwen3 is a hybrid reasoning model: by default it prepends a <think>...</think>
# block, and on edge CPU that reasoning dominates generation time. /no_think is
# Qwen's native soft switch to skip it (kept on both steps, as in the original).
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


def load(cfg: dict, profile: dict) -> "CpuBackend":
    dev_cfg = json.loads(_CONFIG_PATH.read_text())
    cpu_cfg = dev_cfg.get("llm_cpu", {})
    host    = cpu_cfg.get("host", "http://127.0.0.1:11434")
    model   = cpu_cfg.get("model", "qwen3:1.7b")

    start_and_wait(model, host=host)
    llm = LLM(endpoint=host + _CHAT_PATH, api_key=cpu_cfg.get("api_key", ""), model=model)
    return CpuBackend(llm)


class CpuBackend:
    model_name     = _MODEL_NAME
    model_params_b = _MODEL_PARAMS_B

    def __init__(self, llm: LLM):
        self._llm = llm

    def dispatch(self, user_text: str, tool_defs: list):
        messages = [
            {"role": "system", "content": DISPATCH_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        resp = self._llm.chat(messages, tools=to_openai_tools(tool_defs), tool_choice="auto")
        logger.info("dispatch response: %s",
                    json.dumps(resp.get("choices", [{}])[0].get("message", {})))
        return parse_openai_dispatch(resp)

    def answer(self, user_text: str, command: str, network_reply: list) -> str:
        reply_json = json.dumps(network_reply, ensure_ascii=False)
        prompt = (
            f"User request: {user_text}\n"
            f"Command executed: {command}\n\n"
            f"RESULT DATA (write your reply from this; the 'from' field is the device "
            f"that ran the tool, so name it in your answer)\n"
            f"{reply_json}\n"
        )
        return self._llm.ask(ANSWER_SYSTEM, prompt, temperature=0.7)
