"""
LeaderLogic/raspberry_cpuhailo/leader.py

Leader preset: Raspberry Pi 5 + Hailo AI HAT, split inference.
Dispatch (query -> tool) runs on FunctionGemma via CPU Ollama (purpose-built
for tool-calling, cheap); answer (tool result -> reply) runs on the Hailo NPU
(qwen3 .hef), which is more expensive but only invoked once per turn.

Exposed:
    load(cfg, profile) -> CpuHailoBackend
"""

import json
import logging
from pathlib import Path

from LeaderLogic.hailo_backend import HailoLLM
from LeaderLogic.http_backend import LLM
from LeaderLogic.ollama_loader import start_and_wait
from LeaderLogic.run_leader import parse_openai_dispatch, to_openai_tools

logger = logging.getLogger(__name__)

_MODEL_NAME     = "functiongemma:270m (dispatch, CPU) + qwen3:1.7b (answer, Hailo)"
_MODEL_PARAMS_B = 1.7   # dominated by the Hailo answer step
_CHAT_PATH      = "/v1/chat/completions"
_CONFIG_PATH    = Path(__file__).parent / "config.json"

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


def load(cfg: dict, profile: dict) -> "CpuHailoBackend":
    dev_cfg  = json.loads(_CONFIG_PATH.read_text())
    tool_cfg = dev_cfg.get("llm_cpu_tool_calling", {})
    host     = tool_cfg.get("host", "http://127.0.0.1:11434")
    model    = tool_cfg.get("model", "functiongemma")

    start_and_wait(model, host=host)
    dispatch_llm = LLM(endpoint=host + _CHAT_PATH, api_key=tool_cfg.get("api_key", ""), model=model)

    hef_path = dev_cfg.get("hailo", {}).get("hef_path")
    hailo    = HailoLLM(hef_path)
    hailo.activate()

    return CpuHailoBackend(dispatch_llm, hailo)


class CpuHailoBackend:
    model_name     = _MODEL_NAME
    model_params_b = _MODEL_PARAMS_B

    def __init__(self, dispatch_llm: LLM, hailo: HailoLLM):
        self._dispatch_llm = dispatch_llm
        self._hailo        = hailo

    def dispatch(self, user_text: str, tool_defs: list):
        messages = [
            {"role": "system", "content": DISPATCH_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        # No tool_choice arg: FunctionGemma dispatch used chat()'s default
        # ("required"), unlike the CPU-qwen preset which forces "auto".
        resp = self._dispatch_llm.chat(messages, tools=to_openai_tools(tool_defs))
        logger.info("dispatch response: %s",
                    json.dumps(resp.get("choices", [{}])[0].get("message", {})))
        return parse_openai_dispatch(resp)

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
