"""
LeaderLogic/generic/leader.py

Leader preset: any device without a specific hardware preset. Uses llama.cpp
(via llama-cpp-python), CPU only. Picks its GGUF from the device SCORE (when
LLM_detection_with_score is True):

    score >= 170   -> qwen3:1.7b
    score < 170    -> functiongemma:270m

When LLM_detection_with_score is False, always functiongemma:270m.

Exposed:
    load(cfg, profile) -> LlamaCppBackend
"""

import logging

from LeaderLogic.llama_cpp_backend import LlamaCppBackend, load_llm

log = logging.getLogger(__name__)

# When True, pick the model from the device score; when False, always functiongemma.
LLM_detection_with_score = False

# (params_b, hf_repo, filename, label, is_functiongemma)
_QWEN = (1.7, "unsloth/Qwen3-1.7B-GGUF",
         "Qwen3-1.7B-Q4_K_M.gguf", "qwen3:1.7b", False)
_FUNCTIONGEMMA = (0.27, "unsloth/functiongemma-270m-it-GGUF",
                  "functiongemma-270m-it-Q4_K_M.gguf", "functiongemma:270m", True)
_SCORE_QWEN = 170

_DISPATCH_SYSTEM = (
    "You are a smart-home agent dispatcher. "
    "Call exactly ONE tool that best matches the user's intent. "
    "Extract any names or object targets precisely from the user's message. "
    "If nothing matches, reply normally without calling any tool."
)
_DISPATCH_SYSTEM_FG = (
    "You are a smart-home agent dispatcher. "
    "You are a model that can do function calling with the following functions"
)
_ANSWER_SYSTEM = (
    "You are a smart home assistant. Turn the data into a short, friendly reply to the user. "
    "Reply in English. Summarize the result in natural language. "
    "DO NOT OUTPUT JSON, field names, brackets, or raw data structures. "
    "ALWAYS MENTION IN THE REPLY WHERE THE RESULTS ARE COMING FROM: the 'from' value is the "
    "device that executed the tool, and the user MUST be told which device(s) reported it."
)


def load(cfg: dict, profile: dict) -> LlamaCppBackend:
    score = (profile or {}).get("score", 0)
    params_b, repo, filename, label, is_fg = _select_model(score)

    log.info("loading %s (%sB params)", label, params_b)
    llm = load_llm(repo, filename, is_fg)

    dispatch_system = _DISPATCH_SYSTEM_FG if is_fg else _DISPATCH_SYSTEM
    return LlamaCppBackend(llm, is_fg, dispatch_system, _ANSWER_SYSTEM,
                            model_name=label, model_params_b=params_b)


def _select_model(score: float):
    """Pick (params_b, repo, filename, label, is_fg) from the device score."""
    if not LLM_detection_with_score:
        log.info("LLM_detection_with_score=False -> functiongemma:270m")
        return _FUNCTIONGEMMA
    chosen = _QWEN if score >= _SCORE_QWEN else _FUNCTIONGEMMA
    log.info("score=%s -> %s", score, chosen[3])
    return chosen
