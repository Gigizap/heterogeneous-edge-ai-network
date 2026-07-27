"""
LeaderLogic/stm32mp257fdk/leader.py

Leader preset: STM32MP257F-DK, CPU-only, always FunctionGemma (the only model
small enough to run acceptably on the Cortex-A35 without a NPU). Uses
llama.cpp (cross-compiled llama-cpp-python; see README.md).

Dispatch and answer share the SAME system prompt (_SYSTEM_PROMPT_FG), unlike
the generic preset: on this weak CPU, keeping the leading prompt tokens
byte-identical across the dispatch -> answer switch lets llama-cpp-python
reuse that prefix's KV cache instead of recomputing it from scratch every step.

Exposed:
    load(cfg, profile) -> LlamaCppBackend
"""

import logging

from LeaderLogic.llama_cpp_backend import LlamaCppBackend, load_llm

log = logging.getLogger(__name__)

_PARAMS_B = 0.27
_REPO     = "unsloth/functiongemma-270m-it-GGUF"
_FILENAME = "functiongemma-270m-it-Q4_K_M.gguf"
_LABEL    = "functiongemma:270m"

_SYSTEM_PROMPT_FG = (
    "You are a smart-home agent dispatcher. "
    "You are a model that can do function calling with the following functions. "
    "Call exactly ONE tool that best matches the user's intent, or reply normally if nothing matches. "
    "When you are instead given the result of a tool that already ran, turn that data into a "
    "short, friendly reply: summarize it in natural language and mention all the important data "
    "received from the tool and the available sensors when replying."
)


def load(cfg: dict, profile: dict) -> LlamaCppBackend:
    log.info("loading %s (%sB params)", _LABEL, _PARAMS_B)
    llm = load_llm(_REPO, _FILENAME, is_fg=True)
    return LlamaCppBackend(llm, is_fg=True,
                            dispatch_system=_SYSTEM_PROMPT_FG,
                            answer_system=_SYSTEM_PROMPT_FG,
                            model_name=_LABEL, model_params_b=_PARAMS_B)
