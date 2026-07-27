"""
LeaderLogic/hailo_backend.py

Thin wrapper around the Hailo NPU native LLM (hailo_platform.genai), shared by
every leader preset that runs inference on the Hailo AI HAT (raspberry_hailo
for dispatch+answer, raspberry_cpuhailo for answer only).
"""

import logging

logger = logging.getLogger(__name__)


class HailoLLM:
    def __init__(self, hef_path: str, temperature: float = 0.1, seed: int = 42,
                 max_tokens: int = 512):
        self._hef_path    = str(hef_path)
        self._temperature = temperature
        self._seed        = seed
        self._max_tokens  = max_tokens
        self.vdevice = None
        self.llm     = None

    def activate(self):
        """Acquire the Hailo device + load the HEF. Call before first use."""
        if self.llm is not None:
            return
        from hailo_platform import VDevice
        from hailo_platform.genai import LLM as _HailoLLM

        params = VDevice.create_params()
        params.group_id = "1"
        self.vdevice = VDevice(params)
        self.llm     = _HailoLLM(self.vdevice, self._hef_path)
        logger.info("Hailo LLM loaded (%s)", self._hef_path)

    def deactivate(self):
        if self.llm is not None:
            try:
                self.llm.release()
            except Exception as e:
                logger.warning("llm.release() failed: %s", e)
            self.llm = None
        if self.vdevice is not None:
            try:
                self.vdevice.release()
            except Exception as e:
                logger.warning("vdevice.release() failed: %s", e)
            self.vdevice = None

    def generate(self, messages: list, tools=None) -> str:
        self.llm.clear_context()
        parts = []
        with self.llm.generate(
            prompt=messages, tools=tools,
            temperature=self._temperature, seed=self._seed,
            max_generated_tokens=self._max_tokens,
        ) as gen:
            for token in gen:
                parts.append(token)
        # Hailo's genai LLM emits the qwen3 turn-end token as a literal string; strip it.
        return "".join(parts).replace("<|im_end|>", "")
