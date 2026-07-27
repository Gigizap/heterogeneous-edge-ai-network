"""
LeaderLogic/http_backend.py

Minimal OpenAI-compatible /v1/chat/completions HTTP client, shared by every
leader preset that talks to a model over HTTP (Ollama, CPU or FunctionGemma)
instead of loading it in-process.
"""

import requests


class LLM:
    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint
        self.model    = model
        self.headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def ask(self, system: str, user: str, temperature: float = 0.1) -> str:
        payload = {
            "model":       self.model,
            "messages":    [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temperature,
        }
        resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def chat(self, messages: list, tools: list = None,
             temperature: float = 0, seed: int = 42,
             tool_choice: str = "required") -> dict:
        """Full chat completion with optional tool calling. Returns the raw API response.

        tool_choice "required" forces a tool call; "auto" lets the model answer
        in plain text when nothing matches (so dispatch can short-circuit).
        """
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = tool_choice
        resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
