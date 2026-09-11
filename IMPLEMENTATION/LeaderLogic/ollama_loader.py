"""
ollama_loader.py
Starts an ollama-compatible server and blocks until the model is ready.
"""

import subprocess
import time
import logging
import requests

log = logging.getLogger(__name__)

POLL_INTERVAL = 1.0
BOOT_TIMEOUT  = 60.0


def start_and_wait(model: str, host: str = "http://localhost:11434", cmd: list = None) -> subprocess.Popen:
    health_url = f"{host}/api/tags"
    pull_url   = f"{host}/api/pull"

    try:
        resp = requests.get(health_url, timeout=2)
        if resp.status_code == 200:
            log.info("server already running, skipping launch")
            return None
    except requests.exceptions.ConnectionError:
        pass

    if cmd is None:
        cmd = ["ollama", "serve"]

    log.info("starting server (%s)…", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(model in m for m in models):
                    log.info("model '%s' ready", model)
                    return proc
                log.info("server up, model '%s' not found - pulling…", model)
                requests.post(pull_url, json={"model": model, "stream": False}, timeout=300)
                log.info("model '%s' pulled", model)
                return proc
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            log.warning("health check error: %s", e)

        time.sleep(POLL_INTERVAL)

    proc.terminate()
    raise RuntimeError(f"ollama did not become ready within {BOOT_TIMEOUT}s")


"""
# Regular ollama
start_and_wait("llama3.2:3b")

# Hailo-ollama
start_and_wait("qwen2.5-instruct:1.5b", host="http://localhost:8000", cmd=["hailo-ollama", "serve"])
"""