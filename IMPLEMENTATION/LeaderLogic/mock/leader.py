"""
LeaderLogic/mock/leader.py

Mock leader - NO LLM, for testing the election / backup / failover machinery
without the cost (or hardware) of real inference.

It exposes the legacy full-boot leader-module contract (not the load() ->
Backend contract the other presets use, since it isn't a pipeline at all):

    boot(cfg, agent_id, transport, discovery, bot, profile=None) -> MockWorkflow
    module-level MODEL_NAME / MODEL_PARAMS_B   (read for the capability broadcast)

...but instead of dispatching to sensing agents through an LLM, handle() just
echoes the user's message back over Telegram. No model is downloaded or loaded,
and no LeaderNetwork is created. That is enough to exercise, on any device:
  - the query-received -> backed-up path (main._on_telegram_message),
  - conversation backup / restore (get_history / load_history),
  - mid-reply failover resume (main._resume_pending_requests),
  - split-brain reconciliation (election runs regardless of the leader module).

Selected automatically once this folder exists (see
ElectionLogic.identity.discover_leader_presets()); set
"leader_preset": "mock" in device_profile.json to use it.
"""

import logging
import threading
from typing import Dict, List

log = logging.getLogger(__name__)

# Read by main.py for the capability broadcast / /status. A mock runs no model.
MODEL_NAME     = "mock (no LLM)"
MODEL_PARAMS_B = 0.0


# ── boot ──────────────────────────────────────────────────────────────────────

def boot(cfg: dict, agent_id: str, transport, discovery, bot, profile: dict = None) -> "MockWorkflow":
    """No model load and no network wiring - just a bot-echo workflow.

    transport / discovery / cfg are accepted to match the leader-module contract
    but are unused: the mock never dispatches a tool to the sensing agents.
    """
    log.info("mock leader ready - no LLM loaded")
    return MockWorkflow(agent_id=agent_id, bot=bot)


# ── MockWorkflow ────────────────────────────────────────────────────────────────

class MockWorkflow:
    """Minimal leader exposing main.py's interface, with no inference."""

    def __init__(self, agent_id: str, bot=None):
        self._agent_id     = agent_id
        self._bot          = bot
        self._histories:   Dict[int, List[dict]] = {}
        self._history_lock = threading.Lock()

    # ── main.py interface ─────────────────────────────────────────────────────

    def handle(self, chat_id: int, text: str) -> None:
        """Echo the message back. Records the turn so the backup slot advances
        from 'pending' (last turn = user) to 'answered' (last turn = assistant),
        exactly like a real leader - which is what main.py backs up after us."""
        reply = f"[mock leader {self._agent_id}] received: {text}"
        with self._history_lock:
            h = self._histories.setdefault(chat_id, [])
            h.append({"role": "user",      "content": text})
            h.append({"role": "assistant", "content": reply})
        log.info("mock handle chat %s: %r", chat_id, text)
        if self._bot:
            self._bot.send_message(chat_id, reply)
        else:
            log.info("mock answer: %s", reply)

    # ── history (read/written by backup_manager via main.py) ──────────────────

    def get_history(self, chat_id: int) -> List[dict]:
        with self._history_lock:
            return list(self._histories.get(chat_id, []))

    def load_history(self, chat_id: int, messages: List[dict]) -> None:
        with self._history_lock:
            self._histories[chat_id] = list(messages)

    # ── /status ───────────────────────────────────────────────────────────────

    def status_extra(self) -> str:
        return "\nWorkflow: mock (echo, no LLM)"
