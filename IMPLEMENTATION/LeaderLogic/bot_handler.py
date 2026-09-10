# bothandler.py
import requests
import threading
import time
import logging

log = logging.getLogger(__name__)


# Persistent command keyboard, docked under the text input. Buttons send their
# own label as a plain message, so they reach the same command branches a typed
# command does - no extra code path. `resize_keyboard` lets each client shrink
# it to fit its own screen: no width or pixel count is assumed anywhere here.
MAIN_KEYBOARD = {
    "keyboard": [
        [{"text": "/capabilities"}, {"text": "/status"}],
        [{"text": "/loop"}, {"text": "/loop off"}],
    ],
    "resize_keyboard": True,
    "is_persistent":   True,
}

# Populates Telegram's native "/" menu. Sent once when polling starts.
_COMMANDS = [
    {"command": "start",        "description": "Show what this network can do"},
    {"command": "capabilities", "description": "Tools available now, and where they run"},
    {"command": "status",       "description": "Current leader, model and peers"},
    {"command": "loop",         "description": "Keep re-running the last tool"},
]


class TelegramBot:
    """
    Minimal Telegram bot using only `requests`.
    Uses long-polling (no webhooks, no external libs).
    """

    def __init__(self, token: str, allowed_users: list, on_user_message):
        """
        token           - Telegram bot token
        allowed_users   - list of int user IDs allowed to interact
        on_user_message - callback(chat_id: int, text: str) called on valid messages
        """
        self.token = token
        self.allowed_users = set(allowed_users)
        self.on_user_message = on_user_message
        self._base = f"https://api.telegram.org/bot{token}"
        self._offset = 0
        self._last_chat_id = None          # used by reply_to_last()
        self._lock = threading.Lock()
        self._active_typing = {}  # { chat_id: threading.Event }
        self._running = True               # cleared by stop() to end polling

    # ------------------------------------------------------------------ public

    def start(self):
        """Start the polling loop in a background thread."""
        self._running = True
        self._register_commands()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        log.info("Telegram long-polling started")

    def stop(self):
        """Stop the polling loop (so a new leader can take over the bot token).
        The in-flight long-poll finishes within ~10s, then the thread exits."""
        self._running = False
        log.info("Telegram polling stop requested")

    def send_message(self, chat_id: int, text: str):
        """Send a message to a specific chat."""
        log.info("message sent to chat %s: %s", chat_id, text)
        try:
            requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10
            )
        except Exception as e:
            log.warning("send_message error: %s", e)

        # Telegram clears typing when a message arrives, so an intermediate
        # message (e.g. "using <tool> on ...") would blank it until the next
        # refresh tick. Re-arm at once if this chat is still working.
        if chat_id in self._active_typing:
            self._send_typing(chat_id)

    def send_card(self, chat_id: int, html: str, reply_markup: dict = None):
        """Send a SYSTEM-authored message rendered as HTML.

        Only for text this repo writes itself (welcome, /status, announcements,
        loop notices), whose markup is known-safe. Model output must keep going
        through send_message: parse_mode makes Telegram reject a whole message
        over one stray character, and answers are free text we do not control.
        """
        log.info("card sent to chat %s: %s", chat_id, html)
        payload = {"chat_id": chat_id, "text": html, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            requests.post(f"{self._base}/sendMessage", json=payload, timeout=10)
        except Exception as e:
            log.warning("send_card error: %s", e)

        if chat_id in self._active_typing:
            self._send_typing(chat_id)

    def reply_to_last(self, text: str):
        """
        Convenience wrapper: reply to whoever sent the last message.
        Safe to call from any thread (e.g. transport callbacks).
        """
        with self._lock:
            chat_id = self._last_chat_id
        if chat_id is not None:
            self.send_message(chat_id, text)
        else:
            log.warning("reply_to_last: no chat yet, dropping: %s", text)

    def start_typing_indicator(self, chat_id: int):
        # Send the FIRST action synchronously so typing shows the instant the
        # message is received: requests.post releases the GIL during its I/O, so
        # it goes out even though the caller thread is about to enter a long,
        # GIL-holding inference call. A background thread then only refreshes it
        # (Telegram's typing state lapses after ~5s) until stop clears it.
        cancel = threading.Event()
        self._active_typing[chat_id] = cancel
        self._send_typing(chat_id)
        def loop():
            while not cancel.wait(timeout=4):
                self._send_typing(chat_id)
        threading.Thread(target=loop, daemon=True).start()

    def stop_typing_indicator(self, chat_id: int):
        cancel = self._active_typing.pop(chat_id, None)
        if cancel:
            cancel.set()

    def _register_commands(self):
        """Fill Telegram's native "/" menu with the commands this bot answers."""
        try:
            requests.post(f"{self._base}/setMyCommands",
                          json={"commands": _COMMANDS}, timeout=10)
            log.info("registered %d bot commands", len(_COMMANDS))
        except Exception as e:
            log.warning("setMyCommands error: %s", e)

    def _send_typing(self, chat_id: int):
        try:
            requests.post(
                f"{self._base}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
                timeout=10,
            )
        except Exception as e:
            log.warning("typing error: %s", e)

    # ----------------------------------------------------------------- private

    def _poll_loop(self):
        while self._running:
            try:
                resp = requests.get(
                    f"{self._base}/getUpdates",
                    params={"offset": self._offset, "timeout": 10},
                    timeout=20
                )
                resp.raise_for_status()
                updates = resp.json().get("result", [])
                for update in updates:
                    if not self._running:
                        break
                    self._offset = update["update_id"] + 1
                    self._handle_update(update)
            except Exception as e:
                if not self._running:
                    break
                log.warning("polling error: %s", e)
                time.sleep(5)
        log.info("Telegram polling stopped")

    def _handle_update(self, update: dict):
        msg = update.get("message")
        if not msg:
            return

        user_id = msg["from"]["id"]
        chat_id = msg["chat"]["id"]
        text = msg.get("text", "").strip()

        if not text:
            return

        # Auth check
        if user_id not in self.allowed_users:
            self.send_message(chat_id, "⛔ Unauthorized.")
            log.warning("rejected message from unauthorized user %s", user_id)
            return

        # Store last chat for reply_to_last()
        with self._lock:
            self._last_chat_id = chat_id

        log.info("message from %s: %s", user_id, text)
        try:
            self.on_user_message(chat_id, text)
        except Exception as e:
            log.error("on_user_message error: %s", e)
            self.send_message(chat_id, f"❌ Internal error: {e}")