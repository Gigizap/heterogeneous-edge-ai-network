"""
utils.py - shared helpers

Add these functions alongside your existing ones (available_ram_gb, etc.)
"""

import os
import sys
import socket
import threading
import asyncio


# -- existing (keep yours) ----------------------------------------------------

def available_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / (1024 ** 3)

def available_cpu_cores() -> int:
    import os
    return os.cpu_count() or 1


# -- new helpers used by main.py ----------------------------------------------

def own_ip() -> str:
    """Best-effort local IP (non-loopback)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if not ip.startswith("127.") and ":" not in ip:
                return ip
    except Exception:
        pass
    return "0.0.0.0"


def start_async_loop() -> asyncio.AbstractEventLoop:
    """
    Start a new asyncio event loop in a background daemon thread.
    Returns the loop so callers can schedule coroutines onto it.
    """
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Give the loop a moment to start before returning
    import time
    time.sleep(0.05)
    return loop


def schedule(loop: asyncio.AbstractEventLoop, coro) -> asyncio.Future:
    """
    Schedule a coroutine on the given event loop from any thread.
    Returns a concurrent.futures.Future (can be ignored or awaited with .result()).
    """
    return asyncio.run_coroutine_threadsafe(coro, loop)

# -- console colours ----------------------------------------------------------
# Used by the first-run setup screen (ElectionLogic/identity.py). Plain ANSI
# SGR codes, which every terminal we target understands, including the
# STM32MP257F-DK serial console. They collapse to "" when stdout is not a TTY
# so redirected output and log files stay free of escape codes.
# Setup text itself stays ASCII-only: colour is optional, but a glyph a bare
# console cannot encode is a UnicodeEncodeError, so we do not use any.

# cmd.exe needs a nudge before it interprets ANSI codes instead of printing
# them raw. Running an empty command is enough; it is a no-op elsewhere.
if sys.platform == "win32":
    os.system("")

_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

RESET  = "\033[0m"  if _TTY else ""
BOLD   = "\033[1m"  if _TTY else ""
DIM    = "\033[2m"  if _TTY else ""
CYAN   = "\033[36m" if _TTY else ""
GREEN  = "\033[32m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
RED    = "\033[31m" if _TTY else ""
