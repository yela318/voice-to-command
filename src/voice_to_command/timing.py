"""Per-stage latency: always collected into a dict, and printed by default.

    t = {}
    with stage("asr", t):
        ...
    # t == {"asr": 1.83}; prints "[timing] asr: 1.83s" to stderr iff enabled()

On by default. `[timing]` lines go to **stderr**, so stdout stays clean for
`--json` output and pipes. The Recognizer sets it from config.timing (default
true); the CLI has --timing; the env var V2C_TIMING overrides both
(V2C_TIMING=0 silences).
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Dict, Optional

_OFF = {"0", "false", "no", "off", ""}


def enabled() -> bool:
    return os.environ.get("V2C_TIMING", "1").strip().lower() not in _OFF


def set_enabled(on: bool) -> None:
    os.environ["V2C_TIMING"] = "1" if on else "0"


def log(message: str) -> None:
    if enabled():
        print("[timing] {}".format(message), file=sys.stderr, flush=True)


@contextlib.contextmanager
def stage(name: str, sink: Optional[Dict[str, float]] = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if sink is not None:
            sink[name] = round(elapsed, 4)
        if enabled():
            print("[timing] {}: {:.2f}s".format(name, elapsed), file=sys.stderr, flush=True)
