"""Warm, persistent push-to-talk loop.

The `v2c listen` CLI spawns a fresh process per utterance, so it reloads the
Whisper model every single time (~seconds). Here the model is loaded and warmed
up once at startup; every utterance after that pays only inference (~0.3-1s for
base.en on a short clip).

    pip install voice-to-command[whisper,mic]
    python examples/listen_loop.py                    # uses voice.toml if present
    python examples/listen_loop.py voice.ko.toml      # speak Korean, get English

Ctrl-C to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from voice_to_command import Recognizer, record

cfg = Path(sys.argv[1] if len(sys.argv) > 1 else "voice.toml")
rec = Recognizer.from_config(cfg) if cfg.exists() else Recognizer.from_dict(
    {"asr": {"backend": "whisper", "language": "en",
             "whisper": {"model": "base.en"},
             "options": {"task": "transcribe"}},
     "translate": {"mode": "never"}}
)

print("loading + warming up model...", flush=True)
rec.transcribe((np.zeros(16000, dtype=np.float32), 16000))  # forces model load now
print("ready.\n", flush=True)

try:
    while True:
        text = rec.transcribe(record()).text
        print("->", text, "\n", flush=True)
except KeyboardInterrupt:
    sys.exit(0)
