"""voice-to-command -- Korean speech to English text, via faster-whisper.

    from voice_to_command import transcribe, record

    transcribe("sample/voice_kor.m4a")   # file  -> "Please give me carrots."
    transcribe(record())          # mic   -> English text

whisper's task="translate" does the Korean -> English in one pass. Whatever
consumes the string is wired up elsewhere.
"""

from .capture import record
from .core import MODEL_SIZE, transcribe, warmup
from .remote import serve_http, transcribe_remote

__version__ = "0.2.0"

__all__ = [
    "transcribe",
    "warmup",
    "record",
    "serve_http",
    "transcribe_remote",
    "MODEL_SIZE",
    "__version__",
]
