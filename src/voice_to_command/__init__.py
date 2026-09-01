"""voice-to-command -- turn a spoken command into a normalized text string.

    from voice_to_command import Recognizer, record

    rec = Recognizer.from_dict({"asr": {"backend": "whisper"}})
    result = rec.transcribe("command.wav")   # or rec.transcribe(record())
    print(result.text)

The library does audio -> ASR -> (optional translation) -> normalization,
and nothing else: no robot, policy, or simulation concepts. ASR and
translation backends are pluggable (see voice_to_command/registry.py).
"""

from . import errors
from .audio import Audio
from .capture import record
from .config import RecognizerConfig
from .recognizer import Recognizer
from .types import ASRResult, Segment, TranscriptResult

__version__ = "0.1.0"

__all__ = [
    "Recognizer",
    "RecognizerConfig",
    "Audio",
    "record",
    "TranscriptResult",
    "ASRResult",
    "Segment",
    "errors",
    "__version__",
]
