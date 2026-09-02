"""voice-to-command -- turn a spoken command into a normalized text string.

    from voice_to_command import Recognizer, record

    rec = Recognizer.from_dict({"asr": {"backend": "whisper"}})
    result = rec.transcribe("command.wav")   # or rec.transcribe(record())
    print(result.text)

The library does audio -> ASR -> normalization, and nothing else: no robot,
policy, or simulation concepts. ASR backends are pluggable (see
voice_to_command/registry.py). The only translation is whisper's own
task="translate" (source speech -> English in one pass).
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
