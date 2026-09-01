"""Register an out-of-tree ASR backend and use it by name.

Any object with .target_sample_rate, .supports_translation_to(lang) and
.transcribe(audio, *, language) works -- no base class required.

    python examples/custom_backend.py path/to/anything.wav
"""

import sys

from voice_to_command import Recognizer
from voice_to_command.registry import register_asr
from voice_to_command.types import ASRResult


@register_asr("echo_filename")
class EchoFilenameBackend:
    target_sample_rate = 16000

    def supports_translation_to(self, language):
        return False

    def transcribe(self, audio, *, language):
        name = audio.path.stem if audio.is_file() else "microphone"
        return ASRResult(text=name.replace("_", " "), language=language or "en")


rec = Recognizer.from_dict({"asr": {"backend": "echo_filename"}})
print(rec.transcribe(sys.argv[1]).text)
