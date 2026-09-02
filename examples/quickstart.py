"""Transcribe an audio file.

    pip install -e .[whisper]
    python examples/quickstart.py voice.m4a
"""

import sys

from voice_to_command import Recognizer

rec = Recognizer.from_dict(
    {
        "asr": {"backend": "whisper", "language": "auto", "whisper": {"model": "small"}},
        "normalize": {"lowercase": True, "strip_punctuation": True},
        "timing": True,
    }
)

result = rec.transcribe(sys.argv[1])
print("text     :", result.text)
print("raw      :", result.raw_text)
print("language :", result.language)
print("timings  :", result.timings)
