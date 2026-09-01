"""Real faster-whisper run. Skipped unless the [whisper] extra is installed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="pip install voice-to-command[whisper]")
pytestmark = pytest.mark.slow


def test_transcribe_silence_returns_string():
    from voice_to_command import Recognizer

    rec = Recognizer.from_dict({"asr": {"backend": "whisper", "whisper": {"model": "tiny"}}})
    silence = (np.zeros(16000, dtype=np.float32), 16000)
    result = rec.transcribe(silence)
    assert isinstance(result.text, str)
    assert "asr" in result.timings
