"""Real faster-whisper run. Skipped unless faster-whisper is installed."""

from __future__ import annotations

import wave

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="pip install -e .")
pytestmark = pytest.mark.slow


def test_transcribe_samples_returns_str():
    from voice_to_command import transcribe

    silence = np.zeros(16000, dtype=np.float32)
    assert isinstance(transcribe(silence), str)


def test_transcribe_file_returns_str(tmp_path):
    from voice_to_command import transcribe

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    assert isinstance(transcribe(str(path)), str)
