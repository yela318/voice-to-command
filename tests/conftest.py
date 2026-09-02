"""Shared fixtures + fake backends so the core can be tested with no heavy deps."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from voice_to_command.registry import register_asr
from voice_to_command.types import ASRResult, Segment


@register_asr("fake")
class FakeASRBackend:
    """Returns a canned result; records what it was called with."""

    target_sample_rate = 16000

    def __init__(self, text="pick up the black ball", language="en", can_translate_to=None):
        self.text = text
        self.language = language
        self.can_translate_to = can_translate_to
        self.calls = []

    @classmethod
    def from_config(cls, asr):
        return cls(**asr.options)

    def supports_translation_to(self, language):
        return language == self.can_translate_to

    def transcribe(self, audio, *, language):
        self.calls.append({"audio": audio, "language": language})
        return ASRResult(
            text=self.text,
            language=self.language,
            segments=(Segment(text=self.text, start=0.0, end=1.0),),
            timings={"asr.fake_stage": 0.0},
        )


@pytest.fixture
def wav_path(tmp_path):
    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    pcm = (tone * 32767).astype("<i2")
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path
