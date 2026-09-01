from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from voice_to_command.audio import Audio, _resample_linear


def test_coerce_variants(wav_path):
    assert Audio.coerce(str(wav_path)).is_file()
    assert Audio.coerce(wav_path).is_file()
    assert Audio.coerce(b"RIFF....")._encoded is not None
    a = Audio.coerce((np.zeros(10, dtype=np.float32), 16000))
    assert a.has_samples()


def test_coerce_bare_ndarray_rejected():
    with pytest.raises(TypeError):
        Audio.coerce(np.zeros(10, dtype=np.float32))


def test_from_samples_normalizes_dtype_and_shape():
    a = Audio.from_samples([[1], [2], [3]], 8000)
    s, sr = a.samples()
    assert s.dtype == np.float32 and s.shape == (3,) and sr == 8000


def test_wav_roundtrip_from_samples():
    tone = (0.2 * np.sin(np.linspace(0, 20, 16000))).astype(np.float32)
    a = Audio.from_samples(tone, 16000)
    blob = a.wav_bytes()
    with wave.open(io.BytesIO(blob), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2


def test_decode_wav_file(wav_path):
    s, sr = Audio.from_file(wav_path).samples()
    assert sr == 16000 and len(s) == 16000


def test_resample_changes_length():
    x = np.zeros(16000, dtype=np.float32)
    assert len(_resample_linear(x, 16000, 8000)) == 8000
    assert _resample_linear(x, 16000, 16000) is x


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Audio.from_file("nope-does-not-exist.wav")
