from __future__ import annotations

import pytest

from voice_to_command.errors import BackendNotAvailable
from voice_to_command.registry import (
    available_asr_backends,
    get_asr_backend,
    get_translator,
    register_asr,
)


def test_builtin_names_listed():
    names = set(available_asr_backends())
    assert {"whisper", "naver_csr"}.issubset(names)


def test_fake_backend_registered_by_conftest():
    assert "fake" in set(available_asr_backends())
    assert get_asr_backend("fake").__name__ == "FakeASRBackend"


def test_unknown_backend_raises():
    with pytest.raises(BackendNotAvailable):
        get_asr_backend("no-such-backend")


def test_unknown_translator_raises():
    with pytest.raises(BackendNotAvailable):
        get_translator("no-such-translator")


def test_register_sets_name_attr():
    @register_asr("temp_backend")
    class Temp:
        target_sample_rate = 16000

        def supports_translation_to(self, language):
            return False

        def transcribe(self, audio, *, language):
            raise NotImplementedError

    assert Temp.name == "temp_backend"
    assert get_asr_backend("temp_backend") is Temp
