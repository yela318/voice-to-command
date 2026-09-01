from __future__ import annotations

import pytest

from voice_to_command import Recognizer


def _rec(**overrides):
    cfg = {
        "asr": {"backend": "fake", "language": "auto"},
        "translate": {"backend": "fake_translator", "target": "en", "mode": "auto"},
    }
    for section, value in overrides.items():
        merged = dict(cfg.get(section, {}))
        merged.update(value)
        cfg[section] = merged
    return Recognizer.from_dict(cfg)


def test_plain_english_no_translation(wav_path):
    rec = _rec(asr={"backend": "fake", "options": {"text": "Pick up the bowl", "language": "en"}})
    result = rec.transcribe(wav_path)
    assert result.text == "Pick up the bowl"
    assert result.translated is False
    assert result.source_text is None
    assert result.language == "en"
    assert "asr" in result.timings


def test_korean_gets_translated_via_translator(wav_path):
    rec = _rec(asr={"backend": "fake", "options": {"text": "그릇을 집어", "language": "ko"}})
    result = rec.transcribe(wav_path)
    assert result.translated is True
    assert result.source_text == "그릇을 집어"
    assert result.text == "[ko->en] 그릇을 집어"
    assert "translate" in result.timings


def test_backend_that_self_translates_skips_translator(wav_path):
    rec = _rec(
        asr={
            "backend": "fake",
            "options": {"text": "pick up the bowl", "language": "ko", "can_translate_to": "en"},
        }
    )
    result = rec.transcribe(wav_path)
    assert result.translated is True
    assert result.source_text is None  # backend already emitted English
    assert result.text == "pick up the bowl"
    assert "translate" not in result.timings


def test_mode_never_disables_translation(wav_path):
    rec = _rec(
        asr={"backend": "fake", "options": {"text": "그릇", "language": "ko"}},
        translate={"backend": "fake_translator", "target": "en", "mode": "never"},
    )
    result = rec.transcribe(wav_path)
    assert result.translated is False
    assert result.text == "그릇"


def test_normalization_applied(wav_path):
    rec = Recognizer.from_dict(
        {
            "asr": {"backend": "fake", "options": {"text": "Pick up the black ball.", "language": "en"}},
            "normalize": {
                "lowercase": True,
                "strip_punctuation": True,
                "phrase_map": {"black ball": "black bowl"},
            },
        }
    )
    result = rec.transcribe(wav_path)
    assert result.text == "pick up the black bowl"
    assert result.raw_text == "Pick up the black ball."


def test_explicit_language_arg_passed_to_backend(wav_path):
    rec = _rec(asr={"backend": "fake", "options": {"text": "x", "language": "en"}})
    rec.transcribe(wav_path, language="ko")
    assert rec.backend.calls[0]["language"] == "ko"


def test_unknown_backend_raises():
    from voice_to_command.errors import BackendNotAvailable

    rec = Recognizer.from_dict({"asr": {"backend": "does-not-exist"}})
    with pytest.raises(BackendNotAvailable):
        _ = rec.backend
