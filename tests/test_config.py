from __future__ import annotations

import textwrap

import pytest

from voice_to_command.config import RecognizerConfig


def test_defaults():
    cfg = RecognizerConfig()
    assert cfg.asr.backend == "whisper"
    assert cfg.asr.whisper.model == "small"
    assert cfg.translate.mode == "auto"
    assert cfg.timing is True


def test_from_dict_nested_and_unknown_keys_go_to_options():
    cfg = RecognizerConfig.from_dict(
        {
            "asr": {
                "backend": "naver_csr",
                "language": "ko",
                "whisper": {"model": "base", "device": "cuda"},
                "timeout": 30,  # unknown ASRConfig field -> options
            },
            "normalize": {"lowercase": True, "phrase_map": {"a": "b"}},
            "timing": True,
        }
    )
    assert cfg.asr.backend == "naver_csr"
    assert cfg.asr.whisper.model == "base"
    assert cfg.asr.whisper.device == "cuda"
    assert cfg.asr.options == {"timeout": 30}
    assert cfg.normalize.lowercase is True
    assert cfg.normalize.phrase_map == {"a": "b"}
    assert cfg.timing is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("V2C_ASR_BACKEND", "naver_csr")
    monkeypatch.setenv("V2C_WHISPER_MODEL", "tiny")
    monkeypatch.setenv("V2C_WHISPER_CPU_THREADS", "4")
    monkeypatch.setenv("V2C_TIMING", "1")
    cfg = RecognizerConfig().with_env_overrides()
    assert cfg.asr.backend == "naver_csr"
    assert cfg.asr.whisper.model == "tiny"
    assert cfg.asr.whisper.cpu_threads == 4
    assert cfg.timing is True


def test_cpu_threads_defaults_zero_and_reads_from_dict():
    assert RecognizerConfig().asr.whisper.cpu_threads == 0
    cfg = RecognizerConfig.from_dict({"asr": {"whisper": {"cpu_threads": 8}}})
    assert cfg.asr.whisper.cpu_threads == 8


def test_env_overrides_does_not_mutate_original(monkeypatch):
    monkeypatch.setenv("V2C_ASR_BACKEND", "naver_csr")
    base = RecognizerConfig()
    _ = base.with_env_overrides()
    assert base.asr.backend == "whisper"


def test_from_toml(tmp_path):
    pytest.importorskip("tomllib", reason="needs tomllib (3.11+) or tomli")
    p = tmp_path / "voice.toml"
    p.write_text(
        textwrap.dedent(
            """
            timing = true
            [asr]
            backend = "whisper"
            [asr.whisper]
            model = "medium"
            [translate]
            target = "en"
            """
        )
    )
    cfg = RecognizerConfig.from_toml(p)
    assert cfg.asr.whisper.model == "medium"
    assert cfg.timing is True
