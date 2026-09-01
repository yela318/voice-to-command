"""Real Naver CSR / Papago calls. Skipped unless requests + credentials exist."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("requests", reason="pip install voice-to-command[naver]")
pytestmark = pytest.mark.network

_HAS_CSR = {"NCP_CLIENT_ID", "NCP_CLIENT_SECRET"} <= set(os.environ)
_HAS_PAPAGO = {"NCP_PAPAGO_CLIENT_ID", "NCP_PAPAGO_CLIENT_SECRET"} <= set(os.environ)


@pytest.mark.skipif(not _HAS_CSR, reason="NCP_CLIENT_ID / NCP_CLIENT_SECRET not set")
def test_csr_english(wav_path):
    from voice_to_command import Recognizer

    rec = Recognizer.from_dict({"asr": {"backend": "naver_csr", "language": "en"}})
    result = rec.transcribe(wav_path)
    assert isinstance(result.text, str)


@pytest.mark.skipif(not _HAS_PAPAGO, reason="NCP_PAPAGO_* not set")
def test_papago_ko_en():
    from voice_to_command.backends.naver_papago import PapagoTranslator

    out = PapagoTranslator().translate("안녕하세요", source="ko", target="en")
    assert isinstance(out, str) and out
