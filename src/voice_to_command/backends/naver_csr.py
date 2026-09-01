"""Speech-to-text via Naver CLOVA Speech Recognition (CSR).  Extra: voice-to-command[naver].

Built for short (<= 60s, <= 3MB) command-style audio. CSR only transcribes
-- it does not translate -- so pair it with the naver_papago translator for
non-English speech. Credentials come from the environment:
    NCP_CLIENT_ID, NCP_CLIENT_SECRET
"""

from __future__ import annotations

import os
from typing import Optional

from ..audio import Audio
from ..config import ASRConfig
from ..errors import BackendNotAvailable, CredentialsMissing, LanguageRequired
from ..registry import register_asr
from ..types import ASRResult

_API_URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt"
_LANG = {"ko": "Kor", "en": "Eng", "ja": "Jpn", "zh": "Chn"}


@register_asr("naver_csr")
class NaverCSRBackend:
    target_sample_rate = 16000

    def __init__(self, language: str = "auto", timeout: float = 15.0):
        self.language = language
        self.timeout = timeout

    @classmethod
    def from_config(cls, asr: ASRConfig) -> "NaverCSRBackend":
        opts = {k: v for k, v in asr.options.items() if k in {"timeout"}}
        return cls(language=asr.language, **opts)

    def supports_translation_to(self, language: str) -> bool:
        return False

    def transcribe(self, audio: Audio, *, language: Optional[str]) -> ASRResult:
        try:
            import requests
        except ImportError as exc:
            raise BackendNotAvailable("naver backend needs: pip install voice-to-command[naver]") from exc

        lang = language or (None if self.language == "auto" else self.language)
        if lang is None:
            raise LanguageRequired(
                "naver_csr needs an explicit language (e.g. 'ko' or 'en'); 'auto' is unsupported"
            )
        csr_lang = _LANG.get(lang, lang)

        try:
            client_id = os.environ["NCP_CLIENT_ID"]
            client_secret = os.environ["NCP_CLIENT_SECRET"]
        except KeyError as exc:
            raise CredentialsMissing(
                "set NCP_CLIENT_ID / NCP_CLIENT_SECRET for the naver_csr backend"
            ) from exc

        # CSR accepts common containers (m4a/aac/...) directly; only synthesize
        # a WAV when we're holding raw mic samples.
        payload = audio.raw_bytes() if audio.is_file() else audio.wav_bytes(self.target_sample_rate)

        response = requests.post(
            _API_URL,
            params={"lang": csr_lang},
            headers={
                "x-ncp-apigw-api-key-id": client_id,
                "x-ncp-apigw-api-key": client_secret,
                "Content-Type": "application/octet-stream",
            },
            data=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return ASRResult(text=response.json()["text"], language=lang)
