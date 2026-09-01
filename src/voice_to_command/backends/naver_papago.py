"""Text translation via Naver Papago NMT.  Extra: voice-to-command[naver].

Papago is a separate NCP Application from CSR, with its own credentials:
    NCP_PAPAGO_CLIENT_ID, NCP_PAPAGO_CLIENT_SECRET
"""

from __future__ import annotations

import os
from typing import Optional

from ..errors import BackendNotAvailable, CredentialsMissing
from ..registry import register_translator

_API_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"


@register_translator("naver_papago")
class PapagoTranslator:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def translate(self, text: str, *, source: Optional[str], target: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise BackendNotAvailable("papago needs: pip install voice-to-command[naver]") from exc

        try:
            client_id = os.environ["NCP_PAPAGO_CLIENT_ID"]
            client_secret = os.environ["NCP_PAPAGO_CLIENT_SECRET"]
        except KeyError as exc:
            raise CredentialsMissing(
                "set NCP_PAPAGO_CLIENT_ID / NCP_PAPAGO_CLIENT_SECRET for the naver_papago translator"
            ) from exc

        response = requests.post(
            _API_URL,
            headers={
                "X-NCP-APIGW-API-KEY-ID": client_id,
                "X-NCP-APIGW-API-KEY": client_secret,
            },
            data={"source": source or "auto", "target": target, "text": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["result"]["translatedText"]
