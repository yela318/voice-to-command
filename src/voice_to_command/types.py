"""Data types and the backend protocols.

A backend is any object with the ASRBackend shape below; it does not need
to subclass anything. Register it with @register_asr (see registry.py).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, Optional, Tuple

try:  # Protocol is stdlib on 3.8+, runtime_checkable too
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - very old Python
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

if TYPE_CHECKING:
    from .audio import Audio


@dataclasses.dataclass(frozen=True)
class Segment:
    """One decoded chunk from an ASR backend."""

    text: str
    start: Optional[float] = None
    end: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class ASRResult:
    """What an ASR backend returns: text in whatever language it produced."""

    text: str
    language: Optional[str] = None
    segments: Tuple[Segment, ...] = ()


@dataclasses.dataclass(frozen=True)
class TranscriptResult:
    """The Recognizer's output after ASR -> translation -> normalization."""

    text: str  # final string: normalized, in the configured target language
    raw_text: str  # ASR output, before normalization
    source_text: Optional[str]  # text before translation (None if not translated)
    language: Optional[str]  # detected / used spoken language
    translated: bool
    timings: Dict[str, float]  # {"asr": 1.83, "translate": 0.31}
    segments: Tuple[Segment, ...] = ()


@runtime_checkable
class ASRBackend(Protocol):
    """Audio in, text out. Instances are reusable and cache their model."""

    name: str
    target_sample_rate: int

    def supports_translation_to(self, language: str) -> bool:
        """True if this backend can emit `language` directly (e.g. Whisper's
        task='translate' to English), letting the Recognizer skip the
        separate Translator step."""
        ...

    def transcribe(self, audio: "Audio", *, language: Optional[str]) -> ASRResult:
        ...


@runtime_checkable
class Translator(Protocol):
    name: str

    def translate(self, text: str, *, source: Optional[str], target: str) -> str:
        ...
