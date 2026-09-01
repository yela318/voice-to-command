"""The Recognizer: audio -> ASR -> (translate?) -> normalize -> TranscriptResult.

Holds one backend instance; the model loads lazily on the first transcribe()
and is cached for the life of the Recognizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from . import timing
from .audio import Audio, AudioLike
from .config import RecognizerConfig
from .normalize import normalize
from .registry import get_asr_backend, get_translator
from .types import TranscriptResult

_TRANSLATE_ALWAYS = "always"
_TRANSLATE_NEVER = "never"


class Recognizer:
    def __init__(self, config: Optional[RecognizerConfig] = None):
        self.config = (config or RecognizerConfig()).with_env_overrides()
        self._backend = None
        self._translator = None

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "Recognizer":
        return cls(RecognizerConfig.from_toml(path))

    @classmethod
    def from_dict(cls, data: dict) -> "Recognizer":
        return cls(RecognizerConfig.from_dict(data))

    # -- lazily built collaborators --------------------------------------
    @property
    def backend(self):
        if self._backend is None:
            cls = get_asr_backend(self.config.asr.backend)
            builder = getattr(cls, "from_config", None)
            self._backend = builder(self.config.asr) if builder else cls()
        return self._backend

    def _get_translator(self):
        if self._translator is None:
            self._translator = get_translator(self.config.translate.backend)()
        return self._translator

    # -- main entry ------------------------------------------------------
    def transcribe(self, audio: AudioLike, *, language: Optional[str] = None) -> TranscriptResult:
        timing.set_enabled(self.config.timing)
        aud = Audio.coerce(audio)

        cfg_lang = self.config.asr.language
        want_lang = language or (None if cfg_lang == "auto" else cfg_lang)

        timings = {}
        with timing.stage("asr", timings):
            asr = self.backend.transcribe(aud, language=want_lang)
        # Fold in any per-stage split the backend measured (e.g. model_load vs
        # infer); "asr" above stays as the total.
        timings.update(getattr(asr, "timings", None) or {})

        raw = asr.text
        spoken = asr.language or want_lang
        target = self.config.translate.target
        mode = self.config.translate.mode

        want_translation = mode == _TRANSLATE_ALWAYS or (
            mode != _TRANSLATE_NEVER and spoken not in (None, target)
        )

        text = raw
        source_text = None
        translated = False
        if want_translation:
            if self.backend.supports_translation_to(target):
                # The backend already emitted target-language text (e.g. Whisper
                # task='translate'); nothing more to do but record the fact.
                translated = True
            else:
                source_text = raw
                with timing.stage("translate", timings):
                    text = self._get_translator().translate(raw, source=spoken, target=target)
                translated = True

        final = normalize(text, self.config.normalize)
        return TranscriptResult(
            text=final,
            raw_text=raw,
            source_text=source_text,
            language=spoken,
            translated=translated,
            timings=timings,
            segments=asr.segments,
        )
