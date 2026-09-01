"""Local speech-to-text via faster-whisper.  Extra: voice-to-command[whisper]."""

from __future__ import annotations

from typing import Optional

from ..audio import Audio
from ..config import ASRConfig
from ..errors import BackendNotAvailable
from ..registry import register_asr
from ..types import ASRResult, Segment


@register_asr("whisper")
class WhisperBackend:
    target_sample_rate = 16000

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "",
        task: str = "translate",
    ):
        self.model_size = model
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        # "translate" always emits English; "transcribe" keeps the spoken language.
        self.task = task
        self._model = None

    @classmethod
    def from_config(cls, asr: ASRConfig) -> "WhisperBackend":
        w = asr.whisper
        return cls(
            model=w.model,
            device=w.device,
            compute_type=w.compute_type,
            **{k: v for k, v in asr.options.items() if k == "task"},
        )

    def supports_translation_to(self, language: str) -> bool:
        return self.task == "translate" and language == "en"

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise BackendNotAvailable(
                    "whisper backend needs: pip install voice-to-command[whisper]"
                ) from exc
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio: Audio, *, language: Optional[str]) -> ASRResult:
        model = self._load()
        # faster-whisper decodes files itself; hand it the path when we have one.
        if audio.is_file():
            source = str(audio.path)
        else:
            source = audio.samples(self.target_sample_rate)[0]

        segments, info = model.transcribe(
            source,
            language=language,
            task=self.task,
            vad_filter=True,  # trims silence, cuts hallucination on short clips
            condition_on_previous_text=False,  # each command is independent
        )
        segs = tuple(
            Segment(text=s.text.strip(), start=s.start, end=s.end) for s in segments
        )
        text = " ".join(s.text for s in segs).strip()
        return ASRResult(text=text, language=getattr(info, "language", None), segments=segs)
