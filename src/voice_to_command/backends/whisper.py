"""Local speech-to-text via faster-whisper.  Extra: voice-to-command[whisper]."""

from __future__ import annotations

from typing import Optional

from .. import timing
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
        cpu_threads: int = 0,
    ):
        self.model_size = model
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        # 0 lets faster-whisper pick; a higher count speeds up CPU inference.
        self.cpu_threads = cpu_threads
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
            cpu_threads=w.cpu_threads,
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
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def transcribe(self, audio: Audio, *, language: Optional[str]) -> ASRResult:
        t: dict = {}

        # First call in the process loads the model (cold start); later calls
        # find it cached and this stage is ~0.
        with timing.stage("asr.model_load", t):
            model = self._load()

        # faster-whisper decodes files itself; hand it the path when we have one.
        if audio.is_file():
            source = str(audio.path)
        else:
            with timing.stage("asr.resample", t):
                source = audio.samples(self.target_sample_rate)[0]

        # model.transcribe() is lazy -- the real work happens while the segment
        # generator is consumed, so time the materialization too.
        with timing.stage("asr.infer", t):
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
        return ASRResult(
            text=text, language=getattr(info, "language", None), segments=segs, timings=t
        )
