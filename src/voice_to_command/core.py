"""Korean speech -> English text, in one step, via faster-whisper.

whisper's task="translate" takes speech in any supported language and emits
English. We pin the source language to Korean and hand it a file path or raw
mic samples. That's the whole library.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy as np

# faster-whisper model size. Must be multilingual (no ".en"). "tiny"/"base"
# mis-hear Korean; "small" is the practical floor, "medium"/"large-v3" cost more
# for marginal gains on short commands. Override with V2C_MODEL.
MODEL_SIZE = os.environ.get("V2C_MODEL", "small")

_model = None


def _load():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is required: pip install -e ."
            ) from exc
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(audio: Union[str, "np.ndarray"]) -> str:
    """Korean speech -> English text.

    `audio` is an audio file path (wav/m4a/mp3/... -- faster-whisper decodes it)
    or 16 kHz mono float32 samples (as returned by `record()`).
    """
    t0 = time.perf_counter()
    segments, _ = _load().transcribe(
        audio,
        language="ko",
        task="translate",
        vad_filter=True,  # trims silence, cuts hallucination on short clips
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    print("[{:.1f}s] {}".format(time.perf_counter() - t0, text), file=sys.stderr)
    return text
