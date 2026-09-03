"""Korean speech -> English text, in one step, via faster-whisper.

whisper's task="translate" takes speech in any supported language and emits
English. We pin the source language to Korean and hand it a file path or raw
mic samples. That's the whole library.

Per-stage timing goes to stderr as `[timing] <stage>: <s>` lines; set
V2C_TIMING=0 to silence them.
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
DEVICE = os.environ.get("V2C_DEVICE") or "cpu"  # cpu | cuda | auto
# Follows the device: int8 on CPU, float16 on a GPU. Override only for odd cards
# -- an old Pascal (GTX 10xx) is slow at float16 and wants V2C_COMPUTE=int8.
COMPUTE = os.environ.get("V2C_COMPUTE") or ("int8" if DEVICE == "cpu" else "float16")

_model = None
_TIMING_OFF = {"0", "false", "no", "off", ""}


def _log(stage: str, seconds: float, note: str = "") -> None:
    if os.environ.get("V2C_TIMING", "1").strip().lower() not in _TIMING_OFF:
        tail = " {}".format(note) if note else ""
        print("[timing] {}: {:.2f}s{}".format(stage, seconds, tail), file=sys.stderr, flush=True)


def _load():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper is required: pip install -e .") from exc
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
    return _model


def warmup() -> None:
    """Load the model and run one throwaway inference, so the first real
    `transcribe()` pays only infer time. Call once at process start; then keep
    the process alive and call `transcribe()` back-to-back.
    """
    import numpy as np

    transcribe(np.zeros(16000, dtype=np.float32))


def transcribe(audio: Union[str, "np.ndarray"]) -> str:
    """Korean speech -> English text.

    `audio` is an audio file path (wav/m4a/mp3/... -- faster-whisper decodes it)
    or 16 kHz mono float32 samples (as returned by `record()`).
    """
    cold = _model is None
    t0 = time.perf_counter()
    model = _load()
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    segments, _ = model.transcribe(
        audio,
        language="ko",
        task="translate",
        vad_filter=True,  # trims silence, cuts hallucination on short clips
        condition_on_previous_text=False,
    )
    # segments is lazy -- the real work (decode for files + VAD + inference)
    # happens while it is consumed here.
    text = " ".join(s.text.strip() for s in segments).strip()
    t_infer = time.perf_counter() - t1

    _log("model_load", t_load, "(cold start)" if cold else "(cached)")
    _log("infer", t_infer)
    _log("total", t_load + t_infer)
    return text
