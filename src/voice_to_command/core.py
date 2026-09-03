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
# None -> _load() asks CTranslate2 what this card actually supports. Set
# V2C_COMPUTE only to force something else.
COMPUTE = os.environ.get("V2C_COMPUTE") or None

_model = None
_TIMING_OFF = {"0", "false", "no", "off", ""}


def _log(stage: str, seconds: float, note: str = "") -> None:
    if os.environ.get("V2C_TIMING", "1").strip().lower() not in _TIMING_OFF:
        tail = " {}".format(note) if note else ""
        print("[timing] {}: {:.2f}s{}".format(stage, seconds, tail), file=sys.stderr, flush=True)


def _compute_type(device: str) -> str:
    """Best compute type CTranslate2 will actually accept here. float16 where the
    card does it efficiently (Ada/Blackwell), int8 on CPU and on older cards --
    CT2 raises ValueError for float16 below compute capability 7.0 (Pascal).
    """
    import ctranslate2

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    supported = ctranslate2.get_supported_compute_types(device)
    for candidate in ("float16", "int8"):
        if candidate in supported:
            return candidate
    return "float32"


def _preload_cuda_libs() -> None:
    """Make the pip-installed CUDA runtime findable without LD_LIBRARY_PATH.

    CTranslate2 dlopen()s `libcublas.so.12` / `libcudnn*.so.9` by soname, but the
    `nvidia-*-cu12` wheels drop them inside site-packages, where the loader does
    not look -- and LD_LIBRARY_PATH only counts if it was set before the process
    started. Loading them here with RTLD_GLOBAL means the later dlopen finds them
    already resident. Best effort: anything missing just stays missing.
    """
    import ctypes
    import glob
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        return
    for root in spec.submodule_search_locations or ():
        for so in sorted(glob.glob(os.path.join(root, "*", "lib", "*.so.*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _load():
    global _model, COMPUTE
    if _model is None:
        if DEVICE != "cpu":
            _preload_cuda_libs()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper is required: pip install -e .") from exc
        COMPUTE = COMPUTE or _compute_type(DEVICE)
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
    return _model


def warmup() -> None:
    """Load the model and run one throwaway inference, so the first real
    `transcribe()` pays only infer time. Call once at process start; then keep
    the process alive and call `transcribe()` back-to-back.
    """
    import numpy as np

    t0 = time.perf_counter()
    model = _load()
    # vad_filter=False on purpose: VAD drops a silent clip entirely, no segment
    # is produced and the encoder never runs -- which hides a broken CUDA setup
    # until the first real recording. Consume the generator so it actually runs.
    segments, _ = model.transcribe(
        np.zeros(16000, dtype=np.float32), language="ko", task="translate", vad_filter=False
    )
    list(segments)
    _log("warmup", time.perf_counter() - t0, "({}/{})".format(DEVICE, COMPUTE))


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

    _log("model_load", t_load, "({}, {}/{})".format(
        "cold start" if cold else "cached", DEVICE, COMPUTE))
    _log("infer", t_infer)
    _log("total", t_load + t_infer)
    return text
