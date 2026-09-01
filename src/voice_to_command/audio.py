"""The Audio type: one normalized representation for every kind of input.

Accepts a file path, encoded bytes, a (samples, sample_rate) pair, or an
existing Audio. Decoding compressed containers to PCM is done lazily and
needs the [audio] extra (PyAV); WAV is handled with the stdlib.
"""

from __future__ import annotations

import dataclasses
import io
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .errors import DecodeError

AudioLike = Union[str, Path, bytes, bytearray, "Audio", Tuple[np.ndarray, int]]


@dataclasses.dataclass
class Audio:
    _samples: Optional[np.ndarray] = None
    _sample_rate: Optional[int] = None
    _path: Optional[Path] = None
    _encoded: Optional[bytes] = None

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Audio":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return cls(_path=p)

    @classmethod
    def from_samples(cls, samples: np.ndarray, sample_rate: int) -> "Audio":
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        return cls(_samples=arr, _sample_rate=int(sample_rate))

    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray]) -> "Audio":
        """Encoded audio bytes (a wav/m4a/... container), not raw PCM."""
        return cls(_encoded=bytes(data))

    @classmethod
    def coerce(cls, obj: AudioLike) -> "Audio":
        if isinstance(obj, Audio):
            return obj
        if isinstance(obj, (str, Path)):
            return cls.from_file(obj)
        if isinstance(obj, (bytes, bytearray)):
            return cls.from_bytes(obj)
        if isinstance(obj, tuple) and len(obj) == 2:
            return cls.from_samples(obj[0], obj[1])
        if isinstance(obj, np.ndarray):
            raise TypeError("raw ndarray needs a rate: pass (samples, sample_rate)")
        raise TypeError("unsupported audio input: {!r}".format(type(obj)))

    # -- accessors ------------------------------------------------------
    @property
    def path(self) -> Optional[Path]:
        return self._path

    def is_file(self) -> bool:
        return self._path is not None

    def has_samples(self) -> bool:
        return self._samples is not None

    def raw_bytes(self) -> bytes:
        """Original encoded container bytes when we have them (many cloud
        APIs accept m4a/aac directly); otherwise a freshly encoded WAV."""
        if self._encoded is not None:
            return self._encoded
        if self._path is not None:
            return self._path.read_bytes()
        return self.wav_bytes()

    def samples(self, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Decode to float32 mono in [-1, 1], resampled to `sample_rate` if given."""
        s, sr = self._decode()
        if sample_rate is not None and sr != sample_rate:
            s = _resample_linear(s, sr, sample_rate)
            sr = sample_rate
        return s, sr

    def wav_bytes(self, sample_rate: Optional[int] = None) -> bytes:
        s, sr = self.samples(sample_rate)
        pcm16 = (np.clip(s, -1.0, 1.0) * 32767.0).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def duration_seconds(self) -> Optional[float]:
        if self._samples is not None and self._sample_rate:
            return len(self._samples) / self._sample_rate
        return None

    # -- internals ------------------------------------------------------
    def _decode(self) -> Tuple[np.ndarray, int]:
        if self._samples is not None and self._sample_rate is not None:
            return self._samples, self._sample_rate

        blob = self._encoded if self._encoded is not None else (
            self._path.read_bytes() if self._path is not None else None
        )
        if blob is None:
            raise DecodeError("Audio has no samples and no source to decode")

        # Fast path: WAV via the standard library, no extra needed.
        try:
            return _decode_wav(blob)
        except wave.Error:
            pass

        # Anything else (m4a/mp3/ogg/flac) needs PyAV.
        try:
            import av  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise DecodeError(
                "decoding compressed audio needs the [audio] extra: pip install voice-to-command[audio]"
            ) from exc
        return _decode_with_av(av, blob)


def _decode_wav(blob: bytes) -> Tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(blob), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise DecodeError("only 16-bit PCM WAV is supported without the [audio] extra")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def _decode_with_av(av, blob: bytes) -> Tuple[np.ndarray, int]:  # pragma: no cover
    with av.open(io.BytesIO(blob)) as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        chunks = []
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise DecodeError("no audio frames decoded")
    return np.concatenate(chunks).astype(np.float32), sr


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample -- adequate for short speech commands. For
    high-quality resampling, decode via the [audio] extra instead."""
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / float(sr_in)))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)
