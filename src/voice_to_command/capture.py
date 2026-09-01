"""Microphone capture. Push-to-talk only for now: Enter to start, Enter to stop.

Needs the [mic] extra (sounddevice) and the PortAudio system library.
"""

from __future__ import annotations

import numpy as np

from .audio import Audio
from .errors import BackendNotAvailable

DEFAULT_SAMPLE_RATE = 16000  # what Whisper and Naver CSR both expect


def record(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    start_prompt: str = "Press Enter to start recording...",
    stop_prompt: str = "Recording... press Enter again to stop.",
) -> Audio:
    """Block for a push-to-talk recording and return it as an Audio."""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing -> sounddevice raises OSError on import
        raise BackendNotAvailable(
            "microphone capture needs PortAudio "
            "(Linux: apt-get install libportaudio2, Mac: brew install portaudio); "
            + str(exc)
        ) from exc
    except ImportError as exc:
        raise BackendNotAvailable(
            "microphone capture needs the [mic] extra: pip install voice-to-command[mic]"
        ) from exc

    input(start_prompt)
    print(stop_prompt, flush=True)

    frames = []

    def callback(indata, _frame_count, _time_info, status):
        if status:
            print(status, flush=True)
        frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()

    if not frames:
        raise RuntimeError("no audio was captured -- check the microphone input")

    samples = np.concatenate(frames, axis=0).reshape(-1)
    return Audio.from_samples(samples, sample_rate)
