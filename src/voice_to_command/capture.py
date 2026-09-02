"""Microphone capture. Push-to-talk: Enter to start, Enter to stop.

Needs the [mic] extra (sounddevice) and the PortAudio system library.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000  # what whisper expects


def record(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Block for a push-to-talk recording; return 16 kHz mono float32 samples."""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing
        raise RuntimeError(
            "microphone capture needs PortAudio "
            "(Linux: apt-get install libportaudio2, Mac: brew install portaudio); " + str(exc)
        ) from exc
    except ImportError as exc:
        raise RuntimeError("microphone capture needs the [mic] extra: pip install -e .[mic]") from exc

    input("Press Enter to start recording...")
    print("Recording... press Enter again to stop.", flush=True)

    frames = []

    def callback(indata, _frame_count, _time_info, status):
        if status:
            print(status, flush=True)
        frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()

    if not frames:
        raise RuntimeError("no audio was captured -- check the microphone input")
    return np.concatenate(frames, axis=0).reshape(-1)
