"""Microphone capture. Push-to-talk: Enter to start, Enter to stop.

Needs the [mic] extra (sounddevice) and the PortAudio system library.
"""

from __future__ import annotations

import os
import sys

import numpy as np

SAMPLE_RATE = 16000  # what whisper expects
# Which input to record from: an index or a substring of the device name (e.g.
# V2C_MIC=Britz). Unset -> the system default, which on a desktop is usually the
# onboard analog jack rather than the USB headset you actually mean.
MIC = os.environ.get("V2C_MIC") or None


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

    device = int(MIC) if MIC and MIC.isdigit() else MIC
    info = sd.query_devices(device, kind="input")  # raises if the name matches nothing

    # Most USB headsets only offer 44100/48000; asking for 16000 makes PortAudio
    # raise "Invalid sample rate". Record at whatever the device does, resample
    # after.
    rate = sample_rate
    try:
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)
    except Exception:
        rate = int(info["default_samplerate"])
    print(
        "mic: {} @ {} Hz{}".format(
            info["name"], rate, "" if rate == sample_rate else " -> resampled to {}".format(sample_rate)
        ),
        file=sys.stderr,
    )

    input("Press Enter to start recording...")
    print("Recording... press Enter again to stop.", flush=True)

    frames = []

    def callback(indata, _frame_count, _time_info, status):
        if status:
            print(status, flush=True)
        frames.append(indata.copy())

    with sd.InputStream(
        device=device, samplerate=rate, channels=1, dtype="float32", callback=callback
    ):
        input()

    if not frames:
        raise RuntimeError("no audio was captured -- check the microphone input")
    return _resample(np.concatenate(frames, axis=0).reshape(-1), rate, sample_rate)


def _resample(samples: "np.ndarray", src: int, dst: int) -> "np.ndarray":
    """Linear resample. Good enough for speech -- whisper cares about the 0-8 kHz
    band and tolerates the mild aliasing this leaves behind.
    """
    if src == dst or samples.size == 0:
        return samples
    n = int(round(samples.size * dst / float(src)))
    at = np.linspace(0, samples.size - 1, n)
    return np.interp(at, np.arange(samples.size), samples).astype(np.float32)
