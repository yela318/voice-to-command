"""Push-to-talk: record from the mic, then transcribe.

    pip install voice-to-command[whisper,mic]
    python examples/live_mic.py
"""

from voice_to_command import Recognizer, record

rec = Recognizer.from_dict({"asr": {"backend": "whisper"}, "timing": True})

audio = record()  # Enter to start, Enter to stop
result = rec.transcribe(audio)
print(result.text)
