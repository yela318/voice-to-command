# voice-to-command

Korean speech → English text, in one step, via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`task="translate"`).
Give it a mic recording or an audio file; get back an English string. Whatever
consumes the string is wired up elsewhere.

## Install

```bash
git clone https://github.com/yela318/voice-to-command.git
cd voice-to-command
pip install -e .          # add .[mic] for microphone input
```

## Use

```bash
v2c voice_kor.m4a         # a file → English text on stdout
v2c --listen             # the mic (Enter to start, Enter to stop) → English text
```

```python
from voice_to_command import transcribe, record

transcribe("voice_kor.m4a")     # -> "Please give me carrots."
transcribe(record())            # microphone
```

`transcribe()` accepts an audio file path (wav/m4a/mp3/… — faster-whisper
decodes it) or 16 kHz mono float32 samples. It returns the English string and
prints `[<seconds>] <text>` to stderr.

## Notes

- **Model:** faster-whisper `small`, multilingual, CPU. `tiny`/`base` mis-hear
  Korean. Change it with `V2C_MODEL=medium …` or by editing `MODEL_SIZE` in
  `src/voice_to_command/core.py`.
- **Cold start:** the first call downloads (~460 MB) and loads the model (~5 s),
  then caches it for the process — reuse one process for repeated calls.
- **Direction:** whisper's translate task only ever produces **English**.
- `voice_kor.m4a` is a sample recording ("당근을 주세요"); `voice.m4a` is an
  English one.

## Layout

```
src/voice_to_command/
  core.py      _load() caches the model; transcribe(path | samples) -> str
  capture.py   record() — push-to-talk mic, returns float32 samples
  cli.py       v2c <file> | v2c --listen
```

MIT licensed — see [LICENSE](LICENSE).
