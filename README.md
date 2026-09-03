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
v2c --serve              # stay resident: warm the model once, then loop
```

`v2c <file>` and `v2c --listen` each spawn a fresh process, so they pay the
model cold start (~5 s) every time. `v2c --serve` warms once and keeps going:
mic push-to-talk on repeat when run in a terminal, or one audio-file path per
line from stdin when piped (`printf 'a.m4a\nb.m4a\n' | v2c --serve`). Ctrl-C or
EOF quits.

```python
from voice_to_command import transcribe, record

transcribe("voice_kor.m4a")     # -> "Please give me carrots."
transcribe(record())            # microphone
```

For a long-running process, warm the model once at startup; every call after
that is just infer time:

```python
from voice_to_command import transcribe, record, warmup

warmup()                        # ~5 s once
while True:
    text = transcribe(record())  # model_load: 0.00s (cached)
    ...                          # hand `text` to whatever comes next
```

`transcribe()` accepts an audio file path (wav/m4a/mp3/… — faster-whisper
decodes it) or 16 kHz mono float32 samples, and returns the English string.
Per-stage timing goes to stderr:

```
[timing] model_load: 6.21s (cold start)   # ~0.00s (cached) on later calls
[timing] infer: 1.83s                      # decode + VAD + translation
[timing] total: 8.04s
```

## Config (env vars)

| var | default | |
|---|---|---|
| `V2C_MODEL` | `small` | faster-whisper size; multilingual only (no `.en`). `tiny`/`base` mis-hear Korean |
| `V2C_DEVICE` | `cpu` | `cpu` \| `cuda` \| `auto` |
| `V2C_COMPUTE` | auto | picked from what CTranslate2 reports for the device: `float16` on a card that does it efficiently, else `int8`. Set it only to force something else |
| `V2C_TIMING` | `1` | `0` silences the `[timing]` lines |

```bash
V2C_DEVICE=cuda v2c --serve             # GPU
$env:V2C_DEVICE="cuda"; v2c --serve     # same on Windows PowerShell
```

CTranslate2 needs the **CUDA 12** runtime (`libcublas.so.12`, cuDNN 9) — a CUDA
13 install does not satisfy it, the soname differs. `pip install
nvidia-cublas-cu12 nvidia-cudnn-cu12` is enough: `_load()` preloads those wheels
itself, so `LD_LIBRARY_PATH` does not have to be set.

## Notes

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
