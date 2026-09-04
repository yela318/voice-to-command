# voice-to-command

Korean (or English) speech → English text, in one step, via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`task="translate"`).
Give it a mic recording or an audio file; get back an English string. The source
language is auto-detected, so Korean and English clips both work. Whatever
consumes the string is wired up elsewhere.

## Install

```bash
git clone https://github.com/yela318/voice-to-command.git
cd voice-to-command
pip install -e .          # add .[mic] for microphone input
```

## Use

```bash
v2c sample/voice_kor.m4a  # a file → English text on stdout
v2c --listen             # the mic (Enter to start, Enter to stop) → English text
v2c --serve              # stay resident: warm the model once, then loop
v2c --serve --http       # GPU box: HTTP inference server for remote clients
v2c <file> --server_ip HOST   # run inference on that remote server, not locally
```

`v2c <file>` and `v2c --listen` each spawn a fresh process, so they pay the
model cold start (~5 s) every time. `v2c --serve` warms once and keeps going:
mic push-to-talk on repeat when run in a terminal, or one audio-file path per
line from stdin when piped (`printf 'a.m4a\nb.m4a\n' | v2c --serve`). Ctrl-C or
EOF quits.

```python
from voice_to_command import transcribe, record

transcribe("sample/voice_kor.m4a")   # -> "Please give me carrots."
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
| `V2C_LANG` | auto | source language. Unset = auto-detect per clip (Korean + English both fine). Set `V2C_LANG=ko` to pin it for Korean-only use |
| `V2C_DEVICE` | `cpu` | `cpu` \| `cuda` \| `auto` |
| `V2C_COMPUTE` | auto | picked from what CTranslate2 reports for the device: `float16` on a card that does it efficiently, else `int8`. Set it only to force something else |
| `V2C_MIC` | system default | which input to record from: a device index or a substring of its name (`V2C_MIC=Britz`). List them with `python -c "import sounddevice as sd; print(sd.query_devices())"` |
| `V2C_SERVER_IP` | — | run inference on a remote `v2c --serve --http` host instead of loading the model locally (same as `--server_ip`) |
| `V2C_SERVER_PORT` | `8756` | port for the server (both the `--serve --http` listener and the client, same as `--server_port`) |
| `V2C_HTTP_HOST` | `0.0.0.0` | interface the `--serve --http` listener binds to |
| `V2C_TIMING` | `1` | `0` silences the `[timing]` lines |

```bash
V2C_DEVICE=cuda v2c --serve             # GPU
$env:V2C_DEVICE="cuda"; v2c --serve     # same on Windows PowerShell
```

CTranslate2 needs the **CUDA 12** runtime (`libcublas.so.12`, cuDNN 9) — a CUDA
13 install does not satisfy it, the soname differs. `pip install
nvidia-cublas-cu12 nvidia-cudnn-cu12` is enough: `_load()` preloads those wheels
itself, so `LD_LIBRARY_PATH` does not have to be set.

```bash
conda install -c conda-forge portaudio
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
V2C_DEVICE=cuda v2c --serve
```

## Remote GPU

Run the model on a box that has a GPU, drive it from a laptop that does not.
The client never loads the model — it ships the audio bytes over plain HTTP and
prints what comes back. stdlib only, no auth: use it on a trusted LAN, or tunnel
it over SSH.

```bash
# on the GPU box — warm once, then answer /transcribe until Ctrl-C
V2C_DEVICE=cuda v2c --serve --http            # binds 0.0.0.0:8756

# on the laptop — inference happens on gpu-box, text comes back
v2c sample/voice_kor.m4a --server_ip gpu-box
v2c --listen             --server_ip gpu-box  # mic is local, inference remote
v2c --serve              --server_ip gpu-box  # resident client loop
export V2C_SERVER_IP=gpu-box                  # or set it once and drop the flag
```

`--server_port` / `V2C_SERVER_PORT` (default `8756`) sets the port on both ends.
Across an untrusted network, tunnel instead of exposing the port:
`ssh -N -L 8756:localhost:8756 gpu-box`, then `--server_ip 127.0.0.1`. To keep
the server up across reboots, wrap `v2c --serve --http` in a systemd unit.

## Notes

- **Cold start:** the first call downloads (~460 MB) and loads the model (~5 s),
  then caches it for the process — reuse one process for repeated calls.
- **Direction:** whisper's translate task only ever produces **English**.
- **Mixed KO/EN:** with `V2C_LANG` unset, whisper detects the language per clip;
  measured ko-probability stays 0.93–0.99 on the short `voice_kor_*` samples and
  English clips come back verbatim.
- `sample/` holds TTS clips: `voice_kor*` (Korean) and `voice_eng_*` (the English
  match) for carrot / banana / lemon / pineapple / apple, plus `voice.m4a`
  (another English one). Each `voice_kor_*` / `voice_eng_<fruit>` pair says the
  same thing in the two languages.

## Layout

```
src/voice_to_command/
  core.py      _load() caches the model; transcribe(path | samples) -> str
  capture.py   record() — push-to-talk mic, returns float32 samples
  remote.py    serve_http() + transcribe_remote() — run inference on a GPU box
  cli.py       v2c <file> | v2c --listen | v2c --serve [--http] [--server_ip …]
sample/        TTS clips: voice_kor* / voice_eng_* pairs, voice.m4a
```

MIT licensed — see [LICENSE](LICENSE).
