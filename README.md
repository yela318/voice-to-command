# voice-to-command

Speak a command, get back a clean text string. `v2c` is the front half of a
voice pipeline — **audio → ASR → normalization** — on faster-whisper.
Whatever consumes the text lives in another repo.

Whisper-only. Naver CLOVA CSR + Papago live in a separate repo:
[voice-to-command-NAVER-CSR](https://github.com/yela318/voice-to-command-NAVER-CSR).

## Preset

`voice.toml` in the repo root is a ready-to-use config for spoken-English
commands (English-only model, `task="transcribe"`, no translation). Every key
is commented — copy it and edit. To have whisper translate other languages to
English in one pass, set `language` to the spoken language and
`[asr.options] task = "translate"`.

## Running it

Clone and install editable, so `v2c` and the library track your checkout:

```bash
git clone https://github.com/yela318/voice-to-command.git
cd voice-to-command
pip install -e .[whisper]        # add ,mic for microphone capture
```

```bash
v2c transcribe voice.m4a -c voice.toml        # audio file → text
v2c listen               -c voice.toml        # microphone (push-to-talk) → text
v2c backends                                  # list available backends
v2c check [--backend whisper] [voice.m4a]     # import + load smoke test
```

`voice.m4a` (English) and `voice_kor.m4a` (Korean) in the repo root are sample
recordings to try the model on. faster-whisper decodes m4a itself, so no
`[audio]` extra is needed for a file path.

```bash
v2c transcribe voice.m4a     -c voice.toml                    # English → English
v2c transcribe voice_kor.m4a --language ko --model small      # Korean → Korean text
```

The Korean clip needs a multilingual model (`small`, not `base.en`). To get
**English** out of it, use a config with `[asr.options] task = "translate"`
(there is no CLI flag for `task`).

`transcribe` / `listen` also take `--json`, `--timing`, and the overrides
`--backend --model --device --language`.

From Python — build the `Recognizer` once and reuse it, so the model stays
loaded between calls:

```python
from voice_to_command import Recognizer, record

rec = Recognizer.from_config("voice.toml")
rec.transcribe("voice.m4a").text              # or: rec.transcribe(record()).text
```

`examples/listen_loop.py [config]` is a warm push-to-talk loop built on that.

## Result

`transcribe()` returns a `TranscriptResult`: `text` (final — normalized),
`raw_text` (before normalization), `language`, `translated` (true when whisper
emitted English via `task="translate"`), `source_text` (always `None` in this
build), `segments`, and `timings` like `{"asr": 1.83, "asr.infer": 1.6}`.

---

## Configuration

Config is resolved in three layers, each overriding the previous: **defaults**
→ **dict / TOML file** → **`V2C_*` env vars**. Credentials are read only from
the environment, never from a config file.

Every key is documented inline in `voice.toml`. The full reference — all
sections, env var names, normalization order — is in
[docs/configuration.md](docs/configuration.md).

## Backends

Core install is `numpy` only; everything else is a lazily-imported extra.

| extra | pulls | for |
|---|---|---|
| `[whisper]` | faster-whisper | the `whisper` ASR backend (only built-in) |
| `[mic]` | sounddevice | microphone capture (needs the PortAudio system lib) |
| `[audio]` | av | decode m4a / mp3 / … to PCM |
| `[toml]` | tomli | read TOML config on Python < 3.11 |
| `[all]` | all of the above | |

`whisper`'s `task="translate"` emits English directly. For Naver CLOVA CSR /
Papago, use [voice-to-command-NAVER-CSR](https://github.com/yela318/voice-to-command-NAVER-CSR).
Writing your own backend: [docs/writing-a-backend.md](docs/writing-a-backend.md).

## Latency

The first `whisper` call in a process loads the model (cold start); keep one
`Recognizer` alive (see `examples/listen_loop.py`) and later calls skip it.

Every run prints a stage breakdown to **stderr** (stdout stays clean for
`--json` and pipes). Silence it with `timing = false` or `V2C_TIMING=0`.

```
[timing] asr.model_load: 6.21s     # ~0 once the model is cached
[timing] asr.infer: 1.83s          # decode + VAD + transcription
[timing] asr: 8.04s                # total
```

To cut `asr.infer`: raise `cpu_threads`, use a smaller model (`tiny.en`,
`distil-small.en`), or set `device = "cuda"`.

## Consuming repos

Downstream code installs this repo (`pip install -e path/to/voice-to-command`,
or `pip install "voice-to-command @ git+https://github.com/yela318/voice-to-command.git"`
until it's on PyPI) and adds its own policy client + rollout loop. The
`transcribe()` / `TranscriptResult` contract is stable; backends may change
under it.

```python
rec = Recognizer.from_config("voice.toml")
prompt = rec.transcribe(audio_or_path_or_record()).text
```

## Development

```bash
pip install -e .[dev]
pytest                                    # core tests, no heavy deps
ruff check src tests examples
```

MIT licensed — see [LICENSE](LICENSE).
