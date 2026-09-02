# voice-to-command

Speak a command, get back a clean text string. `v2c` is the front half of a
voice pipeline — **audio → ASR → (optional translation) → normalization**.
Whatever consumes the text lives in another repo.

## Presets

Two ready-to-use configs in the repo root — pick by what you speak:

| file | you speak | you get |
|---|---|---|
| `voice.toml` | English | English — English-only Whisper model, no translation |
| `voice.ko.toml` | Korean | English — Whisper translates in one pass, no Papago or credentials |

Every key in them is commented; copy one and edit.

## Running it

Clone and install editable, so `v2c` and the library track your checkout:

```bash
git clone https://github.com/yela318/voice-to-command.git
cd voice-to-command
pip install -e .[whisper]        # add ,mic for microphone capture
```

```bash
v2c transcribe FILE  -c voice.toml            # audio file  → text
v2c listen           -c voice.ko.toml         # microphone (push-to-talk) → text
v2c translate TEXT   --source ko --target en  # text only, runs the translator
v2c backends                                  # list available backends
v2c check [--backend whisper] [FILE]          # import + credential smoke test
```

`transcribe` / `listen` also take `--json`, `--timing`, and the overrides
`--backend --model --device --language --target`.

From Python — build the `Recognizer` once and reuse it, so the model stays
loaded between calls:

```python
from voice_to_command import Recognizer, record

rec = Recognizer.from_config("voice.ko.toml")
rec.transcribe("command.wav").text            # or: rec.transcribe(record()).text
```

`examples/listen_loop.py [config]` is a warm push-to-talk loop built on that.

## Result

`transcribe()` returns a `TranscriptResult`: `text` (final — normalized, target
language), `raw_text` (before normalization), `source_text` (before
translation, or `None`), `language`, `translated`, `segments`, and `timings`
like `{"asr": 1.83, "asr.infer": 1.6, "translate": 0.31}`.

---

## Configuration

Config is resolved in three layers, each overriding the previous: **defaults**
→ **dict / TOML file** → **`V2C_*` env vars**. Credentials are read only from
the environment, never from a config file.

Every key is documented inline in `voice.toml`. The full reference — all
sections, env var names, normalization order — is in
[docs/configuration.md](docs/configuration.md).

## Backends

Core install is `numpy` only; each backend is a lazily-imported extra.

| backend | kind | extra | needs |
|---|---|---|---|
| `whisper` | ASR | `[whisper]` | — (local; `task="translate"` emits English directly) |
| `naver_csr` | ASR | `[naver]` | `NCP_CLIENT_ID` / `NCP_CLIENT_SECRET`, explicit language |
| `naver_papago` | translator | `[naver]` | `NCP_PAPAGO_CLIENT_ID` / `NCP_PAPAGO_CLIENT_SECRET` |

Other extras: `[mic]` (microphone, needs PortAudio), `[audio]` (m4a/mp3
decode), `[all]`. Writing your own: [docs/writing-a-backend.md](docs/writing-a-backend.md).

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
