# voice-to-command

Turn a spoken command into a normalized text string.

`v2c` does **audio → ASR → (optional translation) → normalization**, and
nothing else — no robot, policy, or simulation concepts. It's the reusable
front half of a voice-driven pipeline; whatever consumes the text (a
simulator, a VLA policy, a real robot) lives in a separate repo.

```python
from voice_to_command import Recognizer, record

rec = Recognizer.from_dict({"asr": {"backend": "whisper"}})

result = rec.transcribe("command.wav")     # or rec.transcribe(record())
result.text          # "pick up the black bowl"  (normalized, target language)
result.raw_text      # ASR output before normalization
result.source_text   # text before translation, or None
result.language      # detected / used spoken language
result.timings       # {"asr": 1.83, "translate": 0.31}
```

## Install

```bash
pip install voice-to-command[whisper]          # local faster-whisper
pip install voice-to-command[naver]            # Naver CLOVA CSR + Papago
pip install voice-to-command[mic]              # microphone capture (needs PortAudio)
pip install voice-to-command[whisper,mic]      # typical laptop setup
pip install voice-to-command[all]
```

Core install pulls only `numpy`; every backend dependency is an extra and
imported lazily, so an unused backend costs nothing.

## Backends

| name | kind | extra | notes |
|---|---|---|---|
| `whisper` | ASR | `whisper` | local; `task="translate"` emits English in one step |
| `naver_csr` | ASR | `naver` | cloud; needs `NCP_CLIENT_ID` / `NCP_CLIENT_SECRET`; requires an explicit language |
| `naver_papago` | translator | `naver` | cloud; needs `NCP_PAPAGO_CLIENT_ID` / `NCP_PAPAGO_CLIENT_SECRET` |

Third-party backends register under the `voice_to_command.asr_backends` entry-point
group — see [docs/writing-a-backend.md](docs/writing-a-backend.md).

## Configuration

Build a `RecognizerConfig` from a dict, a TOML file, or the environment.
Credentials are **only** read from the environment, never from config.

`voice.toml` in the repo root is a ready-made preset for **spoken-English**
commands — pinned language, English-only model, plain transcribe task, no
translator. Use it directly:

```bash
v2c listen -c voice.toml
```
```python
Recognizer.from_config("voice.toml")
```

```toml
# voice.toml
timing = false

[asr]
backend  = "whisper"
language = "en"           # "auto" | "en" | "ko" | ...  (naver_csr needs an explicit one)

[asr.whisper]
model        = "base.en"  # tiny | base | small | medium | large-v3 (+ ".en" English-only)
device       = "cpu"      # cpu | cuda  (cuda auto-selects float16)
compute_type = ""         # "" = pick from device

[asr.options]
task = "transcribe"       # "translate" makes Whisper emit English in one step

[translate]
mode   = "never"          # auto = translate iff spoken != target | always | never
target = "en"
backend = "naver_papago"

[normalize]
lowercase        = true
strip_punctuation = true
# phrase_map     = { "black ball" = "black bowl" }
```

Env overrides (applied on top of any config): `V2C_ASR_BACKEND`,
`V2C_ASR_LANGUAGE`, `V2C_WHISPER_MODEL`, `V2C_WHISPER_DEVICE`,
`V2C_WHISPER_COMPUTE_TYPE`, `V2C_TRANSLATE_MODE`,
`V2C_TRANSLATE_TARGET`, `V2C_TIMING`.

## CLI

```bash
v2c transcribe command.wav [--json] [--backend ...] [--model ...] [--device ...]
v2c listen                         # push-to-talk mic → text
v2c translate "안녕하세요" --source ko --target en
v2c backends                       # list installed / available backends
v2c check --backend whisper [command.wav]   # import + credential smoke test
```

## Latency

Every stage is timed into `result.timings` always; the `[timing] <stage>: <s>`
lines are *printed* only when timing is on (`timing = true`, `--timing`, or
`V2C_TIMING=1`). The first `whisper` call in a process also pays a one-time
model load.

## Relation to other repos

- **This repo** — audio → text only. Stable `transcribe()` / `TranscriptResult`
  contract; backends may churn under it.
- **Consumers** (e.g. voice-to-simulation, a future real-robot runtime) depend on
  `v2c` and add the policy client + environment loop themselves:

  ```python
  from voice_to_command import Recognizer, record
  rec = Recognizer.from_config("voice.toml")
  prompt = rec.transcribe(audio_or_path_or_record()).text
  # ... hand `prompt` to the policy / rollout
  ```

## Development

```bash
pip install -e .[dev]
pytest                     # core tests, no heavy deps
pytest -m "not slow and not network"   # same, explicit
```

## License

MIT — see [LICENSE](LICENSE).
