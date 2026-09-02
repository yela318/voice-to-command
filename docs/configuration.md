# Configuration

Three ways to build a `RecognizerConfig`, in increasing precedence:

1. `RecognizerConfig()` defaults
2. `RecognizerConfig.from_dict(...)` / `from_toml(path)`
3. `V2C_*` environment variables (applied by `with_env_overrides()`, which
   `Recognizer.__init__` always calls)

Credentials are **never** part of config — a backend that needs them reads
them straight from the environment.

## Sections

### `[asr]`

| key | default | meaning |
|---|---|---|
| `backend` | `"whisper"` | registered backend name |
| `language` | `"auto"` | spoken language (`"auto"` lets whisper detect it) |
| `whisper.model` | `"small"` | `tiny \| base \| small \| medium \| large-v3` |
| `whisper.device` | `"cpu"` | `cpu \| cuda` (cuda ⇒ `float16` unless `compute_type` set) |
| `whisper.compute_type` | `""` | override precision, e.g. `int8_float16` |
| `whisper.cpu_threads` | `0` | `0` = faster-whisper default; raise to use more CPU cores |

Any key under `[asr]` that isn't a known field is collected into
`asr.options` and passed as kwargs to that backend's `from_config`.

### `[translate]`

| key | default | meaning |
|---|---|---|
| `mode` | `"auto"` | `auto` (count as translated iff spoken ≠ target) `\| always \| never` |
| `target` | `"en"` | target language code |

There is **no external translator** in this build. The only way to get
`target`-language text is the ASR backend emitting it itself — whisper's
`task="translate"` → English, for which `supports_translation_to("en")` is
true. If the backend can't, the text stays in the spoken language and
`result.translated` is `False`. (A `translate.backend` key in an old config
is ignored.)

### `[normalize]`

Applied in this order: `phrase_map` (regex, case-insensitive) → strip
punctuation → lowercase → collapse whitespace.

### top level

| key | default | meaning |
|---|---|---|
| `timing` | `true` | print `[timing] <stage>: <s>` lines to **stderr**; `false` (or `V2C_TIMING=0`) silences them. `result.timings` is filled either way. |

## Environment overrides

`V2C_ASR_BACKEND`, `V2C_ASR_LANGUAGE`, `V2C_WHISPER_MODEL`,
`V2C_WHISPER_DEVICE`, `V2C_WHISPER_COMPUTE_TYPE`, `V2C_WHISPER_CPU_THREADS`,
`V2C_TRANSLATE_MODE`, `V2C_TRANSLATE_TARGET`, `V2C_TIMING`.
