# Configuration

Three ways to build a `RecognizerConfig`, in increasing precedence:

1. `RecognizerConfig()` defaults
2. `RecognizerConfig.from_dict(...)` / `from_toml(path)`
3. `V2C_*` environment variables (applied by `with_env_overrides()`, which
   `Recognizer.__init__` always calls)

Credentials are **never** part of config — backends read `NCP_CLIENT_ID`,
`NCP_PAPAGO_CLIENT_ID`, etc. straight from the environment.

## Sections

### `[asr]`

| key | default | meaning |
|---|---|---|
| `backend` | `"whisper"` | registered backend name |
| `language` | `"auto"` | spoken language; `naver_csr` rejects `"auto"` |
| `whisper.model` | `"small"` | `tiny \| base \| small \| medium \| large-v3` |
| `whisper.device` | `"cpu"` | `cpu \| cuda` (cuda ⇒ `float16` unless `compute_type` set) |
| `whisper.compute_type` | `""` | override precision, e.g. `int8_float16` |

Any key under `[asr]` that isn't a known field is collected into
`asr.options` and passed as kwargs to that backend's `from_config`.

### `[translate]`

| key | default | meaning |
|---|---|---|
| `mode` | `"auto"` | `auto` (translate iff spoken ≠ target) `\| always \| never` |
| `target` | `"en"` | target language code |
| `backend` | `"naver_papago"` | translator name |

If the ASR backend reports `supports_translation_to(target)` (Whisper's
`task="translate"` → English), the separate translator step is skipped.

### `[normalize]`

Applied in this order: `phrase_map` (regex, case-insensitive) → strip
punctuation → lowercase → collapse whitespace.

### top level

| key | default |
|---|---|
| `timing` | `false` |

## Environment overrides

`V2C_ASR_BACKEND`, `V2C_ASR_LANGUAGE`, `V2C_WHISPER_MODEL`,
`V2C_WHISPER_DEVICE`, `V2C_WHISPER_COMPUTE_TYPE`,
`V2C_TRANSLATE_MODE`, `V2C_TRANSLATE_TARGET`, `V2C_TIMING`.
