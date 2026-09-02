# Backends

## whisper (ASR, extra: `whisper`)

Local [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The model
loads on the first `transcribe()` and is cached on the backend instance.

- `task="translate"` (default) always outputs English; `supports_translation_to("en")`
  is then true and the Recognizer marks the result `translated`. Set
  `task="transcribe"` (via `asr.options`) to keep the spoken language.
- `device="cuda"` auto-selects `compute_type="float16"`.
- `cpu_threads` (default `0` = faster-whisper's own default) raises the CPU
  thread count for inference.
- Rough CPU transcription time for a ~5s clip: `tiny` ≈1s, `base` ≈2s,
  `small` ≈5s (model already downloaded, load excluded).

This build ships whisper only. Naver CLOVA CSR + Papago live in a separate
repo: <https://github.com/yela318/voice-to-command-NAVER-CSR>.

## Writing your own

See [writing-a-backend.md](writing-a-backend.md).
