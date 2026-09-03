# Changelog

## [0.2.0] - unreleased

Reduced to the minimum: **Korean speech → English text via faster-whisper
`task="translate"`**, file or microphone input.

### Added
- `warmup()` (load the model + one throwaway inference) and `v2c --serve` (stay
  resident: mic loop on a TTY, one file path per line from stdin when piped).
- Per-stage `[timing]` lines on stderr (`model_load` / `infer` / `total`).
- `V2C_DEVICE` for GPU inference (default `cpu`). `V2C_COMPUTE` follows it
  (`int8` on CPU, `float16` on a GPU) and only needs setting for an odd card.

### Removed
- The pluggable backend registry, the config system (`RecognizerConfig` / TOML
  files), text normalization, the timing module, `Recognizer` /
  `TranscriptResult`, the `Audio` type, all CLI subcommands, `voice.toml` /
  `voice.ko.toml`, `docs/`, and `examples/`. Config is now four env vars
  (`V2C_MODEL` / `_DEVICE` / `_COMPUTE` / `_TIMING`), no file, no loader.
- Naver CLOVA CSR + Papago were already split to
  <https://github.com/yela318/voice-to-command-NAVER-CSR> in 0.1.x.

### API
- `voice_to_command.transcribe(path | samples) -> str`
- `voice_to_command.record() -> np.ndarray`
- `voice_to_command.warmup() -> None`
- `v2c <file>` / `v2c --listen` / `v2c --serve`

## [0.1.0]
- Initial scaffold: audio → ASR → (translation) → normalization, pluggable
  backends, `v2c` CLI. Superseded by 0.2.0.
