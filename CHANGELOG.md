# Changelog

## [0.2.0] - unreleased

Reduced to the minimum: **Korean speech → English text via faster-whisper
`task="translate"`**, file or microphone input.

### Removed
- The pluggable backend registry, config system (`RecognizerConfig` / TOML /
  `V2C_*` env), text normalization, the timing module, `Recognizer` /
  `TranscriptResult`, the `Audio` type, all CLI subcommands, `voice.toml` /
  `voice.ko.toml`, `docs/`, and `examples/`.
- Naver CLOVA CSR + Papago were already split to
  <https://github.com/yela318/voice-to-command-NAVER-CSR> in 0.1.x.

### API
- `voice_to_command.transcribe(path | samples) -> str`
- `voice_to_command.record() -> np.ndarray`
- `v2c <file>` / `v2c --listen`

## [0.1.0]
- Initial scaffold: audio → ASR → (translation) → normalization, pluggable
  backends, `v2c` CLI. Superseded by 0.2.0.
