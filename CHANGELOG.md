# Changelog

## [0.2.0] - unreleased

Reduced to the minimum: **Korean speech → English text via faster-whisper
`task="translate"`**, file or microphone input.

### Added
- `warmup()` (load the model + one throwaway inference, VAD off so the encoder
  really runs) and `v2c --serve` (stay resident: mic loop on a TTY, one file
  path per line from stdin when piped).
- Per-stage `[timing]` lines on stderr (`model_load` / `infer` / `total`).
- `V2C_DEVICE` for GPU inference (default `cpu`). On a non-CPU device `_load()`
  preloads the `nvidia-*-cu12` wheels so CTranslate2 finds `libcublas.so.12`
  without `LD_LIBRARY_PATH`. The compute type is picked
  from `ctranslate2.get_supported_compute_types()` — `float16` where the card
  does it efficiently, else `int8`; `V2C_COMPUTE` forces a specific one.

### Added
- `V2C_MIC` picks the input device by index or name substring; `record()` also
  prints which device and rate it settled on to stderr.

### Added
- `V2C_LANG` sets the source language; unset it (the default) and whisper
  auto-detects per clip, so Korean *and* English inputs both translate to
  English. `V2C_LANG=ko` re-pins it.
- Remote GPU: `v2c --serve --http` runs a stdlib HTTP inference server on the
  GPU box; `--server_ip` / `--server_port` (env `V2C_SERVER_IP` /
  `V2C_SERVER_PORT`) make any client send audio there instead of loading the
  model locally. New `remote.serve_http()` / `remote.transcribe_remote()`.
- Sample clips moved to `sample/`; added `voice_eng_*` English matches for
  carrot / banana / lemon / pineapple / apple.

### Fixed
- `record()` no longer forces 16 kHz on the input device. Most USB headsets only
  offer 44100/48000 and PortAudio answered with `Invalid sample rate` (-9997);
  it now records at the device's own rate and resamples to 16 kHz.

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
- `voice_to_command.serve_http(port=8756) -> int`
- `voice_to_command.transcribe_remote(path | samples, ip, port=8756) -> str`
- `v2c <file>` / `v2c --listen` / `v2c --serve` / `v2c --serve --http` /
  `v2c <file> --server_ip HOST`

## [0.1.0]
- Initial scaffold: audio → ASR → (translation) → normalization, pluggable
  backends, `v2c` CLI. Superseded by 0.2.0.
