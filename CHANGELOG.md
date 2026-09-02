# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer.

## [Unreleased]

### Added
- Initial scaffold: `Recognizer`, `RecognizerConfig`, `Audio`, `record()`.
- Backend registry with lazy built-in and `voice_to_command.asr_backends` entry-point
  discovery.
- Backend: `whisper` (faster-whisper), with `cpu_threads` and a per-stage
  timing split (`asr.model_load` / `asr.resample` / `asr.infer`).
- Text normalization (`phrase_map`, punctuation, case, whitespace).
- Per-stage timing in `TranscriptResult.timings`; `[timing]` lines print to
  stderr by default.
- `v2c` CLI: `transcribe`, `listen`, `backends`, `check`.

### Changed
- **Whisper-only.** Naver CLOVA CSR + Papago and the external `Translator`
  abstraction (`register_translator`, `translate.backend`, `v2c translate`)
  moved to a separate repo:
  <https://github.com/yela318/voice-to-command-NAVER-CSR>. `translate.mode` /
  `translate.target` remain and gate whisper's own `task="translate"`.
- Timing is on by default and prints to stderr (was off, stdout).

## [0.1.0] - unreleased
- First tagged release.
