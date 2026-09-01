# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer.

## [Unreleased]

### Added
- Initial scaffold: `Recognizer`, `RecognizerConfig`, `Audio`, `record()`.
- Backend registry with lazy built-ins and `voice_to_command.asr_backends` entry-point
  discovery.
- Backends: `whisper` (faster-whisper), `naver_csr`, `naver_papago`.
- Text normalization (`phrase_map`, punctuation, case, whitespace).
- Per-stage timing collected into `TranscriptResult.timings`.
- `v2c` CLI: `transcribe`, `listen`, `translate`, `backends`, `check`.

## [0.1.0] - unreleased
- First tagged release.
