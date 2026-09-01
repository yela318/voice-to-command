# Backends

## whisper (ASR, extra: `whisper`)

Local [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The model
loads on the first `transcribe()` and is cached on the backend instance.

- `task="translate"` (default) always outputs English; `supports_translation_to("en")`
  is then true and the Papago step is skipped. Set `task="transcribe"` (via
  `asr.options`) to keep the spoken language.
- `device="cuda"` auto-selects `compute_type="float16"`.
- Rough CPU transcription time for a ~5s clip: `tiny` ≈1s, `base` ≈2s,
  `small` ≈5s (model already downloaded, load excluded).

## naver_csr (ASR, extra: `naver`)

[Naver CLOVA Speech Recognition](https://api.ncloud-docs.com/docs/ai-naver-clovaspeechrecognition-stt).
Cloud API, built for short (≤ 60s, ≤ 3MB) command audio.

- Credentials: `NCP_CLIENT_ID`, `NCP_CLIENT_SECRET`.
- Needs an explicit language (`ko`/`en`/`ja`/`zh`); `"auto"` raises `LanguageRequired`.
- Does not translate — pair with `naver_papago` for non-English speech.
- File inputs are sent as-is (m4a/aac accepted); mic samples are sent as WAV.

## naver_papago (translator, extra: `naver`)

[Papago NMT](https://api.ncloud-docs.com/docs/ai-naver-papagonmt). A **separate**
NCP Application from CSR.

- Credentials: `NCP_PAPAGO_CLIENT_ID`, `NCP_PAPAGO_CLIENT_SECRET`.

## Writing your own

See [writing-a-backend.md](writing-a-backend.md).
