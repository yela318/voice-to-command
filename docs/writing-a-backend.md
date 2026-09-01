# Writing a backend

An ASR backend is any object with this shape (no base class needed):

```python
class MyBackend:
    name = "my_backend"            # set for you by @register_asr
    target_sample_rate = 16000     # what transcribe() wants raw samples in

    # optional: build from ASRConfig; if absent, MyBackend() is called
    @classmethod
    def from_config(cls, asr):     # asr: voice_to_command.config.ASRConfig
        return cls(**asr.options)

    def supports_translation_to(self, language: str) -> bool:
        return False

    def transcribe(self, audio, *, language):   # audio: voice_to_command.Audio
        # audio.is_file()            -> str(audio.path) for libs that decode themselves
        # audio.samples(rate)        -> (np.float32 mono, rate)
        # audio.wav_bytes(rate)      -> 16-bit PCM WAV bytes
        # audio.raw_bytes()          -> original container bytes when available
        return ASRResult(text="...", language=language, segments=(...,))
```

### Optional: report your own timing split

`ASRResult.timings` (a `{stage: seconds}` dict, default empty) lets a backend
break its share down; the Recognizer merges it into `TranscriptResult.timings`
next to the `asr` total. Use `voice_to_command.timing.stage`, which also prints
`[timing]` lines when timing is on. Namespace the keys with an `asr.` prefix:

```python
from voice_to_command import timing

def transcribe(self, audio, *, language):
    t = {}
    with timing.stage("asr.model_load", t):
        model = self._load()
    with timing.stage("asr.infer", t):
        text = model.run(audio.samples(self.target_sample_rate)[0])
    return ASRResult(text=text, language=language, timings=t)
```

## In-tree

Add a module under `voice_to_command/backends/`, decorate with `@register_asr("name")`,
and add it to `_BUILTIN_ASR` in `voice_to_command/registry.py` with a
`"module:Class"` string so it's imported lazily.

## Out-of-tree (separate package)

Register via entry points; no change to voice-to-command:

```toml
# pyproject.toml of your package
[project.entry-points."voice_to_command.asr_backends"]
my_backend = "my_package.backend:MyBackend"
```

After `pip install my-package`, `v2c backends` lists `my_backend` and
`Recognizer.from_dict({"asr": {"backend": "my_backend"}})` works.

## Translators

Same idea with `@register_translator("name")` and a
`translate(text, *, source, target) -> str` method. Only in-tree
registration (`_BUILTIN_TRANSLATORS`) is wired today.

## Errors to raise

`BackendNotAvailable` (missing dep), `CredentialsMissing`,
`LanguageRequired`, `AudioTooLong`, `DecodeError` — all from `voice_to_command.errors`.
