# tests/data

Drop small sample recordings here for the integration tests, e.g.:

- `command_en.wav` — short English robot command
- `command_ko.wav` — the Korean equivalent

The unit tests don't need these — `tests/conftest.py` synthesizes a WAV
fixture on the fly. Integration tests that use real files should
`pytest.importorskip` their backend and skip if the file is absent.
