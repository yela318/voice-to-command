"""Client/server round trip with the whisper model stubbed out (no download)."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from voice_to_command import remote


@pytest.fixture
def server(monkeypatch):
    """A real _Handler on an ephemeral port; core.transcribe is faked."""
    seen = {}

    def fake_transcribe(audio, translate=None):
        seen["audio"] = audio
        seen["translate"] = translate
        return "please give me a carrot"

    monkeypatch.setattr("voice_to_command.core.transcribe", fake_transcribe)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), remote._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[0], httpd.server_address[1], seen
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_file_roundtrip(server, tmp_path):
    ip, port, seen = server
    path = tmp_path / "clip.bin"
    path.write_bytes(b"RIFFfake-audio")

    assert remote.transcribe_remote(str(path), ip, port) == "please give me a carrot"
    # a file path arrives at the model as a binary stream, not a str
    assert hasattr(seen["audio"], "read")


def test_samples_roundtrip(server):
    ip, port, seen = server
    samples = np.linspace(-0.1, 0.1, 800, dtype=np.float32)

    assert remote.transcribe_remote(samples, ip, port) == "please give me a carrot"
    got = seen["audio"]
    assert isinstance(got, np.ndarray) and got.dtype == np.dtype("<f4")
    np.testing.assert_allclose(got, samples, atol=1e-7)
    assert seen["translate"] is False


def test_translate_flag_is_forwarded(server, tmp_path):
    ip, port, seen = server
    path = tmp_path / "clip.bin"
    path.write_bytes(b"RIFFfake")

    remote.transcribe_remote(str(path), ip, port, translate=True)
    assert seen["translate"] is True


def test_server_error_is_raised(server, monkeypatch):
    ip, port, _ = server

    def boom(_audio, translate=None):
        raise ValueError("bad frame")

    monkeypatch.setattr("voice_to_command.core.transcribe", boom)
    with pytest.raises(RuntimeError, match="bad frame"):
        remote.transcribe_remote(np.zeros(4, dtype=np.float32), ip, port)


def test_unreachable_server():
    with pytest.raises(RuntimeError, match="can't reach"):
        remote.transcribe_remote(np.zeros(4, dtype=np.float32), "127.0.0.1", 9)
