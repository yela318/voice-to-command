from __future__ import annotations

import pytest

from voice_to_command.cli import main


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "v2c" in capsys.readouterr().out


def test_no_args_errors(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
    assert "audio file" in capsys.readouterr().err


def test_http_without_serve_errors(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--http"])
    assert e.value.code == 2
    assert "--serve" in capsys.readouterr().err


def test_server_ip_routes_to_remote(monkeypatch, capsys):
    calls = {}

    def fake_remote(audio, ip, port):
        calls["args"] = (audio, ip, port)
        return "hi"

    monkeypatch.setattr("voice_to_command.remote.transcribe_remote", fake_remote)
    assert main(["clip.wav", "--server_ip", "gpu-box", "--server_port", "9000"]) == 0
    assert calls["args"] == ("clip.wav", "gpu-box", 9000)
    assert capsys.readouterr().out.strip() == "hi"


def test_serve_http_invokes_server(monkeypatch):
    monkeypatch.setattr("voice_to_command.remote.serve_http", lambda port: port)
    assert main(["--serve", "--http"]) == 8756
