from __future__ import annotations

import json

from voice_to_command.cli import main


def test_backends_lists_fake(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "fake" in out.split()
    assert "whisper" in out.split()


def test_transcribe_plain_text(capsys, wav_path):
    code = main(["transcribe", str(wav_path), "--backend", "fake"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "pick up the black ball"


def test_transcribe_json(capsys, wav_path):
    code = main(["transcribe", str(wav_path), "--backend", "fake", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["raw_text"] == "pick up the black ball"
    assert "asr" in payload["timings"]


def test_check_reports_importable(capsys, wav_path):
    code = main(["check", str(wav_path), "--backend", "fake"])
    assert code == 0
    out = capsys.readouterr().out
    assert "importable" in out
    assert "transcript" in out


def test_unknown_backend_exits_2(capsys, wav_path):
    assert main(["transcribe", str(wav_path), "--backend", "nope"]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_file_exits_2(capsys):
    assert main(["transcribe", "no-such-file.wav", "--backend", "fake"]) == 2
    assert "error:" in capsys.readouterr().err
