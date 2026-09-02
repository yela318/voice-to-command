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
