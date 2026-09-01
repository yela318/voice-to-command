from __future__ import annotations

from voice_to_command.config import NormalizeConfig
from voice_to_command.normalize import normalize


def test_noop_default_just_trims_and_collapses():
    cfg = NormalizeConfig()
    assert normalize("  pick   up  the bowl \n", cfg) == "pick up the bowl"


def test_phrase_map_case_insensitive():
    cfg = NormalizeConfig(phrase_map={"black ball": "black bowl"})
    assert normalize("Pick up the BLACK BALL", cfg) == "Pick up the black bowl"


def test_strip_punctuation_and_lowercase():
    cfg = NormalizeConfig(lowercase=True, strip_punctuation=True)
    assert normalize("Pick up the bowl, please!", cfg) == "pick up the bowl please"


def test_phrase_map_runs_before_lowercase():
    cfg = NormalizeConfig(lowercase=True, phrase_map={"Bowl": "Plate"})
    # phrase_map matched on raw text (case-insensitive), then lowercased
    assert normalize("put it in the Bowl", cfg) == "put it in the plate"
