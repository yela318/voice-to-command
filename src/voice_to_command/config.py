"""Configuration: plain dataclasses, buildable from a dict, a TOML file, or env.

Credentials are never read from here -- backends read those straight from
the environment (NCP_CLIENT_ID, NCP_PAPAGO_CLIENT_ID, ...).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Union


@dataclasses.dataclass
class WhisperConfig:
    model: str = "small"  # tiny | base | small | medium | large-v3
    device: str = "cpu"  # cpu | cuda
    compute_type: str = ""  # "" -> float16 on cuda, int8 on cpu


@dataclasses.dataclass
class ASRConfig:
    backend: str = "whisper"
    language: str = "auto"  # "auto" | "en" | "ko" | ...
    whisper: WhisperConfig = dataclasses.field(default_factory=WhisperConfig)
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)  # extra kwargs for other backends


@dataclasses.dataclass
class TranslateConfig:
    mode: str = "auto"  # auto (translate iff spoken != target) | always | never
    target: str = "en"
    backend: str = "naver_papago"


@dataclasses.dataclass
class NormalizeConfig:
    lowercase: bool = False
    strip_punctuation: bool = False
    collapse_whitespace: bool = True
    phrase_map: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RecognizerConfig:
    asr: ASRConfig = dataclasses.field(default_factory=ASRConfig)
    translate: TranslateConfig = dataclasses.field(default_factory=TranslateConfig)
    normalize: NormalizeConfig = dataclasses.field(default_factory=NormalizeConfig)
    timing: bool = False

    # -- builders ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecognizerConfig":
        data = data or {}
        asr_in = dict(data.get("asr", {}))
        whisper_in = dict(asr_in.pop("whisper", {}))
        known_asr = {f.name for f in dataclasses.fields(ASRConfig)}
        options = dict(asr_in.pop("options", {}))
        options.update({k: asr_in.pop(k) for k in list(asr_in) if k not in known_asr})
        asr = ASRConfig(whisper=WhisperConfig(**whisper_in), options=options, **asr_in)

        return cls(
            asr=asr,
            translate=TranslateConfig(**data.get("translate", {})),
            normalize=NormalizeConfig(**data.get("normalize", {})),
            timing=bool(data.get("timing", False)),
        )

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "RecognizerConfig":
        return cls.from_dict(_load_toml(Path(path)))

    def with_env_overrides(self) -> "RecognizerConfig":
        """Return a copy with V2C_* environment variables applied on top."""
        out = _clone(self)
        env = os.environ

        out.asr.backend = env.get("V2C_ASR_BACKEND", out.asr.backend)
        out.asr.language = env.get("V2C_ASR_LANGUAGE", out.asr.language)
        out.asr.whisper.model = env.get("V2C_WHISPER_MODEL", out.asr.whisper.model)
        out.asr.whisper.device = env.get("V2C_WHISPER_DEVICE", out.asr.whisper.device)
        out.asr.whisper.compute_type = env.get(
            "V2C_WHISPER_COMPUTE_TYPE", out.asr.whisper.compute_type
        )
        out.translate.mode = env.get("V2C_TRANSLATE_MODE", out.translate.mode)
        out.translate.target = env.get("V2C_TRANSLATE_TARGET", out.translate.target)
        if "V2C_TIMING" in env:
            out.timing = env["V2C_TIMING"].strip().lower() not in {"0", "false", "no", "off", ""}
        return out


def _clone(cfg: RecognizerConfig) -> RecognizerConfig:
    return RecognizerConfig(
        asr=ASRConfig(
            backend=cfg.asr.backend,
            language=cfg.asr.language,
            whisper=WhisperConfig(**dataclasses.asdict(cfg.asr.whisper)),
            options=dict(cfg.asr.options),
        ),
        translate=TranslateConfig(**dataclasses.asdict(cfg.translate)),
        normalize=NormalizeConfig(
            lowercase=cfg.normalize.lowercase,
            strip_punctuation=cfg.normalize.strip_punctuation,
            collapse_whitespace=cfg.normalize.collapse_whitespace,
            phrase_map=dict(cfg.normalize.phrase_map),
        ),
        timing=cfg.timing,
    )


def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "reading TOML config on Python < 3.11 needs the [toml] extra "
                "(pip install voice-to-command[toml]); or build RecognizerConfig.from_dict(...) yourself"
            ) from exc
    with path.open("rb") as fh:
        return tomllib.load(fh)
