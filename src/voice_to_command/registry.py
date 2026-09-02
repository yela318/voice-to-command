"""Backend registry.

Built-in backends are wired in the table below and imported lazily (only
when selected), so faster-whisper stays optional. Third-party backends
register via the "voice_to_command.asr_backends" entry-point group.

    from voice_to_command.registry import register_asr

    @register_asr("my_backend")
    class MyBackend:
        target_sample_rate = 16000
        def supports_translation_to(self, lang): return False
        def transcribe(self, audio, *, language): ...
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Iterable

from .errors import BackendNotAvailable

_ASR: Dict[str, type] = {}

_BUILTIN_ASR = {
    "whisper": "voice_to_command.backends.whisper:WhisperBackend",
}
_ASR_ENTRYPOINT_GROUP = "voice_to_command.asr_backends"


def register_asr(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name  # type: ignore[attr-defined]
        _ASR[name] = cls
        return cls

    return deco


def get_asr_backend(name: str) -> type:
    if name in _ASR:
        return _ASR[name]
    if name in _BUILTIN_ASR:
        return _import_target(name, _BUILTIN_ASR[name])
    for ep in _iter_entry_points(_ASR_ENTRYPOINT_GROUP):
        if ep.name == name:
            return ep.load()
    raise BackendNotAvailable(
        "unknown ASR backend {!r}; available: {}".format(name, sorted(available_asr_backends()))
    )


def available_asr_backends() -> Iterable[str]:
    names = set(_ASR) | set(_BUILTIN_ASR)
    names |= {ep.name for ep in _iter_entry_points(_ASR_ENTRYPOINT_GROUP)}
    return names


def _import_target(name: str, spec: str) -> type:
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendNotAvailable(
            "backend {!r} is missing an optional dependency: {}".format(name, exc)
        ) from exc
    return getattr(module, attr)


def _iter_entry_points(group: str):
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return []
    eps = entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        return list(eps.select(group=group))
    return list(eps.get(group, []))  # Python 3.8 / 3.9
