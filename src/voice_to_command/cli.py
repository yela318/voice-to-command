"""`v2c` command-line entry point (argparse, no third-party deps).

    v2c transcribe command.wav [--json]
    v2c listen
    v2c translate "안녕하세요" --source ko --target en
    v2c backends
    v2c check --backend whisper [command.wav]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import List, Optional

from . import __version__
from .config import RecognizerConfig
from .errors import V2CError
from .recognizer import Recognizer
from .registry import available_asr_backends, get_asr_backend, get_translator


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="path to a TOML config file")
    p.add_argument("--language", help="override asr.language ('auto', 'en', 'ko', ...)")
    p.add_argument("--model", help="override asr.whisper.model")
    p.add_argument("--device", help="override asr.whisper.device (cpu|cuda)")
    p.add_argument("--target", help="override translate.target")
    p.add_argument("--timing", action="store_true", help="print [timing] lines")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")


def _build_config(args: argparse.Namespace) -> RecognizerConfig:
    cfg = RecognizerConfig.from_toml(args.config) if args.config else RecognizerConfig()
    if args.backend:
        cfg.asr.backend = args.backend
    if args.language:
        cfg.asr.language = args.language
    if args.model:
        cfg.asr.whisper.model = args.model
    if args.device:
        cfg.asr.whisper.device = args.device
    if getattr(args, "target", None):
        cfg.translate.target = args.target
    if args.timing:
        cfg.timing = True
    return cfg


def _emit(result, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2))
    else:
        print(result.text)


def _cmd_transcribe(args: argparse.Namespace) -> int:
    rec = Recognizer(_build_config(args))
    _emit(rec.transcribe(args.audio), args.json)
    return 0


def _cmd_listen(args: argparse.Namespace) -> int:
    from .capture import record

    rec = Recognizer(_build_config(args))
    _emit(rec.transcribe(record()), args.json)
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    cfg = RecognizerConfig.from_toml(args.config) if args.config else RecognizerConfig()
    translator = get_translator(cfg.translate.backend)()
    print(translator.translate(args.text, source=args.source, target=args.target))
    return 0


def _cmd_backends(_args: argparse.Namespace) -> int:
    for name in sorted(available_asr_backends()):
        print(name)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    cls = get_asr_backend(args.backend)  # raises BackendNotAvailable if the extra is missing
    print("backend {!r} importable: {}".format(args.backend, cls.__name__))
    if args.audio:
        rec = Recognizer(_build_config(args))
        result = rec.transcribe(args.audio)
        print("transcript: {!r}".format(result.text))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v2c", description="Audio -> text command")
    parser.add_argument("--version", action="version", version="v2c {}".format(__version__))
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p_t = sub.add_parser("transcribe", help="transcribe an audio file")
    p_t.add_argument("audio")
    p_t.add_argument("--backend", help="override asr.backend")
    _add_common(p_t)
    p_t.set_defaults(func=_cmd_transcribe)

    p_l = sub.add_parser("listen", help="record from the mic, then transcribe")
    p_l.add_argument("--backend", help="override asr.backend")
    _add_common(p_l)
    p_l.set_defaults(func=_cmd_listen)

    p_tr = sub.add_parser("translate", help="translate text with the configured translator")
    p_tr.add_argument("text")
    p_tr.add_argument("--source", default="auto")
    p_tr.add_argument("--target", default="en")
    p_tr.add_argument("-c", "--config")
    p_tr.set_defaults(func=_cmd_translate)

    p_b = sub.add_parser("backends", help="list available ASR backends")
    p_b.set_defaults(func=_cmd_backends)

    p_c = sub.add_parser("check", help="smoke-test a backend (and optionally an audio file)")
    p_c.add_argument("audio", nargs="?")
    p_c.add_argument("--backend", default="whisper")
    _add_common(p_c)
    p_c.set_defaults(func=_cmd_check)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (V2CError, FileNotFoundError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
