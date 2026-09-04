"""`v2c` -- speech to text (Korean stays Korean, English stays English).

    v2c sample/voice_kor.m4a   # a file  -> text on stdout
    v2c --listen               # the mic -> text on stdout (one shot)
    v2c --serve                # stay resident: warm the model once, then loop
    v2c --translate            # translate the speech to English instead
    v2c --serve --http         # GPU box: HTTP inference server for remote clients
    v2c <file> --server_ip HOST # run inference on that remote server, not locally
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, List, Optional

from . import __version__
from .core import TRANSLATE, warmup


def _serve(transcribe: Callable[..., str], warm: bool = True) -> int:
    """Warm the model (unless inference is remote), then keep going. Mic
    push-to-talk when stdin is a TTY; otherwise read one audio-file path per line
    from stdin. Ctrl-C / EOF quits."""
    if warm:
        warmup()
    print("ready", flush=True)
    if sys.stdin.isatty():
        from .capture import record

        try:
            while True:
                print(transcribe(record()), flush=True)
        except KeyboardInterrupt:
            return 0
    for line in sys.stdin:
        path = line.strip()
        if path:
            print(transcribe(path), flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="v2c", description="speech -> text (whisper)")
    p.add_argument("audio", nargs="?", help="audio file (wav/m4a/mp3/...)")
    p.add_argument("--listen", action="store_true", help="record one clip from the microphone")
    p.add_argument("--serve", action="store_true", help="stay resident: warm once, then loop")
    p.add_argument(
        "--translate",
        action="store_true",
        help="translate the speech to English (default: transcribe in the spoken language; "
        "env: V2C_TRANSLATE=1)",
    )
    p.add_argument(
        "--http", action="store_true", help="with --serve: HTTP inference server for remote clients"
    )
    p.add_argument(
        "--server_ip",
        help="run inference on a remote `v2c --serve --http` host instead of locally "
        "(env: V2C_SERVER_IP)",
    )
    p.add_argument(
        "--server_port",
        type=int,
        help="port for --server_ip (default 8756, env: V2C_SERVER_PORT)",
    )
    p.add_argument("--version", action="version", version="v2c {}".format(__version__))
    args = p.parse_args(argv)

    ip = args.server_ip or os.environ.get("V2C_SERVER_IP")
    port = args.server_port or int(os.environ.get("V2C_SERVER_PORT") or 8756)
    translate = args.translate or TRANSLATE

    if args.http and not args.serve:
        p.error("--http only makes sense with --serve")
    if args.http and ip:
        p.error("--http runs a server; --server_ip is for the client")

    if args.serve and args.http:
        from .remote import serve_http

        return serve_http(port)

    if ip:
        from .remote import transcribe_remote

        def run(audio):
            return transcribe_remote(audio, ip, port, translate=translate)

    else:
        from .core import transcribe

        def run(audio):
            return transcribe(audio, translate=translate)

    if not (args.serve or args.listen or args.audio):
        p.error("give an audio file, or --listen / --serve for the microphone")

    try:
        if args.serve:
            return _serve(run, warm=ip is None)
        if args.listen:
            from .capture import record

            print(run(record()))
        else:
            print(run(args.audio))
    except RuntimeError as exc:  # unreachable server, missing deps, no mic -- no traceback
        print("v2c: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
