"""`v2c` -- Korean speech to English text.

    v2c voice_kor.m4a      # a file  -> English text on stdout
    v2c --listen           # the mic -> English text on stdout (one shot)
    v2c --serve            # stay resident: warm the model once, then loop
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .core import transcribe, warmup


def _serve() -> int:
    """Warm the model, then keep going. Mic push-to-talk when stdin is a TTY;
    otherwise read one audio-file path per line from stdin. Ctrl-C / EOF quits."""
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
    p = argparse.ArgumentParser(prog="v2c", description="Korean speech -> English text (whisper)")
    p.add_argument("audio", nargs="?", help="audio file (wav/m4a/mp3/...)")
    p.add_argument("--listen", action="store_true", help="record one clip from the microphone")
    p.add_argument("--serve", action="store_true", help="stay resident: warm once, then loop")
    p.add_argument("--version", action="version", version="v2c {}".format(__version__))
    args = p.parse_args(argv)

    if args.serve:
        return _serve()
    if args.listen:
        from .capture import record

        print(transcribe(record()))
    elif args.audio:
        print(transcribe(args.audio))
    else:
        p.error("give an audio file, or --listen / --serve for the microphone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
