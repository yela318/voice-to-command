"""`v2c` -- Korean speech to English text.

    v2c voice_kor.m4a      # a file  -> English text on stdout
    v2c --listen           # the mic -> English text on stdout
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from . import __version__
from .core import transcribe


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="v2c", description="Korean speech -> English text (whisper)")
    p.add_argument("audio", nargs="?", help="audio file (wav/m4a/mp3/...)")
    p.add_argument("--listen", action="store_true", help="record from the microphone instead")
    p.add_argument("--version", action="version", version="v2c {}".format(__version__))
    args = p.parse_args(argv)

    if args.listen:
        from .capture import record

        print(transcribe(record()))
    elif args.audio:
        print(transcribe(args.audio))
    else:
        p.error("give an audio file, or --listen for the microphone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
