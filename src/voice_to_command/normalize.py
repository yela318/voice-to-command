"""Text post-processing applied to the ASR/translation output.

Order: phrase_map (on the raw text) -> strip punctuation -> lowercase ->
collapse whitespace. phrase_map is a small domain fixup, e.g.
{"black ball": "black bowl"} for a mis-heard object name.
"""

from __future__ import annotations

import re
import string

from .config import NormalizeConfig

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str, cfg: NormalizeConfig) -> str:
    out = text
    for src, dst in cfg.phrase_map.items():
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    if cfg.strip_punctuation:
        out = out.translate(_PUNCT)
    if cfg.lowercase:
        out = out.lower()
    if cfg.collapse_whitespace:
        out = " ".join(out.split())
    return out.strip()
