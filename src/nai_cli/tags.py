"""Tag suggestions, looked up the way the official prompt box does."""

import re
from dataclasses import dataclass

from nai_cli import errors
from nai_cli import prompt as P
from nai_cli.models import Model

# The official pattern only removes a slash or bracket followed by ]; kept as is so queries match
_STRIP = re.compile(r"[/\[{}]\]")
JAPANESE_RATIO = 0.3
FURRY_V3 = "nai-diffusion-furry-3"
# The official prompt box waits this long after typing before it asks
PAUSE = 0.5


@dataclass(frozen=True)
class Lookup:
    query: str
    text: str
    model: str
    lexicon: str | None
    lang: str | None


def is_japanese(text: str) -> bool:
    found = len(P.CJK.findall(text))
    return found > 0 and found / P.js_length(text) > JAPANESE_RATIO


def lookup(query: str, model: Model, furry: bool) -> Lookup:
    if furry and not model.furry_mode:
        raise errors.NaiError(
            errors.MODEL_UNSUPPORTED, f"{model.name} has no furry mode", feature="furry", model=model.id
        )
    text = P.js_trim(_STRIP.sub("", query))
    if not text:
        raise errors.NaiError(errors.BAD_ARGS, f"empty tag query: {query!r}")
    if model.is_v5:
        request_model, lexicon = model.id, "furryv5" if furry else "animev5"
    else:
        request_model, lexicon = FURRY_V3 if furry else model.id, None
    return Lookup(query, text, request_model, lexicon, "jp" if is_japanese(query) else None)
