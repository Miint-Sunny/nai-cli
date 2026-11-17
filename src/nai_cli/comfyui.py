"""ComfyUI multi-character prompts, split into a base prompt and characters at their regions' centres."""

import math
import re
from dataclasses import dataclass

from nai_cli import prompt as P

# A comma ends the tag before COUPLE as a space does
_COUPLE = re.compile(r"(?:^|[\s,]+)COUPLE\b")
_COUPLE_REGION = re.compile(r"^\s*\(([^)]+)\)")
_AND = re.compile(r"\s+AND\s+")
_REGION_CALL = re.compile(r"\b(?:MASK|AREA)\s*\(")
_MASK = re.compile(r"\bMASK\s*\(([^)]+)\)")
_AREA = re.compile(r"\bAREA\s*\(([^)]+)\)")
_IMASK = re.compile(r"\bIMASK\s*\([^)]*\)")
_FILL = re.compile(r"\bFILL\s*\(\s*\)")
_TRAILING_WEIGHT = re.compile(r"\s*:\s*[\d.]+\s*$")
_LINE_START = re.compile(r"(?:^|[\r\n])[ \t]*$")
_SPACES = " \t\r\n"
# Stands in for quoted text; no syntax matches it
_BLANK = "\0"


@dataclass(frozen=True)
class Character:
    prompt: str
    # The centre of the region it was given, within 0-1
    center: tuple[float, float] | None


@dataclass(frozen=True)
class Split:
    base: str
    characters: tuple[Character, ...]


@dataclass(frozen=True)
class _Text:
    """A prompt beside a copy of it with quoted text blanked, so syntax is found only outside quotes and cut from
    both at the same places."""

    text: str
    bare: str

    @classmethod
    def of(cls, text: str) -> "_Text":
        bare = list(text)
        for start, end in P.quoted_ranges(text):
            # Single quotes are apostrophes as often, as in '90s or girls', so a pair spanning tags is not quoted
            if text[end - 1] in "'\u2019" and any(c in ",\r\n" for c in text[start:end]):
                continue
            bare[start:end] = _BLANK * (end - start)
        return cls(text, "".join(bare))

    def __getitem__(self, at: slice) -> "_Text":
        return _Text(self.text[at], self.bare[at])

    def __add__(self, other: "_Text") -> "_Text":
        return _Text(self.text + other.text, self.bare + other.bare)

    def cut(self, pattern: re.Pattern[str]) -> list["_Text"]:
        parts, start = [], 0
        for match in pattern.finditer(self.bare):
            parts.append(self[start : match.start()])
            start = match.end()
        return [*parts, self[start:]]

    def without(self, pattern: re.Pattern[str]) -> "_Text":
        parts = self.cut(pattern)
        return sum(parts[1:], parts[0])

    def strip(self) -> "_Text":
        return self[len(self.bare) - len(self.bare.lstrip()) : len(self.bare.rstrip())]

    def take(self, pattern: re.Pattern[str]) -> tuple["_Text", tuple[float, float] | None]:
        """Without the first call, and the centre of the region it gave."""
        match = pattern.search(self.bare)
        if match is None:
            return self, None
        return self[: match.start()] + self[match.end() :], _center(self.text[match.start(1) : match.end(1)])


def split(text: str) -> Split | None:
    """COUPLE, AND with MASK or AREA, or a base prompt with a character after each | line; None for other
    prompts. How many characters a model takes is checked with the request."""
    typed = _Text.of(text)
    if _COUPLE.search(typed.bare):
        found = _couple(typed)
    elif _AND.search(typed.bare) and _REGION_CALL.search(typed.bare):
        found = _and_mask(typed)
    else:
        found = _pipes(typed)
    return found if found is not None and found.characters else None


def _couple(typed: _Text) -> Split:
    base, *parts = typed.cut(_COUPLE)
    characters = []
    for part in parts:
        rest = part.strip()
        region = _COUPLE_REGION.match(rest.bare)
        # A bracket that gives no region is a prompt of its own, such as (1girl:1.2)
        center = _center(rest.text[region.start(1) : region.end(1)]) if region else None
        if region and center is not None:
            rest = rest[region.end() :]
        rest, masked = rest.take(_MASK)
        characters.append((rest, center or masked))
    return Split(_tidy(base.without(_FILL).text), _characters(characters))


def _and_mask(typed: _Text) -> Split:
    characters = []
    for part in typed.cut(_AND):
        rest = part.strip().without(_TRAILING_WEIGHT)
        rest, masked = rest.take(_MASK)
        rest, area = rest.take(_AREA)
        characters.append((rest, masked or area))
    # Every part is a character; there is no shared prompt
    return Split("", _characters(characters))


def _characters(parts: list[tuple[_Text, tuple[float, float] | None]]) -> tuple[Character, ...]:
    out = []
    for text, center in parts:
        prompt = _tidy(text.without(_IMASK).text)
        if prompt:
            out.append(Character(prompt, center))
    return tuple(out)


def _center(params: str) -> tuple[float, float] | None:
    """(x1 x2) or (x1 x2, y1 y2, weight...); y spans the whole height when left out."""
    ranges = [p.split() for p in params.split(",")]
    try:
        x1, x2 = (float(v) for v in ranges[0][:2])
        y1, y2 = (float(v) for v in ranges[1][:2]) if len(ranges) > 1 and len(ranges[1]) >= 2 else (0.0, 1.0)
    except ValueError:
        return None
    # nan and inf parse as floats but place nothing
    if not all(math.isfinite(v) for v in (x1, x2, y1, y2)):
        return None
    return _clamp((x1 + x2) / 2), _clamp((y1 + y2) / 2)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _tidy(text: str) -> str:
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"^\s*,\s*|\s*,\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _pipes(typed: _Text) -> Split | None:
    """A | at the start of a line, or between spaces, outside brackets, quotes and ||a|b|| groups."""
    text, bare = typed.text, typed.bare
    parts = []
    start = 0
    depth = {"{": 0, "[": 0, "(": 0}
    closing = {"}": "{", "]": "[", ")": "("}
    escaped = grouped = False
    i = 0
    while i < len(bare):
        ch = bare[i]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch in depth:
            depth[ch] += 1
        elif ch in closing:
            depth[closing[ch]] = max(0, depth[closing[ch]] - 1)
        elif ch == "|" and bare[i + 1 : i + 2] == "|":
            grouped = not grouped
            i += 1
        elif ch == "|" and not grouped and not any(depth.values()) and _divides(text, i):
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    parts.append(text[start:].strip())
    # A | on the first line leaves the base prompt empty
    base, *characters = parts
    characters = [p for p in characters if p]
    if not characters:
        return None
    return Split(base, tuple(Character(p, None) for p in characters))


def _divides(text: str, at: int) -> bool:
    if _LINE_START.search(text, 0, at):
        return True
    return 0 < at < len(text) - 1 and text[at - 1] in _SPACES and text[at + 1] in _SPACES
