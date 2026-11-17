"""Prompt tidying a user switches on: commas and spaces, Stable Diffusion weights, a space before numeric closings."""

import re
from typing import Any

from nai_cli import prompt as P

_HORIZONTAL_SPACE = re.compile(r"[ \t\f\u00a0\u3000]+")
_LITERAL_START = "\ue000"
_LITERAL_END = "\ue001"

_BRACKETS = {"(": ")", "[": "]"}
_SD_ESCAPABLE = "()[]\\"
_SD_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_WEIGHT_TOLERANCE = 0.00001


def format_prompt(text: str, underscores: bool = False) -> str:
    """Normalize full-width commas, collapse whitespace, drop empty tags and put one space after commas.
    Optionally replace underscores. Line breaks and indentation are kept."""
    ranges = P.literal_ranges(text)
    lead = _LITERAL_START
    while lead in text:
        lead += _LITERAL_START
    literals: list[str] = []
    masked: list[str] = []
    cursor = 0
    for start, end in ranges:
        masked.append(text[cursor:start])
        masked.append(f"{lead}{len(literals)}{_LITERAL_END}")
        literals.append(text[start:end])
        cursor = end
    masked.append(text[cursor:])
    lines = re.split(r"(\r\n|\r|\n)", "".join(masked))
    formatted = "".join(part if i % 2 else _format_line(part, underscores) for i, part in enumerate(lines))
    return re.sub(re.escape(lead) + r"(\d+)" + _LITERAL_END, lambda m: literals[int(m.group(1))], formatted)


def convert_sd_weights(text: str) -> str:
    """(tag:1.2) and [tag:0.8] become 1.2::tag:: and 0.8::tag::. Brackets without a number stay,
    since NAI reads them too, and escaped round brackets lose the backslash. Literal ranges such as quoted text
    and text: segments are sent as written."""
    out, last = [], 0
    for start, end in P.literal_ranges(text):
        out += [_convert_sd_segment(text[last:start]), text[start:end]]
        last = end
    out.append(_convert_sd_segment(text[last:]))
    return "".join(out)


def space_numeric_closings(text: str) -> str:
    """Insert a space when a closing :: follows a digit, point or minus sign; nothing else changes."""
    positions = [m.start for m in P.weight_marks(text) if m.kind == "numeric_close"]
    for pos in reversed(positions):
        text = f"{text[:pos]} {text[pos:]}"
    return text


def _format_line(line: str, underscores: bool) -> str:
    stripped = line.strip(" \t")
    if not stripped:
        return line
    lead = line[: len(line) - len(line.lstrip(" \t"))]
    tail = line[len(line.rstrip(" \t")) :]
    content = _HORIZONTAL_SPACE.sub(" ", stripped).replace("，", ",")
    tags = [t.strip() for t in content.split(",")]
    if underscores:
        tags = [_HORIZONTAL_SPACE.sub(" ", t.replace("_", " ")).strip() for t in tags]
    formatted = ", ".join(t for t in tags if t)
    if content.endswith(",") and formatted:
        formatted += ","
    return f"{lead}{formatted}{tail}"


def _convert_sd_segment(text: str) -> str:
    if not _has_sd_weights(text):
        return _unescape(text, "()")
    parts: list[tuple[str, float]] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text) and text[i + 1] in _SD_ESCAPABLE:
            parts.append((text[i + 1], 1.0))
            i += 2
            continue
        if c in _BRACKETS:
            close = _matching_close(text, i)
            content = text[i + 1 : close] if close >= 0 else ""
            weight = _explicit_weight(content) if close >= 0 else None
            if weight is not None:
                parts.append((_unescape(content[: content.rfind(":")].strip()), weight))
                i = close + 1
                continue
        parts.append((c, 1.0))
        i += 1
    merged: list[list[Any]] = []
    for piece, weight in parts:
        if merged and abs(merged[-1][1] - weight) < _WEIGHT_TOLERANCE:
            merged[-1][0] += piece
        else:
            merged.append([piece, weight])
    out = ""
    open_weight = False
    for piece, weight in merged:
        piece = _unescape(piece)
        number = _weight_text(weight)
        if open_weight:
            out += "::"
        if number != "1":
            piece = piece.strip()
            trailing = P.NUMBER_BEFORE_CLOSE.search(out + number)
            # A digit already before it would join the weight into a longer number
            out += ("" if trailing is not None and trailing.group(0) == number else " ") + f"{number}::{piece}"
            open_weight = True
        else:
            out += piece
            open_weight = False
    return out + ("::" if open_weight else "")


def _has_sd_weights(text: str) -> bool:
    for i, c in enumerate(text):
        if c in _BRACKETS and not _escaped(text, i):
            close = _matching_close(text, i)
            if close >= 0 and _explicit_weight(text[i + 1 : close]) is not None:
                return True
    return False


def _matching_close(text: str, start: int) -> int:
    opening, closing = text[start], _BRACKETS[text[start]]
    depth = 1
    for i in range(start + 1, len(text)):
        if text[i] == opening and not _escaped(text, i):
            depth += 1
        elif text[i] == closing and not _escaped(text, i):
            depth -= 1
            if depth == 0:
                return i
    return -1


def _escaped(text: str, index: int) -> bool:
    count = 0
    while index - count - 1 >= 0 and text[index - count - 1] == "\\":
        count += 1
    return count % 2 == 1


def _explicit_weight(content: str) -> float | None:
    """The weight of (text:weight); text holding another colon is not one."""
    colon = content.rfind(":")
    if colon <= 0 or colon == len(content) - 1:
        return None
    before, after = content[:colon].strip(), content[colon + 1 :].strip()
    if not _SD_NUMBER.fullmatch(after) or not before or ":" in before:
        return None
    return float(after)


def _unescape(text: str, chars: str = _SD_ESCAPABLE) -> str:
    out, i = [], 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in chars:
            out.append(text[i + 1])
            i += 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _weight_text(weight: float) -> str:
    fixed = f"{weight:.5f}"
    trimmed = re.sub(r"\.?0+$", "", fixed)
    return "0" if trimmed == "-0" else trimmed
