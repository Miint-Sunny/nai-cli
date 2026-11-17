"""Prompt text rules: quality suffix, V5 automatic text block, UC presets and nsfw."""

import re
import unicodedata
from dataclasses import dataclass

MIX_SEPARATOR = "|"
MIX_ESCAPE = "||"
MAX_MIX_CHUNKS = 6

# JavaScript's \s, which the official rules use: unlike Python's it excludes \x1c-\x1f and \x85 and includes \ufeff
JS_SPACE = "\t\n\v\f\r \u00a0\u1680" + "".join(map(chr, range(0x2000, 0x200B))) + "\u2028\u2029\u202f\u205f\u3000\ufeff"
_S = "[" + re.escape(JS_SPACE) + "]"
TEXT_MARKER = re.compile(r"(?:^|" + _S + r"|[,.:\[\]{}、。])text:(?!:)", re.IGNORECASE)
ENHANCE_HINT = ", -2::upscaled, blurry::,"
_ENHANCE_HINT_PRESENT = "upscaled, blurry"
AUTO_TEXT_MARKER = "teXt:"
_GENERATED_MARKER = re.compile(r"(?:^|" + _S + r"|[,.:\[\]{}、。])teXt:(?!:)")
CJK = re.compile(r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]")
_QUOTE_PAIRS = {'"': '"', "“": "”", "「": "」", "'": "'", "‘": "’"}
_SINGLE_QUOTE_BOUNDARY = set(JS_SPACE + ",.")
NSFW = "nsfw"
FURRY_PREFIX = "fur dataset, "
_DATASET_PREFIXES = ("fur dataset", "background dataset")
_NSFW = re.compile(r"[\{\[]*nsfw[\}\]]*", re.IGNORECASE)
_NSFW_WITH_SEP = re.compile(r"[\{\[]*nsfw[\}\]]*" + _S + "*,?" + _S + "*", re.IGNORECASE)
_TRAILING_SEPARATORS = re.compile("[" + re.escape(JS_SPACE) + ",]+$")


def js_trim(text: str) -> str:
    return text.strip(JS_SPACE)


def split_mix_chunks(prompt: str) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    escaped = False
    i = 0
    while i < len(prompt):
        ch = prompt[i]
        if ch == MIX_SEPARATOR and i + 1 < len(prompt) and prompt[i + 1] == MIX_SEPARATOR:
            escaped = not escaped
            buf.append(MIX_ESCAPE)
            i += 2
            continue
        if ch == MIX_SEPARATOR and not escaped:
            chunks.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    chunks.append("".join(buf))
    if len(chunks) <= MAX_MIX_CHUNKS:
        return chunks
    return [*chunks[: MAX_MIX_CHUNKS - 1], MIX_SEPARATOR.join(chunks[MAX_MIX_CHUNKS - 1 :])]


def js_length(text: str) -> int:
    # JavaScript string length counts UTF-16 code units
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def longest_alternatives(prompt: str) -> str:
    """Each ||a|b|| group becomes its longest option, ties to the later one, as the official token counter does."""
    parts = prompt.split(MIX_ESCAPE)
    for i in range(1, len(parts), 2):
        best, *rest = parts[i].split(MIX_SEPARATOR)
        for option in rest:
            if js_length(best) <= js_length(option):
                best = option
        parts[i] = best
    return "".join(parts)


def apply_furry(prompt: str) -> str:
    """Furry mode starts the prompt with fur dataset unless it already names a dataset."""
    return prompt if prompt.startswith(_DATASET_PREFIXES) else FURRY_PREFIX + prompt


def append_suffix(prompt: str, suffix: str | None) -> str:
    """Appends ", suffix" without trimming, so "wind," becomes "wind,, suffix" as in the official client."""
    if not suffix:
        return prompt
    return f"{prompt}, {suffix}" if prompt else suffix


def apply_suffix(prompt: str, suffix: str | None) -> str:
    """The suffix goes into the first mix segment before the first text:, and every text: marker takes the first one's spelling."""
    if not suffix:
        return prompt
    chunks = split_mix_chunks(prompt)
    match = TEXT_MARKER.search(chunks[0])
    if match is None:
        chunks[0] = append_suffix(chunks[0], suffix)
    else:
        pieces = TEXT_MARKER.split(chunks[0])
        pieces[0] = append_suffix(pieces[0], suffix)
        chunks[0] = match.group(0).join(pieces)
    return MIX_SEPARATOR.join(chunks)


def compose_suffix(quality: str | None, transparent: bool) -> str | None:
    if not transparent:
        return quality or None
    return f"transparent background, {quality}" if quality else "transparent background"


@dataclass
class AutoTextCharacter:
    prompt: str
    x: float = 0.5
    y: float = 0.5


def quoted_ranges(prompt: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(prompt):
        opening = prompt[cursor]
        closing = _QUOTE_PAIRS.get(opening)
        accepts = closing is not None and (opening != "'" or _is_single_quote_boundary(prompt, cursor - 1))
        if not accepts:
            cursor += 1
            continue
        apostrophe_style = closing in ("'", "’")
        end = cursor + 1
        while end < len(prompt) and (
            prompt[end] != closing or (apostrophe_style and _is_letter_or_number(prompt, end + 1))
        ):
            end += 1
        if end >= len(prompt):
            cursor += 1
            continue
        ranges.append((cursor, end + 1))
        cursor = end + 1
    return ranges


def _is_letter_or_number(text: str, index: int) -> bool:
    # JavaScript sees half of a surrogate pair there, which is neither
    if not 0 <= index < len(text) or ord(text[index]) > 0xFFFF:
        return False
    return unicodedata.category(text[index])[0] in "LN"


def _is_single_quote_boundary(text: str, index: int) -> bool:
    if index < 0 or index >= len(text):
        return True
    return text[index] in _SINGLE_QUOTE_BOUNDARY


def _extract_quoted(prompt: str) -> list[str]:
    out = []
    for start, end in quoted_ranges(prompt):
        value = js_trim(prompt[start + 1 : end - 1])
        if value:
            out.append(value)
    return out


def _split_rows(chars: list[AutoTextCharacter]) -> list[list[AutoTextCharacter]]:
    if len(chars) <= 1:
        return [chars]
    total_span = chars[-1].y - chars[0].y
    split_index, largest_gap = 1, -1.0
    for i in range(1, len(chars)):
        gap = chars[i].y - chars[i - 1].y
        if gap > largest_gap:
            largest_gap, split_index = gap, i
    if total_span <= 0.15 and largest_gap <= 0.1:
        return [chars]
    return [*_split_rows(chars[:split_index]), *_split_rows(chars[split_index:])]


def _reading_order(chars: list[AutoTextCharacter]) -> list[AutoTextCharacter]:
    by_y = sorted(chars, key=lambda c: c.y)
    return [c for row in _split_rows(by_y) for c in sorted(row, key=lambda c: c.x)]


def auto_text_block(prompt: str, chars: list[AutoTextCharacter], use_coords: bool) -> str | None:
    enabled = [c for c in chars if c.prompt]
    if TEXT_MARKER.search(prompt) or any(TEXT_MARKER.search(c.prompt) for c in enabled):
        return None
    texts = _collect_quoted(split_mix_chunks(prompt)[0], enabled, use_coords)
    if not texts:
        return None
    return f"{AUTO_TEXT_MARKER} " + "\n\n".join(texts)


def apply_auto_text(prompt: str, chars: list[AutoTextCharacter], use_coords: bool) -> str:
    """V5 copies quoted text into a trailing teXt: block so the model renders it as text."""
    block = auto_text_block(prompt, chars, use_coords)
    if block is None:
        return prompt
    chunks = split_mix_chunks(prompt)
    base = _TRAILING_SEPARATORS.sub("", chunks[0])
    chunks[0] = block if not base else f"{base}, {block}"
    return MIX_SEPARATOR.join(chunks)


def strip_auto_text(prompt: str, chars: list[AutoTextCharacter], use_coords: bool) -> str:
    out = []
    for chunk in split_mix_chunks(prompt):
        match = _GENERATED_MARKER.search(chunk)
        if match is None:
            out.append(chunk)
            continue
        prefix = chunk[: match.start()]
        expected = "\n\n".join(_collect_quoted(prefix, chars, use_coords))
        if js_trim(chunk[match.end() :]) != expected:
            out.append(chunk)
            continue
        out.append(_TRAILING_SEPARATORS.sub("", prefix))
    return MIX_SEPARATOR.join(out)


def _collect_quoted(base: str, chars: list[AutoTextCharacter], use_coords: bool) -> list[str]:
    ordered = _reading_order(chars) if use_coords else chars
    groups = [_extract_quoted(base), *(_extract_quoted(c.prompt) for c in ordered)]
    combined = "".join(t for g in groups for t in g)
    cjk = len(CJK.findall(combined))
    # Quote order is reversed within each group when the text is mostly CJK
    if cjk and cjk / js_length(combined) > 0.3:
        groups = [list(reversed(g)) for g in groups]
    return [t for g in groups for t in g]


def add_enhance_hint(text: str) -> str:
    """Add the Enhance down-weighting before the first text: segment or at the end, unless already present."""
    if _ENHANCE_HINT_PRESENT in text:
        return text
    match = TEXT_MARKER.search(text)
    if match is None:
        return text + ENHANCE_HINT
    return text[: match.start()] + ENHANCE_HINT + text[match.start() :]


def compose_negative(negative: str, preset_text: str, nsfw: bool) -> str:
    """Prefix the first mix segment with the UC preset and then nsfw; the user's text is untouched."""
    chunks = negative.split(MIX_SEPARATOR)
    if preset_text:
        chunks[0] = f"{preset_text}, {chunks[0]}" if chunks[0] else preset_text
    text = MIX_SEPARATOR.join(chunks)
    return f"{NSFW}, {text}" if nsfw else text


def contains_nsfw(prompt: str) -> bool:
    return bool(_NSFW.search(prompt))


def official_character_prompt(text: str) -> str:
    """The official client sends the first 1girl and 1boy of a character prompt as girl and boy."""
    return text.replace("1girl", "girl", 1).replace("1boy", "boy", 1)


def remove_nsfw(negative: str) -> str:
    """Remove nsfw, {nsfw} and [nsfw] with their trailing comma; text without nsfw is untouched."""
    if not negative or not _NSFW.search(negative):
        return negative
    result = _NSFW_WITH_SEP.sub("", negative)
    result = re.sub("," + _S + "*,", ",", result)
    result = re.sub("^" + _S + "*," + _S + "*", "", result)
    result = re.sub(_S + "*," + _S + "*$", "", result)
    return js_trim(result)


def nsfw_removal_damage(text: str) -> list[str]:
    """Tags containing more than a bare nsfw, which removal would break, such as 2::nsfw:: becoming 2::::."""
    return [t.strip() for t in text.split(",") if _NSFW.search(t) and not _NSFW.fullmatch(t.strip())]


def strip_nsfw_prefix(negative: str) -> str:
    """Strip a leading nsfw and its ", ", the only form recognized when reading images."""
    if not negative.startswith(NSFW):
        return negative
    return negative[len(NSFW) :].removeprefix(", ")


def strip_uc_preset(negative: str, preset_text: str) -> tuple[bool, str]:
    """Inverse of compose_negative after nsfw removal: drop the preset and its ", " from the first mix segment."""
    chunks = negative.split(MIX_SEPARATOR)
    if not preset_text or not chunks[0].startswith(preset_text):
        return False, negative
    chunks[0] = chunks[0][len(preset_text) :].removeprefix(", ")
    return True, MIX_SEPARATOR.join(chunks)


def strip_suffix(prompt: str, suffix: str) -> tuple[bool, str]:
    """Inverse of apply_suffix, recovering the prompt as written."""
    chunks = split_mix_chunks(prompt)
    match = TEXT_MARKER.search(chunks[0])
    pieces = [chunks[0]] if match is None else TEXT_MARKER.split(chunks[0])
    head = pieces[0]
    if head == suffix:
        pieces[0] = ""
    elif head.endswith(", " + suffix):
        pieces[0] = head[: -len(", " + suffix)]
    else:
        return False, prompt
    chunks[0] = pieces[0] if match is None else match.group(0).join(pieces)
    return True, MIX_SEPARATOR.join(chunks)


_WEIGHT_NUMBER = re.compile(r"(?<![\w.])(-?[0-9]*\.?[0-9]+)$")
# A number right before :: starts a new weight, so one before a closing :: weights every following tag
NUMBER_BEFORE_CLOSE = re.compile(r"-?\d*\.?\d*$")
_SPACED_NUMBER = re.compile(r"(?:^|" + _S + r"|,)(-?\d*\.?\d* )$")
_NUMBER_AFTER = re.compile(r"^ ?-?\d*\.?\d*(?![\d.-]*::)")
# Only numbers up to 20 are flagged as misplaced weights
_PLAUSIBLE_WEIGHT = 21


@dataclass
class WeightMark:
    kind: str
    start: int
    end: int
    number: str = ""


def weight_marks(text: str) -> list[WeightMark]:
    """Classify each :: of n::...:: pairs: open, close, numeric_close, stray, spaced, number_after or unclosed."""
    marks: list[WeightMark] = []
    open_at: WeightMark | None = None
    for m in re.finditer("::", text):
        before = text[: m.start()]
        opening = _WEIGHT_NUMBER.search(before)
        if open_at is not None:
            content = text[open_at.end : m.start()]
            if opening and content.strip() == opening.group(1):
                open_at = WeightMark("open", opening.start(), m.end(), opening.group(1))
                marks.append(open_at)
                continue
            if opening and text[open_at.end : opening.start()].rstrip(" \t").endswith((",", MIX_SEPARATOR, "\n")):
                # A number as its own tag switches weights without closing
                open_at = WeightMark("switch", opening.start(), m.end(), opening.group(1))
                marks.append(open_at)
                continue
            number = NUMBER_BEFORE_CLOSE.search(before).group(0)
            marks.append(WeightMark("numeric_close" if number else "close", m.start(), m.end(), number))
            open_at = None
            continue
        if opening:
            open_at = WeightMark("open", opening.start(), m.end(), opening.group(1))
            marks.append(open_at)
            continue
        if before[-4:].lower() == "text":
            continue
        spaced = _SPACED_NUMBER.search(before)
        spaced_weight = _as_weight(spaced.group(1)) if spaced else None
        after = _NUMBER_AFTER.search(text[m.end() :])
        after_weight = _as_weight(after.group(0)) if after and after.group(0).strip() else None
        if spaced_weight is not None and abs(spaced_weight) < _PLAUSIBLE_WEIGHT:
            # Meant to open a weight, so the next :: closes it instead of being reported as stray
            open_at = WeightMark("spaced", spaced.start(1), m.end(), spaced.group(1).strip())
            marks.append(open_at)
        elif after_weight is not None and abs(after_weight) < _PLAUSIBLE_WEIGHT:
            marks.append(WeightMark("number_after", m.start(), m.end() + len(after.group(0)), after.group(0).strip()))
        else:
            marks.append(WeightMark("stray", m.start(), m.end(), NUMBER_BEFORE_CLOSE.search(before).group(0)))
    if open_at is not None:
        marks.append(WeightMark("unclosed", open_at.start, open_at.end, open_at.number))
    return marks


def _as_weight(number: str) -> float | None:
    try:
        return float(number)
    except ValueError:
        return None


# Underscores in Danbooru kaomoji are part of the face and must survive underscore replacement
_KAOMOJI = [
    "(o)_(o)",
    "<o>_<o>",
    "<|>_<|>",
    "||_||",
    "0_0",
    "+_+",
    "+_-",
    "._.",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
]
_KAOMOJI_PATTERN = re.compile(
    r"(?<![^\W_])(?:" + "|".join(re.escape(k) for k in sorted(_KAOMOJI, key=len, reverse=True)) + r")(?![^\W_])"
)
_ANGLE = re.compile(r"<[^<>\n]*>")


def literal_ranges(text: str) -> list[tuple[int, int]]:
    """Spans that tidying and lint leave alone: quoted text copied into teXt:, text: segments up to the end
    of their mix segment, <markers> and kaomoji."""
    ranges = list(quoted_ranges(text))
    for m in TEXT_MARKER.finditer(text):
        end = m.end()
        while end < len(text):
            if text[end] == MIX_SEPARATOR:
                if end + 1 < len(text) and text[end + 1] == MIX_SEPARATOR:
                    end += 2
                    continue
                break
            end += 1
        ranges.append((m.end() - len("text:"), end))
    ranges += [(m.start(), m.end()) for m in _ANGLE.finditer(text)]
    ranges += [(m.start(), m.end()) for m in _KAOMOJI_PATTERN.finditer(text)]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def with_fixed(text: str, prefix: str, suffix: str) -> str:
    """Fixed tags around the prompt, joined with commas; empty parts drop out."""
    return ", ".join(part for part in (prefix, text, suffix) if js_trim(part))


def fill_drawn(text: str, stored: str, drawn: str) -> str:
    """Each ||a|b|| group of text replaced by the option an image drew for the same group, found by matching
    the image's stored prompt against the prompt it actually sent; text is kept when they do not match."""
    groups, stored_parts = text.split(MIX_ESCAPE), stored.split(MIX_ESCAPE)
    if len(groups) < 3 or len(groups) % 2 == 0 or len(stored_parts) % 2 == 0:
        return text
    pattern = "".join(
        re.escape(part) if i % 2 == 0 else "(" + "|".join(map(re.escape, part.split(MIX_SEPARATOR))) + ")"
        for i, part in enumerate(stored_parts)
    )
    match = re.fullmatch(pattern, drawn)
    if match is None:
        return text
    out, searched_from = [], 0
    for i, part in enumerate(groups):
        if i % 2 == 0:
            out.append(part)
            continue
        # Groups pair up in order, so a repeated group takes the next draw of that group
        found = next((j for j in range(searched_from, len(stored_parts) // 2) if stored_parts[2 * j + 1] == part), None)
        if found is None:
            out.append(f"{MIX_ESCAPE}{part}{MIX_ESCAPE}")
        else:
            out.append(match.group(found + 1))
            searched_from = found + 1
    return "".join(out)
