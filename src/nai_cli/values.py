"""Values as files and options give them: strict JSON, numbers that are not booleans, paths and option text."""

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, TypeGuard

_DIGITS = re.compile(r"[0-9]+")


def parse_json(text: str | bytes) -> Any:
    """json.loads for what files and replies hold; NaN, Infinity and numbers no float can hold are refused as
    ValueError."""
    try:
        return json.loads(text, parse_constant=_not_json, parse_float=_finite_float, parse_int=_float_sized_int)
    except RecursionError as e:
        raise ValueError("JSON is nested too deeply") from e


def _not_json(name: str) -> None:
    raise ValueError(f"{name} is not a JSON value")


def _finite_float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"{text} is too large for a number")
    return number


def _float_sized_int(text: str) -> int:
    number = int(text)
    if abs(number) > sys.float_info.max:
        raise ValueError(f"{text[:20]}... is too large for a number")
    return number


def to_json(value: Any, indent: int | None = None) -> str:
    """Strict JSON text that always encodes as UTF-8: NaN and Infinity are refused as ValueError, and lone
    surrogates, which POSIX file names and JSON escapes can carry, are written as \\u escapes."""
    text = json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False)
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


# Above this a float has gaps between whole numbers, so JavaScript writes it as it is
_JS_SAFE_INTEGER = 2**53


def as_js(value: Any) -> Any:
    """value with every whole float made an int, as JSON.stringify writes 4.0 as 4, so a request body is the
    official client's byte for byte."""
    if isinstance(value, float) and value.is_integer() and abs(value) < _JS_SAFE_INTEGER:
        return int(value)
    if isinstance(value, dict):
        return {k: as_js(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [as_js(v) for v in value]
    return value


def path(text: str | Path) -> Path:
    """A path as typed or written in the config; ~ is the home folder on every platform, not only in a shell."""
    try:
        return Path(text).expanduser()
    except RuntimeError as e:
        # No home folder is known, as in some containers
        raise ValueError(f"{text}: {e}") from None


# Bundles held in an image keep both suffixes, so a number goes before them
_DOUBLE_SUFFIXES = (".naiv4vibebundle.png",)


def split_name(path: Path) -> tuple[str, str]:
    """The name before its suffix, and the suffix as written."""
    lower = path.name.lower()
    suffix = next((s for s in _DOUBLE_SUFFIXES if lower.endswith(s)), path.suffix)
    stem = path.name[: len(path.name) - len(suffix)]
    return stem, path.name[len(stem) :]


def is_whole(value: Any) -> TypeGuard[int]:
    # bool is a subclass of int
    return isinstance(value, int) and not isinstance(value, bool)


def is_digits(text: str) -> bool:
    """ASCII digits only; str.isdigit also takes ² and other digits int() refuses."""
    return _DIGITS.fullmatch(text) is not None


def option(flag: str, text: str) -> str:
    """flag=text, which nai reads back as text: after =, since it may start with -, and a leading @ doubled, since
    @PATH reads a file."""
    return f"{flag}={'@' + text if text.startswith('@') else text}"


def unsendable(text: str) -> str | None:
    """Why no request can carry text, if none can: a NUL, or half of a surrogate pair, which POSIX arguments that
    are not UTF-8 and JSON escapes can hold."""
    if "\x00" in text:
        return "a NUL character"
    if any("\ud800" <= ch <= "\udfff" for ch in text):
        return "text that is not valid UTF-8"
    return None


def is_number(value: Any) -> TypeGuard[int | float]:
    """Finite and within float range too, since Python reads NaN, Infinity and integers of any size."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return abs(value) <= sys.float_info.max if isinstance(value, int) else math.isfinite(value)
