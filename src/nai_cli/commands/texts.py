"""Text typed for an option: @FILE reads a file, @- standard input and @@ a literal @."""

import re
from typing import Any

from nai_cli import errors, fs, values
from nai_cli.commands.context import Context

STDIN = "@-"
# A path separator or a file extension, and no comma as tag lists have
_LOOKS_LIKE_PATH = re.compile(r"^[^,]*(?:[\\/]|\.[A-Za-z]\w*$)[^,]*$")
# Starts the way only a path does, so a comma later on does not matter
_PATH_START = re.compile(r"^(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|~[\\/]|[\\/])")


def read_typed(opts: dict[str, Any], names: tuple[str, ...], ctx: Context) -> dict[str, Any]:
    """The named options with their text as meant: a single value or a list for repeated options, None when not
    given. Only one of them may read standard input."""
    readers = [(name, i) for name in names for i, value in enumerate(_values(opts.get(name))) if value == STDIN]
    if len(readers) > 1:
        raise errors.NaiError(errors.BAD_ARGS, "only one option can read standard input with @-")
    # Standard input is taken as it is, so a leading @ there reads no file
    piped = ctx.read_stdin().strip() if readers else None
    out: dict[str, Any] = {}
    for name in names:
        value = opts.get(name)
        if isinstance(value, list):
            out[name] = [_sendable(piped if item == STDIN else text_arg(item)) for item in value]
        elif value is not None:
            out[name] = _sendable(piped if value == STDIN else text_arg(value))
        else:
            out[name] = None
    return out


def _sendable(text: str) -> str:
    # Checked once read, since a file or standard input can hold what an argument cannot
    reason = values.unsendable(text)
    if reason is not None:
        raise errors.NaiError(errors.BAD_ARGS, f"{reason} in {text!r}, which no request can carry")
    return text


def _values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def text_arg(value: str) -> str:
    """A value of @PATH is read from that file, and @@ stands for a literal @. Other text starting with @,
    such as the tag @_@, is sent as typed unless it looks like a path."""
    if not value.startswith("@"):
        return value
    if value.startswith("@@"):
        return value[1:]
    target = value[1:]
    try:
        path = values.path(target)
    except (OSError, ValueError, RuntimeError):
        path = None
    is_file, exists = (fs.is_file(path), fs.exists(path)) if path is not None else (False, False)
    if is_file:
        try:
            # Notepad saves UTF-8 with a byte order mark
            return path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError) as e:
            raise errors.NaiError(errors.BAD_ARGS, f"cannot read {path}: {e}") from e
    if target and (exists or _PATH_START.match(target) or _LOOKS_LIKE_PATH.match(target)):
        raise errors.NaiError(
            errors.BAD_ARGS, f"{value}: no such file to read the text from; write @@ to send text starting with @"
        )
    return value
