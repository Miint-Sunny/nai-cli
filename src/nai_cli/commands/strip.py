"""strip: copies of images without generation settings, EXIF or stealth data, for sharing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import errors, images, meta
from nai_cli.commands import copies
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class StripSettings:
    targets: copies.Targets
    # --keep-format: each copy in its source's format; without it every copy is a PNG
    keep_format: bool


def resolve(opts: dict[str, Any], ctx: Context) -> StripSettings:
    keep_format = bool(opts.get("keep_format"))
    return StripSettings(copies.resolve(opts, ctx, copies.FORMATS if keep_format else copies.PNG), keep_format)


def run(s: StripSettings, ctx: Context) -> dict[str, Any]:
    warnings: list[str] = []

    def make(source: Path) -> bytes:
        data = images.strip_in_format(source, warnings) if s.keep_format else images.strip_file(source)
        return checked(data, source.name)

    return copies.write(s.targets, "clean", make, warnings)


def checked(data: bytes, name: str) -> bytes:
    """The bytes of a strip copy, or bad_args naming the file when they still hold generation settings, stealth
    data, EXIF or XMP, or cannot be read back. Every copy is checked before any is written, so one refused leaves
    none behind."""
    left = meta.left_in(data)
    if left:
        raise errors.NaiError(errors.BAD_ARGS, f"{name} would still hold {', '.join(left)}; no copy is written")
    return data
