"""strip: copies of images without generation settings, EXIF or stealth data, for sharing."""

from typing import Any

from nai_cli import images
from nai_cli.commands import copies
from nai_cli.commands.context import Context


def resolve(opts: dict[str, Any], ctx: Context) -> copies.Targets:
    return copies.resolve(opts, ctx)


def run(s: copies.Targets, ctx: Context) -> dict[str, Any]:
    return copies.write(s, "clean", images.strip_file)
