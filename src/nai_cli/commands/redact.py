"""redact: copies of images with regions pixelated, blurred or filled, for sharing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import errors, images, inpaint, share
from nai_cli.commands import copies, spec
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class RedactSettings:
    targets: copies.Targets
    redaction: share.Redaction
    keep_metadata: bool
    segment: inpaint.Segment


def resolve(opts: dict[str, Any], ctx: Context) -> RedactSettings:
    amount = opts.get("amount")
    if amount is not None and not 0 < amount <= 1:
        raise errors.NaiError(errors.BAD_ARGS, "--amount is a fraction of the short edge, above 0 and at most 1")
    opacity = opts.get("opacity")
    if opacity is not None and not 0 < opacity <= 1:
        raise errors.NaiError(errors.BAD_ARGS, "--opacity is above 0 and at most 1")
    redaction = share.Redaction(
        regions=tuple(inpaint.parse_region(r) for r in opts["region"]),
        effect=opts.get("effect") or "pixelate",
        amount=amount,
        color=copies.color(opts.get("color"), (0, 0, 0)),
        opacity=1.0 if opacity is None else opacity,
        invert=bool(opts.get("invert")),
    )
    return RedactSettings(
        copies.resolve(opts, ctx), redaction, bool(opts.get("keep_metadata")), spec.segmenter(ctx.config.local.sam_dir)
    )


def run(s: RedactSettings, ctx: Context) -> dict[str, Any]:
    warnings: list[str] = []

    def make(source: Path) -> bytes:
        return copies.finish(
            source, share.redact(images.open_still(source), s.redaction, s.segment), s.keep_metadata, warnings
        )

    return copies.write(s.targets, "redacted", make, warnings)
