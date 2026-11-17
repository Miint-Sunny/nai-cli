"""upscale and the Director tools: priced by the source's size, paid under the lock, saved like generations."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nai_cli import api, cost, decode, errors, images, meta, naming, pixel_snap, presets, values
from nai_cli.charge import authorize, keep_usable, price_json
from nai_cli.commands import output, texts
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class SnapOptions:
    # "auto", 0 for none, or a palette size
    colors: int | str
    max_detail: float | None
    # Scale the cells back up to about the source's size
    upscale: bool


@dataclass(frozen=True)
class ImageJob:
    command: str
    image: Path
    output: output.Output
    dry_run: bool
    yes: bool
    # The {type} a file name template gives the results
    kind: str
    # Names of the images of the reply, for the default file names
    outputs: tuple[str, ...] = ()
    tool: str | None = None
    prompt: str = ""
    defry: int = 0
    snap: SnapOptions | None = None


def resolve_upscale(opts: dict[str, Any], ctx: Context) -> ImageJob:
    return ImageJob(
        "upscale",
        Path(opts["image"]),
        output.resolve(opts.get("output"), ctx.config),
        bool(opts.get("dry_run")) or ctx.dry_run,
        bool(opts.get("yes")),
        "upscale",
    )


def resolve_tool(opts: dict[str, Any], ctx: Context) -> ImageJob:
    tool = opts["tool"]
    emotion = opts.get("emotion") or presets.EMOTIONS[0]
    prompt = texts.read_typed(opts, ("prompt",), ctx)["prompt"] or ""
    return ImageJob(
        f"tool-{tool}",
        Path(opts["image"]),
        output.resolve(opts.get("output"), ctx.config),
        bool(opts.get("dry_run")) or ctx.dry_run,
        bool(opts.get("yes")),
        # The official client names emotion results by the emotion too
        f"emotion-{emotion}" if tool == "emotion" else tool,
        presets.TOOL_OUTPUTS.get(tool, ()),
        tool,
        presets.emotion_prompt(emotion, prompt) if tool == "emotion" else prompt,
        opts.get("defry") or 0,
        _snap_options(opts) if tool == presets.PIXEL_SNAP else None,
    )


def _snap_options(opts: dict[str, Any]) -> SnapOptions:
    text = str(opts.get("colors") or "auto").strip().lower()
    if text in ("auto", "off"):
        colors: int | str = 0 if text == "off" else "auto"
    elif values.is_digits(text) and presets.SNAP_COLORS[0] <= int(text) <= presets.SNAP_COLORS[1]:
        colors = int(text)
    else:
        low, high = presets.SNAP_COLORS
        raise errors.NaiError(errors.BAD_ARGS, f"--colors is auto, off or {low} to {high}: {text}")
    max_detail = None if opts.get("avoid_over_refining") else pixel_snap.MAX_DETAIL
    return SnapOptions(colors, max_detail, bool(opts.get("upscale")))


def run_upscale(s: ImageJob, ctx: Context) -> dict[str, Any]:
    # Upscale never resizes the source; larger images are refused
    result, out = _run(s, ctx, lambda w, h, opus: cost.upscale(w, h), _as_is, lambda c, data, w, h: c.upscale(data))
    if out:
        ow, oh = decode.size_of_bytes(out[0])
        result["size"] = f"{ow}x{oh}"
    return result


def run_tool(s: ImageJob, ctx: Context) -> dict[str, Any]:
    if s.snap is not None:
        return _run_snap(s, s.snap, ctx)
    tool = s.tool or ""
    result, _ = _run(
        s,
        ctx,
        lambda w, h, opus: cost.augment(w, h, bg_removal=tool == "bg-removal", opus=opus),
        images.tool_image,
        lambda c, data, w, h: c.augment(data, tool, w, h, s.prompt, s.defry, expected=max(len(s.outputs), 1)),
    )
    return result


def _run_snap(s: ImageJob, o: SnapOptions, ctx: Context) -> dict[str, Any]:
    """Runs here, as the official client runs it in the browser, so it sends nothing and costs nothing."""
    # One picture is snapped; an animation would lose its other frames
    img = images.open_still(s.image)
    size = {"size": f"{img.width}x{img.height}"}
    # Snapped in a dry run too, since an image it cannot snap fails there
    snapped = pixel_snap.snap(np.asarray(img.convert("RGBA")), o.colors, o.max_detail)
    if s.dry_run:
        output.check(s.output)
        return {"ok": True, "dry_run": True, **size, "cost": {"estimate": 0}, "cells": _cells(snapped)}
    rows, cols = snapped.rgb.shape[:2]
    alpha = snapped.alpha if snapped.alpha is not None else np.full((rows, cols), 255, np.uint8)
    result = Image.fromarray(np.dstack([snapped.rgb, alpha]), "RGBA")
    if o.upscale:
        # Half up, as the browser rounds the factor
        factor = max(1, int((img.width / cols + img.height / rows) / 2 + 0.5))
        result = result.resize((cols * factor, rows * factor), Image.Resampling.NEAREST)
    output.ready(s.output)
    warnings = [f"the image was {snapped.note}"] if snapped.note else []
    paths = _save(s, ctx, output.keep_free, [images.to_png(result)], _source_naming(s), warnings)
    out = {
        "ok": True,
        "images": paths,
        **size,
        "cost": {"estimate": 0, "spent": 0},
        "cells": _cells(snapped),
        "pitch": [round(p, 3) for p in snapped.pitch],
        "warnings": warnings,
    }
    if snapped.palette_size is not None:
        out["colors"] = snapped.palette_size
    return out


def _cells(snapped: pixel_snap.Snapped) -> str:
    rows, cols = snapped.rgb.shape[:2]
    return f"{cols}x{rows}"


def _as_is(img: Image.Image) -> tuple[bytes, int, int]:
    return images.to_png(img), img.width, img.height


def _run(
    s: ImageJob,
    ctx: Context,
    price_of: Callable[[int, int, bool], int],
    prepare: Callable[[Image.Image], tuple[bytes, int, int]],
    send: Callable[[api.Client, bytes, int, int], list[bytes]],
) -> tuple[dict[str, Any], list[bytes]]:
    """Price by source size, confirm under the lock, save, then measure the spend. prepare turns the source into
    the image and size to send."""
    img = images.open_image(s.image)
    w, h = img.size
    if price_of(w, h, True) < 0:
        raise errors.NaiError(errors.BAD_ARGS, f"{w}x{h} exceeds the maximum area of this endpoint")
    data, send_w, send_h = prepare(img)
    if s.dry_run:
        output.check(s.output)
        warnings: list[str] = []
        account = ctx.dry_account(warnings)
        price = price_of(w, h, account.opus if account else True)
        dry = {"ok": True, "dry_run": True, "size": f"{w}x{h}", "sent_size": f"{send_w}x{send_h}"}
        if warnings:
            dry["warnings"] = warnings
        dry["cost"] = price_json(price, account)
        refused = ctx.refusal(price, account, s.yes, dry["cost"])
        return ({**dry, "refused": refused} if refused else dry), []
    # Fails without a token before any folder is made
    ctx.client()
    output.ready(s.output)
    warnings: list[str] = []
    named = _source_naming(s)
    with ctx.charged(warnings, command=s.command, source=str(s.image)) as charge:
        price = price_of(w, h, charge.before.opus)
        authorize(price, max(len(s.outputs), 1), charge, ctx.config, s.yes, price_json(price, charge.before))

        def save(pictures: list[bytes | None]) -> list[str]:
            return _save(s, ctx, charge.keep, pictures, named, warnings)

        try:
            out = charge.pay(send, charge.client, data, send_w, send_h)
        except api.DamagedImages as e:
            keep_usable(e, save, warnings)
        paths = save(list(out))
    result: dict[str, Any] = {
        "ok": True,
        "images": paths,
        "cost": price_json(price, None, charge),
        "warnings": warnings,
    }
    if charge.client.retries:
        result["retries"] = charge.client.retries
    return result, out


def _source_naming(s: ImageJob) -> naming.Meta | None:
    """A tool's result is named and filed by the settings of the image it worked on, as the official client
    names it."""
    try:
        record = meta.parse(s.image)
    except errors.NaiError:
        # Without generation settings the default name is kept, as the official client keeps the source's name
        return None
    return output.recorded(record, s.kind)


def _save(
    s: ImageJob,
    ctx: Context,
    keep: output.Keep,
    pictures: list[bytes | None],
    named: naming.Meta | None,
    warnings: list[str],
) -> list[str]:
    """Named by the template when it has {type}, as the official client names these results, else the source's
    name and the tool's; numbered when taken. Either way a result of several keeps its label, as the official
    client adds it."""
    paths = []
    seed = named.seed if named is not None and named.seed is not None else 0
    template = s.output.template
    templated = template is not None and naming.names_type(template)
    for i, data in enumerate(pictures):
        if data is None:
            continue
        image = output.named(s.output, named, data, ctx.now(), templated=templated, index=i)
        label = s.outputs[i] if i < len(s.outputs) else (str(i) if len(pictures) > 1 else "")
        tail = f"_{label}" if label else ""
        if image.name is None and s.output.destination.exact is None:
            # A long source name is cut before the tool's name, which tells the results apart
            image = replace(image, name=s.image.stem)
            tail = f"_{_named_by(s)}{tail}"
        paths.append(output.save_image(keep, data, s.output, image, warnings, seed=seed, suffix=tail))
    return paths


def _named_by(s: ImageJob) -> str:
    # The official client names an emotion result by the tool, then the emotion
    return f"emotion_{s.kind.removeprefix('emotion-')}" if s.tool == "emotion" else s.kind
