"""vibe: write Vibes to a .naiv4vibe or .naiv4vibebundle file, or into a PNG, encoding the images that need it."""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import caches, cost, decode, errors, fs, images, models, naming, values, vibe
from nai_cli.charge import authorize, price_json
from nai_cli.commands import output
from nai_cli.commands.context import Context
from nai_cli.commands.generate import remember
from nai_cli.commands.references import VibeRef, load_vibes, vibe_ref


@dataclass(frozen=True)
class VibeSettings:
    model: models.Model
    sources: tuple[VibeRef, ...]
    # The name given with -o, before fitting it to the file system
    exact: Path | None
    into: Path | None
    keep_image: bool
    dry_run: bool
    yes: bool


def resolve(opts: dict[str, Any], ctx: Context) -> VibeSettings:
    model = models.resolve(opts.get("model") or "v4.5")
    if not model.vibe:
        raise errors.NaiError(errors.MODEL_UNSUPPORTED, f"{model.name} does not support Vibe", feature="vibe")
    exact = values.path(opts["output"]) if opts.get("output") else None
    return VibeSettings(
        model=model,
        sources=tuple(vibe_ref(v, opts.get("strength"), opts.get("info")) for v in opts["sources"]),
        exact=exact,
        into=Path(opts["into"]) if opts.get("into") else None,
        keep_image=opts.get("keep_image") is not False,
        dry_run=bool(opts.get("dry_run")) or ctx.dry_run,
        yes=bool(opts.get("yes")),
    )


def run(s: VibeSettings, ctx: Context) -> dict[str, Any]:
    sources = load_vibes(s.sources, s.model.id)
    bundled = len(sources) > 1 or s.into is not None or str(s.exact or "").lower().endswith(".naiv4vibebundle")
    target = _target(s, bundled, ctx)
    # Made and tried before paying, as for images; a dry run only looks
    (output.check_folder if s.dry_run else output.ready_folder)(target.parent, target if s.exact else None)
    holder = _holding_png(s.into) if s.into else None
    export = _Export(s.model.id, sources, [None] * len(sources), bundled, holder, not s.keep_image)
    export.look_up()
    # A damaged image fails here, before any other is paid for
    for encode in export.encodes():
        vibe.check_image(encode.source)
    result: dict[str, Any] = {"ok": True, "file": str(target), "model": s.model.id, "vibes": len(sources)}
    if s.dry_run:
        warnings: list[str] = []
        account = ctx.dry_account(warnings)
        to_encode = len(export.encodes())
        price = cost.VIBE_ENCODE_COST * to_encode
        result.update(
            file=str(fs.free_name(target)), cached=not to_encode, dry_run=True, cost=price_json(price, account)
        )
        if warnings:
            result["warnings"] = warnings
        refused = ctx.refusal(price, account, s.yes, result["cost"])
        return {**result, "refused": refused} if refused else result
    if not export.pending:
        return _unpaid(result, export, target)
    return _encode(s, ctx, export, target, result)


@dataclass
class _Export:
    """The Vibes to write; encodings are filled in as they are found or paid for."""

    model_id: str
    sources: list[vibe.VibeSource]
    found: list[str | None]
    bundled: bool
    holder: bytes | None
    without_images: bool

    @property
    def pending(self) -> list[int]:
        return [i for i, encoding in enumerate(self.found) if encoding is None]

    def encodes(self) -> list[vibe.Encode]:
        return vibe.encodes([(i, self.sources[i]) for i in self.pending], self.model_id)

    def look_up(self) -> None:
        # Encodings come from the files or the cache, so an image encoded before costs nothing again
        for i in self.pending:
            self.found[i] = caches.vibe_encoding(self.sources[i], self.model_id)

    def content(self) -> str | bytes:
        entries = [
            vibe.export(source, self.model_id, self.found[i], self.without_images)
            for i, source in enumerate(self.sources)
        ]
        text = vibe.dumps(vibe.bundle(entries) if self.bundled else entries[0])
        return vibe.embed(self.holder, text) if self.holder is not None else text


def _unpaid(result: dict[str, Any], export: _Export, target: Path) -> dict[str, Any]:
    file = output.save_unpaid(export.content(), target)
    return {**result, "file": file, "cached": True, "cost": {"estimate": 0, "spent": 0}}


def _encode(s: VibeSettings, ctx: Context, export: _Export, target: Path, result: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    with ctx.charged(warnings, command="vibe", output=str(target)) as charge:
        client = charge.client
        # Another process may have encoded the same images while this one waited
        export.look_up()
        if not export.pending:
            return _unpaid(result, export, target)
        encodes = export.encodes()
        price = cost.VIBE_ENCODE_COST * len(encodes)
        authorize(price, 0, charge, ctx.config, s.yes, price_json(price, charge.before))
        for encode in encodes:
            source = encode.source
            image = base64.b64decode(source.image_b64 or "")
            encoding = charge.pay(client.encode_vibe, image, s.model.id, source.information_extracted, source.mask)
            for i in encode.positions:
                export.found[i] = encoding
            remember(source, s.model.id, encoding, warnings)
        result["file"] = charge.keep(export.content(), target, image=False)
    result.update(cached=False, cost=price_json(price, None, charge), warnings=warnings)
    if client.retries:
        result["retries"] = client.retries
    return result


def _target(s: VibeSettings, bundled: bool, ctx: Context) -> Path:
    first = s.sources[0].path
    # Cut to fit as an image's -o is, since a name refused on saving fails after paying
    if s.exact is not None and s.exact.name in ("", ".."):
        raise errors.NaiError(errors.BAD_ARGS, "-o needs a file name")
    return naming.fitted(s.exact) if s.exact else naming.vibe_path(first, bundled, s.into, ctx.now())


def _holding_png(path: Path) -> bytes:
    """Decoded in full before anything is paid for, so a damaged image fails here and not after encoding."""
    img = images.open_image(path)
    try:
        data = path.read_bytes()
    except OSError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    return data if data.startswith(decode.PNG_SIGNATURE) else images.to_png(img)
