"""Sharing copies named after their source with a label, written all or none."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from nai_cli import decode, errors, fs, images, meta, naming, values
from nai_cli.commands import output, sources
from nai_cli.commands.context import Context
from nai_cli.commands.output import IMAGE_SUFFIXES
from nai_cli.commands.sources import Source

PNG = (".png",)
# With strip --keep-format a copy has the format of its source, one of these
FORMATS = (".png", ".webp", ".jpg", ".jpeg")
# The format each suffix names, as decode.format_of names formats
_FAMILIES = {".png": "PNG", ".webp": "WEBP", ".jpg": "JPEG", ".jpeg": "JPEG"}


@dataclass(frozen=True)
class Targets:
    sources: tuple[Source, ...]
    # A file name for one image; taken, it gets the next number, as every name does
    exact: Path | None
    folder: Path | None
    # Without -o: the output folder and its folder rule, as generated images are filed
    filed: naming.Destination
    # --rename: by the day and a number in it, not after the source
    rename: bool
    # The suffixes the copies may have
    formats: tuple[str, ...]
    when: datetime
    dry_run: bool


def resolve(opts: dict[str, Any], ctx: Context, formats: tuple[str, ...] = PNG) -> Targets:
    given = tuple(map(Path, opts["images"]))
    found = sources.expand(given, sources.SUFFIXES, bool(opts.get("recursive")))
    if not found:
        raise errors.NaiError(errors.BAD_ARGS, "no images in the paths given")
    out = values.path(opts["output"]) if opts.get("output") else None
    exact = folder = None
    if out is not None and out.suffix.lower() in formats:
        # A folder stands for as many images as it holds
        if len(found) > 1 or any(fs.is_folder(p) for p in given):
            raise errors.NaiError(errors.BAD_ARGS, "-o names one file; give a folder for several images")
        if opts.get("rename"):
            raise errors.NaiError(
                errors.BAD_ARGS, "--rename names copies by date and number: give -o a folder, or no -o"
            )
        exact = out
    elif out is not None:
        if out.suffix.lower() in IMAGE_SUFFIXES and not fs.is_folder(out):
            raise _no_name_for(out, formats)
        folder = out
    targets = Targets(
        found, exact, folder, output.filed(ctx.config), bool(opts.get("rename")), formats, ctx.now(), ctx.dry_run
    )
    return targets


def _no_name_for(out: Path, formats: tuple[str, ...]) -> errors.NaiError:
    if formats == PNG:
        return errors.NaiError(errors.BAD_ARGS, f"copies are saved as PNG; give -o a .png name or a folder: {out}")
    return errors.NaiError(
        errors.BAD_ARGS,
        f"copies keep the format of their source; give -o a name ending in {', '.join(formats)}, or a folder: {out}",
    )


def write(t: Targets, label: str, make: Callable[[Path], bytes], warnings: list[str] | None = None) -> dict[str, Any]:
    """NAME_label.EXT in the output folder, or NAME.EXT in the folder given under the place its source has in the
    folder it came from, EXT being the copy's; or, with --rename, by date and number there. warnings are those make
    adds and those about the names."""
    # Every image is made before anything is written, so a bad one leaves no partial set
    copies = [make(source.path) for source in t.sources]
    numbers = fs.Numbers(t.when)
    targets = [_target(t, source, label, data, numbers) for source, data in zip(t.sources, copies, strict=True)]
    notes = [*_download_names(t), *(warnings or [])]
    noted = {"warnings": notes} if notes else {}
    if t.dry_run:
        for target in targets:
            output.check_folder(target.parent, target if target == t.exact else None)
        return {"ok": True, "dry_run": True, "files": _listing(t.sources, _free_names(targets)), **noted}
    written: list[Path] = []
    try:
        for target, data in zip(targets, copies, strict=True):
            written.append(fs.create_numbered(target, data))
    except BaseException as e:
        for path in written:
            fs.remove(path)
        if isinstance(e, OSError | ValueError):
            raise errors.NaiError(errors.BAD_ARGS, f"cannot write {target}: {e}") from e
        raise
    return {"ok": True, "files": _listing(t.sources, written), **noted}


def _free_names(targets: list[Path]) -> list[Path]:
    """The names the copies would be written under, each after those before it."""
    taken: set[Path] = set()
    out = []
    for target in targets:
        path = fs.free_name(target, taken)
        taken.add(path)
        out.append(path)
    return out


def _target(t: Targets, source: Source, label: str, data: bytes, numbers: fs.Numbers) -> Path:
    """The name given with -o, if it ends as the copy's format does; else a name in the folder -o gives, under its
    source's place, or where a generated image is filed: by date and number with --rename, else after its source.
    A copy's suffix is its format's."""
    suffix = _suffix(source, data)
    if t.exact is not None:
        if _FAMILIES[t.exact.suffix.lower()] != decode.format_of(data):
            kind = decode.format_of(data).lower()
            raise errors.NaiError(
                errors.BAD_ARGS, f"{source.path.name} is copied as {kind}: give -o a {suffix} name or a folder"
            )
        return t.exact
    if t.folder is not None:
        folder, kept = t.folder / source.relative.parent, ""
    else:
        folder, kept = t.filed.folder_for(_kept(data, t.when)), "_" + label
    if t.rename:
        return numbers.name(folder, suffix)
    return folder / f"{naming.fit(source.path.stem, kept)}{suffix}"


def _suffix(source: Source, data: bytes) -> str:
    """A copy's suffix by the format of its bytes: its source's own spelling of that format where the source has
    one."""
    own = source.path.suffix.lower()
    if _FAMILIES.get(own) == decode.format_of(data):
        return own
    return {"PNG": ".png", "WEBP": ".webp"}.get(decode.format_of(data), ".jpg")


def _download_names(t: Targets) -> list[str]:
    """A copy named after its source keeps the start of the prompt and the seed where the source has NovelAI's
    download name."""
    if t.rename or t.exact is not None:
        return []
    count = sum(naming.download_name(source.path.stem) for source in t.sources)
    if not count:
        return []
    copies = "copy is" if count == 1 else "copies are"
    return [
        f"{count} {copies} named after NovelAI's download name, which holds the start of the prompt and the seed; "
        "--rename names copies by date and number instead"
    ]


def _kept(data: bytes, when: datetime) -> naming.Meta:
    # Filed by the settings the copy keeps, so one without them is filed by date and size alone
    width, height = decode.size_of_bytes(data)
    try:
        record = meta.parse_raw(meta.raw_of(data, "the copy"))
    except errors.NaiError:
        return naming.Meta(date=when, width=width, height=height)
    return replace(output.recorded(record, ""), date=when, width=width, height=height)


def _listing(found: tuple[Source, ...], files: list[Path]) -> list[dict[str, str]]:
    return [{"source": str(a.path), "file": str(b)} for a, b in zip(found, files, strict=True)]


def color(text: str | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if text is None:
        return default
    value = text.strip().removeprefix("#")
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise errors.NaiError(errors.BAD_ARGS, f"a colour is #RRGGBB: {text}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def finish(source: Path, edited: Image.Image, keep_metadata: bool, warnings: list[str]) -> bytes:
    """The edited pixels as a bare PNG; with keep_metadata, the source's PNG text is carried over as it is. Settings
    it keeps in EXIF or in the alpha channel, which the edit loses, are written as text in place of PNG text of the
    same keywords, so the copy reads back as the source did."""
    out = images.strip_metadata(edited)
    if not keep_metadata:
        return out
    data = source.read_bytes()
    try:
        settings = meta.raw_of(data, source.name)
    except errors.NaiError:
        settings = {}
    if not settings or "Comment" in meta.png_text_chunks(data):
        # The settings are the PNG text itself, or there are none
        return images.copy_text_chunks(data, out)
    out = images.copy_text_chunks(data, out, leaving=set(settings))
    return images.add_text(out, _writable(settings, source.name, warnings))


def _writable(settings: dict[str, str], name: str, warnings: list[str]) -> dict[str, str]:
    # EXIF and alpha-channel settings may hold any key, which PNG text cannot always carry
    out = {}
    for key, text in settings.items():
        try:
            decode.itxt(key, text)
        except ValueError as e:
            warnings.append(f"{name}: {e}, so the copy leaves it out")
            continue
        out[key] = text
    return out
