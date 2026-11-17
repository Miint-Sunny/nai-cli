"""Sharing copies named after their source with a label, written all or none."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from nai_cli import decode, errors, fs, images, meta, naming, values
from nai_cli.commands import output
from nai_cli.commands.context import Context
from nai_cli.commands.output import IMAGE_SUFFIXES


@dataclass(frozen=True)
class Targets:
    images: tuple[Path, ...]
    # A .png name for one image; taken, it gets the next number, as every name does
    exact: Path | None
    folder: Path | None
    # Without -o: the output folder and its folder rule, as generated images are filed
    filed: naming.Destination
    when: datetime
    dry_run: bool


def resolve(opts: dict[str, Any], ctx: Context) -> Targets:
    sources = tuple(map(Path, opts["images"]))
    given = values.path(opts["output"]) if opts.get("output") else None
    exact = folder = None
    if given is not None and given.suffix.lower() == ".png":
        if len(sources) > 1:
            raise errors.NaiError(errors.BAD_ARGS, "-o names one file; give a folder for several images")
        exact = given
    elif given is not None:
        if given.suffix.lower() in IMAGE_SUFFIXES and not fs.is_folder(given):
            raise errors.NaiError(errors.BAD_ARGS, f"copies are saved as PNG; give -o a .png name or a folder: {given}")
        folder = given
    return Targets(sources, exact, folder, output.filed(ctx.config), ctx.now(), ctx.dry_run)


def write(t: Targets, label: str, make: Callable[[Path], bytes], warnings: list[str] | None = None) -> dict[str, Any]:
    """NAME_label.png in the output folder, or NAME.png in the folder given; warnings are those make adds."""
    # Every image is made before anything is written, so a bad one leaves no partial set
    copies = [make(source) for source in t.images]
    targets = [_target(t, source, label, data) for source, data in zip(t.images, copies, strict=True)]
    noted = {"warnings": warnings} if warnings else {}
    if t.dry_run:
        for target in targets:
            output.check_folder(target.parent, target if target == t.exact else None)
        return {"ok": True, "dry_run": True, "files": _listing(t.images, _free_names(targets)), **noted}
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
    return {"ok": True, "files": _listing(t.images, written), **noted}


def _free_names(targets: list[Path]) -> list[Path]:
    """The names the copies would be written under, each after those before it."""
    taken: set[Path] = set()
    out = []
    for target in targets:
        path = fs.free_name(target, taken)
        taken.add(path)
        out.append(path)
    return out


def _target(t: Targets, source: Path, label: str, data: bytes) -> Path:
    if t.exact is not None:
        return t.exact
    if t.folder is not None:
        return t.folder / f"{naming.fit(source.stem)}.png"
    return t.filed.folder_for(_kept(data, t.when)) / f"{naming.fit(source.stem, '_' + label)}.png"


def _kept(data: bytes, when: datetime) -> naming.Meta:
    # Filed by the settings the copy keeps, so one without them is filed by date and size alone
    width, height = decode.size_of_bytes(data)
    try:
        record = meta.parse_raw(meta.raw_of(data, "the copy"))
    except errors.NaiError:
        return naming.Meta(date=when, width=width, height=height)
    return replace(output.recorded(record, ""), date=when, width=width, height=height)


def _listing(sources: tuple[Path, ...], files: list[Path]) -> list[dict[str, str]]:
    return [{"source": str(a), "file": str(b)} for a, b in zip(sources, files, strict=True)]


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
