"""Where a command saves images and what it names them; checked before anything is paid for."""

import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from nai_cli import config, decode, errors, fs, naming, values

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif")


@dataclass(frozen=True)
class Output:
    destination: naming.Destination
    # The config's file name template; a file name given with -o replaces it
    template: naming.Template | None


def resolve(target: str | None, cfg: config.Config) -> Output:
    """-o for this run: a .png name, or a folder without the config's folder rule."""
    template = _template(cfg, "filename", naming.parse)
    if not target:
        return Output(filed(cfg), template)
    path = values.path(target)
    suffix = path.suffix.lower()
    if suffix == ".png":
        return Output(naming.Destination(path.parent, exact=path), None)
    if suffix in IMAGE_SUFFIXES and not fs.is_folder(path):
        raise errors.NaiError(errors.BAD_ARGS, f"images are saved as PNG; give -o a .png name or a folder: {path}")
    return Output(naming.Destination(path), template)


def filed(cfg: config.Config) -> naming.Destination:
    """The output folder with the config's folder rule."""
    return naming.Destination(cfg.output_dir, _template(cfg, "folder", naming.parse_folders))


def recorded(record: dict[str, Any], kind: str) -> naming.Meta:
    """What a template names an image by, from the settings read back from it."""
    seed, steps, scale = record.get("seed"), record.get("steps"), record.get("scale")
    return naming.Meta(
        prompt=str(record.get("user_prompt", record.get("prompt")) or ""),
        seed=seed if values.is_whole(seed) else None,
        model=record.get("model"),
        steps=steps if values.is_whole(steps) else None,
        cfg=scale if values.is_number(scale) else None,
        sampler=str(record.get("sampler") or ""),
        type=kind,
        characters=tuple(naming.Character(str(c.get("prompt") or "")) for c in record.get("characters") or []),
    )


def _template(cfg: config.Config, key: str, parse: Callable[[str], naming.Template]) -> naming.Template | None:
    text = getattr(cfg, key)
    if not text:
        return None
    # The official client falls back to its default name instead; refusing here comes before any spend
    try:
        return parse(text)
    except ValueError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"{cfg.path}: [output] {key}: {e}") from None


def ready(output: Output) -> None:
    """Fail on an unusable output folder before anything is charged; the folders under it are made on saving. A
    name the system refuses falls back to the default name, so -o needs no trying here."""
    ready_folder(output.destination.checked_folder)


def ready_folder(folder: Path, name: Path | None = None) -> None:
    """name is a file given with -o, tried as well, as a file other than an image has no fallback name."""
    # A file is written and removed, since Windows takes NUL and names ending in a dot for folders that exist
    with _refused(folder, name):
        fs.make_folder(folder)
        fs.try_names(folder, name)


def check(output: Output) -> None:
    check_folder(output.destination.checked_folder)


def check_folder(folder: Path, name: Path | None = None) -> None:
    """ready_folder for a dry run, which leaves nothing behind."""
    with _refused(folder, name):
        fs.try_names(folder, name)


@contextmanager
def _refused(folder: Path, name: Path | None) -> Iterator[None]:
    try:
        yield
    except (OSError, ValueError) as e:
        where = f"as {name}" if name else f"in the output folder {folder}"
        raise errors.NaiError(errors.BAD_ARGS, f"cannot save {where}: {e}") from e


def save_unpaid(data: bytes | str, path: Path) -> str:
    """Write a result nothing was paid for, under path or the next free number."""
    try:
        return str(fs.create_numbered(path, data))
    except (OSError, ValueError) as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot write {path}: {e}") from e


@dataclass(frozen=True)
class Named:
    """One image's values for the folder rule and the template, and the name the template gives, if any."""

    meta: naming.Meta
    name: str | None


def named(
    output: Output, base: naming.Meta | None, data: bytes, when: datetime, templated: bool = True, **fields: Any
) -> Named:
    """base is None when the source of a tool's result has no settings: the default name, in the folder of
    the time. templated is False where the template does not name such results."""
    try:
        width, height = decode.size_of_bytes(data)
    except decode.BAD_IMAGE:
        width = height = None
    meta = replace(base or naming.Meta(), id=str(uuid.uuid4()), date=when, width=width, height=height, **fields)
    if output.template is None or base is None or not templated:
        return Named(meta, None)
    return Named(meta, naming.render(output.template, meta) or naming.fallback(meta))


# Saves bytes under a path or the next free number there, returning where
Keep = Callable[[bytes, Path], str]


def keep_free(data: bytes, path: Path) -> str:
    """Keep for results nothing was paid for."""
    try:
        return str(fs.create_numbered(path, data))
    except (OSError, ValueError) as e:
        raise errors.NaiError(errors.SAVE_FAILED, f"cannot save the result: {e}") from e


def save_image(
    keep: Keep,
    data: bytes,
    output: Output,
    image: Named,
    warnings: list[str],
    seed: int = 0,
    index: int = 0,
    suffix: str = "",
) -> str:
    destination = output.destination
    chosen = destination.image_path(image.meta, seed, index, suffix, image.name)
    try:
        return keep(data, chosen)
    except errors.NaiError as e:
        fallback = destination.fallback_path(image.meta, seed, index, suffix)
        if e.code != errors.SAVE_FAILED or fallback == chosen:
            raise
        # A name or folder the file system refuses must not lose an image already made
        reason = e.__cause__ or e.message
        warnings.append(f"cannot save as {chosen.name}: {reason}; the default name in {fallback.parent} is used")
        return keep(data, fallback)
