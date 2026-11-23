"""The images a command works on: those named on the command line, and those inside the folders named there."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from nai_cli import fs

# What a folder is searched for by the commands that copy images; nai meta has its own, the formats the official
# client writes settings into
SUFFIXES = (".png", ".webp", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Source:
    """An image a command works on, and where it lies inside the folder given for it: its name alone when the image
    itself was given."""

    path: Path
    relative: Path


def expand(given: Iterable[Path], suffixes: tuple[str, ...], recursive: bool) -> tuple[Source, ...]:
    """The images in the paths given, by name inside each folder: an image stands for itself, and a folder for the
    images in it, or with recursive in it and in the folders inside it, known by their suffix whatever its case."""
    out: list[Source] = []
    for path in given:
        if not fs.is_folder(path):
            out.append(Source(path, Path(path.name)))
            continue
        pattern = "**/*" if recursive else "*"
        found = sorted(p for p in path.glob(pattern) if p.suffix.lower() in suffixes and fs.is_file(p))
        out += [Source(p, p.relative_to(path)) for p in found]
    return tuple(out)
