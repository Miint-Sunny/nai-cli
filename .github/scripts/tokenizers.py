"""Fetch the official tokenizer files into a folder, kept by the CI cache, then install them where nai-cli reads them.

Usage: tokenizers.py download DIR | install DIR
"""

import shutil
import sys
import time
from pathlib import Path

import httpx

from nai_cli import api, caches, tokenizer

# Seconds between attempts; the download fails the run only after all of them
DELAYS = [5, 15, 30, 60]


def _usable(kind: str, path: Path) -> bool:
    try:
        tokenizer.parse(kind, path.read_bytes())
    except (OSError, ValueError):
        return False
    return True


def _fetch(kind: str) -> bytes:
    name = tokenizer.FILES[kind]
    for attempt, delay in enumerate([*DELAYS, None], 1):
        try:
            raw = api.fetch_tokenizer(name, tokenizer.VERSION)
            tokenizer.parse(kind, raw)
            return raw
        except (httpx.HTTPError, ValueError) as e:
            reason = f"{name} attempt {attempt}: {type(e).__name__}: {e}"
            if delay is None:
                print(f"::error::{reason}")
                sys.exit(1)
            print(f"::warning::{reason}; retrying in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def download(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for kind, name in tokenizer.FILES.items():
        path = folder / name
        # A restored cache leaves nothing to download
        if _usable(kind, path):
            print(f"{name}: cached")
            continue
        path.write_bytes(_fetch(kind))
        print(f"{name}: downloaded")


def install(folder: Path) -> None:
    for kind, name in tokenizer.FILES.items():
        source = folder / name
        if not _usable(kind, source):
            print(f"::error::{source} is missing or is not a {kind} tokenizer file")
            sys.exit(1)
        target = caches.tokenizer_path(kind)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(f"{name}: installed to {target}")


if __name__ == "__main__":
    commands = {"download": download, "install": install}
    if len(sys.argv) != 3 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](Path(sys.argv[2]))
