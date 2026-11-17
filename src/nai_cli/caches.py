"""Files kept in the data folder between runs: the official tokenizer files and the Vibe encodings paid for."""

import contextlib
import functools
import hashlib
from pathlib import Path

import httpx

from nai_cli import api, config, fs, tokenizer, vibe


def tokenizer_path(kind: str) -> Path:
    return config.data_dir() / "tokenizers" / f"v{tokenizer.VERSION}" / tokenizer.FILES[kind]


@functools.cache
def load_tokenizer(kind: str) -> tokenizer.Tokenizer | None:
    """Read from the data folder, downloading on first use; None when neither works, and counts are then estimated."""
    path = tokenizer_path(kind)
    try:
        return tokenizer.parse(kind, path.read_bytes())
    except (OSError, ValueError):
        pass
    try:
        raw = api.fetch_tokenizer(tokenizer.FILES[kind], tokenizer.VERSION)
        loaded = tokenizer.parse(kind, raw)
    except (httpx.HTTPError, ValueError):
        return None
    # Counting still works for this run when the file cannot be kept; the next one downloads again
    with contextlib.suppress(OSError):
        fs.replace(path, raw)
    return loaded


def vibe_encoding(source: vibe.VibeSource, model: str) -> str | None:
    """The encoding the Vibe keeps, else one paid for before with the same settings; None when it needs encoding."""
    stored = vibe.stored_encoding(source, model)
    if stored is not None:
        return stored
    try:
        cached = _vibe_path(source, model).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        # Missing or unreadable, so the image is encoded again
        return None
    return cached if cached and vibe.is_encoding(cached) else None


def remember_vibe(source: vibe.VibeSource, model: str, encoding: str) -> None:
    # Lookups outside the generation lock must never read half a file
    fs.replace(_vibe_path(source, model), encoding)


def _vibe_path(source: vibe.VibeSource, model: str) -> Path:
    key = vibe.encoding_key(source, model)
    return config.data_dir() / "vibe-cache" / f"{hashlib.sha256(key.encode()).hexdigest()}.txt"
