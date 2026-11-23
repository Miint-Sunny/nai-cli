import json
from pathlib import Path

import httpx
import pytest
from helpers import FakeClient

from nai_cli import api, caches, config

# Settings from real images for round-trip tests; not tracked, skipped when absent
PRIVATE = Path(__file__).resolve().parent / "private"
# Kept so tests may replace caches.load_tokenizer
_LOAD = caches.load_tokenizer


def _rows(name: str) -> list[dict]:
    path = PRIVATE / name
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def tagged() -> list[dict]:
    return _rows("tagged.json")


def untagged() -> list[dict]:
    return _rows("untagged.jsonl")


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Keep config, ledger, cache and images in a temporary directory."""
    monkeypatch.delenv(config.TOKEN_ENV, raising=False)
    for name, folder in (
        # Images default to Pictures under the home folder on macOS
        ("HOME", "home"),
        ("USERPROFILE", "home"),
        ("APPDATA", "config"),
        ("XDG_CONFIG_HOME", "config"),
        ("LOCALAPPDATA", "local"),
        ("XDG_DATA_HOME", "local"),
    ):
        monkeypatch.setenv(name, str(tmp_path / folder))
    monkeypatch.setattr(api, "fetch_tokenizer", _offline)
    _LOAD.cache_clear()
    cfg = config.default_path()
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        f'novelai_token = ""\n[output]\ndir = "{(tmp_path / "out").as_posix()}"\n',
        encoding="utf-8",
    )
    yield tmp_path
    _LOAD.cache_clear()


def _offline(name: str, version: int) -> bytes:
    # Tests never contact NovelAI, so token counts fall back to estimates unless a test supplies a tokenizer file
    raise httpx.ConnectError(f"offline: {name} v{version}")


@pytest.fixture
def with_token(monkeypatch, isolated_dirs):
    path = config.default_path()
    path.write_text(
        path.read_text(encoding="utf-8").replace('novelai_token = ""', 'novelai_token = "pst-x"'), encoding="utf-8"
    )
    monkeypatch.setattr(api, "Client", FakeClient)
