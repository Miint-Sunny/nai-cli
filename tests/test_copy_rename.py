"""Copies named --rename: by the day of the run and a number that counts on in each folder from the highest one
its files of that day hold."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from helpers import run, run_batch
from PIL import Image

from nai_cli.commands.context import Context

DAY = datetime(2026, 11, 23, 14, 15, 16)
COMMANDS = [["strip"], ["redact", "--region", "rect:0,0,0.5,0.5"], ["watermark", "--text", "me"]]
# NovelAI's download name: the start of the prompt, a space, s- and the seed
DOWNLOADED = "1girl, solo, smile s-1234.png"


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """The run takes its day from the context's clock."""
    create = Context.create

    def with_the_clock(*args, **options) -> Context:
        ctx = create(*args, **options)
        ctx.now = lambda: DAY
        return ctx

    monkeypatch.setattr(Context, "create", with_the_clock)


def _pic(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "gray").save(path)
    return path


def _names(out: dict) -> list[str]:
    return [Path(f["file"]).name for f in out["files"]]


def _taken(folder: Path, *names: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"taken")
    return folder


def test_numbers_count_on_from_the_highest_of_the_day_whatever_the_suffix(capsys, tmp_path):
    out = _taken(
        tmp_path / "out", "20261123-0003.webp", "20261123-0002.png", "20261122-0100.png", "20261123-7.png", "x-0009.png"
    )
    code, copies = run(capsys, "strip", str(_pic(tmp_path / "a.png")), "--rename", "-o", str(out))
    assert code == 0 and _names(copies) == ["20261123-0004.png"]
    assert (out / "20261123-0002.png").read_bytes() == b"taken"


def test_numbers_of_other_days_do_not_count(capsys, tmp_path):
    out = _taken(tmp_path / "out", "20261122-0100.png", "20261124-0001.png")
    code, copies = run(capsys, "strip", str(_pic(tmp_path / "a.png")), "--rename", "-o", str(out))
    assert code == 0 and _names(copies) == ["20261123-0001.png"]


@pytest.mark.parametrize("command", COMMANDS)
def test_copies_of_one_run_are_numbered_in_order(capsys, tmp_path, command):
    out = tmp_path / "out"
    code, copies = run(
        capsys, *command, str(_pic(tmp_path / "b.png")), str(_pic(tmp_path / "a.png")), "--rename", "-o", str(out)
    )
    assert code == 0 and _names(copies) == ["20261123-0001.png", "20261123-0002.png"]
    assert all((out / name).is_file() for name in _names(copies))


def test_each_folder_numbers_on_its_own(capsys, tmp_path):
    folder = tmp_path / "in"
    for name in ("a.png", "sub/b.png", "sub/deep/c.png"):
        _pic(folder / name)
    out = _taken(tmp_path / "out", "20261123-0005.png")
    _taken(out / "sub", "20261123-0009.jpg")
    code, copies = run(capsys, "strip", str(folder), "-r", "-o", str(out), "--rename")
    assert code == 0, copies
    assert [str(Path(f["file"]).relative_to(out)) for f in copies["files"]] == [
        "20261123-0006.png",
        "sub/20261123-0010.png",
        "sub/deep/20261123-0001.png",
    ]


def test_without_o_renamed_copies_are_filed_where_copies_go(capsys, tmp_path):
    code, copies = run(capsys, "strip", str(_pic(tmp_path / "b.png")), str(_pic(tmp_path / "a.png")), "--rename")
    assert code == 0, copies
    folders = {Path(f["file"]).parent for f in copies["files"]}
    assert len(folders) == 1 and all(Path(f["file"]).is_file() for f in copies["files"])
    assert _names(copies) == ["20261123-0001.png", "20261123-0002.png"]


@pytest.mark.parametrize(
    ("source", "options", "warns"),
    [
        (DOWNLOADED, [], True),
        (DOWNLOADED, ["-o", "OUT"], True),
        (DOWNLOADED, ["--rename"], False),
        (DOWNLOADED, ["-o", "OUT/one.png"], False),
        ("pic.png", [], False),
        ("s-1234.png", [], False),
    ],
)
def test_a_copy_keeps_its_sources_download_name_unless_told_otherwise(capsys, tmp_path, source, options, warns):
    options = [option.replace("OUT", str(tmp_path / "out")) for option in options]
    source = _pic(tmp_path / "in" / source)
    code, copies = run(capsys, "strip", str(source), *options)
    assert code == 0, copies
    assert any("--rename" in w and "seed" in w for w in copies.get("warnings", [])) is warns, copies


def test_a_dry_run_names_what_the_real_run_writes(capsys, tmp_path):
    out = _taken(tmp_path / "out", "20261123-0001.png")
    first, second = _pic(tmp_path / "a.png"), _pic(tmp_path / "b.png")
    jobs = tmp_path / "jobs.jsonl"
    argv = ["strip", str(first), str(second), "--rename", "-o", str(out)]
    jobs.write_text(json.dumps(argv) + "\n", encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert code == 0 and _names(lines[0]) == ["20261123-0002.png", "20261123-0003.png"]
    code, real = run(capsys, *argv)
    assert code == 0 and lines[0]["files"] == real["files"]


def test_rename_and_a_file_name_are_bad_args(capsys, tmp_path):
    code, out = run(capsys, "strip", str(_pic(tmp_path / "a.png")), "--rename", "-o", str(tmp_path / "one.png"))
    assert (code, out["code"]) == (1, "bad_args") and "folder" in out["message"]
    assert not list(tmp_path.rglob("one.png"))
