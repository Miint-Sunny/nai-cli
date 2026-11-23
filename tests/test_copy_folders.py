"""Folders given to the commands that copy images: the images inside them, with -r those inside their folders too,
and each copy keeping its place under -o."""

import json
from pathlib import Path

import pytest
from helpers import run, run_batch
from PIL import Image
from samples import nai_image

from nai_cli import cli

COMMANDS = [["strip"], ["redact", "--region", "rect:0,0,0.5,0.5"], ["watermark", "--text", "me"]]
# In name order, the text file and the GIF left out
TOP = ["a.png", "b.png", "e.png"]
ALL = [*TOP, "sub/c.png", "sub/deep/d.png"]


def _tree(root: Path) -> Path:
    """A folder holding images named a.PNG, b.png, e.jpg, c.png and d.png, the last two in folders inside it, and
    a text file and a GIF; made in an order other than their names."""
    folder = root / "in"
    (folder / "sub" / "deep").mkdir(parents=True)
    for name in ("e.jpg", "sub/deep/d.png", "a.PNG", "notes.txt", "sub/c.png", "b.png", "x.gif"):
        if Path(name).suffix.lower() in (".png", ".jpg"):
            Image.new("RGB", (16, 16), "gray").save(folder / name)
        else:
            (folder / name).write_text("x", encoding="utf-8")
    return folder


def _names(out: dict, folder: Path) -> list[str]:
    return [str(Path(f["file"]).relative_to(folder)) for f in out["files"]]


@pytest.mark.parametrize("command", COMMANDS)
def test_a_folder_means_the_images_in_it_in_name_order(capsys, tmp_path, command):
    folder, out = _tree(tmp_path), tmp_path / "out"
    code, first = run(capsys, *command, str(folder), "-o", str(out))
    assert code == 0, first
    assert _names(first, out) == TOP
    assert [Path(f["source"]).name for f in first["files"]] == ["a.PNG", "b.png", "e.jpg"]


@pytest.mark.parametrize("command", COMMANDS)
def test_recursive_reaches_the_folders_inside_and_each_copy_keeps_its_place(capsys, tmp_path, command):
    folder, out = _tree(tmp_path), tmp_path / "out"
    code, copies = run(capsys, *command, str(folder), "-r", "-o", str(out))
    assert code == 0, copies
    assert _names(copies, out) == ALL
    assert all((out / name).is_file() for name in ALL)


@pytest.mark.parametrize("command", COMMANDS)
def test_without_o_each_image_of_a_folder_is_filed_as_its_copy(capsys, tmp_path, command):
    folder = _tree(tmp_path)
    code, copies = run(capsys, *command, str(folder), "-r")
    label = {"strip": "clean", "redact": "redacted", "watermark": "watermarked"}[command[0]]
    assert code == 0 and sorted(Path(f["file"]).name for f in copies["files"]) == [
        f"{stem}_{label}.png" for stem in "abcde"
    ]


def _planned(capsys, tmp_path, *argv) -> dict:
    """A copy command has no --dry-run of its own: a batch run as a dry run runs it as its dry run."""
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(list(argv)) + "\n", encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert code == 0 and lines[0]["type"] == "job", lines
    return lines[0]


def test_a_dry_run_names_what_the_real_run_writes(capsys, tmp_path):
    folder, out = _tree(tmp_path), tmp_path / "out"
    planned = _planned(capsys, tmp_path, "strip", str(folder), "-r", "-o", str(out))
    assert planned["dry_run"] is True and not out.exists()
    code, written = run(capsys, "strip", str(folder), "-r", "-o", str(out))
    assert code == 0 and planned["files"] == written["files"]
    # Names the first run took get the next number, as a dry run of the second says
    again = _planned(capsys, tmp_path, "strip", str(folder), "-r", "-o", str(out))
    assert code == 0 and _names(again, out) == [name[:-4] + "-1.png" for name in ALL]


def test_a_folder_with_a_file_name_is_bad_args(capsys, tmp_path):
    folder = _tree(tmp_path)
    code, out = run(capsys, "strip", str(folder), "-o", str(tmp_path / "one.png"))
    assert (code, out["code"]) == (1, "bad_args") and "give a folder" in out["message"]
    assert not list(tmp_path.rglob("one.png"))


def test_when_no_image_is_given_it_is_bad_args(capsys, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("x", encoding="utf-8")
    for given in (empty, notes):
        code, out = run(capsys, "strip", str(given), "-o", str(tmp_path / "out"))
        assert (code, out["code"]) == (1, "bad_args"), out


def test_meta_reads_png_and_webp_from_a_folder_as_before(capsys, tmp_path):
    folder = tmp_path / "in"
    (folder / "sub").mkdir(parents=True)
    nai_image(folder / "b.PNG", w=64, h=64, seed=2)
    nai_image(folder / "a.png", w=64, h=64)
    Image.new("RGB", (16, 16)).save(folder / "sub" / "c.webp")
    Image.new("RGB", (16, 16)).save(folder / "d.jpg")
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    for options, found in (([], ["a.png", "b.PNG"]), (["-r"], ["a.png", "b.PNG", "sub/c.webp"])):
        code = cli.main(["meta", str(folder), *options])
        lines = capsys.readouterr().out.splitlines()
        read = [Path(json.loads(line)["image"]).relative_to(folder).as_posix() for line in lines[:-1]]
        assert code == 0 and read == found
    empty = tmp_path / "empty"
    empty.mkdir()
    code = cli.main(["meta", str(empty)])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["meta"]["images"] == 0
