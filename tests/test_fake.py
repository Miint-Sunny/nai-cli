"""nai fake: copies whose settings are made up, from a preset, a template, a text or single fields, and written into
both layers as nai reword writes them."""

import json
import random
import re
from pathlib import Path

import pytest
from helpers import run, run_batch
from PIL import Image, PngImagePlugin
from samples import layers_of, settings

from nai_cli import config, errors, fake, request

SPACES = " " * 512
TAUNTS = ", ".join(["杂鱼~♥"] * 64)
FIELDS = ["Title", "Description", "Software", "Source", "Generation time", "Comment"]


def _plain(path: Path, size=(128, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)
    return path


def _folder() -> Path:
    folder = config.config_dir() / "fakes"
    folder.mkdir(parents=True)
    return folder


def _comment(copy) -> dict:
    return json.loads(layers_of(copy)[0]["Comment"])


@pytest.mark.parametrize(("spec", "text"), [("1", SPACES), ("2", TAUNTS), ("made up", "made up")])
def test_a_preset_or_a_text_fills_every_field(capsys, tmp_path, spec, text):
    source = _plain(tmp_path / "a.png")
    out = tmp_path / "out"
    code, made = run(capsys, "fake", str(source), "--preset", spec, "-o", str(out))
    assert code == 0 and made == {
        "ok": True,
        "files": [{"source": str(source), "file": str(out / "a.png")}],
        "warnings": [],
    }
    text_fields, hidden = layers_of(out / "a.png")
    assert text_fields == dict.fromkeys(FIELDS, text)
    assert hidden == {key: value for key, value in text_fields.items() if key != "Title"}


def test_a_preset_of_the_user_replaces_the_built_in_of_the_same_name(capsys, tmp_path):
    preset = {"Title": "A title", "Comment": {"prompt": "mine", "seed": 7, "width": 1, "height": 1, "steps": 12}}
    (_folder() / "1.json").write_text(json.dumps(preset), encoding="utf-8")
    source = _plain(tmp_path / "a.png")
    out = tmp_path / "out"
    code, made = run(capsys, "fake", str(source), "--preset", "1", "-o", str(out))
    text_fields, hidden = layers_of(out / "a.png")
    comment = json.loads(text_fields["Comment"])
    assert code == 0, made
    assert (
        text_fields["Title"] == "A title"
        and text_fields["Description"] == "mine"
        and text_fields["Software"] == "NovelAI"
    )
    assert comment["prompt"] == "mine" and comment["seed"] == 7 and comment["steps"] == 12
    # The image's size, not the preset's
    assert (comment["width"], comment["height"]) == (128, 64)
    assert 2 <= float(text_fields["Generation time"]) < 9 and "signed_hash" not in comment
    assert hidden == {key: value for key, value in text_fields.items() if key != "Title"}


def test_a_template_is_a_json_file_named_with_an_at_sign(capsys, tmp_path):
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "_note": "left out",
                "Source": "NovelAI Diffusion V4.5 4BDE2A90",
                "Comment": json.dumps({"prompt": "from a template", "seed": 3}),
            }
        ),
        encoding="utf-8",
    )
    source = _plain(tmp_path / "a.png")
    out = tmp_path / "out"
    code, made = run(capsys, "fake", str(source), "--preset", f"@{template}", "--set", "seed=9", "-o", str(out))
    text_fields, _ = layers_of(out / "a.png")
    assert code == 0, made
    assert text_fields["Source"] == "NovelAI Diffusion V4.5 4BDE2A90" and "_note" not in text_fields
    assert json.loads(text_fields["Comment"]) == {"prompt": "from a template", "seed": 9, "width": 128, "height": 64}
    code, made = run(capsys, "fake", str(source), "--preset", "@/nowhere.json", "-o", str(out))
    assert (code, made["code"]) == (1, "bad_args")


def test_set_alone_keeps_what_the_image_had(capsys, tmp_path):
    source = tmp_path / "in" / "a.png"
    text = settings(prompt="1girl, smile", steps=30)
    source.parent.mkdir(parents=True)
    info = PngImagePlugin.PngInfo()
    for key, value in text.items():
        info.add_text(key, value)
    Image.new("RGB", (128, 64)).save(source, pnginfo=info)
    out = tmp_path / "out"
    code, made = run(capsys, "fake", str(source), "--set", "seed=7", "--set", "uc=very lowres", "-o", str(out))
    kept, hidden = layers_of(out / "a.png")
    comment = json.loads(kept["Comment"])
    assert code == 0, made
    assert comment["seed"] == 7 and comment["uc"] == "very lowres"
    assert comment["prompt"] == "1girl, smile" and comment["steps"] == 30 and comment["scale"] == 4.0
    assert "signed_hash" not in comment and kept["Description"] == "1girl, smile" and kept["Source"] == text["Source"]
    assert hidden == {key: value for key, value in kept.items() if key != "Title"}
    # The source is never touched
    assert "signed_hash" in json.loads(layers_of(source)[0]["Comment"])


def test_set_changes_a_preset(capsys, tmp_path):
    source = _plain(tmp_path / "a.png")
    out = tmp_path / "out"
    code, made = run(
        capsys, "fake", str(source), "--preset", "1", "--set", "seed=7", "--set", "prompt=cat", "-o", str(out)
    )
    text_fields, _ = layers_of(out / "a.png")
    assert code == 0, made
    assert json.loads(text_fields["Comment"]) == {"prompt": "cat", "uc": SPACES, "seed": 7, "width": 128, "height": 64}
    assert text_fields["Description"] == SPACES


@pytest.fixture
def drawn_seeds():
    """The seeds two images draw once the random source is reset, the same after every reset."""
    before = random.getstate()
    random.seed(2026)
    seeds = [request.random_seed(), request.random_seed()]
    yield seeds
    random.setstate(before)


def test_a_null_seed_is_drawn_for_each_image(capsys, tmp_path, drawn_seeds):
    (_folder() / "seeds.json").write_text(
        json.dumps({"Generation time": "1", "Comment": {"prompt": "x", "seed": None}}), encoding="utf-8"
    )
    first, second = _plain(tmp_path / "a.png"), _plain(tmp_path / "b.png")
    out = tmp_path / "out"
    for _ in range(2):
        random.seed(2026)
        code, made = run(capsys, "fake", str(first), str(second), "--preset", "seeds", "-o", str(out))
        assert code == 0, made
    seeds = [_comment(entry["file"])["seed"] for entry in made["files"]]
    assert seeds == drawn_seeds and seeds[0] != seeds[1]


def test_the_size_is_the_images_whatever_is_set(capsys, tmp_path):
    source = _plain(tmp_path / "a.png")
    out = tmp_path / "out"
    argv = ["--set", "width=999", "--set", "height=999", "--set", "seed=1"]
    code, made = run(capsys, "fake", str(source), *argv, "-o", str(out))
    assert code == 0, made
    comment = _comment(out / "a.png")
    assert (comment["width"], comment["height"]) == (128, 64) and comment["seed"] == 1


def test_neither_a_preset_nor_a_field_is_bad_args(capsys, tmp_path):
    source = _plain(tmp_path / "a.png")
    out = tmp_path / "out"
    for argv in ([], ["--set", "no equals sign"]):
        code, refused = run(capsys, "fake", str(source), *argv, "-o", str(out))
        assert (code, refused["code"]) == (1, "bad_args"), refused
    assert not out.exists()


def test_copies_are_named_fake_and_take_the_other_options(capsys, tmp_path):
    source = _plain(tmp_path / "in" / "a.png")
    _plain(tmp_path / "in" / "sub" / "b.png")
    out = tmp_path / "out"
    code, filed = run(capsys, "fake", str(source), "--preset", "1")
    assert code == 0 and Path(filed["files"][0]["file"]).name == "a_fake.png", filed
    code, made = run(capsys, "fake", str(tmp_path / "in"), "-r", "-o", str(out), "--preset", "1", "--rename")
    written = [Path(entry["file"]) for entry in made["files"]]
    assert code == 0 and [path.parent for path in written] == [out, out / "sub"], made
    assert all(re.fullmatch(r"[0-9]{8}-[0-9]{4}\.png", path.name) for path in written), made
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(["fake", str(source), "--preset", "1", "-o", str(tmp_path / "dry")]), encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert code == 0 and lines[0]["dry_run"] is True and not (tmp_path / "dry").exists()


def test_set_values_are_json_or_text():
    assert fake.parse_set("seed=7") == ("seed", 7)
    assert fake.parse_set("uc=lowres") == ("uc", "lowres")
    assert fake.parse_set('a={"b": 1}') == ("a", {"b": 1})
    assert fake.parse_set("flag=true") == ("flag", True)
    for item in ("novalue", "=1"):
        with pytest.raises(errors.NaiError):
            fake.parse_set(item)


def test_a_preset_file_holds_its_comment_as_text_or_as_an_object():
    read = fake.read_preset('{"_note": 1, "Title": "t", "Comment": "{\\"prompt\\": \\"p\\"}"}')
    assert read == {"Title": "t", "Comment": {"prompt": "p"}}
    assert fake.read_preset('{"Comment": "plain text"}') == {"Comment": "plain text"}
    for text in ("[1, 2]", "not json"):
        with pytest.raises(errors.NaiError):
            fake.read_preset(text)
