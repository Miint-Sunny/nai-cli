"""The two copies of the settings NovelAI writes to an image, its text and its alpha channel: which one is read,
and whether they agree."""

import json
from pathlib import Path

import pytest
from helpers import run
from PIL import Image
from samples import in_low_bits, latin1_description, layered_png, settings, stealth_payload, with_comment_field

from nai_cli import cli, meta

PROMPT = "1girl, smile"
CHANGED = "1girl, changed by another tool"


def _changed(**changes) -> dict[str, str]:
    """NovelAI's settings as another tool leaves the text: changed, and no longer signed."""
    return with_comment_field(settings(), signed_hash=None, **changes)


def _meta(capsys, path, *options) -> dict:
    code, out = run(capsys, "meta", str(path), *options)
    assert code == 0, out
    return out


def test_agreeing_copies_are_read_from_the_text(tmp_path):
    path = layered_png(tmp_path / "a.png", settings(), settings())
    layers = meta.layers_of(path.read_bytes(), path.name)
    assert list(layers.found) == ["png_text", "stealth"] and layers.signed == ["png_text", "stealth"]
    assert layers.used == "png_text" and layers.differences == []
    assert meta.read_raw(path) == settings()


def test_text_changed_after_novelai_signed_gives_way_to_the_stealth_data(capsys, tmp_path):
    path = layered_png(tmp_path / "a.png", _changed(prompt=CHANGED), settings())
    out = _meta(capsys, path)
    assert out["layers"] == {
        "found": ["png_text", "stealth"],
        "signed": ["stealth"],
        "used": "stealth",
        "differences": ["Comment.prompt"],
    }
    assert out["prompt"] == PROMPT
    [warning] = out["warnings"]
    assert "Comment.prompt" in warning and "stealth" in warning and "signed" in warning


def test_two_signed_copies_that_disagree_are_read_from_the_text(capsys, tmp_path):
    path = layered_png(tmp_path / "a.png", with_comment_field(settings(), prompt=CHANGED), settings())
    out = _meta(capsys, path)
    assert out["layers"]["signed"] == ["png_text", "stealth"]
    assert out["layers"]["used"] == "png_text" and out["layers"]["differences"] == ["Comment.prompt"]
    assert out["prompt"] == CHANGED
    assert any("Comment.prompt" in w and "png_text" in w for w in out["warnings"])


def test_the_tool_and_model_names_are_not_differences(capsys, tmp_path):
    text = {**settings(), "Software": "Other Client", "Source": "NAI-Diffusion-v5-Full"}
    out = _meta(capsys, layered_png(tmp_path / "a.png", text, settings()))
    assert out["layers"]["differences"] == [] and out["layers"]["used"] == "png_text"
    assert out["warnings"] == [] and out["software"] == "Other Client"


def test_fields_only_the_text_keeps_are_not_differences(tmp_path):
    text = with_comment_field(settings(), reference_image_multiple=["ENCODING"], reference_strength_multiple=[0.6])
    path = layered_png(tmp_path / "a.png", text, settings())
    layers = meta.layers_of(path.read_bytes(), path.name)
    assert layers.differences == [] and layers.used == "png_text"
    # The Vibes are kept, as a rerun needs them
    assert meta.read_raw(path) == text


@pytest.mark.parametrize(("text", "stealth", "found"), [(None, "settings", "stealth"), ("settings", None, "png_text")])
def test_one_copy_alone_is_read(tmp_path, text, stealth, found):
    path = layered_png(tmp_path / "a.png", settings() if text else None, settings() if stealth else None)
    layers = meta.layers_of(path.read_bytes(), path.name)
    assert list(layers.found) == [found] and layers.used == found and layers.differences is None


def test_nuls_before_an_exif_description_are_not_a_difference(capsys, tmp_path):
    text = settings(prompt="1girl, 猫")
    exif = Image.Exif()
    exif[0x010D] = text["Title"]
    exif[0x010E] = "\0\0\0\0" + text["Description"]
    exif[0x0131] = text["Source"]
    exif.get_ifd(0x8769)[0x9286] = b"ASCII\0\0\0" + json.dumps(text).encode()
    path = tmp_path / "a.webp"
    in_low_bits(Image.new("RGBA", (96, 96), (10, 20, 30, 255)), stealth_payload(settings(prompt="1girl, 猫"))).save(
        path, "WEBP", lossless=True, exact=True, exif=exif.tobytes()
    )
    out = _meta(capsys, path)
    assert out["layers"]["used"] == "exif" and out["layers"]["differences"] == [] and out["warnings"] == []
    assert out["prompt"] == "1girl, 猫"


def test_a_no_break_space_written_as_one_byte_is_not_a_difference(tmp_path):
    path = layered_png(tmp_path / "a.png", settings(prompt="猫, a\u00a0b"), settings(prompt="猫, a\u00a0b"))
    layers = meta.layers_of(latin1_description(path).read_bytes(), path.name)
    assert layers.differences == [] and layers.used == "png_text"


def test_layer_reads_one_copy_only(capsys, tmp_path):
    code, out = run(capsys, "meta", str(layered_png(tmp_path / "text.png", settings())), "--layer", "stealth")
    assert code == 1 and out["code"] == "no_metadata"
    code, out = run(capsys, "meta", str(layered_png(tmp_path / "stealth.png", None, settings())), "--layer", "text")
    assert code == 1 and out["code"] == "no_metadata"
    both = layered_png(tmp_path / "both.png", _changed(prompt=CHANGED), settings())
    out = _meta(capsys, both, "--layer", "stealth")
    assert out["layers"] == {
        "found": ["png_text", "stealth"],
        "signed": ["stealth"],
        "used": "stealth",
        "differences": ["Comment.prompt"],
    }
    assert out["prompt"] == PROMPT
    out = _meta(capsys, both, "--layer", "text")
    assert out["layers"]["used"] == "png_text" and out["prompt"] == CHANGED
    assert any("--layer" in w and "png_text" in w for w in out["warnings"])


def test_the_summary_counts_images_whose_copies_disagree(capsys, tmp_path):
    folder = tmp_path / "pics"
    layered_png(folder / "a.png", settings(), settings())
    layered_png(folder / "b.png", _changed(prompt=CHANGED), settings())
    layered_png(folder / "c.png", None, settings())
    code = cli.main(["meta", str(folder)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and lines[-1]["meta"]["layers_differ"] == 1
    assert [line["layers"]["used"] for line in lines[:-1]] == ["png_text", "stealth", "stealth"]


def test_reruns_and_kept_copies_follow_the_stealth_data(capsys, tmp_path):
    source = layered_png(tmp_path / "a.png", _changed(prompt=CHANGED), settings())
    # Without its Title, which the alpha channel does not keep
    hidden = {k: v for k, v in settings().items() if k != "Title"}
    assert meta.raw_of(source.read_bytes(), source.name) == hidden
    code, out = run(capsys, "watermark", str(source), "--text", "me", "--keep-metadata")
    assert code == 0, out
    kept = meta.png_text_chunks(Path(out["files"][0]["file"]).read_bytes())
    assert kept["Comment"] == settings()["Comment"] and kept["Description"] == PROMPT
