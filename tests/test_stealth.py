"""Stealth data in the lowest bits of an image's channels: read from the alpha channel or the colours, and cleared
from the colours of a copy, where they are the picture itself."""

import math

import numpy as np
import pytest
from helpers import run
from PIL import Image
from samples import in_low_bits, settings, stealth_payload

from nai_cli import decode, images, meta, stealth

SIZE = (64, 64)


def _pattern(size=SIZE) -> Image.Image:
    """A picture whose pixels all differ, so that a change to any of them shows."""
    return Image.fromarray((np.arange(size[0] * size[1] * 3) * 7 % 256).reshape(size[1], size[0], 3).astype(np.uint8))


def _hidden(magic: bytes, seed: int = 5) -> Image.Image:
    """The pattern with NovelAI's settings in the lowest bits of its colours."""
    return in_low_bits(_pattern(), stealth_payload(settings(seed=seed), magic), "rgb")


def _taken(block: stealth.Block, size=SIZE) -> np.ndarray:
    """Which of an image's lowest bits a block in the colours takes."""
    where = np.arange(block.bits)
    pixels = where // 3
    taken = np.zeros((size[1], size[0], 3), dtype=bool)
    taken[pixels % size[1], pixels // size[1], where % 3] = True
    return taken


@pytest.mark.parametrize("magic", [b"stealth_rgbinfo", b"stealth_rgbcomp"])
def test_settings_hidden_in_the_colours_are_read(capsys, tmp_path, magic):
    path = tmp_path / "a.png"
    _hidden(magic).save(path)
    code, out = run(capsys, "meta", str(path))
    assert code == 0, out
    assert out["seed"] == 5 and out["prompt"] == "1girl, smile"


def test_the_alpha_channel_is_read_before_the_colours():
    in_alpha = in_low_bits(_pattern().convert("RGBA"), stealth_payload(settings(seed=9)), "alpha")
    both = in_low_bits(in_alpha, stealth_payload(settings(seed=1), b"stealth_rgbcomp"), "rgb")
    assert stealth.find(both, decode.STEALTH_MAX_PIXELS).channel is stealth.Channel.ALPHA
    assert meta.stealth_text(both)["Comment"] == settings(seed=9)["Comment"]


def test_large_images_are_skipped_as_the_official_reader_does(monkeypatch):
    hidden = _hidden(b"stealth_rgbcomp")
    assert stealth.find(hidden, 100) is None and stealth.find(hidden, math.inf) is not None
    monkeypatch.setattr(decode, "STEALTH_MAX_PIXELS", 100)
    assert meta.stealth_text(hidden) == {}


def test_a_source_alpha_is_cleaned_only_behind_novelai_magic():
    """The official client cleans an image's alpha when it starts with stealth_pngcomp, and only then."""
    plain = Image.new("RGBA", (96, 96), (10, 20, 30, 254))
    assert not stealth.begins_in_alpha(plain, decode.STEALTH_MAX_PIXELS)
    assert images.clear_stealth_alpha(plain).getchannel("A").getextrema() == (254, 254)
    other_tool = in_low_bits(plain, stealth_payload(settings(), b"stealth_pnginfo"), "alpha")
    assert not stealth.begins_in_alpha(other_tool, decode.STEALTH_MAX_PIXELS)
    novelai = in_low_bits(plain, stealth_payload(settings()), "alpha")
    assert stealth.begins_in_alpha(novelai, decode.STEALTH_MAX_PIXELS)
    assert images.clear_stealth_alpha(novelai).getchannel("A").getextrema() == (255, 255)


def test_clearing_touches_only_the_bits_the_block_takes():
    hidden = _hidden(b"stealth_rgbinfo")
    block = stealth.find(hidden, math.inf)
    cleared = np.asarray(stealth.clear(hidden, block)).astype(int)
    before = np.asarray(_pattern()).astype(int)
    taken = _taken(block)
    # The bits are cleared, so an odd value loses one and an even one is the original
    assert np.array_equal(cleared[taken], before[taken] & ~1) and (cleared[~taken] == before[~taken]).all()


def test_a_strip_copy_drops_stealth_data_hidden_in_the_colours(capsys, tmp_path):
    source = tmp_path / "a.png"
    hidden = _hidden(b"stealth_rgbcomp")
    hidden.save(source)
    block = stealth.find(hidden, math.inf)
    code, out = run(capsys, "strip", str(source))
    assert code == 0, out
    with Image.open(out["files"][0]["file"]) as copy:
        assert copy.mode == "RGB" and stealth.find(copy, math.inf) is None
        change = np.abs(np.asarray(copy, dtype=int) - np.asarray(hidden, dtype=int))
    taken = _taken(block)
    assert change.max() == 1 and change[taken].any() and not change[~taken].any()


@pytest.mark.parametrize("argv", [["redact", "--region", "rect:0.5,0.5,1,1"], ["watermark", "--text", "me"]])
def test_edited_copies_drop_it_too(capsys, tmp_path, argv):
    source = tmp_path / "a.png"
    block = stealth.find(_hidden(b"stealth_rgbcomp"), math.inf)
    _hidden(b"stealth_rgbcomp").save(source)
    code, out = run(capsys, *argv, str(source))
    assert code == 0, out
    with Image.open(out["files"][0]["file"]) as copy:
        assert stealth.find(copy, math.inf) is None
        # The edits stay away from the top left corner, where the block lies, so its bits were there to clear
        assert not (np.asarray(copy.convert("RGB")) & 1)[_taken(block)].any()


@pytest.mark.parametrize("length", [0, 7, SIZE[0] * SIZE[1] * 3, SIZE[0] * SIZE[1] * 3 - 152 + 8])
def test_a_magic_with_a_length_that_does_not_fit_is_not_stealth_data(capsys, tmp_path, length):
    path = tmp_path / "a.png"
    in_low_bits(_pattern(), b"stealth_rgbcomp" + length.to_bytes(4, "big") + b"\0" * 64, "rgb").save(path)
    assert stealth.find(Image.open(path), math.inf) is None
    code, out = run(capsys, "meta", str(path))
    assert (code, out["code"]) == (1, "no_metadata")
