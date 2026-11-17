"""Pixel snapping: the grid of scaled pixel art found again, its colours, and the tool run locally."""

from pathlib import Path

import numpy as np
import pytest
from helpers import run
from PIL import Image
from samples import pixel_art

from nai_cli import errors, pixel_snap


@pytest.mark.parametrize(("rows", "cols", "pitch"), [(24, 20, 8), (18, 30, 6.5)])
def test_the_grid_and_colours_come_back(rows, cols, pitch):
    art, rgba = pixel_art(rows, cols, pitch)
    out = pixel_snap.snap(rgba)
    assert out.rgb.shape == art.shape and out.alpha is None
    assert np.abs(out.rgb.astype(int) - art).max() <= 12
    assert out.palette_size == len(np.unique(art.reshape(-1, 3), axis=0))
    assert all(abs(p - pitch) < 0.1 for p in out.pitch)


def test_palette_size_limits_colours():
    _, rgba = pixel_art(20, 20, 8, colors=40)
    assert pixel_snap.snap(rgba, colors=16).palette_size <= 16
    assert pixel_snap.snap(rgba, colors=0).palette_size is None


def test_transparent_cells_keep_their_alpha():
    _, rgba = pixel_art(16, 16, 8, colors=24)
    rgba[:64, :, 3] = 0
    out = pixel_snap.snap(rgba)
    assert out.alpha is not None and set(np.unique(out.alpha)) == {0, 255}
    assert (out.alpha[0] == 0).all() and (out.alpha[-1] == 255).all()


def test_images_without_a_grid_are_refused():
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., 3] = 255
    with pytest.raises(errors.NaiError):
        pixel_snap.snap(rgba)
    with pytest.raises(errors.NaiError):
        pixel_snap.snap(np.zeros((64, 64, 4), np.uint8))


def test_tool_runs_locally_and_saves(capsys, isolated_dirs, tmp_path):
    _, rgba = pixel_art(16, 12, 8)
    source = tmp_path / "art.png"
    Image.fromarray(rgba, "RGBA").save(source)
    code, out = run(capsys, "tool", "pixel-snap", str(source), "-o", str(tmp_path / "snap.png"))
    assert code == 0 and out["cells"] == "12x16" and out["cost"]["estimate"] == 0
    with Image.open(out["images"][0]) as img:
        assert img.size == (12, 16)
    code, out = run(capsys, "tool", "pixel-snap", str(source), "--upscale", "-o", str(tmp_path / "big.png"))
    with Image.open(out["images"][0]) as img:
        assert img.size == (96, 128)
    assert Path(out["images"][0]).name == "big.png"


def test_bad_colour_counts_are_refused(capsys, tmp_path):
    source = tmp_path / "art.png"
    Image.new("RGB", (8, 8)).save(source)
    for count in ("300", "0", "-4"):
        code, out = run(capsys, "tool", "pixel-snap", str(source), "--colors", count)
        assert code == 1 and out["code"] == "bad_args", count


def test_a_palette_below_the_slider_can_be_typed(capsys, tmp_path):
    source = tmp_path / "art.png"
    Image.fromarray(np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)).save(source)
    code, out = run(capsys, "tool", "pixel-snap", str(source), "--colors", "2")
    assert code == 0 and out["colors"] <= 2


def test_snap_is_free_with_no_account_in_its_cost_and_refuses_animations(capsys, tmp_path):
    still = tmp_path / "still.png"
    Image.fromarray(pixel_art(16, 12, 8)[1], "RGBA").save(still)
    code, out = run(capsys, "tool", "pixel-snap", str(still), "--dry-run")
    assert code == 0 and out["cost"] == {"estimate": 0} and out["cells"] == "12x16"
    # A dry run snaps too, so an image the real run refuses is refused there
    flat = tmp_path / "flat.png"
    Image.new("RGB", (8, 8), "red").save(flat)
    for argv in ([], ["--dry-run"]):
        code, out = run(capsys, "tool", "pixel-snap", str(flat), *argv)
        assert (code, out["code"]) == (1, "bad_args") and "grid" in out["message"]
    moving = tmp_path / "moving.png"
    frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
    frames[0].save(moving, save_all=True, append_images=frames[1:])
    code, out = run(capsys, "tool", "pixel-snap", str(moving))
    assert (code, out["code"]) == (1, "bad_args") and "animated" in out["message"]
