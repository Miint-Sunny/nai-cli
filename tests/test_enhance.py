import pytest
from helpers import run
from PIL import Image
from samples import nai_image

from nai_cli import enhance, errors, images, models
from nai_cli import prompt as P

V5 = models.resolve("v5")
V45 = models.resolve("v4.5")


def test_scales_like_official():
    assert enhance.scales(832, 1216, V5) == ["max", "1.5", "1"]
    assert enhance.scales(832, 1216, V45) == ["1.5", "1"]
    assert enhance.scales(1024, 1024, V5) == ["max", "1.5", "1"]
    assert enhance.scales(512, 768, V45) == ["2", "1.5", "1"]
    assert enhance.scales(1664, 2432, V5) == []
    assert enhance.scales(1216, 2048, V5) == ["max", "1"]
    assert enhance.scales(1216, 2112, V5) == ["1"]


def test_plan_sizes():
    plan = enhance.plan(832, 1216, V5, None, "1.5", None, None)
    # The base image is floored first; the requested size is then rounded to a multiple of 64
    assert plan.image_size == (1248, 1824) and plan.request_size == (1280, 1856) and plan.billed_size == (1280, 1856)
    assert (plan.strength, plan.noise, plan.level) == (0.5, 0.0, 3) and not plan.upscaled
    top = enhance.plan(832, 1216, V5, 5, None, None, 0.05)
    assert top.upscaled and top.request_size == (832, 1216) and top.billed_size == (1440, 2144)
    assert (top.strength, top.noise) == (0.7, 0.05)
    assert enhance.plan(832, 1216, V45, None, None, None, None).scale == "1.5"
    with pytest.raises(errors.NaiError):
        enhance.plan(832, 1216, V45, None, "max", None, None)
    with pytest.raises(errors.NaiError):
        enhance.plan(1664, 2432, V5, None, None, None, None)


def test_max_billed_size():
    assert enhance.max_billed_size(832, 1216) == (1440, 2144)
    assert enhance.max_billed_size(1024, 1024) == (1760, 1760)
    assert enhance.max_billed_size(512, 512) == (1024, 1024)


def test_variation_samples():
    assert enhance.variation_samples(832, 1216) == 3
    assert enhance.variation_samples(1536, 1536) == 3


def test_enhance_hint_like_official():
    assert P.add_enhance_hint("a, b") == "a, b, -2::upscaled, blurry::,"
    assert P.add_enhance_hint("a, b text: hi") == "a, b, -2::upscaled, blurry::, text: hi"
    assert P.add_enhance_hint("a,text: x") == "a, -2::upscaled, blurry::,,text: x"
    assert P.add_enhance_hint("x, upscaled, blurry") == "x, upscaled, blurry"


@pytest.mark.parametrize("size", [(1, 1), (8, 600), (600, 8)])
def test_tiny_images_are_refused_not_crashed(capsys, tmp_path, size):
    img = tmp_path / "tiny.png"
    Image.new("RGB", size, "red").save(img)
    for argv in (["enhance", "--image", str(img), "--prompt", "x", "--dry-run"], ["tool", "pixel-snap", str(img)]):
        code, out = run(capsys, *argv)
        assert out["code"] != "internal" and (code == 0 or out["code"] == "bad_args"), (argv, out)


def test_enhance_from_image(capsys, tmp_path, monkeypatch):
    img = tmp_path / "pic.png"
    nai_image(img)
    sent = []
    original = images.source_image
    monkeypatch.setattr(
        images,
        "source_image",
        lambda path, w, h, keep_alpha: sent.append((w, h)) or original(path, w, h, keep_alpha),
    )
    code, out = run(capsys, "enhance", "--image", str(img), "--dry-run")
    p = out["body"]["parameters"]
    assert (
        code == 0
        and p["upscaled_enhance"] is True
        and (p["width"], p["height"]) == (832, 1216)
        and sent == [(832, 1216)]
    )
    assert p["strength"] == 0.5 and p["noise"] == 0 and p["n_samples"] == 1 and out["seed"] != 5
    assert out["enhance"]["scale"] == "max" and out["enhance"]["billed_size"] == "1440x2144"
    assert out["prompt"].startswith("1girl, smile") and "upscaled, blurry" not in out["prompt"]
    code, out = run(
        capsys, "enhance", "--image", str(img), "--scale", "1.5", "--level", "5", "--seed", "7", "--dry-run"
    )
    p = out["body"]["parameters"]
    assert (p["width"], p["height"]) == (1280, 1856) and sent[-1] == (1248, 1824)
    assert p["strength"] == 0.7 and p["noise"] == 0.1 and out["seed"] == 7 and "upscaled_enhance" not in p
    assert out["prompt"].endswith(", -2::upscaled, blurry::,")
    code, out = run(capsys, "enhance", "--image", str(img), "--size", "832x1216", "--dry-run")
    assert out["code"] == "bad_args"
    code, out = run(capsys, "enhance", "--image", str(img), "--model", "v4.5", "--scale", "max", "--dry-run")
    assert out["code"] == "bad_args" and out["choices"] == ["1.5", "1"]


def test_variations_from_image(capsys, tmp_path):
    img = tmp_path / "pic.png"
    nai_image(img)
    code, out = run(capsys, "variations", "--image", str(img), "--dry-run")
    p = out["body"]["parameters"]
    assert code == 0 and p["n_samples"] == 3 and p["strength"] == 0.8 and p["noise"] == 0.1
    assert (p["width"], p["height"]) == (832, 1216) and out["seed"] != 5 and out["prompt"].startswith("1girl, smile")
    code, out = run(capsys, "variations", "--image", str(img), "--n", "1", "--prompt", "cat", "--dry-run")
    assert out["body"]["parameters"]["n_samples"] == 1 and out["prompt"].startswith("cat")


def test_enhance_without_metadata_needs_prompt(capsys, tmp_path):
    img = tmp_path / "plain.png"
    Image.new("RGB", (832, 1216)).save(img)
    code, out = run(capsys, "enhance", "--image", str(img), "--dry-run")
    assert code == 1 and out["code"] == "no_metadata"
    code, out = run(capsys, "enhance", "--image", str(img), "--prompt", "1girl", "--dry-run")
    assert code == 0 and out["enhance"]["scale"] == "max"


def test_variations_size_follows_the_request(capsys, tmp_path):
    big = tmp_path / "big.png"
    Image.new("RGB", (1664, 2432)).save(big)
    code, out = run(capsys, "variations", "--image", str(big), "--prompt", "1girl", "--dry-run")
    assert code == 0 and out["size"] == "832x1216" and out["body"]["parameters"]["n_samples"] == 3
    odd = tmp_path / "odd.png"
    nai_image(odd, w=1467, h=2144)
    code, out = run(capsys, "variations", "--image", str(odd), "--dry-run")
    # Above the limit once on the grid, so the longer side is clamped and the gap reported
    assert code == 0 and out["size"] == "1472x2112"
    assert any("pixel limit" in w for w in out["warnings"])
