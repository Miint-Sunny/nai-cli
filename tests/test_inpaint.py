"""Feathering and compositing digests come from the official client's bundled code on the same inputs."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from helpers import FakeClient, ledger_entry, run
from PIL import Image, PngImagePlugin

from nai_cli import decode, errors, images, inpaint


def _latent_rect() -> np.ndarray:
    m = np.zeros((32, 24), dtype=np.uint8)
    m[6:20, 5:14] = 255
    m[22:26, 18:24] = 255
    return m


def _pattern(w: int, h: int, alpha: str) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., 0] = (x * 37 + y * 11) % 256
    img[..., 1] = (x * 5 + y * 23) % 256
    img[..., 2] = ((x ^ y) * 7) % 256
    if alpha == "opaque":
        img[..., 3] = 255
    elif alpha == "mixed":
        img[..., 3] = np.where(x < 40, 0, np.where((y // 16) % 2 == 0, 255, (x * 13 + y * 3) % 256))
    else:
        bits = np.unpackbits(np.frombuffer(b"stealth_pngcomp", dtype=np.uint8))
        column_major = ((np.arange(w * h) * 7) // 3 % 2).astype(np.uint8)
        column_major[: bits.size] = bits
        img[..., 3] = 254 + column_major.reshape(w, h).T
    return img


def _sha(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def test_feather_matches_official():
    feather = inpaint.feather(Image.fromarray(_latent_rect(), "L"))
    assert feather.size == (192, 256)
    assert _sha(np.asarray(feather)) == "75c41055981fcf60f0cd6933cf770fe103fc2e838001cefb9f98c67e1c8c5860"


@pytest.mark.parametrize(
    ("base", "box", "digest"),
    [
        (("mixed", 300, 260), (24, 8, 128, 192), "97b4ef0c597b7615dae7c822757fb49e169be937fc503d7d79b2726e907f76cd"),
        (("opaque", 192, 256), (0, 0, 192, 256), "f605bb641de1a900fa567d9f4f9d60462f91eecf3e66db66dd67877a915ead98"),
    ],
)
def test_composite_matches_official(base, box, digest):
    kind, w, h = base
    feather = inpaint.feather(Image.fromarray(_latent_rect(), "L"))
    raw = Image.fromarray(_pattern(192, 256, "stealth"), "RGBA")
    out = inpaint.composite(Image.fromarray(_pattern(w, h, kind), "RGBA"), raw, feather, box)
    assert _sha(np.asarray(out)) == digest


def test_parse_region_and_expand():
    r = inpaint.parse_region("-ellipse:0.1,0.2,0.3,0.4")
    assert r.shape == "ellipse" and r.subtract and r.points == [0.1, 0.2, 0.3, 0.4]
    assert inpaint.parse_region("poly:0,0,1,0,0.5,1").shape == "poly"
    for bad in ("rect:0,0,1", "poly:0,0,1,1", "rect:0,0,1.5,1", "circle:0,0,1,1", "rect:a,b,c,d"):
        with pytest.raises(errors.NaiError):
            inpaint.parse_region(bad)
    assert inpaint.parse_expand("64") == (64, 64, 64, 64)
    assert inpaint.parse_expand("0,32,128,0") == (0, 32, 128, 0)
    for bad in ("0", "1,2", "-1,0,0,0", "x"):
        with pytest.raises(errors.NaiError):
            inpaint.parse_expand(bad)


def test_draw_regions_adds_then_subtracts():
    regions = [inpaint.parse_region("rect:0,0,0.5,1"), inpaint.parse_region("-rect:0,0,0.25,1")]
    mask = np.asarray(inpaint.draw_regions(Image.new("L", (100, 10), 0), regions))
    assert mask[5, 10] == 0 and mask[5, 40] == 255 and mask[5, 80] == 0
    grown = np.asarray(inpaint.grow(Image.fromarray(mask, "L"), 3))
    assert grown[5, 23] == 255 and grown[5, 21] == 0 and grown[5, 53] == 255 and grown[5, 55] == 0


def test_expand_canvas_pads_to_grid_on_expanded_side():
    canvas, mask, offset = inpaint.expand_canvas(Image.new("RGB", (832, 1216), "white"), (0, 0, 100, 0))
    assert canvas.size == (960, 1216) and offset == (0, 0)
    # The seam: one latent cell of the source is repainted along the expanded side
    assert mask.getpixel((823, 0)) == 0 and mask.getpixel((824, 0)) == 255 and mask.getpixel((832, 0)) == 255
    assert canvas.getpixel((900, 10))[3] == 0 and canvas.getpixel((10, 10)) == (255, 255, 255, 255)
    # Each side is rounded up on its own, so the source keeps to the latent grid
    canvas, mask, offset = inpaint.expand_canvas(Image.new("RGB", (832, 1216)), (10, 1, 10, 0))
    assert canvas.size == (960, 1280) and offset == (64, 64)
    assert offset[0] % inpaint.LATENT_SCALE == 0 and offset[1] % inpaint.LATENT_SCALE == 0
    assert (
        mask.getpixel((71, 400)) == 255
        and mask.getpixel((72, 400)) == 0
        and mask.getpixel((887, 400)) == 0
        and mask.getpixel((888, 400)) == 255
    )
    assert mask.getpixel((400, 71)) == 255 and mask.getpixel((400, 72)) == 0 and mask.getpixel((400, 1279)) == 0


def test_focus_sizes_follow_official():
    assert inpaint.focus_request_size(384, 1216) == (512, 1792)
    assert inpaint.focus_request_size(320, 320) == (1024, 1024)
    assert inpaint.focus_request_size(1024, 1024) == (1024, 1024)
    latent = Image.new("L", (104, 152), 0)
    latent.paste(255, (40, 60, 50, 70))
    plan = inpaint.plan_focus(latent, 96)
    assert (plan.width, plan.height) == (320, 320) and plan.x % 8 == 0 and plan.y % 8 == 0
    assert plan.x <= 320 and plan.x + plan.width >= 400 and (plan.request_width, plan.request_height) == (1024, 1024)
    corner = Image.new("L", (104, 152), 0)
    corner.paste(255, (0, 0, 4, 4))
    assert inpaint.plan_focus(corner, 96).x == 0


def test_choose_focus():
    small = Image.new("L", (104, 152), 0)
    small.paste(255, (40, 60, 50, 70))
    assert inpaint.choose_focus(small, (832, 1216), 96, None)[0] is not None
    assert inpaint.choose_focus(small, (832, 1216), 96, False)[0] is None
    large = Image.new("L", (104, 152), 255)
    assert inpaint.choose_focus(large, (832, 1216), 96, None)[0] is None
    # Crops of 0.8-1 MP are sent unscaled and free; focusing keeps the resolution of large images
    medium = Image.new("L", (184, 268), 0)
    medium.paste(255, (40, 60, 136, 148))
    plan, warnings = inpaint.choose_focus(medium, (1472, 2144), 96, None)
    assert (plan.width, plan.height) == (960, 896) and (plan.request_width, plan.request_height) == (960, 896)
    assert not warnings
    # A 1000 px wide canvas allows a 960 px crop, which cannot cover a full-width mask
    band = Image.new("L", (125, 75), 0)
    band.paste(255, (0, 30, 125, 40))
    assert inpaint.choose_focus(band, (1000, 600), 96, None)[0] is None
    plan, warnings = inpaint.choose_focus(band, (1000, 600), 96, True)
    assert plan.width == 960 and warnings
    huge = Image.new("L", (208, 304), 255)
    assert inpaint.choose_focus(huge, (1664, 2432), 96, None) == (None, [])
    with pytest.raises(errors.NaiError):
        inpaint.choose_focus(huge, (1664, 2432), 96, True)


def _prepare(source, **kw):
    options = {
        "mask": None,
        "regions": [],
        "grow_pixels": 0,
        "expand": None,
        "size": None,
        "current": (832, 1216),
        "focus": None,
        "padding": 96,
        "keep_alpha": False,
    }
    options.update(kw)
    return inpaint.prepare(source, **options)


def test_prepare_focus_and_whole():
    src = Image.new("RGB", (832, 1216), "gray")
    job = _prepare(src, regions=[inpaint.parse_region("rect:0.4,0.4,0.5,0.5")])
    assert job.focus is not None and (job.width, job.height) == (1024, 1024)
    assert Image.open(io.BytesIO(job.mask)).size == (1024, 1024) and job.feather.size == (1024, 1024)
    assert job.base is job.canvas and job.box[2:] == (job.focus.width, job.focus.height)
    x, y = job.remap(0.45, 0.45)
    assert 0 < x < 1 and 0 < y < 1 and job.remap(0, 0) == (0.0, 0.0)
    whole = _prepare(src, regions=[inpaint.parse_region("rect:0.4,0.4,0.5,0.5")], focus=False)
    assert whole.focus is None and (whole.width, whole.height) == (832, 1216) and whole.box == (0, 0, 832, 1216)
    assert whole.remap(0.3, 0.7) == (0.3, 0.7)
    with pytest.raises(errors.NaiError) as e:
        _prepare(src)
    assert e.value.code == errors.EMPTY_MASK


def test_editor_canvas_size_like_official():
    assert inpaint.editor_canvas_size(832, 1216) == (832, 1216)
    assert inpaint.editor_canvas_size(1467, 2144) == (1472, 2176)
    assert inpaint.editor_canvas_size(1000, 1500) == (1024, 1536)
    assert inpaint.editor_canvas_size(3000, 4000) == (1920, 2560)
    assert inpaint.editor_canvas_size(4000, 3000) == (2560, 1920)


def test_prepare_whole_canvas_like_official():
    nearly_all = [inpaint.parse_region("rect:0.05,0.05,0.95,0.95")]
    stretched = _prepare(Image.new("RGB", (1000, 1500), "gray"), regions=nearly_all)
    # Without focus, canvases up to 3 MP are sent at their own size
    assert stretched.canvas.size == (1024, 1536) and (stretched.width, stretched.height) == (1024, 1536)
    assert stretched.focus is None and any("1024x1536" in w for w in stretched.warnings)
    big_region = [inpaint.parse_region("rect:0.1,0.1,0.75,0.55")]
    shrunk = _prepare(Image.new("RGB", (1664, 2432), "gray"), regions=big_region)
    assert shrunk.focus is None and (shrunk.width, shrunk.height) == (832, 1216)
    assert any("832x1216" in w and "--focus" in w for w in shrunk.warnings)
    assert not _prepare(Image.new("RGB", (832, 1216), "gray"), regions=big_region).warnings
    chosen = _prepare(Image.new("RGB", (1664, 2432), "gray"), regions=big_region, size=(832, 1216))
    assert not chosen.warnings


def test_prepare_mask_file_is_stretched_and_combined():
    src = Image.new("RGB", (832, 1216))
    painted = Image.new("L", (104, 152), 0)
    painted.paste(255, (0, 0, 20, 20))
    job = _prepare(src, mask=painted, regions=[inpaint.parse_region("-rect:0,0,0.1,0.1")], focus=False)
    latent = np.asarray(job.latent)
    assert latent[0, 0] == 0 and latent[18, 18] == 255
    expanded = _prepare(src, mask=painted, expand=(64, 0, 0, 0), focus=False)
    assert expanded.canvas.size == (896, 1216) and np.asarray(expanded.latent)[0, 9] == 255


def test_paste_keeps_canvas_size_and_text_chunks():
    src = Image.new("RGB", (832, 1216), "gray")
    job = _prepare(src, regions=[inpaint.parse_region("rect:0.4,0.4,0.5,0.5")])
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", '{"prompt": "hand"}')
    buf = io.BytesIO()
    Image.new("RGBA", (job.width, job.height), (255, 0, 0, 255)).save(buf, format="PNG", pnginfo=info)
    pasted = job.paste(buf.getvalue())
    out = Image.open(io.BytesIO(pasted))
    assert out.size == (832, 1216) and out.text["Comment"] == '{"prompt": "hand"}'
    center = job.focus.x + job.focus.width // 2, job.focus.y + job.focus.height // 2
    assert out.getpixel(center)[:3] == (255, 0, 0) and out.getpixel((5, 5))[:3] == (128, 128, 128)
    preview = job.preview()
    assert preview.size == (832, 1216) and preview.getpixel(center)[0] > preview.getpixel((5, 5))[0]


def test_copy_text_chunks_goes_after_header():
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "x")
    info.add_itxt("Description", "说明")
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG", pnginfo=info)
    merged = images.copy_text_chunks(buf.getvalue(), images.to_png(Image.new("RGB", (3, 3))))
    kinds = [kind for kind, *_ in decode.png_chunks(merged)]
    assert kinds[:3] == [b"IHDR", b"tEXt", b"iTXt"] and kinds[-1] == b"IEND"
    assert Image.open(io.BytesIO(merged)).text == {"Comment": "x", "Description": "说明"}


def test_expanded_sides_repaint_a_cell_of_the_source():
    _, mask, offset = inpaint.expand_canvas(Image.new("RGB", (128, 128)), (64, 64, 0, 0))
    assert offset == (64, 64)
    # Left and top reach 8 pixels in; the sides not expanded stay as they were
    assert mask.getpixel((71, 100)) == 255 and mask.getpixel((72, 100)) == 0
    assert mask.getpixel((100, 71)) == 255 and mask.getpixel((100, 72)) == 0
    assert mask.getpixel((191, 191)) == 0


def test_a_subtracting_region_takes_the_spelling_of_the_help(capsys, tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (64, 64), "white").save(img)
    outs = []
    for name, subtract in (
        ("spaced", ["--region", "-ellipse:0.2,0.2,0.5,0.5"]),
        ("joined", ["--region=-ellipse:0.2,0.2,0.5,0.5"]),
    ):
        target = tmp_path / f"{name}.png"
        code, out = run(capsys, "redact", str(img), "--region", "rect:0.1,0.1,0.8,0.8", *subtract, "-o", str(target))
        assert code == 0, out
        outs.append(target.read_bytes())
    assert outs[0] == outs[1]
    code, out = run(capsys, "redact", str(img), "--region", "-x")
    assert code == 1 and out["code"] == "bad_args"


def test_inpaint_empty_mask(capsys, tmp_path):
    img, mask = tmp_path / "a.png", tmp_path / "m.png"
    Image.new("RGB", (64, 64)).save(img)
    Image.new("L", (64, 64), 0).save(mask)
    _, out = run(capsys, "inpaint", "--image", str(img), "--mask", str(mask), "--size", "64x64", "--dry-run")
    assert out["code"] == "empty_mask"


def test_inpaint_focus_by_region(capsys, tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (832, 1216)).save(img)
    hand = ("inpaint", "--image", str(img), "--region", "rect:0.4,0.4,0.5,0.5", "--prompt", "hand", "--dry-run")
    code, out = run(capsys, *hand)
    assert code == 0 and out["size"] == "1024x1024" and out["inpaint"]["focus"]["request_size"] == "1024x1024"
    assert out["cost"]["estimate"] == 0 and out["inpaint"]["canvas"] == "832x1216"
    code, out = run(capsys, *hand, "--no-focus")
    assert out["size"] == "832x1216" and out["inpaint"]["focus"] is None
    code, out = run(capsys, *hand, "--size", "832x1216", "--focus")
    assert out["code"] == "bad_args"
    code, out = run(capsys, *hand, "--focus-padding", "500")
    assert out["code"] == "bad_args"
    code, out = run(capsys, "inpaint", "--image", str(img), "--prompt", "x", "--dry-run")
    assert out["code"] == "bad_args"


def test_inpaint_expand_preview(capsys, tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (832, 1216), "white").save(img)
    code, out = run(capsys, "inpaint", "--image", str(img), "--expand", "0,0,100,0", "--prompt", "sky", "--preview")
    assert code == 0 and out["preview"] and out["inpaint"]["canvas"] == "960x1216"
    assert out["size"] == "512x1792" and Image.open(out["preview"]).size == (960, 1216)


def test_inpaint_pastes_back(capsys, with_token, tmp_path, monkeypatch):
    sizes = []

    def generate(self, body):
        p = body["parameters"]
        sizes.append((p["width"], p["height"]))
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", '{"seed": 1}')
        buf = io.BytesIO()
        Image.new("RGBA", (p["width"], p["height"]), (255, 0, 0, 255)).save(buf, format="PNG", pnginfo=info)
        return [buf.getvalue()]

    monkeypatch.setattr(FakeClient, "generate", generate)
    img = tmp_path / "a.png"
    Image.new("RGB", (832, 1216), "gray").save(img)
    code, out = run(
        capsys, "inpaint", "--image", str(img), "--region", "ellipse:0.4,0.4,0.5,0.5", "--prompt", "hand", "--keep-raw"
    )
    assert code == 0 and sizes == [(1024, 1024)] and not out["warnings"]
    merged = Image.open(out["images"][0])
    assert merged.size == (832, 1216) and merged.text["Comment"] == '{"seed": 1}'
    assert merged.getpixel((5, 5))[:3] == (128, 128, 128) and merged.getpixel((374, 547))[:3] == (255, 0, 0)
    assert Image.open(out["raw_images"][0]).size == (1024, 1024)
    assert ledger_entry(out["raw_images"][0])["seed"] == out["seed"]


def test_focus_does_not_hide_bad_positions(capsys, tmp_path):
    img = tmp_path / "s.png"
    Image.new("RGB", (1024, 1024)).save(img)
    for pos in ("nan,0.5", "0.5,9"):
        argv = ["--region", "rect:0.4,0.4,0.5,0.5", "--char", "girl", "--char-pos", pos, "--dry-run"]
        code, out = run(capsys, "inpaint", "--image", str(img), "--prompt", "x", *argv)
        assert code == 1 and out["code"] == "bad_args"


def test_chosen_model_decides_references_when_inpainting(capsys, tmp_path):
    # V5 Curated inpaints with V4.5 Curated, but the official client offers neither feature for it
    for extra in (["--refs", "1"], ["--vibes", "1"]):
        code, out = run(capsys, "cost", "--model", "v5-curated", "--inpaint", *extra)
        assert code == 1 and out["code"] == "model_unsupported" and out["model"] == "nai-diffusion-5-curated"
    code, out = run(capsys, "cost", "--model", "v4.5-curated", "--inpaint", "--refs", "1")
    assert code == 0 and out["model"] == "nai-diffusion-4-5-curated-inpainting"
    img = tmp_path / "s.png"
    Image.new("RGB", (64, 64)).save(img)
    argv = ["--image", str(img), "--region", "rect:0,0,1,1", "--prompt", "x", "--no-focus", "--dry-run"]
    for extra in (["--ref", str(img)], ["--vibe", str(img)]):
        code, out = run(capsys, "inpaint", *argv, "--model", "v5-curated", *extra)
        assert code == 1 and out["code"] == "model_unsupported"
    code, out = run(capsys, "inpaint", *argv, "--model", "v4.5-curated", "--ref", str(img))
    assert code == 0 and out["model"] == "nai-diffusion-4-5-curated-inpainting"
    assert len(out["body"]["parameters"]["director_reference_images"]) == 1
    # Vibes are never sent when inpainting, as in the official client
    code, out = run(capsys, "inpaint", *argv, "--model", "v4.5-curated", "--vibe", str(img))
    assert code == 0 and "reference_image_multiple" not in out["body"]["parameters"]


def test_a_preview_is_a_finished_job_and_a_dry_run_writes_none(capsys, tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (512, 512)).save(img)
    argv = ["inpaint", "--image", str(img), "--region", "rect:0.1,0.1,0.5,0.5", "--prompt", "x", "--preview"]
    code, out = run(capsys, *argv, "--dry-run")
    assert code == 0 and out["dry_run"] is True and out["preview"] is None
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".png"] == ["a.png"]
    code, out = run(capsys, *argv)
    assert code == 0 and "dry_run" not in out and Path(out["preview"]).exists()
