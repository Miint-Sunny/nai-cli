"""Images made with settings in them, damaged or oversized ones, pixel art and Vibe entries, shared by the tests."""

import hashlib
import io
import json
import struct
import zlib

import numpy as np
from PIL import Image, PngImagePlugin


def noisy_png() -> bytes:
    buf = io.BytesIO()
    Image.effect_noise((64, 64), 64).save(buf, format="PNG")
    return buf.getvalue()


# Cut inside the image data, as a dropped connection would
CUT = noisy_png()[:300]


def vibe_entry(**fields) -> dict:
    """A Vibe with the type and id the official client gives it: the hash of its image, or of its first
    encoding."""
    data = {"identifier": "novelai-vibe-transfer", "version": 1, **fields}
    if "image" in data:
        made = {"type": "image", "id": hashlib.sha256(str(data["image"] or "").encode()).hexdigest()}
    else:
        first = next((v["encoding"] for by_model in data["encodings"].values() for v in by_model.values()), "")
        made = {"type": "encoding", "id": hashlib.sha256(first.encode()).hexdigest()}
    return {**made, **data}


def bomb_png() -> bytes:
    """Only a header, declaring far more pixels than Pillow agrees to open."""
    header = b"IHDR" + struct.pack(">IIBBBBB", 20000, 20000, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + header + struct.pack(">I", zlib.crc32(header))


def png_bytes(w=64, h=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def nai_image(
    path, w=832, h=1216, seed=5, prompt="1girl, smile, very aesthetic, masterpiece, no text", uc="lowres", chars=()
):
    center = [{"x": 0.5, "y": 0.5}]
    comment = {
        "prompt": prompt,
        "uc": uc,
        "seed": seed,
        "width": w,
        "height": h,
        "steps": 28,
        "scale": 4.0,
        "sampler": "k_euler_ancestral",
        "cfg_rescale": 0.0,
        "noise_schedule": "karras",
        "n_samples": 1,
        "v4_prompt": {
            "caption": {
                "base_caption": prompt,
                "char_captions": [{"char_caption": c, "centers": center} for c, _ in chars],
            },
            "use_coords": False,
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": uc, "char_captions": [{"char_caption": n, "centers": center} for _, n in chars]}
        },
        "tag_hint_qt": 1,
        "tag_hint_uc_preset": 0,
    }
    info = PngImagePlugin.PngInfo()
    info.add_text("Source", "NovelAI Diffusion V5 0ADF9AB7")
    info.add_text("Comment", json.dumps(comment))
    Image.new("RGB", (w, h), "gray").save(path, pnginfo=info)


def v45_image(path, **settings):
    comment = {
        "prompt": "1girl, smile",
        "uc": "lowres",
        "seed": 5,
        "width": 832,
        "height": 1216,
        "steps": 23,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "noise_schedule": "karras",
        "n_samples": 1,
        **settings,
    }
    info = PngImagePlugin.PngInfo()
    info.add_text("Source", "NovelAI Diffusion V4.5 4BDE2A90")
    info.add_text("Comment", json.dumps(comment))
    Image.new("RGB", (832, 1216), "gray").save(path, pnginfo=info)


def with_comment(path, comment):
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", comment)
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    return path


def pixel_art(rows, cols, pitch, colors=8, noise=4.0, seed=1):
    """Pixel art of rows by cols cells scaled by pitch with noise: the art, and the scaled RGBA pixels."""
    rng = np.random.default_rng(seed)
    art = rng.integers(0, 256, (colors, 3))[rng.integers(0, colors, (rows, cols))]
    ys = (np.arange(int(rows * pitch)) / pitch).astype(int)
    xs = (np.arange(int(cols * pitch)) / pitch).astype(int)
    big = art[ys][:, xs].astype(float) + rng.normal(0, noise, (len(ys), len(xs), 3))
    rgba = np.dstack([np.clip(big, 0, 255), np.full((len(ys), len(xs)), 255)]).astype(np.uint8)
    return art, rgba
