"""Images made with settings in them, damaged or oversized ones, pixel art and Vibe entries, shared by the tests."""

import gzip
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


def settings(prompt="1girl, smile", **comment) -> dict[str, str]:
    """The settings of a NovelAI image as its text keeps them: six fields as text, Comment as JSON."""
    comment = {
        "prompt": prompt,
        "uc": "lowres",
        "seed": 5,
        "width": 64,
        "height": 64,
        "steps": 28,
        "scale": 4.0,
        "sampler": "k_euler_ancestral",
        "signed_hash": "SIGNATURE",
        **comment,
    }
    return {
        "Title": "NovelAI generated image",
        "Description": prompt,
        "Software": "NovelAI",
        "Source": "NovelAI Diffusion V5 0ADF9AB7",
        "Generation time": "6.6",
        "Comment": json.dumps(comment),
    }


def with_comment_field(settings: dict[str, str], **changes) -> dict[str, str]:
    """settings with changes inside its Comment; a field given as None is removed."""
    kept = json.loads(settings["Comment"])
    comment = {k: v for k, v in kept.items() if k not in changes or changes[k] is not None}
    comment.update({k: v for k, v in changes.items() if v is not None})
    return {**settings, "Comment": json.dumps(comment)}


def stealth_payload(settings: dict[str, str], magic: bytes = b"stealth_pngcomp") -> bytes:
    """settings as they are hidden in an image's lowest bits: every field but Title as JSON, gzipped where the
    magic says so, behind the magic and the length in bits, and in the alpha channel before the mark that says no
    error correction follows."""
    data = {k: v for k, v in settings.items() if k != "Title"}
    text = json.dumps(data).encode()
    packed = gzip.compress(text, mtime=0) if magic.endswith(b"comp") else text
    tail = b"\xff\xff\xff\xff" if magic.startswith(b"stealth_png") else b""
    return magic + (len(packed) * 8).to_bytes(4, "big") + packed + tail


def in_low_bits(img: Image.Image, payload: bytes, channels: str = "alpha") -> Image.Image:
    """A copy of img with payload in its lowest bits, column-major as the writers go: the alpha channel's, one bit
    a pixel, or those of red, green and blue, three bits a pixel."""
    arr = np.array(img.convert("RGBA" if channels == "alpha" or "A" in img.getbands() else "RGB"))
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    height = arr.shape[0]
    assert len(bits) <= img.width * height * (1 if channels == "alpha" else 3)
    where = np.arange(len(bits))
    if channels == "alpha":
        rows, cols = where % height, where // height
        arr[rows, cols, 3] = (arr[rows, cols, 3] & 0xFE) | bits
    else:
        pixels = where // 3
        arr[pixels % height, pixels // height, where % 3] = (
            arr[pixels % height, pixels // height, where % 3] & 0xFE
        ) | bits
    return Image.fromarray(arr)


def layered_png(path, text: dict[str, str] | None, stealth: dict[str, str] | None = None, size=(96, 96)):
    """A PNG as NovelAI saves one, in a folder made for it: the settings in its text chunks and in its alpha
    channel; either copy may be left out."""
    img = Image.new("RGBA", size, (10, 20, 30, 255))
    if stealth is not None:
        img = in_low_bits(img, stealth_payload(stealth))
    info = PngImagePlugin.PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, pnginfo=info)
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
