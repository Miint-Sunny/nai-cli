"""Images as the requests send them: opened, converted, resized for import, masked and stripped."""

import io
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageSequence

from nai_cli import decode, errors, presets, resample, stealth
from nai_cli.presets import GRID, MAX_PIXELS, nearest_grid

LATENT_SCALE = 8
PRECISE_REFERENCE_CANVASES = [(1024, 1536), (1536, 1024), (1472, 1472)]
# Masks are binarized on alpha before sending; values above this are repainted
MASK_THRESHOLD = 155


def open_image(path: str | Path) -> Image.Image:
    try:
        img = _open(path)
        img.load()
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    return ImageOps.exif_transpose(img)


# Modes PNG stores as they are; others, such as CMYK from a JPEG, are converted first
_PNG_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "I;16"})


def to_png(img: Image.Image) -> bytes:
    if img.mode == "I":
        # Pillow stores I as 16 bits but no longer takes it as is; I;16 writes the same file
        img = img.convert("I;16")
    elif img.mode not in _PNG_MODES:
        img = rgb_or_rgba(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def copy_text_chunks(source: bytes, target: bytes, leaving: Collection[str] = ()) -> bytes:
    """Copy the text chunks of source verbatim after the IHDR of target, so the composite keeps its metadata;
    chunks of the keywords in leaving are not copied."""
    texts = [
        source[start:end]
        for kind, body, start, end in decode.png_chunks(source)
        if kind in decode.TEXT_CHUNKS and body.split(b"\0", 1)[0].decode("latin1") not in leaving
    ]
    return _after_header(target, b"".join(texts))


def add_text(target: bytes, texts: dict[str, str]) -> bytes:
    """texts written as iTXt chunks after the IHDR of target."""
    return _after_header(target, b"".join(decode.itxt(key, text) for key, text in texts.items()))


def with_settings(img: Image.Image, settings: dict[str, str]) -> bytes:
    """A PNG of the image without what it held, with settings written where NovelAI writes them: each field as a
    text chunk, and all but Title, gzipped, in the lowest bits of an alpha channel, added opaque where the image has
    none. bad_args for an image too small for them."""
    hidden = _bare(img).convert("RGBA")
    try:
        hidden = stealth.embed(hidden, stealth.payload(settings))
    except ValueError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"the image is too small for its settings: {e}") from e
    return add_text(to_png(hidden), dict(settings))


def _after_header(target: bytes, chunks: bytes) -> bytes:
    for kind, _, _, end in decode.png_chunks(target):
        if kind == b"IHDR":
            return target[:end] + chunks + target[end:]
    return target


# Reference sides for imported images: 1216 long and 896 short for portrait
_IMPORT_LONG_SIDE = 1216
_IMPORT_SHORT_SIDE = 896


def import_size(width: int, height: int, current: tuple[int, int]) -> tuple[int, int]:
    """Generation size for an imported image: the current size if the aspect ratio matches and it fits;
    the source size if it is a multiple of 64 within the limit; otherwise scaled from a 1216 long or 896 short side."""
    ratio = width / height
    short, long = (height, width) if ratio > 1 else (width, height)
    oriented = short / long
    # Compared against the portrait-oriented sides while current stays as is
    if current[0] / current[1] == ratio and short <= current[0] and long <= current[1]:
        return current
    if not (short % GRID == 0 and long % GRID == 0 and short * long <= MAX_PIXELS):
        from_long = nearest_grid(_IMPORT_LONG_SIDE * oriented)
        from_short = nearest_grid(_IMPORT_SHORT_SIDE / oriented)
        long_error = abs(from_long / _IMPORT_LONG_SIDE - oriented)
        short_error = abs(_IMPORT_SHORT_SIDE / from_short - oriented)
        if long_error < short_error and from_long * _IMPORT_LONG_SIDE <= MAX_PIXELS:
            short, long = from_long, _IMPORT_LONG_SIDE
        elif _IMPORT_SHORT_SIDE * from_short <= MAX_PIXELS:
            short, long = _IMPORT_SHORT_SIDE, from_short
        else:
            short, long = 512, 512
    return (long, short) if ratio > 1 else (short, long)


def clear_stealth_alpha(img: Image.Image) -> Image.Image:
    """For images carrying stealth data in their alpha channel, alpha 254 becomes 255 and 1 becomes 0, so the bits
    do not show as noise."""
    if not stealth.begins_in_alpha(img, decode.STEALTH_MAX_PIXELS):
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: 255 if v == 254 else 0 if v == 1 else v))
    return out


def source_image(path: str | Path, width: int, height: int, keep_alpha: bool) -> bytes:
    """Base image for img2img and inpainting, stretched to size, on white when the model has no transparency."""
    return to_png(source_canvas(open_image(path), width, height, keep_alpha))


def clean_source(img: Image.Image) -> Image.Image:
    return clear_stealth_alpha(rgb_or_rgba(img))


def source_canvas(img: Image.Image, width: int, height: int, keep_alpha: bool) -> Image.Image:
    img = resample.lanczos(clean_source(img), width, height)
    if not keep_alpha and img.mode == "RGBA":
        img = resample.fill_background(img, (255, 255, 255))
    return img


def mask_coverage(img: Image.Image) -> Image.Image:
    """White or opaque areas are repainted: alpha decides for transparent masks, luminance for opaque ones."""
    if "A" in img.getbands() and img.getextrema()[-1][0] < 255:
        return img.getchannel("A")
    return img.convert("L")


def latent_mask(coverage: Image.Image, width: int, height: int) -> Image.Image:
    """Nearest-neighbour downscale to latent size (1/8) and threshold at 155; returns a 0/255 grayscale image."""
    latent = resample.nearest(coverage, width // LATENT_SCALE, height // LATENT_SCALE)
    binary = latent.point(lambda v: 255 if v > MASK_THRESHOLD else 0)
    if binary.getextrema()[1] == 0:
        raise errors.NaiError(errors.EMPTY_MASK, "the mask selects nothing to repaint")
    return binary


def mask_request(latent: Image.Image, width: int, height: int) -> bytes:
    """Nearest-neighbour upscale back to the generation size on black: white is repainted, black is kept."""
    full = resample.nearest(latent, width, height)
    return to_png(Image.merge("RGBA", (full, full, full, Image.new("L", full.size, 255))))


def tool_image(img: Image.Image) -> tuple[bytes, int, int]:
    """Director tool input, scaled to roughly 1 to 3 megapixels."""
    img = rgb_or_rgba(img)
    width, height = presets.tool_request_size(img.width, img.height)
    return to_png(resample.lanczos(img, width, height)), width, height


def rgb_or_rgba(img: Image.Image) -> Image.Image:
    """img as RGB, or as RGBA when it has an alpha channel; any other mode is converted to one of them."""
    if img.mode in ("RGB", "RGBA"):
        return img
    return img.convert("RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB")


def precise_reference_image(path: str | Path, transparent: bool) -> bytes:
    return to_png(precise_reference_canvas(open_image(path), transparent))


def precise_reference_canvas(img: Image.Image, transparent: bool = False) -> Image.Image:
    """Fit into the closest of three canvases and centre on transparency, which a model without transparency gets
    filled black."""
    img = clean_source(img)
    ratio = img.width / img.height
    cw, ch = min(PRECISE_REFERENCE_CANVASES, key=lambda c: abs(c[0] / c[1] - ratio))
    if ratio > cw / ch:
        width, height = cw, resample.js_round(cw / ratio)
    else:
        width, height = resample.js_round(ch * ratio), ch
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    left, top = resample.js_round((cw - width) / 2), resample.js_round((ch - height) / 2)
    canvas.paste(resample.lanczos(img, width, height).convert("RGBA"), (left, top))
    return canvas if transparent else resample.fill_background(canvas, (0, 0, 0))


def size_of(path: str | Path) -> tuple[int, int]:
    return open_image(path).size


def strip_metadata(img: Image.Image) -> bytes:
    """A PNG of the pixels alone: no text chunks, EXIF or colour profile, and no low alpha bit, where stealth
    settings hide."""
    return to_png(_bare(img))


def strip_file(path: str | Path) -> bytes:
    """strip_metadata for an image file; an animated one stays animated, each frame cleaned, as an APNG."""
    path = Path(path)
    try:
        with _open(path) as img:
            animated = getattr(img, "n_frames", 1) > 1
            frames, durations, loop = _frames(img) if animated else ((), (), 0)
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    return _apng(frames, durations, loop) if animated else strip_metadata(open_image(path))


def strip_in_format(path: str | Path, warnings: list[str]) -> bytes:
    """strip_file for a copy in the format of its source. A JPEG, and a WebP without a transparent pixel, lose only
    the segments and chunks that hold metadata, and what follows their image in the file, and stay as they are
    otherwise; a lossless or an animated WebP is
    cleaned as a PNG is and saved as lossless WebP; a lossy WebP with transparent pixels has to be encoded again,
    with loss, which warnings says. Any other format is saved as a PNG, as strip_file does."""
    path = Path(path)
    try:
        data = path.read_bytes()
        if decode.format_of(data) == "JPEG":
            return decode.strip_jpeg(data)
        if decode.format_of(data) == "WEBP":
            return _strip_webp(data, path, warnings)
    except ValueError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot strip {path}: {e}") from e
    except OSError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    return strip_file(path)


def open_still(path: str | Path) -> Image.Image:
    """open_image for edits that draw on one picture, which an animation is not."""
    try:
        with _open(path) as img:
            animated = getattr(img, "n_frames", 1) > 1
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    if animated:
        raise errors.NaiError(errors.BAD_ARGS, f"{path} is animated; only still images can be edited")
    return open_image(path)


def _open(path: str | Path) -> Image.Image:
    return Image.open(io.BytesIO(Path(path).read_bytes()))


def _strip_webp(data: bytes, path: Path, warnings: list[str]) -> bytes:
    """A WebP without its metadata. A lossy WebP's pixels are no place for stealth data: what it may hold lies in
    its alpha chunk, which goes, unless a pixel is transparent and so the picture has to be encoded again. A lossless
    or an animated WebP may hide settings in its pixels, which are cleaned as a PNG's are."""
    kinds = {kind for kind, _ in decode.webp_chunks(data)}
    try:
        with _open(path) as img:
            animated = getattr(img, "n_frames", 1) > 1
            untouched = b"VP8 " in kinds and (b"ALPH" not in kinds or _opaque(img))
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    if untouched:
        return decode.strip_webp(data)
    if animated:
        with _open(path) as source:
            frames, durations, loop = _frames(source)
        return _webp(
            frames[0], save_all=True, append_images=frames[1:], duration=durations, loop=loop, lossless=True, exact=True
        )
    if b"VP8L" in kinds:
        return _webp(_bare(open_image(path)), lossless=True, exact=True)
    warnings.append(f"{path.name}: it has transparent pixels, so its WebP is encoded again, with loss")
    return _webp(_bare(open_image(path)), quality=95, alpha_quality=100, exact=True)


def _opaque(img: Image.Image) -> bool:
    """Whether no pixel of the image is transparent: every alpha is 254 or more, as NovelAI's stealth data leaves
    them."""
    return "A" not in img.getbands() or img.getchannel("A").getextrema()[0] >= 254


def _frames(img: Image.Image) -> tuple[list[Image.Image], list[int], int]:
    """Every frame of an animation cleaned, how long each shows in milliseconds, and how often the file loops."""
    frames, durations = [], []
    for frame in ImageSequence.Iterator(img):
        # Pillow composites each frame whole, so every frame replaces the last when played
        frames.append(_bare(frame.convert("RGBA")))
        durations.append(frame.info.get("duration", 0))
    return frames, durations, img.info.get("loop", 1)


def _apng(frames: list[Image.Image], durations: list[int], loop: int) -> bytes:
    buf = io.BytesIO()
    frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:], duration=durations, loop=loop)
    return buf.getvalue()


def _webp(img: Image.Image, **saving: Any) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", **saving)
    return buf.getvalue()


def _bare(img: Image.Image) -> Image.Image:
    # Alpha keeps its look: 254 becomes 255 and 1 becomes 0, as cleaning stealth data does
    if "A" in img.getbands() or "transparency" in img.info:
        img = img.convert("RGBA")
        img.putalpha(img.getchannel("A").point(lambda v: 255 if v >= 254 else 0 if v <= 1 else v & 0xFE))
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # Other tools hide settings in the colours, which are the picture itself, so only the bits they take are cleared
    searched = rgb_or_rgba(img)
    if (block := stealth.find(searched, math.inf)) is not None:
        img = stealth.clear(searched, block)
    # A new image carries none of the source's info, which Pillow would otherwise write back
    return Image.frombytes(img.mode, img.size, img.tobytes())
