"""Pixel operations of the official client: Pica Lanczos3, nearest neighbour and background fill, bit-exact."""

import math

import numpy as np
from PIL import Image

_WINDOW = 3.0
_FIXED_ONE = 16383
_EPSILON = 1.1920929e-7


def _lanczos3(x: float) -> float:
    x = abs(x)
    if x >= _WINDOW:
        return 0.0
    if x < _EPSILON:
        return 1.0
    xx = x * math.pi
    return math.sin(xx) / xx * math.sin(xx / 3) / (xx / 3)


def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def _filters(src_size: int, dest_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Source indices and 14-bit fixed-point weights per target pixel, zero-padded to (targets, max taps)."""
    scale = dest_size / src_size
    inverted = 1.0 / scale
    clamped = min(1.0, scale)
    window = _WINDOW / clamped
    rows: list[tuple[int, list[int]]] = []
    for dest in range(dest_size):
        center = (dest + 0.5) * inverted
        first = max(0, math.floor(center - window))
        last = min(src_size - 1, math.ceil(center + window))
        raw = [_lanczos3((pixel + 0.5 - center) * clamped) for pixel in range(first, last + 1)]
        total = sum(raw)
        # Pica stores filter values in a Float32Array and reads them back when normalizing
        stored = [float(np.float32(v)) for v in raw]
        fixed = [js_round(_FIXED_ONE * (v / total)) for v in stored]
        filter_total = sum(v / total for v in stored)
        # Pica corrects the error at half the target size as an index, a no-op past the filter length
        if dest_size >> 1 < len(fixed):
            fixed[dest_size >> 1] += js_round(_FIXED_ONE * (1.0 - filter_total))
        nonzero = [i for i, v in enumerate(fixed) if v]
        if nonzero:
            rows.append((first + nonzero[0], fixed[nonzero[0] : nonzero[-1] + 1]))
        else:
            rows.append((0, []))
    taps = max(1, max(len(w) for _, w in rows))
    index = np.zeros((dest_size, taps), dtype=np.int64)
    weight = np.zeros((dest_size, taps), dtype=np.int64)
    for dest, (start, values) in enumerate(rows):
        index[dest, : len(values)] = np.arange(start, start + len(values))
        weight[dest, : len(values)] = values
    return index, weight


def _convolve(data: np.ndarray, index: np.ndarray, weight: np.ndarray, axis: int) -> np.ndarray:
    out = None
    for k in range(index.shape[1]):
        taken = np.take(data, index[:, k], axis=axis)
        w = weight[:, k].reshape((1, -1, 1) if axis == 1 else (-1, 1, 1))
        out = taken * w if out is None else out + taken * w
    return out


def _trunc_div(numerator: np.ndarray, denominator: np.ndarray | int) -> np.ndarray:
    return np.trunc(numerator / denominator).astype(np.int64)


def lanczos(img: Image.Image, width: int, height: int) -> Image.Image:
    """Pica Lanczos3 resize on RGBA; returns RGB when the source has no alpha."""
    width, height = max(1, width), max(1, height)
    if img.size == (width, height):
        return img.copy()
    has_alpha = img.mode == "RGBA"
    out = _resize_rgba(np.asarray(img.convert("RGBA")), width, height)
    result = Image.fromarray(out, "RGBA")
    return result if has_alpha else result.convert("RGB")


def _resize_rgba(src: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = src.shape[:2]
    hor_index, hor_weight = _filters(src_w, width)
    vert_index, vert_weight = _filters(src_h, height)
    data = src.astype(np.int64)
    out = np.empty((height, width, 4), dtype=np.uint8)
    if (src[..., 3] != 255).any():
        # With alpha, premultiply before convolving and unpremultiply after the vertical pass
        alpha = data[..., 3:4]
        premultiplied = np.concatenate([data[..., :3] * alpha, alpha], axis=2)
        summed = _convolve(premultiplied, hor_index, hor_weight, axis=1)
        summed[..., :3] = _trunc_div(summed[..., :3], 255)
        tmp = np.maximum(summed >> 7, 0) & 0xFFFF
        summed = _convolve(tmp, vert_index, vert_weight, axis=0) >> 7
        a = np.clip((summed[..., 3:4] + 8192) >> 14, 0, 255)
        color = summed[..., :3]
        color = np.where(a > 0, _trunc_div(255 * color, np.maximum(a, 1)), color)
        out[..., :3] = np.clip((color + 8192) >> 14, 0, 255)
        out[..., 3:4] = a
        return out
    summed = _convolve(data, hor_index, hor_weight, axis=1)
    tmp = np.maximum(summed >> 7, 0) & 0xFFFF
    summed = _convolve(tmp, vert_index, vert_weight, axis=0) >> 7
    out[...] = np.clip((summed + 8192) >> 14, 0, 255)
    out[..., 3] = 255
    return out


def nearest(img: Image.Image, width: int, height: int) -> Image.Image:
    """Nearest neighbour sampling at the target pixel centre."""
    width, height = max(1, width), max(1, height)
    src = np.asarray(img)
    rows = np.minimum(src.shape[0] - 1, np.floor((np.arange(height) + 0.5) * src.shape[0] / height).astype(np.int64))
    cols = np.minimum(src.shape[1] - 1, np.floor((np.arange(width) + 0.5) * src.shape[1] / width).astype(np.int64))
    return Image.fromarray(np.ascontiguousarray(src[rows][:, cols]), img.mode)


def fill_background(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Alpha-blend onto a background colour; the result is opaque."""
    data = np.asarray(img.convert("RGBA")).astype(np.float64)
    e = data[..., 3:4] / 255
    s = 1.0 * (1 - e)
    total = e + s
    mixed = (data[..., :3] * e + np.array(color, dtype=np.float64) * s) / total
    # ImageData rounds half to even, while alpha uses Math.round
    rgb = np.where(e == 1, data[..., :3], np.clip(np.rint(mixed), 0, 255))
    alpha = np.where(e == 1, 255, np.floor(255 * total + 0.5))
    out = np.concatenate([rgb, alpha], axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGBA").convert("RGB")
