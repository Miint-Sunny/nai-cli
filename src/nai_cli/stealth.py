"""Stealth data: settings hidden in the lowest bits of an image's channels, where a block lies and how to clear it."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from nai_cli import decode

MAGIC_BITS = 15 * 8
LENGTH_BITS = 32
HEADER_BITS = MAGIC_BITS + LENGTH_BITS


class Channel(Enum):
    ALPHA = "alpha"
    RGB = "rgb"


# Every magic is 15 bytes and says which channels hold the data and whether it is gzipped
NOVELAI_MAGIC = b"stealth_pngcomp"
MAGICS = {
    b"stealth_pnginfo": (Channel.ALPHA, False),
    NOVELAI_MAGIC: (Channel.ALPHA, True),
    b"stealth_rgbinfo": (Channel.RGB, False),
    b"stealth_rgbcomp": (Channel.RGB, True),
}


@dataclass(frozen=True)
class Block:
    """The stealth data an image keeps: the channels it lies in, its data, and the lowest bits it takes from the
    first pixel on."""

    channel: Channel
    compressed: bool
    data: bytes
    # The magic, the length and the data; the alpha channel may hold error correction after them
    bits: int


def find(img: Image.Image, max_pixels: float) -> Block | None:
    """The stealth data in the lowest bits of an RGB or RGBA image, read column-major, the alpha channel before the
    colours. None when the image has more pixels than max_pixels, as the official reader skips large images, or
    when no block starts there."""
    if img.width * img.height > max_pixels:
        return None
    arr = np.asarray(img)
    channels = (Channel.ALPHA, Channel.RGB) if "A" in img.getbands() else (Channel.RGB,)
    return next((block for channel in channels if (block := _block(arr, channel)) is not None), None)


def begins_in_alpha(img: Image.Image, max_pixels: float) -> bool:
    """Whether the alpha channel's lowest bits start with NovelAI's magic, which is all the official client looks
    for before it cleans an image's alpha."""
    if img.width * img.height > max_pixels or "A" not in img.getbands():
        return False
    return _magic(np.asarray(img), Channel.ALPHA) == NOVELAI_MAGIC


def clear(img: Image.Image, block: Block) -> Image.Image:
    """A copy of an RGB or RGBA image without the lowest bits a block in its colours takes. The colours are the
    picture itself, so nothing else changes, and a cleared value by one at most."""
    arr = np.array(img)
    where = np.arange(block.bits)
    pixels = where // 3
    rows, cols = pixels % arr.shape[0], pixels // arr.shape[0]
    arr[rows, cols, where % 3] &= 0xFE
    return Image.fromarray(arr)


def _magic(arr: np.ndarray, channel: Channel) -> bytes:
    per_column = _per_column(arr, channel)
    return np.packbits(_low_bits(arr, channel, -(-MAGIC_BITS // per_column))[:MAGIC_BITS]).tobytes()


def _block(arr: np.ndarray, channel: Channel) -> Block | None:
    """The block that starts in a channel's lowest bits; None unless a magic of that channel starts there, its
    length fits the image, and its data decompresses."""
    magic = _magic(arr, channel)
    if magic not in MAGICS or MAGICS[magic][0] is not channel:
        return None
    per_column = _per_column(arr, channel)
    bits = _low_bits(arr, channel, -(-HEADER_BITS // per_column))
    length = int.from_bytes(np.packbits(bits[MAGIC_BITS:HEADER_BITS]).tobytes(), "big")
    # A length that is not a whole number of bytes, or more than the image holds, is noise behind a magic
    if length <= 0 or length % 8 or HEADER_BITS + length > arr.shape[1] * per_column:
        return None
    bits = _low_bits(arr, channel, -(-(HEADER_BITS + length) // per_column))
    payload = np.packbits(bits[HEADER_BITS : HEADER_BITS + length]).tobytes()
    try:
        data = decode.inflate(payload, decode.GZIP) if MAGICS[magic][1] else payload
    except ValueError:
        return None
    return Block(channel, MAGICS[magic][1], data, HEADER_BITS + length)


def _per_column(arr: np.ndarray, channel: Channel) -> int:
    """The bits a column of the image gives to a channel."""
    return arr.shape[0] if channel is Channel.ALPHA else 3 * arr.shape[0]


def _low_bits(arr: np.ndarray, channel: Channel, columns: int) -> np.ndarray:
    """The lowest bits of the first columns of a channel, column-major: one a pixel from the alpha channel, three
    a pixel from the colours."""
    if channel is Channel.ALPHA:
        return (arr[:, :columns, 3] & 1).T.reshape(-1)
    return (arr[:, :columns, :3] & 1).transpose(1, 0, 2).reshape(-1)
