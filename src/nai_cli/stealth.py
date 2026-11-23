"""Stealth data: settings hidden in the lowest bits of an image's channels, where a block lies, how to clear it
and how NovelAI writes one."""

import gzip
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from nai_cli import decode

# The alpha channel holds every field of the PNG text but Title
HIDDEN = ("Description", "Software", "Source", "Generation time", "Comment")
MAGIC_BITS = 15 * 8
LENGTH_BITS = 32
HEADER_BITS = MAGIC_BITS + LENGTH_BITS
# After the data, the alpha channel names the length of its error correction code, which NovelAI never writes
NO_CORRECTION = 0xFFFFFFFF


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


def payload(settings: dict[str, str]) -> bytes:
    """settings as NovelAI hides them: every field but Title as one JSON text, gzipped behind the magic and the
    length in bits, and before the mark that says no error correction follows. The gzip header names no time, so the
    same settings give the same bytes."""
    text = json.dumps({key: settings[key] for key in HIDDEN if key in settings}, ensure_ascii=False)
    packed = gzip.compress(text.encode(), mtime=0)
    return NOVELAI_MAGIC + (len(packed) * 8).to_bytes(4, "big") + packed + NO_CORRECTION.to_bytes(4, "big")


def embed(img: Image.Image, payload: bytes) -> Image.Image:
    """A copy of an RGBA image with payload in the lowest bits of its alpha channel, column-major as NovelAI writes
    it. ValueError when the image has fewer pixels than the payload has bits."""
    arr = np.array(img)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    height, width = arr.shape[:2]
    if len(bits) > height * width:
        raise ValueError(f"{len(bits)} bits need as many pixels, and the image has {height} times {width}")
    where = np.arange(len(bits))
    rows, cols = where % height, where // height
    arr[rows, cols, 3] = (arr[rows, cols, 3] & 0xFE) | bits
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
