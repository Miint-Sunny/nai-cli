"""Image bytes from files and replies: whether they decode, their size and format, the chunks of a PNG and a WebP,
the segments of a JPEG, and bounded inflating."""

import io
import struct
import zlib
from collections.abc import Iterator

from PIL import Image

# What Pillow raises for a damaged, truncated or oversized image
BAD_IMAGE = (OSError, ValueError, SyntaxError, struct.error, Image.DecompressionBombError)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# The official reader skips stealth data in larger images
STEALTH_MAX_PIXELS = 0x1000000
TEXT_CHUNKS = (b"tEXt", b"iTXt", b"zTXt")
# The marker that ends a JPEG image, with which a copy ends
JPEG_END = b"\xff\xd9"
# Settings and tokenizer files are a few megabytes; anything inflating past this is not one
MAX_INFLATED = 64 * 1024 * 1024
# zlib's wbits for each container
ZLIB, GZIP, RAW = zlib.MAX_WBITS, 16 + zlib.MAX_WBITS, -zlib.MAX_WBITS


def inflate(data: bytes, wbits: int = ZLIB, limit: int = MAX_INFLATED) -> bytes:
    """ValueError for data that is damaged or inflates past limit, so a small file cannot fill the memory."""
    inflater = zlib.decompressobj(wbits)
    try:
        out = inflater.decompress(data, limit)
    except zlib.error as e:
        raise ValueError(f"damaged compressed data: {e}") from e
    if inflater.unconsumed_tail:
        raise ValueError(f"compressed data inflates past {limit} bytes")
    return out


def png_chunks(data: bytes) -> Iterator[tuple[bytes, bytes, int, int]]:
    """Yield (type, data, start, end) per chunk, spans including length and CRC; nothing for non-PNG data."""
    if data[:8] != PNG_SIGNATURE:
        return
    i = 8
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        kind = data[i + 4 : i + 8]
        end = i + 12 + length
        # A chunk cut short ends the file there
        if end > len(data):
            return
        yield kind, data[i + 8 : i + 8 + length], i, end
        if kind == b"IEND":
            return
        i = end


def png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def format_of(data: bytes) -> str | None:
    """PNG, WEBP or JPEG, as Pillow names the format, for bytes that begin like one; None for anything else."""
    if data.startswith(PNG_SIGNATURE):
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return "JPEG" if data[:2] == b"\xff\xd8" else None


def webp_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """(kind, payload) per chunk of a WebP file, in file order: those within the size its RIFF header names, not what
    a tool may have put after them. A chunk cut short ends the file there, and data that is no WebP has none."""
    out: list[tuple[bytes, bytes]] = []
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return out
    # The size counts everything after it, the file's kind included
    end = min(8 + struct.unpack("<I", data[4:8])[0], len(data))
    at = 12
    while at + 8 <= end:
        kind = data[at : at + 4]
        (length,) = struct.unpack("<I", data[at + 4 : at + 8])
        if at + 8 + length > end:
            break
        out.append((kind, data[at + 8 : at + 8 + length]))
        # A chunk of odd length is followed by a pad byte
        at += 8 + length + (length & 1)
    return out


def webp_file(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """A WebP file of the chunks, in order."""
    body = b"".join(
        kind + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")
        for kind, payload in chunks
    )
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body


def strip_webp(data: bytes) -> bytes:
    """A still WebP without the chunks that hold metadata, EXIF, XMP and the alpha channel its stealth data may lie
    in, and without what lies past the size its RIFF header names. The picture stays byte for byte, and so does its
    colour profile."""
    kept = [(kind, payload) for kind, payload in webp_chunks(data) if kind not in (b"EXIF", b"XMP ", b"ALPH")]
    kinds = {kind for kind, _ in kept}
    out = []
    for kind, payload in kept:
        if kind == b"VP8X":
            # VP8X says which optional chunks follow: a colour profile 0x20, alpha 0x10, EXIF 0x08, XMP 0x04
            flags = payload[0] & ~(0x10 | 0x08 | 0x04)
            if b"ICCP" not in kinds:
                flags &= ~0x20
            # With nothing extended left, the simple form of the file has no VP8X
            if not flags:
                continue
            payload = bytes([flags]) + payload[1:]
        out.append((kind, payload))
    return webp_file(out)


def jpeg_segments(data: bytes) -> list[tuple[int, bytes]]:
    """(marker, segment) per segment of the first image of a JPEG in file order, a segment with its marker and its
    length; one of marker 0DA holds a whole scan, its header and its data, which ends at the next marker. What
    follows the image's end marker is left out, as MPF and gain-map JPEGs keep a second image there. ValueError for
    data that is no JPEG, or that ends before its first scan or before its end marker."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    out = []
    at = 2
    scanned = False
    while True:
        if at + 2 > len(data):
            raise _cut_short(scanned)
        if data[at] != 0xFF:
            raise ValueError(f"no marker at byte {at}")
        marker = data[at + 1]
        if marker == 0xFF:
            # A fill byte before the marker
            at += 1
            continue
        if marker == 0xD9:
            if not scanned:
                raise _cut_short(scanned)
            # The image ends here; what follows is not part of it
            return out
        if marker == 0xD8 and scanned:
            # Another image starts where this one ended without its end marker
            raise _cut_short(scanned)
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            out.append((marker, data[at : at + 2]))
            at += 2
            continue
        if at + 4 > len(data):
            raise _cut_short(scanned)
        (length,) = struct.unpack(">H", data[at + 2 : at + 4])
        end = at + 2 + length
        if end > len(data):
            raise _cut_short(scanned)
        if marker == 0xDA:
            end, scanned = _scan_end(data, end), True
        out.append((marker, data[at:end]))
        at = end


def strip_jpeg(data: bytes) -> bytes:
    """A JPEG without the segments that hold settings: every APPn but JFIF's APP0, an APP2 colour profile and
    Adobe's APP14, and the comments, before the first scan and between the scans of a progressive image alike. The
    picture stays byte for byte, and so does its colour profile. What the source keeps after the image's end marker,
    a second image or padding, is left out."""
    kept = b"".join(segment for marker, segment in jpeg_segments(data) if _segment_stays(marker, segment))
    return data[:2] + kept + JPEG_END


def _cut_short(scanned: bool) -> ValueError:
    """The error for a JPEG that ends early: before its first scan, or before its end marker once a scan began."""
    return ValueError("the JPEG ends before its end marker" if scanned else "the JPEG ends before its first scan")


def _scan_end(data: bytes, at: int) -> int:
    """Where the scan data starting at at ends: at the FF of the marker after it. In the data, FF is followed by 00,
    which makes it a data byte, or by D0 to D7, a restart marker; both are the scan's. FF FF is a fill byte before a
    marker, and no data. ValueError for data that ends inside the scan."""
    while True:
        found = data.find(b"\xff", at)
        if found < 0 or found + 1 == len(data):
            raise ValueError("the JPEG ends before its end marker")
        code = data[found + 1]
        if code == 0x00 or 0xD0 <= code <= 0xD7:
            at = found + 2
        elif code == 0xFF:
            at = found + 1
        else:
            return found


def _segment_stays(marker: int, segment: bytes) -> bool:
    return (
        marker not in (0xFE, *range(0xE1, 0xF0))
        or marker == 0xEE
        or (marker == 0xE2 and segment[4:16] == b"ICC_PROFILE\0")
    )


def itxt(key: str, text: str) -> bytes:
    """An uncompressed iTXt chunk, which holds UTF-8 where tEXt holds only Latin-1. ValueError for a keyword PNG
    does not allow or text that is not Unicode."""
    if not valid_keyword(key):
        raise ValueError(f"{key!r} is not a PNG keyword")
    try:
        body = text.encode()
    except UnicodeEncodeError as e:
        raise ValueError(f"the text of {key} is not valid Unicode") from e
    return png_chunk(b"iTXt", key.encode("latin1") + b"\0" * 5 + body)


def valid_keyword(key: str) -> bool:
    """1 to 79 printable Latin-1 characters, without spaces at either end or two together, as PNG requires."""
    return (
        0 < len(key) <= 79
        and all(" " <= c <= "~" or "\xa1" <= c <= "\xff" for c in key)
        and key == key.strip(" ")
        and "  " not in key
    )


def size_of_bytes(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


def verify(data: bytes) -> None:
    """Decode every pixel; opening an image reads only its header, which a truncated file still has."""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
