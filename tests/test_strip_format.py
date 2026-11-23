"""nai strip --keep-format: a copy in its source's format, losing only its metadata where the format lets it, and
every strip copy read back before any is written."""

import io
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from helpers import run
from PIL import Image
from samples import in_low_bits, nai_image, settings, stealth_payload

from nai_cli import decode, errors, images, meta, stealth
from nai_cli.commands import strip

ICC = b"a colour profile, kept where the copy is not encoded again"
ADOBE = b"Adobe\x00\x64\x00\x00\x00\x00\x00"


def _exif() -> bytes:
    """EXIF as NovelAI's WebP keeps settings: the whole metadata in UserComment, with Comment inside it."""
    exif = Image.Exif()
    exif[0x010E] = "the prompt, kept in the description"
    exif.get_ifd(0x8769)[0x9286] = b"ASCII\0\0\0" + json.dumps(settings(prompt="secret")).encode()
    return exif.tobytes()


def _lossy(transparent: bool) -> Image.Image:
    """A picture for a lossy WebP: opaque to the eye with an alpha channel, as NovelAI's lossy WebP is, unless
    transparent."""
    img = Image.new("RGBA", (64, 64), (90, 140, 200, 255))
    img.putpixel((3, 3), (90, 140, 200, 128 if transparent else 254))
    return img


def _save(img: Image.Image, path: Path, fmt: str, **options) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, fmt, **options)
    return path


def _jpeg(img: Image.Image, **options) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", **options)
    return buf.getvalue()


def _make(path: Path) -> Path:
    """An image of the format its suffix names, with EXIF where the format takes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = _exif()
    match path.suffix.lower():
        case ".png":
            nai_image(path, w=64, h=64)
            return path
        case ".webp":
            return _save(_lossy(transparent=False), path, "WEBP", quality=90, exif=exif, icc_profile=ICC)
        case ".jpeg" | ".jpg":
            source = _save(
                _lossy(transparent=False).convert("RGB"), path, "JPEG", quality=90, exif=exif, icc_profile=ICC
            )
            # An APP14 segment, which decoding a colour depends on: written by hand, as Pillow only writes it for CMYK
            data = path.read_bytes()
            path.write_bytes(data[:2] + b"\xff\xee" + struct.pack(">H", 2 + len(ADOBE)) + ADOBE + data[2:])
            return source
        case _:
            return _save(Image.new("RGB", (64, 64), "gray"), path, "GIF")


def _kinds(path: Path) -> list[bytes]:
    return [kind for kind, _ in decode.webp_chunks(path.read_bytes())]


def _holds_nothing(path: Path) -> None:
    with pytest.raises(errors.NaiError) as refused:
        meta.parse(path)
    assert refused.value.code == errors.NO_METADATA
    with Image.open(path) as img:
        assert not img.getexif() and not img.info.get("xmp")
        assert stealth.find(images.rgb_or_rgba(img), math.inf) is None


def test_a_png_copy_is_what_it_was(capsys, tmp_path):
    source = _make(tmp_path / "in" / "a.png")
    out = tmp_path / "out"
    code, kept = run(capsys, "strip", str(source), "--keep-format", "-o", str(out))
    assert code == 0 and Path(kept["files"][0]["file"]).name == "a.png"
    _holds_nothing(Path(kept["files"][0]["file"]))
    code, plain = run(capsys, "strip", str(source), "-o", str(out))
    assert code == 0 and Path(plain["files"][0]["file"]).read_bytes() == Path(kept["files"][0]["file"]).read_bytes()


def test_a_lossless_webp_stays_lossless(capsys, tmp_path):
    source = _save(
        Image.new("RGBA", (64, 64), (10, 20, 30, 255)), tmp_path / "in" / "a.webp", "WEBP", lossless=True, exif=_exif()
    )
    out = tmp_path / "out"
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(out))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and _kinds(copy) == [b"VP8L"], copies
    assert np.array_equal(np.asarray(Image.open(copy)), np.asarray(Image.open(source)))
    _holds_nothing(copy)


def test_a_lossy_webp_without_transparency_loses_only_its_chunks(capsys, tmp_path):
    source = _make(tmp_path / "in" / "a.webp")
    assert {b"VP8X", b"ICCP", b"ALPH", b"EXIF"} <= set(_kinds(source))
    out = tmp_path / "out"
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(out))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and copies.get("warnings", []) == [], copies
    # The alpha chunk goes with the stealth data it may hold; the colour profile stays, so VP8X stays for it alone
    assert _kinds(copy) == [b"VP8X", b"ICCP", b"VP8 "]
    assert dict(decode.webp_chunks(copy.read_bytes()))[b"VP8X"][0] == 0x20
    assert (
        dict(decode.webp_chunks(copy.read_bytes()))[b"VP8 "] == dict(decode.webp_chunks(source.read_bytes()))[b"VP8 "]
    )
    with Image.open(copy) as img:
        assert img.mode == "RGB" and img.info["icc_profile"] == ICC
    _holds_nothing(copy)


def test_a_lossy_webp_with_transparency_is_encoded_again_with_a_warning(capsys, tmp_path):
    source = _save(
        _lossy(transparent=True), tmp_path / "in" / "a.webp", "WEBP", quality=90, exif=_exif(), icc_profile=ICC
    )
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and [w for w in copies["warnings"] if "a.webp" in w and "loss" in w], copies
    assert {b"ALPH", b"VP8 "} <= set(_kinds(copy)) and not {b"EXIF", b"VP8L", b"ICCP"} & set(_kinds(copy))
    with Image.open(copy) as img:
        assert img.mode == "RGBA" and img.getchannel("A").getextrema()[0] == 128
    _holds_nothing(copy)


def test_an_animated_webp_is_cleaned_frame_by_frame_and_stays_animated(capsys, tmp_path):
    frames = [
        in_low_bits(Image.new("RGBA", (64, 64), (40 * i, 0, 0, 255)), stealth_payload(settings())) for i in range(3)
    ]
    source = _save(
        frames[0],
        tmp_path / "in" / "a.webp",
        "WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=[100, 200, 300],
        loop=0,
        lossless=True,
        exif=_exif(),
    )
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and {b"ANIM"} <= set(_kinds(copy)) and b"EXIF" not in _kinds(copy), copies
    # Lossless: every frame holds a VP8L picture, and no frame a lossy VP8 one
    frames_kept = [payload for kind, payload in decode.webp_chunks(copy.read_bytes()) if kind == b"ANMF"]
    assert len(frames_kept) == 3 and all(b"VP8L" in frame and b"VP8 " not in frame for frame in frames_kept)
    with Image.open(copy) as img:
        assert img.n_frames == 3 and img.info["loop"] == 0
        durations = []
        for i in range(img.n_frames):
            img.seek(i)
            img.load()
            durations.append(img.info["duration"])
            assert stealth.find(img.convert("RGBA"), math.inf) is None
        assert durations == [100, 200, 300]


def test_a_jpeg_loses_only_its_segments(capsys, tmp_path):
    source = _save(
        _lossy(transparent=False).convert("RGB"),
        tmp_path / "in" / "a.jpg",
        "JPEG",
        quality=90,
        exif=_exif(),
        icc_profile=ICC,
        comment=b"a comment to go",
    )
    data = source.read_bytes()
    source.write_bytes(data[:2] + b"\xff\xee" + struct.pack(">H", 2 + len(ADOBE)) + ADOBE + data[2:])
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and copy.name == "a.jpg", copies
    kept = copy.read_bytes()
    assert b"secret" not in kept and b"a comment to go" not in kept
    assert b"ICC_PROFILE" in kept and ADOBE in kept
    # The scan is the source's, byte for byte
    scan = lambda path: [segment for marker, segment in decode.jpeg_segments(path.read_bytes()) if marker == 0xDA]  # noqa: E731
    assert scan(copy) == scan(source)
    _holds_nothing(copy)


def test_a_jpeg_copy_ends_at_its_own_end_holding_nothing_of_the_image_after_it(capsys, tmp_path):
    # As the MPF of a phone's JPEG and a gain map's do: a second image, with EXIF of its own, after the first EOI
    first = _jpeg(Image.new("RGB", (64, 64), (200, 50, 50)))
    second = _jpeg(Image.new("RGB", (32, 32), (50, 50, 200)), exif=_exif())
    source = tmp_path / "in" / "a.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(first + second)
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and copy.name == "a.jpg", copies
    kept = copy.read_bytes()
    scan, their_scan = kept[kept.index(b"\xff\xda") :], second[second.index(b"\xff\xda") :]
    # One end marker ends the copy, and nothing of the second image is in it
    assert scan.count(b"\xff\xd9") == 1 and scan.endswith(b"\xff\xd9"), scan.count(b"\xff\xd9")
    assert b"secret" not in kept and their_scan not in kept
    # The first image's scan is the source's, byte for byte, and the copy decodes to that image alone
    assert scan == first[first.index(b"\xff\xda") :]
    with Image.open(copy) as img, Image.open(io.BytesIO(first)) as original:
        assert img.size == (64, 64) and np.array_equal(np.asarray(img), np.asarray(original))
    _holds_nothing(copy)


def test_a_webp_copy_leaves_out_what_lies_past_the_size_its_header_names(capsys, tmp_path):
    source = _make(tmp_path / "in" / "a.webp")
    webp = source.read_bytes()
    assert len(webp) == 8 + struct.unpack("<I", webp[4:8])[0]
    # A chunk past the RIFF size, where no reader of the file looks
    body = b"TRAILING-WEBP-MARKER"
    source.write_bytes(webp + b"JUNK" + struct.pack("<I", len(body)) + body)
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0, copies
    kept = copy.read_bytes()
    assert body not in kept
    assert 8 + struct.unpack("<I", kept[4:8])[0] == len(kept)
    # The picture is the source's, byte for byte, with its colour profile
    assert dict(decode.webp_chunks(kept))[b"VP8 "] == dict(decode.webp_chunks(webp))[b"VP8 "]
    with Image.open(copy) as img:
        assert img.info["icc_profile"] == ICC
    _holds_nothing(copy)


def test_a_progressive_jpeg_loses_what_lies_between_its_scans(capsys, tmp_path):
    source = tmp_path / "in" / "a.jpg"
    source.parent.mkdir(parents=True)
    data = _jpeg(Image.new("RGB", (64, 64), (200, 50, 50)), progressive=True, exif=_exif(), icc_profile=ICC)
    second_scan = data.index(b"\xff\xda", data.index(b"\xff\xda") + 2)
    comment, app1 = b"A comment between two scans", b"AN-APP1-BETWEEN-SCANS"
    between = b"\xff\xfe" + struct.pack(">H", 2 + len(comment)) + comment
    between += b"\xff\xe1" + struct.pack(">H", 2 + len(app1)) + app1
    source.write_bytes(data[:second_scan] + between + data[second_scan:])
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0, copies
    kept = copy.read_bytes()
    assert comment not in kept and app1 not in kept and b"secret" not in kept
    # Still progressive, with every scan the source has, each the source's, and decoding to the same picture
    assert b"\xff\xc2" in kept and kept.count(b"\xff\xda") == data.count(b"\xff\xda")
    assert all(segment in data for marker, segment in decode.jpeg_segments(kept) if marker == 0xDA)
    with Image.open(copy) as img, Image.open(source) as original:
        assert np.array_equal(np.asarray(img), np.asarray(original))
    _holds_nothing(copy)


def test_a_scan_holding_stuffed_bytes_and_restart_markers_is_copied_as_it_is(capsys, tmp_path):
    # In a scan, FF 00 is a data byte FF and FF D0 to FF D7 restart markers, which are the scan's, not its end
    noisy = Image.fromarray(np.random.default_rng(1).integers(0, 256, (128, 128, 3), dtype=np.uint8))
    data = _jpeg(noisy, quality=90, restart_marker_rows=1)
    assert b"\xff\x00" in data and b"\xff\xd0" in data
    source = tmp_path / "in" / "a.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    # Nothing in it is to go, so the copy is the source itself
    assert code == 0 and copy.read_bytes() == data, copies


def test_a_jpeg_with_a_fill_byte_before_its_end_marker_still_copies(capsys, tmp_path):
    # One FF more before a marker is padding, which is neither data nor the marker itself
    data = _jpeg(Image.new("RGB", (64, 64), (200, 50, 50)), comment=b"a comment to go")
    source = tmp_path / "in" / "a.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(data[:-2] + b"\xff" + data[-2:])
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and b"a comment to go" not in copy.read_bytes(), copies
    with Image.open(copy) as img, Image.open(source) as original:
        assert np.array_equal(np.asarray(img), np.asarray(original))
    _holds_nothing(copy)


def test_a_jpeg_that_ends_before_its_end_marker_is_refused_by_name(capsys, tmp_path):
    source = tmp_path / "in" / "cut.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(_jpeg(Image.new("RGB", (64, 64), (200, 50, 50)))[:-2])
    code, refused = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    assert (code, refused["code"]) == (1, "bad_args") and "cut.jpg" in refused["message"], refused
    assert not (tmp_path / "out").exists()


def test_an_image_without_its_end_marker_is_not_read_on_into_the_one_after_it():
    # The first image ends without its end marker, so nothing says where it ends: the next image's start is no end
    first = _jpeg(Image.new("RGB", (64, 64), (200, 50, 50)))
    second = _jpeg(Image.new("RGB", (32, 32), (50, 50, 200)), exif=_exif())
    with pytest.raises(ValueError, match="end marker"):
        decode.strip_jpeg(first[:-2] + second)


def test_any_other_format_is_saved_as_png_as_without_the_flag(capsys, tmp_path):
    source = _make(tmp_path / "in" / "a.gif")
    code, copies = run(capsys, "strip", str(source), "--keep-format", "-o", str(tmp_path / "out"))
    copy = Path(copies["files"][0]["file"])
    assert code == 0 and copy.name == "a.png" and Image.open(copy).format == "PNG"


@pytest.mark.parametrize(
    ("source", "given", "options", "status"),
    [
        ("a.webp", "one.png", [], 0),
        ("a.webp", "one.webp", [], 1),
        ("a.webp", "one.webp", ["--keep-format"], 0),
        ("a.webp", "one.png", ["--keep-format"], 1),
        ("a.jpg", "one.jpeg", ["--keep-format"], 0),
        ("a.jpg", "one.png", ["--keep-format"], 1),
        ("a.jpg", "one.gif", ["--keep-format"], 1),
        ("a.png", "one.png", ["--keep-format"], 0),
        ("a.gif", "one.png", ["--keep-format"], 0),
    ],
)
def test_o_takes_a_file_name_ending_as_its_copy_does(capsys, tmp_path, source, given, options, status):
    path = _make(tmp_path / "in" / source)
    code, out = run(capsys, "strip", str(path), "-o", str(tmp_path / given), *options)
    assert code == status, out
    if status:
        assert out["code"] == "bad_args" and not (tmp_path / given).exists()
    else:
        assert Path(out["files"][0]["file"]).name == given


def test_copies_take_the_suffix_of_their_format(capsys, tmp_path):
    folder = tmp_path / "in"
    for name in ("a.png", "b.webp", "c.jpeg"):
        _make(folder / name)
    out = tmp_path / "out"
    code, copies = run(capsys, "strip", str(folder), "--keep-format", "-o", str(out))
    assert code == 0 and [Path(f["file"]).name for f in copies["files"]] == ["a.png", "b.webp", "c.jpeg"]
    code, copies = run(capsys, "strip", str(folder), "--keep-format")
    assert code == 0 and [Path(f["file"]).name for f in copies["files"]] == [
        "a_clean.png",
        "b_clean.webp",
        "c_clean.jpeg",
    ]
    code, copies = run(capsys, "strip", str(folder), "--keep-format", "--rename", "-o", str(out), "-r")
    suffixes = [Path(f["file"]).suffix for f in copies["files"]]
    assert code == 0 and suffixes == [".png", ".webp", ".jpeg"]


def _xmp_webp(path: Path) -> Path:
    return _save(Image.new("RGB", (64, 64), "gray"), path, "WEBP", lossless=True, xmp=b"<x:xmpmeta/>")


@pytest.mark.parametrize(
    ("left", "make"),
    [
        ("generation settings", lambda p: nai_image(p, w=64, h=64)),
        ("generation settings", lambda p: _save(_lossy(transparent=False), p, "WEBP", quality=90, exif=_exif())),
        (
            "stealth data",
            lambda p: in_low_bits(
                Image.new("RGBA", (96, 96), (10, 20, 30, 255)), stealth_payload({"Software": "another tool"})
            ).save(p),
        ),
        ("EXIF", lambda p: Image.new("RGB", (8, 8)).save(p, exif=_exif())),
        ("XMP", _xmp_webp),
    ],
    ids=["png text", "webp exif", "stealth of another tool", "exif", "xmp"],
)
def test_the_read_back_names_what_a_copy_would_still_hold(tmp_path, left, make):
    copy = tmp_path / "copy.png"
    make(copy)
    with pytest.raises(errors.NaiError) as refused:
        strip.checked(copy.read_bytes(), "copy.png")
    assert refused.value.code == errors.BAD_ARGS, refused.value.message
    assert "copy.png" in refused.value.message and left in refused.value.message


def test_the_read_back_passes_a_clean_copy(tmp_path):
    clean = images.strip_file(_make(tmp_path / "a.png"))
    assert strip.checked(clean, "a_clean.png") is clean


def test_nothing_is_written_when_a_copy_would_still_hold_settings(capsys, tmp_path, monkeypatch):
    clean, dirty = _make(tmp_path / "a.png"), _make(tmp_path / "b.png")
    with_settings = tmp_path / "settings.png"
    nai_image(with_settings, w=64, h=64)
    strip_file = images.strip_file
    monkeypatch.setattr(
        images, "strip_file", lambda path: with_settings.read_bytes() if Path(path) == dirty else strip_file(path)
    )
    out = tmp_path / "out"
    for options in ([], ["--keep-format"]):
        code, refused = run(capsys, "strip", str(clean), str(dirty), "-o", str(out), *options)
        assert (code, refused["code"]) == (1, "bad_args") and "b.png" in refused["message"], refused
    assert not out.exists()
