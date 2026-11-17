"""Read and write .naiv4vibe files; user files are never modified."""

import base64
import binascii
import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from nai_cli import decode, errors, images, models, resample, values
from nai_cli.meta import model_of, png_text_entries, read_raw
from nai_cli.request import check_range

IDENTIFIER = "novelai-vibe-transfer"
BUNDLE_IDENTIFIER = "novelai-vibe-transfer-bundle"
VERSION = 1
MODEL_KEYS = {
    "nai-diffusion-4-5-full": "v4-5full",
    "nai-diffusion-4-5-curated": "v4-5curated",
    "nai-diffusion-4-full": "v4full",
    "nai-diffusion-4-curated-preview": "v4curated",
}
DEFAULT_STRENGTH = 0.6
# Default information extracted: 0.7 for V4.5 Full, 1 otherwise
_DEFAULT_INFORMATION_EXTRACTED = {"nai-diffusion-4-5-full": 0.7}
_FILE_SUFFIXES = (".naiv4vibe", ".naiv4vibebundle", ".json")
# The official client exports Vibes into PNG images under this text chunk, as base64 JSON
_PNG_KEYWORD = "naidata"
_ID = re.compile(r"[0-9a-f]{64}")
_THUMBNAIL = re.compile(r"data:image/[^;]+;base64,")
# The official client stores a JPEG of this longest side at quality 0.8 with each image Vibe
_THUMBNAIL_SIDE = 256
_THUMBNAIL_QUALITY = 80
# The fields the official export writes, in its order; it drops any others
_ENTRY_FIELDS = (
    "identifier",
    "version",
    "type",
    "image",
    "id",
    "encodings",
    "name",
    "thumbnail",
    "createdAt",
    "importInfo",
)
_PARAMS_FIELDS = ("mask", "information_extracted")


def default_information_extracted(model: str) -> float:
    return _DEFAULT_INFORMATION_EXTRACTED.get(model, 1.0)


@dataclass
class VibeSource:
    name: str
    image_b64: str | None
    encodings: dict[str, dict[str, dict[str, Any]]]
    strength: float
    information_extracted: float
    # Only carried over from imported files; the official client has no way to draw one
    mask: Any = None
    # As read from a file, so an export keeps its id, name and thumbnail
    entry: dict[str, Any] | None = None


def params_key(information_extracted: float, mask: Any = None) -> str:
    # Parameters sorted by name, unset ones dropped, as the official client hashes them
    params = {"information_extracted": information_extracted, "mask": mask}
    canonical = ",".join(f"{k}:{_js_text(v)}" for k, v in sorted(params.items()) if v is not None)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load(path: Path, strength: float | None, information_extracted: float | None, model: str) -> list[VibeSource]:
    """Load a .naiv4vibe or .naiv4vibebundle, an image carrying one, or a plain image to encode later.
    Unset values come from the file, then the defaults; a bundle applies them to every Vibe."""
    if path.suffix.lower() in _FILE_SUFFIXES:
        try:
            data = values.parse_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise errors.NaiError(errors.BAD_ARGS, f"cannot read Vibe file {path}: {e}") from e
        return _sources(data, path, strength, information_extracted, model)
    embedded = _embedded(path)
    if embedded is not None:
        return _sources(embedded, path, strength, information_extracted, model)
    png = images.to_png(images.open_image(path))
    return [
        _checked(
            VibeSource(
                name=path.stem,
                image_b64=base64.b64encode(png).decode(),
                encodings={},
                strength=DEFAULT_STRENGTH if strength is None else strength,
                information_extracted=default_information_extracted(model)
                if information_extracted is None
                else information_extracted,
            )
        )
    ]


def _embedded(path: Path) -> dict[str, Any] | None:
    try:
        # The official import reads the first, which an export into the image puts right after IHDR
        text = next((t for key, t in png_text_entries(path.read_bytes()) if key == _PNG_KEYWORD), None)
    except OSError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    if text is None:
        return None
    try:
        data = values.parse_json(base64.b64decode(text))
    except (binascii.Error, ValueError):
        return None
    if isinstance(data, dict) and data.get("identifier") in (IDENTIFIER, BUNDLE_IDENTIFIER):
        return data
    return None


def _sources(
    data: Any, path: Path, strength: float | None, information_extracted: float | None, model: str
) -> list[VibeSource]:
    identifier = data.get("identifier") if isinstance(data, dict) else None
    if identifier == IDENTIFIER:
        return [_source(data, path.stem, strength, information_extracted, model)]
    if identifier == BUNDLE_IDENTIFIER:
        vibes = data.get("vibes")
        if not isinstance(vibes, list) or not vibes or not all(isinstance(v, dict) for v in vibes):
            raise errors.NaiError(errors.BAD_ARGS, f"{path.name} holds no Vibes")
        return [_source(v, f"{path.stem} {i}", strength, information_extracted, model) for i, v in enumerate(vibes, 1)]
    raise errors.NaiError(errors.BAD_ARGS, f"{path.name} is not a Vibe file")


def _source(
    data: dict[str, Any], name: str, strength: float | None, information_extracted: float | None, model: str
) -> VibeSource:
    info = data.get("importInfo")
    info = info if isinstance(info, dict) else {}
    name = data.get("name") or name
    mask = info.get("mask")
    if mask is not None and not _is_base64(mask):
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} has an invalid mask")
    image = _image(data.get("image"), name)
    encodings = _encodings(data.get("encodings"), name)
    _check_id(data, name, encodings)
    stored_info = _stored(info, "information_extracted", default_information_extracted(model), name)
    # The official client locks the slider of a Vibe without its image, since it cannot be encoded again
    if image is None and information_extracted is not None and information_extracted != stored_info:
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"Vibe {name!r} has no source image, so its information extracted stays at {_js_number(stored_info)}",
        )
    return _checked(
        VibeSource(
            name=str(name),
            image_b64=image,
            encodings=encodings,
            strength=strength if strength is not None else _stored(info, "strength", DEFAULT_STRENGTH, name),
            information_extracted=stored_info if information_extracted is None else information_extracted,
            mask=mask,
            entry=data,
        )
    )


def _check_id(data: dict[str, Any], name: str, encodings: dict[str, dict[str, dict[str, Any]]]) -> None:
    """The official import refuses a Vibe whose id does not match its image, or its first encoding."""
    if data.get("type") not in ("image", "encoding"):
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} is neither an image nor an encoding Vibe")
    vibe_id = data.get("id")
    if not isinstance(vibe_id, str) or not _ID.fullmatch(vibe_id):
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} has no valid id")
    if data.get("type") == "image" and _sha256(str(data.get("image") or "")) != vibe_id:
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r}: its image does not match its id")
    if data.get("type") == "encoding":
        first = next((v for by_model in encodings.values() for v in by_model.values()), None)
        if first is None:
            raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} holds no encoding")
        if _sha256(first.get("encoding") or "") != vibe_id:
            raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r}: its encoding does not match its id")
    thumbnail = data.get("thumbnail")
    if thumbnail and not (isinstance(thumbnail, str) and _THUMBNAIL.match(thumbnail)):
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} has an invalid thumbnail")


def from_settings(path: str | Path) -> list[VibeSource]:
    """The Vibes kept in a generated image's settings, as the official import restores them: encodings for the
    model that made the image, with their strengths and information extracted."""
    path = Path(path)
    raw = read_raw(path)
    try:
        settings = values.parse_json(raw["Comment"])
    except (KeyError, ValueError):
        settings = None
    if not isinstance(settings, dict):
        raise errors.NaiError(errors.BAD_ARGS, f"{path.name} has no generation settings")
    encodings = settings.get("reference_image_multiple")
    if not isinstance(encodings, list) or not encodings:
        raise errors.NaiError(errors.BAD_ARGS, f"{path.name} keeps no Vibes in its settings")
    model = model_of(settings, raw.get("Source")) or ""
    key = MODEL_KEYS.get(model)
    if key is None:
        raise errors.NaiError(errors.BAD_ARGS, f"{path.name} was not made with a model that has Vibes")
    out = []
    for i, encoding in enumerate(encodings):
        name = f"{path.stem} {i + 1}"
        if not encoding or not is_encoding(encoding):
            raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} keeps an invalid encoding")
        strength = _listed(settings, "reference_strength_multiple", i, DEFAULT_STRENGTH, name)
        info = _listed(
            settings, "reference_information_extracted_multiple", i, default_information_extracted(model), name
        )
        out.append(_checked(VibeSource(name, None, {key: {"unknown": {"encoding": encoding}}}, strength, info)))
    return out


def _listed(settings: dict[str, Any], key: str, index: int, default: float, name: str) -> float:
    values = settings.get(key)
    value = values[index] if isinstance(values, list) and index < len(values) else None
    return _stored({key: value}, key, default, name)


def _image(value: Any, name: str) -> str | None:
    """The source image in base64, checked on load so a broken file fails before anything is paid for."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} holds an image that cannot be read")
    # Line breaks are allowed, as browsers' base64 readers allow them
    value = "".join(value.split())
    try:
        decode.size_of_bytes(base64.b64decode(value, validate=True))
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} holds an image that cannot be read") from e
    return value


def check_image(source: VibeSource) -> None:
    """Decode the whole source image before paying to encode it; loading a file checked only its header."""
    if source.image_b64 is None:
        return
    try:
        decode.verify(base64.b64decode(source.image_b64))
    except Image.DecompressionBombError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {source.name!r} holds an image past the pixel limit") from e
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {source.name!r} holds a damaged image") from e


def is_encoding(value: Any) -> bool:
    """Text a request can carry as an encoding."""
    return isinstance(value, str) and values.unsendable(value) is None


def _encodings(value: Any, name: str) -> dict[str, dict[str, dict[str, Any]]]:
    """{model key: {parameters key: {encoding, ...}}}; the official import rejects any other shape."""
    if value is None:
        return {}
    well_formed = isinstance(value, dict) and all(
        isinstance(by_model, dict)
        and all(isinstance(v, dict) and is_encoding(v.get("encoding", "")) for v in by_model.values())
        for by_model in value.values()
    )
    if not well_formed:
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} has malformed encodings")
    return value


def _stored(info: dict[str, Any], key: str, default: float, name: str) -> float:
    value = info.get(key)
    # A missing or null value takes the default, as in the official import
    if value is None:
        return default
    if not values.is_number(value):
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {name!r} stores an invalid {key}: {value!r}")
    return float(value)


def _checked(source: VibeSource) -> VibeSource:
    # The official client lets strength go past its slider, so it only has to be a finite number
    if not math.isfinite(source.strength):
        raise errors.NaiError(errors.BAD_ARGS, f"Vibe {source.name!r} has an invalid strength: {source.strength}")
    check_range("information extracted", source.information_extracted, models.INFORMATION_EXTRACTED_RANGE)
    return source


def stored_encoding(source: VibeSource, model: str) -> str | None:
    """The encoding the Vibe keeps for these settings; None when its image is to be looked up or encoded."""
    by_model = source.encodings.get(MODEL_KEYS.get(model, ""), {})
    wanted = by_model.get(params_key(source.information_extracted, source.mask))
    if wanted and wanted.get("encoding"):
        return wanted["encoding"]
    if source.image_b64 is None:
        # Without the source image only the stored information-extracted value can be used
        for variant in by_model.values():
            if variant.get("encoding"):
                return variant["encoding"]
        raise errors.NaiError(
            errors.BAD_ARGS, f"Vibe {source.name!r} has no encoding for {model} and no image to re-encode"
        )
    return None


def _is_base64(value: Any) -> bool:
    if not isinstance(value, str) or not value.isascii():
        return False
    try:
        # Line breaks are allowed, as for the image
        base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, TypeError, ValueError):
        return False
    return True


def encoding_key(source: VibeSource, model: str) -> str:
    """What one encoding serves: the image, the model, the information extracted and the mask."""
    key = f"{source.image_b64}|{model}|{_js_number(source.information_extracted)}"
    if source.mask is not None:
        key += f"|{_js_text(source.mask)}"
    return key


def _js_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _js_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return _js_number(value)
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return str(value)


@dataclass(frozen=True)
class Encode:
    """One encoding to pay for, and the positions of the Vibes it serves."""

    source: VibeSource
    positions: tuple[int, ...]


def encodes(pending: list[tuple[int, VibeSource]], model: str) -> list[Encode]:
    """The Vibes still to encode, one paid encoding for each set of Vibes that would get the same one."""
    groups: dict[str, list[tuple[int, VibeSource]]] = {}
    for i, source in pending:
        groups.setdefault(encoding_key(source, model), []).append((i, source))
    return [Encode(group[0][1], tuple(i for i, _ in group)) for group in groups.values()]


def thumbnail(png: bytes) -> str:
    """The image scaled to 256 on its longer side and saved as a JPEG data URL, transparency on black, as the
    official client keeps it with an image Vibe."""
    with Image.open(io.BytesIO(png)) as img:
        img.load()
        scale = _THUMBNAIL_SIDE / max(img.size)
        small = resample.lanczos(img.convert("RGBA"), math.floor(img.width * scale), math.floor(img.height * scale))
    out = io.BytesIO()
    resample.fill_background(small, (0, 0, 0)).save(out, "JPEG", quality=_THUMBNAIL_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def export(source: VibeSource, model: str, encoding: str | None, without_image: bool) -> dict[str, Any]:
    """One Vibe as the official export writes it: the stored entry, any new encoding, and this use's settings
    as importInfo. without_image keeps only the encoding for this model and settings, as the official export
    can."""
    if source.entry is not None:
        data = dict(source.entry)
    else:
        image_b64 = source.image_b64 or ""
        data = {
            "identifier": IDENTIFIER,
            "version": VERSION,
            "type": "image",
            "image": image_b64,
            "id": _sha256(image_b64),
            "encodings": {},
            "name": source.name,
            "thumbnail": thumbnail(base64.b64decode(image_b64)),
            "createdAt": int(time.time() * 1000),
        }
    key, params = MODEL_KEYS[model], params_key(source.information_extracted, source.mask)
    encodings = {m: dict(variants) for m, variants in (data.get("encodings") or {}).items()}
    if encoding is not None and data.get("type") == "image":
        settings = {"information_extracted": source.information_extracted}
        if source.mask is not None:
            settings["mask"] = source.mask
        encodings.setdefault(key, {})[params] = {"encoding": encoding, "params": settings}
    data["encodings"] = encodings
    info = {"model": model, "information_extracted": source.information_extracted, "strength": source.strength}
    if source.mask is not None:
        info["mask"] = source.mask
    data["importInfo"] = info
    if without_image and data.get("type") == "image":
        kept = encodings.get(key, {}).get(params)
        if not kept or not kept.get("encoding"):
            raise errors.NaiError(errors.BAD_ARGS, f"Vibe {source.name!r} has no encoding for {model} to keep")
        data = {**data, "id": _sha256(kept["encoding"]), "type": "encoding", "encodings": {key: {"unknown": kept}}}
        data.pop("image", None)
        data.pop("thumbnail", None)
    return _official_fields(data)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _official_fields(data: dict[str, Any]) -> dict[str, Any]:
    out = {k: data[k] for k in _ENTRY_FIELDS if data.get(k) is not None}
    out["encodings"] = {
        model: {params: _variant(v) for params, v in variants.items()} for model, variants in out["encodings"].items()
    }
    return out


def _variant(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"encoding": value.get("encoding")}
    params = value.get("params")
    if isinstance(params, dict):
        out["params"] = {k: params[k] for k in _PARAMS_FIELDS if params.get(k) is not None}
    return out


def bundle(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"identifier": BUNDLE_IDENTIFIER, "version": VERSION, "vibes": entries}


def dumps(data: dict[str, Any]) -> str:
    """Indented by two spaces, as the official export writes it."""
    return values.to_json(_js_values(data), indent=2)


def _js_values(value: Any) -> Any:
    # JSON.stringify writes 1.0 as 1
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _js_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_js_values(v) for v in value]
    return value


def embed(png: bytes, text: str) -> bytes:
    """The PNG with text in an uncompressed naidata iTXt chunk right after IHDR, as base64 of its UTF-8, the way
    the official export writes a bundle into an image. Bundles it held before are dropped."""
    chunk = decode.itxt(_PNG_KEYWORD, base64.b64encode(text.encode()).decode("ascii"))
    chunks = list(decode.png_chunks(png))
    if not chunks or chunks[0][0] != b"IHDR":
        raise errors.NaiError(errors.BAD_ARGS, "the image to hold the Vibes is not a PNG")
    kept = [png[start:end] for kind, data, start, end in chunks[1:] if not _holds_bundle(kind, data)]
    return png[: chunks[0][3]] + chunk + b"".join(kept)


def _holds_bundle(kind: bytes, data: bytes) -> bool:
    return kind in decode.TEXT_CHUNKS and data.split(b"\0", 1)[0] == _PNG_KEYWORD.encode("latin1")
