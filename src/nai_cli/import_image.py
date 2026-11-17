"""Settings brought in from a generated image, assigned to the target model the way the official import does."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nai_cli import errors, meta, models, presets, rerun, values, vibe
from nai_cli import request as R

PARTS = ("prompt", "uc", "characters", "settings", "seed", "vibes")
# The official import dialog leaves Vibes off; the Vibes an image keeps are brought along unless left out
DEFAULT_PARTS = ("prompt", "uc", "characters", "vibes")
_BRACKET_CHARS = re.compile(r"[\[\]{}]")
_TIGHT_COMMA = re.compile(r",(?=[^ ])")


@dataclass
class Imported:
    """What the image brings in; None is a setting it does not set, which then comes from elsewhere."""

    model: models.Model
    prompt: str | None = None
    negative: str | None = None
    negative_verbatim: bool | None = None
    official_nsfw: bool | None = None
    uc_preset: str | None = None
    characters: list[R.Character] | None = None
    use_coords: bool | None = None
    quality: str | None = None
    transparent: bool | None = None
    premultiplied_alpha: bool | None = None
    furry: bool | None = None
    auto_text: bool | None = None
    variety: bool | None = None
    legacy_uc: bool | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    cfg_rescale: float | None = None
    noise_schedule: str | None = None
    size: tuple[int, int] | None = None
    # Why size differs from the image's, for the commands that use it
    size_note: str | None = None
    seed: int | None = None
    vibes: list[vibe.VibeSource] | None = None
    normalize_vibes: bool | None = None
    warnings: list[str] = field(default_factory=list)


def parse_parts(value: str | None) -> tuple[str, ...]:
    """--import: a comma-separated list of parts, or all."""
    if not value:
        return DEFAULT_PARTS
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if parts == ["all"]:
        return PARTS
    unknown = [part for part in parts if part not in PARTS]
    if unknown:
        raise errors.NaiError(errors.BAD_ARGS, f"unknown --import parts: {', '.join(unknown)}", choices=[*PARTS, "all"])
    return tuple(parts)


def from_image(
    path: str | Path,
    parts: tuple[str, ...],
    chosen_model: models.Model | None,
    default_model: models.Model,
    configured: Callable[[models.Model], models.Model],
    clean_brackets: bool = False,
    drawn_prompts: bool = False,
    rework: bool = False,
) -> Imported:
    """chosen_model is --model, which the image's model never overrides; drawn_prompts reads the prompts as
    actually sent instead of with their ||a|b|| groups. rework is for enhance and variations, which redo this very
    image and so keep its legacy UC and alpha mode; the import dialog keeps the current ones instead."""
    raw = meta.read_raw(path)
    record = meta.parse_raw(meta.with_drawn_prompts(raw) if drawn_prompts else raw)
    out = Imported(configured(_target_model(record, parts, chosen_model, default_model)))
    clean = _cleaned_brackets if clean_brackets else _sendable
    _import_furry(record, out)
    if "prompt" in parts:
        _import_prompt(record, out, clean)
    if "uc" in parts:
        _import_uc(record, out, clean)
    if "characters" in parts:
        _import_characters(record, out, clean)
    if "settings" in parts:
        _import_settings(record, out, rework)
    if "seed" in parts:
        out.seed = meta.SEED.read(record)
    if "vibes" in parts:
        _import_vibes(Path(path), record, out)
    return out


def _sendable(text: str) -> str:
    reason = values.unsendable(text)
    if reason is not None:
        raise errors.NaiError(errors.BAD_ARGS, f"the image's prompts hold {reason}, which no request can carry")
    return text


def _cleaned_brackets(text: str) -> str:
    """The import dialog's Clean Brackets, which it applies to every prompt: drop [ ] { } and put one space after
    each comma."""
    return _sendable(_TIGHT_COMMA.sub(", ", _BRACKET_CHARS.sub("", text)).replace(" ,", ","))


def _target_model(
    record: dict[str, Any], parts: tuple[str, ...], chosen: models.Model | None, default: models.Model
) -> models.Model:
    # Settings and Vibes belong to the model that made the image, so asking for either switches to it; the default
    # brings Vibes only when the model already is that one
    switching = {"settings"} if parts == DEFAULT_PARTS else {"settings", "vibes"}
    if chosen is not None or not (switching & set(parts)):
        return chosen or default
    image_model = models.MODELS.get(record.get("model") or "")
    return image_model or default


def _import_furry(record: dict[str, Any], out: Imported) -> None:
    # The official import reads furry mode from the prompt whichever parts are chosen
    if out.model.furry_mode and "furry" in record:
        out.furry = bool(record["furry"])


def _import_prompt(record: dict[str, Any], out: Imported, clean: Any) -> None:
    out.prompt = clean(record.get("user_prompt", record.get("prompt") or ""))
    if record.get("auto_text") is False and out.model.auto_text:
        out.auto_text = False


def _import_uc(record: dict[str, Any], out: Imported, clean: Any) -> None:
    verbatim = bool(record.get("negative_verbatim"))
    negative = record["negative"] if verbatim else record.get("user_negative", record.get("negative") or "")
    out.negative = clean(negative)
    out.negative_verbatim = verbatim
    if not verbatim and "official_nsfw" in record:
        out.official_nsfw = bool(record["official_nsfw"])
    out.uc_preset = _uc_preset(record, out.model)


def _uc_preset(record: dict[str, Any], model: models.Model) -> str | None:
    preset = record.get("uc_preset")
    return presets.carry_uc_preset(preset, model) if isinstance(preset, str) else None


def _import_characters(record: dict[str, Any], out: Imported, clean: Any) -> None:
    chars = list(record.get("characters") or [])
    if len(chars) > out.model.max_characters:
        out.warnings.append(
            f"{out.model.name} allows {out.model.max_characters} characters, so "
            f"{len(chars) - out.model.max_characters} of the image's are left out"
        )
        chars = chars[: out.model.max_characters]
    placed = bool(record.get("use_coords")) and all(c["x"] is not None and c["y"] is not None for c in chars)
    out.characters = [
        R.Character(clean(c["prompt"]), clean(c["negative"]), c["x"] if placed else None, c["y"] if placed else None)
        for c in chars
    ]
    out.use_coords = placed


def _import_settings(record: dict[str, Any], out: Imported, rework: bool) -> None:
    """The image's parameters, those the target model has no use for left out with a warning."""
    model = out.model
    quality = record.get("quality")
    if isinstance(quality, str):
        out.quality = quality if quality in presets.quality_tiers(model) else "standard"
    out.transparent = _supported(out, bool(record.get("transparent")), model.transparent, "transparent backgrounds")
    out.variety = _supported(out, bool(record.get("variety")), model.variety_plus, "Variety+")
    if rework:
        if model.transparent and record.get("premultiplied_alpha") is not None:
            out.premultiplied_alpha = bool(record["premultiplied_alpha"])
        out.legacy_uc = _supported(out, bool(record.get("legacy_uc")), model.legacy_uc, "legacy UC")
    for number in meta.NUMERIC_SETTINGS:
        _import_number(record, out, number)
    out.sampler = models.imported_sampler(record.get("sampler"))
    # The official import turns native, which only staging models took, into karras
    schedule = record.get("noise_schedule")
    if schedule == "native":
        schedule = "karras"
    if (model.noise_schedule or model.is_v5) and schedule in models.NOISE_SCHEDULES:
        out.noise_schedule = schedule
    out.size = rerun.rerun_size(record)
    if out.size is not None and out.size != (
        presets.nearest_grid(record["width"]),
        presets.nearest_grid(record["height"]),
    ):
        out.size_note = (
            f"the image is {record['width']}x{record['height']}, above the pixel limit once on the grid, "
            f"so {out.size[0]}x{out.size[1]} is used"
        )
    if record.get("precise_references"):
        out.warnings.append(
            f"the image used {record['precise_references']} precise references, which images do not keep, "
            "so they are left out"
        )


def _import_number(record: dict[str, Any], out: Imported, number: meta.ImageNumber) -> None:
    """Zero is a value like any other; only a missing or unusable one is left to the command line and the config,
    as the official import assigns whatever key the image has."""
    value = number.read(record)
    if value is None:
        return
    if number.feature is not None and not getattr(out.model, number.feature):
        # A model without the feature does not send it, so only a value that would change the image is worth noting
        _supported(out, bool(value), False, number.feature_name)
        return
    setattr(out, number.attr, value)


def _supported(out: Imported, value: bool, supported: bool, feature: str) -> bool:
    if value and not supported:
        out.warnings.append(f"{out.model.name} has no {feature}, so the image's setting is left out")
    return value and supported


def _import_vibes(path: Path, record: dict[str, Any], out: Imported) -> None:
    """The encodings the image keeps, usable only by the model that made them, as the official import restores
    them."""
    stored, used = record.get("stored_vibes", 0), record.get("vibes", 0)
    if stored and out.model.vibe and record.get("model") == out.model.id:
        out.vibes = vibe.from_settings(path)
        if record.get("normalize_vibe_strength") is not None:
            out.normalize_vibes = bool(record["normalize_vibe_strength"])
    elif stored:
        out.warnings.append(
            f"the image's Vibes are encoded for {record.get('model')}, so {out.model.name} cannot use them"
        )
    if used > stored:
        out.warnings.append(f"the image used {used - stored} Vibes its settings do not keep, so they are left out")
