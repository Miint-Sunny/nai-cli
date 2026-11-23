"""Made-up settings written into a copy in place of an image's own: a preset, a template, a text or single fields."""

import json
import random
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from nai_cli import errors, request, stealth, values

# The presets nai-cli brings. 1 fills every field with spaces, which a metadata viewer shows as empty; 2 with a taunt
BUILTIN = {"1": " " * 512, "2": ", ".join(["杂鱼~♥"] * 64)}
# The fields NovelAI writes to an image's PNG text, in its order; the alpha channel hides all but the first
FIELDS = ("Title", *stealth.HIDDEN)
TITLE = "NovelAI generated image"
DEFAULT_UC = (
    "nsfw, lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, "
    "very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, "
    "too many watermarks, negative space, blank page"
)


@dataclass(frozen=True)
class Plan:
    """What the settings of every copy are made of, and the fields --set changes on top."""

    sets: Mapping[str, Any]

    def own(self, read: Callable[[], dict[str, str] | None]) -> dict[str, str] | None:
        """The image's own settings, which only a plan made from them asks for."""
        return None

    def settings(self, own: dict[str, str] | None, size: tuple[int, int]) -> dict[str, str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Text(Plan):
    """Every field holds a text."""

    text: str

    def settings(self, own: dict[str, str] | None, size: tuple[int, int]) -> dict[str, str]:
        return _as_text(_filled(self.text, self.sets), size)


@dataclass(frozen=True)
class Preset(Plan):
    """The fields a preset or a template holds: as text, with Comment as text or as its object."""

    meta: Mapping[str, Any]

    def settings(self, own: dict[str, str] | None, size: tuple[int, int]) -> dict[str, str]:
        meta = deepcopy(self.meta)
        comment = meta.get("Comment")
        if isinstance(comment, str):
            # A preset whose Comment is text: every field starts from that text, then takes the preset's own
            made = _filled(comment, self.sets)
            made.update({key: value for key, value in meta.items() if key != "Comment" and key not in self.sets})
            return _as_text(made, size)
        return _as_text(_made(meta, self.sets, size), size)


@dataclass(frozen=True)
class Own(Plan):
    """The image's own settings with the fields given changed, or NovelAI's defaults where it keeps none."""

    def own(self, read: Callable[[], dict[str, str] | None]) -> dict[str, str] | None:
        return read()

    def settings(self, own: dict[str, str] | None, size: tuple[int, int]) -> dict[str, str]:
        base = opened(own) if own else {}
        if not isinstance(base.get("Comment"), dict):
            base = {}
        return _as_text(_made(base, self.sets, size), size)


def parse_set(item: str) -> tuple[str, Any]:
    """KEY=VALUE as --set takes it, the value read as JSON where it is JSON and as text otherwise; bad_args without
    an = or without a key."""
    key, found, value = item.partition("=")
    if not found or not key.strip():
        raise errors.NaiError(errors.BAD_ARGS, f"--set takes KEY=VALUE: {item}")
    try:
        return key.strip(), values.parse_json(value)
    except ValueError:
        return key.strip(), value


def read_preset(text: str) -> dict[str, Any]:
    """The fields a preset or a template file holds: a JSON object, its keys starting with _ left out and its
    Comment read as the object it holds; bad_args for any other text."""
    try:
        meta = values.parse_json(text)
    except ValueError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"a preset is a JSON object: {e}") from e
    if not isinstance(meta, dict):
        raise errors.NaiError(errors.BAD_ARGS, "a preset is a JSON object")
    return opened({key: value for key, value in meta.items() if not key.startswith("_")})


def opened(meta: dict[str, Any]) -> dict[str, Any]:
    """meta with its Comment, which an image keeps as a JSON text, as the object it holds, where it holds one."""
    try:
        comment = values.parse_json(meta["Comment"])
    except (KeyError, TypeError, ValueError):
        return meta
    return {**meta, "Comment": comment} if isinstance(comment, dict) else meta


def default_comment(width: int, height: int) -> dict[str, Any]:
    """The settings of a NovelAI Diffusion V5 image by default, field by field as its images hold them, so that
    NovelAI's own inspect reads what is written."""
    return {
        "prompt": "",
        "steps": 28,
        "height": height,
        "width": width,
        "scale": 5.0,
        "uncond_scale": 0.0,
        "cfg_rescale": 0.0,
        "seed": request.random_seed(),
        "n_samples": 1,
        "noise_schedule": "karras",
        "legacy_v3_extend": False,
        "reference_information_extracted_multiple": [],
        "reference_strength_multiple": [],
        "v4_prompt": {
            "caption": {"base_caption": "", "char_captions": []},
            "use_coords": False,
            "use_order": True,
            "legacy_uc": False,
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": DEFAULT_UC, "char_captions": []},
            "use_coords": False,
            "use_order": False,
            "legacy_uc": False,
        },
        "director_reference_strengths": None,
        "director_reference_descriptions": None,
        "upscale": None,
        "straight_alpha": True,
        "quality_boost": False,
        "sampler": "k_euler_ancestral",
        "controlnet_strength": 1.0,
        "controlnet_model": None,
        "dynamic_thresholding": False,
        "dynamic_thresholding_percentile": 0.999,
        "dynamic_thresholding_mimic_scale": 10.0,
        "sm": False,
        "sm_dyn": False,
        "skip_cfg_above_sigma": None,
        "skip_cfg_below_sigma": 0.0,
        "deliberate_euler_ancestral_bug": False,
        "prefer_brownian": True,
        "cfg_sched_eligibility": "enable_for_post_summer_samplers",
        "uncond_per_vibe": True,
        "wonky_vibe_correlation": True,
        "stream": "msgpack",
        "tag_hint_transparent_background": None,
        "tag_hint_uc_preset": 2,
        "tag_hint_qt": 1,
        "legacy": False,
        "color_correct": False,
        "version": 1,
        "uc": DEFAULT_UC,
        "request_type": "PromptGenerateRequest",
        "model_name": "NovelAI Diffusion V5",
        "model_hash": "0ADF9AB7",
    }


def _filled(text: str, sets: Mapping[str, Any]) -> dict[str, Any]:
    """Every field holding text, with the fields given changed. A field of Comment's given makes Comment an object
    that holds it, with the text for both prompts."""
    meta: dict[str, Any] = dict.fromkeys(FIELDS, text)
    inner = {}
    for key, value in sets.items():
        if key in FIELDS:
            meta[key] = value
        else:
            inner[key] = value
    if inner:
        meta["Comment"] = {"prompt": text, "uc": text, **inner}
    return meta


def _made(base: Mapping[str, Any], sets: Mapping[str, Any], size: tuple[int, int]) -> dict[str, Any]:
    """base, a preset's or an image's fields with Comment as its object, completed as NovelAI completes the settings
    of an image, with the fields given changed and without its signature."""
    comment = base.get("Comment")
    if not isinstance(comment, dict):
        comment = default_comment(*size)
    meta: dict[str, Any] = {
        "Title": TITLE,
        "Software": "NovelAI",
        "Source": f"{comment.get('model_name') or 'NovelAI Diffusion V5'} {comment.get('model_hash') or '0ADF9AB7'}",
        **base,
        "Comment": comment,
    }
    if "Generation time" not in meta:
        meta["Generation time"] = f"{random.uniform(2, 9):.13f}"
    for key, value in sets.items():
        if key in FIELDS:
            meta[key] = value
        elif key in ("prompt", "uc"):
            _prompt(comment, key, "v4_prompt" if key == "prompt" else "v4_negative_prompt", str(value))
            if key == "prompt":
                meta["Description"] = str(value)
        else:
            comment[key] = value
    meta.setdefault("Description", comment.get("prompt", ""))
    comment.pop("signed_hash", None)
    return meta


def _prompt(comment: dict[str, Any], key: str, prompt: str, value: str) -> None:
    """A prompt changed where it is typed and where V4 and later models read it from."""
    comment[key] = value
    caption = comment.setdefault(prompt, {}).setdefault("caption", {})
    caption["base_caption"] = value
    caption.setdefault("char_captions", [])


def _as_text(meta: dict[str, Any], size: tuple[int, int]) -> dict[str, str]:
    """The settings of one image's copy: every field as the text of a PNG text chunk, Comment as JSON unless it is
    text already; of the image's size, and with a seed of its own where none is given."""
    comment = meta.get("Comment")
    if isinstance(comment, dict):
        comment["width"], comment["height"] = size
        if comment.get("seed") is None:
            comment["seed"] = request.random_seed()
    out: dict[str, str] = {}
    for key in FIELDS:
        if key in meta:
            value = meta[key]
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False) if key == "Comment" else values.to_json(value)
            out[key] = value
    return out
