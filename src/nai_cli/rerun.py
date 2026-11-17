"""Generation settings read from an image turned back into nai gen arguments that rerun it."""

from pathlib import Path
from typing import Any

from nai_cli import presets, tidy, values
from nai_cli.meta import NUMERIC_SETTINGS, SEED
from nai_cli.models import MODELS, NOISE_SCHEDULES, effort_of, imported_sampler, resolve, with_hidden_options

_ON_SWITCHES = (("furry", "--furry"), ("variety", "--variety"), ("legacy_uc", "--legacy-uc"))
_FEATURES = {"transparent": "transparent", "furry": "furry_mode", "variety": "variety_plus", "legacy_uc": "legacy_uc"}


def to_gen_args(
    record: dict[str, Any], image: str | Path | None = None, *, hidden_options: bool = False
) -> list[str] | None:
    """Turn parsed settings into nai gen arguments that rerun the image, or None when its prompts cannot be sent
    again. With the image's path, the Vibes its settings keep come back through --vibes-from."""
    if _unsendable(record) is not None:
        return None
    prompt, negative = _rerun_prompts(record)
    chars = record.get("characters", [])
    args = [values.option("--prompt", prompt), values.option("--neg", negative)]
    args += _switch_args(record, hidden_options)
    args += _setting_args(record)
    if image is not None and _carried_vibes(record):
        args.append(f"--vibes-from={Path(image).resolve()}")
    args += _character_args(record, chars)
    args += _tidy_args([prompt, negative, *(t for c in chars for t in (c["prompt"], c["negative"]))])
    # The stored prompts already hold any fixed tags, and were sent without ComfyUI splitting
    args += ["--no-fixed-tags", "--no-comfyui-characters"]
    return args


def rerun_size(record: dict[str, Any]) -> tuple[int, int] | None:
    """Each side on the grid, as the official client sends an imported size. When that exceeds the limit, which
    the official client would send with no images, the longer side is clamped the way its size box does."""
    width, height = record.get("width"), record.get("height")
    if not all(values.is_whole(v) and v > 0 for v in (width, height)):
        return None
    w, h = presets.nearest_grid(width), presets.nearest_grid(height)
    if w * h > presets.MAX_PIXELS:
        if w >= h:
            w = presets.side_within_limit(h)
        else:
            h = presets.side_within_limit(w)
    return w, h


def rerun_gaps(record: dict[str, Any]) -> list[str]:
    """What gen_args cannot bring back, so a rerun differs from the image."""
    gaps = [
        f"the image's {number.key} is out of range, so gen_args leaves it out"
        for number in (SEED, *NUMERIC_SETTINGS)
        if record.get(number.key) is not None and number.read(record) is None
    ]
    size = rerun_size(record)
    if size and presets.nearest_grid(record["width"]) * presets.nearest_grid(record["height"]) > presets.MAX_PIXELS:
        gaps.append(
            f"the image is {record['width']}x{record['height']}, above the pixel limit once on the grid, "
            f"so gen_args uses {size[0]}x{size[1]}"
        )
    if record.get("precise_references"):
        gaps.append(
            f"the image used {record['precise_references']} precise references, which images do not keep, "
            "so gen_args leaves them out"
        )
    missing = record.get("vibes", 0) - _carried_vibes(record)
    if missing > 0:
        gaps.append(f"the image used {missing} Vibes its settings do not keep, so gen_args leaves them out")
    unsendable = _unsendable(record)
    if unsendable is not None:
        gaps.append(f"the image's prompts hold {unsendable}, which no request can carry, so gen_args is null")
    return gaps


def _unsendable(record: dict[str, Any]) -> str | None:
    texts = [*_rerun_prompts(record), *(t for c in record.get("characters", []) for t in (c["prompt"], c["negative"]))]
    return next(filter(None, map(values.unsendable, texts)), None)


def _rerun_prompts(record: dict[str, Any]) -> tuple[str, str]:
    prompt = record.get("user_prompt", record["prompt"])
    negative = (
        record["negative"] if record.get("negative_verbatim") else record.get("user_negative", record["negative"])
    )
    return prompt, negative


def _switch_args(record: dict[str, Any], hidden_options: bool) -> list[str]:
    args = []
    if record.get("negative_verbatim"):
        args.append("--neg-verbatim")
    elif "official_nsfw" in record:
        # Pin the nsfw rule so config defaults cannot change it
        args.append("--official-nsfw" if record["official_nsfw"] else "--no-official-nsfw")
    if record.get("model"):
        args.append(f"--model={record['model']}")
        # Pinned, so the config's effort cannot turn V5 Full into its medium model
        effort = effort_of(record["model"])
        if effort:
            args.append(f"--effort={effort}")
    if record.get("quality"):
        args.append(f"--quality={record['quality']}")
    if record.get("uc_preset"):
        args.append(f"--uc-preset={record['uc_preset']}")
    if record.get("auto_text") is False:
        args.append("--no-auto-text")
    if record.get("normalize_vibe_strength") is False:
        args.append("--no-normalize-vibe-strength")
    model = MODELS.get(record.get("model") or "")
    if model is not None:
        model = with_hidden_options(model, hidden_options)
    # Only what the image's model takes, as a rerun refuses the rest; an unknown model gets all of it
    takes = {key: model is None or getattr(model, feature) for key, feature in _FEATURES.items()}
    if record.get("transparent") and takes["transparent"]:
        args.append("--transparent")
    if record.get("premultiplied_alpha") is not None and takes["transparent"]:
        # Pinned, so the config default cannot change it
        args.append("--premultiplied-alpha" if record["premultiplied_alpha"] else "--no-premultiplied-alpha")
    args += [flag for key, flag in _ON_SWITCHES if record.get(key) and takes[key]]
    return args


def _setting_args(record: dict[str, Any]) -> list[str]:
    args = [
        f"--{number.attr.replace('_', '-')}={record[number.key]}"
        for number in (SEED, *NUMERIC_SETTINGS)
        if number.read(record) is not None
    ]
    sampler = imported_sampler(record.get("sampler"))
    if sampler is not None:
        args.append(f"--sampler={sampler}")
    schedule = record.get("noise_schedule")
    if schedule == "native":
        # Only staging models took native; the official import gives karras
        schedule = "karras"
    if schedule in NOISE_SCHEDULES:
        args.append(f"--noise-schedule={schedule}")
    size = rerun_size(record)
    if size:
        args.append(f"--size={size[0]}x{size[1]}")
    # Reruns this one image, whatever n the config gives
    args.append("--n=1")
    return args


def _carried_vibes(record: dict[str, Any]) -> int:
    """Vibes --vibes-from brings back: those the settings keep, for a model that has Vibes."""
    model = MODELS.get(record.get("model") or "")
    return record.get("stored_vibes", 0) if model is not None and model.vibe else 0


def _character_args(record: dict[str, Any], chars: list[dict[str, Any]]) -> list[str]:
    # Negatives pair by position, so one differing from the model's fill gives every character one
    filled = resolve(record["model"]).character_negative if record.get("model") else None
    with_negatives = any(c["negative"] != filled for c in chars)
    args = []
    for c in chars:
        args.append(values.option("--char", c["prompt"]))
        if with_negatives:
            args.append(values.option("--char-neg", c["negative"]))
    if record.get("use_coords") and chars and all(c["x"] is not None and c["y"] is not None for c in chars):
        args += [f"--char-pos={c['x']},{c['y']}" for c in chars]
    return args


def _tidy_args(texts: list[str]) -> list[str]:
    """Tidying switched off wherever it would alter the stored text, whatever the config says."""
    args = []
    if any(tidy.format_prompt(t, underscores=True) != t for t in texts):
        args.append("--no-format-prompt")
    if any(tidy.space_numeric_closings(t) != t for t in texts):
        args.append("--no-space-numeric-close")
    if any(tidy.convert_sd_weights(t) != t for t in texts):
        args.append("--no-convert-sd-weights")
    return args
