"""interrogate: tags read from an image by an ONNX tagger on this computer, joined into a prompt."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import errors, fs, images, local_models, tagger
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class InterrogateSettings:
    image: Path
    tagger: tagger.Settings


def resolve(opts: dict[str, Any], ctx: Context) -> InterrogateSettings:
    local = ctx.config.local
    model = opts["model"] if opts.get("model") else local.tagger
    if model is None:
        raise errors.NaiError(
            errors.CONFIG_MISSING, "no tagger model: give --model or set tagger under [local] in the config"
        )
    given = opts.get("labels") or (local.tagger_labels if not opts.get("model") else None)
    labels = given if given else _labels_beside(model)
    kind_text = opts.get("kind") or (local.tagger_kind if not opts.get("model") else "")
    try:
        kind = tagger.Kind(kind_text) if kind_text else tagger.guess_kind(model, labels)
    except ValueError:
        choices = ", ".join(k.value for k in tagger.Kind)
        raise errors.NaiError(errors.BAD_ARGS, f"the tagger kind is one of {choices}: {kind_text}") from None
    thresholds = {}
    for name in ("threshold", "character_threshold"):
        value = opts.get(name)
        if value is not None and not 0 <= value <= 1:
            raise errors.NaiError(errors.BAD_ARGS, f"--{name.replace('_', '-')} is within 0-1: {value}")
        thresholds[name] = tagger.DEFAULT_THRESHOLD if value is None else value
    return InterrogateSettings(
        Path(opts["image"]),
        tagger.Settings(model, labels, kind, categories=_categories(opts.get("categories")), **thresholds),
    )


def _labels_beside(model: Path) -> Path:
    """The first of the tag list's usual places that holds a file, else a tagger's config.json."""
    found = next((p for p in tagger.label_candidates(model) if fs.is_file(p)), None)
    config_file = model.parent / "config.json"
    if found is None and tagger.is_split_config(config_file):
        found = config_file
    if found is None:
        raise errors.NaiError(errors.CONFIG_MISSING, f"no tag list found beside {model}; give it with --labels")
    return found


def _categories(text: str | None) -> tuple[tagger.Category, ...]:
    if not text:
        return tagger.DEFAULT_CATEGORIES
    names = {c.value: c for c in tagger.CHOOSABLE}
    chosen = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = [part for part in chosen if part not in names]
    if unknown or not chosen:
        raise errors.NaiError(errors.BAD_ARGS, f"--categories takes a comma list of {', '.join(names)}: {text}")
    return tuple(dict.fromkeys(names[part] for part in chosen))


def run(s: InterrogateSettings, ctx: Context) -> dict[str, Any]:
    image = images.open_still(s.image)
    labels = tagger.load_labels(s.tagger.labels)
    tags = tagger.interrogate(image, s.tagger, labels, local_models.load(s.tagger.model))
    return {
        "ok": True,
        "image": str(s.image),
        "prompt": ", ".join(t.text for t in tags),
        "tags": [{"tag": t.text, "score": round(t.score, 4), "category": t.category.value} for t in tags],
        "model": str(s.tagger.model),
        "kind": s.tagger.kind.value,
    }
