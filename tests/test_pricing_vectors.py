"""Pricing vectors shared with other implementations of cost.py; `python tests/test_pricing_vectors.py` rewrites them."""

import itertools
import json
from pathlib import Path

from nai_cli import cost, enhance, models

VECTORS = Path(__file__).resolve().parent / "pricing_vectors.json"

_SIZES = [(64, 64), (832, 1216), (1024, 1024), (1024, 1536), (1472, 1472), (1536, 2048)]
_STEPS = [1, 23, 28, 29, 50]
# (opus, quota_exhausted)
_ACCOUNTS = [(True, False), (True, True), (False, False)]
# (n_samples, strength, precise_references, vibes, vibes_to_encode)
_EXTRAS = [
    (2, 1.0, 0, 0, 0),
    (4, 1.0, 0, 0, 0),
    (1, 0.37, 0, 0, 0),
    (3, 0.7, 0, 0, 0),
    (1, 1.0, 2, 0, 0),
    (2, 1.0, 3, 0, 0),
    (1, 1.0, 0, 4, 0),
    (1, 1.0, 0, 6, 1),
    (2, 0.5, 0, 9, 3),
]
_TOOL_SIZES = [(8, 4096), (512, 512), (832, 1216), (1024, 1024), (1024, 1536), (1216, 1664), (1536, 2048)]
_TOOL_SIZES += [(1536, 2112), (4096, 4096)]
_ENHANCE_SIZES = [(64, 64), (512, 768), (832, 1216), (1024, 1024), (1216, 832), (1280, 1536)]


def test_vectors_match_cost():
    assert json.loads(VECTORS.read_text(encoding="utf-8")) == export()


def export() -> dict:
    return {
        "models": _models(),
        "generation": _generation(),
        "upscale": [{"width": w, "height": h, "anlas": cost.upscale(w, h)} for w, h in _TOOL_SIZES],
        "augment": [
            {"width": w, "height": h, "bg_removal": bg, "opus": opus, "anlas": cost.augment(w, h, bg, opus)}
            for (w, h), bg, opus in itertools.product(_TOOL_SIZES, (False, True), (True, False))
        ],
        "enhance_max": [_enhance(w, h) for w, h in _ENHANCE_SIZES],
    }


def write() -> None:
    # One model or case per line keeps diffs readable when a price changes
    sections = []
    for key, value in export().items():
        if isinstance(value, dict):
            rows = [f"{json.dumps(k)}:{_compact(v)}" for k, v in value.items()]
            sections.append(f'"{key}":{{\n' + ",\n".join(rows) + "\n}")
        else:
            sections.append(f'"{key}":[\n' + ",\n".join(_compact(v) for v in value) + "\n]")
    VECTORS.write_text("{\n" + ",\n".join(sections) + "\n}\n", encoding="utf-8", newline="\n")


def _models() -> dict:
    # Keyed by every id a request may carry; inpainting ids price as the model resolve maps them to
    ids = list(models.MODELS) + [m.inpainting_id for m in models.MODELS.values()]
    out = {}
    for model_id in ids:
        m = models.resolve(model_id)
        out[model_id] = {
            "anlas_multiplier": m.anlas_multiplier,
            "medium_effort": m.medium_effort,
            "opus_quota": m.opus_quota,
            "fixed_steps": m.fixed.steps if m.fixed else None,
            "max_enhance": m.max_enhance,
        }
    return out


def _generation() -> list[dict]:
    plain = [(1, 1.0, 0, 0, 0)]
    cases = itertools.product(models.MODELS, _SIZES, _STEPS, _ACCOUNTS, plain)
    extra = itertools.product(
        ("nai-diffusion-5-full", "nai-diffusion-4-5-full"), _SIZES[1:4], (28,), _ACCOUNTS, _EXTRAS
    )
    out = []
    for model_id, (w, h), steps, (opus, exhausted), (n, strength, refs, vibes, encode) in itertools.chain(cases, extra):
        est = cost.generation(models.resolve(model_id), w, h, steps, n, strength, opus, exhausted, refs, vibes, encode)
        out.append(
            {
                "model": model_id,
                "width": w,
                "height": h,
                "steps": steps,
                "n_samples": n,
                "strength": strength,
                "opus": opus,
                "quota_exhausted": exhausted,
                "precise_references": refs,
                "vibes": vibes,
                "vibes_to_encode": encode,
                "anlas": est.anlas,
                "per_sample": est.per_sample,
                "free_by_opus": est.free_by_opus,
                "if_quota_exhausted": est.if_quota_exhausted,
                "valid": est.valid,
            }
        )
    return out


def _enhance(width: int, height: int) -> dict:
    bw, bh = enhance.max_billed_size(width, height)
    return {"width": width, "height": height, "billed_width": bw, "billed_height": bh}


def _compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


if __name__ == "__main__":
    write()
