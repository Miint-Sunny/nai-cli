"""Anlas estimates using the official client's pricing formulas. Actual spend is the balance difference."""

import math
from dataclasses import dataclass

from nai_cli import errors, presets
from nai_cli.models import Model

OPUS_FREE_MAX_PIXELS = 1024 * 1024
OPUS_FREE_MAX_STEPS = 28
MAX_PER_SAMPLE = 140
_AREA = 2.951823174884865e-6
_STEP_AREA = 5.753298233447344e-7
# Medium is priced at a fixed 23 steps divided by 1.75, without rounding
MEDIUM_COST_STEPS = 23
MEDIUM_COST_DIVISOR = 1.75
VIBE_ENCODE_COST = 2
VIBE_EXTRA_COST = 2
VIBE_FREE_COUNT = 4
PRECISE_REFERENCE_COST = 5
# The main generation page's tiers; the Director tools page offers only the first
_UPSCALE_TIERS = [(1048576, 1), (1747627, 2), (2446678, 3), (3145728, 4)]
UPSCALE_MAX_PIXELS = 3145728
TOOL_STEPS = 28


@dataclass
class Estimate:
    anlas: int
    per_sample: int
    free_by_opus: bool
    # Free only while the V5 quota pool lasts; afterwards if_quota_exhausted is charged
    if_quota_exhausted: int
    valid: bool = True


def require_valid(estimate: Estimate) -> None:
    if not estimate.valid:
        raise errors.NaiError(
            errors.BAD_ARGS, "size and steps exceed the NovelAI per-image limit", per_sample=estimate.per_sample
        )


def per_sample(model: Model, width: int, height: int, steps: int, strength: float = 1.0) -> int:
    pixels = width * height
    if model.medium_effort:
        base = (_AREA * pixels + _STEP_AREA * pixels * MEDIUM_COST_STEPS) / MEDIUM_COST_DIVISOR
    else:
        base = _area_price(pixels, steps)
    return max(math.ceil(base * model.anlas_multiplier * strength), 2)


def generation(
    model: Model,
    width: int,
    height: int,
    steps: int,
    n_samples: int = 1,
    strength: float = 1.0,
    opus: bool = True,
    quota_exhausted: bool = False,
    precise_references: int = 0,
    vibes: int = 0,
    vibes_to_encode: int = 0,
) -> Estimate:
    """model is the model that actually runs (the inpainting model for infill); vibes counts only those sent."""
    sample = per_sample(model, width, height, steps, strength)
    if sample > MAX_PER_SAMPLE:
        return Estimate(anlas=-1, per_sample=sample, free_by_opus=False, if_quota_exhausted=-1, valid=False)
    effective_steps = model.fixed.steps if model.fixed else steps
    # The server keeps the image free with precise references and charges only them (measured)
    eligible = opus and effective_steps <= OPUS_FREE_MAX_STEPS and width * height <= OPUS_FREE_MAX_PIXELS
    blocked = model.opus_quota and quota_exhausted
    free = eligible and not blocked
    extra_per_sample = precise_references * PRECISE_REFERENCE_COST
    extra_per_request = max(vibes - VIBE_FREE_COUNT, 0) * VIBE_EXTRA_COST
    one_time = vibes_to_encode * VIBE_ENCODE_COST

    def total(discount: int) -> int:
        samples = sample * max(n_samples - discount, 0) + extra_per_sample * n_samples
        return samples + extra_per_request + one_time

    return Estimate(
        anlas=total(1 if free else 0),
        per_sample=sample,
        free_by_opus=free,
        if_quota_exhausted=total(0) if (eligible and model.opus_quota) else total(1 if free else 0),
    )


def upscale(width: int, height: int) -> int:
    """Priced by the source's area alone, with no Opus discount; returns -1 above 3145728 pixels."""
    pixels = width * height
    if pixels > UPSCALE_MAX_PIXELS:
        return -1
    return next(price for limit, price in _UPSCALE_TIERS if pixels <= limit)


def augment(width: int, height: int, bg_removal: bool = False, opus: bool = True) -> int:
    """Scales the size into [TOOL_MIN_PIXELS, TOOL_MAX_PIXELS], then prices one 28-step image of
    nai-diffusion-3, which the official client uses for every tool; returns -1 above the per-image limit."""
    w, h = presets.grow_to(*presets.shrink_to(width, height, presets.TOOL_MAX_PIXELS), presets.TOOL_MIN_PIXELS)
    price = max(_area_price(w * h, TOOL_STEPS), 2)
    if price > MAX_PER_SAMPLE:
        return -1
    if bg_removal:
        return price * 3 + 5
    if opus and w * h <= OPUS_FREE_MAX_PIXELS:
        return 0
    return price


def _area_price(pixels: int, steps: int) -> int:
    return math.ceil(_AREA * pixels + _STEP_AREA * pixels * steps)
