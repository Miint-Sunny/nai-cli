"""Enhance and Variations levels, scales, sizes and sample counts, as in the official client."""

import math
from dataclasses import dataclass

from nai_cli import errors, presets
from nai_cli.models import Model
from nai_cli.presets import GRID, MAX_PIXELS, nearest_grid

# Levels 1-5 as (strength, noise); the default is level 3
LEVELS = [(0.2, 0.0), (0.4, 0.0), (0.5, 0.0), (0.6, 0.0), (0.7, 0.1)]
DEFAULT_LEVEL = 3
MAX_SCALE = "max"
_FACTORS = (2.0, 1.5, 1.0)
# 832x1216 only offers 1.5x and 1x, and 1.5x is offered even though it is not a multiple of 64
_STANDARD_PORTRAIT = {(832, 1216), (1216, 832)}
# max is offered only below 0.8 of the pixel limit
_MAX_ENHANCE_BELOW = 0.8 * MAX_PIXELS
# max is priced by doubling (rounded to 16), then scaling into the limit (rounded to 32)
_MAX_FACTOR = 2
_MAX_ROUND = 32
VARIATION_STRENGTH = 0.8
VARIATION_NOISE = 0.1
VARIATION_MAX_SAMPLES = 3


def scales(width: int, height: int, model: Model) -> list[str]:
    """Available scales, largest first: within the limit and multiples of 64, led by max for small enough V5 images."""
    if (width, height) in _STANDARD_PORTRAIT:
        factors = [1.5, 1.0]
    else:
        factors = [
            f
            for f in _FACTORS
            if width * f * height * f <= MAX_PIXELS and (width * f) % GRID == 0 and (height * f) % GRID == 0
        ]
    out = [_label(f) for f in factors]
    if model.max_enhance and 0 < width * height < _MAX_ENHANCE_BELOW:
        out.insert(0, MAX_SCALE)
    return out


def _label(factor: float) -> str:
    return f"{factor:g}"


def max_billed_size(width: int, height: int) -> tuple[int, int]:
    """Size the server upscales to at max; pricing uses it."""
    unit = 8 * _MAX_FACTOR
    w, h = width // unit * unit * _MAX_FACTOR, height // unit * unit * _MAX_FACTOR
    if not w or not h:
        raise errors.NaiError(errors.BAD_ARGS, f"{width}x{height}: too small to enhance at max; a side needs {unit}")
    scale = min(1.0, math.sqrt(MAX_PIXELS / (w * h)))
    tw = _MAX_ROUND * math.floor(w * scale / _MAX_ROUND + 0.5)
    th = _MAX_ROUND * math.floor(h * scale / _MAX_ROUND + 0.5)
    if tw * th > MAX_PIXELS:
        tw, th = _MAX_ROUND * math.floor(w * scale / _MAX_ROUND), _MAX_ROUND * math.floor(h * scale / _MAX_ROUND)
    if tw < _MAX_ROUND or th < _MAX_ROUND:
        raise errors.NaiError(errors.BAD_ARGS, f"{width}x{height}: aspect ratio too extreme to enhance")
    return tw, th


@dataclass
class Plan:
    level: int
    scale: str
    strength: float
    noise: float
    # Base image size, requested size and billed size; the first two differ only for 832x1216 at 1.5x
    image_size: tuple[int, int]
    request_size: tuple[int, int]
    billed_size: tuple[int, int]

    @property
    def upscaled(self) -> bool:
        return self.scale == MAX_SCALE

    def to_json(self) -> dict:
        return {
            "level": self.level,
            "scale": self.scale,
            "strength": self.strength,
            "noise": self.noise,
            "image_size": "{}x{}".format(*self.image_size),
            "billed_size": "{}x{}".format(*self.billed_size),
        }


def plan(
    width: int,
    height: int,
    model: Model,
    level: int | None,
    scale: str | None,
    strength: float | None,
    noise: float | None,
) -> Plan:
    """width and height are the source image size. Without a scale the largest available one is used."""
    available = scales(width, height, model)
    if not available:
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"{width}x{height}: no Enhance scale fits (the result must be a multiple of 64 within the limit)",
        )
    if scale is None:
        scale = available[0]
    elif scale not in available:
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"{width}x{height} on {model.name} supports only these scales: {available}",
            choices=available,
        )
    level = DEFAULT_LEVEL if level is None else level
    level_strength, level_noise = LEVELS[level - 1]
    if scale == MAX_SCALE:
        # The image goes as it is; its size is put on the grid like every request
        image_size = (width, height)
        request_size = (nearest_grid(width), nearest_grid(height))
        billed = max_billed_size(width, height)
    else:
        factor = float(scale)
        image_size = (math.floor(width * factor), math.floor(height * factor))
        # The requested size is rounded to the nearest multiple of 64 after the base image is scaled
        request_size = billed = (nearest_grid(image_size[0]), nearest_grid(image_size[1]))
    return Plan(
        level=level,
        scale=scale,
        strength=level_strength if strength is None else strength,
        noise=level_noise if noise is None else noise,
        image_size=image_size,
        request_size=request_size,
        billed_size=billed,
    )


def variation_samples(width: int, height: int) -> int:
    return min(VARIATION_MAX_SAMPLES, presets.max_samples(width, height))
