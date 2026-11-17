"""Sizes, quality tags, UC presets, Director tool limits and images per request."""

import math

from nai_cli import errors
from nai_cli.models import Model

QUALITY_TAGS: dict[str, dict[str, str]] = {
    "v5": {
        "standard": "very aesthetic, masterpiece, no text",
        "light": "very aesthetic, amazing quality, no text",
    },
    "nai-diffusion-4-5-full": {"standard": "very aesthetic, masterpiece, no text"},
    "nai-diffusion-4-5-curated": {"standard": "very aesthetic, masterpiece, no text, -0.8::feet::, rating:general"},
    "nai-diffusion-4-full": {"standard": "no text, best quality, very aesthetic, absurdres"},
    "nai-diffusion-4-curated-preview": {"standard": "rating:general, best quality, very aesthetic, absurdres"},
}

# tag_hint_qt: 0 off, 1 standard, 3 light
QUALITY_HINT = {"none": 0, "standard": 1, "light": 3}

_V5_HEAVY = (
    "lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, "
    "very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, "
    "too many watermarks, negative space, blank page"
)
_FURRY = (
    "{worst quality}, distracting watermark, unfinished, bad quality, {widescreen}, upscale, {sequence}, "
    "{{grandfathered content}}, blurred foreground, chromatic aberration, sketch, everyone, "
    "[sketch background], simple, [flat colors], ych (character), outline, multiple scenes, "
    "[[horror (theme)]], comic"
)

UC_PRESETS: dict[str, dict[str, str]] = {
    "v5": {
        "heavy": _V5_HEAVY,
        "light": (
            "lowres, bad hands, bad anatomy, artistic error, sepia, white haze, worst quality, "
            "very displeasing, jpeg artifacts, 0::ai-generated::"
        ),
        "furry": _FURRY,
        "human": _V5_HEAVY + ", @_@, mismatched pupils, glowing eyes, bad anatomy",
        "none": "",
    },
    "nai-diffusion-4-5-full": {
        "heavy": _V5_HEAVY,
        "light": (
            "lowres, artistic error, scan artifacts, worst quality, bad quality, jpeg artifacts, "
            "multiple views, very displeasing, too many watermarks, negative space, blank page"
        ),
        "furry": _FURRY,
        "human": _V5_HEAVY + ", @_@, mismatched pupils, glowing eyes, bad anatomy",
        "none": "",
    },
    "nai-diffusion-4-5-curated": {
        "heavy": (
            "blurry, lowres, upscaled, artistic error, film grain, scan artifacts, worst quality, "
            "bad quality, jpeg artifacts, very displeasing, chromatic aberration, halftone, multiple views, "
            "logo, too many watermarks, negative space, blank page"
        ),
        "light": (
            "blurry, lowres, upscaled, artistic error, scan artifacts, jpeg artifacts, logo, "
            "too many watermarks, negative space, blank page"
        ),
        "human": (
            "blurry, lowres, upscaled, artistic error, film grain, scan artifacts, bad anatomy, bad hands, "
            "worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, halftone, "
            "multiple views, logo, too many watermarks, @_@, mismatched pupils, glowing eyes, "
            "negative space, blank page"
        ),
        "none": "",
    },
    "nai-diffusion-4-full": {
        "heavy": (
            "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, "
            "very displeasing, chromatic aberration, multiple views, logo, too many watermarks, white blank page, "
            "blank page"
        ),
        "light": (
            "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, white blank page, "
            "blank page"
        ),
        "none": "",
    },
    "nai-diffusion-4-curated-preview": {
        "heavy": (
            "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, "
            "very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts, "
            "white blank page, blank page"
        ),
        "light": (
            "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, logo, dated, "
            "signature, white blank page, blank page"
        ),
        "none": "",
    },
}

UC_PRESET_ID = {"heavy": "heavy", "light": "light", "human": "humanFocus", "furry": "furryFocus", "none": "none"}
UC_PRESET_HINT = {"heavy": 2, "light": 3, "human": 4, "furry": 5, "none": 0}
UC_PRESET_ALIASES = {"humanfocus": "human", "human_focus": "human", "furryfocus": "furry", "off": "none"}
# Fallback order for UC presets when switching models
_UC_FALLBACK = {
    "none": ["none", "light", "heavy"],
    "light": ["light", "none", "heavy"],
    "heavy": ["heavy", "light", "none"],
    "human": ["human", "heavy", "light", "none"],
    "furry": ["furry", "heavy", "light", "none"],
}

SIZE_PRESETS = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
    "portrait-large": (1024, 1536),
    "landscape-large": (1536, 1024),
    "square-large": (1472, 1472),
    "portrait-wallpaper": (1088, 1920),
    "landscape-wallpaper": (1920, 1088),
    "portrait-small": (512, 768),
    "landscape-small": (768, 512),
    "square-small": (640, 640),
}

MAX_PIXELS = 3145728
GRID = 64
# Director tools price at sizes scaled into TOOL_MAX_PIXELS, then up to TOOL_MIN_PIXELS
TOOL_MAX_PIXELS = 3145728
# Requests stay 2000 below the limit and only upscale below TOOL_GROW_BELOW
TOOL_REQUEST_MAX_PIXELS = TOOL_MAX_PIXELS - 2000
TOOL_GROW_BELOW = 1011712
TOOL_MIN_PIXELS = 1048576
AUGMENT_TOOLS = ("bg-removal", "declutter", "declutter-keep-bubbles", "lineart", "sketch", "colorize", "emotion")
# Runs here, as the official client runs it in the browser
PIXEL_SNAP = "pixel-snap"
# Names the official client gives each image a tool returns, in reply order
TOOL_OUTPUTS = {"bg-removal": ("masked", "generated", "blend")}
# fmt: off
EMOTIONS = [
    "neutral", "happy", "sad", "angry", "scared", "surprised", "tired", "excited", "nervous", "thinking", "confused",
    "shy", "disgusted", "smug", "bored", "laughing", "irritated", "aroused", "embarrassed", "worried", "love",
    "determined", "hurt", "playful",
]
# fmt: on
# The emotion and the extra description are sent as one prompt
EMOTION_SEPARATOR = ";;"
DEFRY_MAX = 5
# Samples per request by area: the first tier whose area is not exceeded applies
# fmt: off
_SAMPLE_LIMITS = [
    (3145728, 4), (1572864, 4), (1310720, 4), (524288, 4), (409600, 6), (360448, 8), (262144, 8), (245760, 8)
]


def quality_tags(model: Model, tier: str) -> str:
    tiers = QUALITY_TAGS[_key(model)]
    return tiers.get(tier) or tiers["standard"]


def quality_tiers(model: Model) -> list[str]:
    return ["none", *QUALITY_TAGS[_key(model)]]


def resolve_quality(model: Model, value: str | None) -> str:
    v = (value or model.default_quality).strip().lower()
    v = {"off": "none", "on": "standard", "true": "standard", "false": "none"}.get(v, v)
    if v not in quality_tiers(model):
        raise errors.NaiError(errors.BAD_ARGS, f"quality for {model.name} must be one of {quality_tiers(model)}")
    return v


def uc_presets(model: Model) -> list[str]:
    return list(UC_PRESETS[_key(model)])


def uc_preset_name(value: str) -> str:
    """A UC preset some model has, by its name or an alias."""
    v = value.strip().lower()
    v = UC_PRESET_ALIASES.get(v, v)
    if v not in _UC_FALLBACK:
        raise errors.NaiError(errors.BAD_ARGS, f"unknown UC preset: {value}", choices=list(_UC_FALLBACK))
    return v


def resolve_uc_preset(model: Model, value: str | None) -> str:
    v = uc_preset_name(value or model.default_uc_preset)
    if v not in uc_presets(model):
        raise errors.NaiError(errors.BAD_ARGS, f"UC presets for {model.name}: {uc_presets(model)}")
    return v


def carry_uc_preset(preset: str, model: Model) -> str:
    available = uc_presets(model)
    return next((p for p in _UC_FALLBACK.get(preset, ["none"]) if p in available), "none")


def uc_preset_text(model: Model, preset: str) -> str:
    return UC_PRESETS[_key(model)][preset]
# fmt: on


def _key(model: Model) -> str:
    return "v5" if model.is_v5 else model.id


def max_samples(width: int, height: int) -> int:
    area = width * height
    limit = 0
    for max_area, n in _SAMPLE_LIMITS:
        if area > max_area:
            break
        limit = n
    return limit


def nearest_grid(value: float) -> int:
    """Round to the nearest multiple of 64, ties upward, never below 64."""
    low, high = math.floor(value / GRID) * GRID, math.ceil(value / GRID) * GRID
    picked = low if value - low < high - value else high
    return picked if picked > 0 else GRID


def side_within_limit(other: int) -> int:
    """The longest side on the grid that stays within MAX_PIXELS next to other, as the official size box
    clamps it."""
    return MAX_PIXELS // other // GRID * GRID


def shrink_to(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Scale down proportionally to at most max_pixels, flooring each side to no less than one pixel."""
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def grow_to(width: int, height: int, min_pixels: int) -> tuple[int, int]:
    """Scale up proportionally to min_pixels, flooring each side, so the result may fall slightly short."""
    if width * height >= min_pixels:
        return width, height
    scale = math.sqrt(min_pixels / (width * height))
    return math.floor(width * scale), math.floor(height * scale)


def tool_request_size(width: int, height: int) -> tuple[int, int]:
    w, h = shrink_to(width, height, TOOL_REQUEST_MAX_PIXELS)
    if w * h < TOOL_GROW_BELOW:
        w, h = grow_to(w, h, TOOL_MIN_PIXELS)
    return w, h


def emotion_prompt(emotion: str, extra: str) -> str:
    return f"{emotion}{EMOTION_SEPARATOR}{extra}"


def resolve_size(value: str | None) -> tuple[int, int]:
    if not value:
        return SIZE_PRESETS["portrait"]
    v = value.strip().lower()
    if v in SIZE_PRESETS:
        return SIZE_PRESETS[v]
    try:
        w, h = (int(x) for x in v.replace("×", "x").split("x"))
    except ValueError:
        raise errors.NaiError(
            errors.BAD_ARGS, f"size must be WIDTHxHEIGHT or a preset name: {value}", presets=list(SIZE_PRESETS)
        ) from None
    if w <= 0 or h <= 0 or w % 64 or h % 64:
        raise errors.NaiError(errors.BAD_ARGS, f"width and height must be positive multiples of 64: {w}x{h}")
    if w * h > MAX_PIXELS:
        raise errors.NaiError(errors.BAD_ARGS, f"{w}x{h} exceeds the NovelAI limit of {MAX_PIXELS} pixels")
    return w, h


# Palette sizes for pixel snap: the official slider starts at 16, but takes a smaller number typed in
SNAP_COLORS = (1, 256)
