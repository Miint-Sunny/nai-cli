"""Inpainting: mask building, canvas expansion, focused crops and feathered compositing as in the official client."""

import io
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from nai_cli import errors, images, resample
from nai_cli.images import LATENT_SCALE, MASK_THRESHOLD
from nai_cli.presets import GRID, MAX_PIXELS

# Focused crops are upscaled to about 1 MP unless already above 0.8 of that
FOCUS_TARGET_PIXELS = 1048576
FOCUS_KEEP_RATIO = 0.8
DEFAULT_FOCUS_PADDING = 96
FOCUS_PADDING_RANGE = (32, 192)
EDITOR_MAX_SIDE = 2560
# Feathering: dilate the latent mask by 4, upscale 8x, box blur twice with radius 20
_DILATE_RADIUS = 4
_BLUR_RADIUS = 20
_BLUR_ITERATIONS = 2
# Blur multiplier for radius 20: multiply by 39 and shift right 16, roughly dividing by 41^2
_BLUR_MUL = 39
_BLUR_SHIFT = 16
SHAPES = ("rect", "ellipse", "poly", "object")
_REGION = re.compile(rf"^(-?)({'|'.join(SHAPES)}):(.+)$")
_PREVIEW_MASK_ALPHA = 110
_PREVIEW_MASK_COLOR = (255, 0, 0)
_PREVIEW_FOCUS_COLOR = (0, 255, 0, 255)


@dataclass
class Region:
    shape: str
    points: list[float]
    subtract: bool = False


# The mask, the size of the image, of the object under 0-1 points
Segment = Callable[[Image.Image, list[tuple[float, float]]], Image.Image]


def parse_region(text: str) -> Region:
    """rect:x1,y1,x2,y2, ellipse:x1,y1,x2,y2 (bounding box), poly:x,y,x,y,x,y... or object:x,y... (the object
    under the points, found by the local SAM model), as 0-1 fractions of the canvas. A leading - subtracts."""
    m = _REGION.match(text.strip())
    if not m:
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"region must be rect:x1,y1,x2,y2, ellipse:x1,y1,x2,y2, poly:x,y,x,y,x,y or object:x,y: {text}",
        )
    try:
        points = [float(v) for v in m.group(3).split(",")]
    except ValueError:
        raise errors.NaiError(errors.BAD_ARGS, f"region coordinates must be numbers: {text}") from None
    shape = m.group(2)
    if (
        (shape in ("rect", "ellipse") and len(points) != 4)
        or (shape == "poly" and (len(points) < 6 or len(points) % 2))
        or (shape == "object" and (len(points) < 2 or len(points) % 2))
    ):
        raise errors.NaiError(errors.BAD_ARGS, f"wrong number of coordinates for {shape}: {text}")
    if any(not 0 <= v <= 1 for v in points):
        raise errors.NaiError(
            errors.BAD_ARGS, f"region coordinates are fractions of the canvas and must be within 0-1: {text}"
        )
    return Region(shape, points, subtract=bool(m.group(1)))


def parse_expand(text: str) -> tuple[int, int, int, int]:
    """left,top,right,bottom in pixels, or one number for all sides."""
    try:
        values = [int(v) for v in text.split(",")]
    except ValueError:
        raise errors.NaiError(
            errors.BAD_ARGS, f"expansion must be left,top,right,bottom in pixels or one number: {text}"
        ) from None
    if len(values) == 1:
        values *= 4
    if len(values) != 4 or any(v < 0 for v in values) or not any(values):
        raise errors.NaiError(
            errors.BAD_ARGS, f"expansion needs four non-negative pixel values, at least one positive: {text}"
        )
    return values[0], values[1], values[2], values[3]


def draw_regions(
    mask: Image.Image, regions: list[Region], image: Image.Image | None = None, segment: Segment | None = None
) -> Image.Image:
    """Add or subtract regions in order on a grayscale mask; white is repainted. Object regions segment image,
    which has the mask's size."""
    out = mask.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    for region in regions:
        pairs = [(region.points[i], region.points[i + 1]) for i in range(0, len(region.points), 2)]
        xy = [(x * width, y * height) for x, y in pairs]
        fill = 0 if region.subtract else 255
        if region.shape == "object":
            if image is None or segment is None:
                raise errors.NaiError(errors.BAD_ARGS, "object regions are not available here")
            found = resample.nearest(segment(image, pairs), width, height)
            out.paste(fill, mask=_binary(found))
            continue
        if region.shape == "poly":
            draw.polygon(xy, fill=fill)
            continue
        (x1, y1), (x2, y2) = xy
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if region.shape == "rect":
            draw.rectangle(box, fill=fill)
        else:
            draw.ellipse(box, fill=fill)
    return out


def grow(mask: Image.Image, pixels: int) -> Image.Image:
    if pixels <= 0:
        return mask
    grown = _dilate(np.asarray(_binary(mask)) == 255, pixels)
    return Image.fromarray(grown.astype(np.uint8) * 255, "L")


def _binary(img: Image.Image) -> Image.Image:
    return img.point(lambda v: 255 if v > MASK_THRESHOLD else 0)


def editor_canvas_size(width: int, height: int) -> tuple[int, int]:
    """Editor canvas size: scale down to a 2560 longest side, then round each side up to a multiple of 64."""
    if width > EDITOR_MAX_SIDE or height > EDITOR_MAX_SIDE:
        ratio = width / height
        if ratio > 1:
            width, height = EDITOR_MAX_SIDE, math.floor(EDITOR_MAX_SIDE / ratio)
        else:
            width, height = math.floor(EDITOR_MAX_SIDE * ratio), EDITOR_MAX_SIDE
    return max(GRID, math.ceil(width / GRID) * GRID), max(GRID, math.ceil(height / GRID) * GRID)


def expand_canvas(
    img: Image.Image, edges: tuple[int, int, int, int]
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """Pad with transparent canvas, each expanded side rounded up to a multiple of 64 as the official resize dialog
    rounds it, so the source stays on the latent grid. Returns the canvas, a mask of the new area and the source
    offset. The mask reaches one latent cell into the source along each expanded side, as the official editor
    paints it, so the seam is repainted too."""
    asked = tuple(edge > 0 for edge in edges)
    left, top, right, bottom = (-(-edge // GRID) * GRID for edge in edges)
    # Only a source off the grid needs more, which the editor's canvas never is; it is not a seam to repaint
    extra_w = -(img.width + left + right) % GRID
    extra_h = -(img.height + top + bottom) % GRID
    if right or not left:
        right += extra_w
    else:
        left += extra_w
    if bottom or not top:
        bottom += extra_h
    else:
        top += extra_h
    width, height = img.width + left + right, img.height + top + bottom
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.paste(img.convert("RGBA"), (left, top))
    mask = Image.new("L", (width, height), 255)
    kept = (
        left + (LATENT_SCALE if asked[0] else 0),
        top + (LATENT_SCALE if asked[1] else 0),
        left + img.width - (LATENT_SCALE if asked[2] else 0),
        top + img.height - (LATENT_SCALE if asked[3] else 0),
    )
    if kept[0] < kept[2] and kept[1] < kept[3]:
        mask.paste(0, kept)
    return canvas, mask, (left, top)


@dataclass
class Focus:
    """Crop box in canvas pixels (multiples of 64) and the requested generation size."""

    x: int
    y: int
    width: int
    height: int
    request_width: int
    request_height: int

    @property
    def latent_box(self) -> tuple[int, int, int, int]:
        s = LATENT_SCALE
        return self.x // s, self.y // s, (self.x + self.width) // s, (self.y + self.height) // s

    def to_json(self) -> dict:
        return {
            "crop": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "request_size": f"{self.request_width}x{self.request_height}",
        }


def focus_request_size(width: int, height: int) -> tuple[int, int]:
    """Generation size for a focused crop: below 0.8 MP, scale up within 1 MP in steps of 64."""
    area = width * height
    if area >= FOCUS_KEEP_RATIO * FOCUS_TARGET_PIXELS:
        return width, height
    scale = math.sqrt(FOCUS_TARGET_PIXELS / area)
    return math.floor(math.floor(width * scale) / GRID) * GRID, math.floor(math.floor(height * scale) / GRID) * GRID


def plan_focus(latent: Image.Image, padding: int) -> Focus:
    """Crop box around the mask plus context padding; latent is the latent mask of the whole canvas."""
    box = latent.getbbox()
    if box is None:
        raise errors.NaiError(errors.EMPTY_MASK, "the mask selects nothing to repaint")
    pad = math.ceil(padding / LATENT_SCALE)
    step = GRID // LATENT_SCALE
    if latent.width < step or latent.height < step:
        raise errors.NaiError(errors.BAD_ARGS, f"canvas is smaller than {GRID}x{GRID}; cannot focus")
    x0, x1 = _fit_span(box[0] - pad, box[2] + pad, latent.width, step)
    y0, y1 = _fit_span(box[1] - pad, box[3] + pad, latent.height, step)
    s = LATENT_SCALE
    return Focus(x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s, *focus_request_size((x1 - x0) * s, (y1 - y0) * s))


def _fit_span(start: int, end: int, limit: int, step: int) -> tuple[int, int]:
    """Grow [start, end) symmetrically to a multiple of step, then shift it into [0, limit)."""
    size = min(max(step, math.ceil((end - start) / step) * step), limit // step * step)
    begin = min(max(0, (start + end - size) // 2), limit - size)
    return begin, begin + size


def choose_focus(
    latent: Image.Image, canvas: tuple[int, int], padding: int, wanted: bool | None
) -> tuple[Focus | None, list[str]]:
    """With wanted None, focus automatically when the crop covers the mask, is smaller than the canvas and
    fits in 1 MP, which keeps the full resolution at no cost. Otherwise repaint the whole image."""
    if wanted is False:
        return None, []
    plan = plan_focus(latent, padding)
    area = plan.width * plan.height
    covers = _covers(plan, latent)
    if wanted:
        if area > MAX_PIXELS:
            raise errors.NaiError(
                errors.BAD_ARGS,
                f"focus area {plan.width}x{plan.height} exceeds the generation limit; shrink the mask, "
                "lower --focus-padding or use --no-focus",
            )
        if not covers:
            return plan, [
                f"focus area {plan.width}x{plan.height} does not cover the whole mask; the rest is not repainted"
            ]
        return plan, []
    if covers and area <= FOCUS_TARGET_PIXELS and area < canvas[0] * canvas[1]:
        return plan, []
    return None, []


def _covers(plan: Focus, latent: Image.Image) -> bool:
    """Whether the crop covers the whole mask; it may not when the canvas is not a multiple of 64."""
    box = latent.getbbox()
    x0, y0, x1, y1 = plan.latent_box
    return box is not None and x0 <= box[0] and y0 <= box[1] and x1 >= box[2] and y1 >= box[3]


def feather(latent: Image.Image) -> Image.Image:
    """Feather mask for compositing: dilate, upscale 8x, blur twice; a grayscale image at generation size."""
    dilated = _dilate(np.asarray(latent) == 255, _DILATE_RADIUS)
    values = np.repeat(np.repeat(dilated.astype(np.int64) * 255, LATENT_SCALE, axis=0), LATENT_SCALE, axis=1)
    for _ in range(_BLUR_ITERATIONS):
        summed = _box_sum(_box_sum(values, _BLUR_RADIUS, axis=1), _BLUR_RADIUS, axis=0)
        values = (summed * _BLUR_MUL) >> _BLUR_SHIFT
    return Image.fromarray(values.astype(np.uint8), "L")


def _box_sum(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Window sum of the given radius along one axis, clamping at the edges."""
    n = values.shape[axis]
    padded = np.take(values, np.clip(np.arange(-radius, n + radius), 0, n - 1), axis=axis)
    summed = np.cumsum(padded, axis=axis)
    zero = np.zeros_like(np.take(summed, [0], axis=axis))
    summed = np.concatenate([zero, summed], axis=axis)
    return np.take(summed, np.arange(2 * radius + 1, n + 2 * radius + 1), axis=axis) - np.take(
        summed, np.arange(n), axis=axis
    )


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation with edge clamping, equivalent to ignoring out-of-bounds pixels."""
    return _box_sum(_box_sum(binary.astype(np.int64), radius, axis=1), radius, axis=0) > 0


def composite(base: Image.Image, result: Image.Image, mask: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Composite like the official client: multiply the result's alpha by the feather mask, resize it to
    the box, cut the base by the mask and lay the result over it. Stealth bits are copied column-major."""
    x, y, w, h = box
    raw = np.asarray(result.convert("RGBA"))
    cleaned = np.asarray(images.clear_stealth_alpha(result.convert("RGBA"))).astype(np.float64)
    cleaned[..., 3] = _js_round(cleaned[..., 3] * np.asarray(mask, dtype=np.float64) / 255)
    pasted = resample.lanczos(Image.fromarray(cleaned.astype(np.uint8), "RGBA"), w, h)
    hole = np.asarray(resample.lanczos(Image.merge("RGBA", (mask, mask, mask, mask)), w, h), dtype=np.float64)[..., 3]

    canvas = np.asarray(base.convert("RGBA")).astype(np.float64)
    region = canvas[y : y + h, x : x + w]
    region[..., 3] = _js_round(region[..., 3] * (255 - hole) / 255)
    src = np.asarray(pasted, dtype=np.float64)
    src_a = src[..., 3:4] / 255
    dst_a = region[..., 3:4] / 255
    total = np.minimum(1, src_a + dst_a)
    color = np.minimum(255, (src[..., :3] * src_a + region[..., :3] * dst_a) / np.where(total == 0, 1, total))
    # Uint8ClampedArray rounds colour half to even, while alpha uses Math.round
    region[..., :3] = np.where(total == 0, region[..., :3], np.rint(color))
    region[..., 3:4] = np.where(total == 0, region[..., 3:4], _js_round(255 * total))
    out = np.clip(canvas, 0, 255).astype(np.uint8)

    bits = (raw[..., 3] & 1).T.reshape(-1)
    by_column = out[..., 3].T.reshape(-1)
    count = min(bits.size, by_column.size)
    by_column[:count] = (by_column[:count] & 0xFE) | bits[:count]
    out[..., 3] = by_column.reshape(out.shape[1], out.shape[0]).T
    return Image.fromarray(out, "RGBA")


def _js_round(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5)


@dataclass
class Job:
    """What an inpainting request sends and how the result is composited back."""

    canvas: Image.Image
    latent: Image.Image
    focus: Focus | None
    width: int
    height: int
    image: bytes
    mask: bytes
    base: Image.Image
    feather: Image.Image
    box: tuple[int, int, int, int]
    warnings: list[str] = field(default_factory=list)

    def remap(self, x: float, y: float) -> tuple[float, float]:
        """Map character centres from the canvas into the crop, clamped to 0-1."""
        if self.focus is None:
            return x, y
        f = self.focus
        return (
            min(1.0, max(0.0, (x * self.canvas.width - f.x) / f.width)),
            min(1.0, max(0.0, (y * self.canvas.height - f.y) / f.height)),
        )

    def paste(self, data: bytes) -> bytes:
        result = Image.open(io.BytesIO(data))
        result.load()
        merged = composite(self.base, result, self.feather, self.box)
        return images.copy_text_chunks(data, images.to_png(merged))

    def preview(self) -> Image.Image:
        """Overlay the repainted area as the server sees it (8 px cells) in red, and the focus crop in green."""
        cells = resample.nearest(self.latent, self.latent.width * LATENT_SCALE, self.latent.height * LATENT_SCALE)
        coverage = Image.new("L", self.canvas.size, 0)
        coverage.paste(cells, (0, 0))
        out = self.canvas.convert("RGBA")
        overlay = Image.new("RGBA", out.size, (*_PREVIEW_MASK_COLOR, 0))
        overlay.putalpha(coverage.point(lambda v: _PREVIEW_MASK_ALPHA if v else 0))
        out = Image.alpha_composite(out, overlay)
        if self.focus is not None:
            f = self.focus
            line = max(2, min(out.size) // 200)
            ImageDraw.Draw(out).rectangle(
                (f.x, f.y, f.x + f.width - 1, f.y + f.height - 1), outline=_PREVIEW_FOCUS_COLOR, width=line
            )
        return out

    def to_json(self) -> dict:
        return {
            "canvas": f"{self.canvas.width}x{self.canvas.height}",
            "focus": self.focus.to_json() if self.focus else None,
        }


def prepare(
    source: Image.Image,
    *,
    mask: Image.Image | None,
    regions: list[Region],
    grow_pixels: int,
    expand: tuple[int, int, int, int] | None,
    size: tuple[int, int] | None,
    current: tuple[int, int],
    focus: bool | None,
    padding: int,
    keep_alpha: bool,
    segment: Segment | None = None,
) -> Job:
    """Build the canvas and mask, decide on focus, and prepare the base image, mask and feather mask.
    size applies without focus; by default the canvas is sent as is within the limit, else scaled like an import."""
    canvas, coverage, warnings = _coverage(source, mask, regions, expand, segment)
    coverage = grow(coverage, grow_pixels)
    latent = images.latent_mask(coverage, canvas.width, canvas.height)

    plan, focus_warnings = choose_focus(latent, canvas.size, padding, False if size else focus)
    warnings += focus_warnings
    if plan is not None:
        width, height = plan.request_width, plan.request_height
        crop = canvas.crop((plan.x, plan.y, plan.x + plan.width, plan.y + plan.height))
        sent = images.source_canvas(crop, width, height, keep_alpha)
        request_latent = resample.nearest(latent.crop(plan.latent_box), width // LATENT_SCALE, height // LATENT_SCALE)
        base, box = canvas, (plan.x, plan.y, plan.width, plan.height)
    else:
        width, height = _unfocused_size(canvas, size, current, warnings)
        sent = images.source_canvas(canvas, width, height, keep_alpha)
        request_latent = resample.nearest(latent, width // LATENT_SCALE, height // LATENT_SCALE)
        base, box = sent, (0, 0, width, height)
    if request_latent.getextrema()[1] == 0:
        raise errors.NaiError(errors.EMPTY_MASK, "the mask vanishes at generation size; enlarge it or use --focus")
    return Job(
        canvas=canvas,
        latent=latent,
        focus=plan,
        width=width,
        height=height,
        image=images.to_png(sent),
        mask=images.mask_request(request_latent, width, height),
        base=base,
        feather=feather(request_latent),
        box=box,
        warnings=warnings,
    )


def _coverage(
    source: Image.Image,
    mask: Image.Image | None,
    regions: list[Region],
    expand: tuple[int, int, int, int] | None,
    segment: Segment | None,
) -> tuple[Image.Image, Image.Image, list[str]]:
    """The editor canvas and the area to repaint on it: the painted mask, then the regions in order."""
    src = images.clean_source(source)
    warnings: list[str] = []
    fitted = editor_canvas_size(*src.size)
    if fitted != src.size:
        warnings.append(
            f"source {src.width}x{src.height} is resized to a {fitted[0]}x{fitted[1]} canvas "
            f"(longest side at most {EDITOR_MAX_SIDE}, multiples of {GRID}); the output has this size"
        )
        src = resample.lanczos(src, *fitted)
    if expand:
        canvas, coverage, offset = expand_canvas(src, expand)
    else:
        canvas, coverage, offset = src, Image.new("L", src.size, 0), (0, 0)
    if mask is not None:
        painted = images.mask_coverage(mask)
        if painted.size != src.size:
            painted = resample.nearest(painted, *src.size)
        area = (offset[0], offset[1], offset[0] + src.width, offset[1] + src.height)
        coverage.paste(ImageChops.lighter(coverage.crop(area), _binary(painted)), area[:2])
    return canvas, draw_regions(coverage, regions, canvas, segment), warnings


def _unfocused_size(
    canvas: Image.Image, size: tuple[int, int] | None, current: tuple[int, int], warnings: list[str]
) -> tuple[int, int]:
    if size:
        return size
    if canvas.width * canvas.height <= MAX_PIXELS:
        return canvas.size
    width, height = images.import_size(canvas.width, canvas.height, current)
    warnings.append(
        f"canvas {canvas.width}x{canvas.height} exceeds the generation limit and is sent at {width}x{height}, "
        f"which is also the output size. Use --focus to keep the size (above 1 MP costs Anlas) or shrink the mask"
    )
    return width, height
