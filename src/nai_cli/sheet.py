"""A contact sheet of a grid's images, labelled along both axes."""

from dataclasses import dataclass, replace

from PIL import Image, ImageDraw, ImageFont

from nai_cli import share

# Each image is shrunk to fit this box
CELL_EDGE = 512
_GAP = 8
_FONT_SIZE = 20
_PAD = 10
_BACKGROUND = (24, 24, 24)
_MISSING = (60, 60, 60)
_INK = (235, 235, 235)


@dataclass(frozen=True)
class Sheet:
    # Row by row; None where a cell has no image
    images: list[list[Image.Image | None]]
    column_labels: list[str]
    row_labels: list[str]
    x_title: str
    y_title: str


def render(s: Sheet) -> Image.Image:
    font = share.load_font(None, _FONT_SIZE)
    s = _drawable(s, font)
    present = [img for row in s.images for img in row if img is not None]
    scale = min(1.0, *(CELL_EDGE / max(img.size) for img in present)) if present else 1.0
    cell_w = max((round(img.width * scale) for img in present), default=CELL_EDGE)
    cell_h = max((round(img.height * scale) for img in present), default=CELL_EDGE)
    line = _FONT_SIZE + _PAD
    top = line * (bool(s.x_title) + any(s.column_labels))
    left = max((_width(font, t) for t in (*s.row_labels, s.y_title) if t), default=0)
    left = left + 2 * _PAD if left else 0
    columns, rows = len(s.images[0]), len(s.images)
    canvas = Image.new(
        "RGB", (left + columns * (cell_w + _GAP) + _GAP, top + rows * (cell_h + _GAP) + _GAP), _BACKGROUND
    )
    draw = ImageDraw.Draw(canvas)
    if s.x_title:
        draw.text((left + _GAP, _PAD // 2), s.x_title, font=font, fill=_INK)
    if s.y_title:
        draw.text((_PAD, _PAD // 2), s.y_title, font=font, fill=_INK)
    for column, label in enumerate(s.column_labels):
        draw.text((left + _GAP + column * (cell_w + _GAP), top - line + _PAD // 2), label, font=font, fill=_INK)
    for row, images in enumerate(s.images):
        y = top + _GAP + row * (cell_h + _GAP)
        if row < len(s.row_labels) and s.row_labels[row]:
            draw.text((_PAD, y + cell_h // 2 - _FONT_SIZE // 2), s.row_labels[row], font=font, fill=_INK)
        for column, img in enumerate(images):
            x = left + _GAP + column * (cell_w + _GAP)
            if img is None:
                draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=_MISSING)
                continue
            thumb = img.convert("RGB").resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS
            )
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
    return canvas


def _drawable(s: Sheet, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Sheet:
    # The bitmap font of a Pillow without FreeType raises on text beyond Latin-1, after the cells were paid for
    if isinstance(font, ImageFont.FreeTypeFont):
        return s

    def latin(text: str) -> str:
        return text.encode("latin-1", "replace").decode("latin-1")

    return replace(
        s,
        column_labels=[latin(t) for t in s.column_labels],
        row_labels=[latin(t) for t in s.row_labels],
        x_title=latin(s.x_title),
        y_title=latin(s.y_title),
    )


def _width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    left, _, right, _ = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    return round(right - left)
