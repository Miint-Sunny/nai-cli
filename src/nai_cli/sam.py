"""Masks of the object under a point, from EfficientViT-SAM L0 run on this computer."""

from typing import Any

import numpy as np
from PIL import Image

from nai_cli import errors

ENCODER_SIZE = 512
# The decoder reads points on a canvas of this longest side
DECODER_SIZE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def segment(image: Image.Image, points: list[tuple[float, float]], encoder: Any, decoder: Any) -> Image.Image:
    """A mask the size of image, white over the object the points fall on; points are 0-1 fractions. The encoder
    and decoder are the models local_models.sam loads."""
    width, height = image.size
    embedding = encoder.run(["image_embeddings"], {"input_image": _encoder_input(image)})[0]
    scale_w, scale_h = _longest(width, height, DECODER_SIZE)
    coords = np.array([[[x * scale_w, y * scale_h] for x, y in points]], np.float32)
    labels = np.ones((1, len(points)), np.float32)
    feeds = {"image_embeddings": embedding, "point_coords": coords, "point_labels": labels}
    masks = decoder.run(["masks"], feeds)[0]
    if np.ndim(masks) < 2 or 0 in np.shape(masks):
        raise errors.NaiError(errors.BAD_ARGS, f"{decoder.name} gave no mask: shape {np.shape(masks)}")
    low = np.asarray(masks, np.float32).reshape(-1, *np.shape(masks)[-2:])[0]
    square = _bilinear(low, DECODER_SIZE, DECODER_SIZE)
    logits = _bilinear(square[:scale_h, :scale_w], width, height)
    return Image.fromarray(np.where(logits > 0, 255, 0).astype(np.uint8), "L")


def _longest(width: int, height: int, side: int) -> tuple[int, int]:
    scale = side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _encoder_input(image: Image.Image) -> np.ndarray:
    """The image fitted into the top left of a 512 square, normalised; the rest stays at the mean."""
    w, h = _longest(*image.size, ENCODER_SIZE)
    rgb = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), np.float32) / 255
    tensor = np.zeros((3, ENCODER_SIZE, ENCODER_SIZE), np.float32)
    tensor[:, :h, :w] = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)
    return tensor[None]


def _bilinear(values: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resized with pixel centres aligned, clamped at the edges."""
    rows, top_w = _axis(values.shape[0], height)
    cols, left_w = _axis(values.shape[1], width)
    top = values[rows[0]][:, cols[0]] * (1 - left_w) + values[rows[0]][:, cols[1]] * left_w
    bottom = values[rows[1]][:, cols[0]] * (1 - left_w) + values[rows[1]][:, cols[1]] * left_w
    return top * (1 - top_w[:, None]) + bottom * top_w[:, None]


def _axis(source: int, target: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    at = (np.arange(target) + 0.5) * (source / target) - 0.5
    low = np.floor(at)
    weight = np.clip(at - low, 0, 1)
    low = low.astype(int)
    return (np.clip(low, 0, source - 1), np.clip(low + 1, 0, source - 1)), weight
