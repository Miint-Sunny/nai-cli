"""ONNX models run on this computer, which need the optional onnxruntime install, and the model files they read."""

import functools
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import errors, fs

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

INSTALL_HINT = 'pip install "nai-cli[local]"'


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str


_SAM_REPOSITORY = "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/6a80311888e8c2e2e92f22f554564c0be1f018ae"
# EfficientViT-SAM L0: the encoder, then the decoder
SAM = (
    Asset(
        "l0_encoder.onnx",
        f"{_SAM_REPOSITORY}/onnx/l0_encoder.onnx",
        123112917,
        "7a37ebd16e65c31253416e33897e7e6b07699127765d538e99ecbfaa3ad1e727",
    ),
    Asset(
        "l0_decoder.onnx",
        f"{_SAM_REPOSITORY}/onnx/l0_decoder.onnx",
        16470362,
        "053e72c5703736a8a8a92482c9990c4a26dde9dcb7ff26b05843cf0bc433f42f",
    ),
)


@dataclass(frozen=True)
class Model:
    """A loaded model; run checks it is the model expected, since the user picks the file."""

    name: str
    session: Any

    def inputs(self) -> list[Any]:
        return self.session.get_inputs()

    def outputs(self) -> list[Any]:
        return self.session.get_outputs()

    def run(self, outputs: list[str] | None, feeds: dict[str, Any]) -> list[Any]:
        takes = {i.name for i in self.inputs()}
        gives = {o.name for o in self.outputs()}
        if set(feeds) != takes or not set(outputs or []) <= gives:
            raise errors.NaiError(
                errors.BAD_ARGS,
                f"{self.name} is not the model expected: it takes {sorted(takes)} and gives {sorted(gives)}, "
                f"not {sorted(feeds)} and {sorted(outputs or [])}",
            )
        try:
            return self.session.run(outputs, feeds)
        except Exception as e:
            # onnxruntime raises its own exception types for inputs of the wrong type or shape
            raise errors.NaiError(errors.BAD_ARGS, f"{self.name} cannot run on this input: {e}") from e


def available() -> bool:
    return onnxruntime is not None


def load(path: Path) -> Model:
    """An inference session on the CPU, using half the cores so the computer stays usable."""
    if onnxruntime is None:
        raise errors.NaiError(errors.CONFIG_MISSING, f"local models need onnxruntime: {INSTALL_HINT}")
    if not fs.is_file(path):
        raise errors.NaiError(errors.CONFIG_MISSING, f"model not found: {path}")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
    options.log_severity_level = 3
    try:
        return Model(path.name, onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]))
    except Exception as e:
        # onnxruntime raises its own exception types for unreadable or incompatible models
        raise errors.NaiError(errors.BAD_ARGS, f"cannot load {path.name}: {e}") from e


def sam(folder: Path) -> tuple[Model, Model]:
    """The SAM encoder and decoder, kept for the run so a batch loads them once."""
    if absent := missing(folder, SAM):
        raise errors.NaiError(
            errors.CONFIG_MISSING,
            f"object regions need the SAM model, missing or incomplete {', '.join(a.name for a in absent)} in {folder}: "
            "run nai local download sam",
        )
    return _sam_models(folder)


@functools.lru_cache(maxsize=1)
def _sam_models(folder: Path) -> tuple[Model, Model]:
    encoder, decoder = SAM
    return load(folder / encoder.name), load(folder / decoder.name)


def missing(folder: Path, assets: tuple[Asset, ...]) -> list[Asset]:
    """Files absent or of another size, such as a download cut short."""
    return [a for a in assets if _size(folder / a.name) != a.size]


def unverified(folder: Path, assets: tuple[Asset, ...]) -> list[Asset]:
    """Files missing or whose hash differs, which reads every file whole."""
    absent = missing(folder, assets)
    return [a for a in assets if a in absent or _sha256(folder / a.name) != a.sha256]


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size if fs.is_file(path) else None
    except OSError:
        return None


def _sha256(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None
