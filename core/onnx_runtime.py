"""Optional ONNX Runtime inference path.

If ONNX is enabled (``settings.use_onnx``) and an exported model exists on disk, classification
runs through ONNX Runtime (CPU) instead of PyTorch — faster and lighter, especially with the
quantized model. Everything degrades gracefully: if ONNX is off or no export is present,
callers fall back to the torch path. Generate the exports with ``scripts/export_onnx.py``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings
from .logging_config import get_logger

logger = get_logger(__name__)


def _resolve_onnx_path() -> Path | None:
    """Pick the quantized model if allowed and present, else the fp32 model, else None."""
    quant = Path(settings.onnx_quant_path)
    fp32 = Path(settings.onnx_model_path)
    if settings.onnx_use_quantized and quant.exists():
        return quant
    if fp32.exists():
        return fp32
    return None


@lru_cache(maxsize=1)
def get_onnx_session():
    """Return a cached ONNX Runtime session (CPU), or None if no export is available."""
    path = _resolve_onnx_path()
    if path is None:
        logger.info("No ONNX model found; using the torch inference path.")
        return None
    import onnxruntime as ort  # imported lazily so torch-only installs don't require it

    logger.info("Loading ONNX session from %s", path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_available() -> bool:
    """True if the ONNX path is enabled and a session could be created."""
    return bool(settings.use_onnx) and get_onnx_session() is not None


def run_onnx(batch_np: np.ndarray) -> np.ndarray:
    """Run a normalized (N,3,H,W) float32 batch through ONNX; return softmax probabilities."""
    session = get_onnx_session()
    if session is None:
        raise RuntimeError("ONNX session unavailable; call onnx_available() before run_onnx().")
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch_np.astype(np.float32)})[0]
    # numpically-stable softmax over the class dimension
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
