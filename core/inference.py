"""Inference pipeline — framework-free.

Wraps the classification, TTA, video sampling, and aggregation logic that already worked in
the original ``app.py`` (thresholds and rules are unchanged). Both the Streamlit UI and the
FastAPI service call these functions; neither re-derives the logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image

from .config import settings
from .detection import detect_largest_face
from .logging_config import get_logger
from .models import get_transforms, load_classifier
from .onnx_runtime import onnx_available, run_onnx

logger = get_logger(__name__)


class NoFaceDetectedError(Exception):
    """Raised by high-level helpers when no usable face is found and a face is required."""


@torch.no_grad()
def predict_pil_images(images: list[Image.Image], tta: int | None = None) -> list[dict]:
    """Classify a batch of RGB PIL images; returns ``[{"p_fake", "p_real"}, ...]``.

    Uses test-time augmentation (original + horizontal flip) averaged over ``tta`` passes.
    """
    if not images:
        return []
    if tta is None:
        tta = settings.tta

    tfm = get_transforms()
    batch = torch.stack([tfm(img) for img in images])  # CPU tensor; moved to device below if torch

    # Choose the inference backend: ONNX Runtime (fast/light) if available, else torch.
    use_onnx = onnx_available()
    if use_onnx:
        # ONNX path avoids loading the torch model entirely. The exported checkpoint's class map
        # is {"fake": 0, "real": 1}; output logit order is preserved from export.
        model = device = None
        class_to_idx = {"fake": 0, "real": 1}
    else:
        model, class_to_idx, device = load_classifier()
        batch = batch.to(device)

    all_probs = []
    for i in range(max(1, tta)):
        current = torch.flip(batch, dims=[-1]) if i > 0 else batch
        if use_onnx:
            probs = run_onnx(current.cpu().numpy())
        else:
            logits = model(current)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    i_fake = class_to_idx.get("fake", 0)

    results = []
    for p in avg_probs:
        p_fake = float(p[i_fake])
        results.append({"p_fake": p_fake, "p_real": 1.0 - p_fake})
    return results


def analyze_image(pil_image: Image.Image, tta: int | None = None, require_face: bool = False) -> dict:
    """Full single-image pipeline: detect largest face -> crop -> classify.

    When ``require_face`` is False (Streamlit behavior) the full image is used if no face is
    found. When True (API behavior) a ``NoFaceDetectedError`` is raised instead.
    """
    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    face_bgr = detect_largest_face(bgr)
    faces_detected = 1 if face_bgr is not None else 0

    if face_bgr is None and require_face:
        raise NoFaceDetectedError("No face detected in the image.")

    crop = (
        Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        if face_bgr is not None
        else pil_image.convert("RGB")
    )

    res = predict_pil_images([crop], tta=tta)[0]
    p_fake = res["p_fake"]
    is_fake = p_fake >= settings.classification_threshold
    return {
        "label": "fake" if is_fake else "real",
        "confidence": p_fake if is_fake else 1.0 - p_fake,
        "p_fake": p_fake,
        "p_real": res["p_real"],
        "faces_detected": faces_detected,
    }


def sample_video_faces(
    video_path: str | Path,
    conf_threshold: float | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
):
    """Sample frames from a video and return (face_crops, frame_indices, fps).

    Samples ~``video_target_fps`` frames per second, keeps only faces at least
    ``min_face_size`` px, and stops at ``video_max_frames``. ``progress_callback`` (if given)
    is called with (frames_read, faces_found, total_frames) so UIs can show progress without
    coupling this function to any framework.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, int(fps / settings.video_target_fps)) if fps and fps > 0 else 1

        face_crops: list[Image.Image] = []
        frame_indices: list[int] = []

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % stride == 0:
                face_bgr = detect_largest_face(frame, conf_threshold)
                if (
                    face_bgr is not None
                    and face_bgr.shape[0] >= settings.min_face_size
                    and face_bgr.shape[1] >= settings.min_face_size
                ):
                    face_crops.append(Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)))
                    frame_indices.append(i)
            if len(face_crops) >= settings.video_max_frames:
                break
            if progress_callback is not None:
                progress_callback(i, len(face_crops), total_frames)
            i += 1
    finally:
        cap.release()

    return face_crops, frame_indices, fps


def aggregate_video(p_fakes) -> dict:
    """Aggregate per-frame p(fake) scores into a verdict using the original multi-factor rule."""
    p_fakes = np.asarray(p_fakes, dtype=float)
    n = len(p_fakes)
    mean_p_fake = float(np.mean(p_fakes))
    max_p_fake = float(np.max(p_fakes))
    highly_fake_count = int(np.sum(p_fakes >= settings.high_conf_threshold))
    highly_fake_rate = highly_fake_count / n if n else 0.0

    # FAKE only if the average is high AND there are enough "smoking gun" frames.
    is_fake = (mean_p_fake >= settings.video_avg_threshold) and (
        highly_fake_rate >= settings.highly_fake_rate_threshold
    )

    return {
        "label": "fake" if is_fake else "real",
        "is_fake": is_fake,
        "mean_p_fake": mean_p_fake,
        "max_p_fake": max_p_fake,
        "highly_fake_count": highly_fake_count,
        "highly_fake_rate": highly_fake_rate,
        "num_frames": n,
    }


def analyze_video(
    video_path: str | Path,
    tta: int | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict:
    """Full video pipeline: sample faces -> classify -> aggregate.

    Returns the aggregate verdict plus per-frame data (and the PIL face crops, which UIs use
    for the "most suspicious frames" gallery). Raises ``NoFaceDetectedError`` if no usable
    faces are found.
    """
    face_crops, frame_indices, fps = sample_video_faces(
        video_path, progress_callback=progress_callback
    )
    if not face_crops:
        raise NoFaceDetectedError("No high-quality faces detected in the video.")

    preds = predict_pil_images(face_crops, tta=tta)
    p_fakes = [r["p_fake"] for r in preds]
    aggregate = aggregate_video(p_fakes)

    return {
        "aggregate": aggregate,
        "fps": fps,
        "frame_indices": frame_indices,
        "p_fakes": p_fakes,
        "face_crops": face_crops,
    }
