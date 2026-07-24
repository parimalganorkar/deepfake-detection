"""Face detection — framework-free.

YOLOv8-face is the primary detector; OpenCV's Haar cascade is the fallback when YOLO finds
nothing. Errors are logged (not surfaced via ``st.warning``) so this module is reusable from
both the Streamlit app and the FastAPI service.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import settings
from .logging_config import get_logger
from .models import get_haar_face_detector, get_yolo_face_detector

logger = get_logger(__name__)


def _crop_face(frame_bgr: np.ndarray, box, scale: float = 1.3) -> np.ndarray:
    """Crop a square region around a face box, expanded by ``scale`` and clamped to bounds."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(x2 - x1, y2 - y1) * scale)
    x1_new, y1_new = max(0, cx - side // 2), max(0, cy - side // 2)
    x2_new, y2_new = min(w, x1_new + side), min(h, y1_new + side)
    return frame_bgr[y1_new:y2_new, x1_new:x2_new]


def detect_largest_face(frame_bgr: np.ndarray, conf_threshold: float | None = None):
    """Return the largest face crop (BGR) in ``frame_bgr``, or ``None`` if no face is found.

    Tries YOLOv8-face first, then falls back to Haar cascade.
    """
    if conf_threshold is None:
        conf_threshold = settings.face_detection_confidence

    # --- Primary: YOLOv8-face ---
    try:
        yolo = get_yolo_face_detector()
        device = 0 if (torch.cuda.is_available() and not settings.force_cpu) else "cpu"
        res = yolo.predict(frame_bgr[..., ::-1], conf=conf_threshold, verbose=False, device=device)

        boxes = []
        if res and res[0].boxes:
            for box in res[0].boxes:
                xyxy = box.xyxy.cpu().numpy().squeeze()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                boxes.append((xyxy, area))

        if boxes:
            largest_box, _ = max(boxes, key=lambda item: item[1])
            return _crop_face(frame_bgr, largest_box)
    except Exception as e:  # noqa: BLE001 - detector must never crash the pipeline
        logger.warning("YOLO detector failed: %s", e)

    # --- Fallback: Haar cascade (only runs if YOLO found nothing) ---
    try:
        haar = get_haar_face_detector()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 4, minSize=(64, 64))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            return _crop_face(frame_bgr, (x, y, x + w, y + h))
    except Exception as e:  # noqa: BLE001
        logger.warning("Haar cascade detector failed: %s", e)

    return None
