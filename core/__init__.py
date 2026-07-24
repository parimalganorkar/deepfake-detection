"""core — pure, framework-free deepfake-detection logic.

Imported by both the Streamlit app (``app.py``) and the FastAPI service (``api/``). Nothing in
this package imports ``streamlit`` or ``fastapi``.
"""

from __future__ import annotations

from .config import settings
from .detection import detect_largest_face
from .inference import (
    NoFaceDetectedError,
    aggregate_video,
    analyze_image,
    analyze_video,
    predict_pil_images,
    sample_video_faces,
)
from .models import (
    get_device,
    get_haar_face_detector,
    get_transforms,
    get_yolo_face_detector,
    load_classifier,
    warm_up,
)

__all__ = [
    "settings",
    "detect_largest_face",
    "predict_pil_images",
    "analyze_image",
    "analyze_video",
    "sample_video_faces",
    "aggregate_video",
    "NoFaceDetectedError",
    "load_classifier",
    "get_transforms",
    "get_yolo_face_detector",
    "get_haar_face_detector",
    "get_device",
    "warm_up",
]
