"""Model loading & weight management — framework-free (no Streamlit, no FastAPI).

Weights are auto-downloaded from the Hugging Face Hub and cached locally, so the app works
out-of-the-box on any machine (fixing the old Windows-only ``resnet50_mixed_export\\...`` path
and the manual weight-placement step). Loaders are memoized with ``functools.lru_cache`` so
each model is built exactly once per process, whether the caller is Streamlit or the API.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import models, transforms
from ultralytics import YOLO

from .config import settings
from .logging_config import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Return the inference device, honoring the ``force_cpu`` setting."""
    if settings.force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_weight(filename: str) -> Path:
    """Resolve a weight file from the configured HF repo (cached after first download)."""
    logger.info("Resolving weight '%s' from HF repo '%s'", filename, settings.hf_repo_id)
    path = hf_hub_download(repo_id=settings.hf_repo_id, filename=filename)
    return Path(path)


@lru_cache(maxsize=1)
def load_classifier() -> tuple[nn.Module, dict, torch.device]:
    """Load the fine-tuned ResNet-50 classifier + class map onto the inference device."""
    ckpt_path = download_weight(settings.classifier_filename)
    # weights_only=False is REQUIRED: the checkpoint holds non-tensor data (class_to_idx),
    # and PyTorch 2.6+ flipped this default. Do not change it.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model"])
    model.eval()

    class_to_idx = ckpt.get("class_to_idx", {"fake": 0, "real": 1})
    device = get_device()
    model.to(device)

    logger.info("Classifier loaded on %s (class_to_idx=%s)", device, class_to_idx)
    return model, class_to_idx, device


@lru_cache(maxsize=1)
def get_transforms() -> transforms.Compose:
    """Return the eval-time image transforms for the classifier."""
    return transforms.Compose(
        [
            transforms.Resize((settings.img_size, settings.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(list(settings.mean), list(settings.std)),
        ]
    )


@lru_cache(maxsize=1)
def get_yolo_face_detector() -> YOLO:
    """Load the YOLOv8-face detector (single canonical definition)."""
    weight = download_weight(settings.yolo_filename)
    return YOLO(str(weight))


@lru_cache(maxsize=1)
def get_haar_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's Haar cascade as a fallback face detector."""
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(haar_path)


def warm_up() -> None:
    """Eagerly load every model — used by the API lifespan to fail fast at startup."""
    load_classifier()
    get_transforms()
    get_yolo_face_detector()
    get_haar_face_detector()
