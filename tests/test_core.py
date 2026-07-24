"""Unit tests for the pure `core` logic.

Fast tests (config, cropping, video aggregation) run with no weights. The end-to-end model
test is marked `integration` and skipped unless RUN_INTEGRATION=1.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import settings
from core.detection import _crop_face
from core.inference import aggregate_video

# --- config --------------------------------------------------------------------------------

def test_settings_defaults():
    assert settings.img_size == 224
    assert settings.classification_threshold == 0.5
    assert settings.hf_repo_id == "Parimal/Resnet50_best"
    assert "png" in settings.allowed_img_ext
    assert "mp4" in settings.allowed_vid_ext


# --- cropping ------------------------------------------------------------------------------

def test_crop_face_stays_in_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box near the corner — expanded crop must clamp to the frame, not go negative/overflow.
    crop = _crop_face(frame, (0, 0, 20, 20), scale=2.0)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_crop_face_center_region():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = _crop_face(frame, (80, 80, 120, 120), scale=1.0)
    # 40px box, scale 1.0 -> ~40px square crop.
    assert 30 <= crop.shape[0] <= 50
    assert 30 <= crop.shape[1] <= 50


# --- video aggregation (the multi-factor verdict rule) -------------------------------------

def test_aggregate_video_flags_fake():
    # High average and many highly-confident frames -> FAKE.
    p_fakes = [0.95, 0.9, 0.85, 0.92, 0.88]
    agg = aggregate_video(p_fakes)
    assert agg["is_fake"] is True
    assert agg["label"] == "fake"
    assert agg["num_frames"] == 5
    assert agg["highly_fake_count"] == 5


def test_aggregate_video_real_when_avg_low():
    p_fakes = [0.1, 0.2, 0.15, 0.05, 0.3]
    agg = aggregate_video(p_fakes)
    assert agg["is_fake"] is False
    assert agg["label"] == "real"


def test_aggregate_video_real_when_few_smoking_guns():
    # High-ish average but not enough highly-confident (>=0.80) frames -> REAL by the rule.
    p_fakes = [0.75, 0.72, 0.71, 0.73, 0.70]
    agg = aggregate_video(p_fakes)
    assert agg["highly_fake_count"] == 0
    assert agg["is_fake"] is False


def test_aggregate_video_metrics_match_numpy():
    p_fakes = [0.2, 0.8, 0.6]
    agg = aggregate_video(p_fakes)
    assert agg["mean_p_fake"] == pytest.approx(float(np.mean(p_fakes)))
    assert agg["max_p_fake"] == pytest.approx(0.8)


# --- end-to-end (needs real weights) -------------------------------------------------------

@pytest.mark.integration
def test_analyze_image_end_to_end(synthetic_face_image):
    from core.inference import analyze_image

    result = analyze_image(synthetic_face_image)
    assert result["label"] in {"real", "fake"}
    assert 0.0 <= result["p_fake"] <= 1.0
    assert result["p_real"] == pytest.approx(1.0 - result["p_fake"], abs=1e-6)
    assert result["confidence"] >= 0.5


@pytest.mark.integration
def test_onnx_matches_torch(synthetic_face_image):
    """The ONNX (fp32) path must produce the same probabilities as the torch path."""
    from pathlib import Path

    from core.config import settings
    from core.inference import predict_pil_images

    if not Path(settings.onnx_model_path).exists():
        pytest.skip("ONNX model not exported; run `python -m scripts.export_onnx` first")

    settings.use_onnx = False
    torch_res = predict_pil_images([synthetic_face_image], tta=2)[0]
    try:
        settings.use_onnx = True
        onnx_res = predict_pil_images([synthetic_face_image], tta=2)[0]
    finally:
        settings.use_onnx = False

    assert onnx_res["p_fake"] == pytest.approx(torch_res["p_fake"], abs=1e-3)
