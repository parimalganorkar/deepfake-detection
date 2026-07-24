"""Central configuration for the deepfake-detection system.

All tunables live here as a single ``settings`` object built with pydantic-settings, so any
value can be overridden via environment variables (prefix ``DEEPFAKE_``) or a local ``.env``
file without touching code. This replaces the scattered module-level constants that used to
live at the top of ``app.py``.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPFAKE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model weights (Hugging Face Hub) ---
    hf_repo_id: str = "Parimal/Resnet50_best"
    classifier_filename: str = "resnet50_best.pt"
    yolo_filename: str = "yolov8n-face.pt"

    # --- Image / model input ---
    img_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- Detection & classification thresholds (unchanged from the original app) ---
    face_detection_confidence: float = 0.35  # YOLOv8-face confidence
    classification_threshold: float = 0.50   # tipping point for fake vs. real
    min_face_size: int = 64                  # min px for a valid video face crop

    # --- Video aggregation rules (unchanged from the original app) ---
    video_avg_threshold: float = 0.70        # mean p(fake) needed to consider a video fake
    high_conf_threshold: float = 0.80        # p(fake) for a "highly confident" fake frame
    highly_fake_rate_threshold: float = 0.25  # min rate of highly-confident frames to flag fake

    # --- Video sampling ---
    video_target_fps: int = 5                # aim for ~5 sampled frames per second
    video_max_frames: int = 200             # cap on analyzed frames (memory bound)

    # --- Inference ---
    tta: int = 2                             # test-time augmentation passes (orig + h-flip)
    force_cpu: bool = False                  # pin to CPU even if CUDA is available

    # --- ONNX / CPU optimization ---
    use_onnx: bool = False                   # use the ONNX Runtime path if an exported model exists
    onnx_dir: str = "onnx_models"            # where exported .onnx files live (gitignored)
    onnx_model_filename: str = "resnet50.onnx"        # fp32 export
    onnx_quant_filename: str = "resnet50.quant.onnx"  # dynamically quantized (int8) export
    # Default False: for this conv-heavy ResNet-50, dynamic int8 quantization is ~4x smaller but
    # SLOWER on CPU (dynamic quant only accelerates MatMul/Linear, not conv). fp32 ONNX is 3.6x
    # faster than torch — see docs/onnx_benchmark.md. Set True to trade speed for a smaller model.
    onnx_use_quantized: bool = False

    @property
    def onnx_model_path(self) -> str:
        return f"{self.onnx_dir}/{self.onnx_model_filename}"

    @property
    def onnx_quant_path(self) -> str:
        return f"{self.onnx_dir}/{self.onnx_quant_filename}"

    # --- Uploads (used by the API) ---
    allowed_img_ext: set[str] = {"png", "jpg", "jpeg"}
    allowed_vid_ext: set[str] = {"mp4", "mov", "avi", "mkv"}
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_video_bytes: int = 50 * 1024 * 1024  # 50 MB

    # --- API hardening (used in Phase 1) ---
    api_key: str | None = None               # if set, required on predict endpoints
    rate_limit: str = "30/minute"            # slowapi limit string

    # --- Logging ---
    log_level: str = "INFO"
    json_logs: bool = True


settings = Settings()
