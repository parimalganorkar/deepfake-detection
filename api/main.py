"""FastAPI inference service.

Loads both models once at startup (lifespan), then serves image/video deepfake predictions.
All heavy lifting is delegated to ``core`` — this module is transport + validation + observability.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from PIL import Image
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config import settings
from core.inference import NoFaceDetectedError, analyze_image, analyze_video
from core.logging_config import get_logger
from core.models import warm_up

from .metrics import PREDICTIONS, REQUEST_COUNT, REQUEST_LATENCY, render_latest
from .schemas import (
    ErrorResponse,
    FrameScore,
    HealthResponse,
    ImagePrediction,
    ReadyResponse,
    VideoAggregate,
    VideoPrediction,
)
from .security import limiter, require_api_key

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup so the first request isn't slow (and fail fast if broken)."""
    logger.info("Loading models at startup (weights auto-download from Hugging Face if needed)...")
    app.state.model_loaded = False
    warm_up()
    app.state.model_loaded = True
    logger.info("Models loaded — API ready.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Deepfake Detection API",
    version="1.0.0",
    description="YOLOv8-face + fine-tuned ResNet-50 deepfake classifier. Reuses the exact "
    "inference logic behind the Streamlit demo.",
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
app.state.limiter = limiter


# ---------------------------------------------------------------------------
# Error handling — uniform {"error": "..."} shape; never leak stack traces.
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": f"Rate limit exceeded: {exc.detail}"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


# ---------------------------------------------------------------------------
# Metrics middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUEST_COUNT.labels(endpoint=endpoint, method=request.method, status=str(response.status_code)).inc()
    return response


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_ext(filename: str | None, allowed: set[str]) -> None:
    ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: {sorted(allowed)}.")


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` (+1 to detect overflow); 413 if the upload is too large."""
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds the size limit of {max_bytes // (1024 * 1024)} MB.")
    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness probe — cheap, does not touch the models."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
async def ready(request: Request):
    """Readiness probe — true once models are loaded onto app.state."""
    loaded = bool(getattr(request.app.state, "model_loaded", False))
    return ReadyResponse(status="ready" if loaded else "loading", model_loaded=loaded)


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics():
    payload, content_type = render_latest()
    return Response(content=payload, media_type=content_type)


@app.post(
    "/predict",
    response_model=ImagePrediction,
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def predict(request: Request, file: UploadFile | None = File(default=None)):
    """Classify a single image. Detect the largest face, crop, and run the classifier (TTA)."""
    if file is None:
        raise HTTPException(400, "No file provided.")
    _validate_ext(file.filename, settings.allowed_img_ext)
    contents = await _read_limited(file, settings.max_image_bytes)

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Corrupt or unreadable image.") from None

    start = time.perf_counter()
    try:
        result = analyze_image(img, require_face=True)
    except NoFaceDetectedError:
        raise HTTPException(400, "No face detected in the image.") from None

    result["processing_ms"] = int((time.perf_counter() - start) * 1000)
    PREDICTIONS.labels(endpoint="predict", label=result["label"]).inc()
    return result


@app.post(
    "/predict_video",
    response_model=VideoPrediction,
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
async def predict_video(request: Request, file: UploadFile | None = File(default=None)):
    """Classify a video by sampling frames, scoring each, and aggregating the verdict."""
    if file is None:
        raise HTTPException(400, "No file provided.")
    _validate_ext(file.filename, settings.allowed_vid_ext)
    contents = await _read_limited(file, settings.max_video_bytes)

    suffix = Path(file.filename).suffix or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        start = time.perf_counter()
        try:
            result = analyze_video(tmp_path)
        except NoFaceDetectedError:
            raise HTTPException(400, "No high-quality face detected in the video.") from None
        except ValueError:
            raise HTTPException(400, "Could not open or read the video file.") from None

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        agg = result["aggregate"]
        fps = result["fps"]
        frames = [
            FrameScore(
                frame_index=fi,
                timestamp_s=(fi / fps if fps and fps > 0 else None),
                p_fake=pf,
            )
            for fi, pf in zip(result["frame_indices"], result["p_fakes"], strict=False)
        ]
        PREDICTIONS.labels(endpoint="predict_video", label=agg["label"]).inc()
        return VideoPrediction(
            aggregate=VideoAggregate(**agg), frames=frames, processing_ms=elapsed_ms
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
