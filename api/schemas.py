"""Pydantic request/response models for the API (typed, self-documenting in OpenAPI)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImagePrediction(BaseModel):
    label: str = Field(..., description="'fake' if p_fake >= threshold, else 'real'", examples=["fake"])
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted label")
    p_fake: float = Field(..., ge=0, le=1)
    p_real: float = Field(..., ge=0, le=1)
    faces_detected: int = Field(..., ge=0)
    processing_ms: int = Field(..., ge=0)


class FrameScore(BaseModel):
    frame_index: int
    timestamp_s: float | None = None
    p_fake: float = Field(..., ge=0, le=1)


class VideoAggregate(BaseModel):
    label: str
    is_fake: bool
    mean_p_fake: float
    max_p_fake: float
    highly_fake_count: int
    highly_fake_rate: float
    num_frames: int


class VideoPrediction(BaseModel):
    aggregate: VideoAggregate
    frames: list[FrameScore]
    processing_ms: int = Field(..., ge=0)


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    error: str
