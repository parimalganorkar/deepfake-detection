"""API tests using FastAPI's TestClient.

Models are monkeypatched out (``warm_up`` -> no-op, ``analyze_image`` -> canned result) so these
tests run in milliseconds with no weight download — suitable for CI. The real model path is
covered by the ``integration`` test in ``test_core.py``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main as api_main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "warm_up", lambda: None)

    def fake_analyze_image(img, tta=None, require_face=False):
        return {
            "label": "fake",
            "confidence": 0.9,
            "p_fake": 0.9,
            "p_real": 0.1,
            "faces_detected": 1,
        }

    monkeypatch.setattr(api_main, "analyze_image", fake_analyze_image)
    with TestClient(api_main.app) as c:
        yield c


def _png_bytes(size=(32, 32)) -> bytes:
    img = Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "/docs" in r.headers["location"]


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"deepfake_requests_total" in r.content


def test_predict_ok(client):
    r = client.post("/predict", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "fake"
    assert body["p_fake"] == pytest.approx(0.9)
    assert "processing_ms" in body


def test_predict_no_file_is_400(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert "error" in r.json()


def test_predict_bad_extension_is_400(client):
    r = client.post("/predict", files={"file": ("t.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "error" in r.json()


def test_predict_corrupt_image_is_400(client):
    r = client.post("/predict", files={"file": ("t.png", b"not-a-real-png", "image/png")})
    assert r.status_code == 400
    assert "error" in r.json()


def test_predict_oversize_is_413(client):
    from core.config import settings

    big = b"\x00" * (settings.max_image_bytes + 16)
    r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413
    assert "error" in r.json()


def test_api_key_enforced_when_configured(client, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "api_key", "secret")

    # Missing key -> 401
    r = client.post("/predict", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 401
    assert "error" in r.json()

    # Correct key -> 200
    r = client.post(
        "/predict",
        files={"file": ("t.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "secret"},
    )
    assert r.status_code == 200
