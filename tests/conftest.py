"""Shared pytest fixtures and markers."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that load real weights / run the model (need network + ~283MB download)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless RUN_INTEGRATION=1 is set."""
    if os.getenv("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_INTEGRATION=1 to run (downloads weights, runs model)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def synthetic_face_image() -> Image.Image:
    """A small synthetic RGB image with a crude face-like blob (no real face guaranteed)."""
    arr = np.full((256, 256, 3), 220, dtype=np.uint8)
    arr[80:180, 80:180] = (200, 170, 150)  # face-ish rectangle
    arr[110:125, 105:120] = (40, 40, 40)   # left eye
    arr[110:125, 140:155] = (40, 40, 40)   # right eye
    arr[150:160, 115:145] = (120, 60, 60)  # mouth
    return Image.fromarray(arr)
