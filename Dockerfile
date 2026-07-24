# Deepfake Detection API — CPU-only inference image.
#
# Design notes:
#   * python:3.11-slim base + CPU-only torch (requirements-api.txt pins the PyTorch CPU index).
#     GPU wheels would take this from ~1.5 GB to ~6 GB.
#   * cv2 needs libgl1 + libglib2.0-0 on slim, or it throws a cryptic ImportError on first use.
#   * ultralytics hard-requires the non-headless `opencv-python`, which collides with
#     `opencv-python-headless`. We reinstall headless-only so exactly one cv2 owns the package.
#   * Model weights are NOT baked in — they download from the Hugging Face Hub at startup into
#     HF_HOME, which should be a mounted volume so restarts don't re-download ~289 MB.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

# Runtime libs OpenCV needs on slim images
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so this layer caches across code changes
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt \
    # ultralytics drags in non-headless opencv; force a clean headless-only install
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless>=4.8.1.78,<5.0" \
    && python -c "import cv2; print('cv2 OK', cv2.__version__)"

# Application code — only what the API actually needs (no Streamlit, no notebook, no tests)
COPY core/ ./core/
COPY api/ ./api/

# Run as a non-root user; HF cache dir must be writable by it
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /cache/huggingface \
    && chown -R appuser:appuser /app /cache
USER appuser

EXPOSE 8000

# start-period is generous: the first boot downloads weights from Hugging Face
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

# One worker — the t3.micro deployment target has 1 GB RAM
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
