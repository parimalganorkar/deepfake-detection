# CLAUDE.md

This file tells Claude Code how to work in this repo. Read it fully at the start of every
session before making changes. For *what* we're building and the phase order, see
[`ROADMAP.md`](ROADMAP.md) — this file is the *rules*, the roadmap is the *plan*.

## What this repo is

`deepfake-detection` — a real-time deepfake image/video classifier combining **YOLOv8-face**
for face detection with a **fine-tuned ResNet-50** for real/fake classification (~86% val
accuracy, trained with AMP + EMA + Focal Loss + cosine LR). It began as a single-file
Streamlit demo (`app.py`) with Grad-CAM overlays.

**We are renovating it into a production-grade AI service** — keeping the concept and the
trained model, and rebuilding everything around it: a clean `core/` library, a hardened
FastAPI service, ONNX-optimized CPU inference, a beautified Streamlit UI, Docker, tests + CI,
and an AWS EC2 deployment. The bar is: *modern, production-ready, resume-impactful.*

## Vision & principles

- **Renovation, not rewrite of the model.** The trained ResNet-50 and the YOLOv8-face detector
  stay. We reuse the working inference logic — thresholds, TTA, video aggregation. **Wrap it,
  don't re-derive it.**
- **Clean layering.** `core/` is pure and framework-free (no `streamlit`, no `fastapi`). Both
  `app.py` (Streamlit) and `api/` (FastAPI) import from `core/`. Nothing in `core/` touches
  `st.*` or FastAPI.
- **Production-minded.** Config over constants, logging over `print`/`st.warning`, typed
  interfaces, validated inputs, observable behavior, tested code.
- **Portable.** No Windows-only paths, no implicit GPU assumptions. Runs CPU-only in Docker.

## Model weights (never committed)

Weights are hosted on the Hugging Face Hub and **auto-downloaded at runtime** (never committed,
never baked into images). `.gitignore` excludes `*.pt`/`*.pth`.

- HF repo: **`Parimal/Resnet50_best`**
  - `resnet50_best.pt` (~283 MB) — the classifier checkpoint (a dict with keys `model` and
    `class_to_idx`; load with `torch.load(..., weights_only=False)` — the flag is **required**
    because the checkpoint holds non-tensor data, and PyTorch 2.6+ flipped the default).
  - `yolov8n-face.pt` (~6 MB) — the face detector.
- Download via `huggingface_hub.hf_hub_download(repo_id="Parimal/Resnet50_best", filename=...)`.
  It caches locally, so first run downloads and later runs are instant.

## Target structure

```
deepfake-detection/
├── core/                     # Pure inference logic — no streamlit, no fastapi
│   ├── __init__.py
│   ├── config.py             # pydantic-settings: HF repo, thresholds, size limits, toggles
│   ├── models.py             # HF download, load_classifier(), yolo/haar detectors, transforms
│   ├── detection.py          # _crop_face(), detect_largest_face()  — uses logging
│   ├── inference.py          # predict_pil_images(), analyze_image/_video, sampling + aggregation
│   ├── explain.py            # GradCAM + overlay_cam — matplotlib lives HERE, not in inference
│   ├── onnx_runtime.py       # optional ONNX/quantized inference path (torch fallback)
│   └── logging_config.py     # structured JSON logging setup
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app + lifespan (loads models once onto app.state)
│   ├── schemas.py            # pydantic request/response models
│   ├── security.py           # API-key auth dependency + rate limiting
│   └── metrics.py            # Prometheus metrics + /metrics
├── app.py                    # Streamlit — thin wrapper over core/, light pro dashboard
├── tests/                    # pytest: test_core.py, test_api.py, conftest.py + fixtures
├── scripts/export_onnx.py    # ONNX export + dynamic quantization + benchmark
├── .github/workflows/ci.yml  # lint + tests on push/PR
├── .streamlit/config.toml    # light dashboard theme (palette, fonts)
├── .streamlit/style.css      # injected CSS (cards, KPI accents, verdict chips)
├── pyproject.toml            # pytest + ruff config
├── docs/                     # onnx_benchmark.md, screenshots
├── Dockerfile · docker-compose.yml · .dockerignore
├── requirements.txt          # Streamlit stack
├── requirements-api.txt      # FastAPI + CPU-only torch + onnxruntime + hardening
├── requirements-dev.txt      # pytest, httpx, ruff
├── ROADMAP.md · DEPLOY.md · README.md
└── resnet-50.ipynb · resnet50_report/   (unchanged — training artifacts)
```

**Why `explain.py` is separate:** it imports matplotlib and is used only by Streamlit. The API
never imports it, keeping that dependency out of the API's import path.

## API contract (Phase 1)

- **`GET /`** → `RedirectResponse("/docs")`.
- **`GET /health`** → `{"status": "ok"}` (liveness, cheap).
- **`GET /ready`** → `{"status": "ready", "model_loaded": true}` (readiness — models on `app.state`).
- **`GET /metrics`** → Prometheus exposition (request count, latency, errors).
- **`POST /predict`** — multipart image upload (field `file`; png/jpg/jpeg). Runs
  `detect_largest_face` → crop → `predict_pil_images(tta=2)`. Returns:
  ```json
  {"label":"fake","confidence":0.912,"p_fake":0.912,"p_real":0.088,
   "faces_detected":1,"processing_ms":480}
  ```
  `label` is `"fake"` if `p_fake >= 0.5` (matching `CLASSIFICATION_THRESHOLD`), else `"real"`;
  `confidence` is the probability of the predicted label.
- **`POST /predict_video`** — mp4/mov/avi/mkv upload; samples frames, aggregates via the existing
  rules (`VIDEO_AVG_THRESHOLD`, `HIGH_CONF_THRESHOLD`, `HIGHLY_FAKE_RATE_THRESHOLD`). Returns
  per-frame scores + aggregate verdict. Build **after** `/predict` is solid — video is where
  memory pressure on the t3.micro shows up.
- **Errors:** 400 `{"error":"..."}` for no file / bad extension / corrupt media / no face;
  413 for over size limit (10 MB image, 50 MB video — in `core/config.py`); 500 generic, **no
  stack traces leaked**.
- **Hardening:** API-key header auth + rate limiting on the predict endpoints.

## Docker rules

- Base `python:3.11-slim`. Install **CPU-only torch**:
  `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu`.
  GPU wheels are the difference between a ~1.5 GB and a ~6 GB image — never let them in.
- OpenCV in slim needs `libgl1 libglib2.0-0` (`apt-get install --no-install-recommends`, then
  clean `/var/lib/apt/lists/*`) or `cv2` throws cryptic import errors on the first request.
- Use `opencv-python-headless` in the container (no GUI deps).
- Final image target: **under 2 GB.** If larger, you have GPU wheels — fix them.
- `EXPOSE 8000`; run `uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1` (1 worker —
  the t3.micro has 1 GB RAM).
- **Weights are not baked in.** Mount at runtime (`-v`), or let the app pull from HF at start.
- `.dockerignore`: `.git`, `__pycache__`, `*.pyc`, `.venv`, `venv`, `env`, `.env`, `data/`,
  `checkpoints/`, `*.pt`, `*.pth`, `resnet-50.ipynb`, `resnet50_report/`, `.streamlit/`, `tests/`.

## AWS deployment notes

- Target: **EC2 t3.micro, Ubuntu 24.04 LTS, Free Tier.** 1 vCPU / 1 GB RAM — memory is the
  binding constraint.
- **2 GB swap is mandatory** before pulling/running the image (the t3.micro OOMs on a large
  `docker pull` without swap):
  ```
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```
- Security group: SSH (22) from **my IP only**, HTTP (80) from `0.0.0.0/0`. Never open SSH to the world.
- Container port 8000 → host 80 via `-p 80:8000`; `--restart unless-stopped`.
- Weights: `scp resnet50_best.pt ubuntu@<ip>:~/weights/` and mount, or rely on HF auto-download.
- **Remind me to set a ₹100 billing alarm before launching the instance.** Stop (don't
  terminate) when not demoing. The AWS `.pem` key lives **outside this repo** — mentioning its
  path in a committed file is a bug.

## Non-negotiable constraints

- **Never commit** `*.pt`/`*.pth`, weights directories, `.env`, AWS keys, or `.pem` paths.
  `.gitignore` handles the weights — don't weaken it.
- **`core/` stays framework-free** — no `streamlit`, no `fastapi` imports.
- **Reuse working logic** — wrap the thresholds/TTA/video aggregation, don't rewrite them.
- **Keep the Streamlit app working** — `streamlit run app.py` must run end-to-end at every phase
  (its *look* changes in Phase 3; its *results* must not).
- **CPU-only inference in Docker.** No CUDA wheels in the image.

## Git workflow

- One feature branch per phase (see ROADMAP): `feature/foundation`, `feature/fastapi-backend`,
  `feature/onnx-optimization`, `feature/streamlit-redesign`, `feature/docker`,
  `feature/deployment`, `feature/docs`. Do them in order — each phase unblocks the next.
- **Never commit directly to `main`.** Merge via clean commits after I've reviewed each phase.
- **Do not `git commit`, `git push`, `git checkout` (remote-affecting), or delete branches
  without me saying so in chat.** Before any commit: run `git status` and `git diff --staged`,
  show me the output, and wait. Never `git push --force`; never rewrite pushed history.
- Commit messages: single sentence, imperative (e.g. `Extract inference logic into core module`).

## Ask me before

- Any `git` commit/push/branch-delete/change to `main`.
- Deleting existing files.
- Adding a dependency **not already listed in ROADMAP.md's dependency list** (the ONNX/API/test
  deps there are pre-approved).
- Rewriting logic that already works in `app.py` — the rule is *wrap it*, not *rewrite it*.

## Just do

- Create new files under `core/`, `api/`, `tests/`, `scripts/`.
- Write `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `requirements-*.txt`, `DEPLOY.md`.
- Run the service locally and hit it with `curl` to verify. Run `pytest`. Run `docker build`/`run`.
- Suggest README improvements in draft form for me to review before committing.
- Fix your own bugs when tests fail.

## Known landmines

**Still live — do not regress these:**

- **`torch.load(..., weights_only=False)`** must stay — the checkpoint carries `class_to_idx`.
- **Pin `opencv-python<5.0`.** OpenCV 5.0 **removed `cv2.CascadeClassifier` and ships no Haar
  XMLs**, silently breaking the Haar fallback detector. Both requirements files pin `<5.0`.
- **`streamlit>=1.57`** is required — `app.py` uses `width="stretch"`, `st.metric(border=...)`,
  `st.container(horizontal=...)`, and `st.badge`, none of which exist in older releases.
- **matplotlib lands in the API image anyway** — `ultralytics` hard-requires `matplotlib>=3.3.0`.
  Keeping `core/explain.py` out of the API's import path is still correct layering, but don't
  expect it to remove matplotlib from the image.
- **`load_css()` is `@st.cache_data`** — after editing `.streamlit/style.css` you must restart
  the Streamlit server; a plain rerun serves the cached CSS.
- **OpenCV in slim** needs `libgl1` + `libglib2.0-0` or import fails on first request.
- **t3.micro OOMs** on a large `docker pull` unless swap is enabled first.
- **`env/` is the local venv** — full deps (torch etc.) are a large download.

**Fixed during the renovation (kept for context):**

- ~~Windows path `Path(r"resnet50_mixed_export\resnet50_best.pt")`~~ → replaced by HF
  auto-download in `core/models.py`.
- ~~`get_yolo_face_detector` defined twice in `app.py`~~ → one canonical copy in `core/models.py`.
- ~~`st.warning`/`st.error` used as error handling in pure logic~~ → structured logging in `core/`.
