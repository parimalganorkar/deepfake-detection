# 🗺️ ROADMAP — Renovating `deepfake-detection` into a production AI service

This is the plan we execute. For the *rules of the repo*, see [`CLAUDE.md`](CLAUDE.md).

## The story

An old learning-era project — a single 482-line Streamlit `app.py` doing YOLOv8-face +
ResNet-50 deepfake detection with Grad-CAM — is being **renovated into a modern,
production-ready, resume-impactful AI system**. The concept and the trained model stay; we
rebuild everything around them the way a real project grows: **fix the drawbacks → build a real
API → optimize → beautify the UI → containerize → deploy.**

## Decisions locked

| Decision            | Choice |
|---------------------|--------|
| AI ambition         | Engineering / MLOps focus — reuse the trained model, make everything around it production-grade. No retraining. |
| Production features  | Tests + GitHub Actions CI · structured logging + metrics/monitoring · CPU inference optimization (ONNX + quantization) · API hardening (API-key auth + rate limiting + size limits) |
| Deploy target       | AWS EC2 t3.micro (Free Tier), container-first |

## Drawbacks we are fixing (the "why")

- Monolithic `app.py` mixes UI + model loading + detection + inference + Grad-CAM — untestable, unreusable.
- Not a service — no API; can't be integrated, automated, or load-tested.
- Portability bugs — Windows backslash path, manual weight placement, implicit GPU assumptions.
- No config management; magic constants scattered; bare `except`; `st.warning` as error handling.
- No input validation / size limits; no structured logging; no observability.
- Slow CPU inference (no ONNX/quantization); video loads full frames into memory (OOM risk on 1 GB).
- Zero engineering scaffolding — no tests, CI, containerization, or deployment.

## Pre-approved dependencies

Adding any of these needs no further sign-off:
`huggingface_hub`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic-settings`,
`slowapi`, `prometheus-client` (or `prometheus-fastapi-instrumentator`), `onnx`, `onnxruntime`,
`pytest`, `httpx`, `ruff`, `altair` (Streamlit charts).

**Version constraints that matter:** `opencv-python<5.0` (5.0 dropped `cv2.CascadeClassifier`)
and `streamlit>=1.57` (the UI uses `width="stretch"`, `st.metric(border=)`, `st.badge`).

---

## Phases

### ☑ Phase 0 — Foundation & drawback fixes · `feature/foundation` ✅ DONE
- [x] Clone repo into the env (preserving local `CLAUDE.md` / `env/`).
- [x] Rewrite `CLAUDE.md`; create this `ROADMAP.md`.
- [x] Build `core/` — pure logic extracted from `app.py` (`config`, `models`, `detection`,
      `inference`, `explain`/Grad-CAM, `logging_config`). **Streamlit-free** (verified). Working
      thresholds/TTA/aggregation wrapped, not rewritten.
- [x] Fixed drawbacks: HF auto-download of both weights; cross-platform paths; `pydantic-settings`
      config; structured JSON logging replacing `st.warning`; resolved the duplicate detector fn.
- [x] Refactored `app.py` into a thin wrapper importing `core/`. Runs under Streamlit with 0
      exceptions (verified via `streamlit.testing` AppTest — 2 tabs render, model loads).
- [x] Scaffolded `tests/` (`test_core.py`, `conftest.py`) — 7 fast unit tests + 1 opt-in
      integration test. **All 8 pass** (incl. end-to-end model run).
- [x] Split requirements (`requirements.txt`, `requirements-api.txt`, `requirements-dev.txt`) +
      `pyproject.toml` (pytest/ruff).
- **Extra fix found in verification:** opencv 5.0 removed `cv2.CascadeClassifier` / Haar XMLs
      (broke the fallback detector) → pinned `opencv-python<5.0` in both requirements files.
- **Done:** `core/` imports no Streamlit, the app works, all tests pass. ✅

### ☑ Phase 1 — FastAPI backend · `feature/fastapi-backend` ✅ DONE
- [x] `api/main.py` with lifespan loading both models once onto `app.state` (verified: startup log
      "Models loaded — API ready.").
- [x] Endpoints: `GET /` → `/docs`, `GET /health`, `GET /ready`, `POST /predict`,
      `POST /predict_video`, `GET /metrics`. All verified live via `uvicorn` + real image.
- [x] `api/schemas.py` typed responses; uniform `{"error": "..."}` errors (400/413/429/500, no
      stack-trace leaks); streamed size limits (10 MB image / 50 MB video).
- [x] Hardening: `api/security.py` — opt-in API-key auth (`X-API-Key`) + slowapi rate limiting.
- [x] Observability: structured JSON logging + `api/metrics.py` Prometheus metrics at `/metrics`.
- [x] `tests/test_api.py` (11 TestClient tests, models mocked so CI needs no weights);
      `.github/workflows/ci.yml` runs ruff + pytest on push/PR.
- **Verified live:** `/predict` on a real face → `{"label","confidence","p_fake","p_real",
      "faces_detected":1,"processing_ms":254}`; no-face → `400 {"error":"No face detected..."}`;
      `/health` `/ready` `/metrics` `/docs` all 200. **ruff clean · 17 passed / 1 skipped.** ✅

### ☑ Phase 2 — CPU inference optimization · `feature/onnx-optimization` ✅ DONE
- [x] `scripts/export_onnx.py` — exports ResNet-50 → ONNX, applies dynamic int8 quantization,
      benchmarks torch vs ONNX(fp32) vs ONNX(int8).
- [x] `core/onnx_runtime.py` — config-toggled ONNX Runtime path (`DEEPFAKE_USE_ONNX=1`) with
      automatic torch fallback; wired into `predict_pil_images` (so both `/predict` and the
      Streamlit app benefit). onnxruntime imported lazily.
- [x] Benchmark recorded in `docs/onnx_benchmark.md`.
- **Result:** ONNX fp32 = **3.6× faster** than PyTorch (67.2 → 18.7 ms/img). Dynamic int8 is
      ~4× smaller (89.6 → 22.6 MB) but slower for this conv-heavy net → default is fp32
      (`onnx_use_quantized=False`), a measured, data-driven choice.
- **Verified:** ONNX vs torch parity `abs diff = 1.2e-07`; parity integration test passes.
      **ruff clean · 17 passed / 2 skipped (fast) · 9 passed (integration).** ✅

### ☑ Phase 3 — Streamlit refactor + professional dashboard UI · `feature/streamlit-redesign` ✅ DONE
- [x] `app.py` is a thin wrapper over `core/` (presentation only).
- [x] Premium **light professional dashboard**: `.streamlit/config.toml` (light indigo palette +
      Inter/JetBrains fonts) + `.streamlit/style.css` (injected) — hidden Streamlit chrome,
      gradient hero, KPI stat row, clean white cards with soft shadows, verdict chips, gradient
      confidence bar, key-targeted gradient buttons, styled tabs/uploader/metrics/expanders.
- [x] Modernized per Streamlit best practices: Material Symbols icons, `st.badge`, native
      `st.container(border=True)` result cards, `width="stretch"` (dropped deprecated
      `use_container_width`), **Altair** frame chart (replaced matplotlib `st.pyplot`), and UX
      feedback (`st.spinner`, `st.toast`, success/error states).
- [x] Inference logic + thresholds unchanged.
- **Verified:** ruff clean · AppTest 0 exceptions (2 tabs, 2 buttons, 4 KPI metrics) · Altair
      chart builds · app boots healthy at :8501. ✅

### ☑ Pre-Docker sanity check ✅ DONE
- [x] Structure matches target; no stray files; `core/` framework-free; `api.main` does not
      import `core.explain`.
- [x] ruff clean · **17 passed / 2 skipped** (CI mode) · **19 passed** (integration, real weights
      + ONNX parity) · Streamlit renders 0 exceptions · all API endpoints 200.
- [x] No weights/keys tracked by git; `onnx_models/` (113 MB) ignored.
- **Fixes applied:** bumped `streamlit>=1.57` (the UI would crash on 1.30), declared `altair`
      and `onnxruntime` explicitly, annotated `scikit-learn`/`tqdm` as training-only, added
      `.pytest_cache/`/`.ruff_cache/` to `.gitignore`.
- **Noted:** `matplotlib` ships in the API image regardless (hard dep of `ultralytics`).

### ☑ Phase 4 — Docker · `feature/docker` ✅ DONE
- [x] `Dockerfile` — `python:3.11-slim`, **CPU-only torch** (`2.8.0+cpu`, no CUDA packages),
      `libgl1 libglib2.0-0`, headless-only OpenCV, non-root `appuser`, `HEALTHCHECK`,
      `EXPOSE 8000`, `uvicorn ... --workers 1`.
- [x] Weights **not baked in** — downloaded from HF at startup into `HF_HOME=/cache/huggingface`
      (mounted volume, so restarts are instant).
- [x] `.dockerignore` excludes weights, `onnx_models/`, notebook, training report, tests, docs,
      and the Streamlit UI. `docker-compose.yml` validated (`docker compose config`).
- **Bug caught by containerizing:** `omegaconf` was missing from both requirements files. The
      `yolov8n-face.pt` checkpoint embeds an omegaconf object, so `YOLO()` crashed at startup in a
      clean environment — it only worked locally because omegaconf was installed incidentally.
      Now declared explicitly in `requirements.txt` and `requirements-api.txt`.
- **Verified:** build succeeds · container starts in ~12 s · `HEALTHCHECK` reports `healthy` ·
      `/health` `/ready` `/docs` `/metrics` `/predict` all 200 · prediction **bit-identical to the
      host run** (`p_fake=0.7738938331604004`) · **432 MiB RAM** in use (fits the 1 GB t3.micro) ·
      runs as non-root with exactly one OpenCV package.
- **Image size: 2.66 GB** — above the original 2 GB goal. Not GPU wheels (torch is `+cpu`); the
      bulk is `torch` (722 MB), `polars` via ultralytics (216 MB), `onnx` (86 MB), OpenCV (158 MB).
      Accepted as-is by decision; trimming `onnx` (export-only) and `polars` would be the first
      levers if it ever needs to shrink.

### ☐ Phase 5 — CI/CD + AWS EC2 deployment · `feature/deployment`
- [ ] `DEPLOY.md` runbook: launch t3.micro (Ubuntu 24.04), **2 GB swap first**, install Docker,
      transfer weights, `docker run -p 80:8000 ... --restart unless-stopped`, verify.
- [ ] Security group: SSH(22) my-IP-only, HTTP(80) open. **₹100 billing alarm reminder.**
- [ ] Extend CI toward CD (build image on tag) where sensible.
- **Done when:** API reachable at `http://<ec2-ip>/`; `/docs` screenshot committed to `docs/`.

### ☐ Phase 6 — Docs & resume polish · `feature/docs`
- [ ] README: architecture diagram, feature matrix, API usage (`curl`), local dev (Streamlit +
      API + Docker), Live Demo section + screenshot.
- [ ] Land the resume bullets (FastAPI service, Docker, EC2, ONNX optimization, CI/tests, monitoring).

---

## Resume bullets this unlocks

- Re-architected a monolithic ML demo into a layered, tested Python service (pure `core/` +
  FastAPI + Streamlit), with structured logging and Prometheus metrics.
- Built and hardened a FastAPI deepfake-inference API (API-key auth, rate limiting, request
  validation, interactive OpenAPI docs) loading models once via lifespan management.
- Optimized CPU inference with ONNX Runtime + dynamic quantization (record measured speedup).
- Containerized the service (CPU-only, <2 GB) and deployed it on AWS EC2 with health checks and CI.
