# app.py — AI Deepfake Detector (Streamlit UI)
# Premium light dashboard UI. All inference logic lives in the framework-free `core/` package
# (shared with the FastAPI service). This file is presentation only: layout, styling, and UX.

import io
import os
import tempfile
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.config import settings
from core.detection import detect_largest_face
from core.explain import GradCAM, overlay_cam
from core.inference import aggregate_video, predict_pil_images, sample_video_faces
from core.models import get_transforms, load_classifier
from core.onnx_runtime import onnx_available

# ------------------------------------------------------------------ config
IMG_SIZE = settings.img_size
ALLOWED_IMG = settings.allowed_img_ext
ALLOWED_VID = settings.allowed_vid_ext
FACE_DETECTION_CONFIDENCE = settings.face_detection_confidence
CLASSIFICATION_THRESHOLD = settings.classification_threshold
HIGH_CONF_THRESHOLD = settings.high_conf_threshold
VIDEO_AVG_THRESHOLD = settings.video_avg_threshold
HIGHLY_FAKE_RATE_THRESHOLD = settings.highly_fake_rate_threshold

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon=":material/shield_person:",
    layout="wide",
)


@st.cache_data
def load_css() -> str:
    css_path = Path(__file__).parent / ".streamlit" / "style.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def verdict_chip(is_fake: bool) -> str:
    if is_fake:
        return (
            '<span class="verdict verdict-fake">'
            '<span class="material-symbols-rounded">gpp_bad</span> AI-generated · fake</span>'
        )
    return (
        '<span class="verdict verdict-real">'
        '<span class="material-symbols-rounded">verified_user</span> Likely real</span>'
    )


def confidence_bar(confidence: float, is_fake: bool) -> str:
    fill = "fake-fill" if is_fake else "real-fill"
    pct = confidence * 100
    return (
        '<p class="conf-label">Model confidence</p>'
        f'<div class="confbar"><div class="{fill}" style="width:{pct:.1f}%"></div></div>'
        f'<p class="conf-label" style="text-align:right;margin-top:.4rem;color:#0F172A;'
        f'font-weight:700;">{pct:.1f}%</p>'
    )


# ------------------------------------------------------------------ resources
try:
    with st.spinner("Loading models — downloading weights from Hugging Face on first run…"):
        model, class_to_idx, device = load_classifier()
        TFM = get_transforms()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model: {e}", icon=":material/error:")
    st.stop()

i_fake = class_to_idx.get("fake", 0)
BACKEND = "ONNX Runtime" if onnx_available() else "PyTorch"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(
        '<div style="font-weight:800;font-size:1.2rem;letter-spacing:-.02em;">'
        '<span class="material-symbols-rounded" style="color:#4F46E5;vertical-align:-4px;">'
        'shield_person</span> Deepfake Detector</div>',
        unsafe_allow_html=True,
    )
    st.caption("YOLOv8-face · fine-tuned ResNet-50 · Grad-CAM")
    st.markdown(f":violet-badge[{BACKEND}] :blue-badge[CPU]  :green-badge[~86% val acc]")

    with st.expander("About the model", icon=":material/neurology:", expanded=True):
        st.markdown(
            "A **ResNet-50** (ImageNet-pretrained) fine-tuned on deepfake data "
            "(**WildDeepfake** images + **FaceForensics++** videos). It analyzes the cropped "
            "face for the subtle artifacts left behind by manipulation."
        )
    with st.expander("How to read the result", icon=":material/help:"):
        st.markdown(
            "- **Verdict chip** — the model's call: real vs AI-generated.\n"
            "- **Confidence** — probability of the predicted label.\n"
            "- **Heatmap** — where the model looked (Grad-CAM). Hot = suspicious."
        )

# ------------------------------------------------------------------ hero
st.markdown(
    """
    <div class="hero">
      <span class="eyebrow"><span class="dot"></span> Real-time deepfake analysis</span>
      <h1>Is it real, or AI?</h1>
      <p class="subtitle">Upload an image or a short video. The detector localizes the face,
      classifies it as authentic or manipulated, and shows exactly where the model looked.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
with st.container(horizontal=True, key="kpi_row"):
    st.metric("Classifier", "ResNet-50", border=True)
    st.metric("Validation accuracy", "≈ 86%", border=True)
    st.metric("Face detector", "YOLOv8-face", border=True)
    st.metric("Inference backend", BACKEND, border=True)

st.write("")
steps = [
    ("1", "Upload media", "Choose an image or short video that contains a face.", "indigo"),
    ("2", "Detect face", "YOLOv8-face crops the largest face — Haar cascade as a fallback.", "blue"),
    ("3", "Classify", "The fine-tuned ResNet-50 scores the crop as real or fake, with Grad-CAM.", "violet"),
]
cols = st.columns(3, gap="medium")
for col, (num, title, desc, tint) in zip(cols, steps, strict=True):
    col.markdown(
        f'<div class="card tint-{tint}"><div class="step-num num-{tint}">{num}</div>'
        f"<h4>{title}</h4><p>{desc}</p></div>",
        unsafe_allow_html=True,
    )

st.write("")
tab_img, tab_vid = st.tabs(["Image analysis", "Video analysis"])

# ================================================================== IMAGE TAB
with tab_img:
    files = st.file_uploader(
        "Upload image(s)",
        type=list(ALLOWED_IMG),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    go = st.button(
        "Analyze image(s)",
        type="primary",
        disabled=not files,
        width="stretch",
        key="btn_analyze_images",
    )

    if go and files:
        processed_imgs, face_crops = [], []
        with st.spinner("Detecting faces and classifying…"):
            for f in files:
                try:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    processed_imgs.append(img)
                    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    face_bgr = detect_largest_face(bgr, conf_threshold=FACE_DETECTION_CONFIDENCE)
                    face_crops.append(
                        Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                        if face_bgr is not None
                        else img
                    )
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not process '{f.name}': {e}", icon=":material/warning:")

            results = predict_pil_images(face_crops, tta=settings.tta) if face_crops else []

        st.toast(f"Analyzed {len(results)} image(s)", icon=":material/check_circle:")
        st.subheader(f"Results · {len(results)} image(s)")

        for img, face_crop, res in zip(processed_imgs, face_crops, results, strict=False):
            p_fake = res["p_fake"]
            is_fake = p_fake >= CLASSIFICATION_THRESHOLD
            confidence = p_fake if is_fake else 1 - p_fake

            with st.container(border=True):
                st.markdown(verdict_chip(is_fake), unsafe_allow_html=True)
                st.markdown(confidence_bar(confidence, is_fake), unsafe_allow_html=True)

                with st.spinner("Generating focus map…"):
                    resized = face_crop.resize((IMG_SIZE, IMG_SIZE))
                    x = TFM(resized).unsqueeze(0).to(device)
                    heatmap = overlay_cam(resized, GradCAM(model)(x, class_idx=i_fake))

                c_img, c_heat = st.columns(2)
                c_img.image(img, caption="Original", width="stretch")
                c_heat.image(heatmap, caption="Grad-CAM focus (on face crop)", width="stretch")

                with st.expander("Explain the model's focus", icon=":material/insights:"):
                    st.markdown(
                        "The heatmap shows where the model looked on the cropped face. "
                        "**Hot (red/yellow)** regions are the ones it found most suspicious; "
                        "**cool (blue)** regions were largely ignored. The verdict is driven by "
                        "the features in the hot areas matching patterns learned in training."
                    )

# ================================================================== VIDEO TAB
with tab_vid:
    vf = st.file_uploader(
        "Upload a short video",
        type=list(ALLOWED_VID),
        label_visibility="collapsed",
    )
    go_v = st.button(
        "Analyze video",
        type="primary",
        disabled=(vf is None),
        width="stretch",
        key="btn_analyze_video",
    )

    if go_v and vf is not None:
        progress = st.progress(0, text="Analyzing video…")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vf.name).suffix) as tmp:
            tmp.write(vf.read())
            tmp_path = tmp.name

        try:
            def _on_progress(frames_read, faces_found, total):
                progress.progress(
                    min(1.0, frames_read / max(1, total)),
                    text=f"Scanning video… valid faces found: {faces_found}",
                )

            try:
                face_crops, frame_indices, fps = sample_video_faces(
                    tmp_path, progress_callback=_on_progress
                )
            except ValueError:
                st.error("Could not open the video file.", icon=":material/error:")
                face_crops, frame_indices, fps = [], [], 0

            progress.empty()

            if not face_crops:
                st.warning(
                    "No high-quality faces detected — unable to make a prediction.",
                    icon=":material/no_accounts:",
                )
            else:
                with st.spinner("Classifying sampled frames…"):
                    preds = predict_pil_images(face_crops, tta=settings.tta)
                    p_fakes = np.array([r["p_fake"] for r in preds])
                    agg = aggregate_video(p_fakes)

                is_fake = agg["is_fake"]
                st.toast("Video analysis complete", icon=":material/check_circle:")

                with st.container(border=True):
                    st.markdown("##### Overall verdict")
                    st.markdown(verdict_chip(is_fake), unsafe_allow_html=True)
                    if is_fake:
                        st.error(
                            f"Flagged **fake**: average fake score {agg['mean_p_fake']:.2f} ≥ "
                            f"{VIDEO_AVG_THRESHOLD}, and {agg['highly_fake_rate']:.0%} of frames "
                            f"were highly-confident fakes (≥ {HIGHLY_FAKE_RATE_THRESHOLD:.0%}).",
                            icon=":material/gpp_bad:",
                        )
                    else:
                        st.success(
                            f"Assessed **real**: it did not meet both fake criteria "
                            f"(avg {agg['mean_p_fake']:.2f}, highly-confident rate "
                            f"{agg['highly_fake_rate']:.0%}).",
                            icon=":material/verified_user:",
                        )

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg fake score", f"{agg['mean_p_fake']:.3f}")
                    m2.metric("Max fake score", f"{agg['max_p_fake']:.3f}")
                    m3.metric(
                        "Confident fakes",
                        f"{agg['highly_fake_count']} / {agg['num_frames']}",
                        f"{agg['highly_fake_rate']:.0%}",
                    )

                st.subheader("Fake probability over time")
                df = pd.DataFrame({"Frame": frame_indices, "p(fake)": p_fakes})
                line = (
                    alt.Chart(df)
                    .mark_line(color="#818CF8", point=alt.OverlayMarkDef(color="#6366F1", size=45))
                    .encode(
                        x=alt.X("Frame:Q", title="Frame"),
                        y=alt.Y("p(fake):Q", scale=alt.Scale(domain=[0, 1]), title="p(fake)"),
                        tooltip=["Frame", "p(fake)"],
                    )
                )
                threshold = (
                    alt.Chart(pd.DataFrame({"y": [HIGH_CONF_THRESHOLD]}))
                    .mark_rule(color="#F43F5E", strokeDash=[6, 4])
                    .encode(y="y:Q")
                )
                st.altair_chart((line + threshold).properties(height=260), width="stretch")

                st.subheader("Most suspicious frames")
                order = np.argsort(-p_fakes)[: min(6, len(p_fakes))]
                if p_fakes[order[0]] < CLASSIFICATION_THRESHOLD:
                    st.success(
                        "No frames were flagged as suspicious — the video looks consistently real.",
                        icon=":material/thumb_up:",
                    )
                else:
                    gallery = st.columns(3)
                    shown = 0
                    for k in order:
                        if p_fakes[k] < CLASSIFICATION_THRESHOLD:
                            continue
                        ts = f"{frame_indices[k] / fps:.2f}s" if fps > 0 else f"frame {frame_indices[k]}"
                        gallery[shown % 3].image(
                            face_crops[k], caption=f"{ts} · p(fake) {p_fakes[k]:.3f}", width="stretch"
                        )
                        shown += 1
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
