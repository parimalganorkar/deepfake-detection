# 🛡️ AI-Powered Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-yellow?logo=huggingface&logoColor=white)

**Project:** Deepfake Detection using ResNet-50 + YOLOv8-face  
**GitHub Repo:** [parimalganorkar/deepfake-detection](https://github.com/parimalganorkar/deepfake-detection.git)  
**Model Weights:** [Hugging Face Hub](https://huggingface.co/Parimal/Resnet50_best/tree/main)

---

## 📌 1. Project Description
A production-ready pipeline for detecting deepfakes in images and videos. The system utilizes a hybrid approach:
1.  **Face Detection:** Uses **YOLOv8-face** (with a Haar Cascade fallback) to accurately locate faces in frames.
2.  **Classification:** Extracts face crops and processes them through a fine-tuned **ResNet-50** backbone.
3.  **Explainability:** Includes Grad-CAM visualization to highlight which parts of the face triggered the decision.

The training process involved robust augmentations, mixed-precision (AMP) training, Exponential Moving Average (EMA), Focal Loss for class imbalance, and a Cosine Learning Rate scheduler.

## 🚀 2. Key Features
*   **Dual Mode Inference:** Seamlessly handles both single Images and Video files.
*   **Robust Backbone:** Fine-tuned **ResNet-50** optimized with AdamW and Cosine Annealing.
*   **Smart Face Detection:** Primary detection via YOLOv8-face; automatic fallback to Haar Cascade if YOLO fails.
*   **Video Logic:** Aggregates frame-by-frame predictions using average scoring + "highly fake" frame heuristics for final video classification.
*   **Explainability:** Integrated **Grad-CAM** heatmaps to visualize the model's focus areas.
*   **Interactive UI:** Built with **Streamlit** for easy uploading and real-time analysis.
*   **TTA (Test-Time Augmentation):** Improves inference reliability.

## 🖥️ 3. System & Hardware
*   **Development:** Developed & validated on **Kaggle T4 GPU**.
*   **Recommendation:** A CUDA-enabled NVIDIA GPU is strongly recommended for video processing speed.
*   **Minimum:** Can run on CPU for single-image inference, but video analysis will be slow.

---

## ⚙️ 4. Installation & Execution Guide

Follow these steps to set up the project locally.

### Step 1: Clone the Repository
```bash
git clone https://github.com/parimalganorkar/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
### Step 2: Download Model Weights (Crucial)
The trained model weights are hosted on Hugging Face to keep the Git repository light.
Go to the Hugging Face Repository.
Download the file named resnet50_best.pt (or the specific .pt checkpoint you wish to use).
Move/Paste this file into the root folder of the project (inside deepfake-detection/).