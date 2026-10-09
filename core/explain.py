"""Grad-CAM explainability.

Kept separate from ``inference.py`` because it depends on matplotlib (for the colormap) and is
only used by the Streamlit UI, so the API never pulls plotting code into its import path. Note
this does not keep matplotlib out of the API image — ``ultralytics`` hard-requires it, so it
ships regardless; the separation is about clean layering, not image size.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


class GradCAM:
    """Grad-CAM heatmap generator for a ResNet-style model (default target layer ``layer4``)."""

    def __init__(self, model, target_layer_name: str = "layer4"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        layer = dict(model.named_modules())[target_layer_name]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx: int | None = None) -> np.ndarray:
        x = x.requires_grad_(True)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        score = out[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam


def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a Grad-CAM heatmap onto an image using the 'jet' colormap."""
    img = np.array(img_pil).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")(cam)[..., :3]
    out = np.clip((1 - alpha) * img + alpha * cmap, 0, 1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))
