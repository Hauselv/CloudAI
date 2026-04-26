from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.settings import ensure_parent, resolve_path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * image + alpha * colored)


def simple_gradient_heatmap(model, image_tensor, device) -> np.ndarray:
    if torch is None:
        raise ImportError("torch is required for gradient-based heatmaps")
    image_tensor = image_tensor.to(device).unsqueeze(0)
    image_tensor.requires_grad_(True)
    model.zero_grad()
    output = model(image_tensor).sum()
    output.backward()
    grads = image_tensor.grad.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
    grads -= grads.min()
    if grads.max() > 0:
        grads /= grads.max()
    return grads


def save_overlay(overlay: np.ndarray, path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return path


def fallback_heatmap_from_edges(image_path: str) -> np.ndarray:
    image = read_rgb_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180).astype(np.float32)
    if edges.max() > 0:
        edges /= edges.max()
    return edges
