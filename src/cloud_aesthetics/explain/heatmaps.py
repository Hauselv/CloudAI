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


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    max_value = float(values.max())
    if max_value > 0:
        values = values / max_value
    return values


def feature_diagnostic_map(image: np.ndarray, feature_name: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    feature = feature_name.lower()
    if "cloud" in feature:
        from cloud_aesthetics.preprocessing.image_ops import estimate_cloud_mask

        return estimate_cloud_mask(image).astype(np.float32)
    if "edge" in feature or "glcm" in feature or "contrast" in feature:
        edges = cv2.Canny(gray, 80, 180).astype(np.float32)
        return normalize_map(edges)
    if "brightness" in feature or "dynamic_range" in feature:
        return normalize_map(gray.astype(np.float32))
    if "saturation" in feature or "hue" in feature:
        return normalize_map(hsv[:, :, 1].astype(np.float32))
    if "saliency" in feature or "thirds" in feature or "dominant" in feature:
        from cloud_aesthetics.preprocessing.image_ops import estimate_saliency_map

        return normalize_map(estimate_saliency_map(image))
    if "dog" in feature or "blob" in feature:
        blur_small = cv2.GaussianBlur(gray, (0, 0), 1)
        blur_large = cv2.GaussianBlur(gray, (0, 0), 4)
        return normalize_map(np.abs(blur_small.astype(np.float32) - blur_large.astype(np.float32)))
    if "bottom" in feature:
        height = gray.shape[0]
        mask = np.zeros_like(gray, dtype=np.float32)
        mask[2 * height // 3 :, :] = 1.0
        return mask
    if "top" in feature:
        height = gray.shape[0]
        mask = np.zeros_like(gray, dtype=np.float32)
        mask[: height // 3, :] = 1.0
        return mask
    return fallback_heatmap_from_edges_array(image)


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


def resolve_module_path(root, module_path: str):
    module = root
    for part in module_path.split("."):
        if not part:
            continue
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def available_gradcam_layers(model) -> list[str]:
    candidates = []
    for name, module in model.named_modules():
        if name.startswith("backbone.layer") and name.count(".") == 1:
            candidates.append(name)
    if candidates:
        return candidates
    return [name for name, module in model.named_modules() if name.startswith("backbone") and hasattr(module, "register_forward_hook")][-5:]


def grad_cam_heatmap(model, image_tensor, device, target_layer: str = "backbone.layer4") -> np.ndarray:
    if torch is None:
        raise ImportError("torch is required for Grad-CAM heatmaps")
    model.eval()
    target_module = resolve_module_path(model, target_layer)
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    try:
        tensor = image_tensor.to(device).unsqueeze(0)
        tensor.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        output = model(tensor).sum()
        output.backward()
        if activations is None or gradients is None:
            return simple_gradient_heatmap(model, image_tensor, device)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0)
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_overlay(overlay: np.ndarray, path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return path


def fallback_heatmap_from_edges(image_path: str) -> np.ndarray:
    image = read_rgb_image(image_path)
    return fallback_heatmap_from_edges_array(image)


def fallback_heatmap_from_edges_array(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180).astype(np.float32)
    if edges.max() > 0:
        edges /= edges.max()
    return edges
