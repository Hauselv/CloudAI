from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cloud_aesthetics.settings import resolve_path


def read_rgb_image(path_like: str | Path) -> np.ndarray:
    path = resolve_path(path_like)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_long_edge(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_size / max(height, width)
    if scale >= 1.0:
        return image
    target_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def estimate_cloud_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((brightness > 110) & (saturation < 120)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def estimate_saliency_map(image: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(image)
    saliency = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    saliency = cv2.absdiff(gray, saliency)
    saliency = saliency.astype(np.float32)
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency
