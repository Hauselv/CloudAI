from __future__ import annotations

import cv2
import numpy as np

from cloud_aesthetics.preprocessing.image_ops import estimate_cloud_mask, estimate_saliency_map, rgb_to_gray


def extract_composition_features(image: np.ndarray) -> dict[str, float]:
    gray = rgb_to_gray(image)
    mask = estimate_cloud_mask(image)
    saliency = estimate_saliency_map(image)
    height, width = gray.shape
    horizontal_edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    horizon_index = int(np.argmax(horizontal_edges))
    horizon_confidence = float(horizontal_edges[horizon_index] / (horizontal_edges.mean() + 1e-6))
    vertical_profile = mask.mean(axis=1)
    brightness_profile = gray.mean(axis=1) / 255.0
    flipped = np.fliplr(gray)
    symmetry = float(np.corrcoef(gray.flatten(), flipped.flatten())[0, 1])
    negative_space_fraction = float((mask == 0).mean())
    total_saliency = saliency.sum() + 1e-6
    yy, xx = np.indices(saliency.shape)
    saliency_center_x = float((xx * saliency).sum() / total_saliency / width)
    saliency_center_y = float((yy * saliency).sum() / total_saliency / height)
    thirds_points = np.array(
        [
            [1 / 3, 1 / 3],
            [2 / 3, 1 / 3],
            [1 / 3, 2 / 3],
            [2 / 3, 2 / 3],
        ]
    )
    center = np.array([saliency_center_x, saliency_center_y])
    thirds_distance = float(np.linalg.norm(thirds_points - center, axis=1).min())
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        dominant_area_fraction = float((w * h) / (width * height))
        dominant_center_x = float((x + w / 2) / width)
        dominant_center_y = float((y + h / 2) / height)
    else:
        dominant_area_fraction = 0.0
        dominant_center_x = 0.5
        dominant_center_y = 0.5
    return {
        "horizon_row_fraction": float(horizon_index / max(height - 1, 1)),
        "horizon_confidence": horizon_confidence,
        "vertical_mass_top": float(vertical_profile[: max(1, height // 3)].mean()),
        "vertical_mass_middle": float(vertical_profile[height // 3 : 2 * height // 3].mean()),
        "vertical_mass_bottom": float(vertical_profile[2 * height // 3 :].mean()),
        "brightness_top": float(brightness_profile[: max(1, height // 3)].mean()),
        "brightness_bottom": float(brightness_profile[2 * height // 3 :].mean()),
        "symmetry": symmetry if np.isfinite(symmetry) else 0.0,
        "negative_space_fraction": negative_space_fraction,
        "saliency_center_x": saliency_center_x,
        "saliency_center_y": saliency_center_y,
        "saliency_spread": float(saliency.std()),
        "rule_of_thirds_distance": thirds_distance,
        "dominant_cloud_area_fraction": dominant_area_fraction,
        "dominant_cloud_center_x": dominant_center_x,
        "dominant_cloud_center_y": dominant_center_y,
    }
