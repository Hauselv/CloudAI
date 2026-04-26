from __future__ import annotations

import cv2
import numpy as np

from cloud_aesthetics.preprocessing.image_ops import estimate_cloud_mask


def extract_color_features(image: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    brightness = image.mean(axis=2)
    hue_hist, _ = np.histogram(hsv[:, :, 0], bins=8, range=(0, 180), density=True)
    mask = estimate_cloud_mask(image)
    cloud_fraction = float(mask.mean())
    sky_fraction = float(1.0 - cloud_fraction)
    blue_channel = image[:, :, 2].astype(np.float32)
    red_channel = image[:, :, 0].astype(np.float32)
    return {
        "mean_brightness": float(brightness.mean()),
        "contrast": float(brightness.std()),
        "dynamic_range": float(brightness.max() - brightness.min()),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "lab_a_mean": float(lab[:, :, 1].mean()),
        "lab_b_mean": float(lab[:, :, 2].mean()),
        "blue_orange_balance": float((blue_channel.mean() + 1.0) / (red_channel.mean() + 1.0)),
        "color_temperature_proxy": float(lab[:, :, 2].mean() - lab[:, :, 1].mean()),
        "cloud_area_fraction": cloud_fraction,
        "sky_area_fraction": sky_fraction,
        **{f"hue_bin_{index}": float(value) for index, value in enumerate(hue_hist)},
    }
