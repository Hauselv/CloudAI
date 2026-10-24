"""Color palette features: dominant colors, warmth, harmony and colorfulness.

k-means in CIELAB on a downscaled image gives a small palette; from it we derive the
share of warm (sunset) hues, the share of sky blue, the contrast between palette colors,
how harmonious the hues are (circular concentration), plus Hasler-Suesstrunk
colorfulness. These are closer to how people talk about sunsets than raw hue histograms.
"""

from __future__ import annotations

import cv2
import numpy as np

PALETTE_K = 5


def _kmeans_lab(image_rgb: np.ndarray, k: int = PALETTE_K, max_side: int = 160, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    scale = max_side / max(height, width)
    small = cv2.resize(image_rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else image_rgb
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    k = int(min(k, len(np.unique(lab, axis=0))))
    if k <= 1:
        return lab.mean(axis=0, keepdims=True), np.array([1.0], dtype=np.float32)
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(np.float32)
    weights = counts / counts.sum()
    order = np.argsort(-weights)
    return centers[order], weights[order]


def _lab_to_hsv(centers_lab: np.ndarray) -> np.ndarray:
    lab_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)


def colorfulness(image_rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colorfulness metric."""
    rgb = image_rgb.astype(np.float32)
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def extract_palette_features(image_rgb: np.ndarray) -> dict[str, float]:
    centers, weights = _kmeans_lab(image_rgb)
    hsv = _lab_to_hsv(centers)
    hue_deg = hsv[:, 0] * 2.0  # OpenCV hue is 0-180
    sat = hsv[:, 1] / 255.0
    val = hsv[:, 2] / 255.0
    chroma = np.sqrt(centers[:, 1].astype(np.float32) - 128.0) ** 2 if False else np.sqrt((centers[:, 1] - 128.0) ** 2 + (centers[:, 2] - 128.0) ** 2)
    colored = sat > 0.12
    # Warm sunset hues: red through orange/yellow (330-360 and 0-60), plus pink/magenta (300-330) weighted by saturation.
    warm = ((hue_deg >= 330) | (hue_deg < 60)) & colored
    pink = (hue_deg >= 290) & (hue_deg < 330) & colored
    blue = (hue_deg >= 180) & (hue_deg < 260) & colored
    weights_colored = weights * colored
    warm_share = float((weights * (warm | pink)).sum())
    blue_share = float((weights * blue).sum())
    neutral_share = float((weights * ~colored).sum())
    # Circular concentration of hues among colored palette entries (1 = one hue family, 0 = spread).
    if weights_colored.sum() > 0:
        angles = np.radians(hue_deg)
        resultant = np.hypot((weights_colored * np.cos(angles)).sum(), (weights_colored * np.sin(angles)).sum()) / weights_colored.sum()
        hue_concentration = float(resultant)
    else:
        hue_concentration = 0.0
    # Palette contrast: weighted mean pairwise Delta E (CIE76) between palette colors.
    if len(centers) > 1:
        diffs = centers[:, None, :] - centers[None, :, :]
        delta_e = np.sqrt((diffs**2).sum(axis=2))
        pair_w = weights[:, None] * weights[None, :]
        np.fill_diagonal(pair_w, 0.0)
        palette_contrast = float((delta_e * pair_w).sum() / max(pair_w.sum(), 1e-6))
    else:
        palette_contrast = 0.0
    lightness = centers[:, 0] * 100.0 / 255.0
    return {
        "palette_warm_share": warm_share,
        "palette_blue_share": blue_share,
        "palette_neutral_share": neutral_share,
        "palette_hue_concentration": hue_concentration,
        "palette_contrast": palette_contrast,
        "palette_chroma_mean": float((weights * chroma).sum()),
        "palette_chroma_max": float(chroma.max()),
        "palette_lightness_mean": float((weights * lightness).sum()),
        "palette_lightness_range": float(lightness.max() - lightness.min()),
        "palette_dominant_saturation": float(sat[0]),
        "palette_dominant_value": float(val[0]),
        "palette_dominant_share": float(weights[0]),
        "colorfulness": colorfulness(image_rgb),
    }
