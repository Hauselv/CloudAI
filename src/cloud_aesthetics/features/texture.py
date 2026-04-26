from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import difference_of_gaussians
from skimage.measure import shannon_entropy

from cloud_aesthetics.preprocessing.image_ops import estimate_cloud_mask, rgb_to_gray


def _fractal_dimension(mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return 0.0
    sizes = 2 ** np.arange(1, 6)
    counts = []
    for size in sizes:
        reduced = mask[: mask.shape[0] - (mask.shape[0] % size), : mask.shape[1] - (mask.shape[1] % size)]
        if reduced.size == 0:
            continue
        blocks = reduced.reshape(reduced.shape[0] // size, size, reduced.shape[1] // size, size)
        counts.append(np.sum(blocks.any(axis=(1, 3))))
    counts = np.array(counts, dtype=np.float32)
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0
    coeffs = np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)
    return float(-coeffs[0])


def extract_texture_features(image: np.ndarray) -> dict[str, float]:
    gray = rgb_to_gray(image)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    orientations = (np.rad2deg(np.arctan2(grad_y, grad_x)) + 180.0) % 180.0
    orientation_hist, _ = np.histogram(orientations, bins=8, range=(0, 180), density=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi / 4], levels=256, symmetric=True, normed=True)
    cloud_mask = estimate_cloud_mask(image)
    num_components, _, stats, _ = cv2.connectedComponentsWithStats((cloud_mask * 255).astype(np.uint8))
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_components > 1 else np.array([0])
    blobs = cv2.SimpleBlobDetector_create().detect(gray)
    dog = difference_of_gaussians(gray.astype(np.float32) / 255.0, 1, 4)
    return {
        "edge_density": edge_density,
        "lbp_mean": float(lbp.mean()),
        "lbp_std": float(lbp.std()),
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "entropy": float(shannon_entropy(gray)),
        "blob_count": float(len(blobs)),
        "blob_area_mean": float(component_areas.mean()),
        "connected_components": float(max(num_components - 1, 0)),
        "dog_mean_abs": float(np.abs(dog).mean()),
        "fractal_dimension_proxy": _fractal_dimension(cloud_mask),
        **{f"edge_orientation_bin_{index}": float(value) for index, value in enumerate(orientation_hist)},
    }
