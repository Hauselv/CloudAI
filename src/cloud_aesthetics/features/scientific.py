from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from cloud_aesthetics.preprocessing.image_ops import estimate_cloud_mask, rgb_to_gray


CLOUD_TYPE_TAXONOMY = [
    "cirrus",
    "cirrocumulus",
    "cirrostratus",
    "altocumulus",
    "altostratus",
    "nimbostratus",
    "stratocumulus",
    "stratus",
    "cumulus",
    "cumulonimbus",
]

SCIENTIFIC_CAVEAT = (
    "Cloud type is an image-based scientific estimate from visual cues, not a meteorological observation."
)


@dataclass(frozen=True)
class CloudTypeResult:
    primary_cloud_type: str
    cloud_type_confidence: float
    cloud_type_scores: dict[str, float]
    rule_based_evidence: dict[str, float | str]
    vision_model_vote: str
    scientific_caveat: str


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _normalized_entropy(values: np.ndarray) -> float:
    values = values.astype(np.float32)
    values = values / (values.sum() + 1e-6)
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(len(values)))


def extract_meteorological_features(image: np.ndarray) -> dict[str, float]:
    gray = rgb_to_gray(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cloud_mask = estimate_cloud_mask(image)
    height, width = gray.shape
    upper = cloud_mask[: max(1, height // 3)]
    middle = cloud_mask[height // 3 : 2 * height // 3]
    lower = cloud_mask[2 * height // 3 :]

    edges = cv2.Canny(gray, 60, 160)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    orientations = (np.rad2deg(np.arctan2(grad_y, grad_x)) + 180.0) % 180.0
    orientation_hist, _ = np.histogram(orientations[edges > 0], bins=8, range=(0, 180))

    num_components, _, stats, _ = cv2.connectedComponentsWithStats((cloud_mask * 255).astype(np.uint8))
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if num_components > 1 else np.array([], dtype=np.float32)
    largest_area = float(component_areas.max() / (height * width)) if component_areas.size else 0.0
    small_components = float((component_areas < (height * width * 0.015)).sum()) if component_areas.size else 0.0
    total_components = float(max(num_components - 1, 0))

    vertical_profile = cloud_mask.mean(axis=1)
    vertical_peak = int(np.argmax(vertical_profile)) if vertical_profile.size else 0
    vertical_spread = float(vertical_profile.std())
    cloud_pixels = gray[cloud_mask.astype(bool)]
    cloud_brightness = float(cloud_pixels.mean()) if cloud_pixels.size else float(gray.mean())
    cloud_darkness = 1.0 - cloud_brightness / 255.0
    saturation_mean = float(hsv[:, :, 1].mean() / 255.0)
    layer_score = 1.0 - min(1.0, vertical_spread * 4.0)
    fibrous_score = float(np.percentile(grad_mag, 90) / 255.0)
    cellular_score = min(1.0, small_components / 18.0 + total_components / 60.0)
    towering_score = min(
        1.0,
        float(middle.mean()) * 0.7
        + float(lower.mean()) * 0.35
        + largest_area * 0.8
        + max(0.0, 0.5 - vertical_peak / max(height, 1)) * 0.5,
    )

    return {
        "meteo_cloud_fraction": float(cloud_mask.mean()),
        "meteo_upper_cloud_fraction": float(upper.mean()) if upper.size else 0.0,
        "meteo_middle_cloud_fraction": float(middle.mean()) if middle.size else 0.0,
        "meteo_lower_cloud_fraction": float(lower.mean()) if lower.size else 0.0,
        "meteo_layer_score": float(layer_score),
        "meteo_fibrous_score": float(fibrous_score),
        "meteo_cellular_score": float(cellular_score),
        "meteo_towering_score": float(towering_score),
        "meteo_darkness_score": float(cloud_darkness),
        "meteo_edge_density": float((edges > 0).mean()),
        "meteo_orientation_entropy": _normalized_entropy(orientation_hist.astype(np.float32)),
        "meteo_largest_cloud_area_fraction": largest_area,
        "meteo_small_component_count": small_components,
        "meteo_saturation_mean": saturation_mean,
    }


def classify_cloud_type_from_meteorology(features: dict[str, float]) -> CloudTypeResult:
    upper = features.get("meteo_upper_cloud_fraction", 0.0)
    middle = features.get("meteo_middle_cloud_fraction", 0.0)
    lower = features.get("meteo_lower_cloud_fraction", 0.0)
    layer = features.get("meteo_layer_score", 0.0)
    fibrous = features.get("meteo_fibrous_score", 0.0)
    cellular = features.get("meteo_cellular_score", 0.0)
    towering = features.get("meteo_towering_score", 0.0)
    darkness = features.get("meteo_darkness_score", 0.0)
    cloud_fraction = features.get("meteo_cloud_fraction", 0.0)
    edge_density = features.get("meteo_edge_density", 0.0)
    largest = features.get("meteo_largest_cloud_area_fraction", 0.0)
    orientation_entropy = features.get("meteo_orientation_entropy", 0.0)

    raw_scores = {
        "cirrus": _sigmoid(2.2 * upper + 2.0 * fibrous + 0.8 * orientation_entropy - 1.3 * darkness - 1.0),
        "cirrocumulus": _sigmoid(2.0 * upper + 2.0 * cellular + 1.2 * edge_density - 1.4),
        "cirrostratus": _sigmoid(2.4 * upper + 2.0 * layer + 0.8 * cloud_fraction - 1.6 * cellular - 1.0),
        "altocumulus": _sigmoid(2.2 * middle + 2.2 * cellular + 0.8 * edge_density - 1.2),
        "altostratus": _sigmoid(2.5 * middle + 2.0 * layer + 0.7 * cloud_fraction - 1.3 * cellular - 1.0),
        "nimbostratus": _sigmoid(2.2 * layer + 2.4 * darkness + 1.2 * cloud_fraction - 1.4),
        "stratocumulus": _sigmoid(2.0 * lower + 1.8 * cellular + 1.0 * largest + 0.8 * layer - 1.2),
        "stratus": _sigmoid(2.4 * lower + 2.4 * layer + 0.8 * cloud_fraction - 1.6 * cellular - 1.0),
        "cumulus": _sigmoid(2.2 * lower + 2.0 * largest + 1.4 * cellular + 1.0 * edge_density - 1.2),
        "cumulonimbus": _sigmoid(3.0 * towering + 1.7 * darkness + 1.5 * largest + 1.0 * edge_density - 1.5),
    }
    total = sum(raw_scores.values()) + 1e-6
    scores = {cloud_type: float(value / total) for cloud_type, value in raw_scores.items()}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_type, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = float(max(0.0, min(1.0, top_score + (top_score - runner_up))))
    evidence = {
        "dominant_evidence": _dominant_evidence_name(features),
        "upper_cloud_fraction": upper,
        "middle_cloud_fraction": middle,
        "lower_cloud_fraction": lower,
        "layer_score": layer,
        "fibrous_score": fibrous,
        "cellular_score": cellular,
        "towering_score": towering,
        "darkness_score": darkness,
    }
    return CloudTypeResult(
        primary_cloud_type=top_type,
        cloud_type_confidence=confidence,
        cloud_type_scores=scores,
        rule_based_evidence=evidence,
        vision_model_vote="not_configured",
        scientific_caveat=SCIENTIFIC_CAVEAT,
    )


def _dominant_evidence_name(features: dict[str, float]) -> str:
    candidates = {
        "fibrous high-cloud structure": features.get("meteo_fibrous_score", 0.0) + features.get("meteo_upper_cloud_fraction", 0.0),
        "layered sheet structure": features.get("meteo_layer_score", 0.0),
        "cellular cloud texture": features.get("meteo_cellular_score", 0.0),
        "vertical convective development": features.get("meteo_towering_score", 0.0),
        "dark precipitation-like cloud mass": features.get("meteo_darkness_score", 0.0),
    }
    return max(candidates.items(), key=lambda item: item[1])[0]


def serialize_scientific_result(result: CloudTypeResult) -> dict[str, float | str]:
    return {
        "primary_cloud_type": result.primary_cloud_type,
        "cloud_type_confidence": result.cloud_type_confidence,
        "cloud_type_scores": json.dumps(result.cloud_type_scores, sort_keys=True),
        "rule_based_evidence": json.dumps(result.rule_based_evidence, sort_keys=True),
        "vision_model_vote": result.vision_model_vote,
        "scientific_caveat": result.scientific_caveat,
        **{f"cloud_type_score_{cloud_type}": result.cloud_type_scores.get(cloud_type, 0.0) for cloud_type in CLOUD_TYPE_TAXONOMY},
    }


def unknown_scientific_result(reason: str = "not_full_original") -> dict[str, float | str]:
    scores = {cloud_type: 0.0 for cloud_type in CLOUD_TYPE_TAXONOMY}
    result = CloudTypeResult(
        primary_cloud_type="unknown",
        cloud_type_confidence=0.0,
        cloud_type_scores=scores,
        rule_based_evidence={"reason": reason},
        vision_model_vote="not_configured",
        scientific_caveat=SCIENTIFIC_CAVEAT,
    )
    return serialize_scientific_result(result)


def classify_cloud_type_for_original(image: np.ndarray) -> dict[str, float | str]:
    meteorology = extract_meteorological_features(image)
    result = classify_cloud_type_from_meteorology(meteorology)
    return {**meteorology, **serialize_scientific_result(result)}
