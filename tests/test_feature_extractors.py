from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")

from cloud_aesthetics.features.color import extract_color_features
from cloud_aesthetics.features.composition import extract_composition_features
from cloud_aesthetics.features.scientific import CLOUD_TYPE_TAXONOMY, classify_cloud_type_for_original
from cloud_aesthetics.features.texture import extract_texture_features


def test_feature_extractors_return_expected_keys():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 2] = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    image[:, :, 1] = 180
    image[32:96, 24:104, 0] = 220
    color = extract_color_features(image)
    texture = extract_texture_features(image)
    composition = extract_composition_features(image)
    assert "mean_brightness" in color
    assert "edge_density" in texture
    assert "horizon_confidence" in composition


def test_scientific_cloud_type_layer_returns_taxonomy_scores():
    image = np.full((128, 128, 3), 210, dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.circle(image, (64, 76), 34, (245, 245, 245), -1)
    features = classify_cloud_type_for_original(image)
    assert features["primary_cloud_type"] in CLOUD_TYPE_TAXONOMY
    assert 0.0 <= features["cloud_type_confidence"] <= 1.0
    assert "rule_based_evidence" in features
    for cloud_type in CLOUD_TYPE_TAXONOMY:
        assert f"cloud_type_score_{cloud_type}" in features
