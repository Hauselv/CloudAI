from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage")

from cloud_aesthetics.explain.heatmaps import fallback_heatmap_from_edges, overlay_heatmap_on_image
from cloud_aesthetics.explain.interpretation import build_interpretation
from cloud_aesthetics.explain.regions import build_region_table
from cloud_aesthetics.explain.text_report import build_text_explanation


def test_explanation_helpers_smoke(tmp_path: Path):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(image, (64, 64), 30, (255, 255, 255), -1)
    image_path = tmp_path / "cloud.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    heatmap = fallback_heatmap_from_edges(image_path)
    overlay = overlay_heatmap_on_image(image, heatmap)
    regions = build_region_table(str(image_path), heatmap)
    text = build_text_explanation(
        predicted_score=8.7,
        uncertainty=0.6,
        feature_contributions=pd.DataFrame([{"feature": "edge_density", "delta_prediction": 0.5, "value": 0.2}]),
        concept_scores=pd.DataFrame([{"concept": "concept_dramatic_edges", "score": 0.8}]),
        scientific_context={
            "primary_cloud_type": "cumulus",
            "cloud_type_confidence": 0.7,
            "dominant_evidence": "cellular cloud texture",
        },
    )
    assert overlay.shape == image.shape
    assert not regions.empty
    assert "Predicted score" in text
    assert "Scientific layer estimates cumulus" in text


def test_interpretation_card_handles_prediction_deltas_and_feature_distances():
    baseline = build_interpretation(
        {
            "predicted_score": 8.2,
            "uncertainty": 0.4,
            "top_features": [{"feature": "edge_density", "delta_prediction": 0.6, "value": 0.12}],
            "top_concepts": [{"concept": "concept_dramatic_edges", "score": 0.8}],
            "scientific_cloud_type": {"primary_cloud_type": "cumulus", "cloud_type_confidence": 0.6},
            "ground_truth": [{"mean_score": 8.0, "n_raters": 2}],
        }
    )
    deep = build_interpretation(
        {
            "predicted_score": 5.0,
            "uncertainty": 1.5,
            "top_features": [
                {
                    "feature": "blob_area_mean",
                    "distance_from_typical": 2.0,
                    "z_score": -2.0,
                    "value": 3.0,
                    "typical_value": 10.0,
                    "attribution_kind": "feature_distance",
                }
            ],
            "top_concepts": [],
            "scientific_cloud_type": {},
            "ground_truth": [],
        }
    )
    assert baseline["recommendation"] == "shortlist"
    assert baseline["drivers"][0]["direction"] == "raises"
    assert deep["drivers"][0]["direction"] == "lower_than_typical"
    assert deep["score_interval"]["low"] == 3.5
