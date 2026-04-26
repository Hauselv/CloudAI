from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage")

from cloud_aesthetics.explain.heatmaps import fallback_heatmap_from_edges, overlay_heatmap_on_image
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
    )
    assert overlay.shape == image.shape
    assert not regions.empty
    assert "Predicted score" in text
