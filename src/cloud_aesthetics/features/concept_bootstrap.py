from __future__ import annotations

import numpy as np
import pandas as pd


CONCEPT_COLUMNS = [
    "concept_towering_clouds",
    "concept_dramatic_edges",
    "concept_rim_lighting",
    "concept_crepuscular_rays",
    "concept_layered_depth",
    "concept_cumulus_like",
    "concept_storm_like",
    "concept_dramatic_mood",
    "concept_strong_depth_cues",
    "concept_cloud_sky_contrast",
]


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def infer_concepts_from_features(feature_row: dict[str, float]) -> dict[str, float]:
    towering = sigmoid(
        3.0 * feature_row.get("vertical_mass_middle", 0.0)
        + 2.5 * feature_row.get("dominant_cloud_area_fraction", 0.0)
        - 1.0 * feature_row.get("horizon_row_fraction", 0.0)
    )
    dramatic_edges = sigmoid(
        2.0 * feature_row.get("edge_density", 0.0)
        + 0.02 * feature_row.get("contrast", 0.0)
        + 0.3 * feature_row.get("entropy", 0.0)
    )
    rim_lighting = sigmoid(
        0.02 * feature_row.get("dynamic_range", 0.0)
        + 0.03 * feature_row.get("contrast", 0.0)
        + 0.5 * feature_row.get("horizon_confidence", 0.0)
    )
    crepuscular_rays = sigmoid(
        1.5 * feature_row.get("saliency_spread", 0.0)
        + 1.5 * feature_row.get("edge_orientation_bin_2", 0.0)
        + 1.5 * feature_row.get("edge_orientation_bin_5", 0.0)
    )
    layered_depth = sigmoid(
        2.0 * abs(feature_row.get("brightness_top", 0.0) - feature_row.get("brightness_bottom", 0.0))
        + 1.5 * feature_row.get("vertical_mass_middle", 0.0)
    )
    cumulus_like = sigmoid(
        2.0 * feature_row.get("connected_components", 0.0) / 20.0
        + 1.5 * feature_row.get("blob_count", 0.0) / 20.0
    )
    storm_like = sigmoid(
        0.04 * feature_row.get("contrast", 0.0)
        + 0.4 * feature_row.get("fractal_dimension_proxy", 0.0)
        - 1.0 * feature_row.get("mean_brightness", 0.0) / 255.0
    )
    dramatic_mood = sigmoid(dramatic_edges + storm_like + layered_depth - 1.5)
    strong_depth_cues = sigmoid(layered_depth + feature_row.get("rule_of_thirds_distance", 0.0))
    cloud_sky_contrast = sigmoid(
        0.03 * feature_row.get("dynamic_range", 0.0)
        + 1.5 * abs(feature_row.get("cloud_area_fraction", 0.0) - feature_row.get("sky_area_fraction", 0.0))
    )
    return {
        "concept_towering_clouds": towering,
        "concept_dramatic_edges": dramatic_edges,
        "concept_rim_lighting": rim_lighting,
        "concept_crepuscular_rays": crepuscular_rays,
        "concept_layered_depth": layered_depth,
        "concept_cumulus_like": cumulus_like,
        "concept_storm_like": storm_like,
        "concept_dramatic_mood": dramatic_mood,
        "concept_strong_depth_cues": strong_depth_cues,
        "concept_cloud_sky_contrast": cloud_sky_contrast,
    }


def infer_concepts_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    concept_rows = [infer_concepts_from_features(row) for row in frame.to_dict("records")]
    concept_frame = pd.DataFrame(concept_rows)
    return pd.concat([frame.reset_index(drop=True), concept_frame], axis=1)
