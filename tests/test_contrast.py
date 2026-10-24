from __future__ import annotations

import numpy as np
import pandas as pd

from cloud_aesthetics.eval.contrast import Band, assemble_analysis_table, benjamini_hochberg, contrast_bands, feature_family, judge_feature_table, top_end_trends


def test_feature_family_assignment():
    assert feature_family("clip_concept_z_x") == "skip"
    assert feature_family("clip_concept_boring_sky") == "clip_concept"
    assert feature_family("clip_type_cumulus") == "clip_type"
    assert feature_family("judge_score_mean") == "judge"
    assert feature_family("concept_rim_lighting") == "concept"
    assert feature_family("meteo_layer_score") == "meteo"
    assert feature_family("contrast") == "color"
    assert feature_family("edge_density") == "texture"
    assert feature_family("rule_of_thirds_distance") == "composition"


def test_benjamini_hochberg_is_monotone_and_bounded():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] <= q[2] <= q[1] <= q[3]
    assert abs(q[3] - 0.5) < 1e-9


def test_judge_feature_table_one_hot_and_flags():
    a = pd.DataFrame({"image_id": ["x", "y"], "judge_score": [8.0, 3.0], "judge_confidence": [0.8, 0.6], "judge_cloud_type": ["cumulus", "stratus"], "judge_light": ["golden_hour", "daylight"], "judge_is_sunset": [True, False], "judge_flags": ["", "blurry_or_noisy|landscape_dominant"]})
    b = pd.DataFrame({"image_id": ["x", "y"], "judge_score": [7.0, 3.5], "judge_confidence": [0.7, 0.5], "judge_cloud_type": ["altocumulus", "stratus"], "judge_light": ["golden_hour", "daylight"], "judge_is_sunset": [True, False], "judge_flags": ["", "blurry_or_noisy"]})
    table = judge_feature_table({"a": a, "b": b}).set_index("image_id")
    assert table.loc["x", "judge_score_mean"] == 7.5 and table.loc["x", "judge_score_spread"] == 1.0
    assert table.loc["x", "judge_type_cumulus"] == 0.5 and table.loc["y", "judge_type_stratus"] == 1.0
    assert table.loc["x", "judge_light_golden_hour"] == 1.0
    assert table.loc["y", "judge_flag_blurry_or_noisy"] == 1.0 and table.loc["y", "judge_flag_landscape_dominant"] == 0.5
    assert table.loc["x", "judge_n_flags"] == 0.0 and table.loc["y", "judge_n_flags"] == 1.5


def test_contrast_bands_detects_planted_effect():
    rng = np.random.default_rng(0)
    n = 60
    scores = np.concatenate([np.full(30, 9.5), np.full(30, 8.0)])
    labels = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "mean_score": scores, "std_score": 0.0, "n_raters": 1})
    manifest = pd.DataFrame({"image_id": labels["image_id"], "relative_path": "x.jpg", "kind": "original", "crop_class": "original", "dataset_name": "d"})
    features = pd.DataFrame({"image_id": labels["image_id"], "contrast": np.where(scores > 9, 60, 40) + rng.normal(0, 3, n), "noise_feature": rng.normal(0, 1, n), "scientific_source_image_id": "s"})
    zero_shot = pd.DataFrame({"image_id": labels["image_id"], "clip_concept_spectacular_sky": np.where(scores > 9, 0.02, -0.01) + rng.normal(0, 0.01, n), "clip_concept_z_spectacular_sky": 0.0})
    table = assemble_analysis_table(labels, manifest, features, zero_shot, None)
    assert "clip_concept_z_spectacular_sky" not in table.columns
    result = contrast_bands(table, Band(9.5, 10, "A"), Band(7.5, 8.5, "B"))
    assert result["n_a"] == 30 and result["n_b"] == 30
    frame = result["table"].set_index("feature")
    assert frame.loc["contrast", "cohens_d"] > 2.0 and frame.loc["contrast", "q_mannwhitney"] < 0.01
    assert frame.loc["clip_concept_spectacular_sky", "cohens_d"] > 1.0
    assert abs(frame.loc["noise_feature", "cohens_d"]) < 0.8
    assert result["table"].iloc[0]["feature"] in {"contrast", "clip_concept_spectacular_sky"}
    assert frame.loc["contrast", "direction"].startswith("höher bei A")
    too_small = contrast_bands(table, Band(9.9, 10, "A"), Band(7.5, 8.5, "B"))
    assert too_small["table"].empty and too_small["n_a"] == 0


def test_top_end_trends_ranks_monotone_feature_first():
    n = 40
    scores = np.linspace(7.0, 10.0, n)
    rng = np.random.default_rng(1)
    table = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "mean_score": scores, "trend": scores * 2 + rng.normal(0, 0.1, n), "flat": rng.normal(0, 1, n)})
    trends = top_end_trends(table, min_score=7.0)
    assert trends.iloc[0]["feature"] == "trend" and trends.iloc[0]["spearman"] > 0.95
    assert top_end_trends(table, min_score=9.95).empty
