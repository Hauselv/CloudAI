from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cloud_aesthetics.models.concept import (
    build_concept_table,
    concept_family,
    explain_image,
    load_concept_run,
    select_concept_columns,
    sentence,
    train_concept_suite,
    vocabulary_comparison,
)


def _toy(tmp_path: Path, n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = [f"img{k}" for k in range(n)]
    manifest = pd.DataFrame({"image_id": ids, "relative_path": [f"{i}.jpg" for i in ids], "kind": ["original" if k % 3 else "crop" for k in range(n)], "crop_class": "original", "dataset_name": "toy", "crop_scale": 1.0})
    zero_shot = pd.DataFrame(
        {
            "image_id": ids,
            "clip_concept_layered_depth": rng.normal(size=n),
            "clip_concept_landscape_dominant": rng.normal(size=n),
            "clip_concept_boring_sky": rng.normal(size=n),
            "clip_light_golden_hour": rng.uniform(size=n),
            "clip_concept_z_layered_depth": rng.normal(size=n),  # must be ignored (not a concept family)
        }
    )
    features = pd.DataFrame({"image_id": ids, "palette_neutral_share": rng.uniform(size=n), "entropy": rng.normal(size=n)})
    score = 7 + 1.0 * zero_shot["clip_concept_layered_depth"] - 0.8 * zero_shot["clip_concept_landscape_dominant"] + rng.normal(scale=0.3, size=n)
    labels = pd.DataFrame({"image_id": ids, "mean_score": score.clip(0, 10), "kind": manifest["kind"]})
    splits = pd.DataFrame({"image_id": ids, "partition": ["test" if k % 5 == 0 else "dev" for k in range(n)], "fold": [k % 4 for k in range(n)]})
    table = build_concept_table(manifest, features=features, zero_shot=zero_shot)
    config = {"run_name": "concept_test", "output_dir": str(tmp_path / "artifacts"), "concept_families": ["clip_concept", "clip_light", "palette", "scene"], "regressors": ["lasso"], "bootstrap_samples": 30, "inner_cv": 3, "variants": {"clip": ["clip_concept"], "palette": ["palette"]}}
    return table, labels, splits, config


def test_concept_table_and_column_selection(tmp_path: Path):
    table, _, _, _ = _toy(tmp_path)
    columns = select_concept_columns(table, ["clip_concept", "palette"])
    assert "clip_concept_layered_depth" in columns and "palette_neutral_share" in columns
    assert "clip_concept_z_layered_depth" not in table.columns and "entropy" not in table.columns
    assert concept_family("judge_flag_satellite") == "judge_flag" and concept_family("mean_brightness") is None


def test_concept_suite_recovers_planted_signs_and_is_additive(tmp_path: Path):
    table, labels, splits, config = _toy(tmp_path)
    summary = train_concept_suite(table, labels, splits, config)
    assert summary["kind"] == "concept" and summary["concept_model"]["n_active"] >= 2
    run = load_concept_run(summary["run_id"], tmp_path / "artifacts")
    coefficients = run["concepts"].set_index("concept")
    assert coefficients.loc["clip_concept_layered_depth", "coef_per_sd"] > 0.5
    assert coefficients.loc["clip_concept_landscape_dominant", "coef_per_sd"] < -0.4
    assert coefficients.loc["clip_concept_layered_depth", "sign_agreement"] > 0.95
    assert abs(coefficients.loc["clip_concept_boring_sky", "coef_per_sd"]) < 0.2
    contributions = run["contributions"]
    assert len(contributions) == len(table)
    detail = explain_image(contributions, "img7")
    assert detail.iloc[0]["concept"] in {"clip_concept_layered_depth", "clip_concept_landscape_dominant"}
    row = contributions[contributions["image_id"] == "img7"].iloc[0]
    assert abs(row["intercept"] + detail["contribution"].sum() - row["prediction"]) < 1e-6
    assert summary["models"]["lasso"]["metrics"]["val"]["spearman"] > 0.8
    vocab = run["vocabulary"]
    assert set(vocab["variant"]) == {"clip", "palette"}
    assert vocab.set_index("variant").loc["clip", "oof_spearman"] > vocab.set_index("variant").loc["palette", "oof_spearman"]
    assert "Schichtung" in sentence(run["concepts"])


def test_vocabulary_comparison_skips_empty_families(tmp_path: Path):
    table, labels, splits, config = _toy(tmp_path, n=60)
    frame = vocabulary_comparison(table, labels, splits, config, {"none": ["weather"], "clip": ["clip_concept"]})
    assert list(frame["variant"]) == ["clip"] and "oof_spearman_originals" in frame.columns
