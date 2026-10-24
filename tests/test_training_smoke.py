from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cloud_aesthetics.models.baseline import train_baseline_suite


def _toy_data(n: int = 24):
    features = pd.DataFrame(
        [
            {
                "image_id": f"img{i}",
                "feat_a": float(i),
                "feat_b": float(i % 3),
                "feat_c": float(i * 0.5),
                "kind": "crop" if i % 2 else "original",
                "dataset_name": "toy",
            }
            for i in range(n)
        ]
    )
    labels = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(n)],
            "mean_score": [float(3 + (i % 7)) for i in range(n)],
            "kind": ["crop" if i % 2 else "original" for i in range(n)],
            "dataset_name": ["toy"] * n,
        }
    )
    n_test = n // 4
    n_dev = n - n_test
    partitions = ["dev"] * n_dev + ["test"] * n_test
    folds = [i % 3 for i in range(n_dev)] + [-1] * n_test
    splits = pd.DataFrame({"image_id": [f"img{i}" for i in range(n)], "partition": partitions, "fold": folds})
    return features, labels, splits


def test_baseline_training_smoke(tmp_path: Path):
    features, labels, splits = _toy_data()
    config = {
        "output_dir": str(tmp_path),
        "run_name": "baseline_smoke",
        "target_column": "mean_score",
        "seed": 42,
        "regressors": ["elasticnet", "random_forest"],
    }
    summary = train_baseline_suite(features, labels, splits, config, pairwise_table=pd.DataFrame())
    assert summary["kind"] == "baseline"
    assert summary["evaluation"]["n_folds"] == 3
    assert summary["evaluation"]["n_features"] == 3  # kind/dataset_name are not model inputs
    model = summary["models"]["random_forest"]
    assert set(model["metrics"]) == {"val", "test"}
    assert len(model["cv"]["folds"]) == 3
    assert "mae" in model["cv"]["mean"] and "mae" in model["cv"]["std"]
    assert set(model["by_group"]["val"]["kind"]) == {"crop", "original"}
    assert model["conformal_half_width"] >= 0.0
    predictions = pd.read_parquet(summary["predictions_path"])
    assert len(predictions) == 24
    assert predictions.loc[predictions["partition"] == "dev", "oof_random_forest"].notna().all()
    assert predictions.loc[predictions["partition"] == "test", "pred_random_forest"].notna().all()
    assert json.loads((tmp_path / summary["run_id"] / "summary.json").read_text(encoding="utf-8"))["run_id"] == summary["run_id"]


def test_baseline_single_fold_mode(tmp_path: Path):
    features, labels, splits = _toy_data()
    config = {
        "output_dir": str(tmp_path),
        "run_name": "baseline_single",
        "target_column": "mean_score",
        "cross_validate": False,
        "fold_holdout": 0,
        "regressors": ["elasticnet"],
    }
    summary = train_baseline_suite(features, labels, splits, config)
    assert summary["evaluation"]["n_folds"] == 1
    assert summary["models"]["elasticnet"]["cv"]["folds"] == []
    assert summary["models"]["elasticnet"]["metrics"]["val"]
