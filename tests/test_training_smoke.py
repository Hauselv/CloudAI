from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cloud_aesthetics.models.baseline import train_baseline_suite


def test_baseline_training_smoke(tmp_path: Path):
    features = pd.DataFrame(
        [
            {"image_id": f"img{i}", "feat_a": float(i), "feat_b": float(i % 3), "feat_c": float(i * 0.5)}
            for i in range(12)
        ]
    )
    labels = pd.DataFrame({"image_id": [f"img{i}" for i in range(12)], "mean_score": [float(5 + (i % 5)) for i in range(12)]})
    partitions = ["dev"] * 9 + ["test"] * 3
    folds = [i % 3 for i in range(9)] + [-1] * 3
    splits = pd.DataFrame({"image_id": [f"img{i}" for i in range(12)], "partition": partitions, "fold": folds})
    config = {
        "output_dir": str(tmp_path),
        "run_name": "baseline_smoke",
        "target_column": "mean_score",
        "fold_holdout": 0,
        "seed": 42,
        "regressors": ["elasticnet", "random_forest"],
    }
    summary = train_baseline_suite(features, labels, splits, config, pairwise_table=pd.DataFrame())
    assert summary["kind"] == "baseline"
    assert summary["models"]
