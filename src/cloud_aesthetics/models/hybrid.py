from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from cloud_aesthetics.eval.metrics import compute_regression_metrics
from cloud_aesthetics.models.base import create_run_context, save_run_json


def train_hybrid_model(
    features: pd.DataFrame,
    embeddings: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    run = create_run_context(config["output_dir"], str(config["run_name"]), config)
    merged = (
        features.merge(embeddings, on="image_id", how="inner", suffixes=("_feat", "_embed"))
        .merge(labels[["image_id", str(config["target_column"])]], on="image_id", how="inner")
        .merge(splits[["image_id", "partition", "fold"]], on="image_id", how="inner")
    )
    numeric_columns = [column for column in merged.columns if column not in {"image_id", "partition", "fold", str(config["target_column"])}]
    train_mask = (merged["partition"] == "dev") & (merged["fold"] != int(config.get("fold_holdout", 0)))
    test_mask = merged["partition"] == "test"
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=int(config.get("seed", 42)), max_depth=4)),
        ]
    )
    model.fit(merged.loc[train_mask, numeric_columns], merged.loc[train_mask, str(config["target_column"])])
    preds = model.predict(merged.loc[test_mask, numeric_columns])
    metrics = compute_regression_metrics(
        merged.loc[test_mask, str(config["target_column"])].to_numpy(dtype=np.float32),
        preds,
    )
    summary = {"run_id": run.run_id, "kind": "hybrid", "metrics": metrics}
    save_run_json(summary, run.run_dir / "summary.json")
    return summary
