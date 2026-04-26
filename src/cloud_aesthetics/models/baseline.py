from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cloud_aesthetics.eval.metrics import compute_regression_metrics
from cloud_aesthetics.models.base import RunContext, create_run_context, save_run_json
from cloud_aesthetics.models.ranking import train_pairwise_feature_model
from cloud_aesthetics.models.uncertainty import conformal_interval


def _make_regressors(seed: int) -> dict[str, Pipeline]:
    return {
        "elasticnet": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=seed, min_samples_leaf=2)),
            ]
        ),
        "hist_gb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(random_state=seed, max_depth=4)),
            ]
        ),
    }


def _split_frame(features: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame, target_column: str, fold_holdout: int):
    merged = (
        features.merge(labels[["image_id", target_column]], on="image_id", how="inner")
        .merge(splits[["image_id", "partition", "fold"]], on="image_id", how="inner")
        .sort_values("image_id")
    )
    numeric_columns = [column for column in features.columns if column != "image_id"]
    train_mask = (merged["partition"] == "dev") & (merged["fold"] != fold_holdout)
    val_mask = (merged["partition"] == "dev") & (merged["fold"] == fold_holdout)
    test_mask = merged["partition"] == "test"
    return merged, numeric_columns, train_mask, val_mask, test_mask


def train_baseline_suite(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
    pairwise_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    run = create_run_context(config["output_dir"], str(config["run_name"]), config)
    merged, numeric_columns, train_mask, val_mask, test_mask = _split_frame(
        features,
        labels,
        splits,
        str(config["target_column"]),
        int(config.get("fold_holdout", 0)),
    )
    X_train = merged.loc[train_mask, numeric_columns]
    y_train = merged.loc[train_mask, str(config["target_column"])].to_numpy(dtype=np.float32)
    X_val = merged.loc[val_mask, numeric_columns]
    y_val = merged.loc[val_mask, str(config["target_column"])].to_numpy(dtype=np.float32)
    X_test = merged.loc[test_mask, numeric_columns]
    y_test = merged.loc[test_mask, str(config["target_column"])].to_numpy(dtype=np.float32)
    model_names = config.get("regressors", ["elasticnet", "random_forest", "hist_gb"])
    regressors = _make_regressors(int(config.get("seed", 42)))
    results: dict[str, object] = {"run_id": run.run_id, "kind": "baseline", "models": {}}
    val_predictions = []
    val_index = merged.loc[val_mask, "image_id"].tolist()
    for model_name in model_names:
        estimator = regressors[str(model_name)]
        estimator.fit(X_train, y_train)
        pred_val = estimator.predict(X_val) if len(X_val) else np.array([])
        pred_test = estimator.predict(X_test) if len(X_test) else np.array([])
        metrics = {
            "val": compute_regression_metrics(y_val, pred_val) if len(pred_val) else {},
            "test": compute_regression_metrics(y_test, pred_test) if len(pred_test) else {},
        }
        interval = conformal_interval(y_val, pred_val) if len(pred_val) else 0.0
        model_path = run.run_dir / f"{model_name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(estimator, handle)
        results["models"][str(model_name)] = {
            "metrics": metrics,
            "model_path": str(model_path),
            "conformal_half_width": interval,
        }
        if len(pred_val):
            val_predictions.append(pred_val)
    if val_predictions:
        ensemble = np.vstack(val_predictions)
        results["ensemble_val"] = {
            "image_ids": val_index,
            "mean_prediction": ensemble.mean(axis=0).tolist(),
            "std_prediction": ensemble.std(axis=0).tolist(),
        }
    if pairwise_table is not None and not pairwise_table.empty:
        dev_image_ids = set(merged.loc[train_mask | val_mask, "image_id"])
        test_image_ids = set(merged.loc[test_mask, "image_id"])
        pairwise_result = train_pairwise_feature_model(
            features,
            pairwise_train=pairwise_table[
                pairwise_table["left_image_id"].isin(dev_image_ids) & pairwise_table["right_image_id"].isin(dev_image_ids)
            ],
            pairwise_eval=pairwise_table[
                pairwise_table["left_image_id"].isin(test_image_ids) & pairwise_table["right_image_id"].isin(test_image_ids)
            ],
        )
        if pairwise_result.get("available"):
            model_path = run.run_dir / "pairwise_logreg.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(pairwise_result["model"], handle)
            pairwise_result["model_path"] = str(model_path)
            pairwise_result.pop("model", None)
        results["pairwise"] = pairwise_result
    save_run_json(results, run.run_dir / "summary.json")
    return results
