from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cloud_aesthetics.eval.metrics import compute_regression_metrics
from cloud_aesthetics.models.base import RunContext, create_run_context, save_run_json
from cloud_aesthetics.models.ranking import train_pairwise_feature_model
from cloud_aesthetics.models.uncertainty import conformal_interval

GROUP_COLUMNS = ("kind", "dataset_name")
MIN_GROUP_SIZE = 5


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


def _merge_frames(features: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame, target_column: str):
    label_columns = ["image_id", target_column] + [column for column in GROUP_COLUMNS if column in labels.columns]
    feature_frame = features.drop(columns=[column for column in GROUP_COLUMNS if column in features.columns])
    merged = (
        feature_frame.merge(labels[label_columns], on="image_id", how="inner")
        .merge(splits[["image_id", "partition", "fold"]], on="image_id", how="inner")
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    numeric_columns = [
        column
        for column in feature_frame.columns
        if column != "image_id" and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    return merged, numeric_columns


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not fold_metrics:
        return {"mean": {}, "std": {}}
    frame = pd.DataFrame(fold_metrics).drop(columns=["fold", "n"], errors="ignore")
    return {
        "mean": {column: float(frame[column].mean()) for column in frame.columns},
        "std": {column: float(frame[column].std(ddof=0)) for column in frame.columns},
    }


def _metrics_by_group(frame: pd.DataFrame, predictions: np.ndarray, target_column: str) -> dict[str, dict[str, dict[str, float]]]:
    """Metrics split by crop/original and by dataset, so confounds stay visible."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    if frame.empty or len(predictions) != len(frame):
        return result
    predictions = np.asarray(predictions, dtype=np.float32)
    for column in GROUP_COLUMNS:
        if column not in frame.columns:
            continue
        groups: dict[str, dict[str, float]] = {}
        values = frame[column].fillna("unknown").astype(str).to_numpy()
        for group in sorted(set(values)):
            mask = values == group
            if mask.sum() < MIN_GROUP_SIZE:
                continue
            metrics = compute_regression_metrics(frame.loc[mask, target_column].to_numpy(dtype=np.float32), predictions[mask])
            metrics["n"] = int(mask.sum())
            metrics["mean_target"] = float(frame.loc[mask, target_column].mean())
            metrics["mean_prediction"] = float(predictions[mask].mean())
            groups[group] = metrics
        if groups:
            result[column] = groups
    return result


def train_baseline_suite(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
    pairwise_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Train the feature-based regressors with grouped K-fold CV on the dev partition.

    Every dev fold is held out once, giving out-of-fold (OOF) predictions for all dev
    images. ``metrics.val`` is computed on the pooled OOF predictions, ``cv`` holds the
    per-fold mean and standard deviation, and the final model is refit on the whole dev
    partition before it is evaluated on the held-out ``test`` partition.
    """
    run = create_run_context(config["output_dir"], str(config["run_name"]), config)
    target_column = str(config["target_column"])
    merged, numeric_columns = _merge_frames(features, labels, splits, target_column)
    dev = merged[merged["partition"] == "dev"].reset_index(drop=True)
    test = merged[merged["partition"] == "test"].reset_index(drop=True)
    if dev.empty:
        raise ValueError("Baseline training needs labeled dev images. Run `aggregate-labels` first.")
    folds = sorted(int(fold) for fold in dev["fold"].dropna().unique())
    cross_validate = bool(config.get("cross_validate", True)) and len(folds) >= 2
    y_dev = dev[target_column].to_numpy(dtype=np.float32)
    y_test = test[target_column].to_numpy(dtype=np.float32)
    model_names = [str(name) for name in config.get("regressors", ["elasticnet", "random_forest", "hist_gb"])]
    regressors = _make_regressors(int(config.get("seed", 42)))
    results: dict[str, object] = {
        "run_id": run.run_id,
        "kind": "baseline",
        "evaluation": {
            "scheme": "grouped k-fold CV on dev partition + held-out test" if cross_validate else "single fold holdout",
            "n_folds": len(folds) if cross_validate else 1,
            "n_dev": int(len(dev)),
            "n_test": int(len(test)),
            "n_features": len(numeric_columns),
        },
        "models": {},
    }
    oof_table = dev[["image_id", target_column] + [column for column in GROUP_COLUMNS if column in dev.columns]].copy()
    oof_table["partition"] = "dev"
    test_table = test[["image_id", target_column] + [column for column in GROUP_COLUMNS if column in test.columns]].copy()
    test_table["partition"] = "test"
    oof_predictions: list[np.ndarray] = []

    for model_name in model_names:
        template = regressors[model_name]
        oof = np.full(len(dev), np.nan, dtype=np.float32)
        fold_metrics: list[dict[str, float]] = []
        if cross_validate:
            for fold in folds:
                train_mask = (dev["fold"] != fold).to_numpy()
                val_mask = ~train_mask
                if train_mask.sum() == 0 or val_mask.sum() == 0:
                    continue
                estimator = clone(template)
                estimator.fit(dev.loc[train_mask, numeric_columns], y_dev[train_mask])
                oof[val_mask] = estimator.predict(dev.loc[val_mask, numeric_columns])
                metrics = compute_regression_metrics(y_dev[val_mask], oof[val_mask])
                fold_metrics.append({"fold": fold, "n": int(val_mask.sum()), **metrics})
        else:
            holdout = int(config.get("fold_holdout", 0))
            train_mask = (dev["fold"] != holdout).to_numpy()
            val_mask = ~train_mask
            if val_mask.any() and train_mask.any():
                estimator = clone(template)
                estimator.fit(dev.loc[train_mask, numeric_columns], y_dev[train_mask])
                oof[val_mask] = estimator.predict(dev.loc[val_mask, numeric_columns])
        has_oof = ~np.isnan(oof)
        final_model = clone(template)
        final_model.fit(dev[numeric_columns], y_dev)
        pred_test = final_model.predict(test[numeric_columns]) if len(test) else np.array([], dtype=np.float32)
        val_metrics = compute_regression_metrics(y_dev[has_oof], oof[has_oof]) if has_oof.any() else {}
        test_metrics = compute_regression_metrics(y_test, pred_test) if len(pred_test) else {}
        model_path = run.run_dir / f"{model_name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(final_model, handle)
        results["models"][model_name] = {
            "metrics": {"val": val_metrics, "test": test_metrics},
            "cv": {"folds": fold_metrics, **_summarize_fold_metrics(fold_metrics)},
            "by_group": {
                "val": _metrics_by_group(dev[has_oof].reset_index(drop=True), oof[has_oof], target_column),
                "test": _metrics_by_group(test, pred_test, target_column) if len(pred_test) else {},
            },
            "model_path": str(model_path),
            "conformal_half_width": conformal_interval(y_dev[has_oof], oof[has_oof]) if has_oof.any() else 0.0,
        }
        oof_table[f"oof_{model_name}"] = oof
        test_table[f"pred_{model_name}"] = pred_test if len(pred_test) else np.nan
        if has_oof.any():
            oof_predictions.append(oof)

    if oof_predictions:
        ensemble = np.vstack(oof_predictions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # rows without OOF predictions are NaN by design
            results["ensemble_val"] = {
                "image_ids": dev["image_id"].astype(str).tolist(),
                "mean_prediction": np.nanmean(ensemble, axis=0).tolist(),
                "std_prediction": np.nanstd(ensemble, axis=0).tolist(),
            }
    predictions_path = run.run_dir / "predictions.parquet"
    pd.concat([oof_table, test_table], ignore_index=True).to_parquet(predictions_path, index=False)
    results["predictions_path"] = str(predictions_path)

    if pairwise_table is not None and not pairwise_table.empty:
        dev_image_ids = set(dev["image_id"].astype(str))
        test_image_ids = set(test["image_id"].astype(str))
        pairwise_result = train_pairwise_feature_model(
            features.drop(columns=[column for column in GROUP_COLUMNS if column in features.columns]),
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
