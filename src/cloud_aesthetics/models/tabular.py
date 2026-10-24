"""Shared grouped k-fold training loop for tabular regressors.

Used by the hand-crafted feature baseline and the embedding models so both report the
same metrics: pooled out-of-fold validation, per-fold spread, held-out test, and metrics
per image kind / dataset.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from cloud_aesthetics.eval.metrics import compute_regression_metrics
from cloud_aesthetics.models.base import create_run_context, save_run_json
from cloud_aesthetics.models.uncertainty import conformal_interval

GROUP_COLUMNS = ("kind", "dataset_name")
MIN_GROUP_SIZE = 5


def numeric_input_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in table.columns
        if column != "image_id" and column not in GROUP_COLUMNS and pd.api.types.is_numeric_dtype(table[column])
    ]


def merge_training_frame(table: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame, target_column: str):
    label_columns = ["image_id", target_column] + [column for column in GROUP_COLUMNS if column in labels.columns]
    inputs = table.drop(columns=[column for column in GROUP_COLUMNS if column in table.columns]).copy()
    inputs["image_id"] = inputs["image_id"].astype(str)
    labels = labels.copy()
    labels["image_id"] = labels["image_id"].astype(str)
    splits = splits.copy()
    splits["image_id"] = splits["image_id"].astype(str)
    merged = (
        inputs.merge(labels[label_columns], on="image_id", how="inner")
        .merge(splits[["image_id", "partition", "fold"]], on="image_id", how="inner")
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    return merged, numeric_input_columns(inputs)


def summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not fold_metrics:
        return {"mean": {}, "std": {}}
    frame = pd.DataFrame(fold_metrics).drop(columns=["fold", "n"], errors="ignore")
    return {
        "mean": {column: float(frame[column].mean()) for column in frame.columns},
        "std": {column: float(frame[column].std(ddof=0)) for column in frame.columns},
    }


def metrics_by_group(frame: pd.DataFrame, predictions: np.ndarray, target_column: str) -> dict[str, dict[str, dict[str, float]]]:
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


def train_tabular_suite(
    table: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
    estimators: dict[str, BaseEstimator],
    *,
    kind: str,
    extra_summary: dict[str, object] | None = None,
) -> tuple[dict[str, object], object]:
    """Grouped k-fold CV on the dev partition, refit on all dev rows, evaluate on test.

    Returns the summary dict and the run context. The caller may add more entries
    (for example pairwise metrics) and must save the summary afterwards.
    """
    run = create_run_context(config["output_dir"], str(config["run_name"]), config)
    target_column = str(config.get("target_column", "mean_score"))
    merged, numeric_columns = merge_training_frame(table, labels, splits, target_column)
    dev = merged[merged["partition"] == "dev"].reset_index(drop=True)
    test = merged[merged["partition"] == "test"].reset_index(drop=True)
    if dev.empty:
        raise ValueError("Training needs labeled dev images. Run `aggregate-labels` first.")
    folds = sorted(int(fold) for fold in dev["fold"].dropna().unique())
    cross_validate = bool(config.get("cross_validate", True)) and len(folds) >= 2
    y_dev = dev[target_column].to_numpy(dtype=np.float32)
    y_test = test[target_column].to_numpy(dtype=np.float32)
    summary: dict[str, object] = {
        "run_id": run.run_id,
        "kind": kind,
        "evaluation": {
            "scheme": "grouped k-fold CV on dev partition + held-out test" if cross_validate else "single fold holdout",
            "n_folds": len(folds) if cross_validate else 1,
            "n_dev": int(len(dev)),
            "n_test": int(len(test)),
            "n_features": len(numeric_columns),
        },
        "input_columns": numeric_columns,
        "train_image_ids": dev["image_id"].astype(str).tolist(),
        "models": {},
    }
    if extra_summary:
        summary.update(extra_summary)
    group_cols = [column for column in GROUP_COLUMNS if column in dev.columns]
    oof_table = dev[["image_id", target_column] + group_cols].copy()
    oof_table["partition"] = "dev"
    test_table = test[["image_id", target_column] + group_cols].copy()
    test_table["partition"] = "test"
    oof_predictions: list[np.ndarray] = []

    for model_name, template in estimators.items():
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
        summary["models"][model_name] = {
            "metrics": {"val": val_metrics, "test": test_metrics},
            "cv": {"folds": fold_metrics, **summarize_fold_metrics(fold_metrics)},
            "by_group": {
                "val": metrics_by_group(dev[has_oof].reset_index(drop=True), oof[has_oof], target_column),
                "test": metrics_by_group(test, pred_test, target_column) if len(pred_test) else {},
            },
            "model_path": str(model_path),
            "conformal_half_width": conformal_interval(y_dev[has_oof], oof[has_oof]) if has_oof.any() else 0.0,
            "fitted_params": _fitted_params(final_model),
        }
        oof_table[f"oof_{model_name}"] = oof
        test_table[f"pred_{model_name}"] = pred_test if len(pred_test) else np.nan
        if has_oof.any():
            oof_predictions.append(oof)

    if oof_predictions:
        ensemble = np.vstack(oof_predictions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # rows without OOF predictions are NaN by design
            summary["ensemble_val"] = {
                "image_ids": dev["image_id"].astype(str).tolist(),
                "mean_prediction": np.nanmean(ensemble, axis=0).tolist(),
                "std_prediction": np.nanstd(ensemble, axis=0).tolist(),
            }
    predictions_path = run.run_dir / "predictions.parquet"
    pd.concat([oof_table, test_table], ignore_index=True).to_parquet(predictions_path, index=False)
    summary["predictions_path"] = str(predictions_path)
    save_run_json(summary, run.run_dir / "summary.json")
    return summary, run


def _fitted_params(model: BaseEstimator) -> dict[str, object]:
    """Small, JSON-safe record of data-dependent hyperparameters (e.g. the chosen ridge alpha)."""
    final = model.named_steps["model"] if hasattr(model, "named_steps") else model
    params: dict[str, object] = {}
    for name in ("alpha_", "n_neighbors", "best_params_"):
        value = getattr(final, name, None)
        if value is None:
            continue
        params[name.rstrip("_")] = float(value) if isinstance(value, (int, float, np.floating, np.integer)) else str(value)
    return params
