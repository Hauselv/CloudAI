from __future__ import annotations

import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cloud_aesthetics.models.base import save_run_json
from cloud_aesthetics.models.ranking import train_pairwise_feature_model
from cloud_aesthetics.models.tabular import GROUP_COLUMNS, train_tabular_suite


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


def _attach_global_shap(summary: dict[str, object], run, features: pd.DataFrame) -> None:
    """Mean |SHAP| per feature on the dev rows for every model that has an explainer."""
    try:
        from cloud_aesthetics.explain.feature_importance import load_pickled_model
        from cloud_aesthetics.explain.shap_values import global_shap_importance
    except ImportError:
        return
    train_ids = set(summary.get("train_image_ids", []))
    columns = list(summary.get("input_columns", []))
    frame = features.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    dev = frame[frame["image_id"].isin(train_ids)][columns]
    if dev.empty or not columns:
        return
    for model_name, info in summary["models"].items():
        try:
            model = load_pickled_model(info["model_path"])
            table = global_shap_importance(model, dev)
        except (ValueError, ImportError, RuntimeError):
            continue
        path = run.run_dir / f"shap_importance_{model_name}.parquet"
        table.to_parquet(path, index=False)
        info["shap_importance_path"] = str(path)
        info["shap_top"] = table.head(15)[["feature", "mean_abs_shap", "mean_shap", "direction"]].to_dict("records")
    save_run_json(summary, run.run_dir / "summary.json")


def train_baseline_suite(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
    pairwise_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Train the hand-crafted feature regressors with grouped k-fold CV (see ``models.tabular``)."""
    model_names = [str(name) for name in config.get("regressors", ["elasticnet", "random_forest", "hist_gb"])]
    regressors = _make_regressors(int(config.get("seed", 42)))
    estimators = {name: regressors[name] for name in model_names}
    summary, run = train_tabular_suite(features, labels, splits, config, estimators, kind="baseline")

    if bool(config.get("shap", True)):
        _attach_global_shap(summary, run, features)

    if pairwise_table is not None and not pairwise_table.empty:
        dev_image_ids = set(summary["train_image_ids"])
        predictions = pd.read_parquet(summary["predictions_path"])
        test_image_ids = set(predictions.loc[predictions["partition"] == "test", "image_id"].astype(str))
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
        summary["pairwise"] = pairwise_result
        save_run_json(summary, run.run_dir / "summary.json")
    return summary
