from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("shap")

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cloud_aesthetics.eval.calibration import calibration_table, residual_summary
from cloud_aesthetics.explain.shap_values import global_shap_importance, shap_contributions
from cloud_aesthetics.models.baseline import train_baseline_suite
from cloud_aesthetics.models.ranking import train_ranking_suite


def _frame(n: int = 80, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n), "weak": rng.normal(size=n)})
    y = 5 + 2.0 * X["signal"] + 0.3 * X["weak"] + rng.normal(scale=0.2, size=n)
    return X, y.to_numpy()


def test_shap_contributions_are_additive_for_trees_and_linear():
    X, y = _frame()
    tree = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=60, random_state=0))]).fit(X, y)
    row = X.iloc[[3]]
    table, base, method = shap_contributions(tree, row, X)
    assert method == "tree_shap" and set(table["feature"]) == set(X.columns)
    assert abs(base + table["delta_prediction"].sum() - float(tree.predict(row)[0])) < 1e-6
    assert table.iloc[0]["feature"] == "signal"
    linear = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.01))]).fit(X, y)
    table_lin, base_lin, method_lin = shap_contributions(linear, row, X)
    assert method_lin == "linear_shap"
    assert abs(base_lin + table_lin["delta_prediction"].sum() - float(linear.predict(row)[0])) < 1e-4
    importance = global_shap_importance(tree, X)
    assert importance.iloc[0]["feature"] == "signal" and importance.iloc[0]["direction"] > 0.8


def test_calibration_table_and_residual_summary():
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y_pred = y_true * 0.8 + 1.0  # compressed scale
    table = calibration_table(y_true, y_pred, bin_width=2.0)
    assert table["n"].sum() == 10 and (table["gap"].abs() > 0).any()
    stats = residual_summary(y_true, y_pred, half_width=1.0)
    assert stats["n"] == 10 and stats["spread_ratio"] == pytest.approx(0.8)
    assert 0.0 <= stats["coverage"] <= 1.0 and stats["half_width"] == 1.0


def test_baseline_run_stores_global_shap(tmp_path: Path):
    X, y = _frame(n=36)
    features = X.copy()
    features.insert(0, "image_id", [f"i{k}" for k in range(len(X))])
    labels = pd.DataFrame({"image_id": features["image_id"], "mean_score": np.clip(y, 0, 10)})
    splits = pd.DataFrame({"image_id": features["image_id"], "partition": ["dev"] * 27 + ["test"] * 9, "fold": [k % 3 for k in range(27)] + [-1] * 9})
    summary = train_baseline_suite(features, labels, splits, {"output_dir": str(tmp_path), "run_name": "shap_smoke", "target_column": "mean_score", "regressors": ["random_forest"]})
    info = summary["models"]["random_forest"]
    assert info["shap_top"][0]["feature"] == "signal"
    assert Path(info["shap_importance_path"]).exists()


def test_ranking_suite_trains_on_pairs_and_scores_images(tmp_path: Path):
    rng = np.random.default_rng(3)
    n = 40
    ids = [f"i{k}" for k in range(n)]
    strength = rng.normal(size=n)
    table = pd.DataFrame({"image_id": ids, "a": strength + rng.normal(scale=0.2, size=n), "b": rng.normal(size=n)})
    labels = pd.DataFrame({"image_id": ids, "mean_score": 5 + 2 * strength, "bt_score": 5 + 2 * strength, "duels": 3})
    splits = pd.DataFrame({"image_id": ids, "partition": ["dev"] * 30 + ["test"] * 10, "fold": [k % 3 for k in range(30)] + [-1] * 10})
    rows = []
    for _ in range(300):
        i, j = rng.choice(n, 2, replace=False)
        rows.append({"left_image_id": ids[i], "right_image_id": ids[j], "winner": ids[i] if strength[i] > strength[j] else ids[j], "tie_flag": False, "source": "explicit"})
    pairs = pd.DataFrame(rows)
    summary = train_ranking_suite(table, labels, splits, pairs, {"output_dir": str(tmp_path), "run_name": "rank_smoke", "pair_sources": ["explicit"]})
    model = summary["models"]["pairwise_logreg"]
    assert summary["train_pairs"] > 50 and summary["test_pairs"] > 0
    assert model["metrics"]["test"]["pairwise_accuracy"] > 0.8
    assert model["agreement"]["dev"]["spearman_vs_mean_score"] > 0.8
    assert model["agreement"]["dev"]["spearman_vs_bt_score"] > 0.8
    assert Path(model["rank_scores_path"]).exists()
