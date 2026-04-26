from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cloud_aesthetics.settings import resolve_path


def load_pickled_model(path_like: str | Path):
    path = resolve_path(path_like)
    with path.open("rb") as handle:
        return pickle.load(handle)


def permutation_feature_importance(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 8) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def approximate_local_contributions(model, feature_row: pd.Series, reference_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in reference_frame.columns if column != "image_id"]
    baseline = reference_frame[numeric_columns].median()
    deltas = []
    base_pred = float(model.predict(pd.DataFrame([baseline]))[0])
    for column in numeric_columns:
        probe = baseline.copy()
        probe[column] = feature_row[column]
        pred = float(model.predict(pd.DataFrame([probe]))[0])
        deltas.append({"feature": column, "delta_prediction": pred - base_pred, "value": float(feature_row[column])})
    return pd.DataFrame(deltas).sort_values("delta_prediction", ascending=False, key=np.abs)
