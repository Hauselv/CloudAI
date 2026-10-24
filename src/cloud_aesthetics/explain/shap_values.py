"""SHAP attributions for the sklearn pipelines used by the baseline suite.

Tree models (RandomForest, HistGradientBoosting) get exact TreeExplainer values; linear
models get LinearExplainer on the scaled inputs. Both return additive contributions on the
score scale, so a row's contributions plus the base value equal the prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard
    import shap
except ImportError:  # pragma: no cover
    shap = None


def _split_pipeline(model):
    """(preprocessing steps as a callable, final estimator) for a sklearn Pipeline or bare estimator."""
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        final = steps[-1][1]

        def transform(frame: pd.DataFrame) -> np.ndarray:
            values = frame
            for _, step in steps[:-1]:
                values = step.transform(values)
            return np.asarray(values, dtype=np.float64)

        return transform, final
    return (lambda frame: np.asarray(frame, dtype=np.float64)), model


def _explainer(final, background: np.ndarray):
    name = type(final).__name__
    if name in {"RandomForestRegressor", "HistGradientBoostingRegressor", "GradientBoostingRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor"}:
        return shap.TreeExplainer(final), "tree_shap"
    if hasattr(final, "coef_"):
        return shap.LinearExplainer(final, background), "linear_shap"
    return None, ""


def shap_contributions(model, row: pd.DataFrame, background: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    """Per-feature SHAP values for one row. Returns (table, base_value, method)."""
    if shap is None:
        raise ImportError("shap is required for SHAP attributions")
    transform, final = _split_pipeline(model)
    columns = list(row.columns)
    background_values = transform(background)
    explainer, method = _explainer(final, background_values)
    if explainer is None:
        raise ValueError(f"No SHAP explainer for {type(final).__name__}")
    values = np.asarray(explainer.shap_values(transform(row)))
    if values.ndim == 3:
        values = values[0]
    expected = explainer.expected_value
    base = float(np.ravel(expected)[0]) if np.ndim(expected) else float(expected)
    table = pd.DataFrame({"feature": columns, "delta_prediction": values[0].astype(float), "value": np.asarray(row.iloc[0].to_numpy(), dtype=float)})
    table["attribution_kind"] = method
    return table.sort_values("delta_prediction", key=np.abs, ascending=False).reset_index(drop=True), base, method


def global_shap_importance(model, frame: pd.DataFrame, max_rows: int = 400) -> pd.DataFrame:
    """Mean |SHAP| and mean signed SHAP per feature over (a sample of) the training rows."""
    if shap is None:
        raise ImportError("shap is required for SHAP attributions")
    transform, final = _split_pipeline(model)
    sample = frame if len(frame) <= max_rows else frame.sample(max_rows, random_state=0)
    values_matrix = transform(sample)
    explainer, method = _explainer(final, values_matrix)
    if explainer is None:
        raise ValueError(f"No SHAP explainer for {type(final).__name__}")
    values = np.asarray(explainer.shap_values(values_matrix))
    if values.ndim == 3:
        values = values[0]
    # Correlation between feature value and its SHAP contribution tells the direction of the effect.
    directions = []
    for index in range(values.shape[1]):
        column = values_matrix[:, index]
        if np.std(column) > 0 and np.std(values[:, index]) > 0:
            directions.append(float(np.corrcoef(column, values[:, index])[0, 1]))
        else:
            directions.append(0.0)
    table = pd.DataFrame(
        {
            "feature": list(frame.columns),
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "direction": directions,
            "method": method,
            "n": len(sample),
        }
    )
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
