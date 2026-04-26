from __future__ import annotations

import numpy as np


def summarize_ensemble_predictions(prediction_matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(prediction_matrix, dtype=np.float32)
    return {
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0),
        "lower": np.percentile(matrix, 10, axis=0),
        "upper": np.percentile(matrix, 90, axis=0),
    }


def conformal_interval(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> float:
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    if residuals.size == 0:
        return 0.0
    return float(np.quantile(residuals, 1.0 - alpha))
