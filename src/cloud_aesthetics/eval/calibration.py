from __future__ import annotations

import numpy as np


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1) if len(y_pred) else np.array([0.0, 1.0])
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:], strict=False):
        mask = (y_pred >= start) & (y_pred < end if end < bins[-1] else y_pred <= end)
        if mask.sum() == 0:
            continue
        ece += abs(y_pred[mask].mean() - y_true[mask].mean()) * (mask.sum() / len(y_true))
    return float(ece)
