"""Calibration diagnostics for score predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, bin_width: float = 1.0, low: float = 0.0, high: float = 10.0) -> pd.DataFrame:
    """Mean actual score per predicted-score bin, with counts and the identity reference."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    edges = np.arange(low, high + bin_width, bin_width)
    rows = []
    for start, end in zip(edges[:-1], edges[1:], strict=False):
        mask = (y_pred >= start) & ((y_pred < end) if end < edges[-1] else (y_pred <= end))
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{start:g}–{end:g}",
                "bin_center": float((start + end) / 2.0),
                "n": int(mask.sum()),
                "mean_predicted": float(y_pred[mask].mean()),
                "mean_actual": float(y_true[mask].mean()),
                "std_actual": float(y_true[mask].std(ddof=0)),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["gap"] = frame["mean_actual"] - frame["mean_predicted"]
    return frame


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray, half_width: float | None = None) -> dict[str, float]:
    """Residual statistics plus the empirical coverage of a symmetric interval."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    out = {
        "n": int(len(residual)),
        "bias": float(residual.mean()) if len(residual) else float("nan"),
        "mae": float(np.abs(residual).mean()) if len(residual) else float("nan"),
        "p90_abs_residual": float(np.quantile(np.abs(residual), 0.9)) if len(residual) else float("nan"),
        "spread_ratio": float(y_pred.std(ddof=0) / y_true.std(ddof=0)) if len(residual) > 1 and y_true.std(ddof=0) > 0 else float("nan"),
        "ece": expected_calibration_error(y_true, y_pred) if len(residual) > 1 else float("nan"),
    }
    if half_width is not None and len(residual):
        out["coverage"] = interval_coverage(y_true, y_pred - half_width, y_pred + half_width)
        out["half_width"] = float(half_width)
    return out
