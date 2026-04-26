from __future__ import annotations

import math

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 10) -> float:
    y_true_int = np.clip(np.rint(y_true), min_rating, max_rating).astype(int)
    y_pred_int = np.clip(np.rint(y_pred), min_rating, max_rating).astype(int)
    n = max_rating - min_rating + 1
    conf_mat = np.zeros((n, n), dtype=np.float64)
    for truth, pred in zip(y_true_int, y_pred_int, strict=False):
        conf_mat[truth - min_rating, pred - min_rating] += 1
    expected = np.outer(conf_mat.sum(axis=1), conf_mat.sum(axis=0)) / max(conf_mat.sum(), 1.0)
    weights = np.zeros_like(conf_mat)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    observed = float((weights * conf_mat).sum() / max(conf_mat.sum(), 1.0))
    expected_score = float((weights * expected).sum() / max(expected.sum(), 1.0))
    if expected_score == 0:
        return 1.0
    return float(1.0 - observed / expected_score)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    spearman = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else math.nan
    pearson = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else math.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
        "spearman": float(spearman) if spearman is not None else math.nan,
        "pearson": float(pearson) if pearson is not None else math.nan,
        "qwk": float(quadratic_weighted_kappa(y_true, y_pred)),
    }


def compute_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    score = np.asarray(y_score) if y_score is not None else y_pred
    auc = roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else math.nan
    tau = kendalltau(y_true, score).statistic if len(y_true) > 1 else math.nan
    return {
        "pairwise_accuracy": float(accuracy_score(y_true, y_pred)),
        "pairwise_auc": float(auc) if auc is not None else math.nan,
        "kendall_tau": float(tau) if tau is not None else math.nan,
    }
