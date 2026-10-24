from __future__ import annotations

import numpy as np

from cloud_aesthetics.eval.metrics import compute_regression_metrics, quadratic_weighted_kappa


def test_qwk_keeps_zero_scores_distinct_from_one():
    y_true = np.array([0.0, 0.0, 1.0, 1.0, 10.0, 10.0])
    perfect = quadratic_weighted_kappa(y_true, y_true)
    confused = quadratic_weighted_kappa(y_true, np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0]))
    assert perfect == 1.0
    assert confused < perfect


def test_regression_metrics_contain_expected_keys():
    metrics = compute_regression_metrics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 2.0, 2.5, 4.5]))
    assert set(metrics) == {"mae", "rmse", "r2", "spearman", "pearson", "qwk"}
    assert metrics["mae"] > 0
