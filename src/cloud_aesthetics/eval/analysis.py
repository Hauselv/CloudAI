from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled = np.sqrt(((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b) / (len(group_a) + len(group_b) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled)


def compare_groups(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    high_selector,
    mid_selector,
) -> pd.DataFrame:
    merged = features.merge(labels, on="image_id", how="inner")
    high = merged.loc[high_selector(merged)]
    mid = merged.loc[mid_selector(merged)]
    numeric_columns = [column for column in features.columns if column != "image_id"]
    rows: list[dict[str, float | str]] = []
    for column in numeric_columns:
        high_values = high[column].to_numpy(dtype=np.float32)
        mid_values = mid[column].to_numpy(dtype=np.float32)
        if len(high_values) == 0 or len(mid_values) == 0:
            continue
        statistic, p_value = ttest_ind(high_values, mid_values, equal_var=False)
        rows.append(
            {
                "feature": column,
                "high_mean": float(np.mean(high_values)),
                "mid_mean": float(np.mean(mid_values)),
                "effect_size_d": cohens_d(high_values, mid_values),
                "t_stat": float(statistic),
                "p_value": float(p_value),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "high_mean", "mid_mean", "effect_size_d", "t_stat", "p_value"])
    return pd.DataFrame(rows).sort_values("effect_size_d", ascending=False)


def compute_feature_correlation(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.select_dtypes(include=["number"])
    return numeric.corr()


def compute_pca_projection(features: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    numeric = features.select_dtypes(include=["number"]).fillna(0.0)
    if numeric.empty:
        return pd.DataFrame(columns=["pc1", "pc2"])
    pca = PCA(n_components=min(n_components, numeric.shape[1], len(numeric)))
    projection = pca.fit_transform(numeric)
    columns = [f"pc{index + 1}" for index in range(projection.shape[1])]
    return pd.DataFrame(projection, columns=columns).assign(image_id=features["image_id"].values)
