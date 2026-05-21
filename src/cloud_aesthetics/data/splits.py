from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold


def _make_score_bins(target: pd.Series, n_bins: int) -> pd.Series:
    unique_count = target.nunique(dropna=True)
    if unique_count < 2:
        return pd.Series(np.zeros(len(target), dtype=int), index=target.index)
    bins = max(2, min(n_bins, int(unique_count)))
    try:
        return pd.qcut(target, q=bins, duplicates="drop", labels=False)
    except ValueError:
        return pd.cut(target, bins=bins, labels=False, include_lowest=True)


def _assign_dev_folds(
    split_table: pd.DataFrame,
    dev_frame: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    dev_groups = dev_frame["split_group_id"].astype(str)
    actual_splits = min(max(1, n_splits), len(dev_frame), dev_groups.nunique())
    if actual_splits < 2:
        split_table.loc[split_table["partition"] == "dev", "fold"] = 0
        return split_table
    try:
        splitter = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(dev_frame, dev_frame["score_bin"], groups=dev_groups))
    except ValueError:
        splitter = GroupKFold(n_splits=actual_splits)
        splits = list(splitter.split(dev_frame, groups=dev_groups))
    for fold, (_, val_idx) in enumerate(splits):
        image_ids = set(dev_frame.iloc[val_idx]["image_id"])
        split_table.loc[split_table["image_id"].isin(image_ids), "fold"] = fold
    return split_table


def create_grouped_splits(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    target_column: str = "mean_score",
    n_splits: int = 5,
    score_bins: int = 5,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    merged = manifest.merge(labels[["image_id", target_column]], on="image_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["image_id", "fold", "partition", "score_bin", "split_group_id"])
    merged["score_bin"] = _make_score_bins(merged[target_column], score_bins)
    groups = merged["split_group_id"].astype(str)
    if len(merged) < 3 or groups.nunique() < 2:
        split_table = merged[["image_id", "split_group_id", "score_bin"]].copy()
        split_table["partition"] = "dev"
        split_table["fold"] = 0
        return split_table.sort_values(["partition", "fold", "image_id"]).reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    try:
        train_idx, test_idx = next(splitter.split(merged, merged["score_bin"], groups=groups))
    except ValueError:
        split_table = merged[["image_id", "split_group_id", "score_bin"]].copy()
        split_table["partition"] = "dev"
        split_table["fold"] = 0
        return split_table.sort_values(["partition", "fold", "image_id"]).reset_index(drop=True)
    split_table = merged[["image_id", "split_group_id", "score_bin"]].copy()
    split_table["partition"] = "dev"
    split_table["fold"] = -1
    split_table.loc[test_idx, "partition"] = "test"
    dev_frame = merged.iloc[train_idx].reset_index(drop=True)
    split_table = _assign_dev_folds(split_table, dev_frame, n_splits, random_state)
    return split_table.sort_values(["partition", "fold", "image_id"]).reset_index(drop=True)
