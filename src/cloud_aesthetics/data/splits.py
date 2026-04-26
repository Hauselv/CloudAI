from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def _make_score_bins(target: pd.Series, n_bins: int) -> pd.Series:
    unique_count = target.nunique(dropna=True)
    bins = max(2, min(n_bins, int(unique_count)))
    try:
        return pd.qcut(target, q=bins, duplicates="drop", labels=False)
    except ValueError:
        return pd.cut(target, bins=bins, labels=False, include_lowest=True)


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
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_idx, test_idx = next(splitter.split(merged, merged["score_bin"], groups=groups))
    split_table = merged[["image_id", "split_group_id", "score_bin"]].copy()
    split_table["partition"] = "dev"
    split_table["fold"] = -1
    split_table.loc[test_idx, "partition"] = "test"
    dev_frame = merged.iloc[train_idx].reset_index(drop=True)
    if len(dev_frame) >= n_splits:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (_, val_idx) in enumerate(
            sgkf.split(dev_frame, dev_frame["score_bin"], groups=dev_frame["split_group_id"])
        ):
            image_ids = set(dev_frame.iloc[val_idx]["image_id"])
            split_table.loc[split_table["image_id"].isin(image_ids), "fold"] = fold
    else:
        split_table.loc[split_table["partition"] == "dev", "fold"] = 0
    return split_table.sort_values(["partition", "fold", "image_id"]).reset_index(drop=True)
