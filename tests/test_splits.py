from __future__ import annotations

import pandas as pd

from cloud_aesthetics.data.splits import create_grouped_splits


def test_grouped_splits_keep_groups_together():
    manifest = pd.DataFrame(
        [
            {"image_id": f"img{i}", "split_group_id": f"group{i // 2}", "capture_session_id": f"session{i // 2}"}
            for i in range(10)
        ]
    )
    labels = pd.DataFrame({"image_id": [f"img{i}" for i in range(10)], "mean_score": [float((i % 5) + 5) for i in range(10)]})
    splits = create_grouped_splits(manifest, labels, n_splits=3, score_bins=3, test_fraction=0.2, random_state=7)
    assignment = splits.groupby("split_group_id")[["partition", "fold"]].nunique()
    assert assignment["partition"].max() == 1
    assert assignment["fold"].max() == 1


def test_grouped_splits_fall_back_for_tiny_data():
    manifest = pd.DataFrame([{"image_id": "img0", "split_group_id": "group0", "capture_session_id": "session0"}])
    labels = pd.DataFrame([{"image_id": "img0", "mean_score": 7.0}])
    splits = create_grouped_splits(manifest, labels, n_splits=5, score_bins=5, test_fraction=0.2, random_state=7)
    assert splits.to_dict("records") == [
        {
            "image_id": "img0",
            "split_group_id": "group0",
            "score_bin": 0,
            "partition": "dev",
            "fold": 0,
        }
    ]
