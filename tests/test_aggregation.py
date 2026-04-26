from __future__ import annotations

import pandas as pd

from cloud_aesthetics.data.aggregation import aggregate_ratings, compute_pairwise_win_rate


def test_aggregate_ratings_computes_expected_summary_columns():
    ratings = pd.DataFrame(
        [
            {"rating_id": "1", "image_id": "img1", "rater_id": "a", "rating_session_id": "s1", "raw_score_1_to_10": 8, "rating_timestamp": "2024-01-01T00:00:00+00:00", "note": None},
            {"rating_id": "2", "image_id": "img1", "rater_id": "b", "rating_session_id": "s1", "raw_score_1_to_10": 10, "rating_timestamp": "2024-01-01T00:00:01+00:00", "note": None},
            {"rating_id": "3", "image_id": "img2", "rater_id": "a", "rating_session_id": "s1", "raw_score_1_to_10": 6, "rating_timestamp": "2024-01-01T00:00:02+00:00", "note": None},
            {"rating_id": "4", "image_id": "img2", "rater_id": "b", "rating_session_id": "s1", "raw_score_1_to_10": 7, "rating_timestamp": "2024-01-01T00:00:03+00:00", "note": None},
        ]
    )
    pairwise = pd.DataFrame(
        [
            {"pair_id": "p1", "left_image_id": "img1", "right_image_id": "img2", "rater_id": "a", "winner": "img1", "tie_flag": False, "preference_strength": 0.8, "timestamp": "2024-01-01T00:00:04+00:00"}
        ]
    )
    aggregated = aggregate_ratings(ratings, pairwise)
    row = aggregated.loc[aggregated["image_id"] == "img1"].iloc[0]
    assert row["mean_score"] == 9.0
    assert row["n_raters"] == 2
    assert 0.0 <= row["agreement_index"] <= 1.0
    assert row["pairwise_win_rate"] == 1.0


def test_pairwise_win_rate_handles_ties():
    pairwise = pd.DataFrame(
        [
            {"left_image_id": "img1", "right_image_id": "img2", "winner": None, "tie_flag": True},
        ]
    )
    result = compute_pairwise_win_rate(pairwise)
    assert set(result["pairwise_win_rate"]) == {0.5}
