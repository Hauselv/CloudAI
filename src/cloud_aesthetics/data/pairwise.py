from __future__ import annotations

from itertools import combinations
from uuid import uuid4

import pandas as pd


def generate_pseudo_pairs(ratings: pd.DataFrame, min_score_gap: float = 2.0) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "left_image_id",
                "right_image_id",
                "rater_id",
                "winner",
                "tie_flag",
                "preference_strength",
                "timestamp",
                "source",
            ]
        )
    rows: list[dict[str, object]] = []
    frame = ratings.copy()
    frame["raw_score_1_to_10"] = frame["raw_score_1_to_10"].astype(float)
    for rater_id, group in frame.groupby("rater_id"):
        records = group.sort_values("rating_timestamp").to_dict("records")
        for left, right in combinations(records, 2):
            gap = abs(float(left["raw_score_1_to_10"]) - float(right["raw_score_1_to_10"]))
            if gap < min_score_gap:
                continue
            winner = left["image_id"] if left["raw_score_1_to_10"] > right["raw_score_1_to_10"] else right["image_id"]
            rows.append(
                {
                    "pair_id": str(uuid4()),
                    "left_image_id": left["image_id"],
                    "right_image_id": right["image_id"],
                    "rater_id": rater_id,
                    "winner": winner,
                    "tie_flag": False,
                    "preference_strength": gap,
                    "timestamp": max(left["rating_timestamp"], right["rating_timestamp"]),
                    "source": "pseudo_from_scores",
                }
            )
    return pd.DataFrame(rows)


def merge_pairwise_tables(explicit_pairs: pd.DataFrame, pseudo_pairs: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [explicit_pairs, pseudo_pairs] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["left_image_id", "right_image_id", "rater_id", "timestamp"])
