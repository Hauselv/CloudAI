from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from cloud_aesthetics.data.schemas import empty_table, require_columns


def add_rater_normalized_scores(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return ratings.copy()
    require_columns(ratings, ["rater_id", "raw_score_1_to_10"], "ratings")
    frame = ratings.copy()
    frame["raw_score_1_to_10"] = frame["raw_score_1_to_10"].astype(float)
    grouped = frame.groupby("rater_id")["raw_score_1_to_10"]
    means = grouped.transform("mean")
    stds = grouped.transform(lambda series: float(series.std(ddof=0)) or 1.0)
    frame["rater_z_score"] = (frame["raw_score_1_to_10"] - means) / stds.replace(0.0, 1.0)
    z_mean = frame["rater_z_score"].mean()
    z_std = float(frame["rater_z_score"].std(ddof=0)) or 1.0
    scaled = (frame["rater_z_score"] - z_mean) / z_std
    frame["normalized_score_1_to_10"] = np.clip(5.5 + 1.5 * scaled, 1.0, 10.0)
    return frame


def compute_pairwise_win_rate(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame(columns=["image_id", "pairwise_win_rate"])
    rows: list[dict[str, object]] = []
    for _, row in pairwise.iterrows():
        if bool(row.get("tie_flag", False)):
            left_score = 0.5
            right_score = 0.5
        else:
            winner = row.get("winner")
            left_score = 1.0 if winner == row["left_image_id"] else 0.0
            right_score = 1.0 if winner == row["right_image_id"] else 0.0
        rows.append({"image_id": row["left_image_id"], "result": left_score})
        rows.append({"image_id": row["right_image_id"], "result": right_score})
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("image_id", as_index=False)["result"]
        .mean()
        .rename(columns={"result": "pairwise_win_rate"})
    )


def aggregate_ratings(ratings: pd.DataFrame, pairwise: pd.DataFrame | None = None) -> pd.DataFrame:
    if ratings.empty:
        return empty_table("aggregated")
    frame = add_rater_normalized_scores(ratings)
    grouped = frame.groupby("image_id")
    rows: list[dict[str, object]] = []
    for image_id, group in grouped:
        scores = group["raw_score_1_to_10"].astype(float).to_numpy()
        normalized_scores = group["normalized_score_1_to_10"].astype(float).to_numpy()
        n_raters = len(scores)
        std_score = float(np.std(scores, ddof=0))
        sem_score = float(std_score / math.sqrt(n_raters)) if n_raters else 0.0
        agreement_index = float(np.clip(1.0 - (std_score / 4.5), 0.0, 1.0))
        rows.append(
            {
                "image_id": image_id,
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "trimmed_mean_score": float(trim_mean(scores, 0.1)),
                "std_score": std_score,
                "sem_score": sem_score,
                "n_raters": int(n_raters),
                "agreement_index": agreement_index,
                "normalized_mean_score": float(np.mean(normalized_scores)),
                "pairwise_win_rate": 0.5,
            }
        )
    aggregated = pd.DataFrame(rows)
    if pairwise is not None and not pairwise.empty:
        win_rates = compute_pairwise_win_rate(pairwise)
        aggregated = aggregated.merge(win_rates, on="image_id", how="left", suffixes=("", "_pair"))
        aggregated["pairwise_win_rate"] = aggregated["pairwise_win_rate_pair"].fillna(
            aggregated["pairwise_win_rate"]
        )
        aggregated = aggregated.drop(columns=["pairwise_win_rate_pair"])
    return aggregated.sort_values("image_id").reset_index(drop=True)
