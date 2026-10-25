"""Test-retest reliability of a rater.

Repeated ratings of the same image by the same person give the noise ceiling: no model
can agree with the rater more closely than the rater agrees with themselves. The queue
sprinkles a small share of already-labeled images back in (without showing the old
rating) so this number keeps improving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def repeat_pairs(ratings: pd.DataFrame, rater_id: str) -> pd.DataFrame:
    """First and latest rating per image for a rater, images rated at least twice."""
    if ratings.empty:
        return pd.DataFrame(columns=["image_id", "first_score", "last_score", "n_ratings", "difference"])
    own = ratings[ratings["rater_id"].astype(str) == rater_id].copy()
    own["image_id"] = own["image_id"].astype(str)
    own["raw_score_1_to_10"] = own["raw_score_1_to_10"].astype(float)
    own = own.sort_values("rating_timestamp")
    grouped = own.groupby("image_id")["raw_score_1_to_10"]
    frame = pd.DataFrame({"first_score": grouped.first(), "last_score": grouped.last(), "n_ratings": grouped.size()})
    frame = frame[frame["n_ratings"] >= 2].reset_index()
    frame["difference"] = frame["last_score"] - frame["first_score"]
    return frame


def reliability_summary(pairs: pd.DataFrame) -> dict[str, float]:
    n = int(len(pairs))
    if n == 0:
        return {"n": 0}
    diff = pairs["difference"].to_numpy(dtype=float)
    out = {
        "n": n,
        "mean_abs_difference": float(np.abs(diff).mean()),
        "within_half": float((np.abs(diff) <= 0.5).mean()),
        "within_one": float((np.abs(diff) <= 1.0).mean()),
        "drift": float(diff.mean()),
        "sd_difference": float(diff.std(ddof=0)),
    }
    if n >= 3:
        out["spearman"] = float(spearmanr(pairs["first_score"], pairs["last_score"]).statistic)
        # Reliability of a single rating (intraclass-style): 1 - var(diff)/(2 var(scores))
        pooled = np.concatenate([pairs["first_score"].to_numpy(dtype=float), pairs["last_score"].to_numpy(dtype=float)])
        var_scores = float(pooled.var(ddof=0))
        out["single_rating_reliability"] = float(max(0.0, 1.0 - (diff.var(ddof=0) / 2.0) / var_scores)) if var_scores > 0 else float("nan")
        # Expected maximum Spearman a perfect model could reach against single ratings
        out["noise_ceiling_spearman"] = float(np.sqrt(max(out["single_rating_reliability"], 0.0))) if out["single_rating_reliability"] == out["single_rating_reliability"] else float("nan")
    return out


def select_retest_candidates(
    ratings: pd.DataFrame,
    rater_id: str,
    *,
    count: int,
    seed: int = 0,
    min_days_since: float = 1.0,
    now: pd.Timestamp | None = None,
) -> list[str]:
    """Images rated exactly once by the rater, stratified over the score range, old enough to be forgotten."""
    if ratings.empty or count <= 0:
        return []
    own = ratings[ratings["rater_id"].astype(str) == rater_id].copy()
    if own.empty:
        return []
    own["image_id"] = own["image_id"].astype(str)
    own["raw_score_1_to_10"] = own["raw_score_1_to_10"].astype(float)
    stamps = pd.to_datetime(own["rating_timestamp"], errors="coerce", utc=True)
    reference = now if now is not None else pd.Timestamp.now(tz="UTC")
    own["age_days"] = (reference - stamps).dt.total_seconds() / 86400.0
    counts = own.groupby("image_id").size()
    once = own[own["image_id"].isin(counts[counts == 1].index) & (own["age_days"] >= min_days_since)]
    if once.empty:
        return []
    once = once.assign(bin=pd.cut(once["raw_score_1_to_10"], bins=[-0.1, 3, 6, 8, 10.1], labels=False))
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    bins = [part for _, part in once.groupby("bin")]
    while len(chosen) < count and any(len(part) for part in bins):
        for index, part in enumerate(bins):
            if len(chosen) >= count or part.empty:
                continue
            pick = part.sample(1, random_state=int(rng.integers(0, 2**31 - 1)))
            chosen.append(str(pick.iloc[0]["image_id"]))
            bins[index] = part.drop(pick.index)
    return chosen
