from __future__ import annotations

from itertools import combinations
from uuid import uuid4

import numpy as np
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


def bradley_terry_scores(
    pairs: pd.DataFrame,
    *,
    iterations: int = 200,
    prior: float = 0.5,
    scale: float = 10.0,
) -> pd.DataFrame:
    """Bradley-Terry strengths from pairwise outcomes (ties count half for both sides).

    The MM algorithm with a small symmetric prior (``prior`` pseudo-wins against an
    average opponent) keeps undefeated images finite. Strengths are returned as
    log-odds (``bt_logit``) and rescaled to 0-``scale`` (``bt_score``) for display.
    """
    columns = ["image_id", "bt_logit", "bt_score", "duels", "wins"]
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=columns)
    frame = pairs.dropna(subset=["left_image_id", "right_image_id"]).copy()
    frame["left_image_id"] = frame["left_image_id"].astype(str)
    frame["right_image_id"] = frame["right_image_id"].astype(str)
    ids = sorted(set(frame["left_image_id"]) | set(frame["right_image_id"]))
    index = {image_id: position for position, image_id in enumerate(ids)}
    n = len(ids)
    wins = np.zeros((n, n), dtype=float)  # wins[i, j] = times i beat j
    for _, row in frame.iterrows():
        i, j = index[row["left_image_id"]], index[row["right_image_id"]]
        if bool(row.get("tie_flag", False)) or row.get("winner") in (None, "", float("nan")) or (isinstance(row.get("winner"), float) and np.isnan(row.get("winner"))):
            wins[i, j] += 0.5
            wins[j, i] += 0.5
        elif str(row["winner"]) == row["left_image_id"]:
            wins[i, j] += 1.0
        elif str(row["winner"]) == row["right_image_id"]:
            wins[j, i] += 1.0
    games = wins + wins.T
    strength = np.ones(n)
    total_wins = wins.sum(axis=1) + prior
    for _ in range(iterations):
        denominator = np.zeros(n)
        for i in range(n):
            mask = games[i] > 0
            denominator[i] = (games[i, mask] / (strength[i] + strength[mask])).sum()
        # prior: `prior` pseudo-games against an opponent of strength 1
        denominator += 2 * prior / (strength + 1.0)
        updated = total_wins / np.maximum(denominator, 1e-12)
        updated /= np.exp(np.log(updated).mean())  # fix the scale (geometric mean 1)
        if np.max(np.abs(np.log(updated) - np.log(strength))) < 1e-8:
            strength = updated
            break
        strength = updated
    logit = np.log(strength)
    if n > 1 and logit.std() > 0:
        z = (logit - logit.mean()) / logit.std()
        score = np.clip(scale / 2.0 + z * (scale / 6.0), 0.0, scale)
    else:
        score = np.full(n, scale / 2.0)
    return pd.DataFrame(
        {
            "image_id": ids,
            "bt_logit": logit,
            "bt_score": score,
            "duels": games.sum(axis=1).astype(int),
            "wins": wins.sum(axis=1),
        }
    ).sort_values("bt_logit", ascending=False).reset_index(drop=True)


def select_duel_pair(
    pool: pd.DataFrame,
    existing_pairs: pd.DataFrame,
    *,
    seed: int | None = None,
    max_score_gap: float = 1.5,
) -> tuple[str, str] | None:
    """Pick the next pair for the duel mode.

    ``pool`` has ``image_id`` and an optional ``score`` column (human mean or judge
    estimate). Pairs are drawn among images with the fewest duels so far, preferring
    opponents within ``max_score_gap`` and never repeating a pair.
    """
    if pool is None or len(pool) < 2:
        return None
    rng = np.random.default_rng(seed)
    ids = pool["image_id"].astype(str).tolist()
    scores = dict(zip(ids, pool["score"].astype(float).tolist(), strict=True)) if "score" in pool.columns else {}
    seen: set[frozenset[str]] = set()
    duels: dict[str, int] = {image_id: 0 for image_id in ids}
    if existing_pairs is not None and not existing_pairs.empty:
        for _, row in existing_pairs.iterrows():
            left, right = str(row["left_image_id"]), str(row["right_image_id"])
            seen.add(frozenset((left, right)))
            for image_id in (left, right):
                if image_id in duels:
                    duels[image_id] += 1
    order = sorted(ids, key=lambda image_id: (duels[image_id], rng.random()))
    for anchor in order:
        candidates = [other for other in ids if other != anchor and frozenset((anchor, other)) not in seen]
        if not candidates:
            continue
        if scores:
            close = [other for other in candidates if abs(scores.get(other, 0.0) - scores.get(anchor, 0.0)) <= max_score_gap]
            candidates = close or candidates
        candidates.sort(key=lambda image_id: (duels[image_id], rng.random()))
        return anchor, candidates[0]
    return None


def merge_pairwise_tables(explicit_pairs: pd.DataFrame, pseudo_pairs: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [explicit_pairs, pseudo_pairs] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["left_image_id", "right_image_id", "rater_id", "timestamp"])
