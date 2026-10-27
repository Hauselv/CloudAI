"""Rater analytics: how a person rates, not what they rate.

Sessions are reconstructed from timestamps (a gap longer than ``gap_minutes`` starts a new
one). Because every image has a different "true" quality, the behavioural questions are
asked on *residuals*: the rating minus a per-image reference (by default the mean of the
LLM judges, which exists for every image and is independent of the human's mood). On
residuals we can ask whether the rater drifts within a session, whether the previous
image pulls the next rating along (assimilation) or pushes it away (contrast), and which
images the raters disagree about most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCORE_COLUMN = "raw_score_1_to_10"


def _prepared(ratings: pd.DataFrame, rater_id: str | None = None) -> pd.DataFrame:
    frame = ratings.copy()
    if rater_id is not None:
        frame = frame[frame["rater_id"].astype(str) == str(rater_id)]
    frame["image_id"] = frame["image_id"].astype(str)
    frame["score"] = frame[SCORE_COLUMN].astype(float)
    frame["timestamp"] = pd.to_datetime(frame["rating_timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def sessionize(ratings: pd.DataFrame, rater_id: str, *, gap_minutes: float = 30.0) -> pd.DataFrame:
    """One row per rating of ``rater_id`` with session index, position and minutes since session start."""
    frame = _prepared(ratings, rater_id)
    if frame.empty:
        return frame.assign(session=pd.Series(dtype=int), position=pd.Series(dtype=int), minutes=pd.Series(dtype=float))
    gaps = frame["timestamp"].diff().dt.total_seconds().div(60.0).fillna(np.inf)
    frame["session"] = (gaps > gap_minutes).cumsum().astype(int) - 1
    frame["position"] = frame.groupby("session").cumcount()
    starts = frame.groupby("session")["timestamp"].transform("min")
    frame["minutes"] = (frame["timestamp"] - starts).dt.total_seconds() / 60.0
    frame["seconds_since_previous"] = frame["timestamp"].diff().dt.total_seconds()
    frame.loc[frame["position"] == 0, "seconds_since_previous"] = np.nan
    return frame


def reference_scores(ratings: pd.DataFrame, reference_raters: list[str]) -> pd.Series:
    """Per-image mean score of the reference raters (typically the LLM judges)."""
    frame = ratings[ratings["rater_id"].astype(str).isin([str(r) for r in reference_raters])].copy()
    if frame.empty:
        return pd.Series(dtype=float, name="reference")
    frame["image_id"] = frame["image_id"].astype(str)
    return frame.groupby("image_id")[SCORE_COLUMN].mean().astype(float).rename("reference")


def attach_residuals(sessions: pd.DataFrame, reference: pd.Series | None) -> pd.DataFrame:
    """Residual = own score minus the (centred) reference; without a reference, minus the own mean."""
    frame = sessions.copy()
    if frame.empty:
        frame["residual"] = pd.Series(dtype=float)
        return frame
    if reference is None or reference.empty:
        frame["reference"] = np.nan
        frame["residual"] = frame["score"] - frame["score"].mean()
        return frame
    frame["reference"] = frame["image_id"].map(reference)
    ok = frame["reference"].notna()
    if ok.sum() >= 3:
        # Linear calibration of the reference onto the rater's scale so the residual has no slope by construction.
        slope, intercept = np.polyfit(frame.loc[ok, "reference"], frame.loc[ok, "score"], 1)
        frame["expected"] = intercept + slope * frame["reference"]
    else:
        frame["expected"] = frame["score"].mean()
    frame["residual"] = frame["score"] - frame["expected"].fillna(frame["score"].mean())
    return frame


def session_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per session: size, duration, mean score, mean residual and the within-session trend."""
    rows = []
    for session, part in frame.groupby("session"):
        trend_rho, trend_p = (np.nan, np.nan)
        if len(part) >= 8 and part["residual"].nunique() > 1:
            trend_rho, trend_p = spearmanr(part["position"], part["residual"])
        rows.append(
            {
                "session": int(session),
                "start": part["timestamp"].min(),
                "n": int(len(part)),
                "minutes": float(part["minutes"].max()),
                "seconds_per_image": float(part["seconds_since_previous"].median()) if part["seconds_since_previous"].notna().any() else np.nan,
                "mean_score": float(part["score"].mean()),
                "sd_score": float(part["score"].std(ddof=0)),
                "mean_residual": float(part["residual"].mean()),
                "trend_rho": float(trend_rho) if trend_rho == trend_rho else np.nan,
                "trend_p": float(trend_p) if trend_p == trend_p else np.nan,
                "share_half_steps": float((np.mod(part["score"] * 2, 2) == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def drift_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Does the rater get stricter or milder as a session goes on, and from session to session?"""
    out: dict[str, float] = {"n": int(len(frame)), "n_sessions": int(frame["session"].nunique()) if not frame.empty else 0}
    if len(frame) < 10:
        return out
    rho, p = spearmanr(frame["position"], frame["residual"]) if frame["residual"].nunique() > 1 else (np.nan, np.nan)
    out["within_session_rho"] = float(rho)
    out["within_session_p"] = float(p)
    # Residual change per 50 images within a session (pooled OLS slope), in score points.
    slope = np.polyfit(frame["position"], frame["residual"], 1)[0] if frame["position"].nunique() > 1 else 0.0
    out["residual_change_per_50"] = float(slope * 50)
    by_session = frame.groupby("session")["residual"].mean()
    out["between_session_sd"] = float(by_session.std(ddof=0)) if len(by_session) > 1 else 0.0
    first = frame[frame["position"] < 20]["residual"].mean()
    late = frame[frame["position"] >= 20]["residual"].mean() if (frame["position"] >= 20).any() else np.nan
    out["first20_minus_later"] = float(first - late) if late == late else np.nan
    return out


def anchor_summary(frame: pd.DataFrame, *, max_gap_seconds: float = 600.0, groups: pd.Series | None = None) -> dict[str, float]:
    """Order effects: does the previous image's rating pull the current one (assimilation) or push it (contrast)?

    Uses consecutive ratings within a session no more than ``max_gap_seconds`` apart.
    ``assimilation_r`` correlates the current residual with the previous *residual*;
    ``previous_score_r`` with the previous raw score (what the person last typed).
    ``groups`` (image_id -> split group) drops pairs of images from the same source photo,
    whose residuals are correlated for a content reason rather than an order reason.
    """
    out: dict[str, float] = {"n_pairs": 0, "n_same_group_excluded": 0}
    if frame.empty:
        return out
    frame = frame.sort_values(["session", "position"]).reset_index(drop=True)
    same_session = frame["session"].eq(frame["session"].shift(1))
    close = frame["seconds_since_previous"].le(max_gap_seconds)
    mask = same_session & close & frame["residual"].notna() & frame["residual"].shift(1).notna()
    if groups is not None and not groups.empty:
        group = frame["image_id"].map(groups)
        same_group = group.notna() & group.eq(group.shift(1))
        out["n_same_group_excluded"] = int((mask & same_group).sum())
        mask = mask & ~same_group
    current = frame.loc[mask, "residual"].to_numpy(dtype=float)
    previous_residual = frame["residual"].shift(1)[mask].to_numpy(dtype=float)
    previous_score = frame["score"].shift(1)[mask].to_numpy(dtype=float)
    out["n_pairs"] = int(mask.sum())
    if mask.sum() < 10 or np.std(current) == 0 or np.std(previous_residual) == 0:
        return out
    r1, p1 = pearsonr(previous_residual, current)
    r2, p2 = pearsonr(previous_score, current)
    out.update({"assimilation_r": float(r1), "assimilation_p": float(p1), "previous_score_r": float(r2), "previous_score_p": float(p2)})
    # Effect in score points: expected residual after a previous 9+ vs after a previous <=4.
    high = current[previous_score >= 9.0]
    low = current[previous_score <= 4.0]
    if len(high) >= 5 and len(low) >= 5:
        out["after_high_minus_after_low"] = float(high.mean() - low.mean())
        out["n_after_high"] = int(len(high))
        out["n_after_low"] = int(len(low))
    return out


def score_usage(frame: pd.DataFrame) -> pd.DataFrame:
    """How often each score value is used (0–10 in half steps)."""
    if frame.empty:
        return pd.DataFrame(columns=["score", "n", "share"])
    counts = frame["score"].round(1).value_counts().sort_index()
    return pd.DataFrame({"score": counts.index.astype(float), "n": counts.to_numpy(), "share": (counts / counts.sum()).to_numpy()})


def disputed_images(ratings: pd.DataFrame, *, human_raters: list[str], reference_raters: list[str], min_raters: int = 2) -> pd.DataFrame:
    """Images with the largest disagreement: between humans (if several) and human vs reference."""
    frame = ratings.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["rater_id"] = frame["rater_id"].astype(str)
    frame["score"] = frame[SCORE_COLUMN].astype(float)
    latest = frame.sort_values("rating_timestamp").groupby(["image_id", "rater_id"])["score"].last().unstack()
    humans = [r for r in human_raters if r in latest.columns]
    references = [r for r in reference_raters if r in latest.columns]
    out = pd.DataFrame(index=latest.index)
    if humans:
        out["human_mean"] = latest[humans].mean(axis=1)
        out["human_n"] = latest[humans].notna().sum(axis=1)
        out["human_spread"] = latest[humans].max(axis=1) - latest[humans].min(axis=1)
    if references:
        out["reference_mean"] = latest[references].mean(axis=1)
    if humans and references:
        out["human_vs_reference"] = out["human_mean"] - out["reference_mean"]
    out = out.reset_index()
    if "human_spread" in out.columns and (out["human_n"] >= min_raters).any():
        out["dispute"] = out["human_spread"].where(out["human_n"] >= min_raters, out.get("human_vs_reference", pd.Series(0.0, index=out.index)).abs())
    elif "human_vs_reference" in out.columns:
        out["dispute"] = out["human_vs_reference"].abs()
    else:
        out["dispute"] = 0.0
    out = out[out["dispute"].notna()]
    return out.sort_values("dispute", ascending=False).reset_index(drop=True)


def rater_agreement_matrix(ratings: pd.DataFrame, *, min_common: int = 20) -> pd.DataFrame:
    """Spearman agreement between every pair of raters on their common images."""
    frame = ratings.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["rater_id"] = frame["rater_id"].astype(str)
    frame["score"] = frame[SCORE_COLUMN].astype(float)
    latest = frame.sort_values("rating_timestamp").groupby(["image_id", "rater_id"])["score"].last().unstack()
    raters = list(latest.columns)
    matrix = pd.DataFrame(np.nan, index=raters, columns=raters, dtype=float)
    for a in raters:
        for b in raters:
            both = latest[[a, b]].dropna()
            if a == b:
                matrix.loc[a, b] = 1.0
            elif len(both) >= min_common and both[a].nunique() > 1 and both[b].nunique() > 1:
                matrix.loc[a, b] = float(spearmanr(both[a], both[b]).correlation)
    return matrix


def analyze_rater(
    ratings: pd.DataFrame,
    rater_id: str,
    *,
    reference_raters: list[str] | None = None,
    reference: pd.Series | None = None,
    gap_minutes: float = 30.0,
    max_gap_seconds: float = 600.0,
    groups: pd.Series | None = None,
) -> dict[str, object]:
    """Everything the page needs for one rater. ``reference`` overrides the reference raters."""
    sessions = sessionize(ratings, rater_id, gap_minutes=gap_minutes)
    if reference is None:
        reference = reference_scores(ratings, reference_raters or [])
    frame = attach_residuals(sessions, reference)
    return {
        "frame": frame,
        "sessions": session_table(frame) if not frame.empty else pd.DataFrame(),
        "drift": drift_summary(frame),
        "anchor": anchor_summary(frame, max_gap_seconds=max_gap_seconds, groups=groups),
        "usage": score_usage(frame),
        "reference_available": bool(reference is not None and not reference.empty and frame.get("reference", pd.Series(dtype=float)).notna().any()),
    }


def model_reference(predictions: pd.DataFrame, model_name: str | None = None) -> pd.Series:
    """Per-image out-of-fold prediction from a run's predictions.parquet (test rows use the fitted prediction)."""
    if predictions is None or predictions.empty:
        return pd.Series(dtype=float, name="reference")
    frame = predictions.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    oof_columns = [column for column in frame.columns if column.startswith("oof_")]
    if not oof_columns:
        return pd.Series(dtype=float, name="reference")
    name = model_name or oof_columns[0].removeprefix("oof_")
    oof = frame.get(f"oof_{name}")
    pred = frame.get(f"pred_{name}")
    values = oof if oof is not None else pd.Series(np.nan, index=frame.index)
    if pred is not None:
        values = values.fillna(pred)
    return pd.Series(values.to_numpy(dtype=float), index=frame["image_id"], name="reference").dropna()
