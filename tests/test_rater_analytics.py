from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from cloud_aesthetics.eval.rater_analytics import analyze_rater, anchor_summary, disputed_images, rater_agreement_matrix, sessionize


def _ratings(n: int = 120, seed: int = 0, drift: float = 0.0, anchor: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1, 9, size=n)
    rows = []
    start = pd.Timestamp("2026-04-26T10:00:00Z")
    previous = 5.0
    for k in range(n):
        session = k // 40
        timestamp = start + pd.Timedelta(hours=5 * session) + pd.Timedelta(seconds=20 * (k % 40))
        position = k % 40
        score = truth[k] + drift * position / 40 + anchor * (previous - 5.0) + rng.normal(scale=0.3)
        score = float(np.clip(np.round(score * 2) / 2, 0, 10))
        rows.append({"rating_id": f"r{k}", "image_id": f"img{k}", "rater_id": "friend_a", "rating_session_id": "s", "raw_score_1_to_10": score, "rating_timestamp": timestamp.isoformat(), "note": None})
        previous = score
        for judge in ("llm_a", "llm_b"):
            rows.append({"rating_id": f"{judge}{k}", "image_id": f"img{k}", "rater_id": judge, "rating_session_id": "j", "raw_score_1_to_10": float(np.clip(truth[k] + rng.normal(scale=0.4), 0, 10)), "rating_timestamp": timestamp.isoformat(), "note": None})
    return pd.DataFrame(rows)


def test_sessionize_splits_on_gaps():
    sessions = sessionize(_ratings(), "friend_a", gap_minutes=30)
    assert sessions["session"].nunique() == 3 and sessions["position"].max() == 39
    assert sessions.loc[sessions["position"] == 0, "seconds_since_previous"].isna().all()


def test_drift_and_anchor_are_detected_only_when_planted():
    calm = analyze_rater(_ratings(), "friend_a", reference_raters=["llm_a", "llm_b"])
    assert calm["reference_available"] and calm["drift"]["n_sessions"] == 3
    assert abs(calm["drift"]["within_session_rho"]) < 0.25
    assert abs(calm["anchor"]["assimilation_r"]) < 0.25
    drifting = analyze_rater(_ratings(drift=2.5), "friend_a", reference_raters=["llm_a", "llm_b"])
    assert drifting["drift"]["within_session_rho"] > 0.5 and drifting["drift"]["residual_change_per_50"] > 1.5
    anchored = analyze_rater(_ratings(anchor=0.5), "friend_a", reference_raters=["llm_a", "llm_b"])
    assert anchored["anchor"]["previous_score_r"] > 0.3 and anchored["anchor"]["n_pairs"] > 100
    assert anchored["anchor"]["after_high_minus_after_low"] > 0.5
    assert len(anchored["sessions"]) == 3 and set(anchored["usage"].columns) == {"score", "n", "share"}


def test_disputes_and_agreement_matrix():
    ratings = _ratings(n=40)
    disputes = disputed_images(ratings, human_raters=["friend_a"], reference_raters=["llm_a", "llm_b"])
    assert "human_vs_reference" in disputes.columns and disputes["dispute"].iloc[0] >= disputes["dispute"].iloc[-1]
    matrix = rater_agreement_matrix(ratings, min_common=10)
    assert matrix.loc["llm_a", "llm_b"] > 0.9 and matrix.loc["friend_a", "friend_a"] == 1.0
    empty = anchor_summary(sessionize(ratings.iloc[0:0], "friend_a"))
    assert empty["n_pairs"] == 0
