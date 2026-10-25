from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cloud_aesthetics.eval.reliability import reliability_summary, repeat_pairs, select_retest_candidates
from cloud_aesthetics.judge.triage import build_triage
from cloud_aesthetics.models.baseline import train_baseline_suite
from cloud_aesthetics.models.predict import load_latest_predictions, predict_all, save_predictions


def _ratings() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append({"rating_id": f"r{index}", "image_id": f"i{index}", "rater_id": "me", "rating_session_id": "s", "raw_score_1_to_10": float(index % 11), "rating_timestamp": "2026-01-01T10:00:00+00:00", "note": None})
    rows.append({"rating_id": "x1", "image_id": "i1", "rater_id": "me", "rating_session_id": "s2", "raw_score_1_to_10": 2.0, "rating_timestamp": "2026-02-01T10:00:00+00:00", "note": None})
    rows.append({"rating_id": "x2", "image_id": "i5", "rater_id": "me", "rating_session_id": "s2", "raw_score_1_to_10": 6.0, "rating_timestamp": "2026-02-01T10:00:00+00:00", "note": None})
    rows.append({"rating_id": "x3", "image_id": "i9", "rater_id": "me", "rating_session_id": "s2", "raw_score_1_to_10": 8.0, "rating_timestamp": "2026-02-01T10:00:00+00:00", "note": None})
    rows.append({"rating_id": "o1", "image_id": "i2", "rater_id": "other", "rating_session_id": "s", "raw_score_1_to_10": 5.0, "rating_timestamp": "2026-01-01T10:00:00+00:00", "note": None})
    return pd.DataFrame(rows)


def test_repeat_pairs_and_reliability():
    pairs = repeat_pairs(_ratings(), "me")
    assert set(pairs["image_id"]) == {"i1", "i5", "i9"}
    by_id = pairs.set_index("image_id")
    assert by_id.loc["i1", "first_score"] == 1.0 and by_id.loc["i1", "last_score"] == 2.0 and by_id.loc["i9", "difference"] == -1.0
    summary = reliability_summary(pairs)
    assert summary["n"] == 3 and summary["within_one"] == 1.0 and 0 <= summary["single_rating_reliability"] <= 1
    assert "noise_ceiling_spearman" in summary
    assert reliability_summary(repeat_pairs(_ratings(), "nobody")) == {"n": 0}


def test_select_retest_candidates_is_stratified_and_excludes_repeats():
    ratings = _ratings()
    chosen = select_retest_candidates(ratings, "me", count=4, seed=1, min_days_since=1.0, now=pd.Timestamp("2026-03-01", tz="UTC"))
    assert len(chosen) == 4 and len(set(chosen)) == 4
    assert not {"i1", "i5", "i9"} & set(chosen)  # already repeated
    scores = ratings.set_index("image_id")["raw_score_1_to_10"]
    assert scores[chosen].min() <= 3 and scores[chosen].max() >= 7  # spans the scale
    assert select_retest_candidates(ratings, "me", count=3, now=pd.Timestamp("2026-01-01T12:00", tz="UTC")) == []  # too fresh


def test_predict_all_and_queue_with_predictions_and_retests(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 40
    ids = [f"i{k}" for k in range(n)]
    features = pd.DataFrame({"image_id": ids, "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    scores = np.clip(5 + 2 * features["f1"], 0, 10)
    labels = pd.DataFrame({"image_id": ids[:30], "mean_score": scores[:30]})
    splits = pd.DataFrame({"image_id": ids[:30], "partition": ["dev"] * 24 + ["test"] * 6, "fold": [k % 3 for k in range(24)] + [-1] * 6})
    summary = train_baseline_suite(features, labels, splits, {"output_dir": str(tmp_path / "runs"), "run_name": "pred_smoke", "target_column": "mean_score", "regressors": ["random_forest", "elasticnet"], "shap": False})
    features.to_parquet(tmp_path / "features.parquet", index=False)
    summary["config"] = {"features_path": str(tmp_path / "features.parquet")}
    import json

    (Path(summary["models"]["random_forest"]["model_path"]).parent / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = pd.DataFrame({"image_id": ids, "relative_path": [f"{i}.jpg" for i in ids], "kind": "original", "crop_class": "original", "dataset_name": "toy"})
    predictions = predict_all(Path(summary["models"]["random_forest"]["model_path"]).parent, manifest, labels=labels)
    assert len(predictions) == n and {"pred_random_forest", "pred_elasticnet", "pred_mean", "pred_spread"} <= set(predictions.columns)
    assert predictions["in_training_set"].sum() == 24
    save_predictions(predictions, "pred_smoke", output_dir=tmp_path / "pred")
    assert len(load_latest_predictions(tmp_path / "pred")) == n

    judge = pd.DataFrame({"image_id": ids, "rater_id": "j", "judge_score": np.clip(scores + rng.normal(scale=0.5, size=n), 0, 10), "judge_flags": "", "judge_cloud_type": "cumulus", "judge_light": "daylight", "judge_verdict": "ok"})
    judge.loc[judge["image_id"] == "i35", "judge_score"] = 1.0  # strong disagreement with the model for an unlabeled image
    triage = build_triage({"j": judge}, manifest, labels, predictions=predictions, retest_ids=["i0", "i1"])
    by_id = triage.set_index("image_id")
    assert by_id.loc["i35", "model_judge_gap"] > 2 and "Modell" in by_id.loc["i35", "queue_reason"]
    assert by_id.loc["i0", "retest"] and by_id.loc["i0", "queue_reason"].startswith("Wiederholung")
    unlabeled = by_id.loc[[f"i{k}" for k in range(30, 40)]]
    top_end = unlabeled[unlabeled["estimate"] >= 7.5]
    if not top_end.empty:
        assert top_end["label_rank"].max() < by_id.loc["i0", "label_rank"]  # top-end unlabeled images come before retests
    assert by_id.loc["i0", "label_rank"] < by_id.loc["i2", "label_rank"]  # retests come before plain labeled images
    assert (by_id.loc[~by_id["labeled"] & (by_id["proposed_exclusion"] == ""), "label_rank"] < by_id.loc["i2", "label_rank"]).all()
    assert triage["model_score"].notna().all()
