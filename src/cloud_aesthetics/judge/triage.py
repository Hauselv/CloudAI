"""Turn judge attributes into a labeling queue and exclusion proposals.

Inputs: the attribute tables of one or more model raters (``judge/store.attributes_table``),
the manifest and the human labels. Output: one row per image with the mean judge score,
the spread between judges, raised quality flags, a labeling priority and, where flags
warrant it, a proposed exclusion reason. Nothing here changes labels or exclusions; the
Triage page lets the human accept proposals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cloud_aesthetics.settings import ensure_parent, resolve_path

TRIAGE_PATH = "data/processed/judge/triage.parquet"

# Judge quality flag -> (skip reason, all judges must agree, max mean judge score or None).
# Calibrated on the human labels: only `not_a_cloud_photo` matched the human's own
# exclusions (3/3 rated <= 1). `landscape_dominant` and `airplane_window` images were rated
# 6.9 / 7.2 on average by the human, so they are hints, never proposals.
FLAG_TO_REASON: dict[str, tuple[str, bool, float | None]] = {
    "not_a_cloud_photo": ("not_cloud_training_data", False, None),
    "people_or_objects_dominant": ("not_cloud_training_data", True, 4.0),
    "satellite_or_map": ("satellite", True, 4.0),
}

# Flags worth showing next to an image without proposing anything.
HINT_FLAGS = ("landscape_dominant", "airplane_window", "over_processed", "tiny_detail_crop", "blurry_or_noisy")

# All judges flag "blurry_or_noisy" and the mean judge score is at or below this -> bad_crop
# proposal (9 of 11 such labeled images were rated <= 3 by the human).
LOW_SCORE_EXCLUSION = 3.0


def _split_flags(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [flag for flag in str(value).split("|") if flag]


def build_triage(
    attribute_tables: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    disagreement_threshold: float = 2.0,
    predictions: pd.DataFrame | None = None,
    retest_ids: list[str] | None = None,
) -> pd.DataFrame:
    """One row per manifest image with judge consensus, model predictions, flags, priority and exclusion proposal.

    ``predictions`` (from ``models.predict.predict_all``) adds ``model_score`` / ``model_spread``;
    the priority then rewards expected information: images near the top end, images where
    models and judges disagree, and ``retest_ids`` (already labeled images to rate again
    without seeing the old score).
    """
    frame = manifest.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    keep = [column for column in ("image_id", "relative_path", "kind", "crop_class", "dataset_name", "split_group_id") if column in frame.columns]
    frame = frame[keep].drop_duplicates("image_id")

    score_columns: list[str] = []
    flag_votes: dict[str, dict[str, int]] = {}
    types: dict[str, list[str]] = {}
    lights: dict[str, list[str]] = {}
    verdicts: dict[str, str] = {}
    for rater_id, table in attribute_tables.items():
        if table is None or table.empty:
            continue
        part = table.copy()
        part["image_id"] = part["image_id"].astype(str)
        part = part.drop_duplicates("image_id").set_index("image_id")
        column = f"score_{rater_id}"
        score_columns.append(column)
        frame = frame.merge(part["judge_score"].rename(column), left_on="image_id", right_index=True, how="left")
        for image_id, row in part.iterrows():
            for flag in _split_flags(row.get("judge_flags")):
                flag_votes.setdefault(image_id, {})
                flag_votes[image_id][flag] = flag_votes[image_id].get(flag, 0) + 1
            if isinstance(row.get("judge_cloud_type"), str):
                types.setdefault(image_id, []).append(str(row["judge_cloud_type"]))
            if isinstance(row.get("judge_light"), str):
                lights.setdefault(image_id, []).append(str(row["judge_light"]))
            if isinstance(row.get("judge_verdict"), str) and row["judge_verdict"] and image_id not in verdicts:
                verdicts[image_id] = str(row["judge_verdict"])

    n_judges = len(score_columns)
    if score_columns:
        scores = frame[score_columns]
        frame["judge_mean"] = scores.mean(axis=1)
        frame["judge_spread"] = (scores.max(axis=1) - scores.min(axis=1)).where(scores.notna().sum(axis=1) > 1, 0.0)
        frame["judge_votes"] = scores.notna().sum(axis=1)
    else:
        frame["judge_mean"] = np.nan
        frame["judge_spread"] = np.nan
        frame["judge_votes"] = 0

    def _majority(values: list[str]) -> str:
        return pd.Series(values).mode().iloc[0] if values else ""

    frame["judge_cloud_type"] = frame["image_id"].map(lambda image_id: _majority(types.get(image_id, [])))
    frame["judge_light"] = frame["image_id"].map(lambda image_id: _majority(lights.get(image_id, [])))
    frame["judge_verdict"] = frame["image_id"].map(lambda image_id: verdicts.get(image_id, ""))
    frame["judge_flags"] = frame["image_id"].map(lambda image_id: "|".join(sorted(flag_votes.get(image_id, {}))))
    frame["flag_votes"] = frame["image_id"].map(lambda image_id: sum(flag_votes.get(image_id, {}).values()))

    if labels is not None and not labels.empty:
        human = labels.copy()
        human["image_id"] = human["image_id"].astype(str)
        frame = frame.merge(human[["image_id", "mean_score"]].rename(columns={"mean_score": "human_score"}).drop_duplicates("image_id"), on="image_id", how="left")
    else:
        frame["human_score"] = np.nan
    frame["labeled"] = frame["human_score"].notna()

    proposals: list[str] = []
    proposal_detail: list[str] = []
    hints: list[str] = []
    for _, row in frame.iterrows():
        votes = flag_votes.get(row["image_id"], {})
        mean = row["judge_mean"]
        reason = ""
        detail = ""
        for flag, (mapped, needs_all, max_mean) in FLAG_TO_REASON.items():
            count = votes.get(flag, 0)
            if count == 0:
                continue
            if needs_all and n_judges > 1 and count < n_judges:
                continue
            if max_mean is not None and (pd.isna(mean) or mean > max_mean):
                continue
            reason, detail = mapped, f"{flag} ({count}/{max(n_judges, 1)} judges)"
            break
        if not reason and votes.get("blurry_or_noisy", 0) >= max(n_judges, 1) and pd.notna(mean) and mean <= LOW_SCORE_EXCLUSION:
            reason, detail = "bad_crop", f"blurry_or_noisy by all judges, mean score {mean:.1f}"
        proposals.append(reason)
        proposal_detail.append(detail)
        hints.append("|".join(f"{flag} ({votes[flag]}/{max(n_judges, 1)})" for flag in HINT_FLAGS if votes.get(flag)))
    frame["proposed_exclusion"] = proposals
    frame["proposal_detail"] = proposal_detail
    frame["judge_hints"] = hints

    # Model predictions (optional) and the combined estimate used for the queue.
    frame["model_score"] = np.nan
    frame["model_spread"] = np.nan
    if predictions is not None and not predictions.empty and "pred_mean" in predictions.columns:
        preds = predictions.copy()
        preds["image_id"] = preds["image_id"].astype(str)
        frame = frame.merge(preds[["image_id", "pred_mean", "pred_spread"]].drop_duplicates("image_id"), on="image_id", how="left")
        frame["model_score"] = frame.pop("pred_mean")
        frame["model_spread"] = frame.pop("pred_spread")
    estimates = frame[["judge_mean", "model_score"]]
    frame["estimate"] = estimates.mean(axis=1)
    frame["model_judge_gap"] = (frame["model_score"] - frame["judge_mean"]).abs()

    # Labeling priority: expected information for the 10-vs-8 question.
    # originals first; estimate near the top end; disagreement between judges, between models,
    # and between models and judges; proposals and already labeled images last; retests in between.
    kind_rank = frame["crop_class"].map({"original": 0, "large_crop": 1, "small_crop": 2}).fillna(3) if "crop_class" in frame.columns else pd.Series(3, index=frame.index)
    estimate_fill = frame["estimate"].fillna(frame["judge_mean"]).fillna(-1.0)
    judge_spread = frame["judge_spread"].fillna(0.0)
    model_spread = frame["model_spread"].fillna(0.0)
    gap = frame["model_judge_gap"].fillna(0.0)
    top_end_bonus = np.clip(estimate_fill - 6.0, 0.0, 4.0) * 4.0  # 0 below 6, up to 16 at 10
    disagreement = judge_spread * 2.0 + model_spread * 3.0 + gap * 2.0
    frame["uncertain"] = (judge_spread >= disagreement_threshold) | (gap >= disagreement_threshold) | (model_spread >= 1.5)
    frame["priority_score"] = (-kind_rank * 100.0) + estimate_fill * 3.0 + top_end_bonus + disagreement
    reasons = []
    for _, row in frame.iterrows():
        parts = []
        if pd.notna(row["estimate"]) and row["estimate"] >= 7.5:
            parts.append(f"oberes Ende (Schätzung {row['estimate']:.1f})")
        if pd.notna(row["model_judge_gap"]) and row["model_judge_gap"] >= disagreement_threshold:
            parts.append(f"Modell {row['model_score']:.1f} vs Judges {row['judge_mean']:.1f}")
        elif pd.notna(row["model_spread"]) and row["model_spread"] >= 1.5:
            parts.append(f"Modelle uneinig (±{row['model_spread']:.1f})")
        if judge_spread.loc[_] >= disagreement_threshold:
            parts.append(f"Judges uneinig (±{row['judge_spread']:.1f})")
        reasons.append("; ".join(parts))
    frame["queue_reason"] = reasons
    frame.loc[frame["proposed_exclusion"] != "", "priority_score"] -= 1000.0
    frame.loc[frame["labeled"], "priority_score"] -= 10000.0
    frame["retest"] = False
    if retest_ids:
        mask = frame["image_id"].isin({str(value) for value in retest_ids}) & frame["labeled"]
        frame.loc[mask, "retest"] = True
        # Spread retests through the queue: rank them like a mid-priority unlabeled original.
        frame.loc[mask, "priority_score"] = 20.0 + np.arange(int(mask.sum()))[::-1] * 0.01
        frame.loc[mask, "queue_reason"] = "Wiederholung (Zuverlässigkeit)"
    frame = frame.sort_values("priority_score", ascending=False).reset_index(drop=True)
    frame["label_rank"] = np.arange(1, len(frame) + 1)
    return frame


def save_triage(frame: pd.DataFrame, path_like: str | Path = TRIAGE_PATH) -> Path:
    path = ensure_parent(path_like)
    frame.to_parquet(path, index=False)
    return path


def load_triage(path_like: str | Path = TRIAGE_PATH) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["image_id"] = frame["image_id"].astype(str)
    return frame


def triage_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"images": 0}
    unlabeled = frame[~frame["labeled"]]
    judged = unlabeled[unlabeled["judge_votes"] > 0]
    return {
        "images": int(len(frame)),
        "unlabeled": int(len(unlabeled)),
        "with_model_score": int(frame["model_score"].notna().sum()) if "model_score" in frame.columns else 0,
        "retest_queued": int(frame["retest"].sum()) if "retest" in frame.columns else 0,
        "unlabeled_judged": int(len(judged)),
        "proposed_exclusions": int((frame["proposed_exclusion"] != "").sum()),
        "proposed_exclusions_unlabeled": int((unlabeled["proposed_exclusion"] != "").sum()),
        "uncertain_unlabeled": int(unlabeled["uncertain"].sum()),
        "unlabeled_originals": int((unlabeled.get("crop_class", pd.Series(dtype=str)) == "original").sum()),
        "unlabeled_originals_judge_ge_7": int(((unlabeled.get("crop_class", pd.Series(dtype=str)) == "original") & (unlabeled["judge_mean"] >= 7.0)).sum()),
        "judge_mean_unlabeled": float(judged["judge_mean"].mean()) if not judged.empty else float("nan"),
        "judge_mean_labeled": float(frame.loc[frame["labeled"] & (frame["judge_votes"] > 0), "judge_mean"].mean()) if frame["labeled"].any() else float("nan"),
    }
