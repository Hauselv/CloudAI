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

# Judge quality flag -> skip reason used by the Rate page / exclusions.csv. Flags that need
# both judges to agree are marked; the rest propose on a single vote.
FLAG_TO_REASON: dict[str, tuple[str, bool]] = {
    "not_a_cloud_photo": ("not_cloud_training_data", False),
    "satellite_or_map": ("satellite", False),
    "airplane_window": ("airplane_window", True),
    "landscape_dominant": ("too_much_landscape", True),
    "people_or_objects_dominant": ("not_cloud_training_data", True),
}

LOW_SCORE_EXCLUSION = 1.0  # all judges at or below this score and a "blurry_or_noisy" flag -> bad_crop proposal


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
) -> pd.DataFrame:
    """One row per manifest image with judge consensus, flags, priority and exclusion proposal."""
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
    for _, row in frame.iterrows():
        votes = flag_votes.get(row["image_id"], {})
        reason = ""
        detail = ""
        for flag, (mapped, needs_all) in FLAG_TO_REASON.items():
            count = votes.get(flag, 0)
            if count == 0:
                continue
            if needs_all and n_judges > 1 and count < n_judges:
                continue
            reason, detail = mapped, f"{flag} ({count}/{max(n_judges, 1)} judges)"
            break
        if not reason and votes.get("blurry_or_noisy", 0) >= max(n_judges, 1) and pd.notna(row["judge_mean"]) and row["judge_mean"] <= LOW_SCORE_EXCLUSION:
            reason, detail = "bad_crop", f"blurry_or_noisy by all judges, mean score {row['judge_mean']:.1f}"
        proposals.append(reason)
        proposal_detail.append(detail)
    frame["proposed_exclusion"] = proposals
    frame["proposal_detail"] = proposal_detail

    # Labeling priority for the 10-vs-8 question: unlabeled originals first, high judge scores
    # first, judge disagreement as a tie-breaker (those images carry the most information).
    kind_rank = frame["crop_class"].map({"original": 0, "large_crop": 1, "small_crop": 2}).fillna(3) if "crop_class" in frame.columns else pd.Series(3, index=frame.index)
    mean_fill = frame["judge_mean"].fillna(-1.0)
    spread_fill = frame["judge_spread"].fillna(0.0)
    frame["uncertain"] = spread_fill >= disagreement_threshold
    frame["priority_score"] = (-kind_rank * 100.0) + mean_fill * 5.0 + spread_fill * 2.0
    frame.loc[frame["proposed_exclusion"] != "", "priority_score"] -= 1000.0
    frame.loc[frame["labeled"], "priority_score"] -= 10000.0
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
        "unlabeled_judged": int(len(judged)),
        "proposed_exclusions": int((frame["proposed_exclusion"] != "").sum()),
        "proposed_exclusions_unlabeled": int((unlabeled["proposed_exclusion"] != "").sum()),
        "uncertain_unlabeled": int(unlabeled["uncertain"].sum()),
        "unlabeled_originals": int((unlabeled.get("crop_class", pd.Series(dtype=str)) == "original").sum()),
        "unlabeled_originals_judge_ge_7": int(((unlabeled.get("crop_class", pd.Series(dtype=str)) == "original") & (unlabeled["judge_mean"] >= 7.0)).sum()),
        "judge_mean_unlabeled": float(judged["judge_mean"].mean()) if not judged.empty else float("nan"),
        "judge_mean_labeled": float(frame.loc[frame["labeled"] & (frame["judge_votes"] > 0), "judge_mean"].mean()) if frame["labeled"].any() else float("nan"),
    }
