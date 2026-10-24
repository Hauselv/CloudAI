from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from cloud_aesthetics.data.schemas import SCHEMA_COLUMNS, empty_table
from cloud_aesthetics.settings import ensure_parent, resolve_path
from cloud_aesthetics.utils.io import append_csv_row


def record_rating(
    ratings_dir: str | Path,
    image_id: str,
    rater_id: str,
    score: float,
    rating_session_id: str,
    note: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    event_time = timestamp or datetime.now(timezone.utc)
    row = {
        "rating_id": str(uuid4()),
        "image_id": image_id,
        "rater_id": rater_id,
        "rating_session_id": rating_session_id,
        "raw_score_1_to_10": float(score),
        "rating_timestamp": event_time.isoformat(),
        "note": note,
    }
    target = resolve_path(ratings_dir) / f"{rater_id}.csv"
    append_csv_row(row, target)
    return row


def record_pairwise_preference(
    pairwise_dir: str | Path,
    left_image_id: str,
    right_image_id: str,
    rater_id: str,
    winner: str | None,
    tie_flag: bool,
    preference_strength: float | None = None,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    event_time = timestamp or datetime.now(timezone.utc)
    row = {
        "pair_id": str(uuid4()),
        "left_image_id": left_image_id,
        "right_image_id": right_image_id,
        "rater_id": rater_id,
        "winner": winner,
        "tie_flag": bool(tie_flag),
        "preference_strength": preference_strength,
        "timestamp": event_time.isoformat(),
    }
    target = resolve_path(pairwise_dir) / f"{rater_id}.csv"
    append_csv_row(row, target)
    return row


VOID_FILENAME = "ratings_void.csv"


def void_rating(ratings_dir: str | Path, rating_id: str, reason: str = "undo") -> dict[str, object]:
    """Mark a rating as withdrawn without rewriting the append-only rater files."""
    row = {"rating_id": str(rating_id), "voided_at": datetime.now(timezone.utc).isoformat(), "reason": reason}
    append_csv_row(row, resolve_path(ratings_dir) / VOID_FILENAME)
    return row


def voided_rating_ids(ratings_dir: str | Path) -> set[str]:
    path = resolve_path(ratings_dir) / VOID_FILENAME
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return set(frame["rating_id"].astype(str)) if "rating_id" in frame.columns else set()


def _load_annotation_dir(path_like: str | Path, table_name: str) -> pd.DataFrame:
    root = resolve_path(path_like)
    if not root.exists():
        ensure_parent(root / ".gitkeep")
        return empty_table(table_name)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.csv")):
        if path.name == VOID_FILENAME:
            continue
        frames.append(pd.read_csv(path))
    for path in sorted(root.glob("*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        return empty_table(table_name)
    frame = pd.concat(frames, ignore_index=True)
    return frame.reindex(columns=SCHEMA_COLUMNS[table_name])


def load_raw_scalar_ratings(ratings_dir: str | Path, include_voided: bool = False) -> pd.DataFrame:
    frame = _load_annotation_dir(ratings_dir, "ratings")
    if include_voided or frame.empty:
        return frame
    voided = voided_rating_ids(ratings_dir)
    if not voided:
        return frame
    return frame[~frame["rating_id"].astype(str).isin(voided)].reset_index(drop=True)


def load_raw_pairwise_preferences(pairwise_dir: str | Path) -> pd.DataFrame:
    return _load_annotation_dir(pairwise_dir, "pairwise")
