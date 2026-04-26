from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field


class ImageRecord(BaseModel):
    image_id: str
    relative_path: str
    sha256: str
    width: int
    height: int
    capture_date: str | None = None
    capture_session_id: str
    phash: str
    split_group_id: str


class RatingRecord(BaseModel):
    rating_id: str
    image_id: str
    rater_id: str
    rating_session_id: str
    raw_score_1_to_10: float = Field(ge=0.0, le=10.0)
    rating_timestamp: datetime
    note: str | None = None


class PairwisePreferenceRecord(BaseModel):
    pair_id: str
    left_image_id: str
    right_image_id: str
    rater_id: str
    winner: str | None = None
    tie_flag: bool = False
    preference_strength: float | None = None
    timestamp: datetime


class AggregatedLabelRecord(BaseModel):
    image_id: str
    mean_score: float
    median_score: float
    trimmed_mean_score: float
    std_score: float
    sem_score: float
    n_raters: int
    agreement_index: float
    normalized_mean_score: float
    pairwise_win_rate: float


SCHEMA_COLUMNS: dict[str, list[str]] = {
    "manifest": list(ImageRecord.model_fields.keys()),
    "ratings": list(RatingRecord.model_fields.keys()),
    "pairwise": list(PairwisePreferenceRecord.model_fields.keys()),
    "aggregated": list(AggregatedLabelRecord.model_fields.keys()),
}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{table_name} is missing required columns: {joined}")


def empty_table(table_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA_COLUMNS[table_name])
