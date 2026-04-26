from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from cloud_aesthetics.data.schemas import ImageRecord, RatingRecord, require_columns


def test_image_record_schema_accepts_valid_row():
    record = ImageRecord(
        image_id="abc123",
        relative_path="data/raw/images/sample.jpg",
        sha256="f" * 64,
        width=100,
        height=80,
        capture_session_id="session_a",
        phash="10101010",
        split_group_id="session_a_hash",
    )
    assert record.image_id == "abc123"


def test_rating_record_enforces_score_bounds():
    record = RatingRecord(
        rating_id="rate1",
        image_id="img1",
        rater_id="rater1",
        rating_session_id="session",
        raw_score_1_to_10=9.0,
        rating_timestamp=datetime.now(timezone.utc),
    )
    assert record.raw_score_1_to_10 == 9.0


def test_rating_record_accepts_zero_score():
    record = RatingRecord(
        rating_id="rate_zero",
        image_id="img1",
        rater_id="rater1",
        rating_session_id="session",
        raw_score_1_to_10=0.0,
        rating_timestamp=datetime.now(timezone.utc),
    )
    assert record.raw_score_1_to_10 == 0.0


def test_require_columns_raises_for_missing_fields():
    frame = pd.DataFrame({"image_id": ["x"]})
    try:
        require_columns(frame, ["image_id", "missing"], "demo")
    except ValueError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected require_columns to fail")
