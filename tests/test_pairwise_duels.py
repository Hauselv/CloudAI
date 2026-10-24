from __future__ import annotations

from pathlib import Path

import pandas as pd

from cloud_aesthetics.data.aggregation import aggregate_ratings
from cloud_aesthetics.data.pairwise import bradley_terry_scores, select_duel_pair
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings, record_rating, void_rating, voided_rating_ids


def _pairs(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame([{"left_image_id": left, "right_image_id": right, "winner": winner, "tie_flag": winner is None, "rater_id": "r"} for left, right, winner in rows])


def test_bradley_terry_orders_transitive_results():
    pairs = _pairs([("a", "b", "a"), ("b", "c", "b"), ("a", "c", "a"), ("a", "b", "a"), ("c", "d", "c"), ("b", "d", "b"), ("a", "d", None)])
    table = bradley_terry_scores(pairs)
    order = table["image_id"].tolist()
    assert order[0] == "a" and order[-1] == "d"
    assert order.index("b") < order.index("c")
    by_id = table.set_index("image_id")
    assert by_id.loc["a", "duels"] == 4 and by_id.loc["a", "wins"] == 3.5
    assert 0 <= by_id["bt_score"].min() and by_id["bt_score"].max() <= 10
    assert abs(by_id["bt_logit"].mean()) < 1e-6  # scale fixed at geometric mean 1
    assert bradley_terry_scores(pd.DataFrame()).empty


def test_select_duel_pair_prefers_unseen_close_opponents():
    pool = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "score": [9.5, 9.0, 8.0, 10.0]})
    first = select_duel_pair(pool, pd.DataFrame(), seed=1)
    assert first is not None and len(set(first)) == 2
    seen = _pairs([("a", "b", "a"), ("a", "c", "a"), ("a", "d", "d"), ("b", "c", "b"), ("b", "d", "d")])
    remaining = select_duel_pair(pool, seen, seed=2)
    assert remaining is not None and set(remaining) == {"c", "d"}
    done = _pairs([("a", "b", "a"), ("a", "c", "a"), ("a", "d", "d"), ("b", "c", "b"), ("b", "d", "d"), ("c", "d", "d")])
    assert select_duel_pair(pool, done, seed=3) is None
    assert select_duel_pair(pool.head(1), pd.DataFrame()) is None


def test_void_rating_hides_the_rating_from_loaders(tmp_path: Path):
    ratings_dir = tmp_path / "ratings"
    first = record_rating(ratings_dir, image_id="x", rater_id="me", score=8.0, rating_session_id="s")
    second = record_rating(ratings_dir, image_id="y", rater_id="me", score=3.0, rating_session_id="s")
    void_rating(ratings_dir, second["rating_id"])
    assert voided_rating_ids(ratings_dir) == {second["rating_id"]}
    loaded = load_raw_scalar_ratings(ratings_dir)
    assert loaded["rating_id"].tolist() == [first["rating_id"]]
    assert len(load_raw_scalar_ratings(ratings_dir, include_voided=True)) == 2


def test_aggregate_ratings_adds_bradley_terry_from_explicit_pairs():
    ratings = pd.DataFrame(
        [
            {"rating_id": "1", "image_id": "a", "rater_id": "me", "rating_session_id": "s", "raw_score_1_to_10": 9.5, "rating_timestamp": "2024-01-01T00:00:00+00:00", "note": None},
            {"rating_id": "2", "image_id": "b", "rater_id": "me", "rating_session_id": "s", "raw_score_1_to_10": 9.5, "rating_timestamp": "2024-01-01T00:00:01+00:00", "note": None},
            {"rating_id": "3", "image_id": "c", "rater_id": "me", "rating_session_id": "s", "raw_score_1_to_10": 7.0, "rating_timestamp": "2024-01-01T00:00:02+00:00", "note": None},
        ]
    )
    pairs = _pairs([("a", "b", "a"), ("a", "b", "a"), ("b", "a", "a")])
    pairs["pair_id"] = ["p1", "p2", "p3"]
    pairs["timestamp"] = "2024-01-01T00:00:03+00:00"
    pairs["preference_strength"] = 1.0
    aggregated = aggregate_ratings(ratings, pairs).set_index("image_id")
    assert aggregated.loc["a", "bt_score"] > aggregated.loc["b", "bt_score"]
    assert aggregated.loc["a", "duels"] == 3 and aggregated.loc["c", "duels"] == 0
    assert pd.isna(aggregated.loc["c", "bt_score"])
