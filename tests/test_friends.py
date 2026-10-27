from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cloud_aesthetics.data.ratings import load_raw_pairwise_preferences, load_raw_scalar_ratings, record_rating
from cloud_aesthetics.friends.queue import FriendQueueConfig, build_friend_queue, next_items, progress
from cloud_aesthetics.friends.registry import add_friend, load_friends, resolve_token, revoke_friend
from cloud_aesthetics.friends.server import FriendServerConfig, create_app


def _setup(tmp_path: Path, n: int = 8):
    from PIL import Image

    ids = [f"img{k:02d}" for k in range(n)]
    for k, image_id in enumerate(ids):
        Image.fromarray(np.full((40, 60, 3), 40 + 20 * k, dtype=np.uint8)).save(tmp_path / f"{image_id}.jpg")
    manifest = pd.DataFrame({"image_id": ids, "relative_path": [str(tmp_path / f"{i}.jpg") for i in ids], "kind": ["original"] * 5 + ["crop"] * 3, "crop_class": ["original"] * 5 + ["large_crop", "large_crop", "small_crop"], "dataset_name": "toy", "split_group_id": ids})
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    ratings_dir = tmp_path / "ratings"
    for image_id in ids[:3]:
        record_rating(ratings_dir, image_id=image_id, rater_id="friend_a", score=8.0, rating_session_id="s")
    config = FriendServerConfig(manifest_path=tmp_path / "manifest.parquet", ratings_dir=ratings_dir, pairwise_dir=tmp_path / "pairwise", friends_path=tmp_path / "friends.csv", raters_path=tmp_path / "raters.csv", exclusions_path=tmp_path / "exclusions.csv", map_queue_path=tmp_path / "missing.parquet", triage_path=tmp_path / "missing2.parquet", cache_dir=tmp_path / "cache", image_edge=32, requests_per_minute=1000)
    friend = add_friend("Anna Müller", path_like=config.friends_path, raters_path=config.raters_path)
    return ids, config, friend


def test_registry_tokens_and_revocation(tmp_path: Path):
    _, config, friend = _setup(tmp_path)
    assert friend["rater_id"] == "friend_anna_mueller" and len(friend["token"]) >= 10
    second = add_friend("Anna Müller", path_like=config.friends_path, raters_path=config.raters_path)
    assert second["rater_id"] == "friend_anna_mueller2"
    assert resolve_token(friend["token"], path_like=config.friends_path)["display_name"] == "Anna Müller"
    assert resolve_token("nope", path_like=config.friends_path) is None and resolve_token("../x", path_like=config.friends_path) is None
    assert revoke_friend(friend["token"], path_like=config.friends_path)
    assert resolve_token(friend["token"], path_like=config.friends_path) is None
    raters = pd.read_csv(config.raters_path)
    assert (raters["rater_kind"] == "human").all() and len(load_friends(config.friends_path)) == 2


def test_queue_core_first_then_rest_and_seeded(tmp_path: Path):
    ids, config, friend = _setup(tmp_path)
    manifest = pd.read_parquet(config.manifest_path)
    core = {"img00", "img01", "img02"}
    queue = build_friend_queue(manifest, token="t1", core_rated_ids=core, excluded_ids={"img04"})
    assert set(queue[:3]) == core and "img04" not in queue and "img07" not in queue  # small crop excluded by default
    assert queue == build_friend_queue(manifest, token="t1", core_rated_ids=core, excluded_ids={"img04"})
    other = build_friend_queue(manifest, token="t2", core_rated_ids=core, excluded_ids={"img04"})
    assert set(other[:3]) == core
    assert next_items(queue, {"img00"}, exclude={"img01"}, limit=2) == [i for i in queue if i not in {"img00", "img01"}][:2]
    done = progress(queue, {"img00", "img05"}, core)
    assert done["n_done"] == 2 and done["n_core_done"] == 1 and done["n_core"] == 3


def test_server_flow_rate_undo_duel_and_guards(tmp_path: Path):
    ids, config, friend = _setup(tmp_path)
    client = TestClient(create_app(config))
    token = friend["token"]
    assert client.get("/").status_code == 404 and client.get("/f/invalid").status_code == 404
    assert client.get(f"/api/{token}/image/..%2F..%2Fsecret").status_code == 404
    page = client.get(f"/f/{token}")
    assert page.status_code == 200 and "Wolken bewerten" in page.text and page.headers["X-Frame-Options"] == "DENY"
    nxt = client.get(f"/api/{token}/next?limit=3").json()
    assert nxt["display_name"] == "Anna Müller" and len(nxt["items"]) == 3 and nxt["progress"]["n_core"] == 3
    first = nxt["items"][0]["image_id"]
    assert first in {"img00", "img01", "img02"}
    image = client.get(f"/api/{token}/image/{first}")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg" and len(image.content) > 100
    assert client.get(f"/api/{token}/image/img99").status_code == 404
    assert client.post(f"/api/{token}/rate", json={"image_id": first, "score": 11}).status_code == 422
    rated = client.post(f"/api/{token}/rate", json={"image_id": first, "score": 8.3}).json()
    assert rated["score"] == 8.5 and rated["progress"]["n_done"] == 1
    ratings = load_raw_scalar_ratings(config.ratings_dir)
    own = ratings[ratings["rater_id"] == "friend_anna_mueller"]
    assert len(own) == 1 and own.iloc[0]["raw_score_1_to_10"] == 8.5 and own.iloc[0]["rating_session_id"].startswith("friend_anna_mueller_")
    nxt2 = client.get(f"/api/{token}/next?limit=5&exclude={first}").json()
    assert first not in [item["image_id"] for item in nxt2["items"]]
    assert client.post(f"/api/{token}/undo", json={"rating_id": "does-not-exist"}).status_code == 404
    undone = client.post(f"/api/{token}/undo", json={"rating_id": rated["rating_id"]}).json()
    assert undone["progress"]["n_done"] == 0 and load_raw_scalar_ratings(config.ratings_dir, include_voided=False)["rater_id"].eq("friend_anna_mueller").sum() == 0
    assert client.get(f"/api/{token}/duel").json()["pair"] is None
    for image_id in ids[:3]:
        client.post(f"/api/{token}/rate", json={"image_id": image_id, "score": 9})
    duel = client.get(f"/api/{token}/duel").json()
    assert duel["pair"] and len(duel["pair"]) == 2
    assert client.post(f"/api/{token}/duel", json={"left": duel["pair"][0], "right": duel["pair"][1], "winner": "img99"}).status_code == 400
    saved = client.post(f"/api/{token}/duel", json={"left": duel["pair"][0], "right": duel["pair"][1], "winner": duel["pair"][0]}).json()
    assert saved["winner"] == duel["pair"][0]
    pairs = load_raw_pairwise_preferences(config.pairwise_dir)
    assert len(pairs) == 1 and pairs.iloc[0]["rater_id"] == "friend_anna_mueller"
    revoke_friend(token, path_like=config.friends_path)
    assert client.get(f"/api/{token}/next").status_code == 404


def test_rate_limit(tmp_path: Path):
    _, config, friend = _setup(tmp_path)
    config.requests_per_minute = 3
    client = TestClient(create_app(config))
    codes = [client.get(f"/api/{friend['token']}/next").status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
