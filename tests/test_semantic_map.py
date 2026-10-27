from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cloud_aesthetics.eval.semantic_map import ProjectionParams, build_map_table, build_projection, compute_projection, load_projection, map_regions, map_summary


def _embeddings(n_per: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = np.eye(3, 16) * 6
    rows, ids = [], []
    for blob in range(3):
        for k in range(n_per):
            rows.append(centres[blob] + rng.normal(scale=0.3, size=16))
            ids.append(f"b{blob}_{k}")
    frame = pd.DataFrame(np.array(rows, dtype=np.float32), columns=[f"emb_{i:03d}" for i in range(16)])
    frame.insert(0, "image_id", ids)
    return frame


def test_pca_projection_separates_blobs_and_is_cached(tmp_path: Path):
    frame = _embeddings()
    params = ProjectionParams(model_key="fake", method="pca")
    coords = compute_projection(frame.drop(columns=["image_id"]).to_numpy(), params)
    assert coords.shape == (60, 2)
    first = build_projection(params, output_dir=tmp_path, embeddings=frame)
    assert (tmp_path / "projection_fake_pca.parquet").exists() and (tmp_path / "projection_fake_pca.json").exists()
    cached = build_projection(params, output_dir=tmp_path, embeddings=frame)
    pd.testing.assert_frame_equal(first, cached)
    assert len(load_projection(params, tmp_path)) == 60
    blob_means = first.assign(blob=first["image_id"].str[:2]).groupby("blob")[["x", "y"]].mean()
    spread = np.linalg.norm(blob_means.to_numpy()[:, None, :] - blob_means.to_numpy()[None, :, :], axis=-1)
    assert spread[np.triu_indices(3, 1)].min() > 2.0


def test_map_table_and_regions_find_label_gaps(tmp_path: Path):
    frame = _embeddings()
    params = ProjectionParams(model_key="fake", method="pca")
    projection = build_projection(params, output_dir=tmp_path, embeddings=frame)
    ids = projection["image_id"].tolist()
    manifest = pd.DataFrame({"image_id": ids, "relative_path": [f"{i}.jpg" for i in ids], "kind": "original", "crop_class": "original", "dataset_name": "toy"})
    labeled_ids = [i for i in ids if i.startswith("b0")] + [i for i in ids if i.startswith("b1")][:5]
    labels = pd.DataFrame({"image_id": labeled_ids, "mean_score": [9.5] * 20 + [6.0] * 5, "n_raters": 1})
    zero_shot = pd.DataFrame({"image_id": ids, "clip_cloud_type": ["cirrus" if i.startswith("b2") else "cumulus" for i in ids], "clip_light": "daylight"})
    judge = {"llm_x": pd.DataFrame({"image_id": ids, "judge_score": [8.0 if i.startswith("b2") else 5.0 for i in ids]})}
    predictions = pd.DataFrame({"image_id": ids, "pred_mean": 7.0})
    table = build_map_table(projection, manifest, labels=labels, zero_shot=zero_shot, judge_tables=judge, predictions=predictions)
    assert len(table) == 60 and table["labeled"].sum() == 25
    assert table.loc[table["image_id"] == "b2_0", "judge_mean"].iloc[0] == 8.0
    assert table.loc[table["image_id"] == "b0_0", "estimate"].iloc[0] == 9.5  # human label wins over model
    table, regions = map_regions(table, n_regions=3, seed=1)
    assert set(table["region"]) == {0, 1, 2} and len(regions) == 3
    gap = regions.iloc[0]
    assert gap["n_labeled"] == 0 and gap["top_cloud_type"] == "cirrus" and len(gap["unlabeled_candidates"]) > 0
    tens = regions[regions["n_top"] > 0]
    assert len(tens) == 1 and tens.iloc[0]["mean_score"] == 9.5
    summary = map_summary(table, regions)
    assert summary["regions_without_labels"] == 1 and summary["regions_with_tens"] == 1
