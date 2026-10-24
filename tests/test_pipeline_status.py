from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cloud_aesthetics.cli import aggregate_labels_impl
from cloud_aesthetics.data.manifest import build_manifest
from cloud_aesthetics.data.ratings import record_rating
from cloud_aesthetics.pipeline import pipeline_status, refresh_pipeline, status_frame
from cloud_aesthetics.utils.io import write_table


def _write_rgb(path: Path, image: np.ndarray) -> None:
    assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _setup(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    image_root = tmp_path / "images" / "toy_set"
    image_root.mkdir(parents=True)
    for index in range(8):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[:, :] = [90 + index * 5, 150, 220]
        image[40:100, 30:130] = [240, 240, 240]
        _write_rgb(image_root / f"cloud_{index:02d}.jpg", image)
    manifest = build_manifest(tmp_path / "images", allowed_extensions=[".jpg"], derivative_metadata_path=tmp_path / "none.parquet", exclusions_path=tmp_path / "none.csv")
    manifest_path = tmp_path / "manifest.parquet"
    write_table(manifest, manifest_path)
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        "\n".join(
            [
                f"image_root: {(tmp_path / 'images').as_posix()}",
                f"manifest_path: {manifest_path.as_posix()}",
                f"ratings_dir: {(tmp_path / 'ratings').as_posix()}",
                f"pairwise_dir: {(tmp_path / 'pairwise').as_posix()}",
                f"ratings_path: {(tmp_path / 'ratings.parquet').as_posix()}",
                f"pairwise_path: {(tmp_path / 'pairwise.parquet').as_posix()}",
                f"aggregated_labels_path: {(tmp_path / 'labels.parquet').as_posix()}",
                f"splits_path: {(tmp_path / 'splits.parquet').as_posix()}",
                f"exclusions_path: {(tmp_path / 'exclusions.csv').as_posix()}",
                f"artifacts_root: {(tmp_path / 'runs').as_posix()}",
                f"embeddings_dir: {(tmp_path / 'embeddings').as_posix()}",
                "allowed_extensions: ['.jpg']",
                "capture_session_strategy: parent_dir",
                "split: {n_splits: 2, score_bins: 2, test_fraction: 0.25, random_state: 1}",
            ]
        ),
        encoding="utf-8",
    )
    feature_config = tmp_path / "features.yaml"
    feature_config.write_text(f"image_size: 96\noutput_path: {(tmp_path / 'features.parquet').as_posix()}\n", encoding="utf-8")
    return dataset_config, feature_config, manifest


def test_manifest_carries_kind_and_dataset(tmp_path: Path):
    _, _, manifest = _setup(tmp_path)
    assert set(manifest["kind"]) == {"original"}
    assert set(manifest["dataset_name"]) == {"toy_set"}


def test_pipeline_status_tracks_missing_and_stale_stages(tmp_path: Path):
    dataset_config, feature_config, manifest = _setup(tmp_path)
    by_key = {stage.key: stage for stage in pipeline_status(dataset_config, feature_config)}
    assert by_key["raw_images"].rows == 8
    assert by_key["manifest"].state == "ok"
    assert by_key["raw_ratings"].state == "missing"
    assert by_key["aggregated_labels"].state == "missing"
    assert by_key["features"].state == "missing"
    assert by_key["latest_run"].exists is False

    for index, image_id in enumerate(manifest["image_id"].astype(str)):
        record_rating(tmp_path / "ratings", image_id=image_id, rater_id="tester", score=float(2 + index), rating_session_id="s1")
    by_key = {stage.key: stage for stage in pipeline_status(dataset_config, feature_config)}
    assert by_key["raw_ratings"].rows == 8
    assert by_key["aggregated_labels"].state in {"missing", "stale"}

    aggregate_labels_impl(dataset_config)
    by_key = {stage.key: stage for stage in pipeline_status(dataset_config, feature_config)}
    assert by_key["aggregated_labels"].state == "ok"
    assert by_key["aggregated_labels"].rows == 8
    assert by_key["splits"].state == "ok"
    labels = pd.read_parquet(tmp_path / "labels.parquet")
    assert labels["kind"].tolist() == ["original"] * 8
    assert labels["dataset_name"].tolist() == ["toy_set"] * 8

    report = refresh_pipeline(aggregate=False, features="missing", dataset_config_path=dataset_config, feature_config_path=feature_config)
    assert report["features"] == 8
    by_key = {stage.key: stage for stage in pipeline_status(dataset_config, feature_config)}
    assert by_key["features"].state == "ok"
    features = pd.read_parquet(tmp_path / "features.parquet")
    assert set(features["kind"]) == {"original"}

    frame = status_frame(pipeline_status(dataset_config, feature_config))
    assert list(frame.columns) == ["label", "state", "rows", "updated_at", "note"]
    assert len(frame) == 8
    assert by_key["features"].state == "ok"
    embeddings_stage = {stage.key: stage for stage in pipeline_status(dataset_config, feature_config)}["embeddings"]
    assert embeddings_stage.exists is False
