from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cloud_aesthetics.cli import analyze_batch_impl, ensure_features_for_images
from cloud_aesthetics.data.manifest import build_manifest
from cloud_aesthetics.models.baseline import train_baseline_suite
from cloud_aesthetics.preprocessing.importer import import_private_images
from cloud_aesthetics.utils.io import write_table


def _write_rgb(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(path), bgr)


def test_batch_analysis_writes_predictions_and_skips_existing(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(12):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        image[:, :] = [80 + index * 4, 150 + index * 2, 220]
        image[80:180, 80:240] = [245, 245, 245]
        _write_rgb(source / f"cloud_{index:02d}.jpg", image)

    image_root = tmp_path / "images"
    metadata_path = tmp_path / "image_derivatives.parquet"
    imported = import_private_images(
        source,
        dataset_name="batch_test",
        output_root=image_root,
        derivative_metadata_path=metadata_path,
        batch_id="batch_test_001",
        make_crops=False,
    )
    assert imported["import_batch_id"].nunique() == 1

    manifest = build_manifest(image_root, allowed_extensions=[".jpg"], derivative_metadata_path=metadata_path)
    manifest_path = tmp_path / "manifest.parquet"
    labels_path = tmp_path / "labels.parquet"
    splits_path = tmp_path / "splits.parquet"
    ratings_path = tmp_path / "ratings.parquet"
    pairwise_path = tmp_path / "pairwise.parquet"
    feature_path = tmp_path / "features.parquet"
    write_table(manifest, manifest_path)
    write_table(pd.DataFrame(), ratings_path)
    write_table(pd.DataFrame(), pairwise_path)

    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        "\n".join(
            [
                f"image_root: {image_root.as_posix()}",
                f"manifest_path: {manifest_path.as_posix()}",
                f"ratings_dir: {(tmp_path / 'ratings').as_posix()}",
                f"pairwise_dir: {(tmp_path / 'pairwise').as_posix()}",
                f"ratings_path: {ratings_path.as_posix()}",
                f"pairwise_path: {pairwise_path.as_posix()}",
                f"aggregated_labels_path: {labels_path.as_posix()}",
                f"splits_path: {splits_path.as_posix()}",
                "allowed_extensions: ['.jpg']",
                "capture_session_strategy: parent_dir",
            ]
        ),
        encoding="utf-8",
    )
    feature_config = tmp_path / "features.yaml"
    feature_config.write_text(f"image_size: 128\noutput_path: {feature_path.as_posix()}\n", encoding="utf-8")

    image_ids = manifest["image_id"].astype(str).tolist()
    features = ensure_features_for_images(manifest, image_ids, feature_config)
    labels = pd.DataFrame(
        {
            "image_id": image_ids,
            "mean_score": [float(4 + (index % 6)) for index in range(len(image_ids))],
        }
    )
    splits = pd.DataFrame(
        {
            "image_id": image_ids,
            "partition": ["dev"] * 9 + ["test"] * 3,
            "fold": [index % 3 for index in range(9)] + [-1] * 3,
        }
    )
    write_table(labels, labels_path)
    write_table(splits, splits_path)
    summary = train_baseline_suite(
        features,
        labels,
        splits,
        {
            "output_dir": str(tmp_path / "runs"),
            "run_name": "baseline_batch",
            "target_column": "mean_score",
            "fold_holdout": 0,
            "seed": 42,
            "regressors": ["elasticnet"],
        },
        pairwise_table=pd.DataFrame(),
    )
    model_path = next(iter(summary["models"].values()))["model_path"]
    run_dir = Path(model_path).parent

    first = analyze_batch_impl(run_dir, "batch_test_001", dataset_config, feature_config)
    assert first["image_count"] == len(image_ids)
    assert first["analyzed_count"] == len(image_ids)
    predictions = pd.read_parquet(first["predictions_path"])
    assert set(["batch_id", "run_id", "image_id", "predicted_score", "top_features", "top_concepts"]).issubset(predictions.columns)

    second = analyze_batch_impl(run_dir, "batch_test_001", dataset_config, feature_config)
    assert second["analyzed_count"] == 0
    assert second["skipped_count"] == len(image_ids)
