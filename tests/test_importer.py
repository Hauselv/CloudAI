from __future__ import annotations

import cv2
import numpy as np

from cloud_aesthetics.data.manifest import build_manifest
from cloud_aesthetics.preprocessing.importer import import_private_images


def _write_rgb(path, image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(path), bgr)


def test_import_private_images_creates_crops_and_groups_derivatives(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((640, 800, 3), dtype=np.uint8)
    image[:, :] = [90, 170, 235]
    image[120:360, 160:560] = [245, 245, 245]
    image[520:, :] = [40, 90, 35]
    _write_rgb(source / "clouds.jpg", image)

    metadata_path = tmp_path / "image_derivatives.parquet"
    image_root = tmp_path / "images"
    imported = import_private_images(
        source,
        dataset_name="private_test",
        output_root=image_root,
        derivative_metadata_path=metadata_path,
        max_crops_per_image=3,
        min_sky_fraction=0.6,
        min_cloud_fraction=0.02,
    )

    assert (imported["derivative_kind"] == "original").sum() == 1
    assert (imported["derivative_kind"] == "sky_crop").sum() >= 1
    assert imported["import_batch_id"].nunique() == 1

    manifest = build_manifest(image_root, allowed_extensions=[".jpg"], derivative_metadata_path=metadata_path)
    assert len(manifest) == len(imported)
    assert manifest["split_group_id"].nunique() == 1
    assert manifest["import_batch_id"].nunique() == 1


def test_import_private_images_skips_duplicate_source_crops(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((640, 800, 3), dtype=np.uint8)
    image[:, :] = [90, 170, 235]
    image[120:360, 160:560] = [245, 245, 245]
    _write_rgb(source / "clouds.jpg", image)
    _write_rgb(source / "clouds_copy.jpg", image)

    imported = import_private_images(
        source,
        dataset_name="private_test",
        output_root=tmp_path / "images",
        derivative_metadata_path=tmp_path / "image_derivatives.parquet",
        max_crops_per_image=3,
        min_sky_fraction=0.6,
        min_cloud_fraction=0.02,
    )

    assert (imported["derivative_kind"] == "original").sum() == 2
    assert (imported["derivative_kind"] == "sky_crop").sum() == 3
    assert imported["relative_path"].duplicated().sum() == 0


def test_import_private_images_accepts_explicit_batch_id(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :] = [90, 170, 235]
    _write_rgb(source / "clouds.jpg", image)

    imported = import_private_images(
        source,
        dataset_name="private_test",
        output_root=tmp_path / "images",
        derivative_metadata_path=tmp_path / "image_derivatives.parquet",
        batch_id="batch_test_001",
        make_crops=False,
    )

    assert imported["import_batch_id"].tolist() == ["batch_test_001"]
