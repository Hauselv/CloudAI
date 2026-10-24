from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cloud_aesthetics.data.derivatives import classify_crop, crop_context, crop_scale_table, render_crop_context
from cloud_aesthetics.data.manifest import build_manifest
from cloud_aesthetics.preprocessing.importer import import_private_images


def test_classify_crop_uses_scale_and_pixel_size():
    assert classify_crop("original", None, 3000) == "original"
    assert classify_crop("crop", 0.35, 1058) == "small_crop"
    assert classify_crop("crop", 0.5, 1500) == "large_crop"
    assert classify_crop("crop", 0.9, 700) == "small_crop"
    assert classify_crop("crop", None, 1200) == "large_crop"
    assert classify_crop("crop", None, None) == "large_crop"


def test_crop_scale_table_derives_scale_from_source_size():
    derivatives = pd.DataFrame(
        [
            {"relative_path": "o.jpg", "derivative_kind": "original", "source_image_id": "s", "crop_width": 2000, "crop_height": 1000},
            {"relative_path": "c.jpg", "derivative_kind": "sky_crop", "source_image_id": "s", "crop_width": 500, "crop_height": 500},
        ]
    )
    table = crop_scale_table(derivatives)
    assert table.set_index("relative_path").loc["c.jpg", "crop_scale"] == 0.5


def test_import_records_scale_and_context_is_renderable(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    image[:, :] = [90, 150, 230]
    image[100:500, 200:900] = [240, 240, 240]
    cv2.imwrite(str(source / "sky.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_root = tmp_path / "images"
    metadata_path = tmp_path / "derivatives.parquet"
    imported = import_private_images(
        source,
        dataset_name="ctx",
        output_root=image_root,
        derivative_metadata_path=metadata_path,
        min_crop_size=400,
        crop_scales=(0.5, 0.9),
        min_sky_fraction=0.3,
        min_cloud_fraction=0.01,
        max_crops_per_image=4,
    )
    crops = imported[imported["derivative_kind"] == "sky_crop"]
    assert not crops.empty
    assert set(crops["crop_scale"].round(2)).issubset({0.5, 0.9})
    assert (crops["crop_width"] >= 400).all()
    manifest = build_manifest(image_root, allowed_extensions=[".jpg"], derivative_metadata_path=metadata_path)
    crop_rows = manifest[manifest["kind"] == "crop"]
    assert set(crop_rows["crop_class"]) <= {"small_crop", "large_crop"}
    assert crop_rows["crop_scale"].notna().all()
    derivatives = pd.read_parquet(metadata_path)
    context = crop_context(str(crop_rows.iloc[0]["relative_path"]), derivatives)
    assert context is not None
    assert context.source_width == 1200 and context.source_height == 900
    rendered = render_crop_context(context, max_edge=300)
    assert rendered is not None and rendered.shape[1] == 300
    assert crop_context(str(manifest[manifest["kind"] == "original"].iloc[0]["relative_path"]), derivatives) is None


def test_small_sources_get_no_enlarged_crops(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(source / "tiny.jpg"), image)
    imported = import_private_images(
        source,
        dataset_name="tiny",
        output_root=tmp_path / "images",
        derivative_metadata_path=tmp_path / "d.parquet",
        min_crop_size=768,
        min_sky_fraction=0.0,
        min_cloud_fraction=0.0,
    )
    assert (imported["derivative_kind"] == "sky_crop").sum() == 0
