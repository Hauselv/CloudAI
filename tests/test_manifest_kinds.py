from __future__ import annotations

from pathlib import Path

import pandas as pd

from cloud_aesthetics.data.manifest import dataset_name_from_path, ensure_kind_columns, image_kind_from_path
from cloud_aesthetics.settings import PROJECT_ROOT


def test_image_kind_prefers_derivative_metadata():
    assert image_kind_from_path("data/raw/images/x/originals/a.jpg", "sky_crop") == "crop"
    assert image_kind_from_path("data/raw/images/x/crops/a.jpg", "original") == "original"


def test_image_kind_falls_back_to_crops_folder():
    assert image_kind_from_path("data/raw/images/x/crops/a.jpg") == "crop"
    assert image_kind_from_path("data\\raw\\images\\x\\crops\\a.jpg") == "crop"
    assert image_kind_from_path("data/raw/images/web_sample/cumulus/a.jpg") == "original"


def test_dataset_name_is_first_folder_below_image_root():
    root = PROJECT_ROOT / "data" / "raw" / "images"
    assert dataset_name_from_path("data/raw/images/private_wolken/crops/a.jpg", root) == "private_wolken"
    assert dataset_name_from_path("data/raw/images/web_sample/cumulus/a.jpg", root) == "web_sample"
    assert dataset_name_from_path("data/raw/images/loose.jpg", root) is None
    assert dataset_name_from_path(str(Path("C:/elsewhere/a.jpg")), root) is None


def test_ensure_kind_columns_fills_missing_values():
    manifest = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "relative_path": ["data/raw/images/web_sample/a.jpg", "data/raw/images/private/crops/b.jpg"],
        }
    )
    result = ensure_kind_columns(manifest)
    assert result["kind"].tolist() == ["original", "crop"]
    assert result["dataset_name"].tolist() == ["web_sample", "private"]
    # Existing values are preserved, only gaps are filled.
    partial = manifest.assign(kind=["crop", None], dataset_name=[None, "custom"])
    filled = ensure_kind_columns(partial)
    assert filled["kind"].tolist() == ["crop", "crop"]
    assert filled["dataset_name"].tolist() == ["web_sample", "custom"]
