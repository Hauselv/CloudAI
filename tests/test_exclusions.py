from __future__ import annotations

import cv2
import numpy as np

from cloud_aesthetics.data.exclusions import active_excluded_ids, set_exclusion
from cloud_aesthetics.data.manifest import build_manifest


def test_excluded_image_is_removed_from_manifest(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    image_path = image_root / "sample.jpg"
    assert cv2.imwrite(str(image_path), image)

    manifest = build_manifest(image_root, allowed_extensions=[".jpg"], exclusions_path=tmp_path / "exclusions.csv")
    image_id = manifest.iloc[0]["image_id"]
    set_exclusion(image_id, excluded=True, reason="satellite", path_like=tmp_path / "exclusions.csv")

    assert image_id in active_excluded_ids(tmp_path / "exclusions.csv")
    filtered = build_manifest(image_root, allowed_extensions=[".jpg"], exclusions_path=tmp_path / "exclusions.csv")
    assert filtered.empty


def test_exclusion_can_be_reversed(tmp_path):
    set_exclusion("img1", excluded=True, reason="bad_crop", path_like=tmp_path / "exclusions.csv")
    set_exclusion("img1", excluded=False, reason="restore", path_like=tmp_path / "exclusions.csv")

    assert "img1" not in active_excluded_ids(tmp_path / "exclusions.csv")
