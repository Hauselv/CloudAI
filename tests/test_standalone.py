from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cloud_aesthetics.standalone import build_friend_package, import_friend_label_bundle


def test_build_friend_package_copies_images_and_embeds_manifest(tmp_path: Path) -> None:
    image_path = tmp_path / "cloud.jpg"
    cv2.imwrite(str(image_path), np.full((12, 16, 3), 220, dtype=np.uint8))
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "image_id": "abc123",
                "relative_path": str(image_path),
                "sha256": "abc123",
                "width": 16,
                "height": 12,
                "capture_date": None,
                "capture_session_id": "session",
                "phash": "1010",
                "split_group_id": "group",
            }
        ]
    ).to_parquet(manifest_path)

    package_dir = build_friend_package(
        tmp_path / "friend_package",
        manifest_path=manifest_path,
        package_name="friend_package",
        zip_package=False,
    )

    assert (package_dir / "index.html").exists()
    assert (package_dir / "manifest.json").exists()
    assert (package_dir / "images" / "abc123_cloud.jpg").exists()
    assert '"image_id": "abc123"' in (package_dir / "index.html").read_text(encoding="utf-8")


def test_import_friend_label_bundle_writes_labels_and_images(tmp_path: Path) -> None:
    png_bytes = base64.b64encode(b"tiny-image").decode("ascii")
    bundle_path = tmp_path / "labels.json"
    bundle_path.write_text(
        json.dumps(
            {
                "rater_id": "friend_a",
                "session_id": "session_a",
                "ratings": [{"image_id": "img_a", "score": 8.5, "note": "nice"}],
                "pairwise": [
                    {
                        "left_image_id": "img_a",
                        "right_image_id": "img_b",
                        "winner": "img_a",
                        "tie_flag": False,
                        "preference_strength": 0.75,
                    }
                ],
                "imported_images": [{"image_id": "friend_img", "data_url": f"data:image/png;base64,{png_bytes}"}],
            }
        ),
        encoding="utf-8",
    )

    summary = import_friend_label_bundle(
        bundle_path,
        ratings_dir=tmp_path / "ratings",
        pairwise_dir=tmp_path / "pairwise",
        imported_images_root=tmp_path / "images",
    )

    assert summary == {"ratings": 1, "pairwise": 1, "imported_images": 1}
    assert (tmp_path / "ratings" / "friend_a.csv").exists()
    assert (tmp_path / "pairwise" / "friend_a.csv").exists()
    assert (tmp_path / "images" / "friend_img.png").read_bytes() == b"tiny-image"
