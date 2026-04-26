from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from cloud_aesthetics.data.schemas import SCHEMA_COLUMNS
from cloud_aesthetics.data.exclusions import active_excluded_ids
from cloud_aesthetics.settings import PROJECT_ROOT, ensure_parent, resolve_path
from cloud_aesthetics.utils.io import read_table, write_table


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_phash(path: Path, hash_size: int = 8) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image for phash: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:hash_size, :hash_size]
    threshold = np.median(low_freq[1:, :].ravel())
    bits = low_freq > threshold
    return "".join("1" if value else "0" for value in bits.flatten())


def infer_capture_session(path: Path, image_root: Path, strategy: str = "parent_dir") -> str:
    if strategy == "parent_dir" and path.parent != image_root:
        return path.parent.name
    stem_parts = path.stem.split("_")
    if len(stem_parts) >= 2:
        return "_".join(stem_parts[:2])
    return "default_session"


def project_relative_or_absolute(path: Path) -> str:
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def build_manifest(
    image_root: str | Path,
    allowed_extensions: Iterable[str] | None = None,
    capture_session_strategy: str = "parent_dir",
    derivative_metadata_path: str | Path = "data/raw/metadata/image_derivatives.parquet",
    exclusions_path: str | Path = "data/raw/metadata/exclusions.csv",
) -> pd.DataFrame:
    root = resolve_path(image_root)
    extensions = {extension.lower() for extension in (allowed_extensions or [".jpg", ".jpeg", ".png"])}
    derivative_metadata = read_table(derivative_metadata_path)
    derivative_groups: dict[str, str] = {}
    derivative_batches: dict[str, str] = {}
    if not derivative_metadata.empty and {"relative_path", "source_image_id"}.issubset(derivative_metadata.columns):
        derivative_groups = {
            str(row["relative_path"]): str(row["source_image_id"])
            for _, row in derivative_metadata.dropna(subset=["relative_path", "source_image_id"]).iterrows()
        }
    if not derivative_metadata.empty and {"relative_path", "import_batch_id"}.issubset(derivative_metadata.columns):
        derivative_batches = {
            str(row["relative_path"]): str(row["import_batch_id"])
            for _, row in derivative_metadata.dropna(subset=["relative_path", "import_batch_id"]).iterrows()
        }
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        relative_path = project_relative_or_absolute(path)
        sha256 = compute_sha256(path)
        phash = compute_phash(path)
        capture_session_id = infer_capture_session(path, root, capture_session_strategy)
        image_id = sha256[:16]
        split_group_id = derivative_groups.get(relative_path, f"{capture_session_id}_{phash[:12]}")
        rows.append(
            {
                "image_id": image_id,
                "relative_path": relative_path,
                "sha256": sha256,
                "width": width,
                "height": height,
                "import_batch_id": derivative_batches.get(relative_path),
                "capture_date": None,
                "capture_session_id": capture_session_id,
                "phash": phash,
                "split_group_id": split_group_id,
            }
        )
    frame = pd.DataFrame(rows, columns=SCHEMA_COLUMNS["manifest"])
    if not frame.empty:
        dedupe_subset = ["relative_path"] if derivative_groups else ["sha256"]
        frame = frame.drop_duplicates(subset=dedupe_subset).reset_index(drop=True)
        excluded_ids = active_excluded_ids(exclusions_path)
        if excluded_ids:
            frame = frame[~frame["image_id"].astype(str).isin(excluded_ids)].reset_index(drop=True)
    return frame


def save_manifest(frame: pd.DataFrame, path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    return write_table(frame, path)
