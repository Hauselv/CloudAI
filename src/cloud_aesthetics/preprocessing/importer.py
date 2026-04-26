from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cloud_aesthetics.data.manifest import compute_sha256, project_relative_or_absolute
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.settings import ensure_parent, resolve_path
from cloud_aesthetics.utils.io import read_table, write_table


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DERIVATIVE_COLUMNS = [
    "relative_path",
    "dataset_name",
    "source_relative_path",
    "source_sha256",
    "source_image_id",
    "derivative_kind",
    "crop_x",
    "crop_y",
    "crop_width",
    "crop_height",
    "sky_fraction",
    "cloud_fraction",
]


@dataclass(frozen=True)
class CropCandidate:
    x: int
    y: int
    width: int
    height: int
    sky_fraction: float
    cloud_fraction: float


def _safe_stem(path: Path, max_length: int = 80) -> str:
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in path.stem)
    return safe[:max_length].strip("_") or "image"


def _safe_dataset_name(dataset_name: str) -> str:
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in dataset_name)
    safe = safe.strip("_")
    if not safe:
        raise ValueError("Dataset name must contain at least one letter or number.")
    return safe


def estimate_sky_cloud_masks(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    blue_sky = (hue >= 85) & (hue <= 135) & (saturation >= 20) & (value >= 70)
    bright_cloud = (saturation <= 95) & (value >= 115)
    hazy_sky = (saturation <= 70) & (value >= 145)
    sky_mask = (blue_sky | bright_cloud | hazy_sky).astype(np.uint8)
    cloud_mask = ((saturation <= 85) & (value >= 125)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
    return sky_mask, cloud_mask


def _crop_candidates(
    image_rgb: np.ndarray,
    *,
    min_size: int,
    scales: tuple[float, ...],
    stride_fraction: float,
    min_sky_fraction: float,
    min_cloud_fraction: float,
    max_crops: int,
) -> list[CropCandidate]:
    height, width = image_rgb.shape[:2]
    sky_mask, cloud_mask = estimate_sky_cloud_masks(image_rgb)
    candidates: list[CropCandidate] = []
    short_edge = min(width, height)
    for scale in scales:
        crop_size = int(round(short_edge * scale))
        crop_size = max(min_size, crop_size)
        if crop_size > width or crop_size > height:
            continue
        step = max(1, int(round(crop_size * stride_fraction)))
        x_positions = sorted(set(list(range(0, width - crop_size + 1, step)) + [width - crop_size]))
        y_positions = sorted(set(list(range(0, height - crop_size + 1, step)) + [height - crop_size]))
        for y in y_positions:
            for x in x_positions:
                sky_crop = sky_mask[y : y + crop_size, x : x + crop_size]
                cloud_crop = cloud_mask[y : y + crop_size, x : x + crop_size]
                sky_fraction = float(sky_crop.mean())
                cloud_fraction = float(cloud_crop.mean())
                if sky_fraction < min_sky_fraction or cloud_fraction < min_cloud_fraction:
                    continue
                candidates.append(CropCandidate(x, y, crop_size, crop_size, sky_fraction, cloud_fraction))
    candidates.sort(key=lambda item: (item.sky_fraction + item.cloud_fraction * 0.5, item.width), reverse=True)
    selected: list[CropCandidate] = []
    for candidate in candidates:
        if len(selected) >= max_crops:
            break
        selected.append(candidate)
    return selected


def _write_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"Unable to write image: {path}")


def import_private_images(
    source_dir: str | Path,
    *,
    dataset_name: str,
    output_root: str | Path = "data/raw/images",
    derivative_metadata_path: str | Path = "data/raw/metadata/image_derivatives.parquet",
    copy_originals: bool = True,
    make_crops: bool = True,
    max_crops_per_image: int = 8,
    min_crop_size: int = 384,
    min_sky_fraction: float = 0.72,
    min_cloud_fraction: float = 0.08,
) -> pd.DataFrame:
    source_root = resolve_path(source_dir)
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    output = resolve_path(output_root) / _safe_dataset_name(dataset_name)
    originals_dir = output / "originals"
    crops_dir = output / "crops"
    rows: list[dict[str, object]] = []

    for source_path in sorted(source_root.rglob("*")):
        if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_rgb = read_rgb_image(source_path)
        source_sha256 = compute_sha256(source_path)
        source_image_id = source_sha256[:16]
        original_relative_path = project_relative_or_absolute(source_path)
        imported_relative_path = original_relative_path
        if copy_originals:
            original_name = f"{source_image_id}_{_safe_stem(source_path)}{source_path.suffix.lower()}"
            target_path = originals_dir / original_name
            if not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
            imported_relative_path = project_relative_or_absolute(target_path)
            rows.append(
                {
                    "relative_path": imported_relative_path,
                    "dataset_name": dataset_name,
                    "source_relative_path": original_relative_path,
                    "source_sha256": source_sha256,
                    "source_image_id": source_image_id,
                    "derivative_kind": "original",
                    "crop_x": None,
                    "crop_y": None,
                    "crop_width": image_rgb.shape[1],
                    "crop_height": image_rgb.shape[0],
                    "sky_fraction": None,
                    "cloud_fraction": None,
                }
            )
        if not make_crops:
            continue
        candidates = _crop_candidates(
            image_rgb,
            min_size=min_crop_size,
            scales=(0.35, 0.5, 0.7, 0.9, 1.0),
            stride_fraction=0.5,
            min_sky_fraction=min_sky_fraction,
            min_cloud_fraction=min_cloud_fraction,
            max_crops=max_crops_per_image,
        )
        for index, crop in enumerate(candidates, start=1):
            crop_image = image_rgb[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
            crop_name = f"{source_image_id}_crop{index:02d}_x{crop.x}_y{crop.y}_s{crop.width}.jpg"
            crop_path = crops_dir / crop_name
            _write_image(crop_path, crop_image)
            rows.append(
                {
                    "relative_path": project_relative_or_absolute(crop_path),
                    "dataset_name": dataset_name,
                    "source_relative_path": imported_relative_path,
                    "source_sha256": source_sha256,
                    "source_image_id": source_image_id,
                    "derivative_kind": "sky_crop",
                    "crop_x": crop.x,
                    "crop_y": crop.y,
                    "crop_width": crop.width,
                    "crop_height": crop.height,
                    "sky_fraction": crop.sky_fraction,
                    "cloud_fraction": crop.cloud_fraction,
                }
            )

    new_metadata = pd.DataFrame(rows, columns=DERIVATIVE_COLUMNS)
    existing = read_table(derivative_metadata_path)
    if not existing.empty:
        combined = pd.concat([existing, new_metadata], ignore_index=True)
        combined = combined.drop_duplicates(subset=["relative_path"], keep="last").reset_index(drop=True)
    else:
        combined = new_metadata
    write_table(combined, ensure_parent(derivative_metadata_path))
    return new_metadata
