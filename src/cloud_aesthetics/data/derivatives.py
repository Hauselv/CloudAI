"""Helpers around the importer's derivative metadata (originals and their crops)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cloud_aesthetics.preprocessing.image_ops import read_rgb_image, resize_long_edge
from cloud_aesthetics.settings import resolve_path
from cloud_aesthetics.utils.io import read_table

DERIVATIVES_PATH = "data/raw/metadata/image_derivatives.parquet"

# Crops covering less than this fraction of the source's short edge are detail crops
# without scene context; they are rated very differently from full photos.
SMALL_CROP_MAX_SCALE = 0.5
# Crops with fewer pixels than this on the short edge get upscaled on screen.
SMALL_CROP_MIN_EDGE = 768


@dataclass(frozen=True)
class CropContext:
    source_relative_path: str
    x: int
    y: int
    width: int
    height: int
    source_width: int | None
    source_height: int | None
    crop_scale: float | None


def load_derivatives(path_like: str | Path = DERIVATIVES_PATH) -> pd.DataFrame:
    frame = read_table(path_like)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["relative_path"] = frame["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    return frame


def crop_scale_table(derivatives: pd.DataFrame) -> pd.DataFrame:
    """Per crop path: scale relative to the source short edge (``crop_scale``)."""
    if derivatives.empty or "derivative_kind" not in derivatives.columns:
        return pd.DataFrame(columns=["relative_path", "crop_scale"])
    crops = derivatives[derivatives["derivative_kind"] == "sky_crop"].copy()
    if crops.empty:
        return pd.DataFrame(columns=["relative_path", "crop_scale"])
    if "crop_scale" in crops.columns and crops["crop_scale"].notna().all():
        return crops[["relative_path", "crop_scale"]]
    originals = derivatives[derivatives["derivative_kind"] == "original"][["source_image_id", "crop_width", "crop_height"]]
    originals = originals.rename(columns={"crop_width": "source_width", "crop_height": "source_height"}).drop_duplicates("source_image_id")
    merged = crops.merge(originals, on="source_image_id", how="left")
    short_edge = merged[["source_width", "source_height"]].min(axis=1)
    computed = merged["crop_width"] / short_edge
    if "crop_scale" in merged.columns:
        merged["crop_scale"] = merged["crop_scale"].fillna(computed)
    else:
        merged["crop_scale"] = computed
    return merged[["relative_path", "crop_scale"]]


def classify_crop(kind: object, crop_scale: object, short_edge: object) -> str:
    """``original`` | ``large_crop`` | ``small_crop``."""
    if str(kind) != "crop":
        return "original"
    try:
        scale = float(crop_scale)
    except (TypeError, ValueError):
        scale = float("nan")
    try:
        edge = float(short_edge)
    except (TypeError, ValueError):
        edge = float("nan")
    if (scale == scale and scale < SMALL_CROP_MAX_SCALE) or (edge == edge and edge < SMALL_CROP_MIN_EDGE):
        return "small_crop"
    return "large_crop"


def crop_context(relative_path: str, derivatives: pd.DataFrame | None = None) -> CropContext | None:
    """Return where a crop sits inside its source original, or None for non-crops."""
    frame = derivatives if derivatives is not None else load_derivatives()
    if frame.empty or "relative_path" not in frame.columns:
        return None
    paths = frame["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    key = str(relative_path).replace("\\", "/")
    matches = frame[(paths == key) & (frame["derivative_kind"] == "sky_crop")]
    if matches.empty:
        return None
    row = matches.iloc[0]
    source_path = str(row["source_relative_path"]).replace("\\", "/")
    source = frame[(paths == source_path) & (frame["derivative_kind"] == "original")]
    source_width = int(source.iloc[0]["crop_width"]) if not source.empty else None
    source_height = int(source.iloc[0]["crop_height"]) if not source.empty else None
    scale = None
    if source_width and source_height:
        scale = float(row["crop_width"]) / float(min(source_width, source_height))
    return CropContext(
        source_relative_path=source_path,
        x=int(row["crop_x"]),
        y=int(row["crop_y"]),
        width=int(row["crop_width"]),
        height=int(row["crop_height"]),
        source_width=source_width,
        source_height=source_height,
        crop_scale=scale,
    )


def render_crop_context(context: CropContext, max_edge: int = 1200) -> np.ndarray | None:
    """Downscaled source image with the crop rectangle drawn on it (RGB), or None if missing."""
    path = resolve_path(context.source_relative_path)
    if not path.exists():
        return None
    image = read_rgb_image(path)
    height, width = image.shape[:2]
    small = resize_long_edge(image, max_edge)
    factor = small.shape[1] / width
    x0, y0 = int(round(context.x * factor)), int(round(context.y * factor))
    x1 = int(round((context.x + context.width) * factor))
    y1 = int(round((context.y + context.height) * factor))
    canvas = small.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    mask = np.ones(canvas.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = False
    canvas[mask] = (0.55 * canvas[mask] + 0.45 * overlay[mask]).astype(np.uint8)
    thickness = max(2, canvas.shape[1] // 300)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 200, 40), thickness)
    return canvas
