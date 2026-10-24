"""Capture context table: time, place, sun and weather per image.

Originals are read from EXIF / file names; crops inherit the context of their source
original through the importer's derivative metadata. Weather lookups are optional and
cached; everything else is offline.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from cloud_aesthetics.data.derivatives import load_derivatives
from cloud_aesthetics.features.capture import capture_info
from cloud_aesthetics.features.solar import PHASES, sun_position
from cloud_aesthetics.features.weather import fetch_archive_day, weather_features
from cloud_aesthetics.settings import ensure_parent, resolve_path

CONTEXT_PATH = "data/processed/context/capture_context.parquet"


def _source_map(derivatives: pd.DataFrame) -> dict[str, str]:
    """crop relative_path -> source original relative_path (normalized)."""
    if derivatives.empty or "derivative_kind" not in derivatives.columns:
        return {}
    crops = derivatives[derivatives["derivative_kind"] == "sky_crop"]
    return {
        str(row["relative_path"]).replace("\\", "/"): str(row["source_relative_path"]).replace("\\", "/")
        for _, row in crops.iterrows()
    }


def context_for_original(
    relative_path: str,
    *,
    default_latitude: float | None,
    default_longitude: float | None,
    weather: bool,
    weather_getter: Callable[..., object] | None = None,
) -> dict[str, object]:
    info = capture_info(relative_path, default_latitude=default_latitude, default_longitude=default_longitude)
    row: dict[str, object] = {
        **info.to_dict(),
        "sun_elevation_deg": None,
        "sun_azimuth_deg": None,
        "sun_phase": "unknown",
        "hours_from_sunset": None,
        "month": None,
        "hour_local_estimate": None,
    }
    for phase in PHASES:
        row[f"sun_phase_{phase}"] = None
    capture_utc = datetime.fromisoformat(info.datetime_utc) if info.datetime_utc else None
    if info.date:
        row["month"] = int(info.date[5:7])
    if capture_utc is not None and info.latitude is not None and info.longitude is not None:
        sun = sun_position(capture_utc, info.latitude, info.longitude)
        row.update({"sun_elevation_deg": sun.elevation_deg, "sun_azimuth_deg": sun.azimuth_deg, "sun_phase": sun.phase, "hours_from_sunset": sun.hours_from_sunset})
        for phase in PHASES:
            row[f"sun_phase_{phase}"] = 1.0 if phase == sun.phase else 0.0
        row["hour_local_estimate"] = (capture_utc.hour + capture_utc.minute / 60.0 + round(info.longitude / 15.0)) % 24.0
    if weather and info.date and info.latitude is not None and info.longitude is not None:
        payload = fetch_archive_day(info.latitude, info.longitude, info.date, getter=weather_getter)
        row.update(weather_features(payload, capture_utc))
        row["weather_available"] = payload is not None
    else:
        row.update(weather_features(None, None))
        row["weather_available"] = False
    return row


def build_context_table(
    manifest: pd.DataFrame,
    *,
    default_latitude: float | None = None,
    default_longitude: float | None = None,
    weather: bool = True,
    derivatives: pd.DataFrame | None = None,
    existing: pd.DataFrame | None = None,
    weather_getter: Callable[..., object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Context for every manifest image; originals are computed once, crops copy their source."""
    notify = progress or (lambda message: None)
    derivatives = derivatives if derivatives is not None else load_derivatives()
    source_map = _source_map(derivatives)
    frame = manifest.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["_path"] = frame["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    kinds = frame["kind"].astype(str) if "kind" in frame.columns else pd.Series("original", index=frame.index)
    cache: dict[str, dict[str, object]] = {}
    if existing is not None and not existing.empty and "relative_path" in existing.columns:
        for _, row in existing.iterrows():
            cache[str(row["relative_path"]).replace("\\", "/")] = row.drop(labels=[c for c in ("image_id", "relative_path", "context_source") if c in row.index]).to_dict()
    rows = []
    originals = frame[kinds != "crop"]
    for index, (_, row) in enumerate(originals.iterrows(), start=1):
        path = row["_path"]
        if path not in cache:
            if index % 25 == 0:
                notify(f"context: {index}/{len(originals)} originals")
            cache[path] = context_for_original(path, default_latitude=default_latitude, default_longitude=default_longitude, weather=weather, weather_getter=weather_getter)
        rows.append({"image_id": row["image_id"], "relative_path": path, "context_source": "self", **cache[path]})
    for _, row in frame[kinds == "crop"].iterrows():
        source = source_map.get(row["_path"])
        if source and source not in cache and resolve_path(source).exists():
            cache[source] = context_for_original(source, default_latitude=default_latitude, default_longitude=default_longitude, weather=weather, weather_getter=weather_getter)
        if source and source in cache:
            rows.append({"image_id": row["image_id"], "relative_path": row["_path"], "context_source": "source_original", **cache[source]})
        else:
            rows.append({"image_id": row["image_id"], "relative_path": row["_path"], "context_source": "none", **context_for_original(row["_path"], default_latitude=default_latitude, default_longitude=default_longitude, weather=False)})
    return pd.DataFrame(rows)


def save_context(frame: pd.DataFrame, path_like: str | Path = CONTEXT_PATH) -> Path:
    path = ensure_parent(path_like)
    frame.to_parquet(path, index=False)
    return path


def load_context(path_like: str | Path = CONTEXT_PATH) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["image_id"] = frame["image_id"].astype(str)
    return frame
