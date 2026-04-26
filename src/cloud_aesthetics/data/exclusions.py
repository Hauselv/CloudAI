from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cloud_aesthetics.settings import ensure_parent, resolve_path


EXCLUSION_COLUMNS = ["image_id", "relative_path", "excluded", "reason", "timestamp"]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_exclusions(path_like: str | Path = "data/raw/metadata/exclusions.csv") -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame(columns=EXCLUSION_COLUMNS)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        for values in reader:
            if not values:
                continue
            if header == ["image_id", "excluded", "reason", "timestamp"] and len(values) == 4:
                image_id, excluded, reason, timestamp = values
                relative_path = ""
                if " | " in reason:
                    reason, relative_path = reason.split(" | ", 1)
                rows.append(
                    {
                        "image_id": image_id,
                        "relative_path": relative_path,
                        "excluded": excluded,
                        "reason": reason,
                        "timestamp": timestamp,
                    }
                )
            elif len(values) >= 5:
                image_id, relative_path, excluded, reason = values[:4]
                timestamp = ",".join(values[4:])
                rows.append(
                    {
                        "image_id": image_id,
                        "relative_path": relative_path,
                        "excluded": excluded,
                        "reason": reason,
                        "timestamp": timestamp,
                    }
                )
    return pd.DataFrame(rows, columns=EXCLUSION_COLUMNS)


def active_excluded_ids(path_like: str | Path = "data/raw/metadata/exclusions.csv") -> set[str]:
    frame = load_exclusions(path_like)
    if frame.empty:
        return set()
    frame["excluded"] = frame["excluded"].map(_as_bool)
    latest = frame.dropna(subset=["image_id"]).drop_duplicates(subset=["image_id"], keep="last")
    return set(latest.loc[latest["excluded"], "image_id"].astype(str))


def set_exclusion(
    image_id: str,
    *,
    excluded: bool = True,
    reason: str = "",
    relative_path: str = "",
    path_like: str | Path = "data/raw/metadata/exclusions.csv",
) -> dict[str, object]:
    row = {
        "image_id": str(image_id),
        "relative_path": relative_path,
        "excluded": bool(excluded),
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = ensure_parent(path_like)
    existing = load_exclusions(path) if path.exists() else pd.DataFrame(columns=EXCLUSION_COLUMNS)
    frame = pd.DataFrame([row], columns=EXCLUSION_COLUMNS)
    pd.concat([existing, frame], ignore_index=True).to_csv(path, index=False)
    return row
