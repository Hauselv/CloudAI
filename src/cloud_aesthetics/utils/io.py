from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cloud_aesthetics.settings import ensure_parent, resolve_path


def write_json(data: dict[str, Any], path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return path


def read_json(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_table(frame: pd.DataFrame, path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def read_table(path_like: str | Path) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def append_csv_row(row: dict[str, Any], path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", index=False, header=False)
    else:
        frame.to_csv(path, index=False)
    return path
