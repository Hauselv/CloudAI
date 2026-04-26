from __future__ import annotations

from pathlib import Path

import pandas as pd

from cloud_aesthetics.settings import ensure_parent, resolve_path


def save_features(frame: pd.DataFrame, path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    frame.to_parquet(path, index=False)
    return path


def load_features(path_like: str | Path) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
