from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from cloud_aesthetics.cli import aggregate_labels_impl, explain_impl, extract_features_impl, import_images_impl, train_impl
from cloud_aesthetics.settings import load_yaml, resolve_path
from cloud_aesthetics.utils.io import read_table


def dataset_config() -> dict[str, object]:
    return load_yaml("configs/dataset/default.yaml")


def app_config() -> dict[str, object]:
    return load_yaml("configs/app/default.yaml")


def safe_read_table(path_like: str | Path) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame()
    return read_table(path)


def list_runs() -> list[str]:
    artifacts_root = resolve_path("data/artifacts")
    if not artifacts_root.exists():
        return []
    return sorted([path.name for path in artifacts_root.iterdir() if path.is_dir()], reverse=True)


__all__ = [
    "aggregate_labels_impl",
    "app_config",
    "dataset_config",
    "explain_impl",
    "extract_features_impl",
    "import_images_impl",
    "list_runs",
    "safe_read_table",
    "train_impl",
]
