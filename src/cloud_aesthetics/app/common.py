from __future__ import annotations

import sys
from importlib import import_module, reload
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from cloud_aesthetics.data.exclusions import active_excluded_ids
from cloud_aesthetics.data.manifest import ensure_kind_columns
from cloud_aesthetics.cli import (
    aggregate_labels_impl,
    explain_impl,
    extract_features_impl,
    import_images_impl,
    train_impl,
)
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


def active_manifest(path_like: str | Path) -> pd.DataFrame:
    manifest = safe_read_table(path_like)
    if manifest.empty:
        return manifest
    manifest = ensure_kind_columns(manifest)
    excluded_ids = active_excluded_ids()
    if not excluded_ids:
        return manifest
    return manifest[~manifest["image_id"].astype(str).isin(excluded_ids)].reset_index(drop=True)


def list_runs() -> list[str]:
    artifacts_root = resolve_path("data/artifacts")
    if not artifacts_root.exists():
        return []
    runs = [path for path in artifacts_root.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    runs.sort(key=lambda path: (path / "summary.json").stat().st_mtime, reverse=True)
    return [path.name for path in runs]


def _cli_attr(name: str) -> Any:
    module = import_module("cloud_aesthetics.cli")
    if not hasattr(module, name):
        module = reload(module)
    return getattr(module, name)


def analyze_batch_impl(*args: Any, **kwargs: Any) -> dict[str, object]:
    return _cli_attr("analyze_batch_impl")(*args, **kwargs)


def batch_predictions_path(*args: Any, **kwargs: Any) -> Path:
    return _cli_attr("batch_predictions_path")(*args, **kwargs)


def list_import_batches(*args: Any, **kwargs: Any) -> list[str]:
    return _cli_attr("list_import_batches")(*args, **kwargs)


__all__ = [
    "aggregate_labels_impl",
    "analyze_batch_impl",
    "app_config",
    "active_manifest",
    "batch_predictions_path",
    "dataset_config",
    "explain_impl",
    "extract_features_impl",
    "import_images_impl",
    "list_import_batches",
    "list_runs",
    "safe_read_table",
    "train_impl",
]
