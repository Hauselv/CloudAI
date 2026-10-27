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


def _file_signature(path: Path) -> tuple[str, float, int]:
    stat = path.stat()
    return (str(path), stat.st_mtime, stat.st_size)


def _cached_read_table(signature: tuple[str, float, int]) -> pd.DataFrame:
    import streamlit as st

    @st.cache_data(show_spinner=False, max_entries=64)
    def _read(sig: tuple[str, float, int]) -> pd.DataFrame:
        return read_table(Path(sig[0]))

    return _read(signature)


def safe_read_table(path_like: str | Path) -> pd.DataFrame:
    """Read a table, cached per (path, mtime, size) so pages do not re-read unchanged parquet files."""
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame()
    try:
        return _cached_read_table(_file_signature(path)).copy()
    except Exception:  # noqa: BLE001 - outside a Streamlit session the cache is unavailable
        return read_table(path)


def current_rater(cfg: dict[str, object] | None = None) -> str:
    """Rater chosen in the shared sidebar, else the configured default."""
    import streamlit as st

    default = str((cfg or app_config()).get("default_rater_id", "friend_a"))
    return str(st.session_state.get("rater_id", default) or default)


def human_rater_ids(dataset_cfg: dict[str, object]) -> list[str]:
    from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
    from cloud_aesthetics.judge.store import rater_kinds

    ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
    if ratings.empty:
        return []
    kinds = rater_kinds(dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv"))
    return sorted(r for r in ratings["rater_id"].astype(str).unique() if kinds.get(r, "human") == "human")


def shared_sidebar() -> None:
    """Rater selection and pipeline health, shown on every page above the page's own controls."""
    import streamlit as st

    cfg = app_config()
    dataset_cfg = dataset_config()
    default = str(cfg.get("default_rater_id", "friend_a"))
    options = human_rater_ids(dataset_cfg)
    if default not in options:
        options = [default] + options
    current = str(st.session_state.get("rater_id", default))
    if current not in options:
        options = options + [current]
    with st.sidebar:
        chosen = st.selectbox("Rater", options, index=options.index(current), key="sidebar_rater", help="Gilt für Bewerten, Inspect, Galerie, Quellgruppen, Rater-Analytik und Bericht.")
        st.session_state["rater_id"] = chosen
        status = _cached_pipeline_status()
        stale = [s["label"] for s in status if s["exists"] and s["stale"]]
        missing = [s["label"] for s in status if not s["exists"]]
        if stale:
            st.warning("Veraltet: " + ", ".join(stale), icon="⚠️")
        elif missing:
            st.info("Fehlt: " + ", ".join(missing), icon="⬜")
        else:
            st.caption("✅ Pipeline aktuell")
        st.divider()


def _cached_pipeline_status() -> list[dict[str, object]]:
    import streamlit as st

    @st.cache_data(show_spinner=False, ttl=120)
    def _status() -> list[dict[str, object]]:
        from cloud_aesthetics.pipeline import pipeline_status

        return [{"label": s.label, "exists": bool(s.exists), "stale": bool(s.stale)} for s in pipeline_status()]

    return _status()


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
