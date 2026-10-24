"""On-disk cache for image embeddings.

One parquet file per model under ``data/processed/embeddings/<model_key>.parquet`` with
``image_id`` plus ``emb_000 ...`` columns, and a small ``<model_key>.json`` describing the
model. Extraction is incremental: only images without a cached row are encoded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from cloud_aesthetics.embeddings.backends import EmbeddingBackend, ModelSpec, create_backend
from cloud_aesthetics.settings import ensure_parent, load_yaml, resolve_path

DEFAULT_CONFIG = "configs/embeddings/default.yaml"


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def embedding_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[embedding_columns(frame)].to_numpy(dtype=np.float32)


def embedding_path(model_key: str, output_dir: str | Path = "data/processed/embeddings") -> Path:
    return resolve_path(output_dir) / f"{model_key}.parquet"


def load_embeddings(model_key: str, output_dir: str | Path = "data/processed/embeddings") -> pd.DataFrame:
    path = embedding_path(model_key, output_dir)
    if not path.exists():
        return pd.DataFrame(columns=["image_id"])
    frame = pd.read_parquet(path)
    frame["image_id"] = frame["image_id"].astype(str)
    return frame


def load_embedding_meta(model_key: str, output_dir: str | Path = "data/processed/embeddings") -> dict[str, object]:
    path = embedding_path(model_key, output_dir).with_suffix(".json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_specs(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, ModelSpec]:
    config = load_yaml(config_path)
    return {key: ModelSpec.from_config(key, dict(value)) for key, value in dict(config.get("models", {})).items()}


def _to_frame(image_ids: list[str], matrix: np.ndarray) -> pd.DataFrame:
    columns = [f"emb_{index:03d}" for index in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix.astype(np.float32), columns=columns)
    frame.insert(0, "image_id", [str(value) for value in image_ids])
    return frame


def extract_embeddings(
    manifest: pd.DataFrame,
    spec: ModelSpec,
    *,
    output_dir: str | Path = "data/processed/embeddings",
    batch_size: int = 16,
    backend: EmbeddingBackend | None = None,
    image_ids: Iterable[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Encode manifest images that are not cached yet and return the full cached table."""
    notify = progress or (lambda message: None)
    existing = load_embeddings(spec.key, output_dir)
    cached_ids = set(existing["image_id"]) if not existing.empty else set()
    frame = manifest.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    if image_ids is not None:
        wanted = {str(value) for value in image_ids}
        frame = frame[frame["image_id"].isin(wanted)]
    frame = frame[~frame["image_id"].isin(cached_ids)].drop_duplicates(subset=["image_id"])
    if frame.empty:
        notify(f"{spec.key}: all {len(cached_ids)} embeddings cached")
        return existing
    notify(f"{spec.key}: encoding {len(frame)} images ({len(cached_ids)} cached)")
    backend = backend or create_backend(spec, batch_size=batch_size)
    matrix = backend.encode_images(frame["relative_path"].tolist())
    new_rows = _to_frame(frame["image_id"].tolist(), matrix)
    combined = pd.concat([existing, new_rows], ignore_index=True) if not existing.empty else new_rows
    combined = combined.drop_duplicates(subset=["image_id"], keep="last").reset_index(drop=True)
    path = ensure_parent(embedding_path(spec.key, output_dir))
    combined.to_parquet(path, index=False)
    meta = {
        "model_key": spec.key,
        "backend": spec.backend,
        "architecture": spec.architecture,
        "pretrained": spec.pretrained,
        "image_size": spec.image_size,
        "dim": int(matrix.shape[1]),
        "rows": int(len(combined)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "supports_text": bool(getattr(backend, "supports_text", False)),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return combined


def extract_all_embeddings(
    manifest: pd.DataFrame,
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    only: Iterable[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    config = load_yaml(config_path)
    output_dir = config.get("output_dir", "data/processed/embeddings")
    batch_size = int(config.get("batch_size", 16))
    selected = set(only) if only else None
    report: dict[str, int] = {}
    for key, spec in model_specs(config_path).items():
        if selected and key not in selected:
            continue
        table = extract_embeddings(manifest, spec, output_dir=output_dir, batch_size=batch_size, progress=progress)
        report[key] = int(len(table))
    return report


def load_embedding_table(
    model_keys: list[str],
    output_dir: str | Path = "data/processed/embeddings",
    *,
    prefix_with_model: bool = True,
) -> pd.DataFrame:
    """Inner-join cached embeddings of several models into one wide table."""
    merged: pd.DataFrame | None = None
    for key in model_keys:
        frame = load_embeddings(key, output_dir)
        if frame.empty:
            raise FileNotFoundError(f"No cached embeddings for {key}. Run `extract-embeddings` first.")
        if prefix_with_model:
            frame = frame.rename(columns={column: f"emb_{key}_{column[4:]}" for column in embedding_columns(frame)})
        merged = frame if merged is None else merged.merge(frame, on="image_id", how="inner")
    return merged if merged is not None else pd.DataFrame(columns=["image_id"])
