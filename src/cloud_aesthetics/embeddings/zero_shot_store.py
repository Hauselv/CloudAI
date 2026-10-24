"""Compute and cache the CLIP zero-shot attribute table for all manifest images."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from cloud_aesthetics.embeddings.backends import create_backend
from cloud_aesthetics.embeddings.store import DEFAULT_CONFIG, embedding_matrix, extract_embeddings, load_embeddings, model_specs
from cloud_aesthetics.embeddings.zero_shot import ZeroShotTaxonomy, build_text_bank, zero_shot_table
from cloud_aesthetics.settings import ensure_parent, load_yaml, resolve_path

ZERO_SHOT_FILENAME = "clip_zero_shot.parquet"


def zero_shot_path(output_dir: str | Path = "data/processed/embeddings") -> Path:
    return resolve_path(output_dir) / ZERO_SHOT_FILENAME


def load_zero_shot(output_dir: str | Path = "data/processed/embeddings") -> pd.DataFrame:
    path = zero_shot_path(output_dir)
    if not path.exists():
        return pd.DataFrame(columns=["image_id"])
    frame = pd.read_parquet(path)
    frame["image_id"] = frame["image_id"].astype(str)
    return frame


def compute_zero_shot_table(
    manifest: pd.DataFrame,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    model_key: str = "clip_vit_b32",
    taxonomy: ZeroShotTaxonomy | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """(Re)compute zero-shot attributes for every manifest image using cached CLIP embeddings.

    The table is recomputed as a whole because the concept z-scores depend on the dataset.
    Missing image embeddings are computed on the fly.
    """
    notify = progress or (lambda message: None)
    config = load_yaml(config_path)
    output_dir = str(config.get("output_dir", "data/processed/embeddings"))
    specs = model_specs(config_path)
    if model_key not in specs:
        raise KeyError(f"Model {model_key} is not defined in {config_path}")
    spec = specs[model_key]
    backend = create_backend(spec, batch_size=int(config.get("batch_size", 16)))
    if not getattr(backend, "supports_text", False):
        raise ValueError(f"{model_key} has no text encoder; zero-shot needs a CLIP-style model")
    embeddings = extract_embeddings(manifest, spec, output_dir=output_dir, backend=backend, progress=notify)
    wanted = set(manifest["image_id"].astype(str))
    embeddings = embeddings[embeddings["image_id"].isin(wanted)].reset_index(drop=True)
    notify(f"{model_key}: encoding text prompts")
    bank = build_text_bank(backend, taxonomy)
    notify(f"{model_key}: scoring {len(embeddings)} images")
    table = zero_shot_table(embeddings["image_id"].tolist(), embedding_matrix(embeddings), bank)
    table.to_parquet(ensure_parent(zero_shot_path(output_dir)), index=False)
    return table


def zero_shot_row(image_id: str, output_dir: str | Path = "data/processed/embeddings") -> pd.Series:
    table = load_zero_shot(output_dir)
    if table.empty:
        return pd.Series(dtype=object)
    matches = table[table["image_id"] == str(image_id)]
    return matches.iloc[0] if not matches.empty else pd.Series(dtype=object)
