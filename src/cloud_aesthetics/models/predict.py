"""Score every image with a trained tabular run (baseline features or embeddings).

Used to extend predictions from the labeled set to all images so the labeling queue,
Triage and Gallery can show what the model expects and where models disagree.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cloud_aesthetics.explain.feature_importance import load_pickled_model
from cloud_aesthetics.settings import ensure_parent, resolve_path
from cloud_aesthetics.utils.io import read_table

PREDICTIONS_DIR = "data/processed/predictions"


def _run_dir(run_id: str | Path) -> Path:
    candidate = resolve_path(run_id)
    if candidate.exists():
        return candidate
    return resolve_path("data/artifacts") / str(run_id)


def _input_table(summary: dict[str, object], manifest: pd.DataFrame) -> pd.DataFrame:
    kind = str(summary.get("kind", ""))
    if kind == "embedding":
        from cloud_aesthetics.embeddings.zero_shot_store import load_zero_shot
        from cloud_aesthetics.features.store import load_features
        from cloud_aesthetics.models.embedding import build_embedding_input_table

        config = {
            "embedding_models": summary.get("embedding_models", ["clip_vit_b32"]),
            "embeddings_dir": summary.get("embeddings_dir", "data/processed/embeddings"),
            "include_features": summary.get("include_features", False),
            "include_zero_shot": summary.get("include_zero_shot", False),
        }
        features = load_features(summary.get("config", {}).get("features_path", "data/processed/features/features_v1.parquet"))
        return build_embedding_input_table(config, features=features, zero_shot=load_zero_shot(str(config["embeddings_dir"])))
    if kind == "baseline":
        from cloud_aesthetics.features.store import load_features

        return load_features(summary.get("config", {}).get("features_path", "data/processed/features/features_v1.parquet"))
    raise ValueError(f"predict_all supports baseline and embedding runs, not {kind!r}")


def predict_all(run_id: str | Path, manifest: pd.DataFrame, *, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per manifest image with a prediction per model, their mean and spread."""
    run_dir = _run_dir(run_id)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    table = _input_table(summary, manifest)
    table["image_id"] = table["image_id"].astype(str)
    wanted = set(manifest["image_id"].astype(str))
    table = table[table["image_id"].isin(wanted)].drop_duplicates("image_id").reset_index(drop=True)
    columns = [column for column in summary.get("input_columns", []) if column in table.columns]
    if not columns:
        raise ValueError("Run summary has no input_columns that match the current tables.")
    out = pd.DataFrame({"image_id": table["image_id"]})
    model_names = []
    for model_name, info in summary.get("models", {}).items():
        model = load_pickled_model(info["model_path"])
        out[f"pred_{model_name}"] = np.asarray(model.predict(table[columns]), dtype=float)
        model_names.append(model_name)
    pred_columns = [f"pred_{name}" for name in model_names]
    out["pred_mean"] = out[pred_columns].mean(axis=1)
    out["pred_spread"] = out[pred_columns].max(axis=1) - out[pred_columns].min(axis=1) if len(pred_columns) > 1 else 0.0
    train_ids = {str(value) for value in summary.get("train_image_ids", [])}
    out["in_training_set"] = out["image_id"].isin(train_ids)
    if labels is not None and not labels.empty:
        human = labels.copy()
        human["image_id"] = human["image_id"].astype(str)
        out = out.merge(human[["image_id", "mean_score"]].rename(columns={"mean_score": "human_score"}).drop_duplicates("image_id"), on="image_id", how="left")
    out["run_id"] = run_dir.name
    out["predicted_at"] = datetime.now(timezone.utc).isoformat()
    return out


def save_predictions(frame: pd.DataFrame, run_id: str, output_dir: str | Path = PREDICTIONS_DIR) -> Path:
    path = ensure_parent(resolve_path(output_dir) / f"{run_id}.parquet")
    frame.to_parquet(path, index=False)
    latest = resolve_path(output_dir) / "latest.parquet"
    frame.to_parquet(latest, index=False)
    return path


def load_latest_predictions(output_dir: str | Path = PREDICTIONS_DIR) -> pd.DataFrame:
    frame = read_table(resolve_path(output_dir) / "latest.parquet")
    if frame.empty:
        return frame
    frame["image_id"] = frame["image_id"].astype(str)
    return frame
