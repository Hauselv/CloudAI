"""Regressors on cached image embeddings (CLIP, DINOv2, ...).

The embeddings are precomputed once (``embeddings.store``), so these models train in
seconds on CPU and can be cross-validated properly. ``include_features`` optionally
appends the hand-crafted feature table, ``include_zero_shot`` the CLIP zero-shot
attributes, so the same loop also trains the hybrid variants.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from cloud_aesthetics.embeddings.store import load_embedding_table
from cloud_aesthetics.models.tabular import GROUP_COLUMNS, train_tabular_suite

RIDGE_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def _make_estimators(config: dict[str, object]) -> dict[str, Pipeline]:
    knn_k = int(config.get("knn_neighbors", 10))
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=RIDGE_ALPHAS)),
            ]
        ),
        "knn": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", KNeighborsRegressor(n_neighbors=knn_k, weights="distance", metric="cosine")),
            ]
        ),
        "svr": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(C=float(config.get("svr_c", 3.0)), epsilon=0.3)),
            ]
        ),
    }


def build_embedding_input_table(
    config: dict[str, object],
    *,
    features: pd.DataFrame | None = None,
    zero_shot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join the cached embeddings (and optional extra tables) into one input table."""
    model_keys = [str(key) for key in config.get("embedding_models", ["clip_vit_b32"])]
    output_dir = str(config.get("embeddings_dir", "data/processed/embeddings"))
    table = load_embedding_table(model_keys, output_dir)
    if bool(config.get("include_features", False)) and features is not None and not features.empty:
        extra = features.drop(columns=[column for column in GROUP_COLUMNS if column in features.columns]).copy()
        extra["image_id"] = extra["image_id"].astype(str)
        numeric = ["image_id"] + [column for column in extra.columns if column != "image_id" and pd.api.types.is_numeric_dtype(extra[column])]
        table = table.merge(extra[numeric], on="image_id", how="inner")
    if bool(config.get("include_zero_shot", False)) and zero_shot is not None and not zero_shot.empty:
        extra = zero_shot.copy()
        extra["image_id"] = extra["image_id"].astype(str)
        numeric = ["image_id"] + [column for column in extra.columns if column != "image_id" and pd.api.types.is_numeric_dtype(extra[column])]
        table = table.merge(extra[numeric], on="image_id", how="inner")
    return table


def embedding_input_row(
    image_id: str,
    summary: dict[str, object],
    *,
    manifest: pd.DataFrame,
    features: pd.DataFrame | None = None,
    zero_shot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One-row input table for an image, computing missing embeddings on demand."""
    from cloud_aesthetics.embeddings.store import extract_embeddings, model_specs

    config = {
        "embedding_models": summary.get("embedding_models", ["clip_vit_b32"]),
        "embeddings_dir": summary.get("embeddings_dir", "data/processed/embeddings"),
        "include_features": summary.get("include_features", False),
        "include_zero_shot": summary.get("include_zero_shot", False),
    }
    specs = model_specs()
    for key in config["embedding_models"]:
        if key in specs:
            extract_embeddings(manifest, specs[key], output_dir=str(config["embeddings_dir"]), image_ids=[str(image_id)])
    table = build_embedding_input_table(config, features=features, zero_shot=zero_shot)
    row = table[table["image_id"] == str(image_id)]
    if row.empty:
        raise ValueError(f"No embedding input available for image {image_id}")
    columns = [column for column in summary.get("input_columns", []) if column in row.columns]
    return row[columns].reset_index(drop=True) if columns else row.drop(columns=["image_id"]).reset_index(drop=True)


def labeled_embedding_neighbors(
    image_id: str,
    summary: dict[str, object],
    labels: pd.DataFrame,
    *,
    top_k: int = 6,
) -> pd.DataFrame:
    """Most similar *labeled* training images in embedding space, with their human scores."""
    from cloud_aesthetics.embeddings.store import embedding_matrix, load_embeddings

    model_keys = [str(key) for key in summary.get("embedding_models", ["clip_vit_b32"])]
    output_dir = str(summary.get("embeddings_dir", "data/processed/embeddings"))
    train_ids = {str(value) for value in summary.get("train_image_ids", [])}
    frames = []
    for key in model_keys:
        frame = load_embeddings(key, output_dir)
        if frame.empty:
            continue
        frames.append(frame.set_index("image_id"))
    if not frames:
        return pd.DataFrame(columns=["image_id", "similarity", "mean_score"])
    joined = pd.concat(frames, axis=1, join="inner")
    if str(image_id) not in joined.index:
        return pd.DataFrame(columns=["image_id", "similarity", "mean_score"])
    matrix = embedding_matrix(joined.reset_index())
    norms = (matrix**2).sum(axis=1) ** 0.5
    norms[norms == 0] = 1.0
    matrix = matrix / norms[:, None]
    ids = joined.index.astype(str).to_numpy()
    query = matrix[ids == str(image_id)][0]
    sims = matrix @ query
    result = pd.DataFrame({"image_id": ids, "similarity": sims})
    result = result[(result["image_id"] != str(image_id)) & result["image_id"].isin(train_ids)]
    labels = labels.copy()
    labels["image_id"] = labels["image_id"].astype(str)
    result = result.merge(labels[["image_id", "mean_score"]], on="image_id", how="left")
    return result.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


def train_embedding_suite(
    table: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    model_names = [str(name) for name in config.get("regressors", ["ridge", "knn"])]
    estimators = _make_estimators(config)
    selected = {name: estimators[name] for name in model_names}
    extra = {
        "embedding_models": [str(key) for key in config.get("embedding_models", ["clip_vit_b32"])],
        "embeddings_dir": str(config.get("embeddings_dir", "data/processed/embeddings")),
        "include_features": bool(config.get("include_features", False)),
        "include_zero_shot": bool(config.get("include_zero_shot", False)),
    }
    summary, _ = train_tabular_suite(table, labels, splits, config, selected, kind="embedding", extra_summary=extra)
    return summary
