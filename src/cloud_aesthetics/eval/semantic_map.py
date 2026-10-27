"""Semantic map: a 2-D projection of the image embeddings with everything we know per image.

UMAP (or PCA as the dependency-free fallback) places every image on a plane where
neighbours look alike to CLIP/DINOv2. Colouring the plane by human score, judge score,
model prediction, cloud type or "labeled yes/no" shows where the 10s cluster and which
visual regions have no labels yet. ``map_regions`` turns that into a table: k-means
regions with label coverage, mean scores and example images, so the gaps become a list
of candidate images rather than an impression.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cloud_aesthetics.embeddings.store import embedding_matrix, load_embeddings
from cloud_aesthetics.settings import resolve_path

PROJECTION_DIR = "data/processed/embeddings"
MAP_QUEUE_PATH = "data/processed/embeddings/map_queue.parquet"


@dataclass(frozen=True)
class ProjectionParams:
    model_key: str = "clip_vit_b32"
    method: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    seed: int = 42

    @property
    def stem(self) -> str:
        return f"projection_{self.model_key}_{self.method}"


def compute_projection(matrix: np.ndarray, params: ProjectionParams) -> np.ndarray:
    """2-D coordinates for every row of ``matrix``; UMAP if requested and available, else PCA."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] < 3:
        return np.zeros((matrix.shape[0], 2), dtype=np.float32)
    if params.method == "umap":
        try:
            import umap  # noqa: WPS433 (optional heavy dependency)
        except ImportError:  # pragma: no cover - exercised only without umap-learn
            return compute_projection(matrix, ProjectionParams(params.model_key, "pca", seed=params.seed))
        n_neighbors = int(min(params.n_neighbors, max(2, matrix.shape[0] - 1)))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=params.min_dist, n_components=2, metric="cosine", random_state=params.seed)
        return reducer.fit_transform(matrix).astype(np.float32)
    if params.method == "pca":
        return PCA(n_components=2, random_state=params.seed).fit_transform(matrix).astype(np.float32)
    raise ValueError(f"Unknown projection method {params.method!r}")


def projection_path(params: ProjectionParams, output_dir: str | Path = PROJECTION_DIR) -> Path:
    return resolve_path(output_dir) / f"{params.stem}.parquet"


def load_projection(params: ProjectionParams, output_dir: str | Path = PROJECTION_DIR) -> pd.DataFrame:
    path = projection_path(params, output_dir)
    if not path.exists():
        return pd.DataFrame(columns=["image_id", "x", "y"])
    frame = pd.read_parquet(path)
    frame["image_id"] = frame["image_id"].astype(str)
    return frame


def build_projection(params: ProjectionParams, *, output_dir: str | Path = PROJECTION_DIR, force: bool = False, embeddings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Project the cached embeddings of ``params.model_key`` and store the coordinates next to them.

    The projection is recomputed when forced or when the set of embedded images changed.
    """
    frame = embeddings if embeddings is not None else load_embeddings(params.model_key, output_dir)
    if frame.empty:
        raise ValueError(f"No cached embeddings for {params.model_key!r}. Run `extract-embeddings` first.")
    ids = frame["image_id"].astype(str).tolist()
    existing = load_projection(params, output_dir)
    if not force and not existing.empty and set(existing["image_id"]) == set(ids):
        return existing
    coords = compute_projection(embedding_matrix(frame), params)
    result = pd.DataFrame({"image_id": ids, "x": coords[:, 0], "y": coords[:, 1]})
    path = projection_path(params, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({**asdict(params), "n_images": len(result)}, indent=2), encoding="utf-8")
    return result


def load_judge_tables(root: str | Path = "data/processed/judge") -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    path = resolve_path(root)
    if path.exists():
        for file in sorted(path.glob("*_attributes.parquet")):
            tables[file.stem.replace("_attributes", "")] = pd.read_parquet(file)
    return tables


def build_map_table(
    projection: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    labels: pd.DataFrame | None = None,
    zero_shot: pd.DataFrame | None = None,
    judge_tables: dict[str, pd.DataFrame] | None = None,
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per projected image with coordinates and every per-image signal we have."""
    table = projection.copy()
    table["image_id"] = table["image_id"].astype(str)
    groups = manifest.copy()
    groups["image_id"] = groups["image_id"].astype(str)
    keep = [column for column in ("relative_path", "kind", "crop_class", "dataset_name", "crop_scale", "capture_date") if column in groups.columns]
    table = table.merge(groups[["image_id"] + keep].drop_duplicates("image_id"), on="image_id", how="inner")
    if labels is not None and not labels.empty:
        part = labels.copy()
        part["image_id"] = part["image_id"].astype(str)
        columns = [column for column in ("mean_score", "n_raters", "bt_score") if column in part.columns]
        table = table.merge(part[["image_id"] + columns].drop_duplicates("image_id"), on="image_id", how="left")
    if "mean_score" not in table.columns:
        table["mean_score"] = np.nan
    table["labeled"] = table["mean_score"].notna()
    if zero_shot is not None and not zero_shot.empty:
        part = zero_shot.copy()
        part["image_id"] = part["image_id"].astype(str)
        columns = [column for column in ("clip_cloud_type", "clip_light", "clip_cloud_type_probability", "clip_concept_spectacular_sky", "clip_concept_soft_haze") if column in part.columns]
        table = table.merge(part[["image_id"] + columns].drop_duplicates("image_id"), on="image_id", how="left")
    if judge_tables:
        frames = []
        for rater, part in judge_tables.items():
            if part is None or part.empty:
                continue
            sub = part[["image_id", "judge_score"]].copy()
            sub["image_id"] = sub["image_id"].astype(str)
            frames.append(sub.drop_duplicates("image_id"))
        if frames:
            judge = pd.concat(frames).groupby("image_id")["judge_score"].mean().rename("judge_mean").reset_index()
            table = table.merge(judge, on="image_id", how="left")
    if predictions is not None and not predictions.empty and "pred_mean" in predictions.columns:
        part = predictions[["image_id", "pred_mean"]].copy()
        part["image_id"] = part["image_id"].astype(str)
        table = table.merge(part.drop_duplicates("image_id").rename(columns={"pred_mean": "model_score"}), on="image_id", how="left")
    for column in ("judge_mean", "model_score", "clip_cloud_type", "clip_light"):
        if column not in table.columns:
            table[column] = np.nan
    table["estimate"] = table["mean_score"].fillna(table["model_score"]).fillna(table["judge_mean"])
    return table.reset_index(drop=True)


def map_regions(table: pd.DataFrame, *, n_regions: int = 12, seed: int = 42, examples: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means regions on the 2-D map with label coverage, scores and example images per region.

    ``gap_score`` ranks regions by how many images sit there without any label, weighted by
    how promising the region looks (judge/model estimate), so the top rows are the visual
    neighbourhoods where new labels teach the model the most.
    """
    table = table.copy()
    if table.empty:
        return table, pd.DataFrame()
    k = int(max(1, min(n_regions, len(table))))
    coords = table[["x", "y"]].to_numpy(dtype=np.float32)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(coords)
    table["region"] = model.labels_.astype(int)
    rows = []
    for region, part in table.groupby("region"):
        centre = model.cluster_centers_[region]
        distance = np.linalg.norm(part[["x", "y"]].to_numpy(dtype=np.float32) - centre, axis=1)
        part = part.assign(_distance=distance).sort_values("_distance")
        labeled = part[part["labeled"]]
        unlabeled = part[~part["labeled"]]
        originals = part[part["kind"].astype(str) == "original"] if "kind" in part.columns else part
        estimate = part["estimate"].astype(float)
        rows.append(
            {
                "region": int(region),
                "n": int(len(part)),
                "n_labeled": int(len(labeled)),
                "labeled_share": float(len(labeled) / len(part)),
                "n_originals": int(len(originals)),
                "mean_score": float(labeled["mean_score"].mean()) if not labeled.empty else np.nan,
                "max_score": float(labeled["mean_score"].max()) if not labeled.empty else np.nan,
                "n_top": int((labeled["mean_score"] >= 9.0).sum()),
                "judge_mean": float(part["judge_mean"].astype(float).mean()) if part["judge_mean"].notna().any() else np.nan,
                "model_mean": float(part["model_score"].astype(float).mean()) if part["model_score"].notna().any() else np.nan,
                "estimate_mean": float(estimate.mean()) if estimate.notna().any() else np.nan,
                "top_cloud_type": str(part["clip_cloud_type"].mode().iloc[0]) if part["clip_cloud_type"].notna().any() else "",
                "top_light": str(part["clip_light"].mode().iloc[0]) if part["clip_light"].notna().any() else "",
                "share_original": float((part["kind"].astype(str) == "original").mean()) if "kind" in part.columns else np.nan,
                "x": float(centre[0]),
                "y": float(centre[1]),
                "example_ids": part["image_id"].head(examples).tolist(),
                "unlabeled_candidates": (unlabeled[unlabeled["kind"].astype(str) == "original"] if "kind" in unlabeled.columns else unlabeled)["image_id"].head(examples * 2).tolist(),
            }
        )
    regions = pd.DataFrame(rows)
    promise = regions["estimate_mean"].fillna(regions["estimate_mean"].mean()).fillna(5.0) / 10.0
    regions["gap_score"] = (regions["n"] - regions["n_labeled"]) * (0.5 + promise)
    regions = regions.sort_values("gap_score", ascending=False).reset_index(drop=True)
    return table, regions


def thumbnail_data_uri(path_like: str | Path, size: int = 64) -> str | None:
    """Small JPEG thumbnail as a data URI for drawing images onto the Plotly map."""
    from PIL import Image

    path = resolve_path(path_like)
    if not path.exists():
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def map_summary(table: pd.DataFrame, regions: pd.DataFrame) -> dict[str, object]:
    top = table[table["mean_score"] >= 9.0] if "mean_score" in table.columns else pd.DataFrame()
    return {
        "n_images": int(len(table)),
        "n_labeled": int(table["labeled"].sum()),
        "n_regions": int(len(regions)),
        "regions_without_labels": int((regions["n_labeled"] == 0).sum()) if not regions.empty else 0,
        "unlabeled_in_top3_gaps": int((regions.head(3)["n"] - regions.head(3)["n_labeled"]).sum()) if not regions.empty else 0,
        "regions_with_tens": int((regions["n_top"] > 0).sum()) if not regions.empty else 0,
        "top_images_region_spread": int(top["region"].nunique()) if not top.empty and "region" in top.columns else 0,
    }


def gap_label_queue(table: pd.DataFrame, regions: pd.DataFrame, *, per_region: int = 4) -> pd.DataFrame:
    """Label order that fills the map's gaps: unlabeled originals, regions by gap score, round-robin.

    Within a region the most promising images (human estimate, else model, else judge) come
    first; ``per_region`` images are taken from each region before moving to the next, so a
    labeling session covers several visual neighbourhoods instead of one.
    """
    if table.empty or regions.empty or "region" not in table.columns:
        return pd.DataFrame(columns=["image_id", "region", "map_rank", "map_reason"])
    open_images = table[~table["labeled"]]
    if "kind" in open_images.columns:
        open_images = open_images[open_images["kind"].astype(str) == "original"]
    buckets: list[tuple[dict[str, object], list[str]]] = []
    for row in regions.itertuples():
        part = open_images[open_images["region"] == row.region].copy()
        if part.empty:
            continue
        part["_estimate"] = part["estimate"].astype(float).fillna(-1.0)
        buckets.append((row._asdict(), part.sort_values("_estimate", ascending=False)["image_id"].astype(str).tolist()))
    ordered: list[dict[str, object]] = []
    while buckets:
        next_round = []
        for info, ids in buckets:
            take, rest = ids[:per_region], ids[per_region:]
            mean_text = f", Ø {info['mean_score']:.1f}" if info["mean_score"] == info["mean_score"] else ""
            reason = f"Kartenlücke R{int(info['region'])}: {int(info['n'])} Bilder, {int(info['n_labeled'])} gelabelt{mean_text}, meist {info['top_cloud_type'] or 'unbekannt'}"
            ordered.extend({"image_id": image_id, "region": int(info["region"]), "map_reason": reason} for image_id in take)
            if rest:
                next_round.append((info, rest))
        buckets = next_round
    frame = pd.DataFrame(ordered)
    if frame.empty:
        return pd.DataFrame(columns=["image_id", "region", "map_rank", "map_reason"])
    frame["map_rank"] = np.arange(len(frame))
    return frame[["image_id", "region", "map_rank", "map_reason"]]


def save_map_queue(frame: pd.DataFrame, path_like: str | Path = MAP_QUEUE_PATH) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_map_queue(path_like: str | Path = MAP_QUEUE_PATH) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame(columns=["image_id", "region", "map_rank", "map_reason"])
    frame = pd.read_parquet(path)
    frame["image_id"] = frame["image_id"].astype(str)
    return frame
