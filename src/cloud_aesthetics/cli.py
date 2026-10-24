from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import typer
from sklearn.decomposition import PCA

from cloud_aesthetics.data.aggregation import aggregate_ratings
from cloud_aesthetics.data.exclusions import active_excluded_ids
from cloud_aesthetics.data.manifest import build_manifest, ensure_kind_columns, image_kind_from_path, save_manifest
from cloud_aesthetics.data.pairwise import generate_pseudo_pairs, merge_pairwise_tables
from cloud_aesthetics.data.ratings import load_raw_pairwise_preferences, load_raw_scalar_ratings
from cloud_aesthetics.data.splits import create_grouped_splits
from cloud_aesthetics.data.web_dataset import download_wikimedia_cloud_dataset
from cloud_aesthetics.eval.analysis import compare_groups, compute_feature_correlation, compute_pca_projection
from cloud_aesthetics.eval.neighbors import nearest_neighbors
from cloud_aesthetics.explain.concepts import cloud_type_scores, top_concepts
from cloud_aesthetics.explain.feature_importance import (
    approximate_local_contributions,
    load_pickled_model,
    permutation_feature_importance,
)
from cloud_aesthetics.explain.heatmaps import (
    available_gradcam_layers,
    fallback_heatmap_from_edges,
    grad_cam_heatmap,
    overlay_heatmap_on_image,
    save_overlay,
    simple_gradient_heatmap,
)
from cloud_aesthetics.explain.interpretation import build_interpretation
from cloud_aesthetics.explain.regions import build_region_table
from cloud_aesthetics.explain.text_report import build_text_explanation
from cloud_aesthetics.features.base import extract_and_save_features, extract_features_from_manifest
from cloud_aesthetics.features.store import load_features
from cloud_aesthetics.models.baseline import train_baseline_suite
from cloud_aesthetics.models.deep import CloudRatingNet, train_deep_model
from cloud_aesthetics.models.hybrid import train_hybrid_model
from cloud_aesthetics.models.ranking import train_pairwise_feature_model
from cloud_aesthetics.preprocessing.importer import DEFAULT_CROP_SCALES, DEFAULT_MIN_CROP_SIZE, import_private_images
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.preprocessing.transforms import apply_transform, build_eval_transform
from cloud_aesthetics.settings import PATHS, load_yaml, resolve_path
from cloud_aesthetics.standalone import build_friend_package, import_friend_label_bundle
from cloud_aesthetics.utils.io import read_table, write_json, write_table

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _require_non_empty(frame: pd.DataFrame, name: str, help_text: str) -> None:
    if frame.empty:
        raise typer.BadParameter(f"{name} is empty. {help_text}")


def _load_dataset_config(config_path: str | Path = "configs/dataset/default.yaml") -> dict[str, object]:
    return load_yaml(config_path)


def _filter_active_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    manifest = ensure_kind_columns(manifest)
    excluded_ids = active_excluded_ids()
    if not excluded_ids:
        return manifest
    return manifest[~manifest["image_id"].astype(str).isin(excluded_ids)].reset_index(drop=True)


GROUP_FILTER_KEYS = {"include_kinds": "kind", "include_crop_classes": "crop_class", "include_datasets": "dataset_name"}


def _apply_group_filters(labels: pd.DataFrame, manifest: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict training to image groups named in the model config (e.g. originals only)."""
    active = {GROUP_FILTER_KEYS[key]: {str(value) for value in config[key]} for key in GROUP_FILTER_KEYS if config.get(key)}
    if not active:
        return labels, manifest
    manifest = ensure_kind_columns(manifest)
    labels = attach_manifest_groups(labels, manifest)
    for column, allowed in active.items():
        manifest = manifest[manifest[column].astype(str).isin(allowed)]
        labels = labels[labels[column].astype(str).isin(allowed)]
    return labels.reset_index(drop=True), manifest.reset_index(drop=True)


def attach_manifest_groups(table: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach ``kind`` and ``dataset_name`` from the manifest to a per-image table."""
    if table.empty or manifest.empty or "image_id" not in table.columns:
        return table
    manifest = ensure_kind_columns(manifest)
    group_columns = ["kind", "crop_class", "dataset_name"]
    groups = manifest[["image_id"] + group_columns].copy()
    groups["image_id"] = groups["image_id"].astype(str)
    groups = groups.drop_duplicates(subset=["image_id"], keep="first")
    result = table.drop(columns=[column for column in group_columns if column in table.columns]).copy()
    result["image_id"] = result["image_id"].astype(str)
    return result.merge(groups, on="image_id", how="left")


def _save_processed_tables(config: dict[str, object], ratings: pd.DataFrame, pairwise: pd.DataFrame, aggregated: pd.DataFrame, splits: pd.DataFrame) -> None:
    write_table(ratings, config["ratings_path"])
    write_table(pairwise, config["pairwise_path"])
    write_table(aggregated, config["aggregated_labels_path"])
    write_table(splits, config["splits_path"])


def _resolve_run_dir(run_id: str | Path) -> Path:
    candidate = resolve_path(run_id)
    if candidate.exists() and candidate.is_dir():
        return candidate
    root = PATHS.artifacts_root
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and child.name == str(run_id):
            return child
    raise FileNotFoundError(f"Run not found: {run_id}")


def ingest_images_impl(dataset_config_path: str | Path = "configs/dataset/default.yaml") -> pd.DataFrame:
    config = _load_dataset_config(dataset_config_path)
    manifest = build_manifest(
        image_root=config["image_root"],
        allowed_extensions=config.get("allowed_extensions", []),
        capture_session_strategy=str(config.get("capture_session_strategy", "parent_dir")),
    )
    save_manifest(manifest, config["manifest_path"])
    return manifest


def import_images_impl(
    source: str | Path,
    dataset_name: str,
    dataset_config_path: str | Path = "configs/dataset/default.yaml",
    make_crops: bool = True,
    max_crops_per_image: int = 8,
    min_sky_fraction: float = 0.72,
    min_cloud_fraction: float = 0.08,
    batch_id: str | None = None,
    min_crop_size: int = DEFAULT_MIN_CROP_SIZE,
    crop_scales: tuple[float, ...] = DEFAULT_CROP_SCALES,
) -> dict[str, object]:
    config = _load_dataset_config(dataset_config_path)
    imported = import_private_images(
        source,
        dataset_name=dataset_name,
        output_root=config["image_root"],
        make_crops=make_crops,
        max_crops_per_image=max_crops_per_image,
        min_crop_size=min_crop_size,
        crop_scales=crop_scales,
        min_sky_fraction=min_sky_fraction,
        min_cloud_fraction=min_cloud_fraction,
        batch_id=batch_id,
    )
    manifest = ingest_images_impl(dataset_config_path)
    import_batch_id = ""
    if not imported.empty and "import_batch_id" in imported.columns:
        import_batch_id = str(imported["import_batch_id"].dropna().iloc[0])
    return {
        "batch_id": import_batch_id,
        "imported_count": int(len(imported)),
        "original_count": int((imported["derivative_kind"] == "original").sum()) if not imported.empty else 0,
        "crop_count": int((imported["derivative_kind"] == "sky_crop").sum()) if not imported.empty else 0,
        "manifest_count": int(len(manifest)),
    }


def aggregate_labels_impl(dataset_config_path: str | Path = "configs/dataset/default.yaml") -> dict[str, pd.DataFrame]:
    config = _load_dataset_config(dataset_config_path)
    ratings = load_raw_scalar_ratings(config["ratings_dir"])
    if not ratings.empty and not bool(config.get("include_llm_raters", False)):
        # Model raters (LLM judges) are kept out of the human consensus unless asked for.
        from cloud_aesthetics.judge.store import human_rater_ids

        humans = human_rater_ids(ratings["rater_id"].astype(str).unique(), config.get("raters_path", "data/raw/metadata/raters.csv"))
        ratings = ratings[ratings["rater_id"].astype(str).isin(humans)].reset_index(drop=True)
    explicit_pairwise = load_raw_pairwise_preferences(config["pairwise_dir"])
    pseudo_pairwise = generate_pseudo_pairs(ratings, min_score_gap=2.0)
    pairwise = merge_pairwise_tables(explicit_pairwise, pseudo_pairwise)
    aggregated = aggregate_ratings(ratings, explicit_pairwise)
    manifest = _filter_active_manifest(read_table(config["manifest_path"]))
    aggregated = attach_manifest_groups(aggregated, manifest)
    split_cfg = config["split"]
    splits = create_grouped_splits(
        manifest=manifest,
        labels=aggregated,
        target_column="mean_score",
        n_splits=int(split_cfg["n_splits"]),
        score_bins=int(split_cfg["score_bins"]),
        test_fraction=float(split_cfg["test_fraction"]),
        random_state=int(split_cfg["random_state"]),
    )
    _save_processed_tables(config, ratings, pairwise, aggregated, splits)
    return {"ratings": ratings, "pairwise": pairwise, "aggregated": aggregated, "splits": splits}


def extract_features_impl(
    feature_config_path: str | Path,
    dataset_config_path: str | Path = "configs/dataset/default.yaml",
) -> pd.DataFrame:
    dataset_config = _load_dataset_config(dataset_config_path)
    manifest = _filter_active_manifest(read_table(dataset_config["manifest_path"]))
    return extract_and_save_features(manifest, feature_config_path)


def _derive_embedding_table(features: pd.DataFrame, n_components: int = 16) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in features.columns
        if column != "image_id" and pd.api.types.is_numeric_dtype(features[column])
    ]
    numeric = features[numeric_columns].fillna(0.0)
    n_components = min(n_components, numeric.shape[0], numeric.shape[1]) if not numeric.empty else 0
    if n_components <= 0:
        return pd.DataFrame({"image_id": features["image_id"]})
    projection = PCA(n_components=n_components, random_state=42).fit_transform(numeric)
    columns = [f"embed_{index:02d}" for index in range(projection.shape[1])]
    return pd.DataFrame(projection, columns=columns).assign(image_id=features["image_id"].values)


def train_impl(model_config_path: str | Path, dataset_config_path: str | Path = "configs/dataset/default.yaml") -> dict[str, object]:
    config = load_yaml(model_config_path)
    dataset_config = _load_dataset_config(dataset_config_path)
    kind = str(config["kind"])
    labels = read_table(config.get("labels_path", dataset_config["aggregated_labels_path"]))
    manifest = _filter_active_manifest(read_table(config.get("manifest_path", dataset_config["manifest_path"])))
    splits = read_table(dataset_config["splits_path"])
    pairwise = read_table(dataset_config["pairwise_path"])
    _require_non_empty(manifest, "image manifest", "Run `ingest-images` after placing images under `data/raw/images`.")
    _require_non_empty(labels, "aggregated labels", "Record ratings, then run `aggregate-labels`.")
    _require_non_empty(splits, "split table", "Run `aggregate-labels` to generate grouped splits.")
    labels, manifest = _apply_group_filters(labels, manifest, config)
    _require_non_empty(labels, "filtered labels", "The include_kinds / include_crop_classes / include_datasets filters left no labeled images.")
    if kind == "baseline":
        features = load_features(config["features_path"])
        _require_non_empty(features, "feature table", "Run `extract-features --config configs/features/v1.yaml` first.")
        return train_baseline_suite(features, labels, splits, config, pairwise_table=pairwise)
    if kind == "deep":
        return train_deep_model(manifest, labels, splits, config)
    if kind == "embedding":
        from cloud_aesthetics.embeddings.zero_shot_store import load_zero_shot
        from cloud_aesthetics.models.embedding import build_embedding_input_table, train_embedding_suite

        features = load_features(config.get("features_path", "data/processed/features/features_v1.parquet"))
        zero_shot = load_zero_shot(config.get("embeddings_dir", "data/processed/embeddings"))
        table = build_embedding_input_table(config, features=features, zero_shot=zero_shot)
        table = table[table["image_id"].isin(set(manifest["image_id"].astype(str)))]
        _require_non_empty(table, "embedding table", "Run `extract-embeddings` first.")
        return train_embedding_suite(table, labels, splits, config)
    if kind == "hybrid":
        features = load_features(config["features_path"])
        _require_non_empty(features, "feature table", "Run `extract-features --config configs/features/v1.yaml` first.")
        embeddings = _derive_embedding_table(features)
        return train_hybrid_model(features, embeddings, labels, splits, config)
    if kind == "ranking":
        features = load_features(config["features_path"])
        _require_non_empty(features, "feature table", "Run `extract-features --config configs/features/v1.yaml` first.")
        merged = train_pairwise_feature_model(features, pairwise_train=pairwise, pairwise_eval=pairwise)
        from cloud_aesthetics.models.base import create_run_context, save_run_json

        run = create_run_context(config["output_dir"], str(config["run_name"]), config)
        if merged.get("available"):
            with (run.run_dir / "ranking_model.pkl").open("wb") as handle:
                pickle.dump(merged["model"], handle)
            merged.pop("model", None)
        summary = {"run_id": run.run_id, "kind": "ranking", "metrics": merged.get("metrics", {}), "details": merged}
        save_run_json(summary, run.run_dir / "summary.json")
        return summary
    raise ValueError(f"Unsupported model kind: {kind}")


def evaluate_impl(run_id: str | Path, dataset_config_path: str | Path = "configs/dataset/default.yaml") -> dict[str, object]:
    run_dir = _resolve_run_dir(run_id)
    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    dataset_config = _load_dataset_config(dataset_config_path)
    features_path = run_dir / "feature_analysis.csv"
    if not features_path.exists() and resolve_path("data/processed/features/features_v1.parquet").exists():
        features = read_table("data/processed/features/features_v1.parquet")
        labels = read_table(dataset_config["aggregated_labels_path"])
        high_selector = lambda frame: frame["mean_score"] >= frame["mean_score"].quantile(0.9)
        mid_selector = lambda frame: (frame["mean_score"] >= frame["mean_score"].quantile(0.4)) & (
            frame["mean_score"] <= frame["mean_score"].quantile(0.6)
        )
        comparison = compare_groups(features, labels, high_selector, mid_selector)
        comparison.to_csv(features_path, index=False)
        compute_feature_correlation(features).to_csv(run_dir / "feature_correlation.csv")
        compute_pca_projection(features).to_csv(run_dir / "feature_pca.csv", index=False)
    summary["run_dir"] = str(run_dir)
    return summary


def _select_best_baseline_model(summary: dict[str, object]) -> tuple[str, dict[str, object]]:
    models = summary.get("models", {})
    best_name = ""
    best_info: dict[str, object] = {}
    best_score = float("inf")
    for name, info in models.items():
        val_metrics = info.get("metrics", {}).get("val", {})
        mae = val_metrics.get("mae", float("inf"))
        if mae < best_score:
            best_score = mae
            best_name = name
            best_info = info
    return best_name, best_info


def _feature_output_path(feature_config_path: str | Path) -> str | Path:
    return load_yaml(feature_config_path)["output_path"]


def ensure_features_for_images(
    manifest: pd.DataFrame,
    image_ids: list[str],
    feature_config_path: str | Path = "configs/features/v1.yaml",
) -> pd.DataFrame:
    feature_config = load_yaml(feature_config_path)
    feature_path = feature_config["output_path"]
    existing = read_table(feature_path)
    existing_ids = set(existing["image_id"].astype(str)) if not existing.empty and "image_id" in existing.columns else set()
    target_ids = {str(image_id) for image_id in image_ids}
    missing = target_ids - existing_ids
    if missing:
        target_manifest = manifest[manifest["image_id"].astype(str).isin(missing)]
        new_frame = extract_features_from_manifest(target_manifest, feature_config_path)
        combined = pd.concat([existing, new_frame], ignore_index=True) if not existing.empty else new_frame
        combined = combined.drop_duplicates(subset=["image_id"], keep="last").reset_index(drop=True)
        write_table(combined, feature_path)
        return combined
    return existing


def list_import_batches(dataset_config_path: str | Path = "configs/dataset/default.yaml") -> list[str]:
    config = _load_dataset_config(dataset_config_path)
    manifest = read_table(config["manifest_path"])
    if manifest.empty:
        return []
    if "import_batch_id" in manifest.columns:
        batches = manifest["import_batch_id"].dropna().astype(str)
        batch_ids = sorted([batch for batch in batches.unique().tolist() if batch], reverse=True)
        if batch_ids:
            return batch_ids
    if "capture_session_id" in manifest.columns:
        sessions = sorted(
            [session for session in manifest["capture_session_id"].dropna().astype(str).unique().tolist() if session],
            reverse=True,
        )
        if sessions:
            return [f"session:{session}" for session in sessions]
    if "split_group_id" in manifest.columns:
        groups = sorted(
            [group for group in manifest["split_group_id"].dropna().astype(str).unique().tolist() if group],
            reverse=True,
        )
        return [f"group:{group}" for group in groups]
    return []


def _select_batch_manifest(manifest: pd.DataFrame, batch_id: str) -> pd.DataFrame:
    batch_id = str(batch_id)
    if "import_batch_id" in manifest.columns:
        matched = manifest[manifest["import_batch_id"].fillna("").astype(str) == batch_id].copy()
        if not matched.empty:
            return matched
    if batch_id.startswith("session:") and "capture_session_id" in manifest.columns:
        session_id = batch_id.removeprefix("session:")
        return manifest[manifest["capture_session_id"].fillna("").astype(str) == session_id].copy()
    if batch_id.startswith("group:") and "split_group_id" in manifest.columns:
        group_id = batch_id.removeprefix("group:")
        return manifest[manifest["split_group_id"].fillna("").astype(str) == group_id].copy()
    if "capture_session_id" in manifest.columns:
        matched = manifest[manifest["capture_session_id"].fillna("").astype(str) == batch_id].copy()
        if not matched.empty:
            return matched
    if "split_group_id" in manifest.columns:
        matched = manifest[manifest["split_group_id"].fillna("").astype(str) == batch_id].copy()
        if not matched.empty:
            return matched
    return pd.DataFrame()


def _batch_report_dir(run_id: str | Path, batch_id: str) -> Path:
    run_dir = _resolve_run_dir(run_id)
    safe_batch = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in batch_id)
    return run_dir / "batch_reports" / safe_batch


def batch_predictions_path(run_id: str | Path, batch_id: str) -> Path:
    return _batch_report_dir(run_id, batch_id) / "predictions.parquet"


def analyze_batch_impl(
    run_id: str | Path,
    batch_id: str,
    dataset_config_path: str | Path = "configs/dataset/default.yaml",
    feature_config_path: str | Path = "configs/features/v1.yaml",
    overwrite: bool = False,
) -> dict[str, object]:
    dataset_config = _load_dataset_config(dataset_config_path)
    manifest = read_table(dataset_config["manifest_path"])
    if manifest.empty:
        raise ValueError("Manifest is empty. Import images before analyzing a batch.")
    batch_manifest = _select_batch_manifest(manifest, str(batch_id))
    if batch_manifest.empty:
        raise ValueError(f"No images found for import batch: {batch_id}")

    image_ids = batch_manifest["image_id"].astype(str).tolist()
    ensure_features_for_images(manifest, image_ids, feature_config_path)
    report_dir = _batch_report_dir(run_id, batch_id)
    predictions_path = report_dir / "predictions.parquet"
    existing = read_table(predictions_path)
    existing_ids = set(existing["image_id"].astype(str)) if not overwrite and not existing.empty else set()
    rows: list[dict[str, object]] = []
    created_at = datetime.now(UTC).isoformat()

    for _, image_row in batch_manifest.iterrows():
        image_id = str(image_row["image_id"])
        if image_id in existing_ids:
            continue
        explanation = explain_impl(run_id, image_id, dataset_config_path, feature_config_path)
        explanation_path = _resolve_run_dir(run_id) / "explanations" / f"{image_id}.json"
        rows.append(
            {
                "batch_id": str(batch_id),
                "run_id": str(run_id),
                "image_id": image_id,
                "relative_path": str(image_row["relative_path"]),
                "kind": str(image_row["kind"]) if "kind" in image_row and pd.notna(image_row["kind"]) else image_kind_from_path(image_row["relative_path"]),
                "predicted_score": float(explanation.get("predicted_score", float("nan"))),
                "uncertainty": float(explanation.get("uncertainty", 0.0)),
                "top_features": json.dumps(explanation.get("top_features", [])),
                "top_concepts": json.dumps(explanation.get("top_concepts", [])),
                "interpretation": json.dumps(explanation.get("interpretation", {})),
                "recommendation": str(explanation.get("interpretation", {}).get("recommendation", "")),
                "score_band": str(explanation.get("interpretation", {}).get("score_band", {}).get("label", "")),
                "confidence_band": str(explanation.get("interpretation", {}).get("confidence_band", {}).get("label", "")),
                "score_interval_low": float(explanation.get("interpretation", {}).get("score_interval", {}).get("low", float("nan"))),
                "score_interval_high": float(explanation.get("interpretation", {}).get("score_interval", {}).get("high", float("nan"))),
                "explanation_path": str(explanation_path),
                "overlay_path": str(explanation.get("overlay_path", "")),
                "created_at": created_at,
            }
        )

    new_predictions = pd.DataFrame(rows)
    if overwrite:
        combined = new_predictions
    elif not existing.empty:
        combined = pd.concat([existing, new_predictions], ignore_index=True)
        combined = combined.drop_duplicates(subset=["batch_id", "run_id", "image_id"], keep="last").reset_index(drop=True)
    else:
        combined = new_predictions
    write_table(combined, predictions_path)
    return {
        "batch_id": str(batch_id),
        "run_id": str(run_id),
        "image_count": int(len(batch_manifest)),
        "analyzed_count": int(len(new_predictions)),
        "skipped_count": int(len(batch_manifest) - len(new_predictions)),
        "predictions_path": str(predictions_path),
        "report_dir": str(report_dir),
    }


def _zero_shot_context(image_id: str, output_dir: str | Path = "data/processed/embeddings", top_k: int = 6) -> dict[str, object]:
    """Compact CLIP zero-shot summary for one image, or an empty dict when not computed yet."""
    from cloud_aesthetics.embeddings.zero_shot_store import zero_shot_row

    row = zero_shot_row(image_id, output_dir)
    if row.empty:
        return {}
    types = {name[len("clip_type_"):]: float(value) for name, value in row.items() if str(name).startswith("clip_type_")}
    lights = {name[len("clip_light_"):]: float(value) for name, value in row.items() if str(name).startswith("clip_light_")}
    concepts = []
    for name, value in row.items():
        name = str(name)
        if name.startswith("clip_concept_") and not name.startswith("clip_concept_z_"):
            concept = name[len("clip_concept_"):]
            concepts.append({"concept": concept, "score": float(value), "z_score": float(row.get(f"clip_concept_z_{concept}", 0.0))})
    concepts.sort(key=lambda item: item["z_score"], reverse=True)
    return {
        "model": "clip_vit_b32",
        "cloud_type": str(row.get("clip_cloud_type", "unknown")),
        "cloud_type_confidence": float(row.get("clip_cloud_type_confidence", 0.0)),
        "cloud_type_probability": float(row.get("clip_cloud_type_probability", max(types.values()) if types else 0.0)),
        "cloud_type_scores": [{"cloud_type": name, "score": score} for name, score in sorted(types.items(), key=lambda item: item[1], reverse=True)],
        "light": str(row.get("clip_light", "unknown")),
        "light_scores": [{"light": name, "score": score} for name, score in sorted(lights.items(), key=lambda item: item[1], reverse=True)],
        "concepts": concepts[:top_k] + concepts[-2:],
        "all_concepts": concepts,
        "caveat": "CLIP zero-shot attributes are text-prompt similarities, not meteorological observations.",
    }


def explain_impl(
    run_id: str | Path,
    image_id: str,
    dataset_config_path: str | Path = "configs/dataset/default.yaml",
    feature_config_path: str | Path = "configs/features/v1.yaml",
) -> dict[str, object]:
    run_dir = _resolve_run_dir(run_id)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    dataset_config = _load_dataset_config(dataset_config_path)
    manifest = read_table(dataset_config["manifest_path"])
    labels = read_table(dataset_config["aggregated_labels_path"])
    image_matches = manifest.loc[manifest["image_id"].astype(str) == str(image_id)]
    if image_matches.empty:
        raise ValueError(f"Image not found in manifest: {image_id}")
    image_row = image_matches.iloc[0]
    feature_path = _feature_output_path(feature_config_path)
    features = read_table(feature_path) if resolve_path(feature_path).exists() else pd.DataFrame()
    feature_ids = set(features["image_id"].astype(str)) if not features.empty and "image_id" in features.columns else set()
    if str(image_id) not in feature_ids:
        features = ensure_features_for_images(manifest, [str(image_id)], feature_config_path)
    image = read_rgb_image(image_row["relative_path"])
    predicted_score = float("nan")
    uncertainty = 0.0
    feature_row = pd.Series(dtype=float)
    feature_contributions = pd.DataFrame(columns=["feature", "delta_prediction", "value"])
    heatmap_variants: dict[str, str] = {}
    embedding_neighbors = pd.DataFrame()
    if summary["kind"] == "baseline":
        model_name, model_info = _select_best_baseline_model(summary)
        model = load_pickled_model(model_info["model_path"])
        feature_matches = features.loc[features["image_id"].astype(str) == str(image_id)]
        if feature_matches.empty:
            raise ValueError(f"Features not found for image: {image_id}")
        feature_row = feature_matches.iloc[0]
        numeric_row = feature_row.drop(labels=["image_id"])
        numeric_row = numeric_row[pd.to_numeric(numeric_row, errors="coerce").notna()].astype(float)
        expected = getattr(model, "feature_names_in_", None)
        if expected is None and hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                expected = getattr(step, "feature_names_in_", None)
                if expected is not None:
                    break
        if expected is not None:
            numeric_row = numeric_row.reindex(list(expected), fill_value=0.0)
        predicted_score = float(model.predict(pd.DataFrame([numeric_row]))[0])
        uncertainty = float(model_info.get("conformal_half_width", 0.0))
        feature_contributions = approximate_local_contributions(model, feature_row, features)
        if not feature_contributions.empty:
            feature_contributions = feature_contributions.assign(attribution_kind="prediction_delta")
        heatmap = fallback_heatmap_from_edges(image_row["relative_path"])
        heatmap_source = "edge_fallback"
    elif summary["kind"] == "deep":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("torch is required to explain deep runs") from exc
        run_config_path = run_dir / "run.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8")).get("config", {}) if run_config_path.exists() else {}
        config = summary.get("config", {}) or run_config
        model = CloudRatingNet(
            backbone_name=str(config.get("backbone", "resnet18")),
            freeze_backbone=False,
            bounded_output=bool(config.get("bounded_output", False)),
            score_min=float(config.get("score_min", 0.0)),
            score_max=float(config.get("score_max", 10.0)),
        )
        model.load_state_dict(torch.load(summary["model_path"], map_location="cpu"))
        model.eval()
        transformed = apply_transform(build_eval_transform(int(config.get("image_size", 224))), image)
        tensor = torch.tensor(np.transpose(transformed["image"], (2, 0, 1)), dtype=torch.float32)
        predicted_score = float(model(tensor.unsqueeze(0)).detach().cpu().numpy()[0])
        uncertainty = float(summary.get("conformal_half_width", 0.0))
        device = torch.device("cpu")
        gradient_heatmap = simple_gradient_heatmap(model, tensor, device=device)
        heatmap = gradient_heatmap
        heatmap_source = "input_gradient"
        for layer_name in available_gradcam_layers(model):
            try:
                layer_heatmap = grad_cam_heatmap(model, tensor, device=device, target_layer=layer_name)
            except Exception:
                continue
            layer_heatmap_resized = layer_heatmap if layer_heatmap.shape[:2] == image.shape[:2] else cv2.resize(
                layer_heatmap, (image.shape[1], image.shape[0])
            )
            variant_path = run_dir / "explanations" / f"{image_id}_{layer_name.replace('.', '_')}_gradcam.npy"
            variant_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(variant_path, layer_heatmap_resized.astype(np.float32))
            heatmap_variants[f"gradcam:{layer_name}"] = str(variant_path)
        if not features.empty and str(image_id) in set(features["image_id"].astype(str)):
            feature_row = features.loc[features["image_id"].astype(str) == str(image_id)].iloc[0]
            numeric_features = features.select_dtypes(include=["number"])
            reference = numeric_features.median()
            scale = numeric_features.std(ddof=0).replace(0.0, np.nan)
            deltas = []
            for column, value in feature_row[numeric_features.columns].items():
                value_float = float(value)
                typical = float(reference[column])
                std = float(scale[column]) if pd.notna(scale[column]) else 0.0
                z_score = (value_float - typical) / std if std > 0 else 0.0
                deltas.append(
                    {
                        "feature": column,
                        "distance_from_typical": abs(float(z_score)),
                        "z_score": float(z_score),
                        "value": value_float,
                        "typical_value": typical,
                        "attribution_kind": "feature_distance",
                    }
                )
            feature_contributions = pd.DataFrame(deltas).sort_values("distance_from_typical", ascending=False)
    elif summary["kind"] == "embedding":
        from cloud_aesthetics.embeddings.zero_shot_store import load_zero_shot
        from cloud_aesthetics.models.embedding import embedding_input_row, labeled_embedding_neighbors

        model_name, model_info = _select_best_baseline_model(summary)
        model = load_pickled_model(model_info["model_path"])
        zero_shot_table_full = load_zero_shot(str(summary.get("embeddings_dir", "data/processed/embeddings")))
        input_row = embedding_input_row(str(image_id), summary, manifest=manifest, features=features, zero_shot=zero_shot_table_full)
        predicted_score = float(model.predict(input_row)[0])
        uncertainty = float(model_info.get("conformal_half_width", 0.0))
        embedding_neighbors = labeled_embedding_neighbors(str(image_id), summary, labels)
        if not embedding_neighbors.empty and embedding_neighbors["mean_score"].notna().any():
            neighbor_std = float(embedding_neighbors["mean_score"].std(ddof=0))
            uncertainty = float(max(uncertainty * 0.5, neighbor_std)) if uncertainty else neighbor_std
        if not features.empty and str(image_id) in set(features["image_id"].astype(str)):
            feature_row = features.loc[features["image_id"].astype(str) == str(image_id)].iloc[0]
        heatmap = fallback_heatmap_from_edges(image_row["relative_path"])
        heatmap_source = "edge_fallback"
    else:
        heatmap = fallback_heatmap_from_edges(image_row["relative_path"])
        heatmap_source = "edge_fallback"
        if not features.empty and str(image_id) in set(features["image_id"].astype(str)):
            feature_row = features.loc[features["image_id"].astype(str) == str(image_id)].iloc[0]
        else:
            feature_row = pd.Series(dtype=float)
    heatmap_resized = heatmap if heatmap.shape[:2] == image.shape[:2] else cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    overlay = overlay_heatmap_on_image(image, heatmap_resized)
    overlay_path = save_overlay(overlay, run_dir / "explanations" / f"{image_id}_overlay.png")
    heatmap_path = run_dir / "explanations" / f"{image_id}_heatmap.npy"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(heatmap_path, heatmap_resized.astype(np.float32))
    heatmap_variants[heatmap_source] = str(heatmap_path)
    region_table = build_region_table(image_row["relative_path"], heatmap_resized)
    concept_scores = top_concepts(feature_row) if not feature_row.empty else pd.DataFrame(columns=["concept", "score"])
    scientific_cloud_scores = cloud_type_scores(feature_row) if not feature_row.empty else pd.DataFrame(columns=["cloud_type", "score"])
    scientific_context: dict[str, object] = {}
    if not feature_row.empty:
        try:
            evidence = json.loads(str(feature_row.get("rule_based_evidence", "{}")))
        except json.JSONDecodeError:
            evidence = {}
        scientific_context = {
            "primary_cloud_type": str(feature_row.get("primary_cloud_type", "unknown")),
            "cloud_type_confidence": float(feature_row.get("cloud_type_confidence", 0.0)),
            "cloud_type_scores": scientific_cloud_scores.to_dict("records"),
            "rule_based_evidence": evidence,
            "dominant_evidence": evidence.get("dominant_evidence", evidence.get("reason", "visual cloud cues")),
            "vision_model_vote": str(feature_row.get("vision_model_vote", "not_configured")),
            "scientific_caveat": str(feature_row.get("scientific_caveat", "")),
        }
    zero_shot_context = _zero_shot_context(str(image_id))
    if zero_shot_context:
        scientific_context = {
            **scientific_context,
            "rule_based_cloud_type": scientific_context.get("primary_cloud_type", "unknown"),
            "rule_based_confidence": scientific_context.get("cloud_type_confidence", 0.0),
            "primary_cloud_type": zero_shot_context["cloud_type"],
            "cloud_type_confidence": zero_shot_context["cloud_type_confidence"],
            "vision_model_vote": f"{zero_shot_context['cloud_type']} ({zero_shot_context['cloud_type_confidence']:.2f}, CLIP zero-shot)",
            "dominant_evidence": f"CLIP zero-shot: {zero_shot_context['cloud_type']}, light {zero_shot_context['light']}",
            "clip": zero_shot_context,
        }
        if feature_contributions.empty or summary["kind"] == "embedding":
            feature_contributions = pd.DataFrame(
                [
                    {
                        "feature": f"clip_concept_{item['concept']}",
                        "distance_from_typical": abs(float(item["z_score"])),
                        "z_score": float(item["z_score"]),
                        "value": float(item["score"]),
                        "typical_value": 0.0,
                        "attribution_kind": "zero_shot_concept",
                    }
                    for item in zero_shot_context["concepts"]
                ]
            ).sort_values("distance_from_typical", ascending=False)
    ground_truth = []
    if not labels.empty and "image_id" in labels.columns:
        ground_truth = labels.loc[labels["image_id"].astype(str) == str(image_id)].to_dict("records")
    explanation = {
        "run_id": summary["run_id"],
        "image_id": image_id,
        "predicted_score": predicted_score,
        "uncertainty": uncertainty,
        "overlay_path": str(overlay_path),
        "heatmap_path": str(heatmap_path),
        "heatmap_source": heatmap_source,
        "heatmap_variants": heatmap_variants,
        "top_features": feature_contributions.head(8).to_dict("records"),
        "top_concepts": concept_scores.to_dict("records"),
        "scientific_cloud_type": scientific_context,
        "top_regions": region_table.head(8).to_dict("records"),
        "nearest_neighbors": nearest_neighbors(features, image_id).to_dict("records") if not features.empty else [],
        "embedding_neighbors": embedding_neighbors.to_dict("records") if not embedding_neighbors.empty else [],
        "zero_shot": zero_shot_context,
        "ground_truth": ground_truth,
        "scientific_caveat": (
            "Heatmaps and feature attributions reflect model-associated evidence, not literal causal proof of aesthetic preference."
        ),
        "text_explanation": build_text_explanation(
            predicted_score,
            uncertainty,
            feature_contributions,
            concept_scores,
            scientific_context,
        ),
    }
    explanation["interpretation"] = build_interpretation(explanation)
    write_json(explanation, run_dir / "explanations" / f"{image_id}.json")
    return explanation


@app.command("ingest-images")
def ingest_images(dataset_config: str = typer.Option("configs/dataset/default.yaml")) -> None:
    manifest = ingest_images_impl(dataset_config)
    typer.echo(f"Ingested {len(manifest)} images.")


@app.command("import-images")
def import_images(
    source: str = typer.Option(..., help="Directory containing private images to import"),
    dataset_name: str = typer.Option(..., help="Name for the imported dataset folder under data/raw/images"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
    batch_id: str | None = typer.Option(None, help="Optional import batch id. Defaults to a timestamped id."),
    make_crops: bool = typer.Option(True, help="Generate sky/cloud crops from imported images"),
    max_crops_per_image: int = typer.Option(8, min=0, help="Maximum crops to create per source image"),
    min_sky_fraction: float = typer.Option(0.72, min=0.0, max=1.0, help="Minimum sky-like fraction required for each crop"),
    min_cloud_fraction: float = typer.Option(0.08, min=0.0, max=1.0, help="Minimum cloud-like fraction required for each crop"),
    min_crop_size: int = typer.Option(DEFAULT_MIN_CROP_SIZE, min=64, help="Minimum crop edge in pixels; smaller candidates are skipped, never enlarged"),
    crop_scales: str = typer.Option(",".join(str(value) for value in DEFAULT_CROP_SCALES), help="Comma-separated crop sizes as fraction of the source short edge"),
) -> None:
    scales = tuple(float(value) for value in crop_scales.split(",") if value.strip())
    summary = import_images_impl(
        source,
        dataset_name,
        dataset_config,
        make_crops=make_crops and max_crops_per_image > 0,
        max_crops_per_image=max_crops_per_image,
        min_sky_fraction=min_sky_fraction,
        min_cloud_fraction=min_cloud_fraction,
        batch_id=batch_id,
        min_crop_size=min_crop_size,
        crop_scales=scales,
    )
    typer.echo(
        "Imported "
        f"{summary['original_count']} originals and {summary['crop_count']} crops. "
        f"Manifest now contains {summary['manifest_count']} images. "
        f"Batch: {summary['batch_id']}"
    )


@app.command("analyze-batch")
def analyze_batch(
    run: str = typer.Option(..., help="Run id or run directory used for local model predictions"),
    batch_id: str = typer.Option(..., help="Import batch id to analyze"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
    feature_config: str = typer.Option("configs/features/v1.yaml"),
    overwrite: bool = typer.Option(False, help="Recompute existing predictions for this batch"),
) -> None:
    summary = analyze_batch_impl(run, batch_id, dataset_config, feature_config, overwrite=overwrite)
    typer.echo(json.dumps(summary, indent=2))


@app.command("download-web-dataset")
def download_web_dataset(config: str = typer.Option("configs/dataset/web_sample.yaml", help="Web dataset config YAML")) -> None:
    metadata = download_wikimedia_cloud_dataset(config)
    typer.echo(f"Downloaded {len(metadata)} images. Metadata written for {len(metadata)} records.")


@app.command("export-friend-package")
def export_friend_package(
    output: str = typer.Option("data/artifacts/cloud_labeling_friend_package.zip", help="Output .zip path or package directory"),
    manifest_path: str = typer.Option("data/processed/image_manifest.parquet", help="Manifest to package"),
    package_name: str = typer.Option("cloud_labeling_friend_package", help="Stable package name used for browser storage"),
    rater_hint: str = typer.Option("friend", help="Default rater id shown in the standalone app"),
    zip_package: bool = typer.Option(True, help="Write a zip file in addition to the package folder"),
) -> None:
    package_path = build_friend_package(
        output,
        manifest_path=manifest_path,
        package_name=package_name,
        rater_hint=rater_hint,
        zip_package=zip_package,
    )
    typer.echo(f"Friend labeling package written to {package_path}.")


@app.command("import-friend-labels")
def import_friend_labels(
    bundle: str = typer.Option(..., help="JSON label bundle exported by the standalone friend app"),
    ratings_dir: str = typer.Option("data/raw/metadata/ratings"),
    pairwise_dir: str = typer.Option("data/raw/metadata/pairwise"),
    imported_images_root: str = typer.Option("data/raw/images/friend_imports"),
) -> None:
    summary = import_friend_label_bundle(
        bundle,
        ratings_dir=ratings_dir,
        pairwise_dir=pairwise_dir,
        imported_images_root=imported_images_root,
    )
    typer.echo(
        "Imported "
        f"{summary['ratings']} ratings, {summary['pairwise']} pairwise preferences, "
        f"and {summary['imported_images']} friend images."
    )


@app.command("extract-embeddings")
def extract_embeddings(
    config: str = typer.Option("configs/embeddings/default.yaml", help="Embedding models config"),
    model: list[str] = typer.Option(None, "--model", help="Only these model keys (default: all in config)"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
) -> None:
    """Encode all active images with the configured embedding models (incremental cache)."""
    from cloud_aesthetics.embeddings.store import extract_all_embeddings

    dataset_cfg = _load_dataset_config(dataset_config)
    manifest = _filter_active_manifest(read_table(dataset_cfg["manifest_path"]))
    _require_non_empty(manifest, "image manifest", "Run `ingest-images` first.")
    report = extract_all_embeddings(manifest, config, only=model or None, progress=typer.echo)
    typer.echo(json.dumps(report, indent=2))


@app.command("zero-shot")
def zero_shot(
    config: str = typer.Option("configs/embeddings/default.yaml", help="Embedding models config"),
    model: str = typer.Option("clip_vit_b32", help="CLIP model key with cached image embeddings"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
) -> None:
    """Score cloud types, light and concepts for every cached CLIP embedding."""
    from cloud_aesthetics.embeddings.zero_shot_store import compute_zero_shot_table

    dataset_cfg = _load_dataset_config(dataset_config)
    manifest = _filter_active_manifest(read_table(dataset_cfg["manifest_path"]))
    table = compute_zero_shot_table(manifest, config_path=config, model_key=model, progress=typer.echo)
    counts = table["clip_cloud_type"].value_counts().to_dict() if not table.empty else {}
    typer.echo(f"Zero-shot table has {len(table)} rows. Cloud types: {json.dumps(counts)}")


@app.command("judge-images")
def judge_images(
    model: str = typer.Option("claude-opus-5-5", help="Claude model id used as the vision judge"),
    only_labeled: bool = typer.Option(True, help="Judge only images that already have human labels (validation mode)"),
    limit: int | None = typer.Option(None, help="Stop after this many newly judged images"),
    effort: str = typer.Option("medium", help="output_config.effort for the judge"),
    max_edge: int = typer.Option(1024, help="Images are downscaled to this long edge before upload"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
    report: bool = typer.Option(True, help="Write the comparison report against human labels afterwards"),
) -> None:
    """Rate images with a Claude vision judge and store the scores as a separate rater."""
    from cloud_aesthetics.judge.client import ClaudeJudge
    from cloud_aesthetics.judge.store import run_judge

    dataset_cfg = _load_dataset_config(dataset_config)
    manifest = _filter_active_manifest(read_table(dataset_cfg["manifest_path"]))
    _require_non_empty(manifest, "image manifest", "Run `ingest-images` first.")
    image_ids = None
    if only_labeled:
        labels = read_table(dataset_cfg["aggregated_labels_path"])
        _require_non_empty(labels, "aggregated labels", "Run `aggregate-labels` first or pass --no-only-labeled.")
        image_ids = labels["image_id"].astype(str).tolist()
    judge = ClaudeJudge(model=model, effort=effort, max_edge=max_edge)
    summary = run_judge(
        judge,
        manifest,
        image_ids=image_ids,
        limit=limit,
        ratings_dir=dataset_cfg["ratings_dir"],
        raters_path=dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv"),
        progress=typer.echo,
    )
    typer.echo(json.dumps(summary, indent=2))
    if report and summary["judged"] + summary["cached"] > 0:
        judge_report_impl(summary["rater_id"], dataset_config)


def judge_report_impl(rater_id: str, dataset_config_path: str | Path = "configs/dataset/default.yaml", output_dir: str | Path = "data/artifacts/judge_reports") -> dict[str, object]:
    from cloud_aesthetics.judge.store import human_rater_ids
    from cloud_aesthetics.judge.validation import compare_raters, save_report

    dataset_cfg = _load_dataset_config(dataset_config_path)
    ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
    _require_non_empty(ratings, "ratings", "No ratings found.")
    humans = sorted(human_rater_ids(ratings["rater_id"].astype(str).unique(), dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv")))
    manifest = _filter_active_manifest(read_table(dataset_cfg["manifest_path"]))
    result = compare_raters(ratings, model_rater=rater_id, human_raters=humans, manifest=manifest)
    paths = save_report(result, output_dir)
    result["paths"] = paths
    return result


@app.command("judge-report")
def judge_report(
    rater: str = typer.Option(..., help="Model rater id, e.g. llm_claude_opus_5_5"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
) -> None:
    """Compare a model rater with the human labels and write a report."""
    from cloud_aesthetics.judge.validation import report_markdown

    result = judge_report_impl(rater, dataset_config)
    typer.echo(report_markdown(result))
    typer.echo(json.dumps(result["paths"], indent=2))


@app.command("status")
def status(
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
    feature_config: str = typer.Option("configs/features/v1.yaml"),
) -> None:
    """Show which pipeline stages exist and which are stale."""
    from cloud_aesthetics.pipeline import pipeline_status, status_frame

    stages = pipeline_status(dataset_config, feature_config)
    typer.echo(status_frame(stages).to_string(index=False))


@app.command("refresh")
def refresh(
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
    feature_config: str = typer.Option("configs/features/v1.yaml"),
    features: str = typer.Option("missing", help="missing | full | skip"),
    train_config: str | None = typer.Option(None, help="Optional model config to train afterwards"),
) -> None:
    """Aggregate labels, compute features and optionally train in one go."""
    from cloud_aesthetics.pipeline import refresh_pipeline

    report = refresh_pipeline(
        aggregate=True,
        features=features,
        train_config=train_config,
        dataset_config_path=dataset_config,
        feature_config_path=feature_config,
        progress=typer.echo,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("aggregate-labels")
def aggregate_labels(dataset_config: str = typer.Option("configs/dataset/default.yaml")) -> None:
    tables = aggregate_labels_impl(dataset_config)
    typer.echo(
        f"Aggregated {len(tables['ratings'])} ratings across {len(tables['aggregated'])} images and {len(tables['pairwise'])} pairwise examples."
    )


@app.command("extract-features")
def extract_features(
    config: str = typer.Option(..., help="Feature config YAML"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
) -> None:
    frame = extract_features_impl(config, dataset_config)
    typer.echo(f"Extracted features for {len(frame)} images.")


@app.command("train")
def train(
    config: str = typer.Option(..., help="Model config YAML"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
) -> None:
    summary = train_impl(config, dataset_config)
    typer.echo(f"Completed {summary['kind']} run {summary['run_id']}.")


@app.command("evaluate")
def evaluate(
    run: str = typer.Option(..., help="Run id or run directory"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
) -> None:
    summary = evaluate_impl(run, dataset_config)
    typer.echo(json.dumps(summary, indent=2))


@app.command("explain")
def explain(
    run: str = typer.Option(..., help="Run id or run directory"),
    image_id: str = typer.Option(..., help="Target image id"),
    dataset_config: str = typer.Option("configs/dataset/default.yaml"),
    feature_config: str = typer.Option("configs/features/v1.yaml"),
) -> None:
    explanation = explain_impl(run, image_id, dataset_config, feature_config)
    typer.echo(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    app()
