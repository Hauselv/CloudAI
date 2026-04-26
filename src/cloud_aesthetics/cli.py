from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import typer
from sklearn.decomposition import PCA

from cloud_aesthetics.data.aggregation import aggregate_ratings
from cloud_aesthetics.data.manifest import build_manifest, save_manifest
from cloud_aesthetics.data.pairwise import generate_pseudo_pairs, merge_pairwise_tables
from cloud_aesthetics.data.ratings import load_raw_pairwise_preferences, load_raw_scalar_ratings
from cloud_aesthetics.data.splits import create_grouped_splits
from cloud_aesthetics.data.web_dataset import download_wikimedia_cloud_dataset
from cloud_aesthetics.eval.analysis import compare_groups, compute_feature_correlation, compute_pca_projection
from cloud_aesthetics.eval.neighbors import nearest_neighbors
from cloud_aesthetics.explain.concepts import top_concepts
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
from cloud_aesthetics.explain.regions import build_region_table
from cloud_aesthetics.explain.text_report import build_text_explanation
from cloud_aesthetics.features.base import extract_and_save_features
from cloud_aesthetics.features.store import load_features
from cloud_aesthetics.models.baseline import train_baseline_suite
from cloud_aesthetics.models.deep import CloudRatingNet, train_deep_model
from cloud_aesthetics.models.hybrid import train_hybrid_model
from cloud_aesthetics.models.ranking import train_pairwise_feature_model
from cloud_aesthetics.preprocessing.importer import import_private_images
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
) -> dict[str, object]:
    config = _load_dataset_config(dataset_config_path)
    imported = import_private_images(
        source,
        dataset_name=dataset_name,
        output_root=config["image_root"],
        make_crops=make_crops,
        max_crops_per_image=max_crops_per_image,
        min_sky_fraction=min_sky_fraction,
        min_cloud_fraction=min_cloud_fraction,
    )
    manifest = ingest_images_impl(dataset_config_path)
    return {
        "imported_count": int(len(imported)),
        "original_count": int((imported["derivative_kind"] == "original").sum()) if not imported.empty else 0,
        "crop_count": int((imported["derivative_kind"] == "sky_crop").sum()) if not imported.empty else 0,
        "manifest_count": int(len(manifest)),
    }


def aggregate_labels_impl(dataset_config_path: str | Path = "configs/dataset/default.yaml") -> dict[str, pd.DataFrame]:
    config = _load_dataset_config(dataset_config_path)
    ratings = load_raw_scalar_ratings(config["ratings_dir"])
    explicit_pairwise = load_raw_pairwise_preferences(config["pairwise_dir"])
    pseudo_pairwise = generate_pseudo_pairs(ratings, min_score_gap=2.0)
    pairwise = merge_pairwise_tables(explicit_pairwise, pseudo_pairwise)
    aggregated = aggregate_ratings(ratings, explicit_pairwise)
    manifest = read_table(config["manifest_path"])
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
    manifest = read_table(dataset_config["manifest_path"])
    return extract_and_save_features(manifest, feature_config_path)


def _derive_embedding_table(features: pd.DataFrame, n_components: int = 16) -> pd.DataFrame:
    numeric = features.drop(columns=["image_id"]).fillna(0.0)
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
    manifest = read_table(config.get("manifest_path", dataset_config["manifest_path"]))
    splits = read_table(dataset_config["splits_path"])
    pairwise = read_table(dataset_config["pairwise_path"])
    _require_non_empty(manifest, "image manifest", "Run `ingest-images` after placing images under `data/raw/images`.")
    _require_non_empty(labels, "aggregated labels", "Record ratings, then run `aggregate-labels`.")
    _require_non_empty(splits, "split table", "Run `aggregate-labels` to generate grouped splits.")
    if kind == "baseline":
        features = load_features(config["features_path"])
        _require_non_empty(features, "feature table", "Run `extract-features --config configs/features/v1.yaml` first.")
        return train_baseline_suite(features, labels, splits, config, pairwise_table=pairwise)
    if kind == "deep":
        return train_deep_model(manifest, labels, splits, config)
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


def explain_impl(run_id: str | Path, image_id: str, dataset_config_path: str | Path = "configs/dataset/default.yaml") -> dict[str, object]:
    run_dir = _resolve_run_dir(run_id)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    dataset_config = _load_dataset_config(dataset_config_path)
    manifest = read_table(dataset_config["manifest_path"])
    labels = read_table(dataset_config["aggregated_labels_path"])
    features = read_table("data/processed/features/features_v1.parquet") if resolve_path("data/processed/features/features_v1.parquet").exists() else pd.DataFrame()
    image_row = manifest.loc[manifest["image_id"] == image_id].iloc[0]
    image = read_rgb_image(image_row["relative_path"])
    predicted_score = float("nan")
    uncertainty = 0.0
    feature_contributions = pd.DataFrame(columns=["feature", "delta_prediction", "value"])
    heatmap_variants: dict[str, str] = {}
    if summary["kind"] == "baseline":
        model_name, model_info = _select_best_baseline_model(summary)
        model = load_pickled_model(model_info["model_path"])
        feature_row = features.loc[features["image_id"] == image_id].iloc[0]
        numeric_row = feature_row.drop(labels=["image_id"])
        predicted_score = float(model.predict(pd.DataFrame([numeric_row]))[0])
        uncertainty = float(model_info.get("conformal_half_width", 0.0))
        feature_contributions = approximate_local_contributions(model, feature_row, features)
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
        model = CloudRatingNet(backbone_name=str(config.get("backbone", "resnet18")), freeze_backbone=False)
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
        if not features.empty:
            feature_row = features.loc[features["image_id"] == image_id].iloc[0]
            reference = features.drop(columns=["image_id"]).median()
            deltas = []
            for column, value in feature_row.drop(labels=["image_id"]).items():
                deltas.append({"feature": column, "delta_prediction": abs(float(value - reference[column])), "value": float(value)})
            feature_contributions = pd.DataFrame(deltas).sort_values("delta_prediction", ascending=False)
    else:
        heatmap = fallback_heatmap_from_edges(image_row["relative_path"])
        heatmap_source = "edge_fallback"
        if not features.empty:
            feature_row = features.loc[features["image_id"] == image_id].iloc[0]
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
    concept_scores = top_concepts(feature_row) if not features.empty else pd.DataFrame(columns=["concept", "score"])
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
        "top_regions": region_table.head(8).to_dict("records"),
        "nearest_neighbors": nearest_neighbors(features, image_id).to_dict("records") if not features.empty else [],
        "ground_truth": labels.loc[labels["image_id"] == image_id].to_dict("records"),
        "scientific_caveat": (
            "Heatmaps and feature attributions reflect model-associated evidence, not literal causal proof of aesthetic preference."
        ),
        "text_explanation": build_text_explanation(predicted_score, uncertainty, feature_contributions, concept_scores),
    }
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
    make_crops: bool = typer.Option(True, help="Generate sky/cloud crops from imported images"),
    max_crops_per_image: int = typer.Option(8, min=0, help="Maximum crops to create per source image"),
    min_sky_fraction: float = typer.Option(0.72, min=0.0, max=1.0, help="Minimum sky-like fraction required for each crop"),
    min_cloud_fraction: float = typer.Option(0.08, min=0.0, max=1.0, help="Minimum cloud-like fraction required for each crop"),
) -> None:
    summary = import_images_impl(
        source,
        dataset_name,
        dataset_config,
        make_crops=make_crops and max_crops_per_image > 0,
        max_crops_per_image=max_crops_per_image,
        min_sky_fraction=min_sky_fraction,
        min_cloud_fraction=min_cloud_fraction,
    )
    typer.echo(
        "Imported "
        f"{summary['original_count']} originals and {summary['crop_count']} crops. "
        f"Manifest now contains {summary['manifest_count']} images."
    )


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
) -> None:
    explanation = explain_impl(run, image_id, dataset_config)
    typer.echo(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    app()
