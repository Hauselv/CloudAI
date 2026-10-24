"""Pipeline state inspection and refresh helpers.

The processing chain is raw images -> manifest -> raw ratings -> aggregated labels ->
splits -> features -> model runs. Each stage is a file on disk, so staleness can be
derived from modification times and row counts without a database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from cloud_aesthetics.data.exclusions import active_excluded_ids, load_exclusions
from cloud_aesthetics.data.manifest import compute_sha256, project_relative_or_absolute
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.settings import PATHS, load_yaml, resolve_path
from cloud_aesthetics.utils.io import read_table

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class StageStatus:
    key: str
    label: str
    exists: bool
    rows: int
    updated_at: datetime | None
    stale: bool
    note: str = ""
    details: dict[str, object] = field(default_factory=dict)

    @property
    def state(self) -> str:
        if not self.exists:
            return "missing"
        return "stale" if self.stale else "ok"

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row.pop("details", None)
        row["state"] = self.state
        row["updated_at"] = self.updated_at.strftime("%Y-%m-%d %H:%M") if self.updated_at else ""
        return row


def _mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _newest_mtime(paths: list[Path]) -> datetime | None:
    stamps = [_mtime(path) for path in paths if path.exists()]
    stamps = [stamp for stamp in stamps if stamp is not None]
    return max(stamps) if stamps else None


def _is_newer(candidate: datetime | None, reference: datetime | None) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate > reference


def _list_image_files(image_root: Path) -> list[Path]:
    if not image_root.exists():
        return []
    return [path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def _unlisted_image_files(
    image_files: list[Path],
    manifest: pd.DataFrame,
    excluded_ids: set[str],
    exclusions_path: Path,
    max_hash_checks: int = 50,
) -> list[Path]:
    """Image files that are neither in the manifest nor excluded nor byte-identical duplicates.

    Duplicate files share the hash-based image_id and are collapsed by ``ingest-images``, so a
    handful of unlisted paths is hashed to tell true new images from known duplicates.
    """
    if manifest.empty or "relative_path" not in manifest.columns:
        return list(image_files) if manifest.empty else []
    listed = {str(path).replace("\\", "/") for path in manifest["relative_path"]}
    excluded_paths: set[str] = set()
    if exclusions_path.exists():
        exclusions = load_exclusions(exclusions_path)
        excluded_paths = {str(path).replace("\\", "/") for path in exclusions["relative_path"] if str(path)}
    candidates = []
    for path in image_files:
        relative = project_relative_or_absolute(path).replace("\\", "/")
        if relative in listed or relative in excluded_paths:
            continue
        candidates.append(path)
    if not candidates or len(candidates) > max_hash_checks:
        return candidates
    known_ids = set(manifest["image_id"].astype(str)) | excluded_ids
    return [path for path in candidates if compute_sha256(path)[:16] not in known_ids]


def _latest_run(artifacts_root: Path) -> tuple[Path | None, dict[str, object]]:
    if not artifacts_root.exists():
        return None, {}
    runs = [path for path in artifacts_root.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not runs:
        return None, {}
    latest = max(runs, key=lambda path: (path / "summary.json").stat().st_mtime)
    try:
        import json

        summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    return latest, summary


def pipeline_status(
    dataset_config_path: str | Path = "configs/dataset/default.yaml",
    feature_config_path: str | Path = "configs/features/v1.yaml",
) -> list[StageStatus]:
    """Return the state of every pipeline stage, oldest stage first."""
    dataset_cfg = load_yaml(dataset_config_path)
    feature_cfg = load_yaml(feature_config_path)
    image_root = resolve_path(dataset_cfg["image_root"])
    manifest_path = resolve_path(dataset_cfg["manifest_path"])
    ratings_dir = resolve_path(dataset_cfg["ratings_dir"])
    labels_path = resolve_path(dataset_cfg["aggregated_labels_path"])
    splits_path = resolve_path(dataset_cfg["splits_path"])
    features_path = resolve_path(feature_cfg["output_path"])
    exclusions_path = resolve_path(dataset_cfg.get("exclusions_path", "data/raw/metadata/exclusions.csv"))

    stages: list[StageStatus] = []

    image_files = _list_image_files(image_root)
    images_mtime = _newest_mtime(image_files)
    excluded_ids = active_excluded_ids(exclusions_path)
    stages.append(
        StageStatus(
            key="raw_images",
            label="Rohbilder",
            exists=bool(image_files),
            rows=len(image_files),
            updated_at=images_mtime,
            stale=False,
            note=f"{len(excluded_ids)} ausgeschlossen" if excluded_ids else "",
        )
    )

    manifest = read_table(manifest_path)
    manifest_mtime = _mtime(manifest_path)
    manifest_ids = set(manifest["image_id"].astype(str)) if not manifest.empty else set()
    new_files = _unlisted_image_files(image_files, manifest, excluded_ids, exclusions_path)
    manifest_stale = not manifest.empty and (_is_newer(images_mtime, manifest_mtime) or bool(new_files))
    manifest_note = ""
    if new_files:
        manifest_note = f"{len(new_files)} Bilddateien fehlen im Manifest -> ingest-images"
    elif manifest_stale:
        manifest_note = "neuere Bilddateien vorhanden -> ingest-images"
    stages.append(
        StageStatus(
            key="manifest",
            label="Manifest",
            exists=manifest_path.exists() and not manifest.empty,
            rows=int(len(manifest)),
            updated_at=manifest_mtime,
            stale=manifest_stale,
            note=manifest_note,
            details={"missing_kind": "kind" not in manifest.columns},
        )
    )

    raw_ratings = load_raw_scalar_ratings(ratings_dir)
    rating_files = sorted(ratings_dir.glob("*.csv")) + sorted(ratings_dir.glob("*.parquet")) if ratings_dir.exists() else []
    ratings_mtime = _newest_mtime(rating_files)
    raters = int(raw_ratings["rater_id"].nunique()) if not raw_ratings.empty else 0
    rated_images = int(raw_ratings["image_id"].nunique()) if not raw_ratings.empty else 0
    stages.append(
        StageStatus(
            key="raw_ratings",
            label="Rohbewertungen",
            exists=not raw_ratings.empty,
            rows=int(len(raw_ratings)),
            updated_at=ratings_mtime,
            stale=False,
            note=f"{rated_images} Bilder, {raters} Rater" if raters else "noch keine Bewertungen",
            details={"rated_images": rated_images, "raters": raters},
        )
    )

    labels = read_table(labels_path)
    labels_mtime = _mtime(labels_path)
    labels_stale = (not raw_ratings.empty and labels.empty) or _is_newer(ratings_mtime, labels_mtime)
    if not labels.empty and rated_images and len(labels) != rated_images:
        labels_stale = True
    labels_note = ""
    if labels_stale:
        labels_note = "Rohbewertungen neuer oder Tabelle leer -> aggregate-labels"
    elif not labels.empty and "kind" not in labels.columns:
        labels_note = "ohne kind/dataset-Spalten -> aggregate-labels"
        labels_stale = True
    stages.append(
        StageStatus(
            key="aggregated_labels",
            label="Aggregierte Labels",
            exists=labels_path.exists() and not labels.empty,
            rows=int(len(labels)),
            updated_at=labels_mtime,
            stale=labels_stale,
            note=labels_note,
        )
    )

    splits = read_table(splits_path)
    splits_mtime = _mtime(splits_path)
    label_ids = set(labels["image_id"].astype(str)) if not labels.empty else set()
    split_ids = set(splits["image_id"].astype(str)) if not splits.empty else set()
    splits_stale = (not labels.empty and splits.empty) or _is_newer(labels_mtime, splits_mtime) or (
        bool(label_ids) and split_ids != (label_ids & manifest_ids if manifest_ids else label_ids)
    )
    split_note = ""
    if not splits.empty:
        n_test = int((splits["partition"] == "test").sum())
        n_folds = int(splits.loc[splits["partition"] == "dev", "fold"].nunique())
        split_note = f"{n_test} Test, {n_folds} Dev-Folds"
    if splits_stale:
        split_note = (split_note + " | " if split_note else "") + "passt nicht zu den Labels -> aggregate-labels"
    stages.append(
        StageStatus(
            key="splits",
            label="Splits",
            exists=splits_path.exists() and not splits.empty,
            rows=int(len(splits)),
            updated_at=splits_mtime,
            stale=splits_stale,
            note=split_note,
        )
    )

    features = read_table(features_path)
    features_mtime = _mtime(features_path)
    feature_ids = set(features["image_id"].astype(str)) if not features.empty else set()
    missing_features = sorted(manifest_ids - feature_ids)
    features_stale = bool(missing_features) or (not manifest.empty and features.empty)
    feature_note = ""
    if missing_features:
        feature_note = f"{len(missing_features)} Bilder ohne Features -> extract-features"
    elif not features.empty and "kind" not in features.columns:
        feature_note = "ohne kind/dataset-Spalten (optional neu berechnen)"
    stages.append(
        StageStatus(
            key="features",
            label="Features",
            exists=features_path.exists() and not features.empty,
            rows=int(len(features)),
            updated_at=features_mtime,
            stale=features_stale,
            note=feature_note,
            details={"missing_image_ids": missing_features},
        )
    )

    run_dir, summary = _latest_run(resolve_path(dataset_cfg.get("artifacts_root", PATHS.artifacts_root)))
    run_mtime = _mtime(run_dir / "summary.json") if run_dir else None
    run_stale = run_dir is not None and (
        _is_newer(labels_mtime, run_mtime) or _is_newer(splits_mtime, run_mtime) or _is_newer(features_mtime, run_mtime)
    )
    run_note = ""
    if run_dir is not None:
        run_note = f"{summary.get('kind', '?')} run"
        if run_stale:
            run_note += " | Labels/Splits/Features sind neuer -> train"
    stages.append(
        StageStatus(
            key="latest_run",
            label="Letzter Modell-Run",
            exists=run_dir is not None,
            rows=0,
            updated_at=run_mtime,
            stale=run_stale,
            note=run_note,
            details={"run_id": run_dir.name if run_dir else "", "kind": summary.get("kind", "")},
        )
    )
    return stages


def status_frame(stages: list[StageStatus]) -> pd.DataFrame:
    return pd.DataFrame([stage.to_row() for stage in stages])[["label", "state", "rows", "updated_at", "note"]]


def refresh_pipeline(
    *,
    aggregate: bool = True,
    features: str = "missing",
    train_config: str | Path | None = None,
    dataset_config_path: str | Path = "configs/dataset/default.yaml",
    feature_config_path: str | Path = "configs/features/v1.yaml",
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Bring the processed tables up to date.

    ``features`` is ``"missing"`` (compute only images without a feature row), ``"full"``
    (recompute everything) or ``"skip"``.
    """
    from cloud_aesthetics import cli

    report: dict[str, object] = {}
    notify = progress or (lambda message: None)
    if aggregate:
        notify("Labels aggregieren und Splits erzeugen ...")
        tables = cli.aggregate_labels_impl(dataset_config_path)
        report["aggregated_labels"] = int(len(tables["aggregated"]))
        report["splits"] = int(len(tables["splits"]))
    if features == "full":
        notify("Features für alle Bilder berechnen ...")
        frame = cli.extract_features_impl(feature_config_path, dataset_config_path)
        report["features"] = int(len(frame))
    elif features == "missing":
        dataset_cfg = load_yaml(dataset_config_path)
        manifest = cli._filter_active_manifest(read_table(dataset_cfg["manifest_path"]))
        if not manifest.empty:
            notify("Fehlende Features berechnen ...")
            frame = cli.ensure_features_for_images(manifest, manifest["image_id"].astype(str).tolist(), feature_config_path)
            report["features"] = int(len(frame))
    if train_config:
        notify(f"Training mit {train_config} ...")
        summary = cli.train_impl(train_config, dataset_config_path)
        report["run_id"] = str(summary.get("run_id", ""))
    return report
