from __future__ import annotations

from pathlib import Path

import pandas as pd

from cloud_aesthetics.features.color import extract_color_features
from cloud_aesthetics.features.composition import extract_composition_features
from cloud_aesthetics.features.concept_bootstrap import infer_concepts_from_features
from cloud_aesthetics.features.scientific import classify_cloud_type_for_original, unknown_scientific_result
from cloud_aesthetics.features.store import save_features
from cloud_aesthetics.features.texture import extract_texture_features
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image, resize_long_edge
from cloud_aesthetics.settings import load_yaml
from cloud_aesthetics.utils.io import read_table


def _is_full_original(row: pd.Series, derivative_lookup: dict[str, dict[str, str]]) -> bool:
    relative_path = str(row["relative_path"])
    derivative_info = derivative_lookup.get(relative_path, {})
    if derivative_info.get("derivative_kind") == "sky_crop":
        return False
    normalized = relative_path.replace("\\", "/").lower()
    return "/crops/" not in normalized


def _derivative_lookup() -> dict[str, dict[str, str]]:
    metadata = read_table("data/raw/metadata/image_derivatives.parquet")
    if metadata.empty or "relative_path" not in metadata.columns:
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for _, row in metadata.iterrows():
        relative_path = str(row.get("relative_path", ""))
        if not relative_path:
            continue
        lookup[relative_path] = {
            "derivative_kind": str(row.get("derivative_kind", "")),
            "source_image_id": str(row.get("source_image_id", "")),
        }
    return lookup


def extract_feature_row(
    image_id: str,
    relative_path: str,
    image_size: int = 512,
    *,
    is_full_original: bool = True,
    source_image_id: str | None = None,
) -> dict[str, float | str]:
    image = read_rgb_image(relative_path)
    image = resize_long_edge(image, image_size)
    features: dict[str, float | str] = {"image_id": image_id}
    features.update(extract_color_features(image))
    features.update(extract_texture_features(image))
    features.update(extract_composition_features(image))
    if is_full_original:
        features.update(classify_cloud_type_for_original(image))
        features["scientific_source_image_id"] = image_id
    else:
        features.update(unknown_scientific_result())
        features["scientific_source_image_id"] = source_image_id or image_id
    features.update(infer_concepts_from_features(features))
    return features


def extract_features_from_manifest(manifest: pd.DataFrame, config_path: str | Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    image_size = int(config.get("image_size", 512))
    derivative_lookup = _derivative_lookup()
    rows = []
    for _, row in manifest.iterrows():
        derivative_info = derivative_lookup.get(str(row["relative_path"]), {})
        rows.append(
            extract_feature_row(
                str(row["image_id"]),
                str(row["relative_path"]),
                image_size=image_size,
                is_full_original=_is_full_original(row, derivative_lookup),
                source_image_id=derivative_info.get("source_image_id"),
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty or "scientific_source_image_id" not in frame.columns:
        return frame
    original_science = (
        frame[frame["image_id"].astype(str) == frame["scientific_source_image_id"].astype(str)]
        .drop_duplicates(subset=["image_id"], keep="first")
        .set_index("image_id")
    )
    scientific_columns = [
        column
        for column in frame.columns
        if column.startswith("meteo_")
        or column.startswith("cloud_type_")
        or column in {"primary_cloud_type", "rule_based_evidence", "vision_model_vote", "scientific_caveat"}
    ]
    for index, row in frame.iterrows():
        if str(row["image_id"]) == str(row["scientific_source_image_id"]):
            continue
        source_id = str(row["scientific_source_image_id"])
        if source_id in original_science.index:
            source_science = original_science.loc[source_id]
            for column in scientific_columns:
                frame.at[index, column] = source_science[column]
    return frame


def extract_and_save_features(manifest: pd.DataFrame, config_path: str | Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    frame = extract_features_from_manifest(manifest, config_path)
    save_features(frame, config["output_path"])
    return frame
