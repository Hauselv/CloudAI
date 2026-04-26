from __future__ import annotations

from pathlib import Path

import pandas as pd

from cloud_aesthetics.features.color import extract_color_features
from cloud_aesthetics.features.composition import extract_composition_features
from cloud_aesthetics.features.concept_bootstrap import infer_concepts_from_features
from cloud_aesthetics.features.store import save_features
from cloud_aesthetics.features.texture import extract_texture_features
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image, resize_long_edge
from cloud_aesthetics.settings import load_yaml


def extract_feature_row(image_id: str, relative_path: str, image_size: int = 512) -> dict[str, float | str]:
    image = read_rgb_image(relative_path)
    image = resize_long_edge(image, image_size)
    features: dict[str, float | str] = {"image_id": image_id}
    features.update(extract_color_features(image))
    features.update(extract_texture_features(image))
    features.update(extract_composition_features(image))
    features.update(infer_concepts_from_features(features))
    return features


def extract_features_from_manifest(manifest: pd.DataFrame, config_path: str | Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    image_size = int(config.get("image_size", 512))
    rows = [
        extract_feature_row(row["image_id"], row["relative_path"], image_size=image_size)
        for _, row in manifest.iterrows()
    ]
    return pd.DataFrame(rows)


def extract_and_save_features(manifest: pd.DataFrame, config_path: str | Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    frame = extract_features_from_manifest(manifest, config_path)
    save_features(frame, config["output_path"])
    return frame
