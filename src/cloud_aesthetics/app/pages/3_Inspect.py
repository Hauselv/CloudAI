from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import dataset_config, explain_impl, list_runs, safe_read_table
from cloud_aesthetics.explain.heatmaps import feature_diagnostic_map, heatmap_to_rgb, overlay_heatmap_on_image
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Inspect Predictions", layout="wide")
st.title("Inspect Predictions")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])
runs = list_runs()
if not runs:
    st.warning("No runs found yet. Train a model first.")
    st.stop()
if manifest.empty:
    st.warning("No manifest found yet.")
    st.stop()

run_id = st.selectbox("Run", runs)
image_id = st.selectbox("Image", manifest["image_id"].tolist())
overlay_alpha = st.slider("Overlay strength", min_value=0.0, max_value=0.9, value=0.45, step=0.05)
show_heatmap_only = st.checkbox("Show heatmap only", value=False)
if st.button("Generate explanation"):
    explanation = explain_impl(run_id, image_id)
    st.json(explanation)
    image_row = manifest.loc[manifest["image_id"] == image_id].iloc[0]
    image = read_rgb_image(image_row["relative_path"])
    heatmap_variants = explanation.get("heatmap_variants", {})
    heatmap_choice = None
    if heatmap_variants:
        heatmap_choice = st.selectbox("Heatmap source", list(heatmap_variants.keys()))
    heatmap_path = heatmap_variants.get(heatmap_choice) if heatmap_choice else explanation.get("heatmap_path")
    if heatmap_path and Path(heatmap_path).exists():
        heatmap = np.load(heatmap_path)
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        overlay = overlay_heatmap_on_image(image, heatmap, alpha=overlay_alpha)
        col_original, col_overlay = st.columns(2)
        col_original.image(str(resolve_path(image_row["relative_path"])), caption="Original", use_container_width=True)
        if show_heatmap_only:
            col_overlay.image(heatmap_to_rgb(heatmap), caption="Heatmap only", use_container_width=True)
        else:
            col_overlay.image(overlay, caption=f"Overlay alpha {overlay_alpha:.2f}", use_container_width=True)
    elif explanation.get("overlay_path") and Path(explanation["overlay_path"]).exists():
        st.image(explanation["overlay_path"], caption="Heatmap overlay", use_container_width=True)
    if explanation.get("top_features"):
        st.subheader("Top Interpretable Features")
        top_features = pd.DataFrame(explanation["top_features"])
        st.dataframe(top_features, use_container_width=True)
        feature_choice = st.selectbox("Feature diagnostic map", top_features["feature"].tolist())
        diagnostic = feature_diagnostic_map(image, feature_choice)
        diagnostic_overlay = overlay_heatmap_on_image(image, diagnostic, alpha=overlay_alpha)
        col_feature, col_feature_overlay = st.columns(2)
        col_feature.image(heatmap_to_rgb(diagnostic), caption=f"{feature_choice} map", use_container_width=True)
        col_feature_overlay.image(diagnostic_overlay, caption=f"{feature_choice} overlay", use_container_width=True)
    if explanation.get("top_concepts"):
        st.subheader("Top Concepts")
        st.dataframe(pd.DataFrame(explanation["top_concepts"]), use_container_width=True)
    if explanation.get("nearest_neighbors"):
        st.subheader("Nearest Neighbors")
        st.dataframe(pd.DataFrame(explanation["nearest_neighbors"]), use_container_width=True)
    st.info(explanation["scientific_caveat"])
