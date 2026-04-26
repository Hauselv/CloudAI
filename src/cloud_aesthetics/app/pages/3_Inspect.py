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

from cloud_aesthetics.app.common import app_config, dataset_config, explain_impl, list_runs, safe_read_table
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.explain.heatmaps import feature_diagnostic_map, heatmap_to_rgb, overlay_heatmap_on_image
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Inspect Predictions", layout="wide")
st.title("Inspect Predictions")

cfg = app_config()
dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])
runs = [run for run in list_runs() if (resolve_path("data/artifacts") / run / "summary.json").exists()]
if not runs:
    st.warning("No runs found yet. Train a model first.")
    st.stop()
if manifest.empty:
    st.warning("No manifest found yet.")
    st.stop()

ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
rater_id = st.text_input("Rater ID", value=str(cfg.get("default_rater_id", "friend_a")))
rated_ids = set()
if not ratings.empty:
    rated_ids = set(ratings.loc[ratings["rater_id"] == rater_id, "image_id"].dropna().astype(str))

manifest = manifest.copy()
manifest["image_id"] = manifest["image_id"].astype(str)
manifest["rated"] = manifest["image_id"].isin(rated_ids)
manifest["kind"] = manifest["relative_path"].apply(lambda value: "crop" if "/crops/" in str(value).replace("\\", "/") else "original")

if "inspect_image_id" not in st.session_state:
    st.session_state.inspect_image_id = manifest["image_id"].iloc[0]
if "inspect_explanation" not in st.session_state:
    st.session_state.inspect_explanation = None
if "inspect_explanation_key" not in st.session_state:
    st.session_state.inspect_explanation_key = None

run_id = st.selectbox("Run", runs)
st.subheader("Pick Image")
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.2, 1.2, 1, 1])
status_filter = filter_col1.selectbox("Status", ["All", "Labeled", "Open"], key="inspect_status")
kind_filter = filter_col2.selectbox("Kind", ["All", "Originals", "Crops"], key="inspect_kind")
tile_count = filter_col3.selectbox("Tiles", [12, 24, 48], index=1, key="inspect_tiles")

image_view = manifest.copy()
if status_filter == "Labeled":
    image_view = image_view[image_view["rated"]]
elif status_filter == "Open":
    image_view = image_view[~image_view["rated"]]
if kind_filter == "Originals":
    image_view = image_view[image_view["kind"] == "original"]
elif kind_filter == "Crops":
    image_view = image_view[image_view["kind"] == "crop"]

image_view = image_view.reset_index(drop=True)
page_count = max(1, int((len(image_view) + tile_count - 1) // tile_count))
if int(st.session_state.get("inspect_page", 1)) > page_count:
    st.session_state.inspect_page = 1
page = filter_col4.number_input("Page", min_value=1, max_value=page_count, value=1, key="inspect_page")
page_frame = image_view.iloc[(int(page) - 1) * tile_count : int(page) * tile_count]
st.caption(f"Showing {len(page_frame)} of {len(image_view)} images")

tile_columns = st.columns(4)
for index, (_, tile_row) in enumerate(page_frame.iterrows()):
    with tile_columns[index % 4]:
        image_path = resolve_path(tile_row["relative_path"])
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning("Missing image")
        status = "labeled" if bool(tile_row["rated"]) else "open"
        st.caption(f"{status} | {tile_row['kind']}\n\n{tile_row['image_id']}")
        if st.button("Analyze", key=f"inspect_tile_{tile_row['image_id']}", use_container_width=True):
            st.session_state.inspect_image_id = tile_row["image_id"]
            st.session_state.inspect_explanation = None
            st.session_state.inspect_explanation_key = None
            st.rerun()

image_ids = manifest["image_id"].tolist()
selected_index = image_ids.index(st.session_state.inspect_image_id) if st.session_state.inspect_image_id in image_ids else 0
image_id = st.selectbox("Selected image", image_ids, index=selected_index)
if image_id != st.session_state.inspect_image_id:
    st.session_state.inspect_image_id = image_id
    st.session_state.inspect_explanation = None
    st.session_state.inspect_explanation_key = None
    st.rerun()

overlay_alpha = st.slider("Overlay strength", min_value=0.0, max_value=0.9, value=0.45, step=0.05)
show_heatmap_only = st.checkbox("Show heatmap only", value=False)
if st.button("Generate explanation"):
    st.session_state.inspect_explanation = explain_impl(run_id, image_id)
    st.session_state.inspect_explanation_key = (run_id, image_id)

explanation = st.session_state.inspect_explanation
if explanation is not None and st.session_state.inspect_explanation_key == (run_id, image_id):
    image_row = manifest.loc[manifest["image_id"] == image_id].iloc[0]
    image = read_rgb_image(image_row["relative_path"])
    st.subheader("Prediction")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    predicted_score = explanation.get("predicted_score")
    uncertainty = explanation.get("uncertainty")
    ground_truth = explanation.get("ground_truth", [])
    metric_col1.metric("Predicted", f"{float(predicted_score):.2f}" if predicted_score is not None else "-")
    metric_col2.metric("Uncertainty", f"+/- {float(uncertainty):.2f}" if uncertainty is not None else "-")
    if ground_truth:
        metric_col3.metric("Label", f"{float(ground_truth[0].get('mean_score', 0.0)):.2f}")
    else:
        metric_col3.metric("Label", "none")

    heatmap_variants = explanation.get("heatmap_variants", {})
    heatmap_choice = None
    if heatmap_variants:
        choices = list(heatmap_variants.keys())
        default_index = choices.index("gradcam:backbone.layer4") if "gradcam:backbone.layer4" in choices else 0
        heatmap_choice = st.selectbox("Heatmap source", choices, index=default_index)
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
    with st.expander("Debug: raw explanation data"):
        st.json(explanation)
else:
    st.info("Pick an image and click Generate explanation.")
