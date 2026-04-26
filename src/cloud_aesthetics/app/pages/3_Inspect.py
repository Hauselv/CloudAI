from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import dataset_config, explain_impl, list_runs, safe_read_table

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
if st.button("Generate explanation"):
    explanation = explain_impl(run_id, image_id)
    st.json(explanation)
    overlay_path = explanation.get("overlay_path")
    if overlay_path and Path(overlay_path).exists():
        st.image(overlay_path, caption="Heatmap overlay", use_container_width=True)
    if explanation.get("top_features"):
        st.subheader("Top Interpretable Features")
        st.dataframe(pd.DataFrame(explanation["top_features"]), use_container_width=True)
    if explanation.get("top_concepts"):
        st.subheader("Top Concepts")
        st.dataframe(pd.DataFrame(explanation["top_concepts"]), use_container_width=True)
    if explanation.get("nearest_neighbors"):
        st.subheader("Nearest Neighbors")
        st.dataframe(pd.DataFrame(explanation["nearest_neighbors"]), use_container_width=True)
    st.info(explanation["scientific_caveat"])
