from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import aggregate_labels_impl, extract_features_impl, train_impl
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Train Models", layout="wide")
st.title("Train Models")
st.caption("Run ingestion products in sequence, then launch a baseline or deep experiment.")

col1, col2, col3 = st.columns(3)
if col1.button("Aggregate labels"):
    tables = aggregate_labels_impl()
    st.success(f"Aggregated {len(tables['aggregated'])} image labels.")
if col2.button("Extract features"):
    features = extract_features_impl("configs/features/v1.yaml")
    st.success(f"Extracted features for {len(features)} images.")

model_config = st.selectbox(
    "Model config",
    [
        "configs/models/baseline.yaml",
        "configs/models/deep_cnn.yaml",
        "configs/models/hybrid.yaml",
        "configs/models/ranking.yaml",
    ],
)

if col3.button("Train selected config"):
    summary = train_impl(model_config)
    st.success(f"Finished run {summary['run_id']}.")
    st.json(summary)

st.subheader("Config Preview")
with resolve_path(model_config).open("r", encoding="utf-8") as handle:
    st.code(handle.read(), language="yaml")
