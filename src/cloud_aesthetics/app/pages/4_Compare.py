from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import dataset_config, safe_read_table
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Compare Images", layout="wide")
st.title("Compare Images")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])
aggregated = safe_read_table(dataset_cfg["aggregated_labels_path"])
features = safe_read_table("data/processed/features/features_v1.parquet")

if manifest.empty or aggregated.empty:
    st.warning("Need a manifest and aggregated labels before comparison mode is useful.")
    st.stop()

image_ids = manifest["image_id"].tolist()
left_id = st.selectbox("Left image", image_ids, key="compare_left")
right_id = st.selectbox("Right image", [item for item in image_ids if item != left_id], key="compare_right")

left_manifest = manifest.loc[manifest["image_id"] == left_id].iloc[0]
right_manifest = manifest.loc[manifest["image_id"] == right_id].iloc[0]
left_label = aggregated.loc[aggregated["image_id"] == left_id].iloc[0]
right_label = aggregated.loc[aggregated["image_id"] == right_id].iloc[0]

col1, col2 = st.columns(2)
with col1:
    st.image(str(resolve_path(left_manifest["relative_path"])), caption=f"{left_id} | mean {left_label['mean_score']:.2f}", use_container_width=True)
with col2:
    st.image(str(resolve_path(right_manifest["relative_path"])), caption=f"{right_id} | mean {right_label['mean_score']:.2f}", use_container_width=True)

if not features.empty:
    left_features = features.loc[features["image_id"] == left_id].drop(columns=["image_id"]).T.rename(columns={features.loc[features["image_id"] == left_id].index[0]: "left"})
    right_features = features.loc[features["image_id"] == right_id].drop(columns=["image_id"]).T.rename(columns={features.loc[features["image_id"] == right_id].index[0]: "right"})
    comparison = left_features.join(right_features)
    comparison["delta"] = comparison["left"] - comparison["right"]
    comparison = comparison.sort_values("delta", key=lambda series: series.abs(), ascending=False).head(20)
    st.subheader("Top Feature Differences")
    st.dataframe(comparison, use_container_width=True)

high_agreement = aggregated.sort_values("agreement_index", ascending=False).head(10)
st.subheader("High Agreement Examples")
st.dataframe(high_agreement[["image_id", "mean_score", "agreement_index", "std_score"]], use_container_width=True)
