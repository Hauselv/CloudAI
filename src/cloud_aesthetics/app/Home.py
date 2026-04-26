from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import app_config, dataset_config, safe_read_table

cfg = app_config()
st.set_page_config(page_title=cfg.get("title", "Cloud Aesthetics"), layout="wide")
st.title(cfg.get("title", "Cloud Aesthetics Research Workbench"))
st.caption("Local-first tooling for cloud photo rating, modeling, and explainability.")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])
ratings = safe_read_table(dataset_cfg["ratings_path"])
aggregated = safe_read_table(dataset_cfg["aggregated_labels_path"])
pairwise = safe_read_table(dataset_cfg["pairwise_path"])
features = safe_read_table("data/processed/features/features_v1.parquet")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Images", len(manifest))
col2.metric("Raw Ratings", len(ratings))
col3.metric("Aggregated Labels", len(aggregated))
col4.metric("Pairwise Examples", len(pairwise))
col5.metric("Feature Rows", len(features))

if not aggregated.empty:
    st.subheader("Rating Distribution")
    st.bar_chart(aggregated["mean_score"])

if not aggregated.empty:
    st.subheader("Agreement Overview")
    agreement_view = aggregated[["image_id", "mean_score", "std_score", "agreement_index", "n_raters"]].sort_values(
        "agreement_index", ascending=False
    )
    st.dataframe(agreement_view, use_container_width=True)

if cfg.get("show_scientific_caveat", True):
    st.info(
        "Scientific caveat: model explanations show attribution patterns associated with predictions, not proven causal reasons for human preference."
    )
