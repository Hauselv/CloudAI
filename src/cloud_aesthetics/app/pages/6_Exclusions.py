from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.cli import ingest_images_impl
from cloud_aesthetics.data.exclusions import load_exclusions, set_exclusion

st.set_page_config(page_title="Excluded Images", layout="wide")
st.title("Excluded Images")

exclusions = load_exclusions()
if exclusions.empty:
    st.info("No excluded images yet.")
    st.stop()

latest = exclusions.drop_duplicates(subset=["image_id"], keep="last")
active = latest[latest["excluded"].astype(bool)].reset_index(drop=True)

st.metric("Currently excluded", len(active))
if active.empty:
    st.info("No images are currently excluded.")
    st.stop()

st.dataframe(active, use_container_width=True)

selected = st.selectbox(
    "Restore image",
    active["image_id"].astype(str).tolist(),
    format_func=lambda image_id: f"{image_id} - {active.loc[active['image_id'].astype(str) == image_id, 'relative_path'].iloc[0]}",
)

if st.button("Restore to training pool", type="primary"):
    row = active.loc[active["image_id"].astype(str) == selected].iloc[0]
    try:
        set_exclusion(selected, excluded=False, reason="restore", relative_path=str(row.get("relative_path", "")))
    except TypeError:
        set_exclusion(selected, excluded=False, reason="restore")
    ingest_images_impl()
    st.success(f"Restored {selected}.")
    st.rerun()
