from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import app_config, dataset_config, safe_read_table
from cloud_aesthetics.data.ratings import record_pairwise_preference, record_rating
from cloud_aesthetics.settings import resolve_path

cfg = app_config()
st.set_page_config(page_title="Rate Images", layout="wide")
st.title("Rate Cloud Images")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])

if manifest.empty:
    st.warning("No images found yet. Run the ingest command first and place images in data/raw/images.")
    st.stop()

rater_id = st.text_input("Rater ID", value=str(cfg.get("default_rater_id", "friend_a")))
rating_session_id = st.text_input("Rating Session ID", value=datetime.utcnow().strftime("%Y%m%d_session"))

tab_scalar, tab_pairwise = st.tabs(["Scalar Rating", "Pairwise Preference"])

with tab_scalar:
    image_id = st.selectbox("Image", manifest["image_id"].tolist(), format_func=lambda item: f"{item} - {manifest.loc[manifest['image_id'] == item, 'relative_path'].iloc[0]}")
    row = manifest.loc[manifest["image_id"] == image_id].iloc[0]
    st.image(str(resolve_path(row["relative_path"])), use_container_width=True)
    score = st.slider("Score", min_value=1, max_value=10, value=8)
    note = st.text_area("Optional notes")
    if st.button("Save scalar rating"):
        record_rating(dataset_cfg["ratings_dir"], image_id=image_id, rater_id=rater_id, score=score, rating_session_id=rating_session_id, note=note)
        st.success(f"Saved rating for {image_id}.")

with tab_pairwise:
    left_image_id = st.selectbox("Left image", manifest["image_id"].tolist(), key="left_image")
    right_choices = [item for item in manifest["image_id"].tolist() if item != left_image_id]
    right_image_id = st.selectbox("Right image", right_choices, key="right_image")
    col1, col2 = st.columns(2)
    with col1:
        left_row = manifest.loc[manifest["image_id"] == left_image_id].iloc[0]
        st.image(str(resolve_path(left_row["relative_path"])), caption=f"Left: {left_image_id}", use_container_width=True)
    with col2:
        right_row = manifest.loc[manifest["image_id"] == right_image_id].iloc[0]
        st.image(str(resolve_path(right_row["relative_path"])), caption=f"Right: {right_image_id}", use_container_width=True)
    preference = st.radio("Which image is better?", [left_image_id, right_image_id, "Tie"], horizontal=True)
    strength = st.slider("Preference strength", min_value=0.0, max_value=1.0, value=0.5)
    if st.button("Save pairwise preference"):
        winner = None if preference == "Tie" else preference
        record_pairwise_preference(
            dataset_cfg["pairwise_dir"],
            left_image_id=left_image_id,
            right_image_id=right_image_id,
            rater_id=rater_id,
            winner=winner,
            tie_flag=preference == "Tie",
            preference_strength=strength,
        )
        st.success("Saved pairwise annotation.")
