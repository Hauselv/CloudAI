from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, app_config, dataset_config
from cloud_aesthetics.app.ui import render_image_grid, tile_columns
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings

cfg = app_config()
st.set_page_config(page_title="Image Gallery", layout="wide")
st.title("Image Gallery")

dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])

if manifest.empty:
    st.warning("No images found yet.")
    st.stop()

ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
rater_id = st.text_input("Rater ID", value=str(cfg.get("default_rater_id", "friend_a")))
rated_ids = set()
if not ratings.empty:
    rated_ids = set(ratings.loc[ratings["rater_id"] == rater_id, "image_id"].dropna().astype(str))

view = manifest.copy()
view["image_id"] = view["image_id"].astype(str)
view["rated"] = view["image_id"].isin(rated_ids)
view["kind"] = view["relative_path"].apply(lambda value: "crop" if "/crops/" in str(value).replace("\\", "/") else "original")

col1, col2, col3, col4 = st.columns([1.3, 1.3, 1, 1])
status_filter = col1.selectbox("Status", ["All", "Labeled", "Open"])
kind_filter = col2.selectbox("Kind", ["All", "Originals", "Crops"])
page_size = col3.selectbox("Tiles per page", [24, 48, 72, 96], index=1)

if status_filter == "Labeled":
    view = view[view["rated"]]
elif status_filter == "Open":
    view = view[~view["rated"]]

if kind_filter == "Originals":
    view = view[view["kind"] == "original"]
elif kind_filter == "Crops":
    view = view[view["kind"] == "crop"]

view = view.reset_index(drop=True)
page_count = max(1, int((len(view) + page_size - 1) // page_size))
page = col4.number_input("Page", min_value=1, max_value=page_count, value=1)
start = (int(page) - 1) * page_size
page_frame = view.iloc[start : start + page_size]

st.caption(f"Showing {len(page_frame)} of {len(view)} images | labeled {int(view['rated'].sum())} | open {int((~view['rated']).sum())}")

render_image_grid(
    page_frame,
    columns=tile_columns(page_size),
    caption=lambda row: f"{'labeled' if bool(row['rated']) else 'open'} | {row['kind']}\n\n{row['image_id']}",
    empty_text="No images match the selected filters.",
)
