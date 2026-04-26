from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import app_config, dataset_config, safe_read_table
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.settings import resolve_path

cfg = app_config()
st.set_page_config(page_title="Image Gallery", layout="wide")
st.title("Image Gallery")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])

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

columns = st.columns(4)
for index, (_, row) in enumerate(page_frame.iterrows()):
    with columns[index % 4]:
        status = "labeled" if bool(row["rated"]) else "open"
        st.image(str(resolve_path(row["relative_path"])), use_container_width=True)
        st.caption(f"{status} | {row['kind']}\n\n{row['image_id']}")

if page_frame.empty:
    st.info("No images match the selected filters.")
