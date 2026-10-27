from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import current_rater, active_manifest, app_config, dataset_config
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.data.exclusions import set_exclusion, set_exclusions
from cloud_aesthetics.data.manifest import image_kind_from_path
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings

cfg = app_config()
st.set_page_config(page_title="Source Groups", layout="wide")
st.title("Source Groups")

dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])

if manifest.empty:
    st.warning("No active images found.")
    st.stop()

ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
rater_id = st.text_input("Rater ID", value=current_rater(cfg))
rated_ids = set()
if not ratings.empty:
    rated_ids = set(ratings.loc[ratings["rater_id"] == rater_id, "image_id"].dropna().astype(str))

view = manifest.copy()
view["image_id"] = view["image_id"].astype(str)
view["split_group_id"] = view["split_group_id"].fillna("ungrouped").astype(str)
if "import_batch_id" not in view.columns:
    view["import_batch_id"] = ""
view["rated"] = view["image_id"].isin(rated_ids)
if "kind" not in view.columns:
    view["kind"] = view["relative_path"].map(image_kind_from_path)
view["kind"] = view["kind"].fillna(view["relative_path"].map(image_kind_from_path)).astype(str)

group_summary = (
    view.groupby("split_group_id")
    .agg(
        images=("image_id", "count"),
        originals=("kind", lambda values: int((values == "original").sum())),
        crops=("kind", lambda values: int((values == "crop").sum())),
        labeled=("rated", "sum"),
        batch=("import_batch_id", lambda values: next((str(value) for value in values if pd.notna(value) and str(value)), "")),
    )
    .reset_index()
)
group_summary["open"] = group_summary["images"] - group_summary["labeled"]

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.2, 1.2, 1, 1])
status_filter = filter_col1.selectbox("Group status", ["All", "Has open", "Fully labeled"])
batch_values = ["All"] + sorted([batch for batch in group_summary["batch"].dropna().astype(str).unique().tolist() if batch])
batch_filter = filter_col2.selectbox("Batch", batch_values)
sort_by = filter_col3.selectbox("Sort", ["Most images", "Most open", "Most labeled", "Group id"])
groups_per_page = filter_col4.selectbox("Groups per page", [10, 25, 50], index=1)

groups = group_summary.copy()
if status_filter == "Has open":
    groups = groups[groups["open"] > 0]
elif status_filter == "Fully labeled":
    groups = groups[groups["open"] == 0]
if batch_filter != "All":
    groups = groups[groups["batch"] == batch_filter]

if sort_by == "Most images":
    groups = groups.sort_values(["images", "open"], ascending=False)
elif sort_by == "Most open":
    groups = groups.sort_values(["open", "images"], ascending=False)
elif sort_by == "Most labeled":
    groups = groups.sort_values(["labeled", "images"], ascending=False)
else:
    groups = groups.sort_values("split_group_id")

groups = groups.reset_index(drop=True)
page_count = max(1, int((len(groups) + groups_per_page - 1) // groups_per_page))
if int(st.session_state.get("source_group_page", 1)) > page_count:
    st.session_state.source_group_page = 1
page = st.number_input("Group page", min_value=1, max_value=page_count, value=1, key="source_group_page")
group_page = groups.iloc[(int(page) - 1) * groups_per_page : int(page) * groups_per_page]

st.caption(f"Showing {len(group_page)} of {len(groups)} groups | active images {len(view)}")
st.dataframe(group_page, use_container_width=True)

if group_page.empty:
    st.info("No source groups match the filters.")
    st.stop()

group_options = group_page["split_group_id"].tolist()
selected_group = st.selectbox(
    "Selected source group",
    group_options,
    format_func=lambda group_id: (
        f"{group_id} | "
        f"{int(group_page.loc[group_page['split_group_id'] == group_id, 'images'].iloc[0])} images | "
        f"{int(group_page.loc[group_page['split_group_id'] == group_id, 'open'].iloc[0])} open"
    ),
)

group_frame = view[view["split_group_id"] == selected_group].sort_values(["kind", "relative_path"]).reset_index(drop=True)
summary_row = group_summary[group_summary["split_group_id"] == selected_group].iloc[0]

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Images", int(summary_row["images"]))
metric_col2.metric("Originals", int(summary_row["originals"]))
metric_col3.metric("Crops", int(summary_row["crops"]))
metric_col4.metric("Open", int(summary_row["open"]))

action_col1, action_col2 = st.columns([1, 3])
exclude_reason = action_col2.selectbox(
    "Group exclude reason",
    ["not_cloud_training_data", "satellite", "airplane_window", "too_much_landscape", "duplicate", "bad_crop", "other"],
)
if action_col1.button("Exclude whole group", type="primary", use_container_width=True):
    set_exclusions(group_frame[["image_id", "relative_path"]].to_dict("records"), excluded=True, reason=exclude_reason)
    st.success(f"Excluded {len(group_frame)} images from group {selected_group}.")
    st.rerun()

st.subheader("Group Images")


def _exclude_image_action(row: pd.Series, index: int) -> None:
    safe_path_key = str(row["relative_path"]).replace("\\", "_").replace("/", "_").replace(":", "_")
    if st.button("Exclude image", key=f"exclude_{selected_group}_{index}_{row['image_id']}_{safe_path_key}", use_container_width=True):
        set_exclusion(row["image_id"], excluded=True, reason=exclude_reason, relative_path=str(row["relative_path"]))
        st.success(f"Excluded {row['image_id']}.")
        st.rerun()


render_image_grid(
    group_frame,
    caption=lambda row: f"{'labeled' if bool(row['rated']) else 'open'} | {row['kind']}\n\n{row['image_id']}",
    action=_exclude_image_action,
)
