from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import analyze_batch_impl, dataset_config, import_images_impl, list_runs, safe_read_table
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Import Images", layout="wide")
st.title("Import Cloud Images")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])

source = st.text_input("Source folder", value="")
dataset_name = st.text_input("Dataset name", value="private_clouds")

col1, col2 = st.columns(2)
with col1:
    make_crops = st.checkbox("Generate sky/cloud crops", value=True)
    max_crops = st.slider("Max crops per image", min_value=0, max_value=24, value=8)
with col2:
    min_sky = st.slider("Minimum sky fraction", min_value=0.0, max_value=1.0, value=0.72, step=0.01)
    min_cloud = st.slider("Minimum cloud fraction", min_value=0.0, max_value=1.0, value=0.08, step=0.01)

st.subheader("Optional Batch Analysis")
runs = [run for run in list_runs() if (resolve_path("data/artifacts") / run / "summary.json").exists()]
analyze_after_import = st.checkbox("Analyse nach Import starten", value=False, disabled=not runs)
analysis_run = None
if analyze_after_import and runs:
    analysis_run = st.selectbox("Local model run", runs)
elif analyze_after_import:
    st.info("Train a local model before starting analysis after import.")

if st.button("Import images", type="primary"):
    if not source.strip():
        st.error("Please enter a source folder.")
    elif not Path(source).exists():
        st.error("Source folder does not exist.")
    elif not dataset_name.strip():
        st.error("Please enter a dataset name.")
    else:
        with st.spinner("Importing images and generating crops..."):
            summary = import_images_impl(
                source.strip(),
                dataset_name.strip(),
                make_crops=make_crops,
                max_crops_per_image=max_crops,
                min_sky_fraction=min_sky,
                min_cloud_fraction=min_cloud,
            )
            analysis_summary = None
            if analyze_after_import and analysis_run and summary.get("batch_id"):
                analysis_summary = analyze_batch_impl(analysis_run, str(summary["batch_id"]))
        st.success(
            f"Imported {summary['original_count']} originals and {summary['crop_count']} crops. "
            f"Manifest now contains {summary['manifest_count']} images. Batch: {summary['batch_id']}"
        )
        if analysis_summary:
            st.success(
                f"Analyzed {analysis_summary['analyzed_count']} images "
                f"({analysis_summary['skipped_count']} skipped)."
            )

st.subheader("Current Manifest")
st.metric("Images ready for labeling", len(manifest))
if not manifest.empty:
    columns = ["image_id", "relative_path", "width", "height", "split_group_id"]
    if "import_batch_id" in manifest.columns:
        columns.insert(2, "import_batch_id")
    st.dataframe(manifest[columns], use_container_width=True)
