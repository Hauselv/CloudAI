from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import (
    analyze_batch_impl,
    batch_predictions_path,
    dataset_config,
    list_import_batches,
    list_runs,
    safe_read_table,
)
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Batch Report", layout="wide")
st.title("Batch Report")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])
runs = [run for run in list_runs() if (resolve_path("data/artifacts") / run / "summary.json").exists()]
batches = list_import_batches()

if manifest.empty:
    st.warning("No manifest found yet. Import images first.")
    st.stop()
if not batches:
    st.warning("No import batches found yet. Import images with the current importer first.")
    st.stop()
if not runs:
    st.warning("No model runs found yet. Train a local model first.")
    st.stop()

col_batch, col_run, col_action = st.columns([1.4, 1.4, 1])
batch_id = col_batch.selectbox("Import batch", batches)
run_id = col_run.selectbox("Model run", runs)
overwrite = col_action.checkbox("Overwrite", value=False)

predictions_path = batch_predictions_path(run_id, batch_id)
if st.button("Analyze batch", type="primary"):
    with st.spinner("Analyzing batch..."):
        summary = analyze_batch_impl(run_id, batch_id, overwrite=overwrite)
    st.success(
        f"Analyzed {summary['analyzed_count']} images. "
        f"Skipped {summary['skipped_count']} existing predictions."
    )

predictions = safe_read_table(predictions_path)
if predictions.empty:
    st.info("No predictions for this batch/run yet. Click Analyze batch.")
    st.stop()

predictions = predictions.copy()
predictions["predicted_score"] = pd.to_numeric(predictions["predicted_score"], errors="coerce")
predictions["uncertainty"] = pd.to_numeric(predictions["uncertainty"], errors="coerce")
predictions["kind"] = predictions.get("kind", "image")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Images", len(predictions))
metric_col2.metric("Mean score", f"{predictions['predicted_score'].mean():.2f}")
metric_col3.metric("Top score", f"{predictions['predicted_score'].max():.2f}")
metric_col4.metric("Most uncertain", f"{predictions['uncertainty'].max():.2f}")

chart_frame = predictions[["image_id", "predicted_score", "uncertainty"]].set_index("image_id")
st.bar_chart(chart_frame[["predicted_score"]])

filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.2])
kind_filter = filter_col1.selectbox("Kind", ["All"] + sorted(predictions["kind"].dropna().astype(str).unique().tolist()))
sort_by = filter_col2.selectbox("Sort", ["Score high", "Score low", "Uncertainty high", "Newest"])
query = filter_col3.text_input("Feature/concept filter", value="")

view = predictions.copy()
if kind_filter != "All":
    view = view[view["kind"].astype(str) == kind_filter]
if query.strip():
    needle = query.strip().lower()
    view = view[
        view["top_features"].fillna("").astype(str).str.lower().str.contains(needle)
        | view["top_concepts"].fillna("").astype(str).str.lower().str.contains(needle)
    ]
if sort_by == "Score high":
    view = view.sort_values("predicted_score", ascending=False)
elif sort_by == "Score low":
    view = view.sort_values("predicted_score", ascending=True)
elif sort_by == "Uncertainty high":
    view = view.sort_values("uncertainty", ascending=False)
else:
    view = view.sort_values("created_at", ascending=False)


def _names_from_json(value: object, key: str) -> str:
    try:
        records = json.loads(str(value)) if value else []
    except json.JSONDecodeError:
        return ""
    names = [str(record.get(key, "")) for record in records[:4] if isinstance(record, dict)]
    return ", ".join(name for name in names if name)


table = view.copy()
table["top_feature_names"] = table["top_features"].apply(lambda value: _names_from_json(value, "feature"))
table["top_concept_names"] = table["top_concepts"].apply(lambda value: _names_from_json(value, "concept"))
st.subheader("Ranked Images")
st.dataframe(
    table[
        [
            "image_id",
            "kind",
            "predicted_score",
            "uncertainty",
            "top_feature_names",
            "top_concept_names",
            "relative_path",
        ]
    ],
    use_container_width=True,
)

st.subheader("Top / Low / Uncertain")
tab_top, tab_low, tab_uncertain = st.tabs(["Top", "Low", "Uncertain"])
for tab, frame in [
    (tab_top, predictions.sort_values("predicted_score", ascending=False).head(8)),
    (tab_low, predictions.sort_values("predicted_score", ascending=True).head(8)),
    (tab_uncertain, predictions.sort_values("uncertainty", ascending=False).head(8)),
]:
    with tab:
        columns = st.columns(4)
        for index, (_, row) in enumerate(frame.iterrows()):
            with columns[index % 4]:
                image_path = resolve_path(row["relative_path"])
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                st.caption(f"{row['image_id']}\n\nscore {row['predicted_score']:.2f} | +/- {row['uncertainty']:.2f}")

st.subheader("Image Detail")
image_ids = view["image_id"].astype(str).tolist()
if image_ids:
    selected_image = st.selectbox("Image", image_ids)
    selected = predictions[predictions["image_id"].astype(str) == selected_image].iloc[0]
    detail_col1, detail_col2 = st.columns(2)
    image_path = resolve_path(selected["relative_path"])
    if image_path.exists():
        detail_col1.image(str(image_path), caption="Original", use_container_width=True)
    overlay_path = Path(str(selected.get("overlay_path", "")))
    if overlay_path.exists():
        detail_col2.image(str(overlay_path), caption="Explanation overlay", use_container_width=True)
    detail_metric1, detail_metric2 = st.columns(2)
    detail_metric1.metric("Predicted score", f"{float(selected['predicted_score']):.2f}")
    detail_metric2.metric("Uncertainty", f"+/- {float(selected['uncertainty']):.2f}")

    explanation_path = Path(str(selected.get("explanation_path", "")))
    if explanation_path.exists():
        with explanation_path.open("r", encoding="utf-8") as handle:
            explanation = json.load(handle)
        st.write(explanation.get("text_explanation", ""))
        if explanation.get("top_features"):
            st.write("Top features")
            st.dataframe(pd.DataFrame(explanation["top_features"]), use_container_width=True)
        if explanation.get("top_concepts"):
            st.write("Top concepts")
            st.dataframe(pd.DataFrame(explanation["top_concepts"]), use_container_width=True)
        if explanation.get("top_regions"):
            st.write("Top regions")
            st.dataframe(pd.DataFrame(explanation["top_regions"]), use_container_width=True)
        st.info(explanation.get("scientific_caveat", ""))
else:
    st.info("No images match the selected filters.")
