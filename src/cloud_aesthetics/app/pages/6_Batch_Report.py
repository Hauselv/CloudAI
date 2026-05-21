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
from cloud_aesthetics.explain.interpretation import build_interpretation
from cloud_aesthetics.settings import resolve_path


def _json_value(value: object, fallback: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    if value is None or value == "":
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _names_from_json(value: object, key: str) -> str:
    records = _json_value(value, [])
    if not isinstance(records, list):
        return ""
    names = [str(record.get(key, "")) for record in records[:4] if isinstance(record, dict)]
    return ", ".join(name for name in names if name)


def _row_interpretation(row: pd.Series) -> dict[str, object]:
    interpretation = _json_value(row.get("interpretation", ""), {})
    if isinstance(interpretation, dict) and interpretation:
        return interpretation
    return build_interpretation(
        {
            "predicted_score": row.get("predicted_score"),
            "uncertainty": row.get("uncertainty"),
            "top_features": _json_value(row.get("top_features", ""), []),
            "top_concepts": _json_value(row.get("top_concepts", ""), []),
            "ground_truth": [],
            "scientific_cloud_type": {},
        }
    )


def _prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    predictions = frame.copy()
    predictions["predicted_score"] = pd.to_numeric(predictions["predicted_score"], errors="coerce")
    predictions["uncertainty"] = pd.to_numeric(predictions["uncertainty"], errors="coerce")
    predictions["kind"] = predictions.get("kind", "image")
    interpretations = [_row_interpretation(row) for _, row in predictions.iterrows()]
    predictions["recommendation"] = [str(item.get("recommendation", "review")) for item in interpretations]
    predictions["score_band"] = [str(item.get("score_band", {}).get("label", "")) for item in interpretations]
    predictions["confidence_band"] = [str(item.get("confidence_band", {}).get("label", "")) for item in interpretations]
    predictions["score_interval_low"] = [float(item.get("score_interval", {}).get("low", 0.0)) for item in interpretations]
    predictions["score_interval_high"] = [float(item.get("score_interval", {}).get("high", 0.0)) for item in interpretations]
    predictions["interpretation_summary"] = [str(item.get("summary", "")) for item in interpretations]
    predictions["cloud_type"] = [str(item.get("scientific", {}).get("primary_cloud_type", "unknown")) for item in interpretations]
    return predictions


def _concept_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        records = _json_value(row.get("top_concepts", ""), [])
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                rows.append({"concept": str(record.get("concept", "")), "score": float(record.get("score", 0.0))})
    if not rows:
        return pd.DataFrame(columns=["concept", "mean_score", "count"])
    return (
        pd.DataFrame(rows)
        .groupby("concept")
        .agg(mean_score=("score", "mean"), count=("score", "count"))
        .sort_values(["mean_score", "count"], ascending=False)
        .reset_index()
    )


def _render_tiles(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No images in this group.")
        return
    columns = st.columns(4)
    for index, (_, row) in enumerate(frame.iterrows()):
        with columns[index % 4]:
            image_path = resolve_path(row["relative_path"])
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            st.caption(
                f"{row['image_id']}\n\n"
                f"score {float(row['predicted_score']):.2f} "
                f"({float(row['score_interval_low']):.2f}-{float(row['score_interval_high']):.2f}) | "
                f"{row['recommendation']}"
            )


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
    st.warning("No import batches or capture sessions found yet. Import images or re-run ingestion.")
    st.stop()
if not runs:
    st.warning("No model runs found yet. Train a local model first.")
    st.stop()

col_batch, col_run, col_action = st.columns([1.4, 1.4, 1])
batch_id = col_batch.selectbox("Import batch", batches)
run_id = col_run.selectbox("Model run", runs)
overwrite = col_action.checkbox("Overwrite", value=False)
if str(batch_id).startswith(("session:", "group:")):
    st.caption("Using a legacy manifest fallback batch because no import_batch_id was found.")

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

predictions = _prepare_predictions(predictions)

review_mask = (predictions["recommendation"] == "review") | predictions["confidence_band"].isin(["wide", "very_wide"])
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
metric_col1.metric("Images", len(predictions))
metric_col2.metric("Mean score", f"{predictions['predicted_score'].mean():.2f}")
metric_col3.metric("Top score", f"{predictions['predicted_score'].max():.2f}")
metric_col4.metric("Review first", int(review_mask.sum()))
metric_col5.metric("Most uncertain", f"{predictions['uncertainty'].max():.2f}")

chart_frame = predictions[["image_id", "predicted_score", "uncertainty"]].set_index("image_id")
st.bar_chart(chart_frame[["predicted_score"]])

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1.2, 1])
kind_filter = filter_col1.selectbox("Kind", ["All"] + sorted(predictions["kind"].dropna().astype(str).unique().tolist()))
sort_by = filter_col2.selectbox("Sort", ["Score high", "Score low", "Uncertainty high", "Review first", "Newest"])
query = filter_col3.text_input("Feature/concept filter", value="")
recommendation_filter = filter_col4.selectbox(
    "Action",
    ["All"] + sorted([value for value in predictions["recommendation"].dropna().astype(str).unique().tolist() if value]),
)

view = predictions.copy()
if kind_filter != "All":
    view = view[view["kind"].astype(str) == kind_filter]
if recommendation_filter != "All":
    view = view[view["recommendation"].astype(str) == recommendation_filter]
if query.strip():
    needle = query.strip().lower()
    view = view[
        view["top_features"].fillna("").astype(str).str.lower().str.contains(needle)
        | view["top_concepts"].fillna("").astype(str).str.lower().str.contains(needle)
        | view["interpretation_summary"].fillna("").astype(str).str.lower().str.contains(needle)
    ]
if sort_by == "Score high":
    view = view.sort_values("predicted_score", ascending=False)
elif sort_by == "Score low":
    view = view.sort_values("predicted_score", ascending=True)
elif sort_by == "Uncertainty high":
    view = view.sort_values("uncertainty", ascending=False)
elif sort_by == "Review first":
    view = view.assign(review_rank=view["recommendation"].ne("review")).sort_values(
        ["review_rank", "uncertainty"], ascending=[True, False]
    )
else:
    view = view.sort_values("created_at", ascending=False)

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
            "score_interval_low",
            "score_interval_high",
            "uncertainty",
            "score_band",
            "confidence_band",
            "recommendation",
            "top_feature_names",
            "top_concept_names",
            "relative_path",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Batch Signals")
concepts = _concept_summary(predictions)
summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    if not concepts.empty:
        st.dataframe(concepts.head(8), use_container_width=True, hide_index=True)
    else:
        st.info("No concept summary available.")
with summary_col2:
    cloud_counts = predictions["cloud_type"].value_counts().reset_index()
    cloud_counts.columns = ["cloud_type", "images"]
    st.dataframe(cloud_counts, use_container_width=True, hide_index=True)

st.subheader("Decision Groups")
tab_top, tab_low, tab_uncertain, tab_review, tab_originals, tab_crops = st.tabs(
    ["Top", "Low", "Uncertain", "Review First", "Best Originals", "Best Crops"]
)
with tab_top:
    _render_tiles(predictions.sort_values("predicted_score", ascending=False).head(8))
with tab_low:
    _render_tiles(predictions.sort_values("predicted_score", ascending=True).head(8))
with tab_uncertain:
    _render_tiles(predictions.sort_values("uncertainty", ascending=False).head(8))
with tab_review:
    _render_tiles(predictions[review_mask].sort_values("uncertainty", ascending=False).head(8))
with tab_originals:
    _render_tiles(predictions[predictions["kind"] == "original"].sort_values("predicted_score", ascending=False).head(8))
with tab_crops:
    _render_tiles(predictions[predictions["kind"] == "crop"].sort_values("predicted_score", ascending=False).head(8))

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
    detail_metric1, detail_metric2, detail_metric3 = st.columns(3)
    detail_metric1.metric("Predicted score", f"{float(selected['predicted_score']):.2f}")
    detail_metric2.metric("Likely range", f"{float(selected['score_interval_low']):.2f}-{float(selected['score_interval_high']):.2f}")
    detail_metric3.metric("Action", str(selected["recommendation"]))

    explanation_path = Path(str(selected.get("explanation_path", "")))
    if explanation_path.exists():
        with explanation_path.open("r", encoding="utf-8") as handle:
            explanation = json.load(handle)
        interpretation = explanation.get("interpretation", _row_interpretation(selected))
        st.write(str(interpretation.get("summary", explanation.get("text_explanation", ""))))
        if interpretation.get("cautions"):
            st.warning(" ".join(str(item) for item in interpretation["cautions"]))
        if interpretation.get("drivers"):
            st.write("Main signals")
            st.dataframe(pd.DataFrame(interpretation["drivers"]), use_container_width=True, hide_index=True)
        if explanation.get("top_regions"):
            st.write("Top regions")
            st.dataframe(pd.DataFrame(explanation["top_regions"]), use_container_width=True, hide_index=True)
        st.info(explanation.get("scientific_caveat", ""))
else:
    st.info("No images match the selected filters.")
