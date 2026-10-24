from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, dataset_config
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.judge.store import attributes_table, load_raters, rater_kinds
from cloud_aesthetics.judge.validation import compare_raters

st.set_page_config(page_title="Rater Comparison", layout="wide")
st.title("Rater Comparison")
st.caption("Human raters versus model raters (LLM judges). Model raters never enter the human consensus labels.")

dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])
ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
if ratings.empty:
    st.warning("No ratings yet.")
    st.stop()

kinds = rater_kinds(dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv"))
all_raters = sorted(ratings["rater_id"].astype(str).unique().tolist())
model_raters = [rater for rater in all_raters if kinds.get(rater) == "llm"]
human_raters = [rater for rater in all_raters if kinds.get(rater, "human") == "human"]

registry = load_raters(dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv"))
with st.expander("Rater registry", expanded=False):
    counts = ratings.groupby("rater_id")["image_id"].nunique().rename("images").reset_index()
    counts["kind"] = counts["rater_id"].map(lambda value: kinds.get(str(value), "human"))
    st.dataframe(counts.merge(registry, on="rater_id", how="left"), use_container_width=True, hide_index=True)

if not model_raters:
    st.info("No model rater yet. Run `python -m cloud_aesthetics.cli judge-images` to rate the labeled images with a Claude vision judge.")
    st.stop()

col_model, col_humans = st.columns(2)
model_rater = col_model.selectbox("Model rater", model_raters)
selected_humans = col_humans.multiselect("Human raters (mean per image)", human_raters, default=human_raters)
if not selected_humans:
    st.warning("Select at least one human rater.")
    st.stop()

result = compare_raters(ratings, model_rater=model_rater, human_raters=selected_humans, manifest=manifest)
overall = result["overall"]
if overall.get("n", 0) < 3:
    st.warning("Fewer than three images are rated by both sides.")
    st.stop()

metric_cols = st.columns(6)
metric_cols[0].metric("Images", int(overall["n"]))
metric_cols[1].metric("Spearman", f"{overall['spearman']:.2f}")
metric_cols[2].metric("MAE", f"{overall['mae']:.2f}")
metric_cols[3].metric("Bias (model - human)", f"{overall['bias']:+.2f}")
metric_cols[4].metric("Within ±1", f"{overall['within_1']:.0%}")
metric_cols[5].metric("QWK", f"{overall['qwk']:.2f}")

pairs: pd.DataFrame = result["pairs"]
chart_col, group_col = st.columns([1.3, 1])
with chart_col:
    st.caption("Each point is one image: human mean (x) vs model score (y).")
    color = "crop_class" if "crop_class" in pairs.columns else None
    st.scatter_chart(pairs, x="human_score", y="model_score", color=color, use_container_width=True)
with group_col:
    rows = []
    for column, blocks in result["by_group"].items():
        for group, block in blocks.items():
            rows.append({"group": f"{column}={group}", "n": block["n"], "spearman": block.get("spearman"), "mae": block.get("mae"), "bias": block.get("bias")})
    if rows:
        st.write("Agreement by image group")
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
    bands = result.get("score_bands", {})
    if "human_ge_7_5" in bands:
        block = bands["human_ge_7_5"]
        st.write("Top of the scale (human ≥ 7.5)")
        st.caption(
            f"n={block['n']} · Spearman {block.get('spearman', float('nan')):.2f} · MAE {block.get('mae', float('nan')):.2f}. "
            "This is the part that matters for the 10-vs-8 question."
        )
    if "model_mean_for_human_9plus" in bands:
        st.caption(f"Model mean for human ≥ 9: {bands['model_mean_for_human_9plus']:.2f}; for human < 9: {bands['model_mean_for_human_below_9']:.2f}")

st.subheader("Score distributions")
hist = pd.DataFrame(
    {
        "human": pairs["human_score"].round().value_counts().sort_index(),
        "model": pairs["model_score"].round().value_counts().sort_index(),
    }
).fillna(0)
st.bar_chart(hist)

attributes = attributes_table(model_rater)
if not attributes.empty:
    st.subheader("What the judge saw")
    attr = attributes.merge(pairs[["image_id", "human_score", "model_score"]], on="image_id", how="inner")
    col_type, col_light = st.columns(2)
    type_table = attr.groupby("judge_cloud_type").agg(n=("image_id", "count"), human_mean=("human_score", "mean"), model_mean=("model_score", "mean")).round(2).sort_values("n", ascending=False)
    col_type.write("By judged cloud type")
    col_type.dataframe(type_table, use_container_width=True)
    light_table = attr.groupby("judge_light").agg(n=("image_id", "count"), human_mean=("human_score", "mean"), model_mean=("model_score", "mean")).round(2).sort_values("n", ascending=False)
    col_light.write("By judged light")
    col_light.dataframe(light_table, use_container_width=True)
    flags = attr["judge_flags"].fillna("").str.split("|").explode()
    flags = flags[flags != ""]
    if not flags.empty:
        st.write("Quality flags raised by the judge")
        st.dataframe(flags.value_counts().rename("images").reset_index().rename(columns={"index": "flag"}), use_container_width=True, hide_index=True)

st.subheader("Largest disagreements")
st.caption("Where the model and the human differ most. Reading the judge's verdict next to the photo is the fastest way to learn what the human taste model values.")
disagreements = pd.DataFrame(result["largest_disagreements"])
if not disagreements.empty:
    if not attributes.empty:
        disagreements = disagreements.merge(attributes[["image_id", "judge_verdict", "judge_strengths", "judge_weaknesses"]], on="image_id", how="left")
    if "relative_path" not in disagreements.columns:
        disagreements = disagreements.merge(manifest[["image_id", "relative_path"]].assign(image_id=manifest["image_id"].astype(str)), on="image_id", how="left")
    render_image_grid(
        disagreements,
        columns=3,
        caption=lambda row: (
            f"human {float(row['human_score']):.1f} | model {float(row['model_score']):.1f} ({float(row['difference']):+.1f}) | {row.get('crop_class', '')}\n\n"
            f"{row.get('judge_verdict', '') or ''}"
        ),
    )
