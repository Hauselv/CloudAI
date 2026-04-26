from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import aggregate_labels_impl, dataset_config, extract_features_impl, list_runs, safe_read_table, train_impl
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Train Models", layout="wide")
st.title("Train Models")
st.caption("Run ingestion products in sequence, then launch a baseline or deep experiment.")

dataset_cfg = dataset_config()
labels = safe_read_table(dataset_cfg["aggregated_labels_path"])
splits = safe_read_table(dataset_cfg["splits_path"])

if not labels.empty and not splits.empty:
    st.subheader("Validation Split")
    split_view = splits.merge(labels[["image_id", "mean_score"]], on="image_id", how="left")
    split_counts = split_view.groupby(["partition", "fold"], dropna=False).size().reset_index(name="images")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Labeled Images", len(labels))
    col_b.metric("Dev / CV", int((splits["partition"] == "dev").sum()))
    col_c.metric("Held-out Test", int((splits["partition"] == "test").sum()))
    st.dataframe(split_counts, use_container_width=True)
    chart_frame = split_view.assign(split=lambda frame: frame["partition"] + " fold " + frame["fold"].astype(str))
    st.bar_chart(chart_frame.groupby("split")["image_id"].count())
elif labels.empty:
    st.info("No aggregated labels yet. Use 'Aggregate labels' after labeling some images.")
else:
    st.info("No split table yet. Use 'Aggregate labels' to create validation and test splits.")

col1, col2, col3 = st.columns(3)
if col1.button("Aggregate labels"):
    tables = aggregate_labels_impl()
    st.success(f"Aggregated {len(tables['aggregated'])} image labels.")
if col2.button("Extract features"):
    features = extract_features_impl("configs/features/v1.yaml")
    st.success(f"Extracted features for {len(features)} images.")

model_config = st.selectbox(
    "Model config",
    [
        "configs/models/baseline.yaml",
        "configs/models/deep_cnn.yaml",
        "configs/models/deep_cnn_gradcam.yaml",
        "configs/models/hybrid.yaml",
        "configs/models/ranking.yaml",
    ],
)

if col3.button("Train selected config"):
    summary = train_impl(model_config)
    st.success(f"Finished run {summary['run_id']}.")
    st.json(summary)

st.subheader("Run Metrics")
runs = list_runs()
if runs:
    selected_run = st.selectbox("Run to inspect", runs)
    summary_path = resolve_path("data/artifacts") / selected_run / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("kind") == "deep":
            history = summary.get("metrics", {}).get("history", [])
            rows = []
            for item in history:
                epoch = item.get("epoch", 0)
                for partition in ["train", "val"]:
                    metrics = item.get(partition, {})
                    rows.append(
                        {
                            "epoch": epoch,
                            "partition": partition,
                            "loss": metrics.get("loss"),
                            "mae": metrics.get("mae"),
                            "rmse": metrics.get("rmse"),
                            "r2": metrics.get("r2"),
                        }
                    )
            history_frame = pd.DataFrame(rows)
            if not history_frame.empty:
                st.line_chart(history_frame.pivot(index="epoch", columns="partition", values="loss"))
                st.line_chart(history_frame.pivot(index="epoch", columns="partition", values="rmse"))
                st.dataframe(history_frame, use_container_width=True)
            test_metrics = summary.get("metrics", {}).get("test", {})
            if test_metrics:
                st.write("Held-out test metrics")
                st.json(test_metrics)
        elif summary.get("kind") == "baseline":
            rows = []
            for model_name, model_info in summary.get("models", {}).items():
                for partition, metrics in model_info.get("metrics", {}).items():
                    rows.append(
                        {
                            "model": model_name,
                            "partition": partition,
                            "mae": metrics.get("mae"),
                            "rmse": metrics.get("rmse"),
                            "r2": metrics.get("r2"),
                        }
                    )
            metrics_frame = pd.DataFrame(rows)
            if not metrics_frame.empty:
                st.dataframe(metrics_frame, use_container_width=True)
                rmse_chart = metrics_frame.pivot(index="model", columns="partition", values="rmse")
                st.bar_chart(rmse_chart)
        else:
            st.json(summary)
    else:
        st.info("Selected run does not have a summary.json yet.")
else:
    st.info("No training runs found yet.")

st.subheader("Config Preview")
with resolve_path(model_config).open("r", encoding="utf-8") as handle:
    st.code(handle.read(), language="yaml")
