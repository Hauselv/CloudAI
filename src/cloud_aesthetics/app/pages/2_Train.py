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


def _torch_runtime_info() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"available": False, "torch": "not installed", "device": "cpu"}
    if torch.cuda.is_available():
        return {
            "available": True,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda": getattr(torch.version, "cuda", None),
        }
    return {
        "available": True,
        "torch": torch.__version__,
        "device": "CPU",
        "cuda": getattr(torch.version, "cuda", None),
        "hip": getattr(torch.version, "hip", None),
    }


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

runtime = _torch_runtime_info()
runtime_col1, runtime_col2, runtime_col3 = st.columns(3)
runtime_col1.metric("Torch", str(runtime.get("torch", "unknown")))
runtime_col2.metric("Training device", str(runtime.get("device", "CPU")))
runtime_col3.metric("CUDA", str(runtime.get("cuda", None) or "not active"))

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
        "configs/models/deep_cnn_rocm.yaml",
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
            device = summary.get("device", {})
            run_col1, run_col2, run_col3, run_col4 = st.columns(4)
            run_col1.metric("Device", str(device.get("name", device.get("type", "unknown"))))
            run_col2.metric("Epochs ran", int(summary.get("epochs_ran", 0)))
            run_col3.metric("Best epoch", int(summary.get("best_epoch", -1)))
            run_col4.metric("AMP", "on" if summary.get("amp") else "off")
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
            evaluation = summary.get("evaluation", {})
            if evaluation:
                eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
                eval_col1.metric("Scheme", str(evaluation.get("scheme", "")).split(" on ")[0])
                eval_col2.metric("Folds", int(evaluation.get("n_folds", 1)))
                eval_col3.metric("Dev images", int(evaluation.get("n_dev", 0)))
                eval_col4.metric("Test images", int(evaluation.get("n_test", 0)))
            rows = []
            for model_name, model_info in summary.get("models", {}).items():
                cv = model_info.get("cv", {})
                for partition, metrics in model_info.get("metrics", {}).items():
                    row = {
                        "model": model_name,
                        "partition": "val (out-of-fold)" if partition == "val" else partition,
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                        "spearman": metrics.get("spearman"),
                    }
                    if partition == "val" and cv.get("std"):
                        row["mae_fold_std"] = cv["std"].get("mae")
                        row["spearman_fold_std"] = cv["std"].get("spearman")
                    rows.append(row)
            metrics_frame = pd.DataFrame(rows)
            if not metrics_frame.empty:
                st.dataframe(metrics_frame.round(3), use_container_width=True, hide_index=True)
                rmse_chart = metrics_frame.pivot(index="model", columns="partition", values="rmse")
                st.bar_chart(rmse_chart)
            group_rows = []
            for model_name, model_info in summary.get("models", {}).items():
                for partition, groups in model_info.get("by_group", {}).items():
                    for column, values in groups.items():
                        for group, metrics in values.items():
                            group_rows.append(
                                {
                                    "model": model_name,
                                    "partition": partition,
                                    "group": f"{column}={group}",
                                    "n": metrics.get("n"),
                                    "mae": metrics.get("mae"),
                                    "spearman": metrics.get("spearman"),
                                    "mean_target": metrics.get("mean_target"),
                                    "mean_prediction": metrics.get("mean_prediction"),
                                }
                            )
            if group_rows:
                st.write("Metrics by image kind and dataset")
                st.caption("Crops and originals are rated very differently; a model that only separates the two groups looks good overall but weak within each group.")
                st.dataframe(pd.DataFrame(group_rows).round(3), use_container_width=True, hide_index=True)
        else:
            st.json(summary)
    else:
        st.info("Selected run does not have a summary.json yet.")
else:
    st.info("No training runs found yet.")

st.subheader("Config Preview")
with resolve_path(model_config).open("r", encoding="utf-8") as handle:
    st.code(handle.read(), language="yaml")
