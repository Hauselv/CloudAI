from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import aggregate_labels_impl, dataset_config, extract_features_impl, list_runs, safe_read_table, train_impl
from cloud_aesthetics.eval.calibration import calibration_table, residual_summary
from cloud_aesthetics.pipeline import run_leaderboard
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
        "configs/models/embedding_clip.yaml",
        "configs/models/embedding_clip_originals.yaml",
        "configs/models/embedding_dinov2.yaml",
        "configs/models/embedding_hybrid.yaml",
        "configs/models/embedding_clip_bt.yaml",
        "configs/models/ranking_clip.yaml",
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

st.subheader("Leaderboard")
leaderboard = run_leaderboard()
if leaderboard.empty:
    st.info("No runs with metrics yet.")
else:
    st.caption(
        "Sorted by out-of-fold Spearman on the dev partition. Compare the per-kind columns: a model that only "
        "separates crops from originals scores well overall but poorly within originals."
    )
    st.dataframe(leaderboard.round(3), use_container_width=True, hide_index=True)

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
        elif summary.get("kind") in {"baseline", "embedding", "concept"}:
            if summary.get("kind") == "concept":
                info = summary.get("concept_model", {})
                st.caption(f"Konzeptmodell: {info.get('n_active', 0)} von {info.get('n_concepts', 0)} Konzepten behalten; Koeffizienten und Beiträge auf der Seite Konzeptmodell.")
            if summary.get("kind") == "embedding":
                st.caption(
                    "Embedding run: "
                    + ", ".join(summary.get("embedding_models", []))
                    + (" + hand-crafted features" if summary.get("include_features") else "")
                    + (" + CLIP zero-shot" if summary.get("include_zero_shot") else "")
                )
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
                    if model_info.get("fitted_params"):
                        row["params"] = ", ".join(f"{key}={value}" for key, value in model_info["fitted_params"].items())
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

            predictions_path = summary.get("predictions_path")
            if predictions_path and Path(predictions_path).exists():
                st.subheader("Kalibrierung")
                predictions = pd.read_parquet(predictions_path)
                target_column = str(summary.get("config", {}).get("target_column", "mean_score"))
                model_names = list(summary.get("models", {}).keys())
                cal_col1, cal_col2 = st.columns([1, 1])
                cal_model = cal_col1.selectbox("Modell", model_names, key="cal_model")
                cal_part = cal_col2.selectbox("Partition", ["dev (out-of-fold)", "test"], key="cal_part")
                column = f"oof_{cal_model}" if cal_part.startswith("dev") else f"pred_{cal_model}"
                part = predictions[predictions["partition"] == ("dev" if cal_part.startswith("dev") else "test")].dropna(subset=[column, target_column])
                if len(part) >= 5 and target_column in part.columns:
                    half_width = summary["models"][cal_model].get("conformal_half_width")
                    stats = residual_summary(part[target_column].to_numpy(), part[column].to_numpy(), half_width)
                    stat_cols = st.columns(6)
                    stat_cols[0].metric("n", stats["n"])
                    stat_cols[1].metric("Bias (ist − vorhergesagt)", f"{stats['bias']:+.2f}")
                    stat_cols[2].metric("MAE", f"{stats['mae']:.2f}")
                    stat_cols[3].metric("Streuung Vorhersage/Ist", f"{stats['spread_ratio']:.2f}", help="Unter 1 = das Modell drückt die Skala zusammen (typisch bei Regression zur Mitte).")
                    stat_cols[4].metric("ECE", f"{stats['ece']:.2f}", help="Erwarteter Kalibrierungsfehler über Vorhersage-Bins.")
                    if "coverage" in stats:
                        stat_cols[5].metric("Intervall-Abdeckung", f"{stats['coverage']:.0%}", help=f"Anteil der Bilder innerhalb ±{stats['half_width']:.1f} (Konformal-Halbbreite, Ziel 90 %).")
                    cal_table = calibration_table(part[target_column].to_numpy(), part[column].to_numpy())
                    plot_col1, plot_col2 = st.columns(2)
                    with plot_col1:
                        scatter_frame = part[[column, target_column] + [c for c in ("kind", "crop_class", "image_id") if c in part.columns]].rename(columns={column: "vorhergesagt", target_column: "ist"})
                        fig = px.scatter(scatter_frame, x="vorhergesagt", y="ist", color="crop_class" if "crop_class" in scatter_frame.columns else None, hover_data=["image_id"] if "image_id" in scatter_frame.columns else None, opacity=0.7, range_x=[0, 10], range_y=[0, 10])
                        fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line={"color": "gray", "dash": "dash"})
                        if not cal_table.empty:
                            fig.add_scatter(x=cal_table["mean_predicted"], y=cal_table["mean_actual"], mode="lines+markers", name="Mittel je Bin", line={"color": "#e45756", "width": 3})
                        fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend_title_text="")
                        st.plotly_chart(fig, use_container_width=True)
                    with plot_col2:
                        residuals = (part[target_column] - part[column]).rename("Residuum (ist − vorhergesagt)")
                        hist = px.histogram(residuals, nbins=24, labels={"value": "Residuum (ist − vorhergesagt)"})
                        hist.update_layout(showlegend=False, margin={"l": 10, "r": 10, "t": 30, "b": 10})
                        st.plotly_chart(hist, use_container_width=True)
                    if not cal_table.empty:
                        st.dataframe(cal_table.round(2), use_container_width=True, hide_index=True)
                else:
                    st.info("Zu wenige Vorhersagen für eine Kalibrierungsgrafik.")

            shap_models = {name: info for name, info in summary.get("models", {}).items() if info.get("shap_top")}
            if shap_models:
                st.subheader("Globale SHAP-Wichtigkeit")
                shap_model = st.selectbox("Modell (SHAP)", list(shap_models), key="shap_model")
                shap_frame = pd.DataFrame(shap_models[shap_model]["shap_top"])
                shap_frame["Richtung"] = shap_frame["direction"].map(lambda value: "höherer Wert → höherer Score" if value > 0.1 else ("höherer Wert → niedrigerer Score" if value < -0.1 else "gemischt"))
                fig = px.bar(shap_frame.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h", color="direction", color_continuous_scale="RdBu", range_color=[-1, 1], labels={"mean_abs_shap": "mittlerer |SHAP| (Score-Punkte)", "feature": "", "direction": "Richtung"}, height=max(360, 22 * len(shap_frame) + 100))
                fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Mittlerer absoluter SHAP-Beitrag über die Dev-Bilder; die Farbe zeigt, ob höhere Feature-Werte den Score heben (blau) oder senken (rot).")
        elif summary.get("kind") == "ranking":
            st.caption(
                "Ranking run: "
                + (" + ".join(summary.get("embedding_models", [])) or "hand-crafted features")
                + f" · Paare: {'/'.join(summary.get('pair_sources', []))} · {summary.get('train_pairs', 0)} Trainingspaare, {summary.get('test_pairs', 0)} Testpaare"
            )
            for model_name, info in summary.get("models", {}).items():
                test = info.get("metrics", {}).get("test", {})
                agreement = info.get("agreement", {})
                rank_cols = st.columns(4)
                rank_cols[0].metric("Paar-Genauigkeit (Test)", f"{test.get('pairwise_accuracy', float('nan')):.3f}")
                rank_cols[1].metric("Paar-AUC (Test)", f"{test.get('pairwise_auc', float('nan')):.3f}")
                rank_cols[2].metric("ρ Bildscore vs. Mensch (Test)", f"{agreement.get('test', {}).get('spearman_vs_mean_score', float('nan')):.3f}")
                rank_cols[3].metric("ρ Bildscore vs. BT (Test)", f"{agreement.get('test', {}).get('spearman_vs_bt_score', float('nan')):.3f}" if "spearman_vs_bt_score" in agreement.get("test", {}) else "–", help="Erst verfügbar, wenn Duelle vorliegen.")
                scores_path = info.get("rank_scores_path")
                if scores_path and Path(scores_path).exists():
                    scores = pd.read_parquet(scores_path).dropna(subset=["mean_score"])
                    fig = px.scatter(scores, x="rank_score", y="mean_score", color="partition", hover_data=["image_id"], labels={"rank_score": "latenter Rang-Score", "mean_score": "menschlicher Score"}, opacity=0.7)
                    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(summary)
    else:
        st.info("Selected run does not have a summary.json yet.")
else:
    st.info("No training runs found yet.")

st.subheader("Config Preview")
with resolve_path(model_config).open("r", encoding="utf-8") as handle:
    st.code(handle.read(), language="yaml")
