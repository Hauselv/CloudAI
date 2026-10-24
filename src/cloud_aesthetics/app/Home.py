from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import app_config, dataset_config, safe_read_table
from cloud_aesthetics.pipeline import pipeline_status, refresh_pipeline

cfg = app_config()
st.set_page_config(page_title=cfg.get("title", "Cloud Aesthetics"), layout="wide")
st.title(cfg.get("title", "Cloud Aesthetics Research Workbench"))
st.caption("Local-first tooling for cloud photo rating, modeling, and explainability.")

dataset_cfg = dataset_config()

STATE_ICONS = {"ok": "✅", "stale": "⚠️", "missing": "⬜"}


def _render_status_panel() -> None:
    st.subheader("Pipeline Status")
    stages = pipeline_status()
    stale = [stage for stage in stages if stage.exists and stage.stale]
    missing = [stage for stage in stages if not stage.exists]
    if stale:
        st.warning("Veraltet: " + ", ".join(stage.label for stage in stale) + ". Die Hinweise in der Tabelle sagen, welcher Schritt fehlt.")
    elif missing:
        st.info("Noch nicht erzeugt: " + ", ".join(stage.label for stage in missing))
    else:
        st.success("Alle Stufen sind aktuell.")

    rows = []
    for stage in stages:
        rows.append(
            {
                "": STATE_ICONS.get(stage.state, "?"),
                "Stufe": stage.label,
                "Zeilen": str(stage.rows) if stage.rows else "",
                "Aktualisiert (UTC)": stage.updated_at.strftime("%Y-%m-%d %H:%M") if stage.updated_at else "",
                "Hinweis": stage.note or (stage.details.get("run_id", "") if stage.key == "latest_run" else ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    col_agg, col_missing, col_full = st.columns(3)
    if col_agg.button("Labels aggregieren", use_container_width=True):
        with st.spinner("Aggregiere Labels und erzeuge Splits ..."):
            report = refresh_pipeline(aggregate=True, features="skip")
        st.success(f"{report.get('aggregated_labels', 0)} Labels, {report.get('splits', 0)} Split-Zeilen.")
        st.rerun()
    if col_missing.button("Fehlende Features berechnen", use_container_width=True):
        with st.spinner("Berechne Features für Bilder ohne Feature-Zeile ..."):
            report = refresh_pipeline(aggregate=False, features="missing")
        st.success(f"Feature-Tabelle hat jetzt {report.get('features', 0)} Zeilen.")
        st.rerun()
    if col_full.button("Pipeline aktualisieren (Labels + fehlende Features)", use_container_width=True):
        with st.spinner("Pipeline wird aktualisiert ..."):
            report = refresh_pipeline(aggregate=True, features="missing")
        st.success(f"{report.get('aggregated_labels', 0)} Labels, {report.get('features', 0)} Feature-Zeilen.")
        st.rerun()
    st.caption("Ein vollständiger Feature-Neuaufbau und das Training laufen über die Seite Train oder die CLI.")


_render_status_panel()

manifest = safe_read_table(dataset_cfg["manifest_path"])
ratings = safe_read_table(dataset_cfg["ratings_path"])
aggregated = safe_read_table(dataset_cfg["aggregated_labels_path"])
pairwise = safe_read_table(dataset_cfg["pairwise_path"])
features = safe_read_table("data/processed/features/features_v1.parquet")

st.subheader("Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Images", len(manifest))
col2.metric("Raw Ratings", len(ratings))
col3.metric("Aggregated Labels", len(aggregated))
col4.metric("Pairwise Examples", len(pairwise))
col5.metric("Feature Rows", len(features))

if not manifest.empty and "crop_class" in manifest.columns:
    st.subheader("Bildarten und Labelstand")
    overview = manifest.copy()
    overview["image_id"] = overview["image_id"].astype(str)
    labeled_ids = set(aggregated["image_id"].astype(str)) if not aggregated.empty else set()
    overview["labeled"] = overview["image_id"].isin(labeled_ids)
    kind_table = (
        overview.groupby(["dataset_name", "crop_class"], dropna=False)
        .agg(images=("image_id", "count"), labeled=("labeled", "sum"))
        .reset_index()
    )
    kind_table["open"] = kind_table["images"] - kind_table["labeled"]
    if not aggregated.empty and "crop_class" in aggregated.columns:
        means = aggregated.groupby(["dataset_name", "crop_class"], dropna=False)["mean_score"].mean().round(2).rename("mean_score")
        kind_table = kind_table.merge(means.reset_index(), on=["dataset_name", "crop_class"], how="left")
    st.caption(
        "small_crop = Ausschnitt unter 50 % der kurzen Kante oder unter 768 px. Für die 10-vs-8-Frage zählen vor allem Originale."
    )
    st.dataframe(kind_table, use_container_width=True, hide_index=True)

if not aggregated.empty:
    st.subheader("Rating Distribution")
    if "kind" in aggregated.columns:
        hist_rows = []
        for kind, group in aggregated.groupby(aggregated["kind"].fillna("unknown")):
            counts = group["mean_score"].round().value_counts().sort_index()
            for score, count in counts.items():
                hist_rows.append({"score": int(score), "kind": str(kind), "images": int(count)})
        hist = pd.DataFrame(hist_rows)
        if not hist.empty:
            st.bar_chart(hist.pivot(index="score", columns="kind", values="images").fillna(0))
            by_kind = aggregated.groupby(aggregated["kind"].fillna("unknown"))["mean_score"].agg(["count", "mean", "std"]).round(2)
            st.caption("Mittlerer Score nach Bildart. Crops und Originale sollten getrennt interpretiert werden.")
            st.dataframe(by_kind, use_container_width=True)
    else:
        st.bar_chart(aggregated["mean_score"])

    st.subheader("Agreement Overview")
    agreement_view = aggregated[["image_id", "mean_score", "std_score", "agreement_index", "n_raters"]].sort_values(
        "agreement_index", ascending=False
    )
    st.dataframe(agreement_view, use_container_width=True)

if cfg.get("show_scientific_caveat", True):
    st.info(
        "Scientific caveat: model explanations show attribution patterns associated with predictions, not proven causal reasons for human preference."
    )
