from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, dataset_config, safe_read_table
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.embeddings.store import model_specs
from cloud_aesthetics.embeddings.zero_shot_store import load_zero_shot
from cloud_aesthetics.eval.semantic_map import ProjectionParams, build_map_table, build_projection, load_judge_tables, load_projection, map_regions, map_summary, thumbnail_data_uri
from cloud_aesthetics.utils.io import read_table

st.set_page_config(page_title="Karte", layout="wide")
st.title("Semantische Karte")
st.caption(
    "Jedes Bild ein Punkt; Nachbarn sehen für das Embedding-Modell ähnlich aus. Einfärben nach deinem Score zeigt, wo die 10er "
    "liegen; einfärben nach „gelabelt“ zeigt, welche Bildwelten noch keine Labels haben. Regionen sind k-Means-Zellen auf der Karte, "
    "sortiert nach Lücke: viele Bilder, kaum Labels, vielversprechende Schätzung."
)

dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])
if manifest.empty:
    st.warning("Kein Manifest gefunden.")
    st.stop()

COLOR_OPTIONS = {
    "dein Score": "mean_score",
    "Schätzung (Label, sonst Modell, sonst Judge)": "estimate",
    "Modell-Vorhersage": "model_score",
    "Judge-Mittel": "judge_mean",
    "gelabelt?": "labeled",
    "Bildart": "kind",
    "Crop-Klasse": "crop_class",
    "Datensatz": "dataset_name",
    "CLIP-Wolkentyp": "clip_cloud_type",
    "CLIP-Licht": "clip_light",
    "Region": "region",
}
CATEGORICAL = {"labeled", "kind", "crop_class", "dataset_name", "clip_cloud_type", "clip_light", "region"}


@st.cache_data(show_spinner="Berechne Projektion ...")
def _projection(model_key: str, method: str, n_neighbors: int, min_dist: float) -> pd.DataFrame:
    params = ProjectionParams(model_key=model_key, method=method, n_neighbors=n_neighbors, min_dist=min_dist)
    cached = load_projection(params)
    return cached if not cached.empty else build_projection(params)


@st.cache_data(show_spinner="Lade Signale ...")
def _signals() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    return read_table(dataset_cfg["aggregated_labels_path"]), load_zero_shot(), load_judge_tables(), safe_read_table("data/processed/predictions/latest.parquet")


@st.cache_data(show_spinner="Erzeuge Vorschaubilder ...")
def _thumbnails(paths: tuple[str, ...], size: int) -> list[str | None]:
    return [thumbnail_data_uri(path, size) for path in paths]


with st.sidebar:
    st.header("Projektion")
    available = [key for key in model_specs() if load_projection(ProjectionParams(model_key=key)).shape[0] > 0 or key == "clip_vit_b32"]
    model_key = st.selectbox("Embedding", available or ["clip_vit_b32"], help="Projektionen werden gecacht; neue Modelle per `semantic-map --model <key>` berechnen.")
    method = st.selectbox("Methode", ["umap", "pca"])
    n_neighbors = st.slider("UMAP-Nachbarn", 5, 50, 15, 5, disabled=method != "umap")
    min_dist = st.select_slider("UMAP min_dist", [0.0, 0.05, 0.1, 0.25, 0.5], value=0.1, disabled=method != "umap")
    st.header("Darstellung")
    color_label = st.selectbox("Einfärben nach", list(COLOR_OPTIONS))
    kinds = sorted(manifest["kind"].dropna().astype(str).unique().tolist())
    kind_filter = st.multiselect("Bildart", kinds, default=kinds)
    only_labeled = st.checkbox("Nur gelabelte Bilder", value=False)
    n_regions = st.slider("Regionen", 4, 30, 12, 1)
    thumbs = st.slider("Vorschaubilder auf der Karte", 0, 300, 0, 25, help="Zeichnet kleine Bilder an die Punkte (Zufallsauswahl, gelabelte bevorzugt). Ab ~150 wird die Karte träge.")
    thumb_size = st.select_slider("Vorschaugröße", [32, 48, 64, 96], value=48)

projection = _projection(model_key, method, n_neighbors, min_dist)
labels, zero_shot, judge_tables, predictions = _signals()
table = build_map_table(projection, manifest, labels=labels, zero_shot=zero_shot, judge_tables=judge_tables, predictions=predictions)
table, regions = map_regions(table, n_regions=n_regions)
view = table.copy()
if kind_filter and "kind" in view.columns:
    view = view[view["kind"].astype(str).isin(kind_filter)]
if only_labeled:
    view = view[view["labeled"]]
if view.empty:
    st.warning("Keine Bilder in dieser Auswahl.")
    st.stop()

summary = map_summary(table, regions)
cols = st.columns(5)
cols[0].metric("Bilder", summary["n_images"])
cols[1].metric("gelabelt", summary["n_labeled"])
cols[2].metric("Regionen ohne Label", f"{summary['regions_without_labels']} / {summary['n_regions']}")
cols[3].metric("Regionen mit ≥ 9", summary["regions_with_tens"])
cols[4].metric("Ungelabelt in Top-3-Lücken", summary["unlabeled_in_top3_gaps"])

tab_map, tab_regions, tab_selection = st.tabs(["Karte", "Regionen und Lücken", "Auswahl"])

color_column = COLOR_OPTIONS[color_label]
plot = view.copy()
plot["score_text"] = plot["mean_score"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "–")
plot["model_text"] = plot["model_score"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "–")
plot["judge_text"] = plot["judge_mean"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "–")
hover = {"image_id": True, "score_text": True, "model_text": True, "judge_text": True, "clip_cloud_type": True, "clip_light": True, "kind": True, "x": False, "y": False}
if color_column in CATEGORICAL:
    plot[color_column] = plot[color_column].astype(str)
    fig = px.scatter(plot, x="x", y="y", color=color_column, hover_data=hover, custom_data=["image_id"], opacity=0.85, labels={"score_text": "dein Score", "model_text": "Modell", "judge_text": "Judges", "clip_cloud_type": "CLIP-Typ", "clip_light": "CLIP-Licht"})
else:
    values = plot[color_column].astype(float)
    missing = plot[values.isna()]
    present = plot[values.notna()]
    fig = px.scatter(present, x="x", y="y", color=color_column, hover_data=hover, custom_data=["image_id"], opacity=0.9, color_continuous_scale="Viridis", range_color=(0, 10) if color_column in {"mean_score", "estimate", "model_score", "judge_mean"} else None, labels={color_column: color_label, "score_text": "dein Score", "model_text": "Modell", "judge_text": "Judges", "clip_cloud_type": "CLIP-Typ", "clip_light": "CLIP-Licht"})
    if not missing.empty:
        fig.add_trace(go.Scatter(x=missing["x"], y=missing["y"], mode="markers", marker={"color": "rgba(150,150,150,0.35)", "size": 6}, name="ohne Wert", customdata=missing[["image_id"]].to_numpy(), hovertemplate="%{customdata[0]}<br>ohne Wert<extra></extra>"))
fig.update_traces(marker={"size": 7}, selector={"mode": "markers"})
if not regions.empty:
    fig.add_trace(go.Scatter(x=regions["x"], y=regions["y"], mode="text", text=[f"R{r}" for r in regions["region"]], textfont={"size": 12, "color": "rgba(80,80,80,0.9)"}, hoverinfo="skip", showlegend=False, name="Regionen"))
if thumbs > 0:
    candidates = plot.sort_values("labeled", ascending=False)
    chosen = pd.concat([candidates[candidates["labeled"]].sample(min(thumbs, int(candidates["labeled"].sum())), random_state=0) if candidates["labeled"].any() else candidates.iloc[0:0], candidates[~candidates["labeled"]]]).head(thumbs)
    uris = _thumbnails(tuple(chosen["relative_path"].astype(str)), int(thumb_size))
    span_x = float(plot["x"].max() - plot["x"].min()) or 1.0
    span_y = float(plot["y"].max() - plot["y"].min()) or 1.0
    size_x = span_x * (thumb_size / 1400.0)
    size_y = span_y * (thumb_size / 900.0)
    for (_, row), uri in zip(chosen.iterrows(), uris):
        if uri:
            fig.add_layout_image({"source": uri, "x": row["x"], "y": row["y"], "xref": "x", "yref": "y", "sizex": size_x, "sizey": size_y, "xanchor": "center", "yanchor": "middle", "layer": "above", "opacity": 0.95})
fig.update_layout(height=720, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"}, legend_title_text="", dragmode="lasso")

with tab_map:
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=("lasso", "box", "points"), key="semantic_map")
    st.caption("Lasso oder Box ziehen, um Bilder auszuwählen; sie erscheinen im Tab „Auswahl“. Graue Punkte haben für die gewählte Färbung keinen Wert.")

selected_ids: list[str] = []
try:
    points = event.selection.points if event is not None else []
    selected_ids = [str(point.get("customdata", [None])[0]) for point in points if point.get("customdata")]
except AttributeError:
    selected_ids = []

with tab_regions:
    if regions.empty:
        st.info("Keine Regionen.")
    else:
        show = regions[["region", "n", "n_labeled", "labeled_share", "n_originals", "mean_score", "max_score", "n_top", "judge_mean", "model_mean", "top_cloud_type", "top_light", "gap_score"]].copy()
        show.columns = ["Region", "Bilder", "gelabelt", "Anteil gelabelt", "Originale", "Ø Score", "max Score", "≥ 9", "Ø Judge", "Ø Modell", "CLIP-Typ", "CLIP-Licht", "Lücke"]
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)
        region_choice = st.selectbox("Region ansehen", regions["region"].tolist(), format_func=lambda r: f"R{r}: {int(regions.set_index('region').loc[r, 'n'])} Bilder, {int(regions.set_index('region').loc[r, 'n_labeled'])} gelabelt, {regions.set_index('region').loc[r, 'top_cloud_type']}")
        region_row = regions.set_index("region").loc[region_choice]
        part = table[table["region"] == region_choice]
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Typische Bilder (nahe der Mitte)")
            typical = part[part["image_id"].isin(region_row["example_ids"])]
            render_image_grid(typical, columns=2, caption=lambda r: f"Score {r['mean_score']:.1f}" if pd.notna(r["mean_score"]) else f"ungelabelt · Modell {r['model_score']:.1f}" if pd.notna(r["model_score"]) else "ungelabelt")
        with col_b:
            st.subheader("Ungelabelte Originale zum Labeln")
            candidates = part[part["image_id"].isin(region_row["unlabeled_candidates"])]
            render_image_grid(candidates, columns=2, caption=lambda r: f"Modell {r['model_score']:.1f} · Judges {r['judge_mean']:.1f}" if pd.notna(r["model_score"]) else f"Judges {r['judge_mean']:.1f}" if pd.notna(r["judge_mean"]) else r["image_id"], empty_text="Keine ungelabelten Originale in dieser Region.")
        labeled_part = part[part["labeled"]].sort_values("mean_score", ascending=False)
        if not labeled_part.empty:
            st.subheader(f"Gelabelte Bilder der Region ({len(labeled_part)})")
            render_image_grid(labeled_part.head(12), columns=6, caption=lambda r: f"Score {r['mean_score']:.1f} · {r['kind']}")

with tab_selection:
    if not selected_ids:
        st.info("Noch keine Auswahl. Auf der Karte Lasso oder Box ziehen.")
    else:
        chosen = table[table["image_id"].isin(selected_ids)].sort_values("estimate", ascending=False)
        st.caption(f"{len(chosen)} Bilder ausgewählt · {int(chosen['labeled'].sum())} gelabelt · Ø Score {chosen['mean_score'].mean():.2f}" if chosen["labeled"].any() else f"{len(chosen)} Bilder ausgewählt, keines gelabelt")
        if chosen["labeled"].any() and "clip_cloud_type" in chosen.columns:
            st.write("CLIP-Wolkentypen in der Auswahl: " + ", ".join(f"{k} ({v})" for k, v in chosen["clip_cloud_type"].value_counts().head(5).items()))
        render_image_grid(chosen.head(36), columns=6, caption=lambda r: (f"Score {r['mean_score']:.1f}" if pd.notna(r["mean_score"]) else f"ungelabelt · Modell {r['model_score']:.1f}" if pd.notna(r["model_score"]) else "ungelabelt") + f" · {r['kind']}\n\n{r['image_id']}")
        st.download_button("Auswahl als CSV", chosen[["image_id", "relative_path", "kind", "mean_score", "model_score", "judge_mean", "clip_cloud_type", "region"]].to_csv(index=False).encode("utf-8"), file_name="karte_auswahl.csv", mime="text/csv")
