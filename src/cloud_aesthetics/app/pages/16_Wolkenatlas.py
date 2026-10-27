from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, dataset_config, safe_read_table
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.embeddings.zero_shot_store import load_zero_shot
from cloud_aesthetics.eval.cloud_atlas import CLOUD_TYPES, TYPE_NAMES, atlas_summary, atlas_table, consensus_cloud_types, reliability_report, type_by_light
from cloud_aesthetics.eval.semantic_map import load_judge_tables
from cloud_aesthetics.utils.io import read_table

st.set_page_config(page_title="Wolkenatlas", layout="wide")
st.title("Wolkenatlas")
st.caption(
    "Die Fotos nach Wolkentyp, mit Meteorologie und deinen Noten. Der Typ ist ein Mehrheitsvotum aus CLIP-Zero-Shot und den "
    "beiden LLM-Judges (Primärtyp 1 Stimme, Nebentyp ½); „sicher“ heißt, alle drei Primärtypen stimmen überein. Das ist eine "
    "Bildschätzung, keine Beobachtung, und die Quellen sind sich nur mäßig einig: unsichere Bilder sind standardmäßig ausgeblendet."
)

dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])
if manifest.empty:
    st.warning("Kein Manifest.")
    st.stop()


@st.cache_data(show_spinner="Zähle Stimmen ...")
def _atlas() -> pd.DataFrame:
    consensus = consensus_cloud_types(load_zero_shot(), load_judge_tables())
    return atlas_table(consensus, manifest, read_table(dataset_cfg["aggregated_labels_path"]), load_zero_shot(), safe_read_table("data/processed/predictions/latest.parquet"))


table = _atlas()
if table.empty:
    st.warning("Keine Wolkentypen: erst `zero-shot` oder `judge-images` ausführen.")
    st.stop()

with st.sidebar:
    st.header("Auswahl")
    kinds = sorted(table["kind"].dropna().astype(str).unique().tolist())
    kind_filter = st.multiselect("Bildart", kinds, default=["original"] if "original" in kinds else kinds)
    min_reliability = st.selectbox("Mindest-Zuverlässigkeit", ["wahrscheinlich", "sicher", "unsicher"], help="sicher: alle Primärtypen einig · wahrscheinlich: mindestens zwei Primärtypen oder Primär + Nebentyp · unsicher: alles zeigen")
    examples = st.slider("Beispielbilder je Typ", 2, 6, 3)
    min_cell = st.slider("Mindest-Labels je Zelle (Typ × Licht)", 1, 10, 3)

view = table[table["kind"].astype(str).isin(kind_filter)] if kind_filter else table
report = reliability_report(table)
cols = st.columns(5)
cols[0].metric("Bilder mit Typ", report["n"])
cols[1].metric("sicher", report["sicher"])
cols[2].metric("wahrscheinlich", report["wahrscheinlich"])
cols[3].metric("unsicher", report["unsicher"])
cols[4].metric("in Auswahl", len(view))

summary = atlas_summary(view, min_reliability=min_reliability, examples=examples)
if summary.empty:
    st.info("Keine Bilder in dieser Auswahl.")
    st.stop()

tab_overview, tab_atlas, tab_light = st.tabs(["Übersicht", "Atlas", "Typ × Licht"])

with tab_overview:
    plot = summary.dropna(subset=["mean_score"]).copy()
    fig = px.bar(plot, x="type_name", y="mean_score", error_y="sd_score", color="share_top", color_continuous_scale="Viridis", range_color=(0, 0.7), hover_data={"n": True, "n_labeled": True, "n_top": True, "max_score": ":.1f", "type_name": False}, labels={"type_name": "", "mean_score": "Ø deine Note (gelabelte Bilder)", "share_top": "Anteil ≥ 9"})
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    show = summary[["type_name", "n", "n_sure", "n_originals", "n_labeled", "mean_score", "sd_score", "max_score", "n_top", "share_top", "model_mean", "top_light", "share_golden_or_twilight"]].copy()
    show.columns = ["Typ", "Bilder", "davon sicher", "Originale", "gelabelt", "Ø Note", "SD", "max", "≥ 9", "Anteil ≥ 9", "Ø Modell", "häufigstes Licht", "Anteil goldene Stunde/Dämmerung"]
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)
    st.caption("Ø Note nur über gelabelte Bilder; bei kleinen Zahlen (unter 10 Labels) sind die Mittelwerte Anhaltspunkte, keine Befunde.")

with tab_atlas:
    for row in summary.itertuples():
        info = CLOUD_TYPES.get(row.cloud_type)
        with st.container(border=True):
            head, body = st.columns([1.1, 2])
            with head:
                st.subheader(row.type_name)
                if info is not None:
                    st.caption(f"{info.latin} · Stockwerk: {info.level} · Höhe: {info.height_km}")
                score_text = f"Ø {row.mean_score:.1f} (n = {row.n_labeled}, max {row.max_score:.1f}, {row.n_top} × ≥ 9)" if row.mean_score == row.mean_score else "noch keine Labels"
                st.write(f"**{row.n} Bilder**, {row.n_sure} davon sicher, {row.n_originals} Originale · {score_text}")
                if info is not None:
                    st.markdown(f"**Aussehen:** {info.look}")
                    st.markdown(f"**Entstehung:** {info.formation}")
                    st.markdown(f"**Wetterbedeutung:** {info.weather}")
            with body:
                examples_frame = view[view["image_id"].isin(row.example_ids)].copy()
                examples_frame["_order"] = examples_frame["image_id"].map({image_id: index for index, image_id in enumerate(row.example_ids)})
                render_image_grid(
                    examples_frame.sort_values("_order"),
                    columns=min(examples, max(1, len(examples_frame))),
                    caption=lambda r: (f"deine Note {float(r['mean_score']):.1f}" if pd.notna(r["mean_score"]) else f"ungelabelt · Modell {float(r['model_score']):.1f}" if pd.notna(r["model_score"]) else "ungelabelt") + f" · {r['reliability']} · {r.get('clip_light', '')}",
                    empty_text="Keine Beispiele.",
                )

with tab_light:
    tiers = {"sicher": 2, "wahrscheinlich": 1, "unsicher": 0}
    reliable = view[view["reliability"].map(tiers).fillna(0) >= tiers.get(min_reliability, 1)]
    grid = type_by_light(reliable, min_n=min_cell)
    if grid.empty or grid.notna().sum().sum() == 0:
        st.info("Zu wenige gelabelte Bilder je Zelle. Mindest-Labels senken oder mehr Bildarten einschließen.")
    else:
        fig = px.imshow(grid, text_auto=".1f", zmin=0, zmax=10, color_continuous_scale="Viridis", aspect="auto", labels={"color": "Ø Note", "x": "CLIP-Lichtphase", "y": ""})
        fig.update_layout(height=80 + 42 * len(grid), margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Hängt die 10 am Licht oder an der Wolke? Zellen mit weniger Labels als eingestellt bleiben leer. Lichtphase ist die CLIP-Schätzung, nicht der Sonnenstand aus EXIF (der steht im Kontextblock des Inspect).")
