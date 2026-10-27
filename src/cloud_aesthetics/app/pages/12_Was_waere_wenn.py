from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, dataset_config, list_runs
from cloud_aesthetics.explain.counterfactual import EDITS, apply_edits, best_single_edits, load_scorer, resize_for_speed, sensitivity_sweep
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.settings import resolve_path
from cloud_aesthetics.utils.io import read_table

st.set_page_config(page_title="Was wäre wenn", layout="wide")
st.title("Was wäre wenn: Kontrafaktischer Explorer")
st.caption(
    "Verändere ein Foto kontrolliert und sieh, wie der CLIP-Score reagiert. Die Kurven zeigen, wie empfindlich das Modell "
    "für jeden Eingriff ist; der Vorschlag nennt den Eingriff mit dem größten Gewinn. Das sind Aussagen über das Modell, "
    "nicht über dich; ob du zustimmst, ist genau die Frage, die deine Labels beantworten."
)

dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])
labels = read_table(dataset_cfg["aggregated_labels_path"])
runs = [run for run in list_runs() if run.startswith("embed_clip")]
if manifest.empty or not runs:
    st.warning("Es wird ein Embedding-Run auf CLIP (embedding_clip.yaml) und ein Manifest benötigt.")
    st.stop()


@st.cache_resource(show_spinner="Lade CLIP und Modell ...")
def _scorer(run_id: str):
    return load_scorer(run_id)


with st.sidebar:
    run_id = st.selectbox("Run", runs)
    scorer, summary, model_name = _scorer(run_id)
    st.caption(f"Modell: {model_name} auf {', '.join(summary.get('embedding_models', []))}")
    manifest = manifest.copy()
    manifest["image_id"] = manifest["image_id"].astype(str)
    labeled = labels.copy()
    labeled["image_id"] = labeled["image_id"].astype(str)
    view = manifest.merge(labeled[["image_id", "mean_score"]], on="image_id", how="left")
    kind_choice = st.selectbox("Bildart", ["Originale", "Alle"])
    if kind_choice == "Originale" and "kind" in view.columns:
        view = view[view["kind"] == "original"]
    only_labeled = st.checkbox("Nur gelabelte Bilder", value=True)
    if only_labeled:
        view = view[view["mean_score"].notna()]
    view = view.sort_values("mean_score", ascending=False, na_position="last").reset_index(drop=True)
    options = view["image_id"].tolist()
    if not options:
        st.warning("Keine Bilder in dieser Auswahl.")
        st.stop()
    image_id = st.selectbox("Bild", options, format_func=lambda value: f"{value}  (Score {float(view.set_index('image_id').loc[value, 'mean_score']):.1f})" if pd.notna(view.set_index("image_id").loc[value, "mean_score"]) else value)
    points = st.slider("Stützstellen pro Kurve", 5, 11, 7, 2)

row = view[view["image_id"] == image_id].iloc[0]
original = read_rgb_image(row["relative_path"])
base = resize_for_speed(original, 768)

st.subheader("Eingriffe")
params: dict[str, float] = {}
slider_cols = st.columns(3)
for index, spec in enumerate(EDITS.values()):
    with slider_cols[index % 3]:
        params[spec.key] = st.slider(f"{spec.label} {spec.unit}".strip(), float(spec.low), float(spec.high), float(spec.neutral), float(spec.step), key=f"cf_{spec.key}")
if st.button("Zurücksetzen"):
    for spec in EDITS.values():
        st.session_state[f"cf_{spec.key}"] = float(spec.neutral)
    st.rerun()

edited = apply_edits(base, params)
scores = scorer.score([base, edited])
col_a, col_b = st.columns(2)
col_a.image(base, caption=f"Original · Modell {scores[0]:.2f}" + (f" · du {float(row['mean_score']):.1f}" if pd.notna(row.get("mean_score")) else ""), use_container_width=True)
col_b.image(edited, caption=f"Bearbeitet · Modell {scores[1]:.2f} ({scores[1] - scores[0]:+.2f})", use_container_width=True)

st.subheader("Empfindlichkeit")
with st.spinner("Rechne Sensitivitätskurven ..."):
    sweep = sensitivity_sweep(scorer, base, points=points)
best = best_single_edits(sweep)
top = best.iloc[0]
if top["best_delta"] > 0.05:
    st.success(f"Größter Gewinn laut Modell: {top['label']} auf {top['best_value']:g} ({top['best_delta']:+.2f} Punkte). Größter Verlust: {best.sort_values('worst_delta').iloc[0]['label']} auf {best.sort_values('worst_delta').iloc[0]['worst_value']:g} ({best['worst_delta'].min():+.2f}).")
else:
    st.info("Kein Eingriff hebt den Score nennenswert; das Bild liegt für das Modell nahe an seinem lokalen Optimum.")
fig = px.line(sweep, x="value", y="score", color="label", facet_col="label", facet_col_wrap=3, markers=True, labels={"value": "Wert", "score": "Modell-Score", "label": ""}, height=520)
fig.update_xaxes(matches=None, showticklabels=True)
fig.update_layout(showlegend=False, margin={"l": 10, "r": 10, "t": 40, "b": 10})
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig, use_container_width=True)
st.dataframe(best.round(3), use_container_width=True, hide_index=True)

st.subheader("Systematik über ein Score-Band")
st.caption(
    "Die CLI `counterfactual-sweep` rechnet dieselben Kurven für alle Bilder eines Bandes (z. B. deine 8er) und mittelt sie. "
    "Liegt eine Auswertung vor, erscheint sie hier."
)
sweep_dir = resolve_path("data/artifacts/counterfactual")
if sweep_dir.exists():
    summaries = sorted(sweep_dir.glob("*_summary.csv"))
    if summaries:
        chosen = st.selectbox("Auswertung", [path.stem.replace("_summary", "") for path in summaries])
        table = pd.read_csv(sweep_dir / f"{chosen}_summary.csv")
        fig2 = px.line(table, x="value", y="mean_delta", color="label", facet_col="label", facet_col_wrap=3, markers=True, labels={"value": "Wert", "mean_delta": "Ø Δ Score", "label": ""}, height=480)
        fig2.update_xaxes(matches=None, showticklabels=True)
        fig2.update_layout(showlegend=False, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig2.add_hline(y=0, line_color="gray", line_width=1)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine Band-Auswertung vorhanden.")
