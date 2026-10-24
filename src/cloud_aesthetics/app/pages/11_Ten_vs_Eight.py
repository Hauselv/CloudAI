from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import dataset_config
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.data.exclusions import active_excluded_ids
from cloud_aesthetics.eval.contrast import FAMILY_LABELS, Band, contrast_bands, load_analysis_table, top_end_trends
from cloud_aesthetics.explain.interpretation import FEATURE_LABELS

st.set_page_config(page_title="10 vs 8", layout="wide")
st.title("10 vs 8: Was trennt die Ausnahmebilder von den guten?")
st.caption(
    "Zwei Bänder deines eigenen Scores werden über alle Signale verglichen: handgebaute Features, CLIP-Zero-Shot, "
    "Judge-Attribute. Effektstärke (Cohen's d), Rangtest (Mann-Whitney) und FDR-korrigierte q-Werte. "
    "Korrelation ist kein Kausalbeweis, aber hier entstehen die Hypothesen."
)


@st.cache_data(show_spinner="Lade Merkmalstabelle ...")
def _table() -> pd.DataFrame:
    return load_analysis_table(dataset_config())


table = _table()
if table.empty or "mean_score" not in table.columns:
    st.warning("Keine aggregierten Labels gefunden. Erst `aggregate-labels` ausführen.")
    st.stop()
excluded = active_excluded_ids()
table = table[~table["image_id"].astype(str).isin(excluded)]

with st.sidebar:
    st.header("Auswahl")
    classes = ["original", "large_crop", "small_crop"]
    crop_filter = st.multiselect("Bildart", classes, default=["original"], help="Für die 10-vs-8-Frage zählen Originale; Crops werden anders bewertet.")
    datasets = sorted(table["dataset_name"].dropna().astype(str).unique().tolist()) if "dataset_name" in table.columns else []
    dataset_filter = st.multiselect("Datensatz", datasets, default=datasets)
    a_low, a_high = st.slider("Band A (die Ausnahmebilder)", 0.0, 10.0, (9.5, 10.0), 0.5)
    b_low, b_high = st.slider("Band B (die guten)", 0.0, 10.0, (7.5, 8.5), 0.5)
    family_options = {label: key for key, label in FAMILY_LABELS.items()}
    chosen_families = st.multiselect("Merkmalsfamilien", list(family_options), default=list(family_options))
    top_k = st.slider("Top-Merkmale", 10, 60, 25, 5)
    q_threshold = st.select_slider("q-Wert-Schwelle markieren", [0.01, 0.05, 0.1, 0.2, 1.0], value=0.1)

subset = table.copy()
if crop_filter and "crop_class" in subset.columns:
    subset = subset[subset["crop_class"].astype(str).isin(crop_filter)]
if dataset_filter and "dataset_name" in subset.columns:
    subset = subset[subset["dataset_name"].astype(str).isin(dataset_filter)]

band_a = Band(a_low, a_high, f"{a_low:g}–{a_high:g}")
band_b = Band(b_low, b_high, f"{b_low:g}–{b_high:g}")
result = contrast_bands(subset, band_a, band_b)
metric_cols = st.columns(4)
metric_cols[0].metric(f"Band A ({band_a.label})", result["n_a"])
metric_cols[1].metric(f"Band B ({band_b.label})", result["n_b"])
metric_cols[2].metric("Bilder in Auswahl", len(subset))
metric_cols[3].metric("Signale verglichen", len(result["table"]))
if result["table"].empty:
    st.warning("Zu wenige Bilder in einem der Bänder (mindestens 5 pro Band). Bänder oder Bildart anpassen.")
    st.stop()

frame: pd.DataFrame = result["table"]
selected_keys = {family_options[label] for label in chosen_families}
frame = frame[frame["family"].isin(selected_keys)].reset_index(drop=True)


def _label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix in ("clip_concept_", "clip_type_", "clip_light_", "judge_type_", "judge_light_", "judge_flag_", "judge_", "concept_", "meteo_", "cloud_type_score_"):
        if feature.startswith(prefix):
            return f"{feature[len(prefix):].replace('_', ' ')} ({prefix.rstrip('_').replace('_', ' ')})"
    return feature.replace("_", " ")


frame["label"] = frame["feature"].map(_label)
frame["significant"] = frame["q_mannwhitney"] <= q_threshold

tab_effects, tab_explore, tab_trend, tab_examples = st.tabs(["Effekte", "Merkmal erkunden", "Trend am oberen Ende", "Beispiele"])

with tab_effects:
    top = frame.head(top_k).iloc[::-1]
    fig = px.bar(
        top,
        x="cohens_d",
        y="label",
        orientation="h",
        color="family_label",
        hover_data={"mean_a": ":.3g", "mean_b": ":.3g", "q_mannwhitney": ":.3f", "rank_biserial": ":.2f", "label": False, "family_label": False},
        labels={"cohens_d": f"Cohen's d (positiv = höher in Band A {band_a.label})", "label": "", "family_label": "Familie"},
        height=max(420, 22 * len(top) + 120),
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(legend_title_text="", margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
    show = frame.head(top_k)[["label", "family_label", "mean_a", "mean_b", "cohens_d", "rank_biserial", "p_mannwhitney", "q_mannwhitney", "significant"]].copy()
    show.columns = ["Merkmal", "Familie", f"Ø Band A", f"Ø Band B", "Cohen's d", "Rang-biserial", "p (MW)", "q (FDR)", f"q ≤ {q_threshold}"]
    st.dataframe(show.round(3), use_container_width=True, hide_index=True)
    n_sig = int(frame["significant"].sum())
    st.caption(
        f"{n_sig} von {len(frame)} Signalen liegen unter q ≤ {q_threshold}. Bei kleinen Bändern sind Effektstärken um |d| ≈ 0,5 und darüber "
        "interessant, auch wenn der q-Wert nicht reicht; alles darunter ist mit dieser Stichprobe Rauschen."
    )

with tab_explore:
    feature = st.selectbox("Merkmal", frame["feature"].tolist(), format_func=_label)
    row = frame[frame["feature"] == feature].iloc[0]
    st.caption(f"d = {row['cohens_d']:+.2f} · Ø Band A {row['mean_a']:.3g} vs Ø Band B {row['mean_b']:.3g} · q = {row['q_mannwhitney']:.3f} · {row['direction']}")
    scores = subset["mean_score"].astype(float)
    plot_frame = subset[["image_id", "mean_score", feature] + [column for column in ("crop_class", "relative_path") if column in subset.columns]].copy()
    plot_frame["band"] = "übrige"
    plot_frame.loc[band_a.mask(scores), "band"] = f"A {band_a.label}"
    plot_frame.loc[band_b.mask(scores), "band"] = f"B {band_b.label}"
    col_strip, col_scatter = st.columns(2)
    with col_strip:
        strip = px.strip(plot_frame[plot_frame["band"] != "übrige"], x="band", y=feature, color="band", hover_data=["image_id"], labels={feature: _label(feature), "band": ""})
        strip.update_layout(showlegend=False, margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(strip, use_container_width=True)
    with col_scatter:
        scatter = px.scatter(plot_frame, x=feature, y="mean_score", color="band", hover_data=["image_id"], labels={feature: _label(feature), "mean_score": "dein Score"}, opacity=0.75)
        scatter.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend_title_text="")
        st.plotly_chart(scatter, use_container_width=True)
    if "relative_path" in plot_frame.columns:
        st.write("Höchste und niedrigste Werte dieses Merkmals innerhalb beider Bänder")
        both = plot_frame[plot_frame["band"] != "übrige"].dropna(subset=[feature])
        col_hi, col_lo = st.columns(2)
        with col_hi:
            st.caption("hoch")
            render_image_grid(both.sort_values(feature, ascending=False).head(6), columns=3, caption=lambda r: f"{_label(feature)} {float(r[feature]):.3g} | Score {float(r['mean_score']):.1f}\n\n{r['image_id']}")
        with col_lo:
            st.caption("niedrig")
            render_image_grid(both.sort_values(feature, ascending=True).head(6), columns=3, caption=lambda r: f"{_label(feature)} {float(r[feature]):.3g} | Score {float(r['mean_score']):.1f}\n\n{r['image_id']}")

with tab_trend:
    min_score = st.slider("Nur Bilder mit Score ≥", 5.0, 9.0, 7.0, 0.5)
    trends = top_end_trends(subset, min_score=min_score)
    trends = trends[trends["family"].isin(selected_keys)] if not trends.empty else trends
    if trends.empty:
        st.info("Zu wenige Bilder für eine Trendanalyse.")
    else:
        trends = trends.head(top_k).copy()
        trends["label"] = trends["feature"].map(_label)
        st.caption(
            f"Spearman-Korrelation jedes Signals mit deinem Score, nur innerhalb der Bilder ab {min_score:g} (n={int(trends['n'].max())}). "
            "Das ist die direkte Frage: Was unterscheidet unter den guten Bildern die besseren?"
        )
        fig = px.bar(trends.iloc[::-1], x="spearman", y="label", orientation="h", color="family_label", hover_data={"q_value": ":.3f", "label": False, "family_label": False}, labels={"spearman": "Spearman ρ", "label": ""}, height=max(420, 22 * len(trends) + 120))
        fig.add_vline(x=0, line_width=1, line_color="gray")
        fig.update_layout(legend_title_text="", margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(trends[["label", "family_label", "spearman", "p_value", "q_value", "n"]].round(3), use_container_width=True, hide_index=True)

with tab_examples:
    if "relative_path" in subset.columns:
        col_a, col_b = st.columns(2)
        examples_a = subset[subset["image_id"].astype(str).isin(result["ids_a"])].sort_values("mean_score", ascending=False)
        examples_b = subset[subset["image_id"].astype(str).isin(result["ids_b"])].sort_values("mean_score", ascending=False)
        with col_a:
            st.subheader(f"Band A {band_a.label} ({len(examples_a)})")
            render_image_grid(examples_a.head(24), columns=3, caption=lambda r: f"Score {float(r['mean_score']):.1f} | {r.get('crop_class', '')}\n\n{r['image_id']}")
        with col_b:
            st.subheader(f"Band B {band_b.label} ({len(examples_b)})")
            render_image_grid(examples_b.head(24), columns=3, caption=lambda r: f"Score {float(r['mean_score']):.1f} | {r.get('crop_class', '')}\n\n{r['image_id']}")
