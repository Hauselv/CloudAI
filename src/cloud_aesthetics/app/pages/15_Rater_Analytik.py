from __future__ import annotations

import glob
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

from cloud_aesthetics.app.common import active_manifest, dataset_config, list_runs
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.eval.rater_analytics import analyze_rater, disputed_images, model_reference, rater_agreement_matrix
from cloud_aesthetics.judge.store import rater_kinds
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Rater-Analytik", layout="wide")
st.title("Rater-Analytik: Wie wird bewertet?")
st.caption(
    "Nicht *was* gut ist, sondern *wie* geurteilt wird: Sitzungen aus den Zeitstempeln, Drift innerhalb und zwischen Sitzungen, "
    "Anker-Effekte durch das Vorgängerbild, Skalennutzung und Streitbilder. Alle Verhaltensfragen laufen auf Residuen "
    "(eigene Note minus Referenz pro Bild), damit der Bildinhalt herausgerechnet ist."
)

dataset_cfg = dataset_config()
ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
if ratings.empty:
    st.warning("Noch keine Bewertungen.")
    st.stop()
manifest = active_manifest(dataset_cfg["manifest_path"])
manifest["image_id"] = manifest["image_id"].astype(str)
kinds = rater_kinds(dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv"))
all_raters = sorted(ratings["rater_id"].astype(str).unique().tolist())
humans = [r for r in all_raters if kinds.get(r, "human") == "human"]
judges = [r for r in all_raters if kinds.get(r) == "llm"]

with st.sidebar:
    rater_id = st.selectbox("Rater", humans or all_raters)
    reference_choice = st.selectbox("Referenz pro Bild", ["Judges (LLM-Mittel)", "Modell (out-of-fold)", "keine (eigener Mittelwert)"], help="Wovon die eigene Note abgezogen wird, um den Bildinhalt herauszurechnen. Judges sind unabhängig von deiner Stimmung, das Modell kennt deinen Geschmack.")
    gap_minutes = st.slider("Neue Sitzung ab Pause (Minuten)", 5, 120, 30, 5)
    max_gap = st.slider("Anker: max. Abstand zum Vorgänger (Sekunden)", 30, 1800, 600, 30)
    exclude_same = st.selectbox("Paare gleicher Quelle ausschließen", ["Split-Gruppe (Original + Crops)", "Aufnahmesitzung", "nicht ausschließen"], help="Aufeinanderfolgende Bilder derselben Quelle haben ähnliche Residuen aus Inhaltsgründen; sie würden einen Anker-Effekt vortäuschen.")

reference = None
reference_raters = judges
if reference_choice.startswith("Modell"):
    runs = [run for run in list_runs() if run.startswith("embed_clip_v1")] or [run for run in list_runs() if run.startswith("embed_")]
    if runs:
        predictions_path = resolve_path("data/artifacts") / runs[0] / "predictions.parquet"
        reference = model_reference(pd.read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame(), "ridge")
        st.caption(f"Modellreferenz: {runs[0]} (out-of-fold Ridge; Testbilder mit angepasster Vorhersage)")
    if reference is None or reference.empty:
        st.warning("Keine Modellvorhersagen gefunden, nutze Judges als Referenz.")
        reference = None
elif reference_choice.startswith("keine"):
    reference_raters = []
    reference = pd.Series(dtype=float)

groups = None
if exclude_same.startswith("Split") and "split_group_id" in manifest.columns:
    groups = manifest.set_index("image_id")["split_group_id"]
elif exclude_same.startswith("Aufnahme") and "capture_session_id" in manifest.columns:
    groups = manifest.set_index("image_id")["capture_session_id"]

result = analyze_rater(ratings, rater_id, reference_raters=reference_raters, reference=reference, gap_minutes=gap_minutes, max_gap_seconds=max_gap, groups=groups)
frame: pd.DataFrame = result["frame"]
if frame.empty:
    st.info("Keine Bewertungen mit Zeitstempel für diesen Rater.")
    st.stop()
drift = result["drift"]
anchor = result["anchor"]
sessions: pd.DataFrame = result["sessions"]

cols = st.columns(6)
cols[0].metric("Bewertungen", drift["n"])
cols[1].metric("Sitzungen", drift["n_sessions"])
cols[2].metric("Sekunden je Bild", f"{sessions['seconds_per_image'].median():.0f}" if not sessions.empty and sessions["seconds_per_image"].notna().any() else "–")
cols[3].metric("Drift je 50 Bilder", f"{drift.get('residual_change_per_50', float('nan')):+.2f}", help="Residuenänderung über 50 Bilder innerhalb einer Sitzung (positiv = wird milder).")
cols[4].metric("Anker r", f"{anchor.get('assimilation_r', float('nan')):.2f}" if "assimilation_r" in anchor else "–", help="Korrelation des Residuums mit dem Residuum des Vorgängerbilds. Positiv = Assimilation, negativ = Kontrast.")
cols[5].metric("nach 9+ minus nach ≤4", f"{anchor.get('after_high_minus_after_low', float('nan')):+.2f}" if "after_high_minus_after_low" in anchor else "–", help="Mittleres Residuum nach einem hoch bewerteten Vorgänger minus nach einem niedrig bewerteten, in Score-Punkten.")

tab_course, tab_sessions, tab_anchor, tab_scale, tab_disputes, tab_matrix = st.tabs(["Verlauf", "Sitzungen", "Anker", "Skala", "Streitbilder", "Rater-Matrix"])

with tab_course:
    plot = frame.copy()
    plot["index"] = np.arange(len(plot))
    plot["Sitzung"] = plot["session"].astype(str)
    plot["gleitend"] = plot["residual"].rolling(15, center=True, min_periods=5).mean()
    fig = px.scatter(plot, x="index", y="residual", color="Sitzung", hover_data={"image_id": True, "score": True, "index": False}, labels={"index": "Bewertung Nr.", "residual": "Residuum (eigene Note − Referenz)"}, opacity=0.6)
    fig.add_trace(go.Scatter(x=plot["index"], y=plot["gleitend"], mode="lines", name="gleitendes Mittel (15)", line={"color": "black", "width": 2}))
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    text = f"Innerhalb der Sitzungen: Spearman(Position, Residuum) = {drift.get('within_session_rho', float('nan')):+.3f} (p = {drift.get('within_session_p', float('nan')):.2f}). "
    if "first20_minus_later" in drift and drift["first20_minus_later"] == drift["first20_minus_later"]:
        text += f"Die ersten 20 Bilder einer Sitzung liegen im Mittel {drift['first20_minus_later']:+.2f} Punkte über dem Rest (Aufwärm-Effekt, wenn deutlich positiv). "
    text += f"Zwischen den Sitzungen schwankt das mittlere Residuum um ±{drift.get('between_session_sd', 0.0):.2f}."
    st.caption(text)

with tab_sessions:
    if sessions.empty:
        st.info("Keine Sitzungen.")
    else:
        show = sessions.copy()
        show["start"] = pd.to_datetime(show["start"]).dt.strftime("%Y-%m-%d %H:%M")
        show.columns = ["Sitzung", "Start (UTC)", "Bilder", "Dauer (min)", "s je Bild", "Ø Note", "SD Note", "Ø Residuum", "Trend ρ", "Trend p", "Anteil ,5"]
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)
        st.caption("Trend ρ: Spearman zwischen Position und Residuum innerhalb der Sitzung (ab 8 Bildern). Anteil ,5: wie oft halbe Noten vergeben wurden.")

with tab_anchor:
    if "assimilation_r" not in anchor:
        st.info(f"Zu wenige aufeinanderfolgende Paare ({anchor.get('n_pairs', 0)}).")
    else:
        st.write(
            f"{anchor['n_pairs']} Paare direkt aufeinanderfolgender Bewertungen (≤ {max_gap} s Abstand), "
            f"{anchor.get('n_same_group_excluded', 0)} Paare gleicher Quelle ausgeschlossen. "
            f"Residuum vs. Vorgänger-Residuum: r = {anchor['assimilation_r']:+.2f} (p = {anchor['assimilation_p']:.3g}); "
            f"Residuum vs. Vorgänger-Note: r = {anchor['previous_score_r']:+.2f} (p = {anchor['previous_score_p']:.3g})."
        )
        ordered = frame.sort_values(["session", "position"]).reset_index(drop=True)
        ordered["previous_score"] = ordered["score"].shift(1)
        ordered["previous_residual"] = ordered["residual"].shift(1)
        valid = ordered["session"].eq(ordered["session"].shift(1)) & ordered["seconds_since_previous"].le(max_gap)
        if groups is not None:
            group = ordered["image_id"].map(groups)
            valid &= ~(group.notna() & group.eq(group.shift(1)))
        pairs = ordered[valid].dropna(subset=["previous_score", "residual"])
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.scatter(pairs, x="previous_score", y="residual", trendline="ols", opacity=0.55, hover_data=["image_id"], labels={"previous_score": "Note des Vorgängerbilds", "residual": "Residuum des aktuellen Bilds"})
            fig.add_hline(y=0, line_width=1, line_color="gray")
            fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10})
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            bins = pd.cut(pairs["previous_score"], [-0.1, 4, 6.5, 8.5, 10.1], labels=["Vorgänger ≤ 4", "4,5–6,5", "7–8,5", "9–10"])
            by_bin = pairs.groupby(bins, observed=True)["residual"].agg(["mean", "count", "sem"]).reset_index()
            fig = px.bar(by_bin, x="previous_score", y="mean", error_y="sem", text="count", labels={"previous_score": "", "mean": "mittleres Residuum danach"})
            fig.add_hline(y=0, line_width=1, line_color="gray")
            fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10})
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Lesart: Ein positiver Zusammenhang heißt Assimilation (nach einem starken Bild fällt das nächste milder aus), ein negativer Kontrast. "
            "Vorsicht: Auch mit ausgeschlossenen Quellpaaren können aufeinanderfolgende Bilder inhaltlich ähnlich sein (gleicher Abend, gleiche Website); "
            "die Modellreferenz fängt das besser ab als die Judges. Gegenmittel beim Labeln: Reihenfolge „Random“ oder „Diverse groups“ und Pausen nach extremen Bildern."
        )

with tab_scale:
    usage: pd.DataFrame = result["usage"]
    fig = px.bar(usage, x="score", y="n", labels={"score": "Note", "n": "Anzahl"})
    fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis={"dtick": 0.5})
    st.plotly_chart(fig, use_container_width=True)
    half = float((np.mod(frame["score"] * 2, 2) == 1).mean())
    st.caption(f"Halbe Noten: {half:.0%} aller Bewertungen. Mittel {frame['score'].mean():.2f}, Median {frame['score'].median():.1f}, SD {frame['score'].std(ddof=0):.2f}; {int((frame['score'] >= 9).sum())} Bewertungen ≥ 9, {int((frame['score'] <= 2).sum())} Bewertungen ≤ 2.")

with tab_disputes:
    disputes = disputed_images(ratings, human_raters=humans, reference_raters=judges)
    disputes = disputes.merge(manifest[["image_id", "relative_path", "kind"]], on="image_id", how="inner")
    if disputes.empty or "human_vs_reference" not in disputes.columns:
        st.info("Keine Vergleichsbasis (es braucht Judge-Urteile oder mehrere menschliche Rater).")
    else:
        multi = (disputes.get("human_n", pd.Series(0)) >= 2).any()
        st.caption("Streit zwischen menschlichen Ratern (Spannweite)." if multi else "Nur ein menschlicher Rater: Streit = Abstand zwischen deiner Note und dem Judge-Mittel. Das sind die Bilder, an denen dein Geschmack am deutlichsten vom Modell-Konsens abweicht.")
        n_show = st.slider("Bilder je Seite", 4, 16, 8, 4, key="dispute_n")
        col_up, col_down = st.columns(2)
        with col_up:
            st.subheader("Du deutlich höher")
            render_image_grid(disputes[disputes["human_vs_reference"] > 0].head(n_show), columns=2, caption=lambda r: f"du {float(r['human_mean']):.1f} · Judges {float(r['reference_mean']):.1f} · {r['kind']}\n\n{r['image_id']}")
        with col_down:
            st.subheader("Du deutlich niedriger")
            render_image_grid(disputes[disputes["human_vs_reference"] < 0].head(n_show), columns=2, caption=lambda r: f"du {float(r['human_mean']):.1f} · Judges {float(r['reference_mean']):.1f} · {r['kind']}\n\n{r['image_id']}")

with tab_matrix:
    matrix = rater_agreement_matrix(ratings)
    fig = px.imshow(matrix, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu", labels={"color": "Spearman"})
    fig.update_layout(height=120 + 60 * len(matrix), margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Spearman-Übereinstimmung je Rater-Paar auf den gemeinsam bewerteten Bildern (mindestens 20). Sobald Freunde labeln, erscheinen sie hier.")
