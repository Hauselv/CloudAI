from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, dataset_config, list_runs
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.models.concept import FAMILY_LABELS, concept_label, explain_image, load_concept_run, sentence
from cloud_aesthetics.pipeline import run_leaderboard
from cloud_aesthetics.settings import resolve_path
from cloud_aesthetics.utils.io import read_table

st.set_page_config(page_title="Konzeptmodell", layout="wide")
st.title("Konzeptmodell: das Rating als lesbare Summe benannter Konzepte")
st.caption(
    "Ein Lasso auf benannten Konzepten (CLIP-Konzepte, Licht, Judge-Flags, Farbpalette, Szenenanteil) statt auf anonymen "
    "Embedding-Dimensionen. Weniger genau als CLIP-Ridge, dafür vollständig lesbar: Jeder Koeffizient ist ein Satz. "
    "Koeffizienten gelten je Standardabweichung des Konzepts; der Bootstrap-Balken zeigt, wie stabil das Vorzeichen ist."
)

dataset_cfg = dataset_config()
runs = [run for run in list_runs() if run.startswith("concept_")]
if not runs:
    st.warning("Noch kein Konzept-Run. Erst `train --config configs/models/concept_bottleneck.yaml` ausführen.")
    st.stop()


@st.cache_data(show_spinner="Lade Run ...")
def _run(run_id: str) -> dict[str, object]:
    return load_concept_run(run_id)


@st.cache_data(show_spinner=False)
def _leaderboard() -> pd.DataFrame:
    return run_leaderboard()


with st.sidebar:
    run_id = st.selectbox("Run", runs)
    data = _run(run_id)
    summary: dict[str, object] = data["summary"]
    info = summary.get("concept_model", {})
    st.caption(f"{info.get('n_concepts', 0)} Konzepte angeboten, {info.get('n_active', 0)} behalten · α = {float(info.get('alpha') or 0):.3f}")
    subset = summary.get("config", {}).get("include_kinds")
    st.caption("Trainiert auf: " + ("/".join(str(v) for v in subset) if subset else "alle Bildarten"))
    min_agreement = st.slider("Mindest-Vorzeichenstabilität", 0.0, 1.0, 0.8, 0.05, help="Anteil der Bootstrap-Wiederholungen mit gleichem Vorzeichen wie der Punktschätzer.")

concepts: pd.DataFrame = data["concepts"]
contributions: pd.DataFrame = data["contributions"]
predictions: pd.DataFrame = data["predictions"]
vocabulary: pd.DataFrame = data["vocabulary"]
primary = str(info.get("primary", "lasso"))
metrics = summary.get("models", {}).get(primary, {}).get("metrics", {})
by_kind = summary.get("models", {}).get(primary, {}).get("by_group", {}).get("val", {}).get("kind", {})

cols = st.columns(4)
cols[0].metric("OOF-Spearman (alle)", f"{metrics.get('val', {}).get('spearman', float('nan')):.2f}")
cols[1].metric("OOF-Spearman Originale", f"{by_kind.get('original', {}).get('spearman', float('nan')):.2f}" if by_kind.get("original") else "–")
cols[2].metric("OOF-MAE", f"{metrics.get('val', {}).get('mae', float('nan')):.2f}")
cols[3].metric("Test-Spearman", f"{metrics.get('test', {}).get('spearman', float('nan')):.2f}")

board = _leaderboard()
if not board.empty:
    reference = board[(board["kind"] == "embedding") & board["model"].eq("ridge")].sort_values("val_spearman", ascending=False)
    if not reference.empty:
        best = reference.iloc[0]
        st.caption(
            f"Zum Vergleich das beste CLIP-Ridge ({best['run_id']}): OOF-Spearman {float(best['val_spearman']):.2f}, "
            f"Originale {float(best['val_spearman_originals']) if pd.notna(best['val_spearman_originals']) else float('nan'):.2f}. "
            "Die Lücke ist der Preis der Lesbarkeit."
        )

st.info(sentence(concepts))

tab_coef, tab_vocab, tab_image, tab_residual = st.tabs(["Koeffizienten", "Vokabular-Vergleich", "Bild erklären", "Was das Modell nicht erklärt"])

with tab_coef:
    active = concepts[concepts["active"]].copy()
    if "sign_agreement" in active.columns:
        active["stabil"] = active["sign_agreement"] >= min_agreement
    else:
        active["stabil"] = True
    active = active.sort_values("coef_per_sd")
    fig = go.Figure()
    for family, part in active.groupby("family", sort=False):
        error = {}
        if "ci_low" in part.columns:
            error = {"error_x": {"type": "data", "symmetric": False, "array": (part["ci_high"] - part["coef_per_sd"]).to_numpy(), "arrayminus": (part["coef_per_sd"] - part["ci_low"]).to_numpy(), "thickness": 1.2}}
        fig.add_trace(
            go.Bar(
                x=part["coef_per_sd"],
                y=part["label"],
                orientation="h",
                name=FAMILY_LABELS.get(str(family), str(family)),
                marker={"opacity": [1.0 if flag else 0.35 for flag in part["stabil"]]},
                customdata=part[["sign_agreement", "coef_per_unit"]].to_numpy() if "sign_agreement" in part.columns else None,
                hovertemplate="%{y}<br>%{x:+.2f} je SD<br>Vorzeichen stabil in %{customdata[0]:.0%}<br>%{customdata[1]:+.3g} je Einheit<extra></extra>" if "sign_agreement" in part.columns else None,
                **error,
            )
        )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(barmode="overlay", height=max(420, 24 * len(active) + 140), margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend_title_text="", xaxis_title="Score-Punkte je Standardabweichung (90 %-Bootstrap-Intervall)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Blasse Balken: Vorzeichen unterhalb der eingestellten Stabilität. Konzepte mit Koeffizient 0 wurden vom Lasso verworfen und fehlen hier.")
    show_columns = [column for column in ("label", "family_label", "coef_per_sd", "coef_per_unit", "ci_low", "ci_high", "sign_agreement", "selection_rate", "input_mean", "input_sd") if column in concepts.columns]
    show = concepts[concepts["active"]][show_columns].copy()
    show.columns = [{"label": "Konzept", "family_label": "Familie", "coef_per_sd": "je SD", "coef_per_unit": "je Einheit", "ci_low": "CI 5 %", "ci_high": "CI 95 %", "sign_agreement": "Vorzeichen stabil", "selection_rate": "ausgewählt", "input_mean": "Ø Eingabe", "input_sd": "SD Eingabe"}[column] for column in show_columns]
    st.dataframe(show.round(3), use_container_width=True, hide_index=True)
    dropped = concepts[~concepts["active"]]
    if not dropped.empty:
        with st.expander(f"{len(dropped)} verworfene Konzepte"):
            st.write(", ".join(sorted(dropped["label"].tolist())))

with tab_vocab:
    if vocabulary.empty:
        st.info("Dieser Run hat keine Varianten (Block `variants` in der Config).")
    else:
        st.caption("Dasselbe Modell, andere Wörter: Welches Vokabular erklärt das Rating out-of-fold am besten?")
        vocab = vocabulary.copy()
        long = vocab.melt(id_vars=["variant"], value_vars=[column for column in ("oof_spearman", "oof_spearman_originals") if column in vocab.columns], var_name="metric", value_name="spearman")
        long["metric"] = long["metric"].map({"oof_spearman": "alle Bilder", "oof_spearman_originals": "nur Originale"})
        fig = px.bar(long, x="spearman", y="variant", color="metric", barmode="group", orientation="h", labels={"spearman": "OOF-Spearman", "variant": "", "metric": ""}, height=max(320, 40 * len(vocab) + 120))
        fig.add_vline(x=0, line_width=1, line_color="gray")
        fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
        show = vocab[[column for column in ("variant", "families", "n_concepts", "n_active", "n_dev", "oof_spearman", "oof_spearman_originals", "oof_mae") if column in vocab.columns]].copy()
        show.columns = [{"variant": "Variante", "families": "Familien", "n_concepts": "Konzepte", "n_active": "behalten", "n_dev": "n", "oof_spearman": "ρ alle", "oof_spearman_originals": "ρ Originale", "oof_mae": "MAE"}[column] for column in show.columns]
        st.dataframe(show.round(3), use_container_width=True, hide_index=True)

manifest = active_manifest(dataset_cfg["manifest_path"])
labels = read_table(dataset_cfg["aggregated_labels_path"])
manifest["image_id"] = manifest["image_id"].astype(str)
labels["image_id"] = labels["image_id"].astype(str)
view = manifest.merge(labels[["image_id", "mean_score"]], on="image_id", how="left")
if not predictions.empty:
    oof_column = f"oof_{primary}"
    pred = predictions[["image_id", oof_column, "partition"]].rename(columns={oof_column: "oof"}) if oof_column in predictions.columns else pd.DataFrame(columns=["image_id", "oof", "partition"])
    pred["image_id"] = pred["image_id"].astype(str)
    view = view.merge(pred, on="image_id", how="left")
view = view.merge(contributions[["image_id", "prediction"]], on="image_id", how="left")

with tab_image:
    options = view.sort_values("mean_score", ascending=False, na_position="last")
    only_labeled = st.checkbox("Nur gelabelte Bilder", value=True, key="concept_only_labeled")
    if only_labeled:
        options = options[options["mean_score"].notna()]
    options = options[options["image_id"].isin(set(contributions["image_id"].astype(str)))]
    if options.empty:
        st.info("Keine Bilder mit Konzeptbeiträgen in dieser Auswahl.")
    else:
        lookup = options.set_index("image_id")
        image_id = st.selectbox("Bild", options["image_id"].tolist(), format_func=lambda value: f"{value}  (Score {float(lookup.loc[value, 'mean_score']):.1f})" if pd.notna(lookup.loc[value, "mean_score"]) else value)
        row = lookup.loc[image_id]
        detail = explain_image(contributions, image_id)
        col_img, col_plot = st.columns([1, 1.4])
        with col_img:
            path = resolve_path(row["relative_path"])
            if path.exists():
                st.image(str(path), use_container_width=True)
            human = f"{float(row['mean_score']):.1f}" if pd.notna(row.get("mean_score")) else "–"
            oof_text = f" · out-of-fold {float(row['oof']):.2f}" if "oof" in row and pd.notna(row.get("oof")) else ""
            st.caption(f"Dein Score {human} · Konzeptmodell {float(row['prediction']):.2f}{oof_text} · {row.get('kind', '')}")
        with col_plot:
            if detail.empty:
                st.info("Alle Beiträge sind 0; das Modell sagt für dieses Bild den Mittelwert voraus.")
            else:
                intercept = float(contributions["intercept"].iloc[0])
                ordered = detail.iloc[::-1]
                fig = go.Figure(
                    go.Waterfall(
                        orientation="h",
                        measure=["absolute"] + ["relative"] * len(ordered) + ["total"],
                        y=["Mittelwert"] + ordered["label"].tolist() + ["Vorhersage"],
                        x=[intercept] + ordered["contribution"].tolist() + [0],
                        text=[f"{intercept:.2f}"] + [f"{value:+.2f}" for value in ordered["contribution"]] + [f"{float(row['prediction']):.2f}"],
                        textposition="outside",
                        connector={"line": {"color": "rgba(120,120,120,0.4)"}},
                        increasing={"marker": {"color": "#2a9d8f"}},
                        decreasing={"marker": {"color": "#e76f51"}},
                        totals={"marker": {"color": "#264653"}},
                    )
                )
                fig.update_layout(height=max(360, 26 * len(ordered) + 140), margin={"l": 10, "r": 40, "t": 30, "b": 10}, xaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

with tab_residual:
    if "oof" not in view.columns or view["oof"].isna().all():
        st.info("Keine Out-of-fold-Vorhersagen in diesem Run.")
    else:
        st.caption(
            "Bilder, bei denen die Konzepte am weitesten danebenliegen (out-of-fold, also ohne das Bild gesehen zu haben). "
            "Unterschätzte Bilder tragen etwas, das kein Konzept benennt: Kandidaten für neue Konzept-Prompts."
        )
        scored = view[view["oof"].notna() & view["mean_score"].notna()].copy()
        scored["residual"] = scored["mean_score"].astype(float) - scored["oof"].astype(float)
        kinds = sorted(scored["kind"].dropna().astype(str).unique().tolist()) if "kind" in scored.columns else []
        kind_filter = st.multiselect("Bildart", kinds, default=[k for k in kinds if k == "original"] or kinds, key="concept_residual_kind")
        if kind_filter:
            scored = scored[scored["kind"].astype(str).isin(kind_filter)]
        n_show = st.slider("Bilder je Seite", 4, 16, 8, 4, key="concept_residual_n")
        col_under, col_over = st.columns(2)
        with col_under:
            st.subheader("Unterschätzt (du: höher)")
            render_image_grid(scored.sort_values("residual", ascending=False).head(n_show), columns=2, caption=lambda r: f"du {float(r['mean_score']):.1f} · Modell {float(r['oof']):.1f} (Δ {float(r['residual']):+.1f})\n\n{r['image_id']}")
        with col_over:
            st.subheader("Überschätzt (du: niedriger)")
            render_image_grid(scored.sort_values("residual", ascending=True).head(n_show), columns=2, caption=lambda r: f"du {float(r['mean_score']):.1f} · Modell {float(r['oof']):.1f} (Δ {float(r['residual']):+.1f})\n\n{r['image_id']}")
        fig = px.scatter(scored, x="oof", y="mean_score", color="kind" if "kind" in scored.columns else None, hover_data=["image_id"], labels={"oof": "Konzeptmodell (out-of-fold)", "mean_score": "dein Score"}, opacity=0.7)
        fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line={"color": "gray", "width": 1, "dash": "dot"})
        fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
