from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import dataset_config
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.cli import judge_triage_impl
from cloud_aesthetics.data.exclusions import active_excluded_ids, set_exclusions
from cloud_aesthetics.judge.triage import load_triage, triage_summary

st.set_page_config(page_title="Triage", layout="wide")
st.title("Triage: Vorsortierung durch die Judges")
st.caption(
    "Die Modell-Rater sortieren die ungelabelten Bilder vor. Nichts wird automatisch ausgeschlossen oder gelabelt; "
    "diese Seite schlägt vor, du entscheidest."
)

dataset_cfg = dataset_config()
col_refresh, col_info = st.columns([1, 3])
if col_refresh.button("Triage neu berechnen", use_container_width=True):
    with st.spinner("Kombiniere Judge-Urteile ..."):
        judge_triage_impl()
    st.rerun()

triage = load_triage()
if triage.empty:
    st.info("Noch keine Triage-Tabelle. Erst `judge-images --no-only-labeled` laufen lassen, dann hier neu berechnen.")
    st.stop()
excluded = active_excluded_ids()
triage = triage[~triage["image_id"].isin(excluded)].reset_index(drop=True)
summary = triage_summary(triage)
col_info.caption(
    f"{summary['images']} aktive Bilder · {summary['unlabeled']} ungelabelt, davon {summary['unlabeled_judged']} von Judges bewertet · "
    f"{summary['unlabeled_originals']} ungelabelte Originale ({summary['unlabeled_originals_judge_ge_7']} mit Judge ≥ 7) · "
    f"{summary['proposed_exclusions_unlabeled']} Exclusion-Vorschläge · {summary['uncertain_unlabeled']} unsichere Bilder"
)

tab_next, tab_uncertain, tab_exclude, tab_overview = st.tabs(["Als nächstes labeln", "Unsicher", "Exclusion-Vorschläge", "Übersicht"])

unlabeled = triage[~triage["labeled"] & (triage["proposed_exclusion"] == "")]


def _caption(row: pd.Series) -> str:
    spread = f" ±{float(row['judge_spread']):.1f}" if pd.notna(row.get("judge_spread")) else ""
    hints = f"\n\nHinweise: {row['judge_hints']}" if isinstance(row.get("judge_hints"), str) and row["judge_hints"] else ""
    return (
        f"Judge {float(row['judge_mean']):.1f}{spread} | {row.get('crop_class', '')} | {row.get('judge_cloud_type', '')}, {row.get('judge_light', '')}\n\n"
        f"{row.get('judge_verdict', '') or ''}{hints}\n\n{row['image_id']}"
    )


with tab_next:
    st.write("Ungelabelte Bilder in Label-Reihenfolge: Originale zuerst, dann nach Judge-Score. Die Rate-Seite kennt diese Reihenfolge als „Judge-Vorschlag“.")
    kind_choice = st.selectbox("Bildart", ["Nur Originale", "Alle"], key="triage_kind")
    view = unlabeled if kind_choice == "Alle" else unlabeled[unlabeled["crop_class"] == "original"]
    page_size = st.selectbox("Kacheln", [12, 24, 48], index=1, key="triage_tiles")
    page_count = max(1, (len(view) + page_size - 1) // page_size)
    page = st.number_input("Seite", min_value=1, max_value=page_count, value=1, key="triage_page")
    st.caption(f"{len(view)} Bilder")
    render_image_grid(view.iloc[(int(page) - 1) * page_size : int(page) * page_size], caption=_caption, columns=4)

with tab_uncertain:
    st.write("Bilder, bei denen die Judges um mindestens zwei Punkte auseinanderliegen. Hier ist ein menschliches Label am informativsten.")
    view = unlabeled[unlabeled["uncertain"]].sort_values("judge_spread", ascending=False)
    st.caption(f"{len(view)} Bilder")
    render_image_grid(view.head(24), caption=_caption, columns=4)

with tab_exclude:
    proposals = triage[(triage["proposed_exclusion"] != "")].copy()
    st.write("Vorschläge aus den Qualitätsflags der Judges. Haken setzen und bestätigen; Ausschlüsse lassen sich auf der Seite Exclusions wieder aufheben.")
    st.caption(
        "Bewusst konservativ: Nur „kein Wolkenfoto“, einstimmige Satelliten-/Objekt-Flags bei niedrigem Judge-Score und einstimmig "
        "„unscharf“ bei Score ≤ 3 werden vorgeschlagen. Landschaft und Flugzeugfenster sind nur Hinweise, weil du solche Bilder oft hoch bewertest."
    )
    reasons = ["Alle"] + sorted(proposals["proposed_exclusion"].unique().tolist())
    reason_choice = st.selectbox("Grund", reasons, key="triage_reason")
    include_labeled = st.checkbox("Auch bereits gelabelte Bilder zeigen", value=False)
    view = proposals if reason_choice == "Alle" else proposals[proposals["proposed_exclusion"] == reason_choice]
    if not include_labeled:
        view = view[~view["labeled"]]
    st.caption(f"{len(view)} Vorschläge")
    selected: list[str] = []
    if not view.empty:
        grid = st.columns(4)
        for index, (_, row) in enumerate(view.head(48).iterrows()):
            with grid[index % 4]:
                path = Path(str(row["relative_path"]))
                full = path if path.is_absolute() else Path(__file__).resolve().parents[4] / path
                if full.exists():
                    st.image(str(full), use_container_width=True)
                human = f" | Mensch {float(row['human_score']):.1f}" if pd.notna(row.get("human_score")) else ""
                st.caption(f"{row['proposed_exclusion']} · {row['proposal_detail']}{human}\n\n{row.get('judge_verdict', '')}\n\n{row['image_id']}")
                if st.checkbox("Ausschließen", key=f"triage_excl_{row['image_id']}"):
                    selected.append(str(row["image_id"]))
        if st.button(f"{len(selected)} ausgewählte Bilder ausschließen", type="primary", disabled=not selected):
            rows = view[view["image_id"].isin(selected)]
            by_reason = rows.groupby("proposed_exclusion")
            for reason, part in by_reason:
                set_exclusions(part[["image_id", "relative_path"]].to_dict("records"), excluded=True, reason=f"{reason} (judge proposal)")
            st.success(f"{len(selected)} Bilder ausgeschlossen.")
            st.rerun()

with tab_overview:
    judged = triage[triage["judge_votes"] > 0]
    if not judged.empty:
        st.write("Judge-Score ungelabelt vs. gelabelt")
        hist = pd.DataFrame(
            {
                "ungelabelt": judged.loc[~judged["labeled"], "judge_mean"].round().value_counts().sort_index(),
                "gelabelt": judged.loc[judged["labeled"], "judge_mean"].round().value_counts().sort_index(),
            }
        ).fillna(0)
        st.bar_chart(hist)
        st.write("Ungelabelte Bilder nach Wolkentyp (Judge-Mehrheit)")
        table = judged[~judged["labeled"]].groupby("judge_cloud_type").agg(n=("image_id", "count"), judge_mean=("judge_mean", "mean")).round(2).sort_values("n", ascending=False)
        st.dataframe(table, use_container_width=True)
        st.write("Flags unter den ungelabelten Bildern")
        flags = judged.loc[~judged["labeled"], "judge_flags"].fillna("").str.split("|").explode()
        flags = flags[flags != ""]
        if not flags.empty:
            st.dataframe(flags.value_counts().rename("images").reset_index().rename(columns={"index": "flag"}), use_container_width=True, hide_index=True)
