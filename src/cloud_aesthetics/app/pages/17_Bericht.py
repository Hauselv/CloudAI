from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import app_config, dataset_config
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.judge.store import rater_kinds
from cloud_aesthetics.reports.year_report import REPORT_DIR, build_year_report_impl, save_report
from cloud_aesthetics.settings import resolve_path

st.set_page_config(page_title="Bericht", layout="wide")
st.title("Bericht: Wolken des Jahres")
st.caption(
    "Ein eigenständiges HTML-Dokument mit den besten Fotos und Erklärkarten (Wolkentyp, Konzeptbeiträge, Judge-Urteile), "
    "dem Konzeptmodell in einem Satz, der 10-gegen-8-Tabelle, dem Wolkenatlas, den Modellen und der Verlässlichkeit. "
    "Zum Teilen die Datei herunterladen; PDF über den Druckdialog des Browsers."
)

dataset_cfg = dataset_config()
cfg = app_config()
ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
kinds = rater_kinds(dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv"))
humans = sorted(r for r in ratings["rater_id"].astype(str).unique() if kinds.get(r, "human") == "human") if not ratings.empty else []
default_rater = str(cfg.get("default_rater_id", "friend_a"))

col_a, col_b, col_c, col_d = st.columns([1.2, 1, 1, 1])
rater_id = col_a.selectbox("Rater", humans or [default_rater], index=(humans.index(default_rater) if default_rater in humans else 0))
top_n = int(col_b.number_input("Beste Fotos", min_value=4, max_value=60, value=12, step=4))
year_text = col_c.text_input("Aufnahmejahr (leer = alle)", value="")
originals_only = col_d.checkbox("Nur Originale", value=True)
title = st.text_input("Titel", value=f"Wolken des Jahres {year_text}".strip())
year = int(year_text) if year_text.strip().isdigit() else None

if st.button("Bericht erzeugen", type="primary"):
    with st.spinner("Sammle Artefakte und rendere ..."):
        data, html = build_year_report_impl({**dataset_cfg, "default_rater_id": default_rater}, rater_id=rater_id, top_n=top_n, year=year, originals_only=originals_only, title=title or None)
        path = save_report(html)
    st.session_state["report_html"] = html
    st.session_state["report_path"] = str(path)
    st.session_state["report_scope"] = data["scope"]

if "report_html" in st.session_state:
    scope = st.session_state["report_scope"]
    st.success(f"Gespeichert unter {st.session_state['report_path']} · {scope['n_images']} Fotos, {scope['n_labeled']} bewertet, {scope['n_top']} mit ≥ 9")
    st.download_button("HTML herunterladen", st.session_state["report_html"].encode("utf-8"), file_name=Path(st.session_state["report_path"]).name, mime="text/html")
    with st.expander("Vorschau", expanded=True):
        components.html(st.session_state["report_html"], height=1400, scrolling=True)

reports_dir = resolve_path(REPORT_DIR)
if reports_dir.exists():
    existing = sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if existing:
        st.subheader("Frühere Berichte")
        rows = [{"Datei": p.name, "Größe (KB)": p.stat().st_size // 1024, "Geändert": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")} for p in existing[:20]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
