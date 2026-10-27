"""Entry point: three work areas, a shared sidebar (rater, pipeline health) and cached table reads.

The page files under ``pages/`` are unchanged Streamlit scripts; ``st.navigation`` groups them
and keeps their previous URL paths so bookmarks keep working.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import app_config, shared_sidebar

cfg = app_config()
st.set_page_config(page_title=cfg.get("title", "Cloud Aesthetics"), page_icon="⛅", layout="wide")

PAGES = Path(__file__).parent / "pages"


def _page(file: str, title: str, icon: str, url_path: str, *, default: bool = False) -> st.Page:
    return st.Page(str(PAGES / file), title=title, icon=icon, url_path=url_path, default=default)


navigation = st.navigation(
    {
        "Daten & Labeln": [
            _page("00_Start.py", "Start & Status", "🏠", "Start", default=True),
            _page("0_Import.py", "Import", "📥", "Import"),
            _page("1_Rate.py", "Bewerten", "⭐", "Rate"),
            _page("10_Triage.py", "Triage", "🎯", "Triage"),
            _page("7_Exclusions.py", "Ausschlüsse", "🚫", "Exclusions"),
            _page("8_Source_Groups.py", "Quellgruppen", "🗂️", "Source_Groups"),
            _page("5_Gallery.py", "Galerie", "🖼️", "Gallery"),
        ],
        "Trainieren & Vergleichen": [
            _page("2_Train.py", "Train & Leaderboard", "🏋️", "Train"),
            _page("4_Compare.py", "Runs vergleichen", "⚖️", "Compare"),
            _page("6_Batch_Report.py", "Batch Report", "📦", "Batch_Report"),
            _page("9_Rater_Comparison.py", "Rater-Vergleich", "🤝", "Rater_Comparison"),
            _page("15_Rater_Analytik.py", "Rater-Analytik", "🧭", "Rater_Analytik"),
        ],
        "Verstehen": [
            _page("3_Inspect.py", "Inspect", "🔍", "Inspect"),
            _page("11_Ten_vs_Eight.py", "10 vs 8", "🔟", "Ten_vs_Eight"),
            _page("12_Was_waere_wenn.py", "Was wäre wenn", "🎛️", "Was_waere_wenn"),
            _page("13_Konzeptmodell.py", "Konzeptmodell", "🧩", "Konzeptmodell"),
            _page("14_Karte.py", "Karte", "🗺️", "Karte"),
            _page("16_Wolkenatlas.py", "Wolkenatlas", "☁️", "Wolkenatlas"),
            _page("17_Bericht.py", "Bericht", "📄", "Bericht"),
        ],
    },
    expanded=True,
)

shared_sidebar()
navigation.run()
