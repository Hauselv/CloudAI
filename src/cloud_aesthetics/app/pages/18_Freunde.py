from __future__ import annotations

import socket
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, dataset_config
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.friends.queue import FriendQueueConfig, build_friend_queue, progress
from cloud_aesthetics.friends.registry import add_friend, friend_link, load_friends, reactivate_friend, revoke_friend

st.set_page_config(page_title="Freunde", layout="wide")
st.title("Freunde: persönliche Label-Links")
st.caption(
    "Jeder Freund bekommt einen Link mit eigenem Token; die Bewertungen landen direkt in einer eigenen Rater-Datei "
    "und tauchen sofort in Rater-Vergleich, Streitbildern und Rater-Matrix auf. Der Link ist der Zugang: nicht öffentlich posten, "
    "bei Bedarf hier sperren. Freunde sehen weder deine Noten noch die Judge-Scores."
)

dataset_cfg = dataset_config()
ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
manifest = active_manifest(dataset_cfg["manifest_path"])

with st.sidebar:
    st.header("Server")
    port = int(st.number_input("Port", 1024, 65535, 8765))
    st.session_state.setdefault("friend_base_url", f"http://localhost:{port}")
    base_url = st.text_input("Öffentliche Adresse (Tunnel-URL)", value=st.session_state["friend_base_url"], help="Nach `friends-tunnel` die ausgegebene https://…trycloudflare.com-Adresse hier eintragen; die Links unten verwenden sie.")
    st.session_state["friend_base_url"] = base_url
    reachable = False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            reachable = True
    except OSError:
        reachable = False
    st.caption(("✅ Server läuft auf Port " if reachable else "⬜ Kein Server auf Port ") + str(port))
    st.code(f"python -m cloud_aesthetics.cli friends-serve --port {port}", language="bash")
    st.code(f"python -m cloud_aesthetics.cli friends-tunnel --port {port}", language="bash")
    st.caption("Zwei Terminals: erst der Server, dann der Tunnel. Der Tunnel druckt die öffentliche Adresse; sie ändert sich bei jedem Start (Quick Tunnel ohne Konto).")

col_add, col_info = st.columns([1, 1.4])
with col_add:
    st.subheader("Freund hinzufügen")
    name = st.text_input("Name", placeholder="z. B. Anna")
    if st.button("Link erzeugen", type="primary", disabled=not name.strip()):
        row = add_friend(name.strip())
        st.success(f"{row['display_name']} angelegt als Rater {row['rater_id']}")
        st.rerun()
with col_info:
    st.subheader("Was Freunde sehen")
    queue_cfg = FriendQueueConfig()
    core_ids = set(ratings.loc[ratings["rater_id"].astype(str) == queue_cfg.core_rater, "image_id"].astype(str)) if not ratings.empty else set()
    sample_queue = build_friend_queue(manifest, token="preview", core_rated_ids=core_ids, config=queue_cfg) if not manifest.empty else []
    n_core = len([i for i in sample_queue if i in core_ids])
    st.write(f"{len(sample_queue)} Bilder (Originale und große Ausschnitte, ohne Ausschlüsse). Zuerst die {n_core} Bilder, die du schon bewertet hast, damit Übereinstimmung messbar wird, dann die Lücken aus Karte und Triage, dann der Rest. Jeder Freund in eigener zufälliger Reihenfolge.")
    st.write(f"Duelle: zwischen den eigenen Bildern ab {queue_cfg.duel_threshold:g}.")

friends = load_friends()
st.subheader("Freunde")
if friends.empty:
    st.info("Noch keine Freunde angelegt.")
else:
    counts = ratings.groupby("rater_id")["image_id"].nunique() if not ratings.empty else pd.Series(dtype=int)
    last = ratings.groupby("rater_id")["rating_timestamp"].max() if not ratings.empty else pd.Series(dtype=str)
    rows = []
    for row in friends.itertuples():
        rated = set(ratings.loc[ratings["rater_id"].astype(str) == row.rater_id, "image_id"].astype(str)) if not ratings.empty else set()
        done = progress(sample_queue, rated, core_ids) if sample_queue else {"n_total": 0, "n_done": 0, "n_core_done": 0, "n_core": 0}
        rows.append({"Name": row.display_name, "Rater": row.rater_id, "aktiv": bool(row.active), "bewertet": int(counts.get(row.rater_id, 0)), "Vergleichsset": f"{done['n_core_done']}/{done['n_core']}", "zuletzt": str(last.get(row.rater_id, ""))[:16], "Link": friend_link(base_url, row.token)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={"Link": st.column_config.LinkColumn("Link")})
    st.caption("Link markieren und kopieren; per Messenger an den Freund schicken. Auf dem Handy öffnen, Lesezeichen setzen.")
    col_a, col_b = st.columns(2)
    with col_a:
        target = st.selectbox("Freund sperren / freischalten", friends["rater_id"].tolist())
        is_active = bool(friends.set_index("rater_id").loc[target, "active"])
        if st.button("Sperren" if is_active else "Wieder freischalten"):
            (revoke_friend if is_active else reactivate_friend)(target)
            st.rerun()
    with col_b:
        st.caption("Gesperrte Links liefern „nicht gefunden“; die Bewertungen bleiben erhalten.")
