from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, app_config, dataset_config
from cloud_aesthetics.data.derivatives import crop_context, load_derivatives, render_crop_context
from cloud_aesthetics.data.exclusions import active_excluded_ids, set_exclusion
from cloud_aesthetics.data.pairwise import bradley_terry_scores, select_duel_pair
from cloud_aesthetics.data.ratings import (
    load_raw_pairwise_preferences,
    load_raw_scalar_ratings,
    record_pairwise_preference,
    record_rating,
    void_rating,
)
from cloud_aesthetics.judge.store import rater_kinds
from cloud_aesthetics.judge.triage import load_triage
from cloud_aesthetics.settings import resolve_path

cfg = app_config()
st.set_page_config(page_title="Rate Images", layout="wide")
st.title("Rate Cloud Images")

dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])
derivatives = load_derivatives()

if manifest.empty:
    st.warning("No images found yet. Run the ingest command first and place images in data/raw/images.")
    st.stop()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 1.2rem; }
    div[data-testid="stImage"] img { max-height: calc(100vh - 310px); object-fit: contain; }
    .cloud-hotkeys { color: #6b7280; font-size: 0.85rem; margin-top: -0.5rem; }
    .cloud-score { font-size: 2.6rem; font-weight: 700; line-height: 1; padding-top: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

SCORE_STEPS = [value / 2.0 for value in range(0, 21)]
HOTKEY_HELP = (
    "Hotkeys: 0–9 Score setzen · + oder = für 10 · . oder , für ±½ · Enter speichern & weiter · ← → blättern · "
    "Backspace oder U rückgängig · Duell: A links, L rechts, T unentschieden, N nächstes Paar"
)


def _install_hotkeys() -> None:
    components.html(
        """
        <script>
        const mappings = {
          ArrowLeft: "Previous",
          ArrowRight: "Next",
          Enter: "Save & Next",
          Backspace: "Undo",
          u: "Undo", U: "Undo",
          "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
          "+": "10", "=": "10",
          ".": "+½", ",": "+½",
          a: "Links besser", A: "Links besser",
          l: "Rechts besser", L: "Rechts besser",
          t: "Unentschieden", T: "Unentschieden",
          n: "Nächstes Duell", N: "Nächstes Duell"
        };
        function clickButton(label) {
          const buttons = Array.from(window.parent.document.querySelectorAll("button"));
          const visible = buttons.filter((button) => button.offsetParent !== null);
          const target = visible.find((button) => button.innerText.trim() === label);
          if (target && !target.disabled) target.click();
        }
        window.parent.document.onkeydown = (event) => {
          const tag = window.parent.document.activeElement?.tagName?.toLowerCase();
          if (tag === "input" || tag === "textarea" || tag === "select") return;
          if (event.ctrlKey || event.metaKey || event.altKey) return;
          const label = mappings[event.key];
          if (!label) return;
          event.preventDefault();
          clickButton(label);
        };
        </script>
        """,
        height=0,
    )


def _all_ratings() -> pd.DataFrame:
    return load_raw_scalar_ratings(dataset_cfg["ratings_dir"])


def _current_rated_ids(ratings: pd.DataFrame, rater_id: str) -> set[str]:
    if ratings.empty:
        return set()
    return set(ratings.loc[ratings["rater_id"].astype(str) == rater_id, "image_id"].dropna().astype(str))


def _clamp_index(index: int, length: int) -> int:
    return max(0, min(index, length - 1))


def _move(delta: int, length: int) -> None:
    st.session_state.rate_index = _clamp_index(int(st.session_state.get("rate_index", 0)) + delta, length)


def _go_to_next_unrated(image_ids: list[str], rated_ids: set[str], also_open: set[str] | None = None) -> None:
    """Next image without a rating from this rater; ``also_open`` (blind retests) count as open once."""
    start = int(st.session_state.get("rate_index", 0))
    done_this_session = {item["image_id"] for item in st.session_state.get("rate_history", [])}
    for offset in range(1, len(image_ids) + 1):
        candidate = (start + offset) % len(image_ids)
        image = image_ids[candidate]
        if image not in rated_ids or (also_open and image in also_open and image not in done_this_session):
            st.session_state.rate_index = candidate
            return
    _move(1, len(image_ids))


def _dataset_order(frame) -> list[str]:
    return frame["image_id"].astype(str).tolist()


def _random_order(frame, seed: int) -> list[str]:
    image_ids = _dataset_order(frame)
    random.Random(seed).shuffle(image_ids)
    return image_ids


def _diverse_group_order(frame, seed: int) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        group_id = str(row.get("split_group_id", "default"))
        grouped.setdefault(group_id, []).append(str(row["image_id"]))
    rng = random.Random(seed)
    group_items = list(grouped.items())
    for _, image_ids in group_items:
        rng.shuffle(image_ids)
    rng.shuffle(group_items)
    ordered: list[str] = []
    while group_items:
        next_round = []
        for group_id, image_ids in group_items:
            ordered.append(image_ids.pop(0))
            if image_ids:
                next_round.append((group_id, image_ids))
        group_items = next_round
    return ordered


def _judge_order(frame) -> list[str]:
    """Triage order from the model raters: unlabeled originals first, high judge scores first."""
    triage = load_triage()
    if triage.empty or "label_rank" not in triage.columns:
        return _dataset_order(frame)
    ranks = triage.set_index("image_id")["label_rank"]
    ids = frame["image_id"].astype(str)
    return ids.iloc[ids.map(ranks).fillna(float("inf")).argsort(kind="stable").to_numpy()].tolist()


def _build_ordered_manifest(frame, mode: str, seed: int):
    if mode == "Random":
        ordered_ids = _random_order(frame, seed)
    elif mode == "Diverse groups":
        ordered_ids = _diverse_group_order(frame, seed)
    elif mode == "Judge-Vorschlag":
        ordered_ids = _judge_order(frame)
    else:
        ordered_ids = _dataset_order(frame)
    order = {image_id: index for index, image_id in enumerate(ordered_ids)}
    ordered = frame.copy()
    ordered["image_id"] = ordered["image_id"].astype(str)
    ordered["_order"] = ordered["image_id"].map(order)
    return ordered.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _format_score(value: float) -> str:
    return f"{value:g}"


rater_id = st.text_input("Rater ID", value=str(cfg.get("default_rater_id", "friend_a")))
rating_session_id = st.text_input("Rating Session ID", value=datetime.utcnow().strftime("%Y%m%d_session"))
ratings = _all_ratings()
rated_ids = _current_rated_ids(ratings, rater_id)
excluded_ids = active_excluded_ids()
kinds = rater_kinds(dataset_cfg.get("raters_path", "data/raw/metadata/raters.csv"))
st.session_state.setdefault("rate_score", 8.0)
st.session_state.setdefault("rate_history", [])

tab_scalar, tab_pairwise = st.tabs(["Scalar Rating", "Duelle & Paarvergleich"])

with tab_scalar:
    order_col1, order_col2, order_col3 = st.columns([1.4, 1, 1])
    order_mode = order_col1.selectbox(
        "Label order",
        ["Diverse groups", "Random", "Judge-Vorschlag", "Dataset order"],
        help="Judge-Vorschlag: Reihenfolge aus der Triage-Seite (ungelabelte Originale zuerst, hohe Judge-Scores zuerst).",
    )
    order_seed = int(order_col2.number_input("Seed", min_value=0, max_value=999999, value=int(st.session_state.get("label_order_seed", 42))))
    st.session_state.label_order_seed = order_seed
    if order_col3.button("New random seed", use_container_width=True):
        st.session_state.label_order_seed = random.randint(0, 999999)
        st.session_state.rate_index = 0
        st.rerun()

    if excluded_ids:
        manifest = manifest[~manifest["image_id"].astype(str).isin(excluded_ids)].reset_index(drop=True)
    kind_options = ["Alle", "Nur Originale", "Nur große Crops", "Nur kleine Crops", "Alle Crops"]
    option_col1, option_col2, option_col3 = st.columns([1.4, 1, 1])
    kind_filter = option_col1.selectbox("Bildart", kind_options, help="Originale und Crops werden sehr unterschiedlich bewertet; für die 10-vs-8-Frage zählen vor allem Originale.")
    quick_mode = option_col2.checkbox("Schnellmodus", value=False, help="Zifferntaste speichert sofort und springt weiter (±½ vorher mit . setzen).")
    show_judges = option_col3.checkbox("Judge-Scores einblenden", value=False, help="Standardmäßig aus, damit die Modelle dein Rating nicht verankern.")
    if "crop_class" in manifest.columns:
        if kind_filter == "Nur Originale":
            manifest = manifest[manifest["kind"] == "original"]
        elif kind_filter == "Nur große Crops":
            manifest = manifest[manifest["crop_class"] == "large_crop"]
        elif kind_filter == "Nur kleine Crops":
            manifest = manifest[manifest["crop_class"] == "small_crop"]
        elif kind_filter == "Alle Crops":
            manifest = manifest[manifest["kind"] == "crop"]
        manifest = manifest.reset_index(drop=True)
    if st.session_state.get("rate_kind_filter") != kind_filter:
        st.session_state.rate_kind_filter = kind_filter
        st.session_state.rate_index = 0
    if manifest.empty:
        st.warning("Keine Bilder in dieser Auswahl.")
        st.stop()
    ordered_manifest = _build_ordered_manifest(manifest, order_mode, order_seed)
    image_ids = ordered_manifest["image_id"].astype(str).tolist()
    if "rate_index" not in st.session_state:
        st.session_state.rate_index = 0
    st.session_state.rate_index = _clamp_index(int(st.session_state.rate_index), len(image_ids))
    rated_count = sum(1 for item in image_ids if item in rated_ids)
    unrated_count = len(image_ids) - rated_count
    col_progress, col_position = st.columns([3, 1])
    col_progress.progress(rated_count / len(image_ids), text=f"Labeled {rated_count} / {len(image_ids)} | Open {unrated_count}")
    col_position.metric("Current", f"{st.session_state.rate_index + 1} / {len(image_ids)}")
    last_retest = st.session_state.get("rate_last_retest")
    if last_retest:
        delta = last_retest["now"] - last_retest["earlier"]
        st.caption(f"Letzte Wiederholung {last_retest['image_id']}: früher {last_retest['earlier']:g}, jetzt {last_retest['now']:g} ({delta:+g}).")

    retest_set = set()
    if order_mode == "Judge-Vorschlag":
        queue_table = load_triage()
        if not queue_table.empty and "retest" in queue_table.columns:
            retest_set = set(queue_table.loc[queue_table["retest"].astype(bool), "image_id"].astype(str))

    def _image_label(item: str) -> str:
        row_for_label = ordered_manifest.loc[ordered_manifest["image_id"].astype(str) == item].iloc[0]
        prefix = "[retest]" if item in retest_set else ("[rated]" if item in rated_ids else "[open]")
        return f"{prefix} {item} - {row_for_label['relative_path']}"

    selected_id = st.selectbox("Image", image_ids, index=st.session_state.rate_index, format_func=_image_label)
    st.session_state.rate_index = image_ids.index(selected_id)
    image_id = selected_id
    row = ordered_manifest.loc[ordered_manifest["image_id"] == image_id].iloc[0]
    image_path = resolve_path(row["relative_path"])
    context = crop_context(str(row["relative_path"]), derivatives) if str(row.get("kind", "")) == "crop" else None
    if context is not None:
        col_crop, col_source = st.columns([1, 1])
        col_crop.image(str(image_path), use_container_width=True)
        col_crop.caption(f"Crop | {row.get('crop_class', 'crop')} | {row['relative_path']}")
        source_view = render_crop_context(context)
        if source_view is not None:
            col_source.image(source_view, use_container_width=True)
            scale_text = f"{context.crop_scale:.0%} der kurzen Kante" if context.crop_scale else ""
            col_source.caption(f"Quellbild mit Ausschnitt | {context.width}x{context.height} px | {scale_text}")
        else:
            col_source.info("Quellbild nicht gefunden.")
    else:
        st.image(str(image_path), use_container_width=True)
        st.caption(f"{row.get('kind', 'original')} | {row['relative_path']}")

    # Queue context: why this image is next, and whether it is a blind retest
    triage_row = None
    triage_table = load_triage()
    if not triage_table.empty:
        hits = triage_table[triage_table["image_id"] == image_id]
        triage_row = hits.iloc[0] if not hits.empty else None
    is_retest = bool(triage_row is not None and bool(triage_row.get("retest", False)) and image_id not in {item["image_id"] for item in st.session_state.rate_history})
    if triage_row is not None and str(triage_row.get("queue_reason", "")):
        st.info(("Wiederholung: Bitte neu bewerten, deine frühere Note bleibt verborgen." if is_retest else "Warum jetzt: " + str(triage_row["queue_reason"])), icon="🎯")

    # Previous ratings for this image (own first), model raters only on request; hidden for blind retests
    previous = ratings[ratings["image_id"].astype(str) == image_id].copy() if not ratings.empty else pd.DataFrame()
    if not previous.empty:
        previous["rater_id"] = previous["rater_id"].astype(str)
        previous["kind"] = previous["rater_id"].map(lambda value: kinds.get(value, "human"))
        if not show_judges:
            previous = previous[previous["kind"] == "human"]
        if is_retest:
            previous = previous[previous["rater_id"] != rater_id]
        if not previous.empty:
            previous = previous.sort_values(["rater_id", "rating_timestamp"])
            parts = []
            for _, item in previous.iterrows():
                stamp = str(item.get("rating_timestamp", ""))[:10]
                who = "du" if item["rater_id"] == rater_id else item["rater_id"]
                parts.append(f"{who}: **{float(item['raw_score_1_to_10']):g}** ({stamp})")
            st.caption("Bisherige Bewertungen · " + " · ".join(parts))
    if show_judges:
        triage = load_triage()
        if not triage.empty:
            hit = triage[triage["image_id"] == image_id]
            if not hit.empty and pd.notna(hit.iloc[0].get("judge_mean")):
                item = hit.iloc[0]
                st.caption(f"Judges: Mittel {float(item['judge_mean']):.1f} (Spannweite {float(item['judge_spread']):.1f}) · {item.get('judge_cloud_type', '')}, {item.get('judge_light', '')} · {item.get('judge_verdict', '')}")

    def _save(score: float) -> None:
        saved = record_rating(dataset_cfg["ratings_dir"], image_id=image_id, rater_id=rater_id, score=score, rating_session_id=rating_session_id, note=st.session_state.get("rate_note", ""))
        history = list(st.session_state.rate_history)
        history.append({"rating_id": saved["rating_id"], "image_id": image_id, "score": score, "index": st.session_state.rate_index})
        st.session_state.rate_history = history[-50:]
        if is_retest and not ratings.empty:
            earlier = ratings[(ratings["image_id"].astype(str) == image_id) & (ratings["rater_id"].astype(str) == rater_id)]["raw_score_1_to_10"].astype(float)
            if not earlier.empty:
                st.session_state.rate_last_retest = {"image_id": image_id, "earlier": float(earlier.iloc[-1]), "now": score}
        rated_ids.add(image_id)
        st.session_state.rate_note = ""
        _go_to_next_unrated(image_ids, rated_ids, also_open=retest_set)
        st.rerun()

    score_col, buttons_col = st.columns([1, 6])
    with score_col:
        st.markdown(f'<div class="cloud-score">{_format_score(float(st.session_state.rate_score))}</div>', unsafe_allow_html=True)
        st.caption("Score")
    with buttons_col:
        grid = st.columns(12)
        for value in range(11):
            if grid[value].button(str(value), key=f"score_btn_{value}", use_container_width=True):
                st.session_state.rate_score = float(value)
                if quick_mode:
                    _save(float(value))
                st.rerun()
        if grid[11].button("+½", key="score_btn_half", use_container_width=True, help="Halben Punkt dazu oder weg"):
            current = float(st.session_state.rate_score)
            st.session_state.rate_score = min(10.0, current + 0.5) if current == int(current) else current - 0.5
            st.rerun()
    st.session_state.rate_score = st.slider("Feinjustierung", min_value=0.0, max_value=10.0, value=float(st.session_state.rate_score), step=0.5, label_visibility="collapsed")
    st.text_area("Optional notes", key="rate_note")

    skip_reason = st.selectbox("Skip reason", ["not_cloud_training_data", "satellite", "airplane_window", "too_much_landscape", "duplicate", "bad_crop", "other"])

    col_prev, col_skip, col_save, col_undo, col_next, col_unrated = st.columns([1, 1.1, 1.3, 1, 1, 1.4])
    if col_prev.button("Previous", use_container_width=True):
        _move(-1, len(image_ids))
        st.rerun()
    if col_next.button("Next", use_container_width=True):
        _move(1, len(image_ids))
        st.rerun()
    if col_unrated.button("Next unrated", use_container_width=True):
        _go_to_next_unrated(image_ids, rated_ids, also_open=retest_set)
        st.rerun()
    if col_skip.button("Skip / exclude", use_container_width=True):
        set_exclusion(image_id, excluded=True, reason=skip_reason, relative_path=str(row["relative_path"]))
        image_ids.remove(image_id)
        st.session_state.rate_index = _clamp_index(int(st.session_state.get("rate_index", 0)), max(1, len(image_ids)))
        st.rerun()
    if col_save.button("Save & Next", type="primary", use_container_width=True):
        _save(float(st.session_state.rate_score))
    last = st.session_state.rate_history[-1] if st.session_state.rate_history else None
    if col_undo.button("Undo", use_container_width=True, disabled=last is None, help=f"Letzte Bewertung zurücknehmen ({last['score']:g} für {last['image_id']})" if last else "Noch nichts zu widerrufen"):
        void_rating(dataset_cfg["ratings_dir"], last["rating_id"], reason="undo")
        st.session_state.rate_history = st.session_state.rate_history[:-1]
        if last["image_id"] in image_ids:
            st.session_state.rate_index = image_ids.index(last["image_id"])
        st.session_state.rate_score = float(last["score"])
        st.rerun()
    _install_hotkeys()
    st.markdown(f'<div class="cloud-hotkeys">{HOTKEY_HELP}</div>', unsafe_allow_html=True)

with tab_pairwise:
    st.subheader("Duelle am oberen Ende")
    st.caption(
        "Paarvergleiche trennen 10 von 8,5 zuverlässiger als absolute Noten. Der Pool sind deine gut bewerteten Bilder, "
        "wahlweise plus ungelabelte Triage-Kandidaten. Paare mit wenigen Duellen und ähnlichem Score kommen zuerst."
    )
    pool_col1, pool_col2, pool_col3, pool_col4 = st.columns([1.2, 1.2, 1.2, 1])
    pool_min = pool_col1.slider("Gelabelt ab Score", 5.0, 10.0, 8.0, 0.5)
    pool_kind = pool_col2.selectbox("Bildart (Duell)", ["Nur Originale", "Alle"], key="duel_kind")
    include_triage = pool_col3.checkbox("Triage-Kandidaten (Judge ≥ 7) einbeziehen", value=False)
    duel_seed = int(pool_col4.number_input("Seed (Duell)", min_value=0, max_value=999999, value=7))

    human_ratings = ratings[ratings["rater_id"].astype(str).map(lambda value: kinds.get(value, "human") == "human")] if not ratings.empty else ratings
    pool = pd.DataFrame(columns=["image_id", "score"])
    if not human_ratings.empty:
        means = human_ratings.groupby(human_ratings["image_id"].astype(str))["raw_score_1_to_10"].mean().rename("score").reset_index()
        pool = means[means["score"] >= pool_min]
    if include_triage:
        triage = load_triage()
        if not triage.empty:
            extra = triage[(~triage["labeled"]) & (triage["judge_mean"] >= 7.0) & (triage["proposed_exclusion"] == "")][["image_id", "judge_mean"]].rename(columns={"judge_mean": "score"})
            pool = pd.concat([pool, extra], ignore_index=True).drop_duplicates("image_id")
    pool_manifest = manifest.copy()
    pool_manifest["image_id"] = pool_manifest["image_id"].astype(str)
    if pool_kind == "Nur Originale" and "kind" in pool_manifest.columns:
        pool_manifest = pool_manifest[pool_manifest["kind"] == "original"]
    pool = pool[pool["image_id"].astype(str).isin(set(pool_manifest["image_id"]))].reset_index(drop=True)
    pairs_all = load_raw_pairwise_preferences(dataset_cfg["pairwise_dir"])
    my_pairs = pairs_all[pairs_all["rater_id"].astype(str) == rater_id] if not pairs_all.empty else pairs_all
    pool_ids = set(pool["image_id"].astype(str))
    pool_pairs = my_pairs[my_pairs["left_image_id"].astype(str).isin(pool_ids) & my_pairs["right_image_id"].astype(str).isin(pool_ids)] if not my_pairs.empty else my_pairs
    st.caption(f"Pool: {len(pool)} Bilder · deine Duelle im Pool: {len(pool_pairs)}")

    if len(pool) < 2:
        st.info("Mindestens zwei Bilder im Pool nötig. Schwelle senken oder Triage-Kandidaten einbeziehen.")
    else:
        pair_key = (rater_id, pool_min, pool_kind, include_triage, duel_seed, len(pool_pairs))
        if st.session_state.get("duel_pair_key") != pair_key or not st.session_state.get("duel_pair"):
            st.session_state.duel_pair = select_duel_pair(pool, pool_pairs, seed=duel_seed + len(pool_pairs))
            st.session_state.duel_pair_key = pair_key
        pair = st.session_state.duel_pair
        if pair is None:
            st.success("Alle Paare in diesem Pool sind verglichen. Pool erweitern oder Schwelle senken.")
        else:
            left_image_id, right_image_id = pair
            lookup = pool_manifest.set_index("image_id")
            pool_scores = pool.set_index("image_id")["score"]
            col_left, col_right = st.columns(2)
            for column, image, side in ((col_left, left_image_id, "Links"), (col_right, right_image_id, "Rechts")):
                with column:
                    path = resolve_path(lookup.loc[image, "relative_path"])
                    if path.exists():
                        st.image(str(path), use_container_width=True)
                    st.caption(f"{side} · {image} · Score {float(pool_scores.get(image, float('nan'))):.1f}")

            def _record(winner: str | None, tie: bool) -> None:
                record_pairwise_preference(dataset_cfg["pairwise_dir"], left_image_id=left_image_id, right_image_id=right_image_id, rater_id=rater_id, winner=winner, tie_flag=tie, preference_strength=None if tie else 1.0)
                st.session_state.duel_pair = None
                st.rerun()

            duel_cols = st.columns(4)
            if duel_cols[0].button("Links besser", type="primary", use_container_width=True):
                _record(left_image_id, False)
            if duel_cols[1].button("Unentschieden", use_container_width=True):
                _record(None, True)
            if duel_cols[2].button("Rechts besser", type="primary", use_container_width=True):
                _record(right_image_id, False)
            if duel_cols[3].button("Nächstes Duell", use_container_width=True, help="Paar überspringen"):
                st.session_state.duel_pair = select_duel_pair(pool, pd.concat([pool_pairs, pd.DataFrame([{"left_image_id": left_image_id, "right_image_id": right_image_id}])], ignore_index=True), seed=duel_seed + random.randint(0, 9999))
                st.rerun()

        all_pool_pairs = pairs_all[pairs_all["left_image_id"].astype(str).isin(pool_ids) & pairs_all["right_image_id"].astype(str).isin(pool_ids)] if not pairs_all.empty else pairs_all
        if not all_pool_pairs.empty:
            ranking = bradley_terry_scores(all_pool_pairs)
            ranking = ranking.merge(pool.rename(columns={"score": "score_absolut"}), on="image_id", how="left")
            st.write("Bradley-Terry-Rangfolge im Pool (alle Rater)")
            st.caption("bt_score ist auf 0–10 skaliert (Mittel 5, ±3 pro Standardabweichung). Vergleiche die Reihenfolge mit deinen absoluten Scores: Wo sie auseinanderlaufen, ist die absolute Skala zu grob.")
            st.dataframe(ranking[["image_id", "bt_score", "duels", "wins", "score_absolut"]].round(2), use_container_width=True, hide_index=True)

    with st.expander("Freier Paarvergleich (beliebige Bilder)"):
        left_free = st.selectbox("Left image", manifest["image_id"].astype(str).tolist(), key="left_image")
        right_choices = [item for item in manifest["image_id"].astype(str).tolist() if item != left_free]
        right_free = st.selectbox("Right image", right_choices, key="right_image")
        col1, col2 = st.columns(2)
        with col1:
            left_row = manifest.loc[manifest["image_id"].astype(str) == left_free].iloc[0]
            st.image(str(resolve_path(left_row["relative_path"])), caption=f"Left: {left_free}", use_container_width=True)
        with col2:
            right_row = manifest.loc[manifest["image_id"].astype(str) == right_free].iloc[0]
            st.image(str(resolve_path(right_row["relative_path"])), caption=f"Right: {right_free}", use_container_width=True)
        preference = st.radio("Which image is better?", [left_free, right_free, "Tie"], horizontal=True)
        strength = st.slider("Preference strength", min_value=0.0, max_value=1.0, value=0.5)
        if st.button("Save pairwise preference"):
            winner = None if preference == "Tie" else preference
            record_pairwise_preference(dataset_cfg["pairwise_dir"], left_image_id=left_free, right_image_id=right_free, rater_id=rater_id, winner=winner, tie_flag=preference == "Tie", preference_strength=strength)
            st.success("Saved pairwise annotation.")
