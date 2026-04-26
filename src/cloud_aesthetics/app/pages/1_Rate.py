from __future__ import annotations

import base64
import random
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import app_config, dataset_config, safe_read_table
from cloud_aesthetics.cli import ingest_images_impl
from cloud_aesthetics.data.exclusions import active_excluded_ids, set_exclusion
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings, record_pairwise_preference, record_rating
from cloud_aesthetics.settings import resolve_path

cfg = app_config()
st.set_page_config(page_title="Rate Images", layout="wide")
st.title("Rate Cloud Images")

dataset_cfg = dataset_config()
manifest = safe_read_table(dataset_cfg["manifest_path"])

if manifest.empty:
    st.warning("No images found yet. Run the ingest command first and place images in data/raw/images.")
    st.stop()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 1.2rem; }
    div[data-testid="stImage"] img { max-height: calc(100vh - 310px); object-fit: contain; }
    .cloud-rate-image {
        width: 100%;
        max-height: calc(100vh - 300px);
        object-fit: contain;
        background: #111;
        border-radius: 6px;
    }
    .cloud-hotkeys {
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: -0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _image_html(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="cloud-rate-image" src="data:image/{mime};base64,{encoded}" />'


def _install_hotkeys() -> None:
    components.html(
        """
        <script>
        const mappings = {
          ArrowLeft: "Previous",
          ArrowRight: "Next",
          Enter: "Save & Next"
        };
        function clickButton(label) {
          const buttons = Array.from(window.parent.document.querySelectorAll("button"));
          const target = buttons.find((button) => button.innerText.trim() === label);
          if (target && !target.disabled) target.click();
        }
        window.parent.document.onkeydown = (event) => {
          const tag = window.parent.document.activeElement?.tagName?.toLowerCase();
          if (tag === "input" || tag === "textarea" || tag === "select") return;
          const label = mappings[event.key];
          if (!label) return;
          event.preventDefault();
          clickButton(label);
        };
        </script>
        """,
        height=0,
    )


def _current_rated_ids(rater_id: str) -> set[str]:
    ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
    if ratings.empty:
        return set()
    return set(ratings.loc[ratings["rater_id"] == rater_id, "image_id"].dropna().astype(str))


def _current_excluded_ids() -> set[str]:
    return active_excluded_ids()


def _clamp_index(index: int, length: int) -> int:
    return max(0, min(index, length - 1))


def _move(delta: int, length: int) -> None:
    st.session_state.rate_index = _clamp_index(int(st.session_state.get("rate_index", 0)) + delta, length)


def _go_to_next_unrated(image_ids: list[str], rated_ids: set[str]) -> None:
    start = int(st.session_state.get("rate_index", 0))
    for offset in range(1, len(image_ids) + 1):
        candidate = (start + offset) % len(image_ids)
        if image_ids[candidate] not in rated_ids:
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


def _build_ordered_manifest(frame, mode: str, seed: int):
    if mode == "Random":
        ordered_ids = _random_order(frame, seed)
    elif mode == "Diverse groups":
        ordered_ids = _diverse_group_order(frame, seed)
    else:
        ordered_ids = _dataset_order(frame)
    order = {image_id: index for index, image_id in enumerate(ordered_ids)}
    ordered = frame.copy()
    ordered["image_id"] = ordered["image_id"].astype(str)
    ordered["_order"] = ordered["image_id"].map(order)
    return ordered.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


rater_id = st.text_input("Rater ID", value=str(cfg.get("default_rater_id", "friend_a")))
rating_session_id = st.text_input("Rating Session ID", value=datetime.utcnow().strftime("%Y%m%d_session"))
rated_ids = _current_rated_ids(rater_id)
excluded_ids = _current_excluded_ids()

tab_scalar, tab_pairwise = st.tabs(["Scalar Rating", "Pairwise Preference"])

with tab_scalar:
    order_col1, order_col2, order_col3 = st.columns([1.4, 1, 1])
    order_mode = order_col1.selectbox("Label order", ["Diverse groups", "Random", "Dataset order"])
    order_seed = int(order_col2.number_input("Seed", min_value=0, max_value=999999, value=int(st.session_state.get("label_order_seed", 42))))
    st.session_state.label_order_seed = order_seed
    if order_col3.button("New random seed", use_container_width=True):
        st.session_state.label_order_seed = random.randint(0, 999999)
        st.session_state.rate_index = 0
        st.rerun()

    if excluded_ids:
        manifest = manifest[~manifest["image_id"].astype(str).isin(excluded_ids)].reset_index(drop=True)
    ordered_manifest = _build_ordered_manifest(manifest, order_mode, order_seed)
    image_ids = ordered_manifest["image_id"].astype(str).tolist()
    if not image_ids:
        st.warning("All images are currently excluded from the active labeling pool.")
        st.stop()
    if "rate_index" not in st.session_state:
        st.session_state.rate_index = 0
    st.session_state.rate_index = _clamp_index(int(st.session_state.rate_index), len(image_ids))
    rated_count = sum(1 for item in image_ids if item in rated_ids)
    unrated_count = len(image_ids) - rated_count
    col_progress, col_position = st.columns([3, 1])
    col_progress.progress(rated_count / len(image_ids), text=f"Labeled {rated_count} / {len(image_ids)} | Open {unrated_count}")
    col_position.metric("Current", f"{st.session_state.rate_index + 1} / {len(image_ids)}")

    def _image_label(item: str) -> str:
        row_for_label = ordered_manifest.loc[ordered_manifest["image_id"].astype(str) == item].iloc[0]
        prefix = "[rated]" if item in rated_ids else "[open]"
        return f"{prefix} {item} - {row_for_label['relative_path']}"

    selected_id = st.selectbox(
        "Image",
        image_ids,
        index=st.session_state.rate_index,
        format_func=_image_label,
    )
    st.session_state.rate_index = image_ids.index(selected_id)
    image_id = selected_id
    row = ordered_manifest.loc[ordered_manifest["image_id"] == image_id].iloc[0]
    image_path = resolve_path(row["relative_path"])
    st.markdown(_image_html(image_path), unsafe_allow_html=True)
    st.caption(str(row["relative_path"]))

    score = st.slider("Score", min_value=0.0, max_value=10.0, value=8.0, step=0.5)
    note = st.text_area("Optional notes")

    skip_reason = st.selectbox("Skip reason", ["not_cloud_training_data", "satellite", "airplane_window", "too_much_landscape", "duplicate", "bad_crop", "other"])

    col_prev, col_skip, col_save, col_next, col_unrated = st.columns([1, 1.1, 1.3, 1, 1.5])
    if col_prev.button("Previous", use_container_width=True):
        _move(-1, len(image_ids))
        st.rerun()
    if col_next.button("Next", use_container_width=True):
        _move(1, len(image_ids))
        st.rerun()
    if col_unrated.button("Next unrated", use_container_width=True):
        _go_to_next_unrated(image_ids, rated_ids)
        st.rerun()
    if col_skip.button("Skip / exclude", use_container_width=True):
        try:
            set_exclusion(image_id, excluded=True, reason=skip_reason, relative_path=str(row["relative_path"]))
        except TypeError:
            set_exclusion(image_id, excluded=True, reason=f"{skip_reason} | {row['relative_path']}")
        ingest_images_impl()
        image_ids.remove(image_id)
        st.session_state.rate_index = _clamp_index(int(st.session_state.get("rate_index", 0)), max(1, len(image_ids)))
        st.rerun()
    if col_save.button("Save & Next", type="primary", use_container_width=True):
        record_rating(dataset_cfg["ratings_dir"], image_id=image_id, rater_id=rater_id, score=score, rating_session_id=rating_session_id, note=note)
        rated_ids.add(image_id)
        _go_to_next_unrated(image_ids, rated_ids)
        st.rerun()
    _install_hotkeys()
    st.markdown('<div class="cloud-hotkeys">Hotkeys: left previous | right next | Enter save & next</div>', unsafe_allow_html=True)

with tab_pairwise:
    left_image_id = st.selectbox("Left image", manifest["image_id"].tolist(), key="left_image")
    right_choices = [item for item in manifest["image_id"].tolist() if item != left_image_id]
    right_image_id = st.selectbox("Right image", right_choices, key="right_image")
    col1, col2 = st.columns(2)
    with col1:
        left_row = manifest.loc[manifest["image_id"] == left_image_id].iloc[0]
        st.image(str(resolve_path(left_row["relative_path"])), caption=f"Left: {left_image_id}", use_container_width=True)
    with col2:
        right_row = manifest.loc[manifest["image_id"] == right_image_id].iloc[0]
        st.image(str(resolve_path(right_row["relative_path"])), caption=f"Right: {right_image_id}", use_container_width=True)
    preference = st.radio("Which image is better?", [left_image_id, right_image_id, "Tie"], horizontal=True)
    strength = st.slider("Preference strength", min_value=0.0, max_value=1.0, value=0.5)
    if st.button("Save pairwise preference"):
        winner = None if preference == "Tie" else preference
        record_pairwise_preference(
            dataset_cfg["pairwise_dir"],
            left_image_id=left_image_id,
            right_image_id=right_image_id,
            rater_id=rater_id,
            winner=winner,
            tie_flag=preference == "Tie",
            preference_strength=strength,
        )
        st.success("Saved pairwise annotation.")
