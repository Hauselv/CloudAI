from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_aesthetics.app.common import active_manifest, app_config, dataset_config, explain_impl, list_runs
from cloud_aesthetics.app.ui import render_image_grid
from cloud_aesthetics.data.derivatives import crop_context, render_crop_context
from cloud_aesthetics.features.context import load_context
from cloud_aesthetics.data.ratings import load_raw_scalar_ratings
from cloud_aesthetics.explain.heatmaps import feature_diagnostic_map, heatmap_to_rgb, overlay_heatmap_on_image, overlay_signed_heatmap, signed_heatmap_to_rgb
from cloud_aesthetics.explain.interpretation import generate_local_llm_interpretation
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image
from cloud_aesthetics.settings import resolve_path


@st.cache_data(show_spinner=False)
def _context_table() -> pd.DataFrame:
    return load_context()


def _context_row(image_id: str) -> pd.Series | None:
    table = _context_table()
    if table.empty:
        return None
    matches = table[table["image_id"] == image_id]
    return matches.iloc[0] if not matches.empty else None


def _image_kind(relative_path: object) -> str:
    return "crop" if "/crops/" in str(relative_path).replace("\\", "/") else "original"


def _heatmap_label(source: str) -> str:
    if source.startswith("occlusion:"):
        return "Occlusion (CLIP) - rot stützt den Score, blau senkt ihn"
    if source.startswith("occlusion_support:"):
        return "Occlusion support (CLIP) - Regionen, deren Verdecken den Score senkt"
    if source.startswith("gradcam:") and source.endswith(".last_block"):
        return "Grad-CAM auf CLIP-Patch-Tokens (7x7, nur lineares Modell)"
    labels = {
        "input_gradient": "Input gradient - edge-sensitive",
        "edge_fallback": "Edge fallback - diagnostic only",
        "gradcam:backbone.layer1": "Grad-CAM layer 1 - low-level texture",
        "gradcam:backbone.layer2": "Grad-CAM layer 2 - edges and cloud parts",
        "gradcam:backbone.layer3": "Grad-CAM layer 3 - mid-level structures",
        "gradcam:backbone.layer4": "Grad-CAM layer 4 - semantic regions",
    }
    return labels.get(source, source)


def _format_driver_strength(row: pd.Series) -> str:
    if row.get("attribution_kind") == "prediction_delta":
        return f"{float(row.get('effect', 0.0)):+.2f} score"
    return f"{float(row.get('z_score', 0.0)):+.2f} z"


def _driver_table(interpretation: dict[str, object]) -> pd.DataFrame:
    drivers = pd.DataFrame(interpretation.get("drivers", []))
    if drivers.empty:
        return drivers
    direction_labels = {
        "raises": "raises prediction",
        "lowers": "lowers prediction",
        "higher_than_typical": "higher than typical",
        "lower_than_typical": "lower than typical",
    }
    drivers["Signal"] = drivers["label"]
    drivers["Direction"] = drivers["direction"].map(direction_labels).fillna(drivers["direction"])
    drivers["Strength"] = drivers.apply(_format_driver_strength, axis=1)
    drivers["Value"] = drivers["value"].map(lambda value: f"{float(value):.3g}")
    if "typical_value" in drivers.columns:
        drivers["Typical"] = drivers["typical_value"].map(lambda value: f"{float(value):.3g}")
        return drivers[["Signal", "Direction", "Strength", "Value", "Typical"]]
    return drivers[["Signal", "Direction", "Strength", "Value"]]


def _render_interpretation(explanation: dict[str, object], cfg: dict[str, object]) -> None:
    interpretation = explanation.get("interpretation", {})
    if not interpretation:
        st.write(explanation.get("text_explanation", ""))
        return

    interval = interpretation.get("score_interval", {})
    score_band = interpretation.get("score_band", {})
    confidence_band = interpretation.get("confidence_band", {})
    human_label = interpretation.get("human_label", {})
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Predicted", f"{float(interpretation.get('score', 0.0)):.2f}")
    metric_col2.metric(
        "Likely range",
        f"{float(interval.get('low', 0.0)):.2f}-{float(interval.get('high', 0.0)):.2f}",
    )
    metric_col3.metric("Confidence", str(confidence_band.get("title", "Unknown")))
    metric_col4.metric("Action", str(interpretation.get("recommendation", "review")))
    st.write(str(interpretation.get("summary", "")))

    if human_label.get("available"):
        error = float(human_label.get("prediction_error", 0.0))
        st.caption(
            f"Human mean {float(human_label.get('mean_score', 0.0)):.2f}; "
            f"model error {error:+.2f}; "
            f"{int(human_label.get('n_raters', 0))} rater(s)."
        )
    else:
        st.caption(str(score_band.get("title", "Review")) + " based on model output only.")

    drivers = _driver_table(interpretation)
    if not drivers.empty:
        st.write("Main signals")
        st.dataframe(drivers, use_container_width=True, hide_index=True)

    actions = interpretation.get("next_actions", [])
    cautions = interpretation.get("cautions", [])
    if actions:
        st.write("Next action: " + " ".join(str(item) for item in actions))
    if cautions:
        st.warning(" ".join(str(item) for item in cautions))

    llm_cfg = dict(cfg.get("interpretation", {}) or {})
    if st.button("Generate local LLM summary", use_container_width=True):
        with st.spinner("Asking local LLM..."):
            result = generate_local_llm_interpretation(
                explanation,
                interpretation,
                provider=str(llm_cfg.get("llm_provider", "ollama")),
                model=str(llm_cfg.get("ollama_model", "llama3.2:3b-instruct")),
                url=str(llm_cfg.get("ollama_url", "http://localhost:11434/api/generate")),
                timeout_seconds=float(llm_cfg.get("timeout_seconds", 12)),
            )
        st.session_state.inspect_llm_summary = result
    llm_summary = st.session_state.get("inspect_llm_summary")
    if llm_summary:
        if llm_summary.get("available"):
            st.info(str(llm_summary.get("text", "")))
        else:
            st.caption(f"Local LLM unavailable: {llm_summary.get('error', 'unknown error')}")


st.set_page_config(page_title="Inspect Predictions", layout="wide")
st.title("Inspect Predictions")

cfg = app_config()
dataset_cfg = dataset_config()
manifest = active_manifest(dataset_cfg["manifest_path"])
runs = [run for run in list_runs() if (resolve_path("data/artifacts") / run / "summary.json").exists()]
if not runs:
    st.warning("No runs found yet. Train a model first.")
    st.stop()
if manifest.empty:
    st.warning("No manifest found yet.")
    st.stop()

ratings = load_raw_scalar_ratings(dataset_cfg["ratings_dir"])
rater_id = st.sidebar.text_input("Rater ID", value=str(cfg.get("default_rater_id", "friend_a")))
rated_ids = set()
if not ratings.empty:
    rated_ids = set(ratings.loc[ratings["rater_id"] == rater_id, "image_id"].dropna().astype(str))

manifest = manifest.copy()
manifest["image_id"] = manifest["image_id"].astype(str)
manifest["rated"] = manifest["image_id"].isin(rated_ids)
if "kind" not in manifest.columns:
    manifest["kind"] = manifest["relative_path"].apply(_image_kind)
manifest["kind"] = manifest["kind"].fillna(manifest["relative_path"].apply(_image_kind)).astype(str)

if "inspect_image_id" not in st.session_state:
    st.session_state.inspect_image_id = manifest["image_id"].iloc[0]
if "inspect_explanation" not in st.session_state:
    st.session_state.inspect_explanation = None
if "inspect_explanation_key" not in st.session_state:
    st.session_state.inspect_explanation_key = None
if "inspect_llm_summary" not in st.session_state:
    st.session_state.inspect_llm_summary = None

with st.sidebar:
    run_id = st.selectbox("Run", runs)
    status_filter = st.selectbox("Status", ["All", "Labeled", "Open"], key="inspect_status")
    kind_filter = st.selectbox("Kind", ["All", "Originals", "Crops"], key="inspect_kind")
    tile_count = st.selectbox("Tiles", [8, 16, 24, 48], index=2, key="inspect_tiles")

    image_view = manifest.copy()
    if status_filter == "Labeled":
        image_view = image_view[image_view["rated"]]
    elif status_filter == "Open":
        image_view = image_view[~image_view["rated"]]
    if kind_filter == "Originals":
        image_view = image_view[image_view["kind"] == "original"]
    elif kind_filter == "Crops":
        image_view = image_view[image_view["kind"] == "crop"]

    image_view = image_view.reset_index(drop=True)
    page_count = max(1, int((len(image_view) + tile_count - 1) // tile_count))
    if int(st.session_state.get("inspect_page", 1)) > page_count:
        st.session_state.inspect_page = 1
    page = st.number_input("Page", min_value=1, max_value=page_count, value=1, key="inspect_page")
    page_frame = image_view.iloc[(int(page) - 1) * tile_count : int(page) * tile_count]
    st.caption(f"{len(page_frame)} of {len(image_view)} images")
    for _, tile_row in page_frame.iterrows():
        image_path = resolve_path(tile_row["relative_path"])
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        status = "labeled" if bool(tile_row["rated"]) else "open"
        st.caption(f"{status} | {tile_row['kind']}\n\n{tile_row['image_id']}")
        if st.button("Select", key=f"inspect_tile_{tile_row['image_id']}", use_container_width=True):
            st.session_state.inspect_image_id = tile_row["image_id"]
            st.session_state.inspect_explanation = None
            st.session_state.inspect_explanation_key = None
            st.session_state.inspect_llm_summary = None
            st.rerun()

image_ids = manifest["image_id"].tolist()
selected_index = image_ids.index(st.session_state.inspect_image_id) if st.session_state.inspect_image_id in image_ids else 0
detail_col, action_col = st.columns([1.2, 1])
with action_col:
    image_id = st.selectbox("Selected image", image_ids, index=selected_index)
    if image_id != st.session_state.inspect_image_id:
        st.session_state.inspect_image_id = image_id
        st.session_state.inspect_explanation = None
        st.session_state.inspect_explanation_key = None
        st.session_state.inspect_llm_summary = None
        st.rerun()
    overlay_alpha = st.slider("Overlay strength", min_value=0.0, max_value=0.9, value=0.45, step=0.05)
    show_heatmap_only = st.checkbox("Show heatmap only", value=False)
    if st.button("Generate explanation", type="primary", use_container_width=True):
        with st.spinner("Generating explanation..."):
            st.session_state.inspect_explanation = explain_impl(run_id, image_id)
        st.session_state.inspect_explanation_key = (run_id, image_id)
        st.session_state.inspect_llm_summary = None

image_row = manifest.loc[manifest["image_id"] == st.session_state.inspect_image_id].iloc[0]
with detail_col:
    image_path = resolve_path(image_row["relative_path"])
    if image_path.exists():
        st.image(str(image_path), caption=f"{image_row['kind']} | {image_row.get('crop_class', '')} | {image_row['image_id']}", use_container_width=True)
        if str(image_row["kind"]) == "crop":
            context = crop_context(str(image_row["relative_path"]))
            source_view = render_crop_context(context) if context is not None else None
            if source_view is not None:
                scale_text = f" | {context.crop_scale:.0%} of the short edge" if context.crop_scale else ""
                st.image(source_view, caption=f"Source image with crop region{scale_text}", use_container_width=True)
    else:
        st.warning("Selected image is missing on disk.")

explanation = st.session_state.inspect_explanation
if explanation is None or st.session_state.inspect_explanation_key != (run_id, st.session_state.inspect_image_id):
    st.info("Select an image and generate an explanation.")
    st.stop()

image = read_rgb_image(image_row["relative_path"])
tab_overview, tab_heatmap, tab_signals, tab_neighbors, tab_debug = st.tabs(
    ["Interpretation", "Heatmap", "Signals", "Neighbors", "Debug"]
)

with tab_overview:
    _render_interpretation(explanation, cfg)
    if explanation.get("scientific_cloud_type"):
        scientific = explanation["scientific_cloud_type"]
        st.write("Scientific cloud estimate")
        cols = st.columns(3)
        cols[0].metric("Cloud type", str(scientific.get("primary_cloud_type", "unknown")))
        cols[1].metric("Confidence", f"{float(scientific.get('cloud_type_confidence', 0.0)):.2f}")
        cols[2].metric("Vision vote", str(scientific.get("vision_model_vote", "not_configured")))
        if scientific.get("rule_based_cloud_type"):
            st.caption(
                f"Rule-based estimate: {scientific['rule_based_cloud_type']} "
                f"({float(scientific.get('rule_based_confidence', 0.0)):.2f}); CLIP zero-shot is shown as primary."
            )
        if scientific.get("scientific_caveat"):
            st.caption(str(scientific["scientific_caveat"]))
    context_row = _context_row(str(image_row["image_id"]))
    if context_row is not None:
        st.write("Aufnahmekontext")
        ctx_cols = st.columns(4)
        when = str(context_row.get("datetime_utc") or context_row.get("date") or "unbekannt")
        ctx_cols[0].metric("Zeit (UTC)", when[:16].replace("T", " "), help=f"Qualität: {context_row.get('time_quality')} ({context_row.get('time_source')})")
        elevation = context_row.get("sun_elevation_deg")
        ctx_cols[1].metric("Sonnenhöhe", f"{float(elevation):.1f}°" if pd.notna(elevation) else "–", help=str(context_row.get("sun_phase", "")))
        hfs = context_row.get("hours_from_sunset")
        ctx_cols[2].metric("Std. bis Sonnenuntergang", f"{-float(hfs):+.1f} h" if pd.notna(hfs) else "–", help="negativ = vor Sonnenuntergang")
        cover = context_row.get("wx_cloud_cover")
        ctx_cols[3].metric("Bewölkung (Reanalyse)", f"{float(cover):.0f} %" if pd.notna(cover) else "–", help=f"Ort: {context_row.get('location_source')}")
        if pd.notna(context_row.get("wx_cloud_cover_low")):
            st.caption(
                f"Wolken tief/mittel/hoch: {float(context_row['wx_cloud_cover_low']):.0f} / {float(context_row['wx_cloud_cover_mid']):.0f} / "
                f"{float(context_row['wx_cloud_cover_high']):.0f} % · Feuchte {float(context_row['wx_relative_humidity_2m']):.0f} % · "
                f"Druck {float(context_row['wx_surface_pressure']):.0f} hPa · Niederschlag {float(context_row['wx_precipitation']):.1f} mm · "
                f"Temperatur {float(context_row['wx_temperature_2m']):.0f} °C"
            )
    zero_shot = explanation.get("zero_shot") or {}
    if zero_shot:
        st.write("CLIP zero-shot attributes")
        zs_metric1, zs_metric2, zs_metric3 = st.columns(3)
        zs_metric1.metric("CLIP cloud type", str(zero_shot.get("cloud_type", "unknown")).replace("_", " "))
        zs_metric2.metric("Top probability", f"{float(zero_shot.get('cloud_type_probability', 0.0)):.2f}")
        zs_metric3.metric("Margin to 2nd", f"{float(zero_shot.get('cloud_type_confidence', 0.0)):.2f}")
        zs_col1, zs_col2 = st.columns(2)
        type_frame = pd.DataFrame(zero_shot.get("cloud_type_scores", [])).set_index("cloud_type")
        if not type_frame.empty:
            zs_col1.caption("Cloud type probabilities")
            zs_col1.bar_chart(type_frame["score"])
        light_frame = pd.DataFrame(zero_shot.get("light_scores", [])).set_index("light")
        if not light_frame.empty:
            zs_col2.caption(f"Light: {zero_shot.get('light', 'unknown')}")
            zs_col2.bar_chart(light_frame["score"])
        concept_frame = pd.DataFrame(zero_shot.get("all_concepts", []))
        if not concept_frame.empty:
            concept_frame = concept_frame.sort_values("z_score", ascending=False)
            concept_frame["concept"] = concept_frame["concept"].str.replace("_", " ")
            st.caption("Concept signals relative to the dataset (z-score of similarity gain over a neutral cloud prompt)")
            st.bar_chart(concept_frame.set_index("concept")["z_score"])
        st.caption(str(zero_shot.get("caveat", "")))

with tab_heatmap:
    heatmap_variants = explanation.get("heatmap_variants", {})
    heatmap_choice = None
    if heatmap_variants:
        choices = list(heatmap_variants.keys())
        preferred = [choice for choice in choices if choice.startswith("occlusion:")] + [choice for choice in choices if choice == "gradcam:backbone.layer4"]
        default_index = choices.index(preferred[0]) if preferred else 0
        heatmap_choice = st.selectbox("Heatmap source", choices, index=default_index, format_func=_heatmap_label)
    heatmap_path = heatmap_variants.get(heatmap_choice) if heatmap_choice else explanation.get("heatmap_path")
    saliency_info = explanation.get("saliency") or {}
    if saliency_info:
        st.caption(
            f"Occlusion-Raster {saliency_info.get('grid')}×{saliency_info.get('grid')} auf dem Mittelquadrat (so sieht CLIP das Bild). "
            f"Größter Score-Verlust beim Verdecken: {float(saliency_info.get('max_support', 0.0)):.2f} Punkte, größter Gewinn: {float(saliency_info.get('max_detraction', 0.0)):.2f}."
        )
    if heatmap_path and Path(heatmap_path).exists():
        heatmap = np.load(heatmap_path)
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        signed_map = str(heatmap_choice or "").startswith("occlusion:")
        overlay = overlay_signed_heatmap(image, heatmap, alpha=overlay_alpha + 0.3) if signed_map else overlay_heatmap_on_image(image, heatmap, alpha=overlay_alpha)
        col_original, col_overlay = st.columns(2)
        col_original.image(str(resolve_path(image_row["relative_path"])), caption="Original", use_container_width=True)
        if show_heatmap_only:
            col_overlay.image(signed_heatmap_to_rgb(heatmap) if signed_map else heatmap_to_rgb(heatmap), caption="Heatmap only", use_container_width=True)
        else:
            col_overlay.image(overlay, caption=f"{_heatmap_label(str(heatmap_choice or explanation.get('heatmap_source', '')))}", use_container_width=True)
    elif explanation.get("overlay_path") and Path(explanation["overlay_path"]).exists():
        st.image(explanation["overlay_path"], caption="Heatmap overlay", use_container_width=True)
    if explanation.get("top_regions"):
        st.write("Most activated regions")
        st.dataframe(pd.DataFrame(explanation["top_regions"]), use_container_width=True, hide_index=True)

with tab_signals:
    if explanation.get("top_features"):
        top_features = pd.DataFrame(explanation["top_features"])
        feature_choice = st.selectbox("Feature diagnostic map (diagnostic only)", top_features["feature"].tolist())
        diagnostic = feature_diagnostic_map(image, feature_choice)
        diagnostic_overlay = overlay_heatmap_on_image(image, diagnostic, alpha=overlay_alpha)
        col_feature, col_feature_overlay = st.columns(2)
        col_feature.image(heatmap_to_rgb(diagnostic), caption=f"{feature_choice} map", use_container_width=True)
        col_feature_overlay.image(diagnostic_overlay, caption=f"{feature_choice} overlay", use_container_width=True)
        st.dataframe(top_features, use_container_width=True, hide_index=True)
    if explanation.get("top_concepts"):
        concepts = pd.DataFrame(explanation["top_concepts"])
        st.write("Concept signals")
        st.dataframe(concepts, use_container_width=True, hide_index=True)
    concept_runs = [run for run in list_runs() if run.startswith("concept_")]
    if concept_runs:
        from cloud_aesthetics.models.concept import explain_image, load_concept_run

        concept_run = st.selectbox("Konzeptmodell (lesbare Beiträge)", concept_runs, key="inspect_concept_run")
        concept_data = load_concept_run(concept_run)
        detail = explain_image(concept_data["contributions"], str(image_id), limit=10)
        if detail.empty:
            st.caption("Kein Konzeptbeitrag für dieses Bild (nicht in der Konzepttabelle oder alle Beiträge 0).")
        else:
            contribution_row = concept_data["contributions"]
            contribution_row = contribution_row[contribution_row["image_id"].astype(str) == str(image_id)].iloc[0]
            st.write(f"Konzeptmodell: {float(contribution_row['prediction']):.2f} = Mittelwert {float(contribution_row['intercept']):.2f} + Beiträge")
            show = detail[["label", "contribution"]].copy()
            show["contribution"] = show["contribution"].map(lambda value: f"{value:+.2f}")
            show.columns = ["Konzept", "Beitrag"]
            st.dataframe(show, use_container_width=True, hide_index=True)
    if explanation.get("scientific_cloud_type", {}).get("cloud_type_scores"):
        st.write("Cloud-type scores")
        st.dataframe(pd.DataFrame(explanation["scientific_cloud_type"]["cloud_type_scores"]), use_container_width=True, hide_index=True)

with tab_neighbors:
    embedding_neighbors = explanation.get("embedding_neighbors") or []
    if embedding_neighbors:
        st.write("Most similar labeled training images (embedding space)")
        neighbor_frame = pd.DataFrame(embedding_neighbors)
        neighbor_frame["image_id"] = neighbor_frame["image_id"].astype(str)
        neighbor_view = neighbor_frame.merge(manifest[["image_id", "relative_path", "kind"]], on="image_id", how="left")
        render_image_grid(
            neighbor_view,
            columns=3,
            caption=lambda row: (
                f"human {float(row['mean_score']):.1f} | sim {float(row['similarity']):.2f} | {row.get('kind', '')}\n\n{row['image_id']}"
                if pd.notna(row.get("mean_score"))
                else f"sim {float(row['similarity']):.2f} | {row.get('kind', '')}\n\n{row['image_id']}"
            ),
        )
    if explanation.get("nearest_neighbors"):
        st.write("Nearest neighbors in hand-crafted feature space")
        st.dataframe(pd.DataFrame(explanation["nearest_neighbors"]), use_container_width=True, hide_index=True)
    elif not embedding_neighbors:
        st.info("No neighbor table available.")

with tab_debug:
    st.info(explanation["scientific_caveat"])
    st.json(explanation)
