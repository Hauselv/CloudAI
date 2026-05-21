from __future__ import annotations

import json
from typing import Any

import requests


SCORE_BANDS = [
    (8.0, "excellent", "Excellent candidate", "shortlist"),
    (6.5, "strong", "Strong candidate", "keep"),
    (4.5, "mixed", "Mixed signal", "review"),
    (0.0, "weak", "Weak candidate", "deprioritize"),
]

FEATURE_LABELS = {
    "mean_brightness": "overall brightness",
    "contrast": "contrast",
    "dynamic_range": "dynamic range",
    "saturation_mean": "color saturation",
    "cloud_area_fraction": "cloud coverage",
    "sky_area_fraction": "open sky",
    "edge_density": "edge detail",
    "lbp_mean": "local texture average",
    "lbp_std": "local texture variation",
    "glcm_contrast": "texture contrast",
    "glcm_homogeneity": "texture smoothness",
    "glcm_energy": "texture regularity",
    "glcm_correlation": "texture correlation",
    "entropy": "visual complexity",
    "blob_count": "small cloud blob count",
    "blob_area_mean": "average cloud component size",
    "connected_components": "cloud component count",
    "dog_mean_abs": "fine-scale texture energy",
    "fractal_dimension_proxy": "cloud boundary complexity",
    "dominant_cloud_area_fraction": "dominant cloud mass",
    "rule_of_thirds_distance": "composition balance",
    "saliency_center_x": "horizontal visual focus",
    "saliency_center_y": "vertical visual focus",
}

CONCEPT_LABELS = {
    "concept_towering_clouds": "towering clouds",
    "concept_dramatic_edges": "dramatic edges",
    "concept_rim_lighting": "rim lighting",
    "concept_crepuscular_rays": "crepuscular rays",
    "concept_layered_depth": "layered depth",
    "concept_cumulus_like": "cumulus-like texture",
    "concept_storm_like": "storm-like mood",
    "concept_dramatic_mood": "dramatic mood",
    "concept_strong_depth_cues": "depth cues",
    "concept_cloud_sky_contrast": "cloud-sky contrast",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _clean_label(name: object) -> str:
    raw = str(name)
    if raw in FEATURE_LABELS:
        return FEATURE_LABELS[raw]
    if raw in CONCEPT_LABELS:
        return CONCEPT_LABELS[raw]
    return raw.replace("concept_", "").replace("_", " ")


def _score_band(score: float) -> dict[str, object]:
    for threshold, label, title, action in SCORE_BANDS:
        if score >= threshold:
            return {"label": label, "title": title, "threshold": threshold, "action": action}
    return {"label": "unknown", "title": "Unknown", "threshold": None, "action": "review"}


def _confidence_band(uncertainty: float) -> dict[str, object]:
    if uncertainty <= 0.6:
        return {"label": "tight", "title": "Tight", "risk": "low"}
    if uncertainty <= 1.2:
        return {"label": "usable", "title": "Usable", "risk": "medium"}
    if uncertainty <= 2.0:
        return {"label": "wide", "title": "Wide", "risk": "high"}
    return {"label": "very_wide", "title": "Very wide", "risk": "very_high"}


def _driver_rows(top_features: list[dict[str, object]], limit: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in top_features[:limit]:
        has_delta = "delta_prediction" in item
        effect = _as_float(item.get("delta_prediction") if has_delta else item.get("distance_from_typical"))
        z_score = _as_float(item.get("z_score"))
        direction = "raises" if effect >= 0 else "lowers"
        if not has_delta:
            direction = "higher_than_typical" if z_score >= 0 else "lower_than_typical"
        row = {
            "feature": str(item.get("feature", "")),
            "label": _clean_label(item.get("feature", "")),
            "direction": direction,
            "effect": effect,
            "abs_effect": abs(effect),
            "value": _as_float(item.get("value")),
            "attribution_kind": str(item.get("attribution_kind", "prediction_delta" if has_delta else "feature_distance")),
        }
        if not has_delta:
            row["typical_value"] = _as_float(item.get("typical_value"))
            row["z_score"] = z_score
        rows.append(row)
    return sorted(rows, key=lambda row: row["abs_effect"], reverse=True)


def _concept_rows(top_concepts: list[dict[str, object]], limit: int = 4) -> list[dict[str, object]]:
    return [
        {
            "concept": str(item.get("concept", "")),
            "label": _clean_label(item.get("concept", "")),
            "score": _as_float(item.get("score")),
        }
        for item in top_concepts[:limit]
    ]


def _human_label_context(ground_truth: list[dict[str, object]]) -> dict[str, object]:
    if not ground_truth:
        return {"available": False}
    row = ground_truth[0]
    predicted = _as_float(row.get("mean_score"))
    return {
        "available": True,
        "mean_score": predicted,
        "std_score": _as_float(row.get("std_score")),
        "agreement_index": _as_float(row.get("agreement_index")),
        "n_raters": int(_as_float(row.get("n_raters"), 0.0)),
    }


def build_interpretation(explanation: dict[str, object]) -> dict[str, object]:
    """Convert raw attribution output into a compact decision layer."""
    score = _as_float(explanation.get("predicted_score"))
    uncertainty = _as_float(explanation.get("uncertainty"))
    score_band = _score_band(score)
    confidence_band = _confidence_band(uncertainty)
    drivers = _driver_rows(list(explanation.get("top_features", [])))
    concepts = _concept_rows(list(explanation.get("top_concepts", [])))
    scientific = dict(explanation.get("scientific_cloud_type", {}) or {})
    cloud_type = str(scientific.get("primary_cloud_type", "unknown"))
    cloud_confidence = _as_float(scientific.get("cloud_type_confidence"))
    label_context = _human_label_context(list(explanation.get("ground_truth", [])))
    score_interval = {
        "low": max(0.0, score - uncertainty),
        "high": min(10.0, score + uncertainty),
        "raw_low": score - uncertainty,
        "raw_high": score + uncertainty,
    }
    if label_context["available"]:
        label_context["prediction_error"] = score - _as_float(label_context.get("mean_score"))

    strongest_driver = drivers[0]["label"] if drivers else "mixed visual evidence"
    strongest_concept = concepts[0]["label"] if concepts else "no dominant concept"
    attribution_kind = drivers[0]["attribution_kind"] if drivers else "unknown"
    signal_phrase = "model contribution" if attribution_kind == "prediction_delta" else "unusual feature signal"
    summary = (
        f"{score_band['title']} with {confidence_band['title'].lower()} uncertainty. "
        f"The strongest {signal_phrase} is {strongest_driver}; the leading concept is {strongest_concept}."
    )
    if cloud_type != "unknown":
        summary += f" Scientific cues point to {cloud_type} clouds at {cloud_confidence:.2f} confidence."

    cautions: list[str] = []
    if uncertainty > 1.2:
        cautions.append("Prediction interval is wide; prioritize an additional human rating before trusting the rank.")
    if not label_context["available"]:
        cautions.append("No aggregated human label is available for calibration on this image.")
    if cloud_type == "unknown" or cloud_confidence < 0.25:
        cautions.append("Cloud-type estimate is weak; treat the scientific layer as context only.")
    if not concepts or concepts[0]["score"] < 0.55:
        cautions.append("No concept signal clearly dominates the image.")

    next_actions: list[str] = []
    if uncertainty > 1.2:
        next_actions.append("Add or review labels for this image.")
    if score >= 7.5 and uncertainty <= 1.2:
        next_actions.append("Keep in the high-quality shortlist.")
    elif score <= 4.5 and uncertainty <= 1.2:
        next_actions.append("Deprioritize unless it is needed for negative examples.")
    else:
        next_actions.append("Compare against nearest neighbors before final sorting.")

    return {
        "method": "deterministic_v1",
        "score": score,
        "uncertainty": uncertainty,
        "score_interval": score_interval,
        "score_band": score_band,
        "confidence_band": confidence_band,
        "summary": summary,
        "drivers": drivers,
        "concepts": concepts,
        "scientific": {
            "primary_cloud_type": cloud_type,
            "cloud_type_confidence": cloud_confidence,
            "dominant_evidence": str(scientific.get("dominant_evidence", "")),
        },
        "human_label": label_context,
        "cautions": cautions,
        "next_actions": next_actions,
        "recommendation": str(score_band["action"]),
    }


def _llm_prompt(explanation: dict[str, object], interpretation: dict[str, object]) -> str:
    payload = {
        "prediction": {
            "score": interpretation.get("score"),
            "uncertainty": interpretation.get("uncertainty"),
            "score_band": interpretation.get("score_band"),
            "confidence_band": interpretation.get("confidence_band"),
        },
        "drivers": interpretation.get("drivers", [])[:5],
        "concepts": interpretation.get("concepts", [])[:4],
        "scientific": interpretation.get("scientific", {}),
        "cautions": interpretation.get("cautions", []),
        "next_actions": interpretation.get("next_actions", []),
        "raw_text": explanation.get("text_explanation", ""),
    }
    return (
        "You interpret cloud-photo aesthetic model output for a local research tool. "
        "Use only the JSON evidence. Do not invent weather facts, camera metadata, or causal claims. "
        "Write 4 concise bullets: verdict, why, uncertainty, next action.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def generate_local_llm_interpretation(
    explanation: dict[str, object],
    interpretation: dict[str, object] | None = None,
    *,
    provider: str = "ollama",
    model: str = "llama3.2:3b-instruct",
    url: str = "http://localhost:11434/api/generate",
    timeout_seconds: float = 12.0,
) -> dict[str, object]:
    """Ask an optional local LLM for wording while keeping deterministic output authoritative."""
    if provider in {"", "off", "none", "disabled"}:
        return {"available": False, "provider": provider, "text": "", "error": "Local LLM provider is disabled."}
    if provider != "ollama":
        return {"available": False, "provider": provider, "text": "", "error": f"Unsupported provider: {provider}"}
    if interpretation is None:
        interpretation = build_interpretation(explanation)
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "prompt": _llm_prompt(explanation, interpretation),
                "stream": False,
                "options": {"temperature": 0.15, "num_predict": 220},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {"available": False, "provider": provider, "model": model, "text": "", "error": str(exc)}
    return {
        "available": True,
        "provider": provider,
        "model": model,
        "text": str(data.get("response", "")).strip(),
        "error": "",
    }
