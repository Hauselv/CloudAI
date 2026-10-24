"""Structured judgement a vision model returns for one cloud photo.

Plain JSON-friendly types only (the structured-output schema subset is conservative);
ranges and vocabularies are validated in :func:`normalize_judgement` instead.
"""

from __future__ import annotations

from pydantic import BaseModel

CLOUD_TYPE_VOCAB = [
    "cirrus",
    "cirrocumulus",
    "cirrostratus",
    "altocumulus",
    "altostratus",
    "nimbostratus",
    "stratocumulus",
    "stratus",
    "cumulus",
    "cumulonimbus",
    "lenticular",
    "mammatus",
    "contrail",
    "fog",
    "clear_sky",
    "unknown",
]

LIGHT_VOCAB = ["daylight", "golden_hour", "twilight", "blue_hour", "night", "unknown"]

QUALITY_FLAGS = [
    "airplane_window",
    "satellite_or_map",
    "landscape_dominant",
    "people_or_objects_dominant",
    "blurry_or_noisy",
    "over_processed",
    "tiny_detail_crop",
    "not_a_cloud_photo",
]


class CloudJudgement(BaseModel):
    """What the model fills in. Keep field docs short: they become part of the schema."""

    score: float
    confidence: float
    cloud_types: list[str]
    light: str
    is_sunset_or_sunrise: bool
    dominant_colors: list[str]
    strengths: list[str]
    weaknesses: list[str]
    quality_flags: list[str]
    one_sentence_verdict: str


def _clip(value: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if number != number:
        return low
    return max(low, min(high, number))


def normalize_judgement(raw: CloudJudgement | dict[str, object]) -> dict[str, object]:
    """Clamp the score to the 0-10 scale in 0.5 steps and restrict vocabularies."""
    data = raw.model_dump() if isinstance(raw, CloudJudgement) else dict(raw)
    score = round(_clip(data.get("score", 0.0), 0.0, 10.0) * 2.0) / 2.0
    confidence = _clip(data.get("confidence", 0.5), 0.0, 1.0)
    cloud_types = [str(item).strip().lower().replace(" ", "_") for item in data.get("cloud_types", []) or []]
    cloud_types = [item if item in CLOUD_TYPE_VOCAB else "unknown" for item in cloud_types][:3] or ["unknown"]
    light = str(data.get("light", "unknown")).strip().lower().replace(" ", "_")
    flags = [str(item).strip().lower().replace(" ", "_") for item in data.get("quality_flags", []) or []]
    return {
        "score": score,
        "confidence": confidence,
        "cloud_types": cloud_types,
        "primary_cloud_type": cloud_types[0],
        "light": light if light in LIGHT_VOCAB else "unknown",
        "is_sunset_or_sunrise": bool(data.get("is_sunset_or_sunrise", False)),
        "dominant_colors": [str(item).strip().lower() for item in (data.get("dominant_colors", []) or [])][:5],
        "strengths": [str(item).strip() for item in (data.get("strengths", []) or [])][:3],
        "weaknesses": [str(item).strip() for item in (data.get("weaknesses", []) or [])][:3],
        "quality_flags": [flag for flag in flags if flag in QUALITY_FLAGS],
        "one_sentence_verdict": str(data.get("one_sentence_verdict", "")).strip(),
    }
