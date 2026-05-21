from __future__ import annotations

import pandas as pd


def build_text_explanation(
    predicted_score: float,
    uncertainty: float,
    feature_contributions: pd.DataFrame,
    concept_scores: pd.DataFrame,
    scientific_context: dict[str, object] | None = None,
) -> str:
    top_features = feature_contributions.head(4)["feature"].tolist() if not feature_contributions.empty else []
    top_concepts = concept_scores.head(3)["concept"].tolist() if not concept_scores.empty else []
    feature_text = ", ".join(name.replace("_", " ") for name in top_features) or "mixed visual evidence"
    concept_text = ", ".join(name.replace("concept_", "").replace("_", " ") for name in top_concepts) or "no strong concept signals"
    scientific_text = ""
    if scientific_context:
        cloud_type = str(scientific_context.get("primary_cloud_type", "unknown"))
        confidence = float(scientific_context.get("cloud_type_confidence", 0.0))
        evidence = str(scientific_context.get("dominant_evidence", "visual cloud cues"))
        scientific_text = f" Scientific layer estimates {cloud_type} clouds with {confidence:.2f} confidence from {evidence}."
    return (
        f"Predicted score {predicted_score:.2f} +/- {uncertainty:.2f}. "
        f"The model associates this rating with {feature_text}. "
        f"Top concept signals: {concept_text}.{scientific_text} "
        "These attributions describe model behavior rather than proven causal drivers."
    )
