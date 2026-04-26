from __future__ import annotations

import pandas as pd


def build_text_explanation(
    predicted_score: float,
    uncertainty: float,
    feature_contributions: pd.DataFrame,
    concept_scores: pd.DataFrame,
) -> str:
    top_features = feature_contributions.head(4)["feature"].tolist() if not feature_contributions.empty else []
    top_concepts = concept_scores.head(3)["concept"].tolist() if not concept_scores.empty else []
    feature_text = ", ".join(name.replace("_", " ") for name in top_features) or "mixed visual evidence"
    concept_text = ", ".join(name.replace("concept_", "").replace("_", " ") for name in top_concepts) or "no strong concept signals"
    return (
        f"Predicted score {predicted_score:.2f} +/- {uncertainty:.2f}. "
        f"The model associates this rating with {feature_text}. "
        f"Top concept signals: {concept_text}. "
        "These attributions describe model behavior rather than proven causal drivers."
    )
