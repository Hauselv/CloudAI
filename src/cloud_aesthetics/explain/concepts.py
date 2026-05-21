from __future__ import annotations

import pandas as pd

from cloud_aesthetics.features.concept_bootstrap import CONCEPT_COLUMNS
from cloud_aesthetics.features.scientific import CLOUD_TYPE_TAXONOMY


def top_concepts(feature_row: pd.Series, top_k: int = 4) -> pd.DataFrame:
    rows = [{"concept": concept, "score": float(feature_row.get(concept, 0.0))} for concept in CONCEPT_COLUMNS]
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)


def cloud_type_scores(feature_row: pd.Series, top_k: int = 4) -> pd.DataFrame:
    rows = [
        {"cloud_type": cloud_type, "score": float(feature_row.get(f"cloud_type_score_{cloud_type}", 0.0))}
        for cloud_type in CLOUD_TYPE_TAXONOMY
    ]
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)
