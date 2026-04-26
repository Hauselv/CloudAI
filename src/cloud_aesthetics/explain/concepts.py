from __future__ import annotations

import pandas as pd

from cloud_aesthetics.features.concept_bootstrap import CONCEPT_COLUMNS


def top_concepts(feature_row: pd.Series, top_k: int = 4) -> pd.DataFrame:
    rows = [{"concept": concept, "score": float(feature_row.get(concept, 0.0))} for concept in CONCEPT_COLUMNS]
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)
