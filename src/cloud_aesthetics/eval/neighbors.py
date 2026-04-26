from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def nearest_neighbors(feature_table: pd.DataFrame, image_id: str, top_k: int = 5) -> pd.DataFrame:
    indexed = feature_table.set_index("image_id")
    if image_id not in indexed.index:
        return pd.DataFrame(columns=["image_id", "similarity"])
    numeric = indexed.select_dtypes(include=["number"]).fillna(0.0)
    query = numeric.loc[[image_id]]
    sims = cosine_similarity(query, numeric)[0]
    result = pd.DataFrame({"image_id": numeric.index, "similarity": sims})
    return result[result["image_id"] != image_id].sort_values("similarity", ascending=False).head(top_k)
