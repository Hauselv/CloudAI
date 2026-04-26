from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cloud_aesthetics.eval.metrics import compute_ranking_metrics


@dataclass(slots=True)
class PairwiseDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def build_pairwise_feature_dataset(
    features: pd.DataFrame,
    pairwise_table: pd.DataFrame,
    image_ids: set[str] | None = None,
) -> PairwiseDataset:
    feature_indexed = features.set_index("image_id")
    numeric_columns = [column for column in features.columns if column != "image_id"]
    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_meta: list[dict[str, object]] = []
    for _, row in pairwise_table.iterrows():
        left_id = row["left_image_id"]
        right_id = row["right_image_id"]
        if image_ids is not None and (left_id not in image_ids or right_id not in image_ids):
            continue
        if left_id not in feature_indexed.index or right_id not in feature_indexed.index:
            continue
        if bool(row.get("tie_flag", False)):
            continue
        left = feature_indexed.loc[left_id, numeric_columns].to_numpy(dtype=np.float32)
        right = feature_indexed.loc[right_id, numeric_columns].to_numpy(dtype=np.float32)
        winner = row["winner"]
        y = 1 if winner == left_id else 0
        rows_x.append(left - right)
        rows_y.append(y)
        rows_meta.append({"left_image_id": left_id, "right_image_id": right_id, "winner": winner})
    if not rows_x:
        return PairwiseDataset(
            X=np.zeros((0, max(len(numeric_columns), 1)), dtype=np.float32),
            y=np.zeros((0,), dtype=np.int64),
            metadata=pd.DataFrame(rows_meta),
        )
    return PairwiseDataset(X=np.vstack(rows_x), y=np.asarray(rows_y), metadata=pd.DataFrame(rows_meta))


def train_pairwise_feature_model(
    features: pd.DataFrame,
    pairwise_train: pd.DataFrame,
    pairwise_eval: pd.DataFrame | None = None,
    image_ids: set[str] | None = None,
) -> dict[str, object]:
    train_ds = build_pairwise_feature_dataset(features, pairwise_train, image_ids=image_ids)
    if len(train_ds.y) == 0:
        return {"available": False, "reason": "No pairwise examples available"}
    model = LogisticRegression(max_iter=1000)
    model.fit(train_ds.X, train_ds.y)
    result: dict[str, object] = {"available": True, "model": model, "n_examples": int(len(train_ds.y))}
    if pairwise_eval is not None and not pairwise_eval.empty:
        eval_ds = build_pairwise_feature_dataset(features, pairwise_eval, image_ids=image_ids)
        if len(eval_ds.y):
            probs = model.predict_proba(eval_ds.X)[:, 1]
            preds = (probs >= 0.5).astype(int)
            result["metrics"] = compute_ranking_metrics(eval_ds.y, preds, probs)
    return result
