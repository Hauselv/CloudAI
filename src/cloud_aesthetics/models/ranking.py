"""Pairwise ranking models.

``build_pairwise_feature_dataset`` / ``train_pairwise_feature_model`` are the original
feature-difference logistic regression used inside the baseline suite.
``train_ranking_suite`` is the standalone ranking run: it trains on pairwise outcomes
(explicit duels plus pseudo-pairs from scalar labels) with the dev/test partition from the
split table, evaluates pairwise accuracy on test pairs, and scores every image so the
induced ranking can be compared with the human scores and the Bradley-Terry strengths.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cloud_aesthetics.eval.metrics import compute_ranking_metrics
from cloud_aesthetics.models.base import create_run_context, save_run_json
from cloud_aesthetics.models.tabular import GROUP_COLUMNS, numeric_input_columns


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
    numeric_columns = [
        column
        for column in features.columns
        if column != "image_id" and pd.api.types.is_numeric_dtype(features[column])
    ]
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


def _pair_matrix(table: pd.DataFrame, pairs: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Symmetric difference matrix (each pair appears as left-right and right-left)."""
    indexed = table.set_index("image_id")[columns].astype(np.float32)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    meta: list[dict[str, object]] = []
    for _, row in pairs.iterrows():
        left, right = str(row["left_image_id"]), str(row["right_image_id"])
        if left not in indexed.index or right not in indexed.index or bool(row.get("tie_flag", False)):
            continue
        winner = str(row.get("winner"))
        if winner not in {left, right}:
            continue
        diff = indexed.loc[left].to_numpy() - indexed.loc[right].to_numpy()
        label = 1 if winner == left else 0
        xs.extend([diff, -diff])
        ys.extend([label, 1 - label])
        meta.append({"left_image_id": left, "right_image_id": right, "winner": winner, "source": row.get("source", "explicit")})
    if not xs:
        return np.zeros((0, len(columns)), dtype=np.float32), np.zeros((0,), dtype=np.int64), pd.DataFrame(meta)
    return np.vstack(xs), np.asarray(ys), pd.DataFrame(meta)


def train_ranking_suite(
    table: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    pairs: pd.DataFrame,
    config: dict[str, object],
    *,
    extra_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Logistic regression on input differences; dev pairs train, test pairs evaluate."""
    run = create_run_context(config["output_dir"], str(config["run_name"]), config)
    inputs = table.drop(columns=[column for column in GROUP_COLUMNS if column in table.columns]).copy()
    inputs["image_id"] = inputs["image_id"].astype(str)
    columns = numeric_input_columns(inputs)
    split = splits.copy()
    split["image_id"] = split["image_id"].astype(str)
    dev_ids = set(split.loc[split["partition"] == "dev", "image_id"])
    test_ids = set(split.loc[split["partition"] == "test", "image_id"])
    pairs = pairs.copy()
    pairs["left_image_id"] = pairs["left_image_id"].astype(str)
    pairs["right_image_id"] = pairs["right_image_id"].astype(str)
    sources = [str(source) for source in config.get("pair_sources", ["explicit", "pseudo_from_scores"])]
    if "source" in pairs.columns:
        pairs = pairs[pairs["source"].fillna("explicit").astype(str).isin(sources)]
    dev_pairs = pairs[pairs["left_image_id"].isin(dev_ids) & pairs["right_image_id"].isin(dev_ids)]
    test_pairs = pairs[pairs["left_image_id"].isin(test_ids) & pairs["right_image_id"].isin(test_ids)]
    max_pairs = int(config.get("max_train_pairs", 20000) or 0)
    if max_pairs and len(dev_pairs) > max_pairs:
        dev_pairs = dev_pairs.sample(max_pairs, random_state=int(config.get("seed", 42)))
    X_train, y_train, train_meta = _pair_matrix(inputs, dev_pairs, columns)
    if len(y_train) == 0:
        raise ValueError("No training pairs with inputs on both sides. Collect duels or enable pseudo pairs.")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(C=float(config.get("C", 0.5)), max_iter=2000, fit_intercept=False)),
        ]
    )
    model.fit(X_train, y_train)
    summary: dict[str, object] = {
        "run_id": run.run_id,
        "kind": "ranking",
        "input_columns": columns,
        "train_pairs": int(len(train_meta)),
        "test_pairs": 0,
        "pair_sources": sources,
        "models": {},
    }
    if extra_summary:
        summary.update(extra_summary)
    metrics_test: dict[str, float] = {}
    X_test, y_test, test_meta = _pair_matrix(inputs, test_pairs, columns)
    if len(y_test):
        probs = model.predict_proba(X_test)[:, 1]
        metrics_test = compute_ranking_metrics(y_test, (probs >= 0.5).astype(int), probs)
        summary["test_pairs"] = int(len(test_meta))
    # Per-image score: the model's linear score of (x - 0), i.e. the latent strength.
    matrix = inputs[columns]
    latent = model.decision_function(np.asarray(matrix.astype(np.float32)))
    scores = pd.DataFrame({"image_id": inputs["image_id"], "rank_score": latent})
    label_frame = labels.copy()
    label_frame["image_id"] = label_frame["image_id"].astype(str)
    scores = scores.merge(label_frame[["image_id"] + [column for column in ("mean_score", "bt_score", "duels") if column in label_frame.columns]], on="image_id", how="left")
    scores = scores.merge(split[["image_id", "partition"]], on="image_id", how="left")
    agreement: dict[str, dict[str, float]] = {}
    for partition in ("dev", "test"):
        part = scores[scores["partition"] == partition].dropna(subset=["mean_score"])
        block: dict[str, float] = {"n": int(len(part))}
        if len(part) >= 3:
            block["spearman_vs_mean_score"] = float(spearmanr(part["rank_score"], part["mean_score"]).statistic)
            if "bt_score" in part.columns and part["bt_score"].notna().sum() >= 3:
                bt = part.dropna(subset=["bt_score"])
                block["spearman_vs_bt_score"] = float(spearmanr(bt["rank_score"], bt["bt_score"]).statistic)
                block["n_bt"] = int(len(bt))
        agreement[partition] = block
    model_path = run.run_dir / "ranking_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    scores_path = run.run_dir / "rank_scores.parquet"
    scores.to_parquet(scores_path, index=False)
    summary["models"]["pairwise_logreg"] = {
        "metrics": {"val": {"spearman": agreement.get("dev", {}).get("spearman_vs_mean_score", float("nan"))}, "test": metrics_test},
        "agreement": agreement,
        "model_path": str(model_path),
        "rank_scores_path": str(scores_path),
    }
    save_run_json(summary, run.run_dir / "summary.json")
    return summary
