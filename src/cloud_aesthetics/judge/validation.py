"""Compare a model rater with the human labels.

The first question for any LLM judge is simply: how close is it to the people whose
taste we want to understand? This module answers it per image group and lists the
largest disagreements so they can be inspected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from cloud_aesthetics.eval.metrics import quadratic_weighted_kappa


def _agreement_block(human: np.ndarray, model: np.ndarray) -> dict[str, float]:
    human = np.asarray(human, dtype=np.float32)
    model = np.asarray(model, dtype=np.float32)
    n = int(len(human))
    if n < 3:
        return {"n": n}
    diff = model - human
    return {
        "n": n,
        "spearman": float(spearmanr(human, model).statistic),
        "pearson": float(pearsonr(human, model).statistic),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "bias": float(diff.mean()),
        "within_1": float((np.abs(diff) <= 1.0).mean()),
        "within_2": float((np.abs(diff) <= 2.0).mean()),
        "qwk": float(quadratic_weighted_kappa(human, model)),
        "human_mean": float(human.mean()),
        "model_mean": float(model.mean()),
        "human_std": float(human.std(ddof=0)),
        "model_std": float(model.std(ddof=0)),
    }


def compare_raters(
    ratings: pd.DataFrame,
    *,
    model_rater: str,
    human_raters: list[str] | None,
    manifest: pd.DataFrame | None = None,
    top_k: int = 12,
) -> dict[str, object]:
    """Per-image human mean (over ``human_raters``) vs the model rater's score."""
    frame = ratings.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["rater_id"] = frame["rater_id"].astype(str)
    frame["raw_score_1_to_10"] = frame["raw_score_1_to_10"].astype(float)
    model_scores = frame[frame["rater_id"] == model_rater].groupby("image_id")["raw_score_1_to_10"].mean().rename("model_score")
    human_frame = frame[frame["rater_id"] != model_rater]
    if human_raters:
        human_frame = human_frame[human_frame["rater_id"].isin(set(human_raters))]
    human_scores = human_frame.groupby("image_id")["raw_score_1_to_10"].agg(["mean", "count"]).rename(columns={"mean": "human_score", "count": "human_n"})
    merged = human_scores.join(model_scores, how="inner").reset_index()
    if manifest is not None and not manifest.empty:
        groups = manifest[["image_id"] + [column for column in ("kind", "crop_class", "dataset_name", "relative_path") if column in manifest.columns]].copy()
        groups["image_id"] = groups["image_id"].astype(str)
        merged = merged.merge(groups.drop_duplicates("image_id"), on="image_id", how="left")
    merged["difference"] = merged["model_score"] - merged["human_score"]
    result: dict[str, object] = {
        "model_rater": model_rater,
        "human_raters": sorted(human_frame["rater_id"].unique().tolist()),
        "overall": _agreement_block(merged["human_score"].to_numpy(), merged["model_score"].to_numpy()),
        "by_group": {},
        "score_bands": {},
        "largest_disagreements": [],
        "pairs": merged,
    }
    for column in ("crop_class", "kind", "dataset_name"):
        if column not in merged.columns:
            continue
        blocks = {}
        for group, part in merged.groupby(merged[column].fillna("unknown").astype(str)):
            if len(part) >= 5:
                blocks[group] = _agreement_block(part["human_score"].to_numpy(), part["model_score"].to_numpy())
        if blocks:
            result["by_group"][column] = blocks
    # How well does the model separate the top of the human scale? (the 10-vs-8 question)
    top = merged[merged["human_score"] >= 7.5]
    if len(top) >= 5:
        result["score_bands"]["human_ge_7_5"] = _agreement_block(top["human_score"].to_numpy(), top["model_score"].to_numpy())
    excellent = merged["human_score"] >= 9.0
    if excellent.any() and (~excellent).any():
        result["score_bands"]["model_mean_for_human_9plus"] = float(merged.loc[excellent, "model_score"].mean())
        result["score_bands"]["model_mean_for_human_below_9"] = float(merged.loc[~excellent, "model_score"].mean())
    ordered = merged.reindex(merged["difference"].abs().sort_values(ascending=False).index)
    columns = [column for column in ("image_id", "human_score", "model_score", "difference", "crop_class", "dataset_name", "relative_path") if column in ordered.columns]
    result["largest_disagreements"] = ordered[columns].head(top_k).to_dict("records")
    return result


def report_markdown(result: dict[str, object]) -> str:
    overall = result["overall"]
    lines = [
        f"# Rater comparison: {result['model_rater']} vs {', '.join(result['human_raters']) or 'humans'}",
        "",
        f"Images compared: {overall.get('n', 0)}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in ("spearman", "pearson", "mae", "rmse", "bias", "within_1", "within_2", "qwk", "human_mean", "model_mean", "human_std", "model_std"):
        if key in overall:
            lines.append(f"| {key} | {overall[key]:.3f} |")
    for column, blocks in result.get("by_group", {}).items():
        lines += ["", f"## By {column}", "", "| group | n | spearman | mae | bias |", "|---|---|---|---|---|"]
        for group, block in blocks.items():
            lines.append(f"| {group} | {block['n']} | {block.get('spearman', float('nan')):.3f} | {block.get('mae', float('nan')):.2f} | {block.get('bias', float('nan')):+.2f} |")
    bands = result.get("score_bands", {})
    if bands:
        lines += ["", "## Top of the scale", ""]
        if "human_ge_7_5" in bands:
            block = bands["human_ge_7_5"]
            lines.append(f"- Among images the humans rated >= 7.5 (n={block['n']}): spearman {block.get('spearman', float('nan')):.3f}, mae {block.get('mae', float('nan')):.2f}")
        if "model_mean_for_human_9plus" in bands:
            lines.append(f"- Model mean for human >= 9: {bands['model_mean_for_human_9plus']:.2f}; for human < 9: {bands['model_mean_for_human_below_9']:.2f}")
    if result.get("largest_disagreements"):
        lines += ["", "## Largest disagreements", "", "| image | human | model | diff | class |", "|---|---|---|---|---|"]
        for row in result["largest_disagreements"]:
            lines.append(f"| {row['image_id']} | {row['human_score']:.1f} | {row['model_score']:.1f} | {row['difference']:+.1f} | {row.get('crop_class', '')} |")
    return "\n".join(lines) + "\n"


def save_report(result: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    import json

    from cloud_aesthetics.settings import ensure_parent, resolve_path

    directory = resolve_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result['model_rater']}_vs_human"
    pairs: pd.DataFrame = result["pairs"]
    pairs_path = directory / f"{stem}_pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)
    payload = {key: value for key, value in result.items() if key != "pairs"}
    json_path = ensure_parent(directory / f"{stem}.json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path = directory / f"{stem}.md"
    md_path.write_text(report_markdown(result), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "pairs": str(pairs_path)}
