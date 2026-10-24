"""The "10 vs 8" analysis: what separates excellent photos from merely good ones.

Assembles one wide per-image table from every signal source the project has
(hand-crafted features, CLIP zero-shot attributes, judge attributes) and compares two
human score bands with effect sizes, rank tests and false-discovery control. Pure
pandas/scipy; the Streamlit page and the CLI only render the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

from cloud_aesthetics.settings import resolve_path
from cloud_aesthetics.utils.io import read_table

FAMILY_LABELS = {
    "color": "Farbe (handgebaut)",
    "texture": "Textur (handgebaut)",
    "composition": "Komposition (handgebaut)",
    "concept": "Konzept (handgebaut)",
    "meteo": "Meteo-Heuristik",
    "rule_type": "Wolkentyp (Regeln)",
    "clip_concept": "CLIP-Konzept",
    "clip_type": "CLIP-Wolkentyp",
    "clip_light": "CLIP-Licht",
    "judge": "Judge-Attribut",
    "palette": "Farbpalette",
    "sun": "Sonnenstand / Zeit",
    "weather": "Wetter (Reanalyse)",
}

_COLOR = {"mean_brightness", "contrast", "dynamic_range", "saturation_mean", "lab_a_mean", "lab_b_mean", "blue_orange_balance", "color_temperature_proxy", "cloud_area_fraction", "sky_area_fraction"}
_TEXTURE = {"edge_density", "lbp_mean", "lbp_std", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "entropy", "blob_count", "blob_area_mean", "connected_components", "dog_mean_abs", "fractal_dimension_proxy"}


def feature_family(column: str) -> str:
    if column.startswith("clip_concept_z_"):
        return "skip"
    if column.startswith("palette_") or column == "colorfulness":
        return "palette"
    if column.startswith("sun_") or column in {"hours_from_sunset", "month", "hour_local_estimate"}:
        return "sun"
    if column.startswith("wx_"):
        return "weather"
    if column.startswith("clip_concept_"):
        return "clip_concept"
    if column.startswith("clip_type_"):
        return "clip_type"
    if column.startswith("clip_light_"):
        return "clip_light"
    if column.startswith("judge_"):
        return "judge"
    if column.startswith("concept_"):
        return "concept"
    if column.startswith("meteo_"):
        return "meteo"
    if column.startswith("cloud_type_score_") or column == "cloud_type_confidence":
        return "rule_type"
    if column in _COLOR or column.startswith("hue_bin_"):
        return "color"
    if column in _TEXTURE or column.startswith("edge_orientation_bin_"):
        return "texture"
    if column in {"clip_cloud_type_confidence", "clip_cloud_type_probability"}:
        return "clip_type"
    return "composition"


def _numeric(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [column for column in frame.columns if column not in exclude and pd.api.types.is_numeric_dtype(frame[column])]


def judge_feature_table(attribute_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Numeric judge features per image: mean score/confidence, one-hot majority type and light, flags, sunset."""
    frames = []
    for rater, table in attribute_tables.items():
        if table is None or table.empty:
            continue
        part = table.copy()
        part["image_id"] = part["image_id"].astype(str)
        frames.append(part.drop_duplicates("image_id"))
    if not frames:
        return pd.DataFrame(columns=["image_id"])
    stacked = pd.concat(frames, ignore_index=True)
    grouped = stacked.groupby("image_id")
    out = pd.DataFrame(
        {
            "judge_score_mean": grouped["judge_score"].mean(),
            "judge_score_spread": grouped["judge_score"].max() - grouped["judge_score"].min(),
            "judge_confidence_mean": grouped["judge_confidence"].mean(),
            "judge_is_sunset_share": grouped["judge_is_sunset"].mean().astype(float),
            "judge_n_flags": grouped["judge_flags"].apply(lambda values: float(np.mean([len([f for f in str(v).split("|") if f]) for v in values]))),
        }
    )
    for name, column in (("judge_type", "judge_cloud_type"), ("judge_light", "judge_light")):
        shares = pd.crosstab(stacked["image_id"], stacked[column].fillna("unknown").astype(str), normalize="index")
        shares.columns = [f"{name}_{value}" for value in shares.columns]
        out = out.join(shares, how="left")
    flags = stacked[["image_id", "judge_flags"]].copy()
    flags["judge_flags"] = flags["judge_flags"].fillna("").astype(str)
    exploded = flags.assign(flag=flags["judge_flags"].str.split("|")).explode("flag")
    exploded = exploded[exploded["flag"].fillna("") != ""].reset_index(drop=True)
    if not exploded.empty:
        flag_shares = pd.crosstab(exploded["image_id"], exploded["flag"]).reindex(out.index, fill_value=0)
        n_raters = stacked.groupby("image_id").size().reindex(out.index).replace(0, 1)
        flag_shares = flag_shares.div(n_raters, axis=0)
        flag_shares.columns = [f"judge_flag_{value}" for value in flag_shares.columns]
        out = out.join(flag_shares, how="left")
    return out.fillna(0.0).reset_index()


def assemble_analysis_table(
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame | None = None,
    zero_shot: pd.DataFrame | None = None,
    judge_tables: dict[str, pd.DataFrame] | None = None,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per labeled image with the human score, grouping columns and every numeric signal."""
    base = labels.copy()
    base["image_id"] = base["image_id"].astype(str)
    keep = ["image_id", "mean_score", "std_score", "n_raters"]
    base = base[[column for column in keep if column in base.columns]]
    groups = manifest.copy()
    groups["image_id"] = groups["image_id"].astype(str)
    group_columns = [column for column in ("relative_path", "kind", "crop_class", "dataset_name") if column in groups.columns]
    table = base.merge(groups[["image_id"] + group_columns].drop_duplicates("image_id"), on="image_id", how="left")
    if features is not None and not features.empty:
        part = features.copy()
        part["image_id"] = part["image_id"].astype(str)
        numeric = _numeric(part, {"image_id"})
        table = table.merge(part[["image_id"] + numeric].drop_duplicates("image_id"), on="image_id", how="left")
    if zero_shot is not None and not zero_shot.empty:
        part = zero_shot.copy()
        part["image_id"] = part["image_id"].astype(str)
        numeric = [column for column in _numeric(part, {"image_id"}) if not column.startswith("clip_concept_z_")]
        table = table.merge(part[["image_id"] + numeric].drop_duplicates("image_id"), on="image_id", how="left")
    if judge_tables:
        judged = judge_feature_table(judge_tables)
        if not judged.empty:
            table = table.merge(judged, on="image_id", how="left")
    if context is not None and not context.empty:
        part = context.copy()
        part["image_id"] = part["image_id"].astype(str)
        numeric = [column for column in _numeric(part, {"image_id", "latitude", "longitude", "weather_available"}) if column != "wx_hour_offset"]
        table = table.merge(part[["image_id"] + numeric].drop_duplicates("image_id"), on="image_id", how="left")
        for column in ("sun_phase", "time_quality"):
            if column in part.columns:
                table = table.merge(part[["image_id", column]].drop_duplicates("image_id"), on="image_id", how="left")
    return table


@dataclass(frozen=True)
class Band:
    low: float
    high: float
    label: str

    def mask(self, scores: pd.Series) -> pd.Series:
        return (scores >= self.low) & (scores <= self.high)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def contrast_bands(
    table: pd.DataFrame,
    band_a: Band,
    band_b: Band,
    *,
    min_group: int = 5,
    exclude_columns: tuple[str, ...] = ("mean_score", "std_score", "n_raters"),
) -> dict[str, object]:
    """Compare every numeric column between two human score bands."""
    scores = table["mean_score"].astype(float)
    part_a = table[band_a.mask(scores)]
    part_b = table[band_b.mask(scores)]
    result: dict[str, object] = {
        "band_a": band_a,
        "band_b": band_b,
        "n_a": int(len(part_a)),
        "n_b": int(len(part_b)),
        "ids_a": part_a["image_id"].astype(str).tolist(),
        "ids_b": part_b["image_id"].astype(str).tolist(),
        "table": pd.DataFrame(),
    }
    if len(part_a) < min_group or len(part_b) < min_group:
        return result
    rows = []
    for column in _numeric(table, set(exclude_columns) | {"image_id"}):
        family = feature_family(column)
        if family == "skip":
            continue
        a = part_a[column].astype(float).dropna().to_numpy()
        b = part_b[column].astype(float).dropna().to_numpy()
        if len(a) < min_group or len(b) < min_group:
            continue
        if np.nanstd(np.concatenate([a, b])) == 0:
            continue
        try:
            u_stat, p_mw = mannwhitneyu(a, b, alternative="two-sided")
            rank_biserial = float(1.0 - 2.0 * u_stat / (len(a) * len(b)))
        except ValueError:
            p_mw, rank_biserial = 1.0, 0.0
        try:
            _, p_t = ttest_ind(a, b, equal_var=False)
        except ValueError:
            p_t = 1.0
        rows.append(
            {
                "feature": column,
                "family": family,
                "family_label": FAMILY_LABELS.get(family, family),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
                "cohens_d": cohens_d(a, b),
                "rank_biserial": rank_biserial,
                "p_mannwhitney": float(p_mw) if p_mw == p_mw else 1.0,
                "p_welch": float(p_t) if p_t == p_t else 1.0,
                "n_a": int(len(a)),
                "n_b": int(len(b)),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q_mannwhitney"] = benjamini_hochberg(frame["p_mannwhitney"].to_numpy())
        frame["direction"] = np.where(frame["cohens_d"] > 0, f"höher bei {band_a.label}", f"höher bei {band_b.label}")
        frame = frame.reindex(frame["cohens_d"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    result["table"] = frame
    return result


def top_end_trends(table: pd.DataFrame, min_score: float = 7.0, min_n: int = 10) -> pd.DataFrame:
    """Spearman correlation of every signal with the human score among good photos only."""
    subset = table[table["mean_score"].astype(float) >= min_score]
    rows = []
    if len(subset) < min_n:
        return pd.DataFrame(columns=["feature", "family", "family_label", "spearman", "p_value", "n"])
    for column in _numeric(subset, {"image_id", "mean_score", "std_score", "n_raters"}):
        family = feature_family(column)
        if family == "skip":
            continue
        values = subset[column].astype(float)
        mask = values.notna()
        if mask.sum() < min_n or values[mask].nunique() < 3:
            continue
        rho, p = spearmanr(values[mask], subset.loc[mask, "mean_score"].astype(float))
        rows.append({"feature": column, "family": family, "family_label": FAMILY_LABELS.get(family, family), "spearman": float(rho), "p_value": float(p), "n": int(mask.sum())})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["q_value"] = benjamini_hochberg(frame["p_value"].to_numpy())
    return frame.reindex(frame["spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def load_analysis_table(
    dataset_config: dict[str, object],
    *,
    features_path: str | Path = "data/processed/features/features_v1.parquet",
    zero_shot_path: str | Path = "data/processed/embeddings/clip_zero_shot.parquet",
    judge_root: str | Path = "data/processed/judge",
) -> pd.DataFrame:
    """Convenience loader used by the CLI and the page."""
    from cloud_aesthetics.data.manifest import ensure_kind_columns

    labels = read_table(dataset_config["aggregated_labels_path"])
    manifest = ensure_kind_columns(read_table(dataset_config["manifest_path"]))
    features = read_table(features_path)
    zero_shot = read_table(zero_shot_path)
    judge_tables: dict[str, pd.DataFrame] = {}
    root = resolve_path(judge_root)
    if root.exists():
        for path in sorted(root.glob("*_attributes.parquet")):
            judge_tables[path.stem.replace("_attributes", "")] = pd.read_parquet(path)
    from cloud_aesthetics.features.context import load_context

    return assemble_analysis_table(labels, manifest, features, zero_shot, judge_tables, context=load_context())
