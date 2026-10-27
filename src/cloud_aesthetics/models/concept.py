"""Concept bottleneck model: the rating as a sparse linear function of *named* concepts.

Instead of 512 anonymous embedding dimensions, the inputs are things a person can read:
CLIP zero-shot concepts ("golden rim light"), CLIP light phase, judge flags, palette
statistics and the scene fraction. A Lasso keeps only the concepts that carry signal, so
the fitted model reads like a sentence: "+0.8 for layered depth, -0.6 for landscape".

The model is deliberately less accurate than CLIP-Ridge; its job is to be legible. The
grouped CV loop is shared with every other tabular model (``train_tabular_suite``), on top
of which this module adds coefficient tables, bootstrap stability, per-image contributions
and a comparison of concept vocabularies.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cloud_aesthetics.eval.contrast import judge_feature_table
from cloud_aesthetics.models.base import save_run_json
from cloud_aesthetics.models.tabular import GROUP_COLUMNS, merge_training_frame, train_tabular_suite
from cloud_aesthetics.settings import resolve_path
from cloud_aesthetics.utils.io import read_table

LASSO_ALPHAS = np.logspace(-2.5, 0.3, 24)

# Which columns of the wide per-image table belong to which concept family. Families are
# chosen by name in the model config so vocabularies can be compared ("CLIP only", ...).
FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "clip_concept": ("clip_concept_",),
    "clip_light": ("clip_light_",),
    "clip_type": ("clip_type_",),
    "judge_flag": ("judge_flag_", "judge_n_flags", "judge_is_sunset_share"),
    "judge_type": ("judge_type_",),
    "judge_light": ("judge_light_",),
    "judge_score": ("judge_score_mean", "judge_score_spread", "judge_confidence_mean"),
    "palette": ("palette_", "colorfulness"),
    "scene": ("crop_scale", "sky_area_fraction", "cloud_area_fraction", "dominant_cloud_area_fraction"),
    "sun": ("sun_phase_", "sun_elevation_deg", "hours_from_sunset"),
    "weather": ("wx_cloud_cover", "wx_relative_humidity_2m", "wx_precipitation"),
}
DEFAULT_FAMILIES = ("clip_concept", "clip_light", "judge_flag", "judge_light", "palette", "scene")
FAMILY_LABELS = {
    "clip_concept": "CLIP-Konzept",
    "clip_light": "CLIP-Licht",
    "clip_type": "CLIP-Wolkentyp",
    "judge_flag": "Judge-Flag",
    "judge_type": "Judge-Wolkentyp",
    "judge_light": "Judge-Licht",
    "judge_score": "Judge-Score",
    "palette": "Farbpalette",
    "scene": "Szene / Anteil",
    "sun": "Sonnenstand",
    "weather": "Wetter",
}
CONCEPT_LABELS = {
    "clip_concept_dramatic_storm_sky": "dramatischer Sturmhimmel",
    "clip_concept_golden_rim_light": "goldenes Kantenlicht",
    "clip_concept_crepuscular_rays": "Strahlenbüschel",
    "clip_concept_pastel_sunset": "Pastell-Sonnenuntergang",
    "clip_concept_layered_depth": "Schichtung / Tiefe",
    "clip_concept_towering_structure": "aufgetürmte Struktur",
    "clip_concept_fine_texture": "feine Textur",
    "clip_concept_mammatus": "Mammatus",
    "clip_concept_lenticular": "Lenticularis",
    "clip_concept_rainbow": "Regenbogen",
    "clip_concept_high_contrast": "hoher Kontrast",
    "clip_concept_soft_haze": "weicher Dunst",
    "clip_concept_landscape_dominant": "Landschaft dominiert",
    "clip_concept_airplane_window": "Flugzeugfenster",
    "clip_concept_boring_sky": "langweiliger Himmel",
    "clip_concept_spectacular_sky": "spektakulärer Himmel",
    "clip_light_daylight": "Tageslicht (CLIP)",
    "clip_light_golden_hour": "goldene Stunde (CLIP)",
    "clip_light_twilight": "Dämmerung (CLIP)",
    "clip_light_blue_hour": "blaue Stunde (CLIP)",
    "clip_light_night": "Nacht (CLIP)",
    "judge_n_flags": "Anzahl Judge-Flags",
    "judge_is_sunset_share": "Sonnenuntergang (Judge)",
    "judge_score_mean": "Judge-Score (Mittel)",
    "judge_score_spread": "Judge-Uneinigkeit",
    "judge_confidence_mean": "Judge-Konfidenz",
    "palette_warm_share": "warmer Farbanteil",
    "palette_blue_share": "blauer Farbanteil",
    "palette_neutral_share": "neutraler Farbanteil",
    "palette_hue_concentration": "Farbton-Konzentration",
    "palette_contrast": "Palettenkontrast",
    "palette_chroma_mean": "Buntheit (Mittel)",
    "palette_chroma_max": "Buntheit (Max)",
    "palette_lightness_mean": "Helligkeit der Palette",
    "palette_lightness_range": "Helligkeitsspanne",
    "palette_dominant_saturation": "Sättigung der Hauptfarbe",
    "palette_dominant_value": "Helligkeit der Hauptfarbe",
    "palette_dominant_share": "Anteil der Hauptfarbe",
    "colorfulness": "Farbigkeit",
    "crop_scale": "Szenenanteil (1 = ganzes Foto)",
    "sky_area_fraction": "Himmelsanteil",
    "cloud_area_fraction": "Wolkenanteil",
    "dominant_cloud_area_fraction": "größte Wolkenmasse",
    "sun_elevation_deg": "Sonnenhöhe",
    "hours_from_sunset": "Stunden bis Sonnenuntergang",
}


def concept_label(column: str) -> str:
    if column in CONCEPT_LABELS:
        return CONCEPT_LABELS[column]
    for prefix, text in (("judge_flag_", "Flag: "), ("judge_type_", "Judge-Typ: "), ("judge_light_", "Judge-Licht: "), ("clip_type_", "CLIP-Typ: "), ("sun_phase_", "Sonnenphase: "), ("wx_", "Wetter: ")):
        if column.startswith(prefix):
            return text + column[len(prefix) :].replace("_", " ")
    return column.replace("_", " ")


def concept_family(column: str) -> str | None:
    if column.startswith("clip_concept_z_"):
        return None  # z-scored duplicates of the raw concept similarities
    for family, prefixes in FAMILY_PREFIXES.items():
        if any(column == prefix or column.startswith(prefix) for prefix in prefixes):
            return family
    return None


def select_concept_columns(table: pd.DataFrame, families: tuple[str, ...] | list[str]) -> list[str]:
    wanted = set(families)
    return [column for column in table.columns if column != "image_id" and pd.api.types.is_numeric_dtype(table[column]) and concept_family(column) in wanted]


def build_concept_table(
    manifest: pd.DataFrame,
    *,
    features: pd.DataFrame | None = None,
    zero_shot: pd.DataFrame | None = None,
    judge_tables: dict[str, pd.DataFrame] | None = None,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per manifest image with every candidate concept column (labeled or not)."""
    base = manifest.copy()
    base["image_id"] = base["image_id"].astype(str)
    keep = ["image_id"] + [column for column in ("crop_scale",) + GROUP_COLUMNS if column in base.columns]
    table = base[keep].drop_duplicates("image_id")
    if "crop_scale" in table.columns:
        table["crop_scale"] = table["crop_scale"].astype(float).fillna(1.0)
    for part in (features, zero_shot, context):
        if part is None or part.empty:
            continue
        part = part.copy()
        part["image_id"] = part["image_id"].astype(str)
        numeric = [column for column in part.columns if column != "image_id" and column not in table.columns and pd.api.types.is_numeric_dtype(part[column]) and concept_family(column)]
        if numeric:
            table = table.merge(part[["image_id"] + numeric].drop_duplicates("image_id"), on="image_id", how="left")
    if judge_tables:
        judged = judge_feature_table(judge_tables)
        if not judged.empty:
            judged = judged[["image_id"] + [column for column in judged.columns if column != "image_id" and column not in table.columns]]
            table = table.merge(judged, on="image_id", how="left")
    return table


def load_concept_table(
    manifest: pd.DataFrame,
    *,
    features_path: str | Path = "data/processed/features/features_v1.parquet",
    zero_shot_path: str | Path = "data/processed/embeddings/clip_zero_shot.parquet",
    judge_root: str | Path = "data/processed/judge",
) -> pd.DataFrame:
    from cloud_aesthetics.features.context import load_context

    judge_tables: dict[str, pd.DataFrame] = {}
    root = resolve_path(judge_root)
    if root.exists():
        for path in sorted(root.glob("*_attributes.parquet")):
            judge_tables[path.stem.replace("_attributes", "")] = pd.read_parquet(path)
    return build_concept_table(manifest, features=read_table(features_path), zero_shot=read_table(zero_shot_path), judge_tables=judge_tables, context=load_context())


def _make_estimator(name: str, config: dict[str, object]) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    inner_cv = int(config.get("inner_cv", 5))
    if name == "lasso":
        model = LassoCV(alphas=LASSO_ALPHAS, cv=inner_cv, max_iter=20000, random_state=int(config.get("seed", 42)))
    elif name == "elasticnet":
        model = ElasticNetCV(alphas=LASSO_ALPHAS, l1_ratio=[0.5, 0.8, 0.95], cv=inner_cv, max_iter=20000, random_state=int(config.get("seed", 42)))
    elif name == "ridge":
        model = RidgeCV(alphas=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0))
    else:
        raise ValueError(f"Unknown concept regressor {name!r}")
    return Pipeline(steps + [("model", model)])


def _frozen_copy(pipeline: Pipeline) -> Pipeline:
    """Same pipeline with the CV-chosen regularisation frozen, for cheap bootstrap refits."""
    model = pipeline.named_steps["model"]
    if isinstance(model, LassoCV):
        frozen = Lasso(alpha=float(model.alpha_), max_iter=20000)
    elif isinstance(model, ElasticNetCV):
        from sklearn.linear_model import ElasticNet

        frozen = ElasticNet(alpha=float(model.alpha_), l1_ratio=float(model.l1_ratio_), max_iter=20000)
    elif isinstance(model, RidgeCV):
        frozen = Ridge(alpha=float(model.alpha_))
    else:
        frozen = clone(model)
    return Pipeline([("imputer", clone(pipeline.named_steps["imputer"])), ("scaler", clone(pipeline.named_steps["scaler"])), ("model", frozen)])


def coefficient_table(pipeline: Pipeline, columns: list[str], X: pd.DataFrame) -> pd.DataFrame:
    """Coefficients in score points per standard deviation and per raw unit, with the input scale."""
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    coef = np.asarray(model.coef_, dtype=float)
    frame = pd.DataFrame(
        {
            "concept": columns,
            "family": [concept_family(column) or "other" for column in columns],
            "coef_per_sd": coef,
            "coef_per_unit": coef / scale,
            "input_mean": scaler.mean_,
            "input_sd": scale,
            "observed_min": X[columns].min().to_numpy(dtype=float),
            "observed_max": X[columns].max().to_numpy(dtype=float),
        }
    )
    frame["label"] = frame["concept"].map(concept_label)
    frame["family_label"] = frame["family"].map(lambda key: FAMILY_LABELS.get(key, key))
    frame["active"] = frame["coef_per_sd"].abs() > 1e-9
    frame["intercept"] = float(model.intercept_)
    return frame.reindex(frame["coef_per_sd"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def bootstrap_stability(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, *, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Refit the frozen model on bootstrap resamples; report how stable each coefficient is."""
    rng = np.random.default_rng(seed)
    template = _frozen_copy(pipeline)
    coefs = np.zeros((n_boot, X.shape[1]), dtype=float)
    n = len(X)
    for index in range(n_boot):
        rows = rng.integers(0, n, size=n)
        fitted = clone(template).fit(X.iloc[rows], y[rows])
        coefs[index] = fitted.named_steps["model"].coef_
    point = np.asarray(pipeline.named_steps["model"].coef_, dtype=float)
    sign = np.sign(point)
    with np.errstate(invalid="ignore"):
        agreement = np.where(sign != 0, (np.sign(coefs) == sign).mean(axis=0), (coefs == 0).mean(axis=0))
    return pd.DataFrame(
        {
            "concept": list(X.columns),
            "boot_mean": coefs.mean(axis=0),
            "boot_sd": coefs.std(axis=0),
            "ci_low": np.quantile(coefs, 0.05, axis=0),
            "ci_high": np.quantile(coefs, 0.95, axis=0),
            "selection_rate": (np.abs(coefs) > 1e-9).mean(axis=0),
            "sign_agreement": agreement,
        }
    )


def contribution_table(pipeline: Pipeline, X: pd.DataFrame, image_ids: pd.Series) -> pd.DataFrame:
    """Per-image additive contributions (coef * standardised value) for every concept."""
    imputed = pipeline.named_steps["imputer"].transform(X)
    z = pipeline.named_steps["scaler"].transform(imputed)
    coef = np.asarray(pipeline.named_steps["model"].coef_, dtype=float)
    contributions = pd.DataFrame(z * coef, columns=list(X.columns))
    contributions.insert(0, "image_id", image_ids.astype(str).to_numpy())
    contributions["intercept"] = float(pipeline.named_steps["model"].intercept_)
    contributions["prediction"] = contributions.drop(columns=["image_id"]).sum(axis=1)
    return contributions


def explain_image(contributions: pd.DataFrame, image_id: str, *, limit: int | None = None) -> pd.DataFrame:
    """Long table of the non-zero contributions of one image, largest magnitude first."""
    row = contributions[contributions["image_id"].astype(str) == str(image_id)]
    if row.empty:
        return pd.DataFrame(columns=["concept", "label", "family", "contribution"])
    row = row.iloc[0]
    items = [(column, float(row[column])) for column in contributions.columns if column not in {"image_id", "intercept", "prediction"} and abs(float(row[column])) > 1e-9]
    frame = pd.DataFrame(items, columns=["concept", "contribution"])
    frame["label"] = frame["concept"].map(concept_label)
    frame["family"] = frame["concept"].map(lambda column: concept_family(column) or "other")
    frame = frame.reindex(frame["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return frame.head(limit) if limit else frame


def _grouped_oof(template: Pipeline, frame: pd.DataFrame, columns: list[str], target: str) -> np.ndarray:
    oof = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].dropna().unique()):
        val = (frame["fold"] == fold).to_numpy()
        if val.sum() == 0 or (~val).sum() < 5:
            continue
        fitted = clone(template).fit(frame.loc[~val, columns], frame.loc[~val, target].to_numpy(dtype=float))
        oof[val] = fitted.predict(frame.loc[val, columns])
    return oof


def vocabulary_comparison(
    table: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
    variants: dict[str, list[str]],
) -> pd.DataFrame:
    """Out-of-fold quality of the same model with different concept vocabularies.

    Answers "which words explain the rating best": CLIP concepts alone, judge attributes
    alone, palette alone, everything together.
    """
    target = str(config.get("target_column", "mean_score"))
    regressor = str(config.get("regressors", ["lasso"])[0])
    rows = []
    for name, families in variants.items():
        columns = select_concept_columns(table, families)
        if not columns:
            continue
        merged, _ = merge_training_frame(table[["image_id"] + columns], labels, splits, target)
        dev = merged[merged["partition"] == "dev"].reset_index(drop=True)
        if len(dev) < 10:
            continue
        oof = _grouped_oof(_make_estimator(regressor, config), dev, columns, target)
        ok = ~np.isnan(oof)
        y = dev[target].to_numpy(dtype=float)
        row = {
            "variant": name,
            "families": ", ".join(families),
            "n_concepts": len(columns),
            "n_dev": int(ok.sum()),
            "oof_spearman": float(spearmanr(y[ok], oof[ok]).correlation) if ok.sum() > 2 else np.nan,
            "oof_mae": float(np.mean(np.abs(y[ok] - oof[ok]))) if ok.any() else np.nan,
        }
        if "kind" in dev.columns:
            originals = ok & (dev["kind"].astype(str) == "original").to_numpy()
            row["n_originals"] = int(originals.sum())
            row["oof_spearman_originals"] = float(spearmanr(y[originals], oof[originals]).correlation) if originals.sum() > 5 else np.nan
        full = clone(_make_estimator(regressor, config)).fit(dev[columns], y)
        row["n_active"] = int((np.abs(full.named_steps["model"].coef_) > 1e-9).sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("oof_spearman", ascending=False).reset_index(drop=True) if not frame.empty else frame


def train_concept_suite(
    table: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    """Grouped CV like every tabular model, plus coefficient, stability and contribution artifacts."""
    families = [str(item) for item in config.get("concept_families", DEFAULT_FAMILIES)]
    columns = select_concept_columns(table, families)
    if not columns:
        raise ValueError(f"No concept columns found for families {families}. Run zero-shot / judge / extract-features first.")
    inputs = table[["image_id"] + columns]
    regressors = [str(name) for name in config.get("regressors", ["lasso"])]
    estimators = {name: _make_estimator(name, config) for name in regressors}
    extra = {"concept_families": families, "concept_columns": columns}
    summary, run = train_tabular_suite(inputs, labels, splits, config, estimators, kind="concept", extra_summary=extra)

    target = str(config.get("target_column", "mean_score"))
    merged, _ = merge_training_frame(inputs, labels, splits, target)
    dev = merged[merged["partition"] == "dev"].reset_index(drop=True)
    y_dev = dev[target].to_numpy(dtype=float)
    primary = regressors[0]
    with Path(summary["models"][primary]["model_path"]).open("rb") as handle:
        pipeline = pickle.load(handle)
    coefficients = coefficient_table(pipeline, columns, dev)
    n_boot = int(config.get("bootstrap_samples", 200))
    if n_boot > 0:
        stability = bootstrap_stability(pipeline, dev[columns], y_dev, n_boot=n_boot, seed=int(config.get("seed", 42)))
        coefficients = coefficients.merge(stability, on="concept", how="left")
    coefficients.to_parquet(run.run_dir / "concepts.parquet", index=False)
    contributions = contribution_table(pipeline, inputs[columns], inputs["image_id"])
    contributions.to_parquet(run.run_dir / "contributions.parquet", index=False)

    variants = config.get("variants") or {}
    comparison = vocabulary_comparison(table, labels, splits, config, {str(k): [str(f) for f in v] for k, v in variants.items()}) if variants else pd.DataFrame()
    if not comparison.empty:
        comparison.to_parquet(run.run_dir / "vocabulary.parquet", index=False)

    summary["concept_model"] = {
        "primary": primary,
        "n_concepts": len(columns),
        "n_active": int(coefficients["active"].sum()),
        "intercept": float(pipeline.named_steps["model"].intercept_),
        "alpha": summary["models"][primary].get("fitted_params", {}).get("alpha"),
        "bootstrap_samples": n_boot,
        "top_positive": coefficients[coefficients["coef_per_sd"] > 0].head(5)[["concept", "coef_per_sd"]].to_dict("records"),
        "top_negative": coefficients[coefficients["coef_per_sd"] < 0].head(5)[["concept", "coef_per_sd"]].to_dict("records"),
        "concepts_path": str(run.run_dir / "concepts.parquet"),
        "contributions_path": str(run.run_dir / "contributions.parquet"),
        "vocabulary_path": str(run.run_dir / "vocabulary.parquet") if not comparison.empty else None,
    }
    save_run_json(summary, run.run_dir / "summary.json")
    return summary


def load_concept_run(run_id: str, artifacts_root: str | Path = "data/artifacts") -> dict[str, object]:
    run_dir = resolve_path(artifacts_root) / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    out: dict[str, object] = {"summary": summary, "run_dir": run_dir}
    for key in ("concepts", "contributions", "vocabulary", "predictions"):
        path = run_dir / f"{key}.parquet"
        out[key] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return out


def sentence(coefficients: pd.DataFrame, *, limit: int = 6) -> str:
    """The model as one readable German sentence: '+0,8 Schichtung, −0,6 Landschaft, ...' (per SD)."""
    active = coefficients[coefficients["active"]].head(limit)
    if active.empty:
        return "Das Modell hat kein Konzept behalten (nur Mittelwert)."
    parts = [f"{'+' if row.coef_per_sd >= 0 else '−'}{abs(row.coef_per_sd):.2f} {row.label}" for row in active.itertuples()]
    return "Score ≈ Mittelwert " + ", ".join(parts) + " (je Standardabweichung)"
