"""Counterfactual explorer: how does the predicted score react to controlled image edits?

A set of simple, interpretable operators (contrast, saturation, exposure, warmth, zoom,
blur) is applied to a photo; the edited photo is embedded with CLIP and scored by the
trained regressor. Sweeps over each operator give sensitivity curves, and a greedy search
proposes the single edit that raises the score most. Aggregated over a score band this
tells what the model thinks separates an 8 from a 10. These are statements about the
model; whether a human would agree is exactly what the labels are for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class EditSpec:
    key: str
    label: str
    neutral: float
    low: float
    high: float
    step: float
    unit: str = ""


EDITS: dict[str, EditSpec] = {
    "contrast": EditSpec("contrast", "Kontrast", 1.0, 0.5, 1.6, 0.05, "×"),
    "saturation": EditSpec("saturation", "Sättigung", 1.0, 0.0, 2.0, 0.1, "×"),
    "exposure": EditSpec("exposure", "Belichtung", 0.0, -1.0, 1.0, 0.1, "EV"),
    "warmth": EditSpec("warmth", "Wärme", 0.0, -1.0, 1.0, 0.1, ""),
    "zoom": EditSpec("zoom", "Zoom (Ausschnitt)", 1.0, 1.0, 2.0, 0.1, "×"),
    "blur": EditSpec("blur", "Unschärfe", 0.0, 0.0, 6.0, 0.5, "px"),
}


def apply_edit(image_rgb: np.ndarray, key: str, value: float) -> np.ndarray:
    """Apply one operator. Values at the operator's neutral point return the image unchanged."""
    spec = EDITS[key]
    if abs(value - spec.neutral) < 1e-9:
        return image_rgb
    img = image_rgb.astype(np.float32)
    if key == "contrast":
        mean = img.mean(axis=(0, 1), keepdims=True)
        out = (img - mean) * value + mean
    elif key == "saturation":
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    elif key == "exposure":
        out = img * (2.0**value)
    elif key == "warmth":
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + 22.0 * value, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + 8.0 * value, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    elif key == "zoom":
        height, width = image_rgb.shape[:2]
        crop_h, crop_w = int(round(height / value)), int(round(width / value))
        y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
        crop = image_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    elif key == "blur":
        return cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=float(value))
    else:
        raise KeyError(key)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_edits(image_rgb: np.ndarray, params: dict[str, float]) -> np.ndarray:
    out = image_rgb
    for key in ("zoom", "exposure", "contrast", "saturation", "warmth", "blur"):
        if key in params:
            out = apply_edit(out, key, float(params[key]))
    return out


def resize_for_speed(image_rgb: np.ndarray, max_edge: int = 512) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = max_edge / max(height, width)
    if scale >= 1.0:
        return image_rgb
    return cv2.resize(image_rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


class ClipScorer:
    """Scores RGB arrays with a CLIP backend plus a regressor trained on its embeddings."""

    def __init__(self, backend: Any, model: Any, input_columns: list[str], batch_size: int = 16) -> None:
        self.backend = backend
        self.model = model
        self.input_columns = list(input_columns)
        self.batch_size = batch_size

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if torch is None:
            raise ImportError("torch is required for counterfactual scoring")
        from PIL import Image

        chunks = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = torch.stack([self.backend.preprocess(Image.fromarray(img)) for img in images[start : start + self.batch_size]])
                emb = self.backend.model.encode_image(batch).float()
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                chunks.append(emb.cpu().numpy())
        return np.vstack(chunks) if chunks else np.zeros((0, len(self.input_columns)), dtype=np.float32)

    def score(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros(0)
        embeddings = self.embed(images)
        return np.asarray(self.model.predict(pd.DataFrame(embeddings, columns=self.input_columns)), dtype=float)


def sweep_values(spec: EditSpec, points: int = 7) -> list[float]:
    values = np.linspace(spec.low, spec.high, points).tolist()
    if not any(abs(v - spec.neutral) < 1e-9 for v in values):
        values.append(spec.neutral)
    return sorted(round(float(v), 4) for v in values)


def sensitivity_sweep(scorer: ClipScorer, image_rgb: np.ndarray, *, points: int = 7, keys: list[str] | None = None) -> pd.DataFrame:
    """Score of the image under each operator value; one row per (edit, value)."""
    base = resize_for_speed(image_rgb)
    rows: list[dict[str, object]] = []
    images: list[np.ndarray] = []
    for key in keys or list(EDITS):
        for value in sweep_values(EDITS[key], points):
            images.append(apply_edit(base, key, value))
            rows.append({"edit": key, "label": EDITS[key].label, "value": value, "is_neutral": abs(value - EDITS[key].neutral) < 1e-9})
    scores = scorer.score(images)
    frame = pd.DataFrame(rows)
    frame["score"] = scores
    neutral = frame[frame["is_neutral"]].groupby("edit")["score"].first()
    frame["delta"] = frame["score"] - frame["edit"].map(neutral)
    return frame


def best_single_edits(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per operator: the value with the largest gain and the largest loss."""
    rows = []
    for key, part in sweep.groupby("edit"):
        gain = part.loc[part["delta"].idxmax()]
        loss = part.loc[part["delta"].idxmin()]
        rows.append(
            {
                "edit": key,
                "label": EDITS[key].label,
                "best_value": float(gain["value"]),
                "best_delta": float(gain["delta"]),
                "worst_value": float(loss["value"]),
                "worst_delta": float(loss["delta"]),
                "range": float(part["score"].max() - part["score"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("best_delta", ascending=False).reset_index(drop=True)


@dataclass
class BandSweepResult:
    per_image: pd.DataFrame
    summary: pd.DataFrame
    n_images: int
    notes: dict[str, Any] = field(default_factory=dict)


def band_sweep(scorer: ClipScorer, images: list[tuple[str, np.ndarray]], *, points: int = 5, keys: list[str] | None = None) -> BandSweepResult:
    """Sweep many images and aggregate: mean delta per (edit, value) and how often each edit helps."""
    parts = []
    for image_id, image in images:
        sweep = sensitivity_sweep(scorer, image, points=points, keys=keys)
        sweep.insert(0, "image_id", image_id)
        parts.append(sweep)
    if not parts:
        return BandSweepResult(pd.DataFrame(), pd.DataFrame(), 0)
    per_image = pd.concat(parts, ignore_index=True)
    non_neutral = per_image[~per_image["is_neutral"]]
    summary = (
        non_neutral.groupby(["edit", "label", "value"])
        .agg(mean_delta=("delta", "mean"), median_delta=("delta", "median"), share_positive=("delta", lambda s: float((s > 0.05).mean())), n=("delta", "size"))
        .reset_index()
    )
    best = per_image.loc[per_image.groupby(["image_id", "edit"])["delta"].idxmax()]
    best_summary = best.groupby(["edit", "label"]).agg(mean_best_delta=("delta", "mean"), mean_best_value=("value", "mean")).reset_index()
    return BandSweepResult(per_image, summary, len(images), notes={"best_per_edit": best_summary.to_dict("records")})


def save_band_sweep(result: BandSweepResult, output_dir: str | Path, stem: str) -> dict[str, str]:
    from cloud_aesthetics.settings import resolve_path

    directory = resolve_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    per_image = directory / f"{stem}_per_image.parquet"
    summary = directory / f"{stem}_summary.csv"
    result.per_image.to_parquet(per_image, index=False)
    result.summary.to_csv(summary, index=False)
    (directory / f"{stem}_notes.json").write_text(json.dumps({"n_images": result.n_images, **result.notes}, indent=2, default=float), encoding="utf-8")
    return {"per_image": str(per_image), "summary": str(summary)}


def load_scorer(run_id: str | Path, model_name: str | None = None) -> tuple[ClipScorer, dict[str, Any], str]:
    """Build a scorer from an embedding run on a single CLIP model. Returns (scorer, summary, model_name)."""
    from cloud_aesthetics.embeddings.backends import create_backend
    from cloud_aesthetics.embeddings.store import model_specs
    from cloud_aesthetics.explain.feature_importance import load_pickled_model
    from cloud_aesthetics.settings import resolve_path

    run_dir = resolve_path(run_id)
    if not run_dir.exists():
        run_dir = resolve_path("data/artifacts") / str(run_id)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    keys = [str(k) for k in summary.get("embedding_models", [])]
    if summary.get("kind") != "embedding" or len(keys) != 1 or summary.get("include_features") or summary.get("include_zero_shot"):
        raise ValueError("Counterfactuals need an embedding run on exactly one CLIP model without extra inputs (e.g. embedding_clip.yaml).")
    specs = model_specs()
    if keys[0] not in specs or specs[keys[0]].backend != "open_clip":
        raise ValueError(f"{keys[0]} is not an open_clip model")
    backend = create_backend(specs[keys[0]])
    models = summary.get("models", {})
    chosen = model_name or ("ridge" if "ridge" in models else next(iter(models)))
    model = load_pickled_model(models[chosen]["model_path"])
    return ClipScorer(backend, model, list(summary["input_columns"])), summary, chosen
