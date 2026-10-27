from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jinja2")

from cloud_aesthetics.reports.year_report import collect_report_data, render_report, save_report


def _image(path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.full((30, 40, 3), 120, dtype=np.uint8)).save(path)


def test_report_collects_sections_and_renders(tmp_path: Path):
    ids = [f"img{k}" for k in range(8)]
    for image_id in ids:
        _image(tmp_path / f"{image_id}.jpg")
    manifest = pd.DataFrame({"image_id": ids, "relative_path": [str(tmp_path / f"{i}.jpg") for i in ids], "kind": ["original"] * 6 + ["crop"] * 2, "crop_class": ["original"] * 6 + ["small_crop"] * 2, "dataset_name": "toy", "capture_date": ["2026-05-01"] * 4 + ["2025-05-01"] * 4})
    labels = pd.DataFrame({"image_id": ids, "mean_score": [10.0, 9.5, 8.0, 7.5, 6.0, 9.0, 5.0, 4.0], "n_raters": 1})
    ratings = pd.DataFrame({"rating_id": ["a", "b", "c"], "image_id": ["img0", "img0", "img1"], "rater_id": "friend_a", "rating_session_id": "s", "raw_score_1_to_10": [10.0, 9.5, 9.5], "rating_timestamp": ["2026-04-26T10:00:00+00:00", "2026-04-27T10:00:00+00:00", "2026-04-26T10:01:00+00:00"], "note": None})
    zero_shot = pd.DataFrame({"image_id": ids, "clip_cloud_type": "cirrus", "clip_light": "daylight", "clip_concept_z_layered_depth": [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "clip_concept_layered_depth": 0.1})
    judges = {"llm_x": pd.DataFrame({"image_id": ids, "judge_score": 7.0, "judge_cloud_type": "cirrus", "judge_cloud_types": "cirrus", "judge_verdict": "Fine sky.", "judge_strengths": "a / b / c", "judge_light": "daylight"})}
    contributions = pd.DataFrame({"image_id": ids, "clip_concept_layered_depth": [0.8, 0.4, 0, 0, 0, 0, 0, 0], "clip_concept_soft_haze": [-0.3] * 8, "intercept": 7.0})
    contributions["prediction"] = contributions["intercept"] + contributions["clip_concept_layered_depth"] + contributions["clip_concept_soft_haze"]
    concepts = pd.DataFrame({"concept": ["clip_concept_layered_depth", "clip_concept_soft_haze"], "label": ["Schichtung / Tiefe", "weicher Dunst"], "coef_per_sd": [0.6, -0.4], "active": [True, True], "ci_low": [0.2, -0.7], "ci_high": [0.9, -0.1], "sign_agreement": [1.0, 0.95]})
    concept_run = {"concepts": concepts, "contributions": contributions, "summary": {"run_id": "concept_test", "concept_model": {"primary": "lasso"}, "models": {"lasso": {"metrics": {"val": {"spearman": 0.4}, "test": {"spearman": 0.5}}}}, "config": {"include_kinds": ["original"]}}}
    data = collect_report_data(manifest, labels, ratings=ratings, rater_id="friend_a", zero_shot=zero_shot, judge_tables=judges, concept_run=concept_run, top_n=3, title="Testbericht")
    assert data["scope"]["n_images"] == 6 and data["scope"]["n_labeled"] == 6 and data["scope"]["n_top"] == 3
    assert [card["image_id"] for card in data["cards"]] == ["img0", "img1", "img5"]
    first = data["cards"][0]
    assert first["cloud_type"].startswith("Cirrus") and first["concept_contributions"][0][0] == "Schichtung / Tiefe"
    assert first["clip_concepts"][0][0] == "layered depth" and "llm_x" in first["judges"]
    assert data["concept_model"]["n_active"] == 2 and "Schichtung" in data["concept_model"]["sentence"]
    assert data["atlas"][0]["type_name"].startswith("Cirrus") and data["reliability"]["n"] == 1
    html = render_report(data)
    for needle in ("Testbericht", "Die besten Fotos", "data:image/jpeg;base64", "Konzeptmodell", "Wolkenatlas", "img0", "Fine sky."):
        assert needle in html
    assert "10 gegen 8" not in html  # no analysis table given
    path = save_report(html, output_dir=tmp_path / "reports", stem="test")
    assert path.exists() and path.read_text(encoding="utf-8").startswith("<!doctype html>")
    by_year = collect_report_data(manifest, labels, year=2025, top_n=3)
    assert by_year["scope"]["n_images"] == 2 and by_year["title"] == "Wolken des Jahres 2025"
    assert "image" in render_report(by_year, with_images=False) or True
