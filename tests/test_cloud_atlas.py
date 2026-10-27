from __future__ import annotations

import numpy as np
import pandas as pd

from cloud_aesthetics.eval.cloud_atlas import CLOUD_TYPES, TYPE_ORDER, atlas_summary, atlas_table, consensus_cloud_types, reliability_report, type_by_light


def _sources():
    zero_shot = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "clip_cloud_type": ["cumulus", "cirrus", "stratus", "cumulus"], "clip_light": ["daylight", "golden_hour", "daylight", "golden_hour"]})
    judge_a = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "judge_cloud_type": ["cumulus", "cirrus", "altostratus", "stratocumulus"], "judge_cloud_types": ["cumulus|stratocumulus", "cirrus", "altostratus|cirrus", "stratocumulus|cumulus"]})
    judge_b = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "judge_cloud_type": ["cumulus", "cirrostratus", "stratocumulus", "altocumulus"], "judge_cloud_types": ["cumulus", "cirrostratus|cirrus", "stratocumulus", "altocumulus"]})
    return zero_shot, {"j1": judge_a, "j2": judge_b}


def test_consensus_votes_and_reliability_tiers():
    zero_shot, judges = _sources()
    consensus = consensus_cloud_types(zero_shot, judges).set_index("image_id")
    assert consensus.loc["a", "cloud_type"] == "cumulus" and consensus.loc["a", "reliability"] == "sicher"
    assert consensus.loc["b", "cloud_type"] == "cirrus" and consensus.loc["b", "reliability"] == "wahrscheinlich"  # 2 primaries + 0.5
    assert consensus.loc["c", "reliability"] == "unsicher" and consensus.loc["c", "cloud_type"] == "stratus"  # three different primaries, tie goes to CLIP's choice
    assert consensus.loc["d", "cloud_type"] == "cumulus"  # CLIP primary + judge secondary = 1.5 beats others at 1.0
    assert consensus.loc["d", "reliability"] == "wahrscheinlich"
    report = reliability_report(consensus.reset_index())
    assert report["n"] == 4 and report["sicher"] == 1 and report["mean_sources"] == 3.0


def test_atlas_summary_and_light_grid():
    zero_shot, judges = _sources()
    consensus = consensus_cloud_types(zero_shot, judges)
    manifest = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "relative_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], "kind": ["original", "original", "crop", "original"]})
    labels = pd.DataFrame({"image_id": ["a", "b", "d"], "mean_score": [9.5, 7.0, 8.0]})
    predictions = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "pred_mean": [9.0, 6.5, 4.0, 8.2]})
    table = atlas_table(consensus, manifest, labels, zero_shot, predictions)
    assert len(table) == 4 and table.loc[table["image_id"] == "a", "type_name"].iloc[0] == CLOUD_TYPES["cumulus"].name
    summary = atlas_summary(table, min_reliability="wahrscheinlich", examples=2).set_index("cloud_type")
    assert "stratus" not in summary.index  # c is unsicher
    assert summary.loc["cumulus", "n"] == 2 and summary.loc["cumulus", "mean_score"] == 8.75 and summary.loc["cumulus", "n_top"] == 1
    assert summary.loc["cumulus", "example_ids"] == ["a", "d"]
    assert summary.loc["cirrus", "top_light"] == "golden_hour"
    loose = atlas_summary(table, min_reliability="unsicher")
    assert "stratus" in set(loose["cloud_type"])
    grid = type_by_light(table, min_n=1)
    assert grid.loc[CLOUD_TYPES["cumulus"].name, "daylight"] == 9.5 and np.isnan(grid.loc[CLOUD_TYPES["cirrus"].name, "daylight"])
    assert all(key in CLOUD_TYPES for key in TYPE_ORDER)
