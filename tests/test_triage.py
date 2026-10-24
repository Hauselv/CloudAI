from __future__ import annotations

import pandas as pd

from cloud_aesthetics.judge.triage import build_triage, triage_summary


def _attributes(rater: str, scores: dict[str, float], flags: dict[str, str] | None = None, types: dict[str, str] | None = None) -> pd.DataFrame:
    flags = flags or {}
    types = types or {}
    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "rater_id": rater,
                "judge_score": score,
                "judge_flags": flags.get(image_id, ""),
                "judge_cloud_type": types.get(image_id, "cumulus"),
                "judge_light": "daylight",
                "judge_verdict": f"{rater} says {score}",
            }
            for image_id, score in scores.items()
        ]
    )


def test_build_triage_orders_originals_first_and_proposes_conservatively():
    manifest = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d", "e", "f", "g"],
            "relative_path": [f"{i}.jpg" for i in "abcdefg"],
            "kind": ["original", "original", "crop", "crop", "original", "original", "original"],
            "crop_class": ["original", "original", "large_crop", "small_crop", "original", "original", "original"],
            "dataset_name": ["x"] * 7,
        }
    )
    labels = pd.DataFrame({"image_id": ["e"], "mean_score": [9.0]})
    judge_a = _attributes(
        "ra",
        {"a": 8.0, "b": 4.0, "c": 9.0, "d": 1.0, "e": 7.0, "f": 3.0, "g": 2.0},
        flags={"b": "landscape_dominant", "d": "blurry_or_noisy", "f": "satellite_or_map", "g": "not_a_cloud_photo"},
    )
    judge_b = _attributes(
        "rb",
        {"a": 5.0, "b": 4.5, "c": 8.5, "d": 1.0, "e": 7.5, "f": 3.5, "g": 6.0},
        flags={"b": "landscape_dominant", "d": "blurry_or_noisy|tiny_detail_crop", "f": "satellite_or_map"},
    )
    triage = build_triage({"ra": judge_a, "rb": judge_b}, manifest, labels)
    by_id = triage.set_index("image_id")
    assert by_id.loc["a", "judge_mean"] == 6.5 and by_id.loc["a", "judge_spread"] == 3.0 and by_id.loc["a", "uncertain"]
    # landscape_dominant is a hint, never a proposal (the human rates such photos highly)
    assert by_id.loc["b", "proposed_exclusion"] == "" and by_id.loc["b", "judge_hints"] == "landscape_dominant (2/2)"
    assert by_id.loc["d", "proposed_exclusion"] == "bad_crop"  # blurry by all judges and low mean score
    assert by_id.loc["f", "proposed_exclusion"] == "satellite"  # all judges agree and mean score <= 4
    assert by_id.loc["g", "proposed_exclusion"] == "not_cloud_training_data"  # one vote is enough
    assert by_id.loc["c", "proposed_exclusion"] == ""
    assert by_id.loc["e", "labeled"] and by_id.loc["e", "human_score"] == 9.0
    order = triage["image_id"].tolist()
    assert order[0] == "a"  # unlabeled original without proposal ranks first
    assert order.index("b") < order.index("c")  # originals before crops even with hints
    assert order.index("c") < order.index("d")  # large crop before small crop
    assert by_id.loc["e", "priority_score"] < by_id.loc["d", "priority_score"]  # labeled last
    assert set(by_id.loc["d", "judge_flags"].split("|")) == {"blurry_or_noisy", "tiny_detail_crop"}
    summary = triage_summary(triage)
    assert summary["unlabeled"] == 6 and summary["proposed_exclusions"] == 3 and summary["uncertain_unlabeled"] == 2
    assert summary["unlabeled_originals"] == 4


def test_build_triage_handles_single_judge_and_missing_scores():
    manifest = pd.DataFrame({"image_id": ["a", "b"], "relative_path": ["a.jpg", "b.jpg"], "kind": ["original", "crop"], "crop_class": ["original", "small_crop"], "dataset_name": ["x", "x"]})
    judge = _attributes("ra", {"a": 7.0}, flags={"a": "airplane_window|satellite_or_map"})
    triage = build_triage({"ra": judge}, manifest, None)
    by_id = triage.set_index("image_id")
    assert by_id.loc["a", "proposed_exclusion"] == ""  # satellite needs a low mean score; airplane_window is only a hint
    assert by_id.loc["a", "judge_hints"] == "airplane_window (1/1)"
    assert by_id.loc["b", "judge_votes"] == 0 and pd.isna(by_id.loc["b", "judge_mean"])
    assert by_id.loc["a", "judge_spread"] == 0.0
