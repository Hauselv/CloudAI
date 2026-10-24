from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from cloud_aesthetics.judge.client import ClaudeJudge, encode_image_jpeg, estimate_image_tokens, rater_id_for_model
from cloud_aesthetics.judge.schema import CloudJudgement, normalize_judgement
from cloud_aesthetics.judge.store import attributes_table, cached_result, human_rater_ids, run_judge
from cloud_aesthetics.judge.validation import compare_raters, report_markdown


class FakeMessages:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[dict] = []
        self.fail_next = False

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated API failure")
        # The image is identified by the request order in these tests.
        score = self.scores[len(self.calls) - 1]
        parsed = CloudJudgement(
            score=score,
            confidence=0.8,
            cloud_types=["Cumulus", "weird type"],
            light="golden hour",
            is_sunset_or_sunrise=True,
            dominant_colors=["orange", "blue"],
            strengths=["rim light"],
            weaknesses=["flat horizon"],
            quality_flags=["landscape_dominant", "made_up_flag"],
            one_sentence_verdict="Warm evening sky with structure.",
        )
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1200, output_tokens=150, cache_read_input_tokens=0, cache_creation_input_tokens=0))


class FakeClient:
    def __init__(self, scores: dict[str, float]) -> None:
        self.messages = FakeMessages(scores)


def _manifest(tmp_path: Path, n: int = 4) -> pd.DataFrame:
    rows = []
    for index in range(n):
        path = tmp_path / f"cloud_{index}.jpg"
        image = np.full((300, 400, 3), 120 + index * 20, dtype=np.uint8)
        cv2.imwrite(str(path), image)
        rows.append({"image_id": f"img{index}", "relative_path": str(path), "kind": "original", "crop_class": "original", "dataset_name": "toy"})
    return pd.DataFrame(rows)


def test_normalize_judgement_clamps_and_restricts_vocab():
    judgement = normalize_judgement({"score": 11.3, "confidence": 1.7, "cloud_types": ["Cumulus", "nonsense"], "light": "disco", "quality_flags": ["airplane_window", "x"]})
    assert judgement["score"] == 10.0
    assert judgement["confidence"] == 1.0
    assert judgement["cloud_types"] == ["cumulus", "unknown"]
    assert judgement["primary_cloud_type"] == "cumulus"
    assert judgement["light"] == "unknown"
    assert judgement["quality_flags"] == ["airplane_window"]
    assert normalize_judgement({"score": 7.3})["score"] == 7.5


def test_encode_image_downscales(tmp_path: Path):
    path = tmp_path / "big.jpg"
    cv2.imwrite(str(path), np.zeros((1500, 3000, 3), dtype=np.uint8))
    data, (width, height) = encode_image_jpeg(path, max_edge=600)
    assert width == 600 and height == 300
    assert len(data) > 100
    assert estimate_image_tokens(1024, 768) == 1049
    assert rater_id_for_model("claude-opus-5-5") == "llm_claude_opus_5_5"


def test_run_judge_caches_and_records_ratings(tmp_path: Path):
    manifest = _manifest(tmp_path)
    client = FakeClient({0: 8.0, 1: 3.5, 2: 9.5, 3: 6.0})
    judge = ClaudeJudge(model="claude-opus-5-5", client=client)
    ratings_dir = tmp_path / "ratings"
    root = tmp_path / "judge"
    raters = tmp_path / "raters.csv"
    summary = run_judge(judge, manifest, image_ids=["img0", "img1", "img2"], ratings_dir=ratings_dir, root=root, raters_path=raters)
    assert summary["judged"] == 3 and summary["cached"] == 0 and summary["failed"] == 0
    assert summary["input_tokens"] == 3600
    request = client.messages.calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_format"] is CloudJudgement
    assert request["messages"][0]["content"][0]["type"] == "image"
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    cached = cached_result("llm_claude_opus_5_5", "img1", root)
    assert cached["judgement"]["score"] == 3.5
    assert cached["judgement"]["light"] == "golden_hour"
    ratings = pd.read_csv(ratings_dir / "llm_claude_opus_5_5.csv")
    assert ratings["raw_score_1_to_10"].tolist() == [8.0, 3.5, 9.5]
    assert human_rater_ids(["friend_a", "llm_claude_opus_5_5"], raters) == {"friend_a"}

    # Second run: cached images are skipped, the new one is judged, a failure is recorded without aborting.
    client.messages.fail_next = True
    client.messages.scores[3] = 6.0
    client.messages.scores[4] = 6.0
    summary = run_judge(judge, manifest, ratings_dir=ratings_dir, root=root, raters_path=raters, stop_after_errors=3)
    assert summary["cached"] == 3 and summary["failed"] == 1 and summary["judged"] == 0
    attributes = attributes_table("llm_claude_opus_5_5", root)
    assert len(attributes) == 3
    assert set(attributes["judge_cloud_type"]) == {"cumulus"}


def test_compare_raters_reports_agreement_and_groups():
    rows = []
    human = {"a": 9.0, "b": 8.0, "c": 4.0, "d": 2.0, "e": 7.5, "f": 6.0}
    model = {"a": 8.5, "b": 8.5, "c": 5.0, "d": 1.0, "e": 7.0, "f": 6.5}
    for image_id, score in human.items():
        rows.append({"image_id": image_id, "rater_id": "friend_a", "raw_score_1_to_10": score})
        rows.append({"image_id": image_id, "rater_id": "llm_x", "raw_score_1_to_10": model[image_id]})
    ratings = pd.DataFrame(rows)
    manifest = pd.DataFrame({"image_id": list(human), "crop_class": ["original"] * 6, "kind": ["original"] * 6, "dataset_name": ["toy"] * 6, "relative_path": [f"{k}.jpg" for k in human]})
    result = compare_raters(ratings, model_rater="llm_x", human_raters=["friend_a"], manifest=manifest, top_k=2)
    assert result["overall"]["n"] == 6
    assert result["overall"]["spearman"] > 0.9
    assert abs(result["overall"]["bias"] - (sum(model.values()) - sum(human.values())) / 6) < 1e-6
    assert "crop_class" in result["by_group"]
    assert len(result["largest_disagreements"]) == 2
    assert result["largest_disagreements"][0]["image_id"] in {"c", "d"}
    markdown = report_markdown(result)
    assert "Rater comparison" in markdown and "Largest disagreements" in markdown
