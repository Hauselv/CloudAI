from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cloud_aesthetics.judge.gateway import GatewayJudge, extract_json_object, load_gateway_api_key


def _fake_post_factory(content: str, status: int = 200, log: list | None = None):
    def post(url, headers=None, json=None, timeout=None):
        if log is not None:
            log.append({"url": url, "headers": headers, "json": json})
        payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 900, "completion_tokens": 60}}
        return SimpleNamespace(status_code=status, json=lambda: payload, text=content)

    return post


def test_rate_limit_detection_and_wait_parsing():
    from cloud_aesthetics.judge.gateway import is_rate_limited, rate_limit_wait_seconds

    text = '{"detail":"Rate limit exceeded for end_user: x. Limit type: requests. Current limit: 100, Remaining: 0. Limit resets at: 2026-10-24 13:18:24 UTC"}'
    assert is_rate_limited(400, text) and is_rate_limited(429, "") and not is_rate_limited(400, "bad request")
    now = 1_000.0
    import calendar

    reset = calendar.timegm((2026, 10, 24, 13, 18, 24, 0, 0, 0))
    assert rate_limit_wait_seconds(text, now=reset - 100.0) == 102.0
    assert rate_limit_wait_seconds(text, now=reset + 100.0) == 5.0
    assert rate_limit_wait_seconds("no hint", now=now) == 60.0


def test_gateway_judge_waits_on_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import cloud_aesthetics.judge.gateway as gateway_module

    path = tmp_path / "sky.jpg"
    cv2.imwrite(str(path), np.full((100, 100, 3), 200, dtype=np.uint8))
    calls = {"n": 0}
    good = json.dumps({"score": 6.0})

    def post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(status_code=400, json=lambda: {}, text="Rate limit exceeded. Limit resets at: 2000-01-01 00:00:00 UTC")
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": good}, "finish_reason": "stop"}], "usage": {}}, text=good)

    slept: list[float] = []
    monkeypatch.setattr(gateway_module.time, "sleep", lambda seconds: slept.append(seconds))
    judge = GatewayJudge("https://example.test/api", "m", "k", post=post)
    result = judge.judge("img1", path)
    assert result.ok and result.judgement["score"] == 6.0
    assert calls["n"] == 2 and slept == [5.0]


def test_extract_json_object_handles_fences_and_prose():
    assert extract_json_object('{"score": 7}') == {"score": 7}
    assert extract_json_object('```json\n{"score": 7.5}\n```') == {"score": 7.5}
    assert extract_json_object('Here you go: {"score": 8, "light": "night"} thanks') == {"score": 8, "light": "night"}
    assert extract_json_object("no json here") is None


def test_load_gateway_api_key_prefers_env_then_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    secrets = tmp_path / "s.json"
    secrets.write_text(json.dumps({"gateway_api_key": "file-key"}), encoding="utf-8")
    monkeypatch.delenv("CLOUD_JUDGE_GATEWAY_API_KEY", raising=False)
    assert load_gateway_api_key(secrets_file=secrets) == "file-key"
    monkeypatch.setenv("CLOUD_JUDGE_GATEWAY_API_KEY", "env-key")
    assert load_gateway_api_key(secrets_file=secrets) == "env-key"
    monkeypatch.delenv("CLOUD_JUDGE_GATEWAY_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_gateway_api_key(secrets_file=tmp_path / "missing.json")


def test_gateway_judge_builds_openai_request_and_normalizes(tmp_path: Path):
    path = tmp_path / "sky.jpg"
    cv2.imwrite(str(path), np.full((600, 900, 3), 200, dtype=np.uint8))
    log: list = []
    content = json.dumps({"score": 8.3, "confidence": 0.7, "cloud_types": ["Altocumulus"], "light": "golden_hour", "is_sunset_or_sunrise": True, "dominant_colors": ["orange"], "strengths": ["texture"], "weaknesses": [], "quality_flags": [], "one_sentence_verdict": "Nice."})
    judge = GatewayJudge("https://example.test/api/", "qwen3.6-35b", "secret", post=_fake_post_factory(content, log=log), max_edge=300)
    result = judge.judge("img1", path)
    assert result.ok, result.error
    assert result.judgement["score"] == 8.5
    assert result.judgement["primary_cloud_type"] == "altocumulus"
    assert result.input_tokens == 900 and result.output_tokens == 60
    assert judge.rater_id == "llm_qwen3_6_35b"
    request = log[0]
    assert request["url"] == "https://example.test/api/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer secret"
    body = request["json"]
    assert body["model"] == "qwen3.6-35b"
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert result.extra["image_width"] == 300


def test_gateway_judge_records_http_errors_and_bad_json(tmp_path: Path):
    path = tmp_path / "sky.jpg"
    cv2.imwrite(str(path), np.full((100, 100, 3), 200, dtype=np.uint8))
    judge = GatewayJudge("https://example.test/api", "m", "k", post=_fake_post_factory("nope", status=401))
    result = judge.judge("img1", path)
    assert not result.ok and result.error.startswith("HTTP 401")
    judge = GatewayJudge("https://example.test/api", "m", "k", post=_fake_post_factory("I cannot rate this."))
    result = judge.judge("img1", path)
    assert not result.ok and "no JSON" in result.error
