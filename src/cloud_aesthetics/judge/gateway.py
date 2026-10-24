"""Vision judge for OpenAI-compatible gateways (e.g. a university LLM server).

Same rubric, schema and result type as :mod:`cloud_aesthetics.judge.client`, but the
request goes to ``{base_url}/chat/completions`` with an ``image_url`` data URI and JSON
mode. The API key is read from an environment variable or a secrets JSON file (for
example the one the spider crawler already uses) and is never logged or stored here.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from cloud_aesthetics.judge.client import JudgeResult, encode_image_jpeg, estimate_image_tokens, rater_id_for_model
from cloud_aesthetics.judge.prompt import SYSTEM_PROMPT, USER_PROMPT
from cloud_aesthetics.judge.schema import CloudJudgement, normalize_judgement

JSON_FORMAT_HINT = (
    "\n\nReturn exactly one JSON object with these keys and types: "
    + json.dumps(
        {
            "score": "number 0-10 in 0.5 steps",
            "confidence": "number 0-1",
            "cloud_types": ["string"],
            "light": "string",
            "is_sunset_or_sunrise": "boolean",
            "dominant_colors": ["string"],
            "strengths": ["string"],
            "weaknesses": ["string"],
            "quality_flags": ["string"],
            "one_sentence_verdict": "string",
        }
    )
    + ". No prose outside the JSON."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_RESET_AT = re.compile(r"resets at:\s*([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\s*UTC", re.IGNORECASE)
MAX_RATE_LIMIT_WAIT_SECONDS = 20 * 60


def is_rate_limited(status: int, text: str) -> bool:
    lowered = (text or "").lower()[:600]
    return status == 429 or (status == 400 and "rate limit" in lowered)


def rate_limit_wait_seconds(text: str, now: float | None = None, default: float = 60.0) -> float:
    """Seconds to wait before retrying, taken from the gateway's 'Limit resets at' hint when present."""
    from datetime import datetime, timezone

    match = _RESET_AT.search(text or "")
    if not match:
        return default
    reset = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    current = now if now is not None else time.time()
    return float(min(max(reset - current + 2.0, 5.0), MAX_RATE_LIMIT_WAIT_SECONDS))


def load_gateway_api_key(
    *,
    env_var: str = "CLOUD_JUDGE_GATEWAY_API_KEY",
    secrets_file: str | Path | None = None,
    secrets_key: str = "gateway_api_key",
) -> str:
    """Environment variable first, then the named key of a secrets JSON file."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    if secrets_file:
        path = Path(secrets_file)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            value = str(data.get(secrets_key, "") or "").strip()
            if value:
                return value
    raise RuntimeError(
        f"No gateway API key: set {env_var} or point --secrets-file at a JSON file with a '{secrets_key}' entry."
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        pass
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class GatewayJudge:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        max_edge: int = 1024,
        jpeg_quality: int = 88,
        max_tokens: int = 1500,  # verdicts with three cloud types and phrases run to ~500-700 tokens; leave headroom
        temperature: float = 0.2,
        timeout_seconds: float = 120.0,
        disable_thinking: bool = True,
        json_mode: bool = True,
        post: Callable[..., Any] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        rater_prefix: str = "llm",
        on_wait: Callable[[float, str], None] | None = None,
    ) -> None:
        self.on_wait = on_wait
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.max_edge = max_edge
        self.jpeg_quality = jpeg_quality
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.disable_thinking = disable_thinking
        self.json_mode = json_mode
        self.system_prompt = system_prompt
        self.rater_prefix = rater_prefix
        if post is None:
            import requests

            self._post = requests.post
        else:
            self._post = post

    @property
    def rater_id(self) -> str:
        return rater_id_for_model(self.model, prefix=self.rater_prefix)

    def _body(self, data: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt + JSON_FORMAT_HINT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def judge(self, image_id: str, image_path: str | Path) -> JudgeResult:
        data, (width, height) = encode_image_jpeg(image_path, self.max_edge, self.jpeg_quality)
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        body = self._body(data)
        try:
            response = None
            status = 0
            for attempt in range(4):
                response = self._post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout_seconds)
                status = int(getattr(response, "status_code", 200))
                text = str(getattr(response, "text", ""))
                if is_rate_limited(status, text) and attempt < 3:
                    wait = rate_limit_wait_seconds(text)
                    if self.on_wait is not None:
                        self.on_wait(wait, text[:160])
                    time.sleep(wait)
                    continue
                if status >= 500 and attempt < 3:
                    time.sleep(20)
                    continue
                break
            if status >= 400:
                text = str(getattr(response, "text", ""))[:300]
                return JudgeResult(str(image_id), self.model, {}, {}, latency_seconds=time.perf_counter() - started, error=f"HTTP {status}: {text}")
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - recorded per image, the batch continues
            return JudgeResult(str(image_id), self.model, {}, {}, latency_seconds=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
        latency = time.perf_counter() - started
        usage = payload.get("usage") or {}
        choice = (payload.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        raw = extract_json_object(str(content))
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        finish = str(choice.get("finish_reason", "") or "")
        if raw is None:
            return JudgeResult(str(image_id), self.model, {}, {"content": str(content)[:500]}, input_tokens, output_tokens, finish, latency, error="no JSON object in response")
        try:
            validated = CloudJudgement.model_validate({**{"cloud_types": [], "dominant_colors": [], "strengths": [], "weaknesses": [], "quality_flags": [], "one_sentence_verdict": "", "light": "unknown", "is_sunset_or_sunrise": False, "confidence": 0.5}, **raw})
        except Exception as exc:  # noqa: BLE001 - schema drift is recorded, not fatal
            return JudgeResult(str(image_id), self.model, {}, raw, input_tokens, output_tokens, finish, latency, error=f"schema: {exc}")
        return JudgeResult(
            image_id=str(image_id),
            model=self.model,
            judgement=normalize_judgement(validated),
            raw=validated.model_dump(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=finish,
            latency_seconds=latency,
            extra={"image_width": width, "image_height": height, "estimated_image_tokens": estimate_image_tokens(width, height), "gateway": self.base_url},
        )
