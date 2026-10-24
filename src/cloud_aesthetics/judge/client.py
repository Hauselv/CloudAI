"""Vision-model judge built on the Anthropic SDK.

``ClaudeJudge.judge`` sends one downscaled JPEG plus the rubric and returns a normalized
judgement dict with token usage. The SDK client is injectable so tests can run without
network access or credentials.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from cloud_aesthetics.judge.prompt import SYSTEM_PROMPT, USER_PROMPT
from cloud_aesthetics.judge.schema import CloudJudgement, normalize_judgement
from cloud_aesthetics.preprocessing.image_ops import read_rgb_image, resize_long_edge

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_EDGE = 1024
DEFAULT_JPEG_QUALITY = 88


def rater_id_for_model(model: str, prefix: str = "llm") -> str:
    return f"{prefix}_{model.replace('-', '_').replace('.', '_')}"


class SupportsParse(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


@dataclass
class JudgeResult:
    image_id: str
    model: str
    judgement: dict[str, object]
    raw: dict[str, object]
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    latency_seconds: float = 0.0
    error: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error


def encode_image_jpeg(path: str | Path, max_edge: int = DEFAULT_MAX_EDGE, quality: int = DEFAULT_JPEG_QUALITY) -> tuple[str, tuple[int, int]]:
    """Return (base64 JPEG, (width, height)) of the image downscaled to ``max_edge``."""
    image = resize_long_edge(read_rgb_image(path), max_edge)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    return base64.standard_b64encode(np.asarray(buffer).tobytes()).decode("ascii"), (image.shape[1], image.shape[0])


def estimate_image_tokens(width: int, height: int) -> int:
    """Rough token cost of an image input (about one token per 750 pixels)."""
    return int(round(width * height / 750.0))


class ClaudeJudge:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any | None = None,
        max_edge: int = DEFAULT_MAX_EDGE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        effort: str = "medium",
        max_tokens: int = 1500,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.max_edge = max_edge
        self.jpeg_quality = jpeg_quality
        self.effort = effort
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            # Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile.
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def rater_id(self) -> str:
        return rater_id_for_model(self.model)

    def judge(self, image_id: str, image_path: str | Path) -> JudgeResult:
        data, (width, height) = encode_image_jpeg(image_path, self.max_edge, self.jpeg_quality)
        started = time.perf_counter()
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": self.effort},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    }
                ],
                output_format=CloudJudgement,
            )
        except Exception as exc:  # noqa: BLE001 - recorded per image, the batch continues
            return JudgeResult(image_id=str(image_id), model=self.model, judgement={}, raw={}, error=f"{type(exc).__name__}: {exc}", latency_seconds=time.perf_counter() - started)
        latency = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "cache_read_input_tokens", 0) or 0) + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        stop_reason = str(getattr(response, "stop_reason", "") or "")
        parsed = getattr(response, "parsed_output", None)
        if stop_reason == "refusal" or parsed is None:
            details = getattr(response, "stop_details", None)
            reason = f"refusal: {getattr(details, 'category', '')} {getattr(details, 'explanation', '')}".strip() if stop_reason == "refusal" else "no parsed output"
            return JudgeResult(str(image_id), self.model, {}, {}, input_tokens, output_tokens, stop_reason, latency, error=reason)
        raw = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        return JudgeResult(
            image_id=str(image_id),
            model=self.model,
            judgement=normalize_judgement(raw),
            raw=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            latency_seconds=latency,
            extra={"image_width": width, "image_height": height, "estimated_image_tokens": estimate_image_tokens(width, height)},
        )
