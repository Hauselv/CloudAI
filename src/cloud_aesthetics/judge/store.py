"""Persistence for judge runs.

- Raw judgements are cached per image under ``data/processed/judge/<rater_id>/<image_id>.json``
  so re-runs never pay twice.
- Scores are appended to the normal rater CSV (``data/raw/metadata/ratings/<rater_id>.csv``),
  so the whole label tooling treats the model as one more rater.
- The rater registry (``data/raw/metadata/raters.csv``) records which raters are models, and
  the aggregation excludes them from the human consensus unless explicitly asked.
- Attributes (cloud types, light, flags, ...) are collected into one parquet table per rater.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from cloud_aesthetics.data.ratings import record_rating
from cloud_aesthetics.judge.client import ClaudeJudge, JudgeResult
from cloud_aesthetics.settings import ensure_parent, resolve_path
from cloud_aesthetics.utils.io import read_table

JUDGE_ROOT = "data/processed/judge"
RATERS_PATH = "data/raw/metadata/raters.csv"
RATER_COLUMNS = ["rater_id", "rater_kind", "model", "created_at", "note"]


def load_raters(path_like: str | Path = RATERS_PATH) -> pd.DataFrame:
    frame = read_table(path_like)
    if frame.empty:
        return pd.DataFrame(columns=RATER_COLUMNS)
    return frame.reindex(columns=RATER_COLUMNS)


def register_rater(rater_id: str, rater_kind: str, *, model: str = "", note: str = "", path_like: str | Path = RATERS_PATH) -> pd.DataFrame:
    frame = load_raters(path_like)
    if (frame["rater_id"].astype(str) == rater_id).any():
        return frame
    row = pd.DataFrame([{"rater_id": rater_id, "rater_kind": rater_kind, "model": model, "created_at": datetime.now(timezone.utc).isoformat(), "note": note}])
    frame = pd.concat([frame, row], ignore_index=True)
    frame.to_csv(ensure_parent(path_like), index=False)
    return frame


def rater_kinds(path_like: str | Path = RATERS_PATH) -> dict[str, str]:
    frame = load_raters(path_like)
    return {str(row["rater_id"]): str(row["rater_kind"]) for _, row in frame.iterrows()}


def human_rater_ids(all_rater_ids: Iterable[str], path_like: str | Path = RATERS_PATH) -> set[str]:
    """Raters not registered as models count as human (backwards compatible)."""
    kinds = rater_kinds(path_like)
    return {str(rater) for rater in all_rater_ids if kinds.get(str(rater), "human") == "human"}


def judge_dir(rater_id: str, root: str | Path = JUDGE_ROOT) -> Path:
    return resolve_path(root) / rater_id


def cached_result(rater_id: str, image_id: str, root: str | Path = JUDGE_ROOT) -> dict[str, object] | None:
    path = judge_dir(rater_id, root) / f"{image_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_result(result: JudgeResult, rater_id: str, root: str | Path = JUDGE_ROOT) -> Path:
    path = ensure_parent(judge_dir(rater_id, root) / f"{result.image_id}.json")
    payload = {
        "image_id": result.image_id,
        "model": result.model,
        "rater_id": rater_id,
        "judgement": result.judgement,
        "raw": result.raw,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "stop_reason": result.stop_reason,
        "latency_seconds": result.latency_seconds,
        "error": result.error,
        "extra": result.extra,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def attributes_table(rater_id: str, root: str | Path = JUDGE_ROOT) -> pd.DataFrame:
    """Flatten all cached judgements of a rater into one table (also written to parquet)."""
    directory = judge_dir(rater_id, root)
    rows: list[dict[str, object]] = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            judgement = payload.get("judgement") or {}
            if payload.get("error") or not judgement:
                continue
            rows.append(
                {
                    "image_id": str(payload["image_id"]),
                    "rater_id": rater_id,
                    "model": payload.get("model", ""),
                    "judge_score": judgement.get("score"),
                    "judge_confidence": judgement.get("confidence"),
                    "judge_cloud_type": judgement.get("primary_cloud_type"),
                    "judge_cloud_types": "|".join(judgement.get("cloud_types", [])),
                    "judge_light": judgement.get("light"),
                    "judge_is_sunset": bool(judgement.get("is_sunset_or_sunrise", False)),
                    "judge_colors": "|".join(judgement.get("dominant_colors", [])),
                    "judge_strengths": " / ".join(judgement.get("strengths", [])),
                    "judge_weaknesses": " / ".join(judgement.get("weaknesses", [])),
                    "judge_flags": "|".join(judgement.get("quality_flags", [])),
                    "judge_verdict": judgement.get("one_sentence_verdict", ""),
                    "input_tokens": payload.get("input_tokens", 0),
                    "output_tokens": payload.get("output_tokens", 0),
                    "latency_seconds": payload.get("latency_seconds", 0.0),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_parquet(ensure_parent(resolve_path(root) / f"{rater_id}_attributes.parquet"), index=False)
    return frame


def run_judge(
    judge: ClaudeJudge,
    manifest: pd.DataFrame,
    *,
    image_ids: Iterable[str] | None = None,
    limit: int | None = None,
    ratings_dir: str | Path = "data/raw/metadata/ratings",
    root: str | Path = JUDGE_ROOT,
    raters_path: str | Path = RATERS_PATH,
    session_id: str | None = None,
    progress: Callable[[str], None] | None = None,
    stop_after_errors: int = 5,
) -> dict[str, object]:
    """Judge every requested image once, cache results and append scores as a rater."""
    notify = progress or (lambda message: None)
    rater_id = judge.rater_id
    register_rater(rater_id, "llm", model=judge.model, note="vision judge", path_like=raters_path)
    frame = manifest.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    if image_ids is not None:
        wanted = {str(value) for value in image_ids}
        frame = frame[frame["image_id"].isin(wanted)]
    frame = frame.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    session = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_judge")
    done = skipped = failed = 0
    consecutive_errors = 0
    tokens_in = tokens_out = 0
    for _, row in frame.iterrows():
        if limit is not None and done >= limit:
            break
        image_id = str(row["image_id"])
        if cached_result(rater_id, image_id, root) is not None:
            skipped += 1
            continue
        result = judge.judge(image_id, row["relative_path"])
        save_result(result, rater_id, root)
        tokens_in += result.input_tokens
        tokens_out += result.output_tokens
        if not result.ok:
            failed += 1
            consecutive_errors += 1
            notify(f"{image_id}: ERROR {result.error}")
            if consecutive_errors >= stop_after_errors:
                notify(f"Stopping after {consecutive_errors} consecutive errors.")
                break
            continue
        consecutive_errors = 0
        record_rating(ratings_dir, image_id=image_id, rater_id=rater_id, score=float(result.judgement["score"]), rating_session_id=session, note=str(result.judgement.get("one_sentence_verdict", ""))[:300])
        done += 1
        notify(f"{image_id}: {result.judgement['score']:.1f} ({result.judgement.get('primary_cloud_type')}, {result.judgement.get('light')}) {result.latency_seconds:.1f}s")
    attributes = attributes_table(rater_id, root)
    return {
        "rater_id": rater_id,
        "model": judge.model,
        "judged": done,
        "cached": skipped,
        "failed": failed,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "attributes_rows": int(len(attributes)),
        "session_id": session,
    }
