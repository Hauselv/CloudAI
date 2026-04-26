from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloud_aesthetics.settings import ensure_parent, resolve_path


@dataclass(slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    config: dict[str, Any]


def create_run_context(output_dir: str | Path, run_name: str, config: dict[str, Any]) -> RunContext:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_name}_{timestamp}"
    run_dir = resolve_path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context = RunContext(run_id=run_id, run_dir=run_dir, config=config)
    save_run_json({"run_id": run_id, "config": config}, run_dir / "run.json")
    return context


def save_run_json(data: dict[str, Any], path_like: str | Path) -> Path:
    path = ensure_parent(path_like)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return path


def load_run_json(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
