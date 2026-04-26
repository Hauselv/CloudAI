from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "configs"


@dataclass(slots=True)
class AppPaths:
    project_root: Path = PROJECT_ROOT
    data_root: Path = DATA_ROOT
    config_root: Path = CONFIG_ROOT

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def artifacts_root(self) -> Path:
        return self.data_root / "artifacts"


PATHS = AppPaths()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_parent(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
