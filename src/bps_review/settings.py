from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from bps_review.utils.paths import PROJECT_ROOT, WORKSPACE_ROOT


ROOT = PROJECT_ROOT


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def protocol_config() -> dict[str, Any]:
    return _load_yaml(PROJECT_ROOT / "config" / "protocol.yaml")


@lru_cache(maxsize=1)
def query_config() -> dict[str, Any]:
    return _load_yaml(PROJECT_ROOT / "config" / "search_queries.yaml")


@lru_cache(maxsize=1)
def pipeline_config() -> dict[str, Any]:
    return _load_yaml(PROJECT_ROOT / "config" / "pipeline.yaml")


def resolve_path(config_key: str) -> Path:
    relative = pipeline_config()["paths"][config_key]
    path = WORKSPACE_ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    return path
