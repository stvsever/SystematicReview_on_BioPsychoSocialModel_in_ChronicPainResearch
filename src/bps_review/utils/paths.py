from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_workspace_override = os.getenv("BPS_WORKSPACE_ROOT", "").strip()


def _default_workspace_root() -> Path:
    if _workspace_override:
        return Path(_workspace_override).expanduser().resolve()
    src_root = PROJECT_ROOT / "src"
    markers = [src_root / "review_stages", src_root / "data", src_root / "artifacts", src_root / "protocol"]
    if src_root.exists() and any(marker.exists() for marker in markers):
        return src_root
    return PROJECT_ROOT


WORKSPACE_ROOT = _default_workspace_root()
PROJECT_SCOPED_PREFIXES = {
    "config",
    "docker",
    "paper",
    "pyproject.toml",
    "README.md",
    "Makefile",
    ".env",
    ".env.example",
    ".gitignore",
    "src",
}


def project_path(*parts: str) -> Path:
    if not parts:
        return WORKSPACE_ROOT
    if parts[0] in PROJECT_SCOPED_PREFIXES:
        return PROJECT_ROOT.joinpath(*parts)
    return WORKSPACE_ROOT.joinpath(*parts)
