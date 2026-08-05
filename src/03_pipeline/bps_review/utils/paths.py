from __future__ import annotations

import os
from pathlib import Path


# .../src/03_pipeline/bps_review/utils/paths.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
_workspace_override = os.getenv("BPS_WORKSPACE_ROOT", "").strip()

# The workspace (src) is grouped into numbered sections that follow the order of
# the review itself. Code refers to a section by its plain name, and this table
# is the single place that knows where that section physically lives, so the
# grouping can change without touching a single call site.
WORKSPACE_SECTIONS: dict[str, str] = {
    "protocol": "01_protocol",
    "coding_schemes": "02_coding_schemes",
    "pipeline": "03_pipeline",
    "config": "03_pipeline/config",
    "tests": "03_pipeline/tests",
    "notebooks": "04_notebooks",
    "data": "05_data",
    "review_stages": "06_review_stages",
    "semantic": "07_semantic_space",
    "docs": "08_docs",
    "artifacts": "09_artifacts",
}


def _default_workspace_root() -> Path:
    if _workspace_override:
        return Path(_workspace_override).expanduser().resolve()
    src_root = PROJECT_ROOT / "src"
    markers = [src_root / WORKSPACE_SECTIONS[name]
               for name in ("review_stages", "data", "artifacts", "protocol")]
    if src_root.exists() and any(marker.exists() for marker in markers):
        return src_root
    return PROJECT_ROOT


WORKSPACE_ROOT = _default_workspace_root()
# Prefixes that live at the project root rather than inside the workspace (src).
PROJECT_SCOPED_PREFIXES = {
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
    """Resolve a workspace-relative path.

    The first part is a section name from ``WORKSPACE_SECTIONS`` (for example
    ``data``, ``config``, or ``review_stages``), a project-root prefix, or a plain
    directory name that is then used as given.
    """
    if not parts:
        return WORKSPACE_ROOT
    if parts[0] in PROJECT_SCOPED_PREFIXES:
        return PROJECT_ROOT.joinpath(*parts)
    section = WORKSPACE_SECTIONS.get(parts[0])
    if section:
        return WORKSPACE_ROOT.joinpath(section, *parts[1:])
    return WORKSPACE_ROOT.joinpath(*parts)
