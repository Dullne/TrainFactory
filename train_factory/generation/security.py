"""Security boundaries shared by generation routes and workers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from ..auth.dependencies import validate_storage_path
from ..config.settings import get_settings
from ..utils.path_utils import map_storage_path


def get_generation_input_allowed_dirs() -> List[str]:
    """Return configured and legacy storage roots accepted for generation input."""
    candidates: Iterable[object] = (
        get_settings().datasets_dir,
        os.getenv("GENERATION_OUTPUT_DIR", "/app/data/datasets"),
        os.getenv("SYNC_DATA_DIR", "/app/data/sync"),
        "/data",
        "/app/data",
        "/app/datasets",
    )
    return list(dict.fromkeys(str(path) for path in candidates if path))


def resolve_generation_input_path(
    path: str,
    *,
    allowed_dirs: Optional[List[str]] = None,
    map_host_path: bool = False,
) -> str:
    """Map, authorize, and canonicalize a local generation input path.

    The canonical path is returned so a symlink accepted at the API boundary is
    not retained in the persisted task. Workers call this function again just
    before reading to catch changed directory links and legacy task values.
    """
    candidate = path
    if map_host_path:
        candidate, _ = map_storage_path(path)

    roots = (
        get_generation_input_allowed_dirs()
        if allowed_dirs is None
        else allowed_dirs
    )
    validate_storage_path(
        candidate,
        allowed_dirs=roots,
        resource_type="generation input",
    )
    return str(Path(candidate).resolve())
