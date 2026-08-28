"""Path mapping utilities for host-to-container translation."""

import os
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from ..config.settings import get_settings


def _normalize_path(path: str) -> str:
    """Normalize path with user expansion and absolute resolution."""
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def canonicalize_artifact_path(path: Optional[str]) -> Optional[str]:
    """Canonicalize a local artifact path for ownership comparisons."""
    if not isinstance(path, str) or not path.strip():
        return None
    normalized = _normalize_path(path.strip())
    try:
        resolved = os.fspath(Path(normalized).resolve(strict=False))
    except (OSError, RuntimeError) as exc:
        raise ValueError("artifact path could not be canonicalized") from exc
    return os.path.normcase(os.path.normpath(resolved))


def artifact_path_uses_root(
    candidate: Optional[str],
    root: Optional[str],
) -> bool:
    """Return whether ``candidate`` is exactly ``root`` or below it."""
    canonical_candidate = canonicalize_artifact_path(candidate)
    canonical_root = canonicalize_artifact_path(root)
    if canonical_candidate is None or canonical_root is None:
        return False
    try:
        return os.path.commonpath((canonical_candidate, canonical_root)) == (
            canonical_root
        )
    except ValueError:
        return False


def _parse_mapping_item(item: str) -> Optional[Tuple[str, str]]:
    """Parse a single mapping item into (host_prefix, container_prefix)."""
    if "->" in item:
        host, container = item.split("->", 1)
    elif ":" in item:
        host, container = item.split(":", 1)
    else:
        return None

    host = host.strip()
    container = container.strip()
    if not host or not container:
        return None

    return _normalize_path(host), _normalize_path(container)


@lru_cache(maxsize=1)
def get_path_mappings() -> List[Tuple[str, str]]:
    """Load path mappings from settings."""
    raw = get_settings().path_mappings.strip()
    if not raw:
        return []

    mappings: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = _parse_mapping_item(item)
        if parsed:
            mappings.append(parsed)

    # Longest host prefix wins
    mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
    return mappings


def map_storage_path(path: str) -> Tuple[str, Optional[str]]:
    """Map a host path to container path based on PATH_MAPPINGS.

    Returns:
        (mapped_path, original_host_path_if_mapped)
    """
    if not path:
        return path, None

    normalized = _normalize_path(path)
    for host_prefix, container_prefix in get_path_mappings():
        if normalized == host_prefix or normalized.startswith(host_prefix + os.sep):
            rel = os.path.relpath(normalized, host_prefix)
            mapped = container_prefix if rel in (".", "") else os.path.normpath(os.path.join(container_prefix, rel))
            return mapped, normalized

    return normalized, None


def unmap_storage_path(path: str) -> str:
    """Map a container path back to host path based on PATH_MAPPINGS.

    Returns the host path if mapping exists, otherwise returns original path.
    """
    if not path:
        return path

    normalized = _normalize_path(path)
    for host_prefix, container_prefix in get_path_mappings():
        if normalized == container_prefix or normalized.startswith(container_prefix + os.sep):
            rel = os.path.relpath(normalized, container_prefix)
            return host_prefix if rel in (".", "") else os.path.normpath(os.path.join(host_prefix, rel))

    return normalized


def hash_path(path: str, salt: Optional[str] = None) -> str:
    """Generate a deterministic hash for a normalized path (optionally salted)."""
    normalized = _normalize_path(path)
    if salt:
        normalized = f"{normalized}::{salt}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
