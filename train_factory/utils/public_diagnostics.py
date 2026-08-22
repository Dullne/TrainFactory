"""Stable, non-sensitive diagnostics for public task APIs."""

import hashlib
import re
from typing import Any


PUBLIC_TASK_FAILURE_MESSAGE = "Task failed. Check server logs for details."
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _looks_like_local_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    return (
        bool(_WINDOWS_ABSOLUTE_PATH.match(candidate))
        or candidate.startswith("/")
        or candidate.lower().startswith("file://")
    )


def public_safe_identifier(value: Any) -> Any:
    """Hash path-shaped identifiers so legacy dictionary keys stay opaque."""
    if not _looks_like_local_path(value):
        return value
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"local:sha256:{digest}"


def public_task_error_message(error_message: Any) -> str | None:
    """Return a stable public failure message without exposing worker details."""
    if error_message is None:
        return None
    if isinstance(error_message, str) and not error_message.strip():
        return None
    return PUBLIC_TASK_FAILURE_MESSAGE


def _is_error_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return (
        normalized in {
            "_error",
            "error",
            "errors",
            "error_message",
            "exception",
            "exception_message",
            "failure_reason",
            "stderr",
            "stdout",
            "traceback",
            "stack_trace",
        }
        or normalized.endswith("_error")
        or normalized.endswith("_error_message")
    )


def _is_weak_diagnostic_key(key: Any) -> bool:
    return str(key).strip().lower() in {"reason", "detail", "message"}


def _is_path_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return (
        normalized in {"path", "file", "directory", "dir"}
        or normalized.endswith("_path")
        or normalized.endswith("_file")
        or normalized.endswith("_directory")
        or normalized.endswith("_dir")
    )


def sanitize_public_diagnostics(value: Any, *, _diagnostic: bool = False) -> Any:
    """Recursively redact paths and implementation errors from public payloads."""
    if isinstance(value, dict):
        sanitized = {}
        status = str(value.get("status") or "").strip().lower()
        failed_container = (
            status in {"error", "failed", "failure", "skipped"}
            or value.get("skipped") is True
            or any(_is_error_key(key) for key in value)
        )
        for key, item in value.items():
            public_key = public_safe_identifier(key)
            if public_key in sanitized:
                collision_digest = hashlib.sha256(
                    repr(key).encode("utf-8")
                ).hexdigest()
                public_key = f"redacted-key:sha256:{collision_digest}"
                suffix = 1
                while public_key in sanitized:
                    public_key = (
                        f"redacted-key:sha256:{collision_digest}:{suffix}"
                    )
                    suffix += 1
            if _is_path_key(key):
                sanitized[public_key] = None
                continue
            normalized_key = str(key).strip().lower()
            is_error = _is_error_key(key) or (
                (_diagnostic or failed_container)
                and _is_weak_diagnostic_key(key)
            )
            if is_error and not isinstance(item, (dict, list)):
                sanitized[public_key] = public_task_error_message(item)
                continue
            child_diagnostic = is_error or (
                failed_container
                and normalized_key in {"details", "diagnostic", "diagnostics"}
            )
            if _diagnostic and normalized_key != "status":
                child_diagnostic = True
            sanitized[public_key] = sanitize_public_diagnostics(
                item,
                _diagnostic=child_diagnostic,
            )
        return sanitized
    if isinstance(value, list):
        return [
            sanitize_public_diagnostics(item, _diagnostic=_diagnostic)
            for item in value
        ]
    if _diagnostic and isinstance(value, str) and value:
        return PUBLIC_TASK_FAILURE_MESSAGE
    if _looks_like_local_path(value):
        return None
    return value
