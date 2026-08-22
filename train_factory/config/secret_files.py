"""Strict, side-effect-free helpers for direct and file-backed secrets."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REQUIRED_SECRET_NAMES = (
    "MYSQL_ROOT_PASSWORD",
    "MYSQL_PASSWORD",
    "MYSQL_URL",
    "JWT_SECRET_KEY",
    "DEFAULT_ADMIN_PASSWORD",
)


def _secret_error(setting_name: str, category: str) -> ValueError:
    return ValueError(f"{setting_name} has {category}")


def _validate_value(value: str, setting_name: str) -> str:
    if value == "":
        raise _secret_error(setting_name, "empty value")
    if "\x00" in value:
        raise _secret_error(setting_name, "NUL byte")
    return value


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def resolve_secret(
    *,
    direct_value: str | None,
    file_path: str | os.PathLike[str] | None,
    setting_name: str,
    max_bytes: int = 65_536,
) -> str | None:
    """Resolve one secret without ever including its value or path in errors."""

    if direct_value is not None and file_path is not None:
        raise _secret_error(setting_name, "conflicting sources")
    if direct_value is not None:
        return _validate_value(direct_value, setting_name)
    if file_path is None:
        return None

    path = Path(file_path)
    try:
        metadata = path.lstat()
    except OSError:
        raise _secret_error(setting_name, "unavailable file") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise _secret_error(setting_name, "invalid file type")
    if metadata.st_size > max_bytes:
        raise _secret_error(setting_name, "file too large")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_metadata.st_mode)
                or _file_identity(opened_metadata) != _file_identity(metadata)
                or opened_metadata.st_size != metadata.st_size
            ):
                raise _secret_error(setting_name, "changed file")
            if opened_metadata.st_size > max_bytes:
                raise _secret_error(setting_name, "file too large")
            chunks = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
    except ValueError:
        raise
    except OSError:
        raise _secret_error(setting_name, "unavailable file") from None

    content = b"".join(chunks)
    if (
        not stat.S_ISREG(after_open.st_mode)
        or not stat.S_ISREG(after_path.st_mode)
        or stat.S_ISLNK(after_path.st_mode)
        or _file_identity(after_open) != _file_identity(metadata)
        or _file_identity(after_path) != _file_identity(metadata)
        or after_open.st_size != metadata.st_size
        or after_path.st_size != metadata.st_size
        or len(content) != metadata.st_size
    ):
        raise _secret_error(setting_name, "changed file")
    if len(content) > max_bytes:
        raise _secret_error(setting_name, "file too large")
    try:
        value = content.decode("utf-8")
    except UnicodeDecodeError:
        raise _secret_error(setting_name, "invalid UTF-8") from None
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return _validate_value(value, setting_name)


def _is_present(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def classify_required_secret_sources(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Require all five deployment secrets to use exactly one source mode."""

    direct_presence = {name: _is_present(environment.get(name)) for name in REQUIRED_SECRET_NAMES}
    file_presence = {name: _is_present(environment.get(f"{name}_FILE")) for name in REQUIRED_SECRET_NAMES}

    if any(direct_presence[name] and file_presence[name] for name in REQUIRED_SECRET_NAMES):
        raise ValueError("required secrets have conflicting sources")

    direct_count = sum(direct_presence.values())
    file_count = sum(file_presence.values())
    if direct_count == len(REQUIRED_SECRET_NAMES) and file_count == 0:
        return {
            "mode": "direct",
            "sources": {name: name for name in REQUIRED_SECRET_NAMES},
        }
    if file_count == len(REQUIRED_SECRET_NAMES) and direct_count == 0:
        return {
            "mode": "files",
            "sources": {name: f"{name}_FILE" for name in REQUIRED_SECRET_NAMES},
        }
    if direct_count and file_count:
        raise ValueError("required secrets have mixed sources")
    raise ValueError("required secrets have incomplete")
