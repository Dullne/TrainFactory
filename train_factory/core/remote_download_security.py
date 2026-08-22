"""Security controls shared by user-triggered remote downloads."""

import os
import re
import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Optional


_REPO_COMPONENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
_REMOTE_REPO_ID = re.compile(
    rf"{_REPO_COMPONENT}/{_REPO_COMPONENT}\Z",
    flags=re.ASCII,
)
_REMOTE_ARTIFACT_NAME = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?\Z",
    flags=re.ASCII,
)


class DownloadConcurrencyExceeded(ValueError):
    """Raised when accepting another download would exceed a configured cap."""


class DownloadSizeExceeded(ValueError):
    """Raised when repository metadata or downloaded files exceed their cap."""


class DownloadMetadataUnavailable(ValueError):
    """Raised when a remote repository cannot be sized before downloading."""


class DownloadSizeVerificationFailed(RuntimeError):
    """Raised when local artifact bytes cannot be measured completely."""


class DownloadPolicyViolation(ValueError):
    """Raised when a remote download mode cannot meet tenant isolation rules."""


class DownloadStorageQuotaExceeded(ValueError):
    """Raised when persisted bytes plus active reservations exceed a quota."""


class DownloadLease:
    """Idempotently releases one slot held by a download."""

    def __init__(self, limiter: "DownloadConcurrencyLimiter", user_key: str):
        self._limiter = limiter
        self._user_key = user_key
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._limiter._release(self._user_key)


class DownloadConcurrencyLimiter:
    """Process-wide accounting for active model and dataset downloads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active_by_user: Counter[str] = Counter()
        self._active_total = 0

    @property
    def active_total(self) -> int:
        with self._lock:
            return self._active_total

    def acquire(
        self,
        user_id: Optional[str],
        *,
        global_limit: int,
        per_user_limit: int,
    ) -> DownloadLease:
        user_key = user_id or "<anonymous>"
        with self._lock:
            if self._active_total >= global_limit:
                raise DownloadConcurrencyExceeded(
                    "Remote download global concurrency limit exceeded"
                )
            if self._active_by_user[user_key] >= per_user_limit:
                raise DownloadConcurrencyExceeded(
                    "Remote download per-user concurrency limit exceeded"
                )
            self._active_total += 1
            self._active_by_user[user_key] += 1
        return DownloadLease(self, user_key)

    def _release(self, user_key: str) -> None:
        with self._lock:
            if self._active_by_user[user_key] <= 0:
                return
            self._active_total -= 1
            self._active_by_user[user_key] -= 1
            if self._active_by_user[user_key] == 0:
                del self._active_by_user[user_key]


download_concurrency_limiter = DownloadConcurrencyLimiter()


def resolve_managed_artifact_directory(
    root: Path,
    artifact_id: str,
    stored_path: Optional[str] = None,
) -> Path:
    """Resolve one exact first-level artifact directory under a managed root."""
    root_path = Path(root).resolve()
    candidate = (root_path / artifact_id).resolve()
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("Artifact directory is outside its managed storage root") from exc
    if len(relative.parts) != 1 or relative.name != artifact_id:
        raise ValueError("Artifact ID does not identify one managed directory")
    if stored_path is not None and Path(stored_path).resolve() != candidate:
        raise ValueError("Stored artifact path does not match its managed directory")
    return candidate


def calculate_directory_size_strict(path: Path | str) -> int:
    """Measure every regular file without following links or hiding I/O errors."""
    root = Path(path)
    if root.is_symlink():
        raise DownloadSizeVerificationFailed(
            "Unable to verify downloaded artifact size: symbolic link encountered"
        )
    if not root.is_dir():
        raise DownloadSizeVerificationFailed(
            "Unable to verify downloaded artifact size: directory is unavailable"
        )

    total_size = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise DownloadSizeVerificationFailed(
                            "Unable to verify downloaded artifact size: "
                            "symbolic link encountered"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total_size += entry.stat(follow_symlinks=False).st_size
                    else:
                        raise DownloadSizeVerificationFailed(
                            "Unable to verify downloaded artifact size: "
                            "unsupported filesystem entry encountered"
                        )
        except DownloadSizeVerificationFailed:
            raise
        except OSError as exc:
            raise DownloadSizeVerificationFailed(
                "Unable to verify downloaded artifact size"
            ) from exc
    return total_size


def reclaim_download_directory(
    path: Path | str,
    *,
    fallback_bytes: int,
) -> tuple[bool, Optional[int]]:
    """Remove a download directory or return bytes that must remain billable."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return True, None
    except OSError:
        try:
            residual_bytes = calculate_directory_size_strict(path)
        except DownloadSizeVerificationFailed:
            residual_bytes = max(0, fallback_bytes)
        return False, residual_bytes
    return True, None


def validate_remote_repo_id(remote_repo: str) -> str:
    """Return a canonical ``namespace/repository`` identifier.

    Requiring exactly two conservative Hub name components keeps download
    libraries from interpreting user input as a URL, local path, or packaged
    dataset loader.
    """
    if (
        not isinstance(remote_repo, str)
        or remote_repo != remote_repo.strip()
        or len(remote_repo) > 96
        or not _REMOTE_REPO_ID.fullmatch(remote_repo)
        or ".." in remote_repo
        or "--" in remote_repo
        or remote_repo.endswith(".git")
    ):
        raise ValueError(
            "Remote repository must be a canonical namespace/repository ID"
        )
    return remote_repo


def validate_remote_artifact_name(name: str, *, kind: str) -> str:
    """Reject remote metadata names that could escape an artifact directory."""
    if (
        not isinstance(name, str)
        or not _REMOTE_ARTIFACT_NAME.fullmatch(name)
        or ".." in name
    ):
        raise ValueError(f"Remote {kind} name is not a safe file component")
    return name


def enforce_download_size(
    actual_bytes: int,
    max_bytes: int,
    *,
    remote_repo: str,
) -> None:
    """Reject a repository or artifact larger than its configured cap."""
    if actual_bytes > max_bytes:
        raise DownloadSizeExceeded(
            f"Remote repository {remote_repo!r} size {actual_bytes} bytes "
            f"exceeds configured limit {max_bytes} bytes"
        )


def _file_size(file_info: Any) -> int:
    if isinstance(file_info, dict):
        value = file_info.get("size")
        if value is None:
            value = file_info.get("Size")
        lfs = file_info.get("lfs")
        if lfs is None:
            lfs = file_info.get("Lfs")
    else:
        value = getattr(file_info, "size", None)
        lfs = getattr(file_info, "lfs", None)
    if value is None and lfs is not None:
        value = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if value is None or isinstance(value, bool):
        raise DownloadMetadataUnavailable("Remote file size metadata is missing")
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise DownloadMetadataUnavailable(
            "Remote file size metadata is invalid"
        ) from exc
    if size < 0:
        raise DownloadMetadataUnavailable("Remote file size metadata is invalid")
    return size


def preflight_remote_repo_size(
    *,
    download_source: str,
    repo_type: str,
    remote_repo: str,
    max_bytes: int,
    anonymous: bool,
    timeout_seconds: int,
) -> Optional[int]:
    """Query Hub file metadata and reject repositories over the limit.

    ``None`` means that the provider returned files without byte-size metadata;
    callers must retain their conservative full-limit storage reservation.
    """
    if repo_type not in {"model", "dataset"}:
        raise ValueError(f"Unsupported repository type: {repo_type}")

    if download_source not in {"huggingface", "modelscope"}:
        raise ValueError(f"Unsupported download source: {download_source}")

    try:
        if download_source == "huggingface":
            from huggingface_hub import HfApi

            api_kwargs = {"token": False} if anonymous else {}
            request_kwargs = {
                "files_metadata": True,
                "timeout": timeout_seconds,
            }
            if anonymous:
                request_kwargs["token"] = False
            api = HfApi(**api_kwargs)
            if repo_type == "model":
                info = api.model_info(remote_repo, **request_kwargs)
            else:
                info = api.dataset_info(remote_repo, **request_kwargs)
            files = info.siblings or []
        else:
            from modelscope_hub.api import HubApi

            api_kwargs = {"token": ""} if anonymous else {}
            api = HubApi(**api_kwargs)
            files = api.list_repo_files(
                remote_repo,
                repo_type,
                revision="master",
                recursive=True,
            )
    except DownloadMetadataUnavailable:
        raise
    except Exception as exc:
        raise DownloadMetadataUnavailable(
            f"Unable to verify remote repository size for {remote_repo!r}"
        ) from exc

    if not files:
        raise DownloadMetadataUnavailable(
            f"Unable to verify remote repository size for {remote_repo!r}"
        )

    if download_source == "modelscope":
        try:
            total_size = sum(_file_size(item) for item in files)
        except DownloadMetadataUnavailable:
            # modelscope_hub.list_repo_files 通常只返回文件路径字符串（无 size
            # 元数据），无法做 size 预检；跳过求和，由下载完成后的
            # enforce_download_size 兜底，避免 ModelScope 下载因此恒 502。
            # None keeps the caller's full-limit reservation; 0 would let
            # concurrent unknown-size downloads bypass admission accounting.
            # 若 SDK/版本返回了带 size 的对象则正常预检。
            return None
        enforce_download_size(total_size, max_bytes, remote_repo=remote_repo)
        return total_size

    try:
        total_size = sum(_file_size(item) for item in files)
    except DownloadMetadataUnavailable as exc:
        raise DownloadMetadataUnavailable(
            f"Unable to verify remote repository size for {remote_repo!r}"
        ) from exc
    enforce_download_size(total_size, max_bytes, remote_repo=remote_repo)
    return total_size
