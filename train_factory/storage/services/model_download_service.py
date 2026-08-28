"""
Model download service.
Supports downloading models from ModelScope and HuggingFace.
"""

import os
import threading
import logging
from typing import Optional, Dict, Any
from train_factory.core.time_utils import now_naive
from uuid import uuid4

from ..database import get_session
from ..entities.model_registry_entity import ModelRegistryDB
from ...utils.path_utils import hash_path
from ...config import settings
from ...core.remote_download_security import (
    calculate_directory_size_strict,
    DownloadLease,
    download_concurrency_limiter,
    enforce_download_size,
    preflight_remote_repo_size,
    reclaim_download_directory,
    resolve_managed_artifact_directory,
    validate_remote_repo_id,
)
from .download_storage_quota_service import (
    DownloadStorageReservation,
    download_storage_quota_service,
)
from .model_registry_service import ModelDeletionInProgressError
from .model_artifact_membership_service import lock_model_artifact_membership

logger = logging.getLogger(__name__)


def _extract_repo_basename(remote_repo: str) -> str:
    """Extract repo name from remote_repo string.

    Handles common patterns:
    - "org/repo"
    - "org/repo/resolve/main" (HF)
    - URLs like "https://huggingface.co/org/repo"
    - ModelScope URLs
    """
    import re

    s = (remote_repo or "").strip()
    # Strip URL schemes
    s = re.sub(r"^[a-zA-Z0-9+.-]+://", "", s)
    # Remove known domains/prefixes
    for prefix in (
        "huggingface.co/",
        "hf.co/",
        "modelscope.cn/",
        "www.modelscope.cn/",
        "modelscope/",
        "huggingface:",
        "modelscope:",
        "models/",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    # Split and pick org/repo
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[-1]
    return "model"


class ModelDownloadService:
    """Model download service for remote repositories."""

    def __init__(self):
        self._download_threads: Dict[str, threading.Thread] = {}
        self._download_progress: Dict[str, Dict[str, Any]] = {}

    def start_download(
        self,
        download_source: str,
        remote_repo: str,
        model_type: str = "embedding",
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a model download task.

        Args:
            download_source: Download source (modelscope/huggingface)
            remote_repo: Remote repository address
            model_type: Model type (embedding/reranker)
            display_name: Display name
            description: Description
            user_id: User ID

        Returns:
            Dict containing download task info
        """
        if settings.auth_enabled:
            remote_repo = validate_remote_repo_id(remote_repo)

        if download_source not in {"huggingface", "modelscope"}:
            raise ValueError(f"Unsupported download source: {download_source}")

        lease = download_concurrency_limiter.acquire(
            user_id,
            global_limit=settings.download_max_concurrent_global,
            per_user_limit=settings.download_max_concurrent_per_user,
        )
        registry_id = None
        model_path = None
        storage_reservation = None
        background_started = False
        try:
            expected_bytes = settings.model_download_max_bytes
            should_preflight = settings.auth_enabled
            if not should_preflight:
                try:
                    validate_remote_repo_id(remote_repo)
                    should_preflight = True
                except ValueError:
                    pass
            if should_preflight:
                metadata_bytes = preflight_remote_repo_size(
                    download_source=download_source,
                    repo_type="model",
                    remote_repo=remote_repo,
                    max_bytes=settings.model_download_max_bytes,
                    anonymous=settings.auth_enabled,
                    timeout_seconds=settings.download_metadata_timeout_seconds,
                )
                if metadata_bytes is not None:
                    expected_bytes = metadata_bytes

            storage_reservation = download_storage_quota_service.reserve(
                user_id,
                expected_bytes=expected_bytes,
                global_limit=settings.download_storage_max_bytes_global,
                per_user_limit=settings.download_storage_max_bytes_per_user,
            )

            # Generate registry ID
            registry_id = str(uuid4())

            # Extract model name
            model_name = _extract_repo_basename(remote_repo)

            # Generate display_name if not provided
            if not display_name:
                timestamp = now_naive().strftime("%Y%m%d_%H%M%S")
                display_name = f"{model_name}_download_{timestamp}"

            # Set download path
            model_path = str(settings.models_dir / registry_id)
            os.makedirs(model_path, exist_ok=True)

            # Create model registry record
            with get_session() as session:
                lock_model_artifact_membership(session)
                model = ModelRegistryDB(
                    model_id=registry_id,
                    model_name=model_name,
                    display_name=display_name,
                    version="v1.0.0",
                    model_type=model_type,
                    model_path=model_path,
                    model_path_hash=hash_path(model_path),
                    base_model_path=remote_repo,
                    description=description
                    or f"Downloaded from {download_source}: {remote_repo}",
                    status="registered",
                    is_latest=True,
                    user_id=user_id,
                    # Download-specific fields
                    source_type="downloaded",
                    download_source=download_source,
                    remote_repo=remote_repo,
                    download_status="pending",
                    download_progress=0,
                )
                session.add(model)
                session.commit()
                session.refresh(model)

                # Initialize progress tracking
                self._download_progress[registry_id] = {
                    "registry_id": registry_id,
                    "user_id": user_id,
                    "model_name": model_name,
                    "display_name": display_name,
                    "download_source": download_source,
                    "remote_repo": remote_repo,
                    "model_path": model_path,
                    "status": "downloading",
                    "progress": 0,
                    "error": None,
                    "created_at": now_naive().isoformat(),
                }

                # Start background download thread
                download_thread = threading.Thread(
                    target=self._download_model_background,
                    args=(registry_id, download_source, remote_repo, model_path),
                    kwargs={
                        "lease": lease,
                        "storage_reservation": storage_reservation,
                        "max_bytes": settings.model_download_max_bytes,
                    },
                    daemon=True,
                )
                self._download_threads[registry_id] = download_thread
                download_thread.start()
                background_started = True

                logger.info(
                    f"Model download task started: registry_id={registry_id}, source={download_source}, repo={remote_repo}"
                )

                return {
                    "registry_id": registry_id,
                    "model_name": model_name,
                    "display_name": display_name,
                    "status": "downloading",
                    "model_path": model_path,
                }
        except Exception:
            if not background_started:
                storage_removed = True
                residual_bytes = None
                if model_path is not None:
                    storage_removed, residual_bytes = reclaim_download_directory(
                        model_path,
                        fallback_bytes=settings.model_download_max_bytes,
                    )
                    if not storage_removed:
                        logger.error(
                            "Failed to roll back model directory %s; "
                            "residual storage remains billable",
                            registry_id,
                        )
                if registry_id is not None:
                    self._download_threads.pop(registry_id, None)
                    self._download_progress.pop(registry_id, None)
                    try:
                        if storage_removed:
                            self._delete_pending_record(registry_id)
                        else:
                            self._update_model_record(
                                registry_id,
                                download_status="failed",
                                file_size=residual_bytes,
                                download_error=(
                                    "Download worker failed to start; "
                                    "residual storage cleanup failed"
                                ),
                            )
                    except Exception as cleanup_error:
                        logger.error(
                            "Failed to roll back pending model row %s (error_type=%s)",
                            registry_id,
                            type(cleanup_error).__name__,
                        )
                if storage_reservation is not None:
                    try:
                        storage_reservation.release()
                    except Exception as cleanup_error:
                        logger.error(
                            "Failed to release model storage reservation (error_type=%s)",
                            type(cleanup_error).__name__,
                        )
                try:
                    lease.release()
                except Exception as cleanup_error:
                    logger.error(
                        "Failed to release model download lease (error_type=%s)",
                        type(cleanup_error).__name__,
                    )
            raise

    @staticmethod
    def _delete_pending_record(registry_id: str) -> None:
        """Remove a committed row when its worker thread never started."""
        with get_session() as session:
            from sqlmodel import select

            model = session.exec(
                select(ModelRegistryDB)
                .where(ModelRegistryDB.model_id == registry_id)
                .with_for_update()
            ).first()
            if model and model.status == "deleting":
                return
            if model and model.download_status == "pending":
                session.delete(model)
                session.commit()

    def _download_model_background(
        self,
        registry_id: str,
        download_source: str,
        remote_repo: str,
        model_path: str,
        lease: Optional[DownloadLease] = None,
        storage_reservation: Optional[DownloadStorageReservation] = None,
        max_bytes: Optional[int] = None,
    ):
        """Background model download."""
        try:
            # Update status to downloading
            self._update_download_status(registry_id, "downloading", 0)

            # 下载中监控：snapshot_download 无法中途取消，预检大小元数据也可能
            # 与实际不一致（TOCTOU）。用守护线程定期测量目录大小，一旦超限
            # 记录错误，下载完成后统一走 enforce + reclaim 清理路径。
            limit_bytes = max_bytes if max_bytes is not None else settings.model_download_max_bytes
            stop_monitor = threading.Event()
            monitor_error: list = []

            def _monitor_download_size() -> None:
                while not stop_monitor.is_set():
                    try:
                        current = self._calculate_model_size(model_path)
                        enforce_download_size(
                            current,
                            limit_bytes,
                            remote_repo=remote_repo,
                        )
                    except Exception as exc:
                        monitor_error.append(exc)
                        stop_monitor.set()
                        return
                    stop_monitor.wait(10)

            monitor = threading.Thread(target=_monitor_download_size, daemon=True)
            monitor.start()
            try:
                if download_source == "modelscope":
                    self._download_from_modelscope(registry_id, remote_repo, model_path)
                elif download_source == "huggingface":
                    self._download_from_huggingface(registry_id, remote_repo, model_path)
                else:
                    raise ValueError(f"Unsupported download source: {download_source}")
            finally:
                stop_monitor.set()

            # 下载中检测到超限：抛出，走统一清理路径
            if monitor_error:
                raise monitor_error[0]

            # Calculate file size
            file_size = self._calculate_model_size(model_path)
            enforce_download_size(
                file_size,
                max_bytes or settings.model_download_max_bytes,
                remote_repo=remote_repo,
            )
            if storage_reservation is not None:
                storage_reservation.verify_actual_size(file_size)

            # Update status to completed
            self._update_download_status(registry_id, "available", 100)
            self._update_model_record(
                registry_id,
                status="available",
                file_size=file_size,
                download_progress=100,
            )

            logger.info(
                f"Model download completed: registry_id={registry_id}, size={file_size} bytes"
            )

        except ModelDeletionInProgressError:
            logger.info(
                "Download worker yielded to model deletion for %s; "
                "downloaded artifacts were left for the delete owner",
                registry_id,
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Model download failed: registry_id={registry_id}, error={error_msg}"
            )
            storage_removed, residual_bytes = reclaim_download_directory(
                model_path,
                fallback_bytes=(
                    max_bytes
                    if max_bytes is not None
                    else settings.model_download_max_bytes
                ),
            )
            if not storage_removed:
                error_msg = f"{error_msg}; residual storage cleanup failed"
                logger.warning(
                    "Failed to clean partial model download %s; %s bytes remain",
                    registry_id,
                    residual_bytes,
                )
            self._update_download_status(registry_id, "failed", 0, error_msg)
            self._update_model_record(
                registry_id,
                download_error=error_msg,
                download_status="failed",
                file_size=residual_bytes,
            )

        finally:
            # Clean up thread record
            self._download_threads.pop(registry_id, None)
            try:
                if lease is not None:
                    lease.release()
            finally:
                if storage_reservation is not None:
                    storage_reservation.release()

    def _download_from_modelscope(
        self, registry_id: str, remote_repo: str, model_path: str
    ):
        """Download from ModelScope."""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            raise ImportError("Please install modelscope: pip install modelscope")

        logger.info(f"Starting download from ModelScope: {remote_repo} -> {model_path}")

        # Update progress
        self._update_download_status(registry_id, "downloading", 10)

        download_kwargs = dict(
            model_id=remote_repo,
            local_dir=model_path,
            revision="master",
        )
        if settings.auth_enabled:
            download_kwargs["token"] = ""
        download_kwargs["max_workers"] = settings.download_max_workers
        downloaded_path = snapshot_download(**download_kwargs)

        self._update_download_status(registry_id, "downloading", 90)
        logger.info(
            f"ModelScope download completed: {remote_repo} -> {downloaded_path}"
        )

    def _download_from_huggingface(
        self, registry_id: str, remote_repo: str, model_path: str
    ):
        """Download from HuggingFace."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "Please install huggingface_hub: pip install huggingface_hub"
            )

        logger.info(
            f"Starting download from HuggingFace: {remote_repo} -> {model_path}"
        )

        # Update progress incrementally
        for progress in [10, 30, 50, 70]:
            self._update_download_status(registry_id, "downloading", progress)

        download_kwargs = dict(
            repo_id=remote_repo,
            local_dir=model_path,
            resume_download=True,
            max_workers=settings.download_max_workers,
        )
        if settings.auth_enabled:
            download_kwargs["token"] = False
        downloaded_path = snapshot_download(**download_kwargs)

        self._update_download_status(registry_id, "downloading", 90)
        logger.info(
            f"HuggingFace download completed: {remote_repo} -> {downloaded_path}"
        )

    def _calculate_model_size(self, model_path: str) -> int:
        """Calculate total model size in bytes."""
        return calculate_directory_size_strict(model_path)

    def _update_download_status(
        self,
        registry_id: str,
        status: str,
        progress: int,
        error: Optional[str] = None,
    ):
        """Update download progress tracking in memory and database."""
        # Persist first so a model delete intent cannot be masked by a stale
        # in-memory progress update.
        self._update_model_record(
            registry_id,
            status="available" if status == "available" else None,
            download_status="completed" if status == "available" else status,
            download_progress=progress,
            download_error=error if error else None,
        )
        if registry_id in self._download_progress:
            self._download_progress[registry_id]["status"] = status
            self._download_progress[registry_id]["progress"] = progress
            if error:
                self._download_progress[registry_id]["error"] = error

    def _update_model_record(
        self,
        registry_id: str,
        status: Optional[str] = None,
        download_status: Optional[str] = None,
        file_size: Optional[int] = None,
        download_progress: Optional[int] = None,
        download_error: Optional[str] = None,
    ):
        """Update model record in database."""
        with get_session() as session:
            from sqlmodel import select

            statement = (
                select(ModelRegistryDB)
                .where(ModelRegistryDB.model_id == registry_id)
                .with_for_update()
            )
            model = session.exec(statement).first()
            if model:
                if model.status == "deleting":
                    raise ModelDeletionInProgressError(
                        f"Model is being deleted: {registry_id}"
                    )
                if status:
                    model.update_status(status)
                    # Backward compatibility: when caller doesn't explicitly set download_status.
                    if download_status is None and status == "available":
                        model.download_status = "completed"
                if download_status is not None:
                    model.download_status = download_status
                if file_size is not None:
                    model.file_size = file_size
                if download_progress is not None:
                    model.download_progress = download_progress
                if download_error:
                    model.download_error = download_error
                model.updated_at = now_naive()
                session.add(model)
                session.commit()

    def cleanup_interrupted_downloads(self) -> int:
        """Fail restart-orphaned downloads and reclaim their managed bytes."""
        from sqlmodel import select

        message = "Download interrupted by server restart"
        with get_session() as session:
            models = session.exec(
                select(ModelRegistryDB).where(
                    ModelRegistryDB.source_type == "downloaded",
                    ModelRegistryDB.status != "deleting",
                    ModelRegistryDB.download_status.in_(["pending", "downloading"]),
                )
            ).all()
            for model in models:
                error_message = message
                storage_removed = False
                residual_bytes = settings.model_download_max_bytes
                try:
                    model_path = resolve_managed_artifact_directory(
                        settings.models_dir,
                        model.model_id,
                        model.model_path,
                    )
                    storage_removed, measured_residual = reclaim_download_directory(
                        model_path,
                        fallback_bytes=settings.model_download_max_bytes,
                    )
                    if measured_residual is not None:
                        residual_bytes = measured_residual
                except (ValueError, TypeError):
                    storage_removed = False
                if not storage_removed:
                    error_message = f"{message}; residual storage cleanup failed"
                model.download_status = "failed"
                model.download_progress = 0
                model.download_error = error_message
                if storage_removed:
                    model.file_size = None
                else:
                    model.file_size = residual_bytes
                model.updated_at = now_naive()
                session.add(model)
                self._download_threads.pop(model.model_id, None)
                self._download_progress.pop(model.model_id, None)
            if models:
                session.commit()
        return len(models)

    def get_download_progress(self, registry_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress from memory or database."""
        # First try memory cache
        if registry_id in self._download_progress:
            return self._download_progress.get(registry_id)

        # Fallback to database (for when service was restarted)
        with get_session() as session:
            from sqlmodel import select

            statement = select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == registry_id
            )
            model = session.exec(statement).first()
            if model and model.source_type == "downloaded":
                progress_status = model.status
                if model.download_status == "completed":
                    progress_status = "available"
                elif model.download_status in {"pending", "downloading", "failed"}:
                    progress_status = model.download_status
                return {
                    "registry_id": registry_id,
                    "model_name": model.model_name,
                    "display_name": model.display_name,
                    "download_source": model.download_source,
                    "remote_repo": model.remote_repo,
                    "model_path": model.model_path,
                    "status": progress_status,
                    "progress": model.download_progress or 0,
                    "error": model.download_error,
                    "created_at": model.created_at.isoformat()
                    if model.created_at
                    else None,
                }
        return None

    def is_downloading(self, registry_id: str) -> bool:
        """Check if model is being downloaded."""
        return (
            registry_id in self._download_threads
            and self._download_threads[registry_id].is_alive()
        )

    def list_downloads(self, user_id: Optional[str] = None) -> list:
        """List download tasks, optionally filtered by owner.

        When ``user_id`` is falsy (auth disabled / anonymous) all tasks are
        returned; otherwise only tasks owned by that user are returned so one
        tenant cannot enumerate another tenant's downloads.
        """
        downloads = list(self._download_progress.values())
        if user_id:
            downloads = [d for d in downloads if d.get("user_id") == user_id]
        return downloads


# Singleton instance
_model_download_service = None


def get_model_download_service() -> ModelDownloadService:
    """Get model download service singleton."""
    global _model_download_service
    if _model_download_service is None:
        _model_download_service = ModelDownloadService()
    return _model_download_service


# Export singleton
model_download_service = get_model_download_service()
