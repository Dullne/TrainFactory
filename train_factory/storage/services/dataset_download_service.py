"""
Dataset download service.
Supports downloading datasets from HuggingFace and ModelScope.
"""

import json
import os
import shutil
import threading
import logging
from typing import Optional, Dict, Any, List
from train_factory.core.time_utils import now_naive
from uuid import uuid4

from ..database import get_session
from ..entities.dataset_entity import DatasetDB
from ...utils.path_utils import hash_path
from ...config import settings
from ...core.remote_download_security import (
    calculate_directory_size_strict,
    DownloadPolicyViolation,
    DownloadStorageQuotaExceeded,
    DownloadLease,
    download_concurrency_limiter,
    enforce_download_size,
    preflight_remote_repo_size,
    reclaim_download_directory,
    resolve_managed_artifact_directory,
    validate_remote_artifact_name,
    validate_remote_repo_id,
)
from .download_storage_quota_service import (
    DownloadStorageReservation,
    download_storage_quota_service,
)

logger = logging.getLogger(__name__)


class DatasetDownloadService:
    """Dataset download service for remote repositories."""

    def __init__(self):
        self._download_threads: Dict[str, threading.Thread] = {}
        self._download_progress: Dict[str, Dict[str, Any]] = {}

    def start_download(
        self,
        source_type: str,
        remote_repo: str,
        dataset_name: str,
        dataset_type: str = "custom",
        usage: str = "train",
        model_type: Optional[List[str]] = None,
        hf_subset: Optional[str] = None,
        output_format: str = "jsonl",
        clean_cache: bool = True,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a dataset download task.

        Args:
            source_type: Download source (huggingface/modelscope)
            remote_repo: Remote repository address
            dataset_name: Name for the dataset
            dataset_type: Dataset type
            usage: Dataset usage (train/eval/test)
            model_type: Applicable model tags (e.g. ["embedding", "rerank"])
            hf_subset: HuggingFace subset name
            output_format: Output format (parquet/jsonl/json/arrow), default parquet
            clean_cache: Whether to clean HuggingFace cache after download, default True
            description: Description
            user_id: User ID

        Returns:
            Dict containing download task info
        """
        # 无条件强制 org/repo 形态：无论 auth 是否开启都拒绝把用户输入当
        # 本地路径/URL 解析（否则 legacy 模式可读服务器任意文件）。
        remote_repo = validate_remote_repo_id(remote_repo)
        if hf_subset is not None:
            # subset 会进入缓存目录路径构造：拒绝 "../" 等逃逸组件
            hf_subset = validate_remote_artifact_name(hf_subset, kind="subset")

        if source_type not in {"huggingface", "modelscope"}:
            raise ValueError(f"Unsupported source type: {source_type}")
        if settings.auth_enabled and source_type == "modelscope":
            raise DownloadPolicyViolation(
                "ModelScope dataset downloads are disabled while authentication "
                "is enabled because the SDK cannot guarantee anonymous isolation"
            )
        if settings.auth_enabled and output_format != "jsonl":
            raise DownloadPolicyViolation(
                "Authenticated remote dataset downloads only support streaming JSONL"
            )

        lease = download_concurrency_limiter.acquire(
            user_id,
            global_limit=settings.download_max_concurrent_global,
            per_user_limit=settings.download_max_concurrent_per_user,
        )
        dataset_id = None
        storage_path = None
        storage_reservation = None
        background_started = False
        try:
            expected_bytes = settings.dataset_download_max_bytes
            should_preflight = True
            if should_preflight:
                metadata_bytes = preflight_remote_repo_size(
                    download_source=source_type,
                    repo_type="dataset",
                    remote_repo=remote_repo,
                    max_bytes=settings.dataset_download_max_bytes,
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

            # Generate dataset ID
            dataset_id = str(uuid4())

            # Set download path
            storage_path = str(settings.datasets_dir / dataset_id)
            os.makedirs(storage_path, exist_ok=True)

            # Create dataset record
            with get_session() as session:
                dataset = DatasetDB(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    display_name=dataset_name,
                    storage_path=storage_path,
                    storage_path_hash=hash_path(storage_path),
                    dataset_type=dataset_type,
                    usage=usage,
                    model_type=model_type,
                    file_format=output_format,
                    source_type=source_type,
                    remote_repo=remote_repo,
                    hf_subset=hf_subset,
                    description=description
                    or f"Downloaded from {source_type}: {remote_repo}",
                    status="downloading",
                    user_id=user_id,
                )
                session.add(dataset)
                session.commit()
                session.refresh(dataset)

                # Initialize progress tracking
                self._download_progress[dataset_id] = {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "remote_repo": remote_repo,
                    "storage_path": storage_path,
                    "status": "downloading",
                    "progress": 0,
                    "error": None,
                    "created_at": now_naive().isoformat(),
                    "user_id": user_id,
                }

                # Start background download thread
                download_thread = threading.Thread(
                    target=self._download_dataset_background,
                    args=(
                        dataset_id,
                        source_type,
                        remote_repo,
                        storage_path,
                        hf_subset,
                        output_format,
                        clean_cache,
                    ),
                    kwargs={
                        "lease": lease,
                        "storage_reservation": storage_reservation,
                        "max_bytes": settings.dataset_download_max_bytes,
                    },
                    daemon=True,
                )
                self._download_threads[dataset_id] = download_thread
                download_thread.start()
                background_started = True

                logger.info(
                    f"Dataset download task started: dataset_id={dataset_id}, source={source_type}, repo={remote_repo}"
                )

                return {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "status": "downloading",
                    "storage_path": storage_path,
                }
        except Exception:
            if not background_started:
                storage_removed = True
                residual_bytes = None
                if storage_path is not None:
                    storage_removed, residual_bytes = reclaim_download_directory(
                        storage_path,
                        fallback_bytes=settings.dataset_download_max_bytes,
                    )
                    if not storage_removed:
                        logger.error(
                            "Failed to roll back dataset directory %s; "
                            "residual storage remains billable",
                            dataset_id,
                        )
                if dataset_id is not None:
                    self._download_threads.pop(dataset_id, None)
                    self._download_progress.pop(dataset_id, None)
                    try:
                        if storage_removed:
                            self._delete_pending_record(dataset_id)
                        else:
                            self._update_dataset_record(
                                dataset_id,
                                status="error",
                                file_size=residual_bytes,
                                error_message=(
                                    "Download worker failed to start; "
                                    "residual storage cleanup failed"
                                ),
                            )
                    except Exception as cleanup_error:
                        logger.error(
                            "Failed to roll back pending dataset row %s (error_type=%s)",
                            dataset_id,
                            type(cleanup_error).__name__,
                        )
                if storage_reservation is not None:
                    try:
                        storage_reservation.release()
                    except Exception as cleanup_error:
                        logger.error(
                            "Failed to release dataset storage reservation (error_type=%s)",
                            type(cleanup_error).__name__,
                        )
                try:
                    lease.release()
                except Exception as cleanup_error:
                    logger.error(
                        "Failed to release dataset download lease (error_type=%s)",
                        type(cleanup_error).__name__,
                    )
            raise

    @staticmethod
    def _delete_pending_record(dataset_id: str) -> None:
        """Remove a committed row when its worker thread never started."""
        with get_session() as session:
            from sqlmodel import select

            dataset = session.exec(
                select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
            ).first()
            if dataset and dataset.status == "downloading":
                session.delete(dataset)
                session.commit()

    def _download_dataset_background(
        self,
        dataset_id: str,
        source_type: str,
        remote_repo: str,
        storage_path: str,
        hf_subset: Optional[str] = None,
        output_format: str = "jsonl",
        clean_cache: bool = True,
        lease: Optional[DownloadLease] = None,
        storage_reservation: Optional[DownloadStorageReservation] = None,
        max_bytes: Optional[int] = None,
    ):
        """Background dataset download."""
        try:
            # Update status to downloading
            self._update_download_status(dataset_id, "downloading", 0)

            if source_type == "huggingface":
                num_rows, file_size = self._download_from_huggingface(
                    dataset_id,
                    remote_repo,
                    storage_path,
                    hf_subset,
                    output_format,
                    clean_cache,
                    max_bytes=max_bytes,
                    storage_reservation=storage_reservation,
                )
            elif source_type == "modelscope":
                num_rows, file_size = self._download_from_modelscope(
                    dataset_id,
                    remote_repo,
                    storage_path,
                    output_format,
                    clean_cache,
                    max_bytes=max_bytes,
                )
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            enforce_download_size(
                file_size,
                max_bytes
                if max_bytes is not None
                else settings.dataset_download_max_bytes,
                remote_repo=remote_repo,
            )
            if storage_reservation is not None:
                storage_reservation.verify_actual_size(file_size)

            # Update status to completed
            self._update_download_status(dataset_id, "ready", 100)
            self._update_dataset_record(
                dataset_id, status="ready", num_rows=num_rows, file_size=file_size
            )

            logger.info(
                f"Dataset download completed: dataset_id={dataset_id}, rows={num_rows}"
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Dataset download failed: dataset_id={dataset_id}, error={error_msg}"
            )
            storage_removed, residual_bytes = reclaim_download_directory(
                storage_path,
                fallback_bytes=(
                    max_bytes
                    if max_bytes is not None
                    else settings.dataset_download_max_bytes
                ),
            )
            if not storage_removed:
                error_msg = f"{error_msg}; residual storage cleanup failed"
                logger.warning(
                    "Failed to clean partial dataset download %s; %s bytes remain",
                    dataset_id,
                    residual_bytes,
                )
            self._update_download_status(dataset_id, "failed", 0, error_msg)
            self._update_dataset_record(
                dataset_id,
                status="error",
                file_size=residual_bytes,
                error_message=error_msg,
            )

        finally:
            # Clean up thread record
            self._download_threads.pop(dataset_id, None)
            try:
                if lease is not None:
                    lease.release()
            finally:
                if storage_reservation is not None:
                    storage_reservation.release()

    def _download_from_huggingface(
        self,
        dataset_id: str,
        remote_repo: str,
        storage_path: str,
        hf_subset: Optional[str] = None,
        output_format: str = "jsonl",
        clean_cache: bool = True,
        max_bytes: Optional[int] = None,
        storage_reservation: Optional[DownloadStorageReservation] = None,
    ) -> tuple:
        """Download from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        logger.info(
            f"Starting download from HuggingFace: {remote_repo} -> {storage_path} (format={output_format})"
        )
        self._update_download_status(dataset_id, "downloading", 10)

        load_kwargs = {}
        secure_cache_dir = os.path.join(storage_path, ".download-cache")
        if settings.auth_enabled:
            load_kwargs.update(
                token=False,
                trust_remote_code=False,
                cache_dir=secure_cache_dir,
                streaming=True,
            )
        else:
            # legacy（auth 关闭）模式：仍禁止远端代码执行（安全关键），
            # 但不强制匿名 token（保持原有调用形态）
            load_kwargs.update(trust_remote_code=False)

        try:
            # Load dataset
            if hf_subset:
                dataset = load_dataset(remote_repo, hf_subset, **load_kwargs)
            else:
                dataset = load_dataset(remote_repo, **load_kwargs)

            self._update_download_status(dataset_id, "downloading", 50)

            # Save to specified format
            num_rows = 0
            for split_name, split_data in dataset.items():
                if settings.auth_enabled:
                    split_rows = self._save_streaming_jsonl_split(
                        split_data,
                        storage_path,
                        split_name,
                        remote_repo=remote_repo,
                        max_bytes=max_bytes
                        if max_bytes is not None
                        else settings.dataset_download_max_bytes,
                        storage_reservation=storage_reservation,
                    )
                else:
                    self._save_split(
                        split_data,
                        storage_path,
                        split_name,
                        output_format,
                    )
                    split_rows = len(split_data)
                num_rows += split_rows
                logger.info(
                    "Saved %s split: %s rows (%s)",
                    split_name,
                    split_rows,
                    output_format,
                )

            self._update_download_status(dataset_id, "downloading", 90)

            # Count the per-task cache before deleting it so compressed Hub
            # payloads cannot bypass the final on-disk size check.
            staged_size = self._calculate_size(storage_path)
            enforce_download_size(
                staged_size,
                max_bytes
                if max_bytes is not None
                else settings.dataset_download_max_bytes,
                remote_repo=remote_repo,
            )
        finally:
            if settings.auth_enabled:
                shutil.rmtree(secure_cache_dir, ignore_errors=True)

        file_size = self._calculate_size(storage_path)

        # Clean HuggingFace cache if requested
        if clean_cache and not settings.auth_enabled:
            self._clean_hf_cache(remote_repo, hf_subset)

        logger.info(
            f"HuggingFace download completed: {remote_repo}, total rows: {num_rows}"
        )
        return num_rows, file_size

    def _download_from_modelscope(
        self,
        dataset_id: str,
        remote_repo: str,
        storage_path: str,
        output_format: str = "jsonl",
        clean_cache: bool = True,
        max_bytes: Optional[int] = None,
    ) -> tuple:
        """Download from ModelScope."""
        try:
            from modelscope.msdatasets import MsDataset
        except ImportError:
            raise ImportError("Please install modelscope: pip install modelscope")

        logger.info(
            f"Starting download from ModelScope: {remote_repo} -> {storage_path} (format={output_format})"
        )
        self._update_download_status(dataset_id, "downloading", 10)

        secure_cache_dir = os.path.join(storage_path, ".download-cache")
        load_kwargs = {}
        if settings.auth_enabled:
            load_kwargs.update(
                token="",
                trust_remote_code=False,
                cache_dir=secure_cache_dir,
            )

        try:
            # Load dataset
            dataset = MsDataset.load(remote_repo, **load_kwargs)
            self._update_download_status(dataset_id, "downloading", 50)

            # Convert to HuggingFace dataset and save
            hf_dataset = dataset.to_hf_dataset()

            num_rows = 0
            splits = (
                hf_dataset.items()
                if hasattr(hf_dataset, "items")
                else [("train", hf_dataset)]
            )
            for split_name, split_data in splits:
                self._save_split(split_data, storage_path, split_name, output_format)
                num_rows += len(split_data)

            self._update_download_status(dataset_id, "downloading", 90)

            staged_size = self._calculate_size(storage_path)
            enforce_download_size(
                staged_size,
                max_bytes
                if max_bytes is not None
                else settings.dataset_download_max_bytes,
                remote_repo=remote_repo,
            )
        finally:
            if settings.auth_enabled:
                shutil.rmtree(secure_cache_dir, ignore_errors=True)

        file_size = self._calculate_size(storage_path)

        logger.info(
            f"ModelScope download completed: {remote_repo}, total rows: {num_rows}"
        )
        return num_rows, file_size

    @staticmethod
    def _save_split(split_data, storage_path: str, split_name: str, output_format: str):
        """Save a dataset split in the specified format."""
        split_name = validate_remote_artifact_name(split_name, kind="dataset split")
        if output_format == "jsonl":
            output_file = os.path.join(storage_path, f"{split_name}.jsonl")
            split_data.to_json(output_file, orient="records", lines=True)
        elif output_format == "parquet":
            output_file = os.path.join(storage_path, f"{split_name}.parquet")
            split_data.to_parquet(output_file)
        elif output_format == "json":
            output_file = os.path.join(storage_path, f"{split_name}.json")
            split_data.to_json(output_file, orient="records", lines=False)
        elif output_format == "arrow":
            split_data.save_to_disk(os.path.join(storage_path, split_name))
        else:
            output_file = os.path.join(storage_path, f"{split_name}.jsonl")
            split_data.to_json(output_file, orient="records", lines=True)

    @staticmethod
    def _save_streaming_jsonl_split(
        split_data,
        storage_path: str,
        split_name: str,
        *,
        remote_repo: str,
        max_bytes: int,
        storage_reservation: Optional[DownloadStorageReservation] = None,
    ) -> int:
        """Write an iterable split incrementally with a pre-write disk guard."""
        split_name = validate_remote_artifact_name(split_name, kind="dataset split")
        output_file = os.path.join(storage_path, f"{split_name}.jsonl")
        row_count = 0
        logical_output_bytes = 0
        with open(output_file, "wb") as handle:
            for row in split_data:
                encoded = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                current_bytes = calculate_directory_size_strict(storage_path)
                on_disk_output_bytes = os.fstat(handle.fileno()).st_size
                projected_bytes = (
                    current_bytes
                    - on_disk_output_bytes
                    + logical_output_bytes
                    + len(encoded)
                )
                enforce_download_size(
                    projected_bytes,
                    max_bytes,
                    remote_repo=remote_repo,
                )
                if (
                    storage_reservation is not None
                    and projected_bytes > storage_reservation.reserved_bytes
                ):
                    reserve_target = min(
                        max_bytes,
                        max(
                            projected_bytes,
                            storage_reservation.reserved_bytes + 8 * 1024**2,
                        ),
                    )
                    try:
                        storage_reservation.verify_actual_size(reserve_target)
                    except DownloadStorageQuotaExceeded:
                        if reserve_target == projected_bytes:
                            raise
                        storage_reservation.verify_actual_size(projected_bytes)
                handle.write(encoded)
                logical_output_bytes += len(encoded)
                row_count += 1
        return row_count

    def _clean_hf_cache(self, remote_repo: str, hf_subset: Optional[str] = None):
        """Clean HuggingFace datasets cache for the given repo."""
        try:
            import shutil

            cache_base = os.path.expanduser("~/.cache/huggingface/datasets")
            # HF cache dir name: org___dataset_name (slashes replaced with ___)
            cache_name = remote_repo.replace("/", "___").replace("-", "_").lower()
            cache_dir = os.path.join(cache_base, cache_name)

            # Try exact match first, then glob
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
                logger.info(f"Cleaned HuggingFace cache: {cache_dir}")
            else:
                # Try glob match (HF may use different naming)
                import glob

                matches = glob.glob(os.path.join(cache_base, f"*{cache_name}*"))
                for match in matches:
                    if os.path.isdir(match):
                        shutil.rmtree(match)
                        logger.info(f"Cleaned HuggingFace cache: {match}")
        except Exception as e:
            logger.warning(f"Failed to clean HuggingFace cache for {remote_repo}: {e}")

    def _calculate_size(self, path: str) -> int:
        """Calculate total size in bytes."""
        return calculate_directory_size_strict(path)

    def _update_download_status(
        self,
        dataset_id: str,
        status: str,
        progress: int,
        error: Optional[str] = None,
    ):
        """Update download progress tracking."""
        if dataset_id in self._download_progress:
            self._download_progress[dataset_id]["status"] = status
            self._download_progress[dataset_id]["progress"] = progress
            if error:
                self._download_progress[dataset_id]["error"] = error

    def _update_dataset_record(
        self,
        dataset_id: str,
        status: Optional[str] = None,
        num_rows: Optional[int] = None,
        file_size: Optional[int] = None,
        error_message: Optional[str] = None,
    ):
        """Update dataset record in database."""
        with get_session() as session:
            from sqlmodel import select

            statement = select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
            dataset = session.exec(statement).first()
            if dataset:
                if status:
                    dataset.update_status(status, error_message)
                if num_rows is not None:
                    dataset.num_rows = num_rows
                if file_size is not None:
                    dataset.file_size = file_size
                if error_message and not status:
                    dataset.error_message = error_message
                    dataset.updated_at = now_naive()
                elif not status:
                    dataset.updated_at = now_naive()
                session.add(dataset)
                session.commit()

    def cleanup_interrupted_downloads(self) -> int:
        """Fail restart-orphaned downloads and reclaim their managed bytes."""
        from sqlmodel import select

        message = "Download interrupted by server restart"
        with get_session() as session:
            datasets = session.exec(
                select(DatasetDB).where(
                    DatasetDB.source_type.in_(["huggingface", "modelscope"]),
                    DatasetDB.status.in_(["pending", "downloading"]),
                )
            ).all()
            for dataset in datasets:
                error_message = message
                storage_removed = False
                residual_bytes = settings.dataset_download_max_bytes
                try:
                    storage_path = resolve_managed_artifact_directory(
                        settings.datasets_dir,
                        dataset.dataset_id,
                        dataset.storage_path,
                    )
                    storage_removed, measured_residual = reclaim_download_directory(
                        storage_path,
                        fallback_bytes=settings.dataset_download_max_bytes,
                    )
                    if measured_residual is not None:
                        residual_bytes = measured_residual
                except (ValueError, TypeError):
                    storage_removed = False
                if not storage_removed:
                    error_message = f"{message}; residual storage cleanup failed"
                dataset.status = "error"
                dataset.error_message = error_message
                if storage_removed:
                    dataset.file_size = None
                else:
                    dataset.file_size = residual_bytes
                dataset.updated_at = now_naive()
                session.add(dataset)
                self._download_threads.pop(dataset.dataset_id, None)
                self._download_progress.pop(dataset.dataset_id, None)
            if datasets:
                session.commit()
        return len(datasets)

    def get_download_progress(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress."""
        return self._download_progress.get(dataset_id)

    def is_downloading(self, dataset_id: str) -> bool:
        """Check if dataset is being downloaded."""
        return (
            dataset_id in self._download_threads
            and self._download_threads[dataset_id].is_alive()
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
_dataset_download_service = None


def get_dataset_download_service() -> DatasetDownloadService:
    """Get dataset download service singleton."""
    global _dataset_download_service
    if _dataset_download_service is None:
        _dataset_download_service = DatasetDownloadService()
    return _dataset_download_service


# Export singleton
dataset_download_service = get_dataset_download_service()
