"""
Data Generation API routes.

Provides endpoints for dataset generation functionality.
"""

from dataclasses import dataclass
import logging
import os
import shutil
import stat
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field, model_validator

from ...auth.dependencies import get_current_user
from ...auth.resource_provenance import ResourceProvenanceError
from ...config.settings import get_settings
from ...evaluation.dataset_access import (
    UnsupportedEvaluationDatasetStorageError,
    resolve_managed_local_dataset,
)
from ...generation.pipeline import (
    MAX_GENERATION_BATCH_SIZE,
    MAX_GENERATION_EMBEDDING_CONCURRENCY,
    MAX_GENERATION_LLM_CONCURRENCY,
    MAX_GENERATION_LLM_ENDPOINTS,
    MAX_GENERATION_LLM_TOKENS,
    MAX_GENERATION_RERANK_CONCURRENCY,
    MAX_GENERATION_RETRIES,
    MAX_GENERATION_RETRIEVAL_TOP_K,
    MAX_GENERATION_TIMEOUT,
    DatasetGenerationPipeline,
    PipelineConfig,
    generation_attempt_key,
    resolve_generation_attempt_output_path,
    validate_generation_custom_prompts,
    validate_generation_resource_config,
    validate_generation_step_limits,
)
from ...generation.prompts import SOURCE_TYPES, LENGTH_TYPES
from ...generation.security import (
    get_generation_input_allowed_dirs,
    resolve_generation_input_path as _resolve_generation_input_path,
)
from ...storage.entities.generation_task_entity import GenerationStatus
from ...storage.services.model_config_service import model_config_service
from ...storage.services.outbound_endpoint_policy import validate_user_outbound_url
from ...storage.services.generation_task_service import generation_task_service
from ...storage.services.generation_publication_service import (
    generation_publication_service,
)
from ...storage.services.training_task_service import training_task_service
from ...storage.services.evaluation_task_service import evaluation_task_service
from ...storage.services.dataset_service import (
    DatasetConsumptionUnavailableError,
    DatasetDeletionOwnerConflictError,
    dataset_service,
)
from ...storage.services.milvus_collection_service import (
    MilvusCollectionUnavailableError,
    milvus_collection_service,
)
from ...storage.services.dataset_lineage_service import dataset_lineage_service
from ...storage.services.dataset_asset_service import dataset_asset_service
from ...storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    BackgroundTaskCapacityExceeded,
    background_task_admission_service,
)
from ...utils.public_diagnostics import public_task_error_message

logger = logging.getLogger(__name__)

router = APIRouter()

# 所有生成任务的输出统一存放目录（与数据集同一挂载点下）
# 容器内 /app/data/datasets -> 宿主机 项目/data/datasets
GENERATION_OUTPUT_DIR = os.environ.get("GENERATION_OUTPUT_DIR", "/app/data/datasets")


def _generation_attempt_directory(task_id: str, run_token: str) -> Path:
    """Resolve one exact, opaque attempt directory under the managed root."""
    root = Path(GENERATION_OUTPUT_DIR).resolve()
    output_path = resolve_generation_attempt_output_path(
        str(root / f"generated_{task_id}.jsonl"),
        task_id,
        run_token,
    )
    attempt_dir = output_path.parent.resolve()
    expected_parts = (
        f"generation_{task_id}",
        generation_attempt_key(task_id, run_token),
    )
    try:
        relative = attempt_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("Generation attempt directory escaped managed storage") from exc
    if relative.parts != expected_parts:
        raise ValueError("Generation attempt directory is not exactly scoped")
    return attempt_dir


def _cleanup_generation_attempt_directory(task_id: str, run_token: str) -> None:
    """Remove only the exact opaque directory owned by one attempt."""
    attempt_dir = _generation_attempt_directory(task_id, run_token)
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)


_FILE_ATTRIBUTE_REPARSE_POINT = getattr(
    stat,
    "FILE_ATTRIBUTE_REPARSE_POINT",
    0x400,
)


def _is_symlink_or_reparse(file_stat: Any) -> bool:
    return bool(
        stat.S_ISLNK(file_stat.st_mode)
        or (
            getattr(file_stat, "st_file_attributes", 0)
            & _FILE_ATTRIBUTE_REPARSE_POINT
        )
    )


def _same_file_identity(left: Any, right: Any) -> bool:
    try:
        return os.path.samestat(left, right)
    except (AttributeError, TypeError):
        return (
            getattr(left, "st_dev", None),
            getattr(left, "st_ino", None),
        ) == (
            getattr(right, "st_dev", None),
            getattr(right, "st_ino", None),
        )


def _same_checkpoint_snapshot(left: Any, right: Any) -> bool:
    if not _same_file_identity(left, right):
        return False
    for field in ("st_size", "st_mtime_ns", "st_ctime_ns"):
        if getattr(left, field, None) != getattr(right, field, None):
            return False
    return True


def _absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_managed_generation_path(
    path: Path | str,
    *,
    managed_root: Path,
    require_exists: bool,
) -> tuple[Path, Path]:
    """Validate lexical and resolved containment beneath the managed root."""
    lexical = _absolute_path(path)
    try:
        relative = lexical.relative_to(managed_root)
    except ValueError as exc:
        raise ValueError(
            "Generation restart checkpoint escaped managed generation storage"
        ) from exc

    current = managed_root
    paths = [managed_root]
    for part in relative.parts:
        current = current / part
        paths.append(current)
    for component in paths:
        try:
            component_stat = os.lstat(component)
        except FileNotFoundError:
            if require_exists:
                raise ValueError(
                    "Generation restart checkpoint path is unavailable"
                ) from None
            break
        if _is_symlink_or_reparse(component_stat):
            raise ValueError(
                "Generation restart checkpoint path contains a symlink or reparse point"
            )

    try:
        resolved_root = managed_root.resolve(strict=True)
        resolved = lexical.resolve(strict=require_exists)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Generation restart checkpoint escaped managed generation storage"
        ) from exc
    return lexical, resolved


def _require_directory_identity(
    directory: Path,
    expected_stat: Any,
    *,
    managed_root: Path,
) -> None:
    _require_managed_generation_path(
        directory,
        managed_root=managed_root,
        require_exists=True,
    )
    current_stat = os.lstat(directory)
    if (
        _is_symlink_or_reparse(current_stat)
        or not stat.S_ISDIR(current_stat.st_mode)
        or not _same_file_identity(current_stat, expected_stat)
    ):
        raise ValueError(
            "Generation restart checkpoint destination changed during copy"
        )


def _unlink_if_same_file(path: Optional[Path], expected_stat: Any) -> None:
    if path is None or expected_stat is None:
        return
    try:
        current_stat = os.lstat(path)
        if _same_file_identity(current_stat, expected_stat):
            path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "Could not remove rejected generation checkpoint temporary %s",
            path,
        )


def _require_open_file_identity(
    path: Path,
    file_descriptor: int,
    expected_stat: Any,
    *,
    managed_root: Path,
    error_message: str,
) -> None:
    _require_managed_generation_path(
        path,
        managed_root=managed_root,
        require_exists=True,
    )
    descriptor_stat = os.fstat(file_descriptor)
    path_stat = os.lstat(path)
    if (
        _is_symlink_or_reparse(path_stat)
        or not _same_file_identity(descriptor_stat, expected_stat)
        or not _same_file_identity(path_stat, expected_stat)
    ):
        raise ValueError(error_message)


def _unlink_same_file_at(
    directory_fd: Optional[int],
    name: str,
    expected_stat: Any,
) -> None:
    if directory_fd is None or expected_stat is None:
        return
    try:
        current_stat = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if _same_file_identity(current_stat, expected_stat):
            os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "Could not remove rejected generation checkpoint file %s by directory fd",
            name,
        )


def _supports_checkpoint_directory_fd() -> bool:
    return bool(
        getattr(os, "O_DIRECTORY", 0)
        and os.open in os.supports_dir_fd
        # os.replace accepts src_dir_fd/dst_dir_fd on POSIX but CPython lists
        # the underlying rename capability (not the alias) in supports_dir_fd.
        and os.rename in os.supports_dir_fd
    )


def _supports_windows_checkpoint_anchor() -> bool:
    return os.name == "nt"


def _copy_restart_checkpoint(
    source_path: str,
    destination: Path,
    *,
    expected_directory: Path,
) -> str:
    """Copy one immutable checkpoint through identity-fenced file descriptors."""
    managed_root = _absolute_path(GENERATION_OUTPUT_DIR)
    source, _source_resolved = _require_managed_generation_path(
        source_path,
        managed_root=managed_root,
        require_exists=True,
    )
    source_before = os.lstat(source)
    if _is_symlink_or_reparse(source_before):
        raise ValueError(
            "Generation restart checkpoint source is a symlink or reparse point"
        )
    if not stat.S_ISREG(source_before.st_mode):
        raise ValueError("Generation restart checkpoint is not a regular file")
    if not source.is_file():
        raise ValueError("Generation restart checkpoint is not a regular file")

    expected_directory = _absolute_path(expected_directory)
    destination = _absolute_path(destination)
    if destination.parent != expected_directory:
        raise ValueError("Generation restart checkpoint destination escaped attempt")
    _require_managed_generation_path(
        expected_directory.parent,
        managed_root=managed_root,
        require_exists=False,
    )
    expected_directory.mkdir(parents=True, exist_ok=True)
    _require_managed_generation_path(
        expected_directory,
        managed_root=managed_root,
        require_exists=True,
    )
    directory_stat = os.lstat(expected_directory)
    if (
        _is_symlink_or_reparse(directory_stat)
        or not stat.S_ISDIR(directory_stat.st_mode)
    ):
        raise ValueError(
            "Generation restart checkpoint destination contains a symlink or reparse point"
        )

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    source_fd: Optional[int] = None
    target_fd: Optional[int] = None
    directory_fd: Optional[int] = None
    anchor_fd: Optional[int] = None
    anchor_stat: Any = None
    anchor_real_path: Optional[Path] = None
    temporary_stat: Any = None
    temporary_real_path: Optional[Path] = None
    published_real_path: Optional[Path] = None
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        source_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, source_flags)
        source_opened = os.fstat(source_fd)
        source_after = os.lstat(source)
        _require_managed_generation_path(
            source,
            managed_root=managed_root,
            require_exists=True,
        )
        if (
            not stat.S_ISREG(source_opened.st_mode)
            or source_opened.st_nlink != 1
            or not _same_checkpoint_snapshot(source_before, source_opened)
            or not _same_checkpoint_snapshot(source_after, source_opened)
        ):
            raise ValueError(
                "Generation restart checkpoint changed during restart copy"
            )

        if _supports_checkpoint_directory_fd():
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            directory_fd = os.open(expected_directory, directory_flags)
            if not _same_file_identity(
                os.fstat(directory_fd),
                directory_stat,
            ):
                raise ValueError(
                    "Generation restart checkpoint destination changed during copy"
                )
        elif _supports_windows_checkpoint_anchor():
            anchor_path = expected_directory / (
                f".checkpoint-anchor.{uuid4().hex}.lock"
            )
            _require_directory_identity(
                expected_directory,
                directory_stat,
                managed_root=managed_root,
            )
            anchor_flags = (
                os.O_RDONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOINHERIT", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            anchor_fd = os.open(anchor_path, anchor_flags, 0o600)
            anchor_stat = os.fstat(anchor_fd)
            anchor_real_path = anchor_path.resolve(strict=True)
            _require_open_file_identity(
                anchor_path,
                anchor_fd,
                anchor_stat,
                managed_root=managed_root,
                error_message=(
                    "Generation restart checkpoint destination anchor changed"
                ),
            )
        else:
            raise ValueError(
                "Generation restart checkpoint secure directory anchoring unavailable"
            )

        _require_directory_identity(
            expected_directory,
            directory_stat,
            managed_root=managed_root,
        )
        target_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if directory_fd is not None:
            target_fd = os.open(
                temporary.name,
                target_flags,
                0o600,
                dir_fd=directory_fd,
            )
        else:
            target_fd = os.open(temporary, target_flags, 0o600)
        temporary_stat = os.fstat(target_fd)
        temporary_real_path = temporary.resolve(strict=True)
        _require_managed_generation_path(
            temporary,
            managed_root=managed_root,
            require_exists=True,
        )
        temporary_after = os.lstat(temporary)
        if (
            not stat.S_ISREG(temporary_stat.st_mode)
            or not _same_file_identity(temporary_after, temporary_stat)
        ):
            raise ValueError(
                "Generation restart checkpoint temporary changed during copy"
            )

        copied_bytes = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            copied_bytes += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
        os.fsync(target_fd)
        source_after_copy = os.fstat(source_fd)
        source_path_after_copy = os.lstat(source)
        _require_managed_generation_path(
            source,
            managed_root=managed_root,
            require_exists=True,
        )
        if (
            copied_bytes != source_opened.st_size
            or not _same_checkpoint_snapshot(source_opened, source_after_copy)
            or not _same_checkpoint_snapshot(source_path_after_copy, source_opened)
        ):
            raise ValueError(
                "Generation restart checkpoint changed during restart copy"
            )
        _require_directory_identity(
            expected_directory,
            directory_stat,
            managed_root=managed_root,
        )
        if anchor_fd is not None:
            _require_open_file_identity(
                anchor_real_path or anchor_path,
                anchor_fd,
                anchor_stat,
                managed_root=managed_root,
                error_message=(
                    "Generation restart checkpoint destination anchor changed"
                ),
            )
        temporary_after_fd = os.fstat(target_fd)
        temporary_after = os.lstat(temporary)
        if (
            not _same_file_identity(temporary_after_fd, temporary_stat)
            or not _same_file_identity(temporary_after, temporary_stat)
        ):
            raise ValueError(
                "Generation restart checkpoint temporary changed during copy"
            )
        os.close(target_fd)
        target_fd = None
        if os.path.lexists(destination):
            raise ValueError(
                "Generation restart checkpoint destination already exists"
            )
        _require_directory_identity(
            expected_directory,
            directory_stat,
            managed_root=managed_root,
        )
        if anchor_fd is not None:
            _require_open_file_identity(
                anchor_real_path or anchor_path,
                anchor_fd,
                anchor_stat,
                managed_root=managed_root,
                error_message=(
                    "Generation restart checkpoint destination anchor changed"
                ),
            )
        if directory_fd is not None:
            os.replace(
                temporary.name,
                destination.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            os.replace(temporary, destination)
        published_real_path = destination.resolve(strict=True)
        _require_managed_generation_path(
            destination,
            managed_root=managed_root,
            require_exists=True,
        )
        published_stat = os.lstat(destination)
        if not _same_file_identity(published_stat, temporary_stat):
            raise ValueError(
                "Generation restart checkpoint destination changed during copy"
            )
        _require_directory_identity(
            expected_directory,
            directory_stat,
            managed_root=managed_root,
        )
        if anchor_fd is not None:
            _require_open_file_identity(
                anchor_real_path or anchor_path,
                anchor_fd,
                anchor_stat,
                managed_root=managed_root,
                error_message=(
                    "Generation restart checkpoint destination anchor changed"
                ),
            )
        temporary_stat = None
    finally:
        if target_fd is not None:
            os.close(target_fd)
        if source_fd is not None:
            os.close(source_fd)
        if temporary_stat is not None:
            _unlink_same_file_at(
                directory_fd,
                temporary.name,
                temporary_stat,
            )
            _unlink_same_file_at(
                directory_fd,
                destination.name,
                temporary_stat,
            )
            _unlink_if_same_file(temporary_real_path, temporary_stat)
            _unlink_if_same_file(published_real_path, temporary_stat)
        if directory_fd is not None:
            os.close(directory_fd)
        if anchor_fd is not None:
            os.close(anchor_fd)
            _unlink_if_same_file(anchor_real_path, anchor_stat)
    return str(destination)

# 运行中的 Pipeline 引用（用于停止任务）
_running_pipelines: Dict[str, DatasetGenerationPipeline] = {}
_running_pipeline_tokens: Dict[str, str] = {}
_generation_state_lock = threading.RLock()
_RUN_TOKEN_UNSET = object()


@dataclass(frozen=True)
class _GenerationCancellationRequest:
    owner: object
    run_token: Optional[str]


_generation_cancellation_requests: Dict[
    str,
    _GenerationCancellationRequest,
] = {}


def _get_running_pipeline(
    task_id: str,
) -> tuple[Optional[DatasetGenerationPipeline], Optional[str]]:
    with _generation_state_lock:
        return (
            _running_pipelines.get(task_id),
            _running_pipeline_tokens.get(task_id),
        )


def _install_running_pipeline(
    task_id: str,
    pipeline: DatasetGenerationPipeline,
    run_token: str,
) -> None:
    with _generation_state_lock:
        _running_pipelines[task_id] = pipeline
        _running_pipeline_tokens[task_id] = run_token


def _remove_running_pipeline(
    task_id: str,
    *,
    pipeline: Optional[DatasetGenerationPipeline] = None,
    run_token: Optional[str] = None,
) -> Optional[DatasetGenerationPipeline]:
    with _generation_state_lock:
        current_pipeline = _running_pipelines.get(task_id)
        current_token = _running_pipeline_tokens.get(task_id)
        if pipeline is not None and current_pipeline is not pipeline:
            return None
        if run_token is not None and current_token != run_token:
            return None
        removed = _running_pipelines.pop(task_id, None)
        _running_pipeline_tokens.pop(task_id, None)
        return removed


def _is_current_pipeline(
    task_id: str,
    pipeline: DatasetGenerationPipeline,
    run_token: str,
) -> bool:
    with _generation_state_lock:
        return (
            _running_pipelines.get(task_id) is pipeline
            and _running_pipeline_tokens.get(task_id) == run_token
        )


def _request_generation_cancellation(
    task_id: str,
    owner: Optional[object] = None,
    *,
    run_token: Any = _RUN_TOKEN_UNSET,
) -> _GenerationCancellationRequest:
    with _generation_state_lock:
        cancellation_owner = (
            owner
            if owner is not None
            else _running_pipelines.get(task_id) or object()
        )
        cancellation_token = (
            _running_pipeline_tokens.get(task_id)
            if run_token is _RUN_TOKEN_UNSET
            else run_token
        )
        request = _GenerationCancellationRequest(
            owner=cancellation_owner,
            run_token=cancellation_token,
        )
        _generation_cancellation_requests[task_id] = request
        return request


def _clear_generation_cancellation(
    task_id: str,
    owner: Optional[object] = None,
    *,
    run_token: Any = _RUN_TOKEN_UNSET,
) -> None:
    with _generation_state_lock:
        request = _generation_cancellation_requests.get(task_id)
        if request is None:
            return
        owner_matches = (
            owner is None
            or request is owner
            or request.owner is owner
        )
        token_matches = (
            run_token is _RUN_TOKEN_UNSET
            or request.run_token == run_token
        )
        if owner_matches and token_matches:
            _generation_cancellation_requests.pop(task_id, None)


def _is_generation_cancellation_requested(
    task_id: str,
    owner: Optional[object] = None,
    *,
    run_token: Any = _RUN_TOKEN_UNSET,
) -> bool:
    with _generation_state_lock:
        request = _generation_cancellation_requests.get(task_id)
        return request is not None and (
            owner is None or request.owner is owner
        ) and (
            run_token is _RUN_TOKEN_UNSET or request.run_token == run_token
        )


def _is_generation_stopped(
    task_id: str,
    *,
    run_token: Any = _RUN_TOKEN_UNSET,
    owner: Optional[object] = None,
) -> bool:
    if _is_generation_cancellation_requested(
        task_id,
        owner,
        run_token=run_token,
    ):
        return True
    task = generation_task_service.get_task_raw(task_id)
    if task is None:
        return True
    if (
        run_token is not _RUN_TOKEN_UNSET
        and task.get("run_token") != run_token
    ):
        return True
    return task.get("status") in {
        GenerationStatus.STOPPING,
        GenerationStatus.STOPPED,
        GenerationStatus.RECOVERING,
        GenerationStatus.RESTARTING,
    }


def _get_owned_generation_dataset(
    dataset_id: Optional[str],
    generation_task_id: str,
    user_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not dataset_id or not user_id:
        return None
    dataset = dataset_service.get_dataset(dataset_id)
    if not (
        dataset
        and dataset.get("user_id") == user_id
        and dataset.get("source_task_type") == "generation"
        and dataset.get("source_task_id") == generation_task_id
    ):
        return None
    return dataset


def _finalize_sync_generation_tracking(
    generation_task_id: str,
    *,
    generation_mode: str,
    output_dataset_id: Optional[str],
    output_sample_count: int,
    qa_dataset_id: Optional[str] = None,
    deep_eval_dataset_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Reconcile a successful generation without bypassing sync ownership."""
    try:
        from ...storage.services.external_sync_service import external_sync_service
        from ...sync.level2_handler import (
            on_generation_completed,
            on_generation_failed,
            on_qa_phase_completed,
        )

        sync_generation = external_sync_service.get_generation_by_task_id(
            generation_task_id
        )
        if not sync_generation:
            return True
        owner_id = user_id or sync_generation.get("user_id")
        sample_count = max(int(output_sample_count or 0), 0)

        def fail_tracking(reason: str) -> bool:
            recovery = on_generation_failed(
                generation_task_id=generation_task_id,
                error=reason,
            )
            return bool(recovery.get("recovered"))

        def complete_tracking(dataset_id: Optional[str], count: int) -> bool:
            completion = on_generation_completed(
                generation_task_id=generation_task_id,
                output_dataset_id=dataset_id,
                output_sample_count=count,
            )
            return bool(
                completion.get("completed")
                or completion.get("already_completed")
            )

        if generation_mode in {"doc_to_training", "qa_to_training"}:
            if not _get_owned_generation_dataset(
                output_dataset_id, generation_task_id, owner_id
            ):
                return fail_tracking(
                    "Sync generation output dataset is missing or invalid"
                )
            if qa_dataset_id:
                qa_dataset = _get_owned_generation_dataset(
                    qa_dataset_id, generation_task_id, owner_id
                )
                if not qa_dataset:
                    return fail_tracking(
                        "Sync QA output dataset is missing or invalid"
                    )
                qa_result = on_qa_phase_completed(
                    generation_task_id=generation_task_id,
                    qa_dataset_id=qa_dataset_id,
                    qa_sample_count=max(int(qa_dataset.get("num_rows") or 0), 0),
                )
                if qa_result.get("conflict"):
                    return False
            return complete_tracking(output_dataset_id, sample_count)

        if generation_mode == "qa_extraction":
            qa_dataset = _get_owned_generation_dataset(
                output_dataset_id, generation_task_id, owner_id
            )
            if not qa_dataset:
                return fail_tracking(
                    "Sync QA output dataset is missing or invalid"
                )
            qa_result = on_qa_phase_completed(
                generation_task_id=generation_task_id,
                qa_dataset_id=output_dataset_id,
                qa_sample_count=max(int(qa_dataset.get("num_rows") or sample_count), 0),
            )
            if qa_result.get("conflict"):
                return False
            return complete_tracking(None, 0)

        if generation_mode in {"doc_to_eval", "qa_to_eval"}:
            if sample_count > 0 and not _get_owned_generation_dataset(
                deep_eval_dataset_id, generation_task_id, owner_id
            ):
                return fail_tracking(
                    "Sync evaluation output dataset is missing or invalid"
                )
            return complete_tracking(None, sample_count)

        return fail_tracking("Sync generation mode is invalid")
    except Exception as exc:
        logger.warning(
            "[sync] Tracking reconciliation failed for generation %s: %s",
            generation_task_id,
            exc,
        )
        return False


def _finalize_sync_generation_failure_tracking(
    generation_task_id: str,
    error: str,
) -> bool:
    try:
        from ...sync.level2_handler import on_generation_failed

        recovery = on_generation_failed(
            generation_task_id=generation_task_id,
            error=error,
        )
        return bool(recovery.get("recovered"))
    except Exception as exc:
        logger.warning(
            "[sync] Failure tracking reconciliation failed for generation %s: %s",
            generation_task_id,
            exc,
        )
        return False


def _reject_pending_sync_tracking(
    generation_task_id: str,
    *,
    reject_failed: bool = False,
) -> None:
    """Prevent lifecycle changes from orphaning sync-owned queued batches."""
    from ...storage.services.external_sync_service import external_sync_service

    tracking = external_sync_service.get_generation_by_task_id(generation_task_id)
    if tracking and tracking.get("status") == "pending":
        raise HTTPException(
            status_code=409,
            detail="Sync generation tracking is still pending reconciliation",
        )
    if reject_failed and tracking and tracking.get("status") == "failed":
        raise HTTPException(
            status_code=409,
            detail=(
                "Recovered sync generations cannot be restarted directly; "
                "trigger a new generation from the sync task"
            ),
        )


def _finalize_stopped_generation(task_id: str, reason: str) -> None:
    """Idempotently reconcile sync bookkeeping for a stopped generation."""
    try:
        from ...sync.level2_handler import on_generation_failed

        on_generation_failed(generation_task_id=task_id, error=reason)
    except Exception as exc:
        logger.warning(
            "[sync] Failed to rollback stopped generation %s: %s",
            task_id,
            exc,
        )


def _stop_generation_execution(
    task_id: str,
    reason: str,
    pipeline: Optional[DatasetGenerationPipeline] = None,
    *,
    run_token: Any = _RUN_TOKEN_UNSET,
) -> None:
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception as exc:
            logger.warning("Failed to stop generation pipeline %s: %s", task_id, exc)

    task = generation_task_service.get_task_raw(task_id)
    owns_attempt = bool(
        task
        and (
            run_token is _RUN_TOKEN_UNSET
            or task.get("run_token") == run_token
        )
    )
    stopped = bool(
        owns_attempt and task.get("status") == GenerationStatus.STOPPED
    )
    if owns_attempt and task.get("status") in {
        GenerationStatus.PENDING,
        GenerationStatus.RUNNING,
        GenerationStatus.STOPPING,
    }:
        try:
            status_kwargs = (
                {}
                if run_token is _RUN_TOKEN_UNSET
                else {"expected_run_token": run_token}
            )
            current_status = task.get("status")
            if current_status == GenerationStatus.RUNNING:
                stopping = generation_task_service.update_status(
                    task_id,
                    GenerationStatus.STOPPING,
                    **status_kwargs,
                )
                if not stopping:
                    current = generation_task_service.get_task_raw(task_id)
                    current_status = current.get("status") if current else None
            if current_status in {
                GenerationStatus.PENDING,
                GenerationStatus.STOPPING,
            }:
                stopped = generation_task_service.update_status(
                    task_id,
                    GenerationStatus.STOPPED,
                    **status_kwargs,
                )
        except Exception as exc:
            logger.warning("Failed to persist stopped generation %s: %s", task_id, exc)
            stopped = False
        if not stopped:
            current = generation_task_service.get_task_raw(task_id)
            stopped = bool(
                current
                and current.get("status") == GenerationStatus.STOPPED
                and (
                    run_token is _RUN_TOKEN_UNSET
                    or current.get("run_token") == run_token
                )
            )

    if stopped:
        _finalize_stopped_generation(task_id, reason)


def _is_anonymous_user(current_user: Dict[str, Any]) -> bool:
    return current_user.get("user_id") in (None, "anonymous")


def _verify_owned_resource(
    resource: Optional[Dict[str, Any]],
    current_user: Dict[str, Any],
    resource_name: str,
) -> Dict[str, Any]:
    if not resource:
        raise HTTPException(status_code=404, detail=f"{resource_name} not found")
    if _is_anonymous_user(current_user):
        return resource
    if resource.get("user_id") != current_user.get("user_id"):
        raise HTTPException(
            status_code=403,
            detail=f"Not authorized to access this {resource_name.lower()}",
        )
    return resource


def _verify_task_access(task_id: str, current_user: Dict[str, Any]) -> Dict[str, Any]:
    """验证任务访问权限并返回任务数据"""
    task = generation_task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _verify_owned_resource(task, current_user, "Task")


def _resolve_owned_generation_dataset(
    dataset_id: str,
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    dataset = dataset_service.get_dataset(dataset_id)
    dataset = _verify_owned_resource(dataset, current_user, "Dataset")
    if dataset.get("status") == "deleting":
        raise HTTPException(
            status_code=409,
            detail="Dataset deletion is in progress",
        )
    if _is_anonymous_user(current_user):
        return dataset
    try:
        return resolve_managed_local_dataset(
            dataset_id,
            user_id=current_user["user_id"],
        )
    except UnsupportedEvaluationDatasetStorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResourceProvenanceError as exc:
        raise HTTPException(
            status_code=403,
            detail="Dataset does not have verifiable API-managed provenance",
        ) from exc


def _resolve_generation_task_input(task: Dict[str, Any]) -> str:
    stored_path = task.get("input_path")
    if not get_settings().auth_enabled:
        if not stored_path:
            raise ResourceProvenanceError("Generation input path is missing")
        return stored_path

    user_id = task.get("user_id")
    dataset_id = task.get("source_dataset_id")
    if not user_id or user_id == "anonymous":
        raise ResourceProvenanceError("Authenticated generation task owner is missing")
    if not dataset_id:
        raise ResourceProvenanceError(
            "Authenticated generation task is not bound to a dataset"
        )
    dataset = resolve_managed_local_dataset(dataset_id, user_id=user_id)
    return dataset["storage_path"]


def _require_task_collection_ownership(
    task: Dict[str, Any],
    collection_name: Optional[str] = None,
) -> Optional[str]:
    """Revalidate a persisted Milvus name immediately before it is reused."""
    name = collection_name or task.get("milvus_collection")
    if not name or not get_settings().auth_enabled:
        return name

    user_id = task.get("user_id")
    if not user_id or user_id == "anonymous":
        raise ResourceProvenanceError(
            "Authenticated generation task owner is missing"
        )

    registered = milvus_collection_service.get_by_name(name)
    if registered:
        if registered.get("user_id") != user_id:
            raise ResourceProvenanceError(
                "Milvus collection owner does not match generation task owner"
            )
        return name

    # Older sync tasks created deterministic collections before registry rows.
    # Preserve only a fully-owned sync linkage; an existing registry entry above
    # always takes precedence.
    try:
        from ...storage.services.external_sync_service import external_sync_service

        sync_generation = external_sync_service.get_generation_by_task_id(
            task.get("task_id", "")
        )
        if sync_generation and sync_generation.get("user_id") == user_id:
            sync_task = external_sync_service.get_task_raw(
                sync_generation.get("task_id", "")
            )
            if (
                sync_task
                and sync_task.get("user_id") == user_id
                and sync_task.get("milvus_collection_name") == name
            ):
                return name
    except Exception as exc:
        logger.warning(
            "Could not prove legacy sync collection ownership for task %s: %s",
            task.get("task_id"),
            type(exc).__name__,
        )

    raise ResourceProvenanceError(
        "Milvus collection does not have verifiable task ownership"
    )


def _file_has_data(path: Optional[str]) -> bool:
    """Check whether a file exists and contains non-empty content."""
    if not path:
        return False
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _build_task_stages(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build normalized stage status view from existing task fields.

    This is a read-only status projection and does not alter task execution logic.
    """
    mode = task.get("generation_mode", "doc_to_training")
    task_status = task.get("status")
    running = task_status in {
        GenerationStatus.RUNNING,
        GenerationStatus.STOPPING,
        GenerationStatus.PUBLISHING,
        GenerationStatus.RECOVERING,
        GenerationStatus.RESTARTING,
    }
    completed = task_status == GenerationStatus.COMPLETED

    has_embedding = bool(task.get("embedding_config"))
    output_format = task.get("output_format", "universal")
    pos_neg_resume_supported = output_format == "universal"
    qa_output_path = task.get("qa_output_path")
    qa_filtered_path = task.get("qa_filtered_path")
    output_path = task.get("output_path")
    deep_eval_path = task.get("deep_eval_path")

    qa_ready = _file_has_data(qa_output_path)
    qa_filtered_ready = _file_has_data(qa_filtered_path)
    output_ready = _file_has_data(output_path)
    deep_eval_ready = _file_has_data(deep_eval_path)

    def _state(is_done: bool, is_running: bool, skipped: bool = False) -> str:
        if skipped:
            return "skipped"
        if is_done:
            return "completed"
        if is_running:
            return "running"
        return "pending"

    stages: List[Dict[str, Any]] = []

    if mode == "doc_to_training":
        qa_running = running and not qa_ready
        filter_skipped = not has_embedding
        filter_running = running and has_embedding and qa_ready and not qa_filtered_ready
        output_running = running and (qa_filtered_ready if has_embedding else qa_ready)
        output_done = completed or output_ready

        stages.append({
            "stage": "qa_extraction",
            "label": "QA Extraction",
            "status": _state(qa_ready, qa_running),
            "checkpoint_path": qa_output_path,
            "resume_supported": True,
            "resume_ready": qa_ready,
        })
        stages.append({
            "stage": "qa_filter",
            "label": "QA Filter & Vector Ingest",
            "status": _state(qa_filtered_ready, filter_running, skipped=filter_skipped),
            "checkpoint_path": qa_filtered_path,
            "resume_supported": has_embedding,
            "resume_ready": qa_filtered_ready,
        })
        stages.append({
            "stage": "pos_neg_generation",
            "label": "Pos/Neg Generation",
            "status": _state(output_done, output_running),
            "checkpoint_path": output_path,
            "resume_supported": pos_neg_resume_supported,
            "resume_ready": pos_neg_resume_supported and output_ready,
        })
        stages.append({
            "stage": "deep_eval_export",
            "label": "Deep Evaluation Export",
            "status": _state(deep_eval_ready, False, skipped=not bool(deep_eval_path)),
            "checkpoint_path": deep_eval_path,
            "resume_supported": False,
            "resume_ready": deep_eval_ready,
        })
        return stages

    if mode == "qa_to_training":
        filter_skipped = not has_embedding
        filter_done = bool(task.get("filter_stats")) or qa_filtered_ready
        output_done = completed or output_ready
        output_running = running and not output_done

        stages.append({
            "stage": "qa_input",
            "label": "QA Input",
            "status": "completed",
            "checkpoint_path": task.get("input_path"),
            "resume_supported": False,
            "resume_ready": False,
        })
        stages.append({
            "stage": "qa_filter",
            "label": "QA Filter & Vector Ingest",
            "status": _state(filter_done, running and has_embedding and not filter_done, skipped=filter_skipped),
            "checkpoint_path": qa_filtered_path,
            "resume_supported": has_embedding,
            "resume_ready": has_embedding and qa_filtered_ready,
        })
        stages.append({
            "stage": "pos_neg_generation",
            "label": "Pos/Neg Generation",
            "status": _state(output_done, output_running),
            "checkpoint_path": output_path,
            "resume_supported": pos_neg_resume_supported,
            "resume_ready": pos_neg_resume_supported and output_ready,
        })
        stages.append({
            "stage": "deep_eval_export",
            "label": "Deep Evaluation Export",
            "status": _state(deep_eval_ready, False, skipped=not bool(deep_eval_path)),
            "checkpoint_path": deep_eval_path,
            "resume_supported": True,
            "resume_ready": deep_eval_ready,
        })
        return stages

    if mode == "doc_to_eval":
        qa_running = running and not qa_ready
        retrieval_running = running and qa_ready and not deep_eval_ready

        stages.append({
            "stage": "qa_extraction",
            "label": "QA Extraction",
            "status": _state(qa_ready, qa_running),
            "checkpoint_path": qa_output_path,
            "resume_supported": True,
            "resume_ready": qa_ready,
        })
        stages.append({
            "stage": "retrieval_build",
            "label": "Retrieval Build",
            "status": _state(bool(task.get("filter_stats")), retrieval_running),
            "checkpoint_path": task.get("milvus_collection"),
            "resume_supported": False,
            "resume_ready": False,
        })
        stages.append({
            "stage": "deep_eval_export",
            "label": "Deep Evaluation Export",
            "status": _state(completed or deep_eval_ready, running and not deep_eval_ready),
            "checkpoint_path": deep_eval_path,
            "resume_supported": True,
            "resume_ready": deep_eval_ready,
        })
        return stages

    if mode == "qa_to_eval":
        stages.append({
            "stage": "qa_input",
            "label": "QA Input",
            "status": "completed",
            "checkpoint_path": task.get("input_path"),
            "resume_supported": False,
            "resume_ready": False,
        })
        stages.append({
            "stage": "retrieval_build",
            "label": "Retrieval Build",
            "status": _state(bool(task.get("filter_stats")), running and not deep_eval_ready),
            "checkpoint_path": task.get("milvus_collection"),
            "resume_supported": False,
            "resume_ready": False,
        })
        stages.append({
            "stage": "deep_eval_export",
            "label": "Deep Evaluation Export",
            "status": _state(completed or deep_eval_ready, running and not deep_eval_ready),
            "checkpoint_path": deep_eval_path,
            "resume_supported": True,
            "resume_ready": deep_eval_ready,
        })
        return stages

    # Backward-compatible fallback for legacy/unknown modes.
    stages.append({
        "stage": "legacy_pipeline",
        "label": "Legacy Pipeline",
        "status": "completed" if completed else ("running" if running else "pending"),
        "checkpoint_path": output_path,
        "resume_supported": False,
        "resume_ready": False,
    })
    return stages






def _get_default_steps_config(generation_mode: str) -> Dict[str, Dict[str, Any]]:
    """返回指定模式的默认步骤配置（含所有参数默认值）"""
    doc_quality_defaults = {
        "enabled": True,
        "use_llm": True,
        "min_length": 50,
        "min_text_ratio": 0.3,
        "min_unique_chars": 10,
        "min_score": 0.6,
    }
    qa_gen_defaults = {
        "enabled": True,
        "num_qa_per_doc": 1,
        "use_role": False,
        "languages": ["中文"],
    }
    pos_neg_defaults = {
        "enabled": True,
        "num_positive": 5,
        "num_negative": 20,
        "use_role": False,
        "default_length": "medium",
        "neg_detection_mode": "chunk",
        "supplement_positives": True,
        "confirm_positives": False,
        "confirm_negatives": False,
        "answer_rewrite": False,
        "rerank_score_classification": False,
        "evidence_removal": False,
        "evidence_pruning": False,
        "neg_chunk_scoring": False,
        "chunk_eval_mode": "batch",
        "skip_easy_negatives": True,
        "skip_perfect_ap": True,
        "skip_zero_ap": True,
    }
    role_gen_defaults = {"enabled": False, "roles_per_doc": 3}
    validation_defaults = {"enabled": False}

    if generation_mode == "qa_extraction":
        return {
            "doc_quality": doc_quality_defaults,
            "role_gen": role_gen_defaults,
            "qa_gen": qa_gen_defaults,
        }
    elif generation_mode == "qa_to_training":
        return {
            "role_gen": role_gen_defaults,
            "pos_neg_extraction": pos_neg_defaults,
            "validation": validation_defaults,
        }
    elif generation_mode == "doc_to_training":
        return {
            "doc_quality": doc_quality_defaults,
            "role_gen": role_gen_defaults,
            "qa_gen": qa_gen_defaults,
            "pos_neg_extraction": pos_neg_defaults,
            "validation": validation_defaults,
        }
    elif generation_mode == "doc_to_eval":
        return {
            "doc_quality": doc_quality_defaults,
            "role_gen": role_gen_defaults,
            "qa_gen": qa_gen_defaults,
        }
    elif generation_mode == "qa_to_eval":
        return {}
    return {}


def _resolve_model_config(
    config_id: str,
    current_user: Dict[str, Any],
    expected_types: Optional[set] = None,
    *,
    allow_public: bool = False,
) -> Dict[str, Any]:
    config = model_config_service.get_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    owner_id = config.get("user_id")
    user_id = current_user.get("user_id")
    owner_matches = bool(user_id and user_id != "anonymous" and owner_id == user_id)
    public_read = allow_public and not owner_id
    if not owner_matches and not public_read:
        raise HTTPException(status_code=403, detail="无权访问该模型配置")
    if expected_types:
        model_type = config.get("model_type")
        if model_type == "reranker":
            model_type = "rerank"
        if model_type not in expected_types:
            raise HTTPException(status_code=400, detail="模型类型与配置不匹配")
    return config


def _normalize_model_config_endpoints(
    config: Optional[Dict[str, Any]],
    user_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Normalize and validate every model endpoint before task persistence."""
    if not config:
        return config

    normalized = dict(config)
    if normalized.get("endpoint"):
        normalized["endpoint"] = validate_user_outbound_url(
            normalized["endpoint"],
            user_id,
        )

    if normalized.get("endpoints") is not None:
        endpoints = []
        for endpoint in normalized["endpoints"]:
            item = dict(endpoint)
            item["url"] = validate_user_outbound_url(
                item.get("url", ""),
                user_id,
            )
            endpoints.append(item)
        normalized["endpoints"] = endpoints

    return normalized


_GENERATION_MODEL_CONFIG_TYPES = {
    "llm_config": {"llm"},
    "eval_llm_config": {"llm"},
    "embedding_config": {"embedding"},
    "rerank_config": {"rerank"},
}


def _revalidate_task_model_config_ownership(
    task: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Recheck referenced credential ownership immediately before execution."""
    current_user = {"user_id": task.get("user_id")}
    resolved_configs: Dict[str, Dict[str, Any]] = {}
    for field_name, expected_types in _GENERATION_MODEL_CONFIG_TYPES.items():
        config = task.get(field_name) or {}
        config_id = config.get("config_id")
        if field_name == "embedding_config" and not config_id:
            config_id = task.get("embedding_config_id")
        if config_id:
            resolved_configs[field_name] = _resolve_model_config(
                config_id,
                current_user,
                expected_types,
                allow_public=False,
            )
    return resolved_configs


def _refresh_owned_pipeline_model_configs(
    config: PipelineConfig,
    resolved_configs: Dict[str, Dict[str, Any]],
) -> None:
    """Load current credentials only after the ownership checks have passed."""
    for field_name, source in resolved_configs.items():
        runtime_config = dict(getattr(config, field_name) or {})
        runtime_config.update(
            {
                "config_id": source.get("config_id") or runtime_config.get("config_id"),
                "endpoint": source.get("api_endpoint"),
                "model": source.get("model_name"),
                "api_key": source.get("api_key"),
            }
        )
        setattr(config, field_name, runtime_config)


# ===== Request/Response Models =====

class LLMEndpointConfig(BaseModel):
    """LLM 端点配置"""
    url: str = Field(..., min_length=1, max_length=2048, description="API 端点 URL")
    model: str = Field(..., min_length=1, max_length=256, description="模型名称")
    api_key: Optional[str] = Field(default=None, max_length=512, description="API 密钥")


class LLMConfigModel(BaseModel):
    """LLM 配置"""
    config_id: Optional[str] = Field(default=None, max_length=36, description="模型配置 ID")
    endpoints: Optional[List[LLMEndpointConfig]] = Field(
        default=None,
        min_length=1,
        max_length=MAX_GENERATION_LLM_ENDPOINTS,
        description="多端点配置（用于负载均衡）"
    )
    endpoint: Optional[str] = Field(default=None, max_length=2048, description="单端点 URL")
    model: Optional[str] = Field(default=None, max_length=256, description="模型名称")
    api_key: Optional[str] = Field(default=None, max_length=512, description="API 密钥")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=2048, ge=1, le=MAX_GENERATION_LLM_TOKENS, description="最大 token 数")
    timeout: int = Field(default=60, ge=1, le=MAX_GENERATION_TIMEOUT, description="超时时间（秒）")
    max_retries: int = Field(default=3, ge=0, le=MAX_GENERATION_RETRIES, description="最大重试次数")
    concurrency: int = Field(
        default=10,
        ge=1,
        le=MAX_GENERATION_LLM_CONCURRENCY,
        description="LLM API 并发数",
    )


class EmbeddingConfigModel(BaseModel):
    """Embedding 配置"""
    config_id: Optional[str] = Field(default=None, max_length=36, description="模型配置 ID")
    endpoint: Optional[str] = Field(default=None, max_length=2048, description="API 端点 URL")
    model: Optional[str] = Field(default=None, max_length=256, description="模型名称")
    api_key: Optional[str] = Field(default=None, max_length=512, description="API 密钥")
    batch_size: int = Field(default=32, ge=1, le=MAX_GENERATION_BATCH_SIZE, description="批处理大小")
    concurrency: int = Field(
        default=20,
        ge=1,
        le=MAX_GENERATION_EMBEDDING_CONCURRENCY,
        description="Embedding API 并发数",
    )
    similarity_threshold: float = Field(default=0.85, ge=-1.0, le=1.0, description="embedding 余弦相似度筛选阈值")
    retrieval_top_k: int = Field(default=10, ge=1, le=MAX_GENERATION_RETRIEVAL_TOP_K, description="向量检索候选 chunk 数量")


class RerankConfigModel(BaseModel):
    """Rerank 配置"""
    config_id: Optional[str] = Field(default=None, max_length=36, description="模型配置 ID")
    endpoint: Optional[str] = Field(default=None, max_length=2048, description="API 端点 URL")
    model: Optional[str] = Field(default=None, max_length=256, description="模型名称")
    api_key: Optional[str] = Field(default=None, max_length=512, description="API 密钥")
    top_k: int = Field(default=10, ge=1, le=MAX_GENERATION_RETRIEVAL_TOP_K, description="返回 top-k 数量")
    batch_size: int = Field(default=64, ge=1, le=MAX_GENERATION_BATCH_SIZE, description="单次请求最大文档数")
    concurrency: int = Field(
        default=10,
        ge=1,
        le=MAX_GENERATION_RERANK_CONCURRENCY,
        description="Rerank API 并发数",
    )
    rerank_threshold: float = Field(default=1.0, description="rerank 相似度过滤阈值，大于此分数的候选将被过滤（视为近似重复），1 表示不过滤")


class WorkerConfigModel(BaseModel):
    """Worker 配置"""
    timeout_per_doc: int = Field(
        default=300,
        ge=1,
        le=MAX_GENERATION_TIMEOUT,
        description="每个文档的超时时间（秒）",
    )


class StepConfigModel(BaseModel):
    """步骤配置"""
    doc_quality: Optional[Dict[str, Any]] = Field(default=None)
    keypoint_gen: Optional[Dict[str, Any]] = Field(default=None)
    role_gen: Optional[Dict[str, Any]] = Field(default=None)
    qa_gen: Optional[Dict[str, Any]] = Field(default=None)
    pos_neg_extraction: Optional[Dict[str, Any]] = Field(default=None)
    validation: Optional[Dict[str, Any]] = Field(default=None)
    embedding_scoring: Optional[Dict[str, Any]] = Field(default=None)
    rerank_scoring: Optional[Dict[str, Any]] = Field(default=None)
    evaluation: Optional[Dict[str, Any]] = Field(default=None)

    @model_validator(mode="after")
    def validate_fanout(self):
        validate_generation_step_limits(self.model_dump(exclude_none=True))
        return self


class PostProcessConfigModel(BaseModel):
    """后处理配置"""
    dedup: Optional[Dict[str, Any]] = Field(default=None)
    augmentation: Optional[Dict[str, Any]] = Field(default=None)
    evidence_ops: Optional[Dict[str, Any]] = Field(default=None)
    mining: Optional[Dict[str, Any]] = Field(default=None)


class CreateTaskRequest(BaseModel):
    """创建生成任务请求"""
    task_name: str = Field(..., min_length=1, max_length=255, description="任务名称")

    # 输入源：二选一
    dataset_id: Optional[str] = Field(default=None, description="数据集 ID（优先使用）")
    input_path: Optional[str] = Field(default=None, description="输入文件/目录路径")
    content_field: Optional[str] = Field(default=None, description="内容字段名（覆盖数据集配置）")

    input_format: str = Field(default="auto", description="输入格式: auto, jsonl, json, txt")
    output_format: str = Field(default="universal", description="输出格式")
    generation_mode: str = Field(
        default="doc_to_training",
        description="生成模式: doc_to_training, qa_to_training, qa_extraction, doc_to_eval, qa_to_eval"
    )
    pos_neg_method: str = Field(
        default="retrieval",
        description="正负例生成方式: retrieval(向量检索+评估分类), llm(纯LLM生成)"
    )

    llm_config: Optional[LLMConfigModel] = Field(default=None, description="LLM 配置（qa_to_eval 模式可选）")
    eval_llm_config: Optional[LLMConfigModel] = Field(default=None, description="评估 LLM 配置（可选，为空时复用 llm_config）")
    embedding_config: Optional[EmbeddingConfigModel] = Field(default=None)
    rerank_config: Optional[RerankConfigModel] = Field(default=None)

    worker_config: Optional[WorkerConfigModel] = Field(default=None)
    steps: Optional[StepConfigModel] = Field(default=None)
    post_process: Optional[PostProcessConfigModel] = Field(default=None)
    prompts: Optional[Dict[str, str]] = Field(default=None, description="自定义 Prompt")

    auto_register_dataset: bool = Field(default=True, description="是否自动注册数据集")

    # 向量集合选择（可选，不指定则自动创建）
    milvus_collection_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="使用已有的 Milvus 集合名称（不指定则自动创建）"
    )

    @model_validator(mode="after")
    def validate_prompt_limits(self):
        validate_generation_custom_prompts(self.prompts)
        return self


class TaskResponse(BaseModel):
    """任务响应（简洁版，用于列表）"""
    task_id: str
    task_name: str
    status: str
    generation_mode: str = "doc_to_training"
    pos_neg_method: Optional[str] = "retrieval"
    progress: float
    total_docs: int
    processed_docs: int
    output_sample_count: int
    error_message: Optional[str] = None
    created_at: Optional[str] = None


class TaskStatsResponse(BaseModel):
    """Global generation-task counts for the current user."""
    total: int = 0
    pending: int = 0
    running: int = 0
    stopping: int = 0
    publishing: int = 0
    recovering: int = 0
    restarting: int = 0
    completed: int = 0
    failed: int = 0
    stopped: int = 0


class TaskListResponse(BaseModel):
    """Paginated generation-task list."""
    tasks: List[TaskResponse]
    total: int
    stats: TaskStatsResponse
    limit: int
    offset: int


class TaskDetailResponse(BaseModel):
    """任务详情响应（完整版）"""
    task_id: str
    task_name: str
    description: Optional[str] = None
    status: str
    progress: float
    total_docs: int
    processed_docs: int
    output_sample_count: int
    error_message: Optional[str] = None

    # 输入配置
    input_path: str
    input_format: str
    content_field: Optional[str] = None
    generation_mode: str
    pos_neg_method: Optional[str] = "retrieval"

    # 输出配置
    output_path: Optional[str] = None
    output_format: str
    output_dataset_id: Optional[str] = None
    auto_register_dataset: bool = True

    # 模型配置
    llm_config: Optional[Dict[str, Any]] = None
    eval_llm_config: Optional[Dict[str, Any]] = None
    embedding_config: Optional[Dict[str, Any]] = None
    rerank_config: Optional[Dict[str, Any]] = None

    # 处理配置
    worker_config: Optional[Dict[str, Any]] = None
    steps_config: Optional[Dict[str, Any]] = None
    post_process_config: Optional[Dict[str, Any]] = None

    # Training 模式配置
    source_dataset_id: Optional[str] = None
    embedding_config_id: Optional[str] = None
    milvus_collection: Optional[str] = None
    filter_stats: Optional[Dict[str, Any]] = None

    # Doc-to-Training 中间产物
    qa_output_path: Optional[str] = None
    qa_dataset_id: Optional[str] = None
    qa_filtered_path: Optional[str] = None
    qa_filtered_dataset_id: Optional[str] = None

    # 深度评估数据集
    deep_eval_path: Optional[str] = None
    deep_eval_dataset_id: Optional[str] = None

    # 统一阶段状态视图（只读投影，不影响现有流程）
    stages: Optional[List[Dict[str, Any]]] = None

    # 时间戳
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskProgressResponse(BaseModel):
    """任务进度响应"""
    task_id: str
    status: str
    progress: float
    total_docs: int
    processed_docs: int
    output_sample_count: int


class TaskArtifactItemResponse(BaseModel):
    """任务阶段产物项"""
    dataset_id: str
    dataset_name: str
    stage_key: str
    stage_name: str
    artifact_role: str = "intermediate"
    dataset_type: str
    usage: str
    file_format: Optional[str] = None
    num_rows: Optional[int] = None
    file_size: Optional[int] = None
    status: str
    tags: Optional[List[str]] = None
    created_at: Optional[str] = None


class TaskStageArtifactsResponse(BaseModel):
    """任务阶段产物分组"""
    stage_key: str
    stage_name: str
    step_order: int
    stage_status: str
    artifacts: List[TaskArtifactItemResponse]


class TaskArtifactsResponse(BaseModel):
    """任务产物聚合响应"""
    task_id: str
    stages: List[TaskStageArtifactsResponse]
    total_artifacts: int


# ===== API Endpoints =====

@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: CreateTaskRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    创建数据生成任务

    启动后台任务处理文档并生成训练数据。
    """
    # 解析输入源：优先使用 dataset_id
    input_path = request.input_path
    content_field = request.content_field

    if request.dataset_id:
        dataset = _resolve_owned_generation_dataset(request.dataset_id, current_user)

        input_path = dataset.get("storage_path")
        if not input_path:
            raise HTTPException(status_code=400, detail="数据集存储路径为空")

        # 从数据集 extra_metadata 读取 content_field（如果请求中未指定）
        if not content_field:
            extra_metadata = dataset.get("extra_metadata") or {}
            content_field = extra_metadata.get("content_field")

    # 构建步骤配置：先填充模式对应的默认值，再用用户配置覆盖
    default_steps = _get_default_steps_config(request.generation_mode)
    user_steps = {}
    if request.steps:
        steps_dict = request.steps.model_dump(exclude_none=True)
        for step_name, step_cfg in steps_dict.items():
            if step_cfg:
                user_steps[step_name] = step_cfg
    steps_config = {}
    for step_name, default_cfg in default_steps.items():
        if step_name in user_steps:
            steps_config[step_name] = {**default_cfg, **user_steps[step_name]}
        else:
            steps_config[step_name] = default_cfg
    for step_name, cfg in user_steps.items():
        if step_name not in steps_config:
            steps_config[step_name] = cfg

    post_process_config = None
    if request.post_process:
        post_process_config = request.post_process.model_dump(exclude_none=True)

    worker_config = request.worker_config.model_dump() if request.worker_config else {
        "timeout_per_doc": 300,
    }

    # 解析 LLM 配置（支持 config_id 或直连，qa_to_eval 模式可选）
    llm_config_data = None
    if request.llm_config:
        llm_config_data = request.llm_config.model_dump(exclude_none=True)
        if request.llm_config.config_id:
            config = _resolve_model_config(request.llm_config.config_id, current_user, {"llm"})
            llm_config_data = {
                "config_id": request.llm_config.config_id,
                "endpoint": config.get("api_endpoint"),
                "model": config.get("model_name"),
                "api_key": config.get("api_key"),
                "temperature": request.llm_config.temperature,
                "max_tokens": request.llm_config.max_tokens,
                "timeout": request.llm_config.timeout,
                "max_retries": request.llm_config.max_retries,
                "concurrency": request.llm_config.concurrency,
            }
        elif request.llm_config.endpoints:
            llm_config_data = request.llm_config.model_dump(exclude_none=True)
        else:
            if not request.llm_config.endpoint or not request.llm_config.model:
                # An effectively-empty llm_config (no config_id / endpoints /
                # endpoint) is allowed for qa_to_eval, where the LLM is optional.
                # The frontend always sends an llm_config object (with config_id=""),
                # so this empty case must map to the optional path, not a hard 400.
                if request.generation_mode == "qa_to_eval":
                    llm_config_data = None
                else:
                    raise HTTPException(status_code=400, detail="LLM 配置缺少 endpoint 或 model")
    elif request.generation_mode != "qa_to_eval":
        raise HTTPException(status_code=400, detail="LLM 配置为必填项")

    # 解析评估 LLM 配置（可选）
    eval_llm_config_data = None
    if request.eval_llm_config:
        if request.eval_llm_config.config_id:
            config = _resolve_model_config(request.eval_llm_config.config_id, current_user, {"llm"})
            eval_llm_config_data = {
                "config_id": request.eval_llm_config.config_id,
                "endpoint": config.get("api_endpoint"),
                "model": config.get("model_name"),
                "api_key": config.get("api_key"),
                "temperature": request.eval_llm_config.temperature,
                "max_tokens": request.eval_llm_config.max_tokens,
                "timeout": request.eval_llm_config.timeout,
                "max_retries": request.eval_llm_config.max_retries,
                "concurrency": request.eval_llm_config.concurrency,
            }
        else:
            eval_llm_config_data = request.eval_llm_config.model_dump(exclude_none=True)

    user_id = current_user.get("user_id")
    llm_config_data = _normalize_model_config_endpoints(llm_config_data, user_id)
    eval_llm_config_data = _normalize_model_config_endpoints(
        eval_llm_config_data,
        user_id,
    )

    # 解析 Embedding 配置
    embedding_config_data = None
    if request.embedding_config:
        if request.embedding_config.config_id:
            config = _resolve_model_config(request.embedding_config.config_id, current_user, {"embedding"})
            embedding_config_data = {
                "config_id": request.embedding_config.config_id,
                "endpoint": config.get("api_endpoint"),
                "model": config.get("model_name"),
                "api_key": config.get("api_key"),
                "batch_size": request.embedding_config.batch_size,
                "concurrency": request.embedding_config.concurrency,
                "similarity_threshold": request.embedding_config.similarity_threshold,
                "retrieval_top_k": request.embedding_config.retrieval_top_k,
            }
        else:
            embedding_config_data = request.embedding_config.model_dump(exclude_none=True)

    # 解析 Rerank 配置
    rerank_config_data = None
    if request.rerank_config:
        if request.rerank_config.config_id:
            config = _resolve_model_config(request.rerank_config.config_id, current_user, {"rerank"})
            rerank_config_data = {
                "config_id": request.rerank_config.config_id,
                "endpoint": config.get("api_endpoint"),
                "model": config.get("model_name"),
                "api_key": config.get("api_key"),
                "top_k": request.rerank_config.top_k,
                "batch_size": request.rerank_config.batch_size,
                "concurrency": request.rerank_config.concurrency,
                "rerank_threshold": request.rerank_config.rerank_threshold,
            }
        else:
            rerank_config_data = request.rerank_config.model_dump(exclude_none=True)

    embedding_config_data = _normalize_model_config_endpoints(
        embedding_config_data,
        user_id,
    )
    rerank_config_data = _normalize_model_config_endpoints(
        rerank_config_data,
        user_id,
    )

    # 模式验证
    embedding_config_id = None

    if request.generation_mode in ("qa_to_training", "qa_to_eval"):
        # qa 模式必须指定且拥有 QA 数据集：_resolve_owned_generation_dataset
        # 做所有权 + 溯源 + deleting 校验，防止跨租户读取他人 QA 内容。
        if not request.dataset_id:
            raise HTTPException(
                status_code=400,
                detail=f"{request.generation_mode} 模式必须指定 dataset_id（QA 数据集）",
            )
        qa_dataset = _resolve_owned_generation_dataset(request.dataset_id, current_user)
        if qa_dataset.get("dataset_type") != "qa_pair":
            raise HTTPException(
                status_code=400,
                detail=f"{request.generation_mode} 模式的 dataset_id 必须指向 qa_pair 类型的数据集",
            )
        input_path = qa_dataset.get("storage_path")
        if not input_path:
            raise HTTPException(status_code=400, detail="QA 数据集存储路径为空")

    if request.generation_mode in ("doc_to_training", "qa_extraction", "doc_to_eval"):
        # 文档模式需要文档输入源
        if not input_path:
            raise HTTPException(status_code=400, detail="需要文档输入源（dataset_id 或 input_path）")

    if request.generation_mode in ("doc_to_training", "qa_to_training"):
        # 训练数据生成模式：retrieval 需要 embedding
        # 使用已有集合时可由集合注册表回填 embedding 配置（见下方约 763-784 行的
        # auto-fill），因此仅当既无 embedding_config 又未选择已有集合时才报错。
        if (
            request.pos_neg_method == "retrieval"
            and not request.embedding_config
            and not request.milvus_collection_name
        ):
            raise HTTPException(status_code=400, detail="pos_neg_method=retrieval 需要配置 embedding 模型或选择已有集合")
        # rerank 不能单独使用，必须配合 embedding
        if request.rerank_config and not request.embedding_config:
            raise HTTPException(status_code=400, detail="Rerank 模型需要配合 Embedding 模型使用")
        # 记录 embedding config_id（用于去重判断）
        if request.embedding_config and request.embedding_config.config_id:
            embedding_config_id = request.embedding_config.config_id

    if request.generation_mode in ("doc_to_eval", "qa_to_eval"):
        # 评估数据生成模式：必须有 embedding
        if not request.embedding_config:
            raise HTTPException(status_code=400, detail="评估数据生成模式必须配置 Embedding 模型")
        if request.embedding_config.config_id:
            embedding_config_id = request.embedding_config.config_id
        if request.rerank_config and not request.embedding_config:
            raise HTTPException(status_code=400, detail="Rerank 模型需要配合 Embedding 模型使用")

    # 验证已有集合（如果指定）
    existing_collection_name = None
    if request.milvus_collection_name:
        reg = milvus_collection_service.get_by_name(request.milvus_collection_name)
        if not reg:
            raise HTTPException(status_code=404, detail=f"集合 '{request.milvus_collection_name}' 未注册")
        reg = _verify_owned_resource(reg, current_user, "Milvus collection")
        if reg.get("status") != "active":
            raise HTTPException(
                status_code=409,
                detail="The selected Milvus collection is being deleted",
            )
        existing_collection_name = request.milvus_collection_name

        # 从注册表获取 embedding 配置，覆盖/补充请求中的 embedding 信息
        if reg.get("embedding_config_id"):
            embedding_config_id = reg["embedding_config_id"]
            # 如果用户没有提供 embedding_config，从注册表和模型配置自动填充
            if not embedding_config_data and reg.get("embedding_config_id"):
                config = _resolve_model_config(
                    reg["embedding_config_id"],
                    current_user,
                    {"embedding"},
                    allow_public=False,
                )
                embedding_config_data = {
                    "config_id": reg["embedding_config_id"],
                    "endpoint": config.get("api_endpoint"),
                    "model": config.get("model_name"),
                    "api_key": config.get("api_key"),
                    "batch_size": 32,
                    "concurrency": 20,
                    "similarity_threshold": 0.85,
                    "retrieval_top_k": 10,
                }

    # retrieval 已请求，但请求与所选集合都无法提供可用的 embedding 配置时，在此
    # 明确报错，而不是让检索流程在运行时才失败（例如集合是用直连 embedding 注册的，
    # 其 embedding_config_id 为空，auto-fill 无法回填）。
    if (
        request.generation_mode in ("doc_to_training", "qa_to_training")
        and request.pos_neg_method == "retrieval"
        and not embedding_config_data
    ):
        raise HTTPException(
            status_code=400,
            detail="pos_neg_method=retrieval 需要可用的 embedding 配置：请提供 embedding 模型，或选择已注册且带 embedding 配置的集合",
        )

    if input_path:
        input_path = _resolve_generation_input_path(
            input_path,
            map_host_path=not bool(request.dataset_id),
        )
    if not request.dataset_id and not _is_anonymous_user(current_user):
        raise HTTPException(
            status_code=403,
            detail=(
                "Direct generation input paths are disabled while authentication "
                "is enabled; select an owned dataset_id"
            ),
        )

    # Resolve checkpoint reuse before admission so a lookup failure cannot
    # strand a newly-created pending task or its execution lease.
    skip_filter = False
    previous_filter_stats = None
    if (
        request.generation_mode in ("qa_to_training", "doc_to_training")
        and embedding_config_id
        and request.dataset_id
    ):
        existing_task = generation_task_service.find_completed_task(
            source_dataset_id=request.dataset_id,
            embedding_config_id=embedding_config_id,
            user_id=current_user.get("user_id"),
        )
        if existing_task:
            previous_filter_stats = existing_task.get("filter_stats")
            prev_threshold = None
            prev_dedup_enabled = None
            if previous_filter_stats:
                prev_threshold = previous_filter_stats.get("threshold")
                prev_dedup_enabled = previous_filter_stats.get("dedup_enabled")

            current_threshold = (
                embedding_config_data.get("similarity_threshold", 0.85)
                if embedding_config_data
                else 0.85
            )
            current_dedup_enabled = bool(
                post_process_config
                and post_process_config.get("dedup", {}).get("enabled")
            )
            threshold_match = prev_threshold is not None and abs(
                prev_threshold - current_threshold
            ) < 1e-6
            dedup_match = (
                prev_dedup_enabled == current_dedup_enabled
                if prev_dedup_enabled is not None
                else True
            )

            if threshold_match and dedup_match:
                skip_filter = True
                logger.info(
                    "Found existing completed task %s with same "
                    "dataset/embedding/threshold, skipping filter",
                    existing_task.get("task_id"),
                )

    # 创建任务到数据库
    try:
        task, execution_lease = background_task_admission_service.admit_execution(
            "generation",
            None,
            current_user.get("user_id"),
            generation_task_service.create_task,
            task_name=request.task_name,
            input_path=input_path,
            input_format=request.input_format,
            output_format=request.output_format,
            generation_mode=request.generation_mode,
            pos_neg_method=request.pos_neg_method,
            llm_config=llm_config_data,
            eval_llm_config=eval_llm_config_data,
            embedding_config=embedding_config_data,
            rerank_config=rerank_config_data,
            worker_config=worker_config,
            steps_config=steps_config,
            post_process_config=post_process_config,
            custom_prompts=request.prompts,
            content_field=content_field,
            auto_register_dataset=request.auto_register_dataset,
            user_id=current_user.get("user_id"),
            source_dataset_id=request.dataset_id,
            embedding_config_id=embedding_config_id,
            milvus_collection=existing_collection_name,
            similarity_threshold=(
                embedding_config_data.get("similarity_threshold", 0.85)
                if embedding_config_data
                else 0.85
            ),
            retrieval_top_k=(
                embedding_config_data.get("retrieval_top_k", 10)
                if embedding_config_data
                else 10
            ),
            require_source_dataset=bool(request.dataset_id),
        )
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MilvusCollectionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatasetConsumptionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    task_id = task["task_id"]
    created_attempt = generation_task_service.get_task_raw(task_id)
    scheduled_run_token = (
        created_attempt.get("run_token") if created_attempt else None
    )
    if not scheduled_run_token:
        execution_lease.release()
        raise RuntimeError("Generation task was created without an attempt token")

    try:
        # 统一输出路径：所有生成产物放在固定目录下
        output_path = str(
            resolve_generation_attempt_output_path(
                os.path.join(
                    GENERATION_OUTPUT_DIR,
                    f"generated_{task_id}.jsonl",
                ),
                task_id,
                scheduled_run_token,
            )
        )
        os.makedirs(GENERATION_OUTPUT_DIR, exist_ok=True)
        if not generation_task_service.set_output(
            task_id,
            output_path,
            0,
            expected_status=GenerationStatus.PENDING,
            expected_run_token=scheduled_run_token,
        ):
            raise RuntimeError("Generation task lost ownership before scheduling")

        # 启动后台任务
        _llm_cfg = llm_config_data or {}
        config = PipelineConfig(
            input_path=input_path,
            input_format=request.input_format,
            content_field=content_field,
            generation_mode=request.generation_mode,
            pos_neg_method=request.pos_neg_method,
            output_path=output_path,
            output_format=request.output_format,
            llm_config=_llm_cfg,
            eval_llm_config=eval_llm_config_data,
            llm_concurrency=_llm_cfg.get("concurrency", 10),
            embedding_concurrency=embedding_config_data.get("concurrency", 20) if embedding_config_data else 20,
            timeout_per_doc=worker_config.get("timeout_per_doc", 300),
            steps=steps_config,
            post_process=post_process_config,
            custom_prompts=request.prompts,
            embedding_config=embedding_config_data,
            rerank_config=rerank_config_data,
            source_dataset_id=request.dataset_id,
            similarity_threshold=embedding_config_data.get("similarity_threshold", 0.85) if embedding_config_data else 0.85,
            retrieval_top_k=embedding_config_data.get("retrieval_top_k", 10) if embedding_config_data else 10,
            skip_filter=skip_filter,
            previous_filter_stats=previous_filter_stats,
            task_id=task_id,
            run_token=scheduled_run_token,
            existing_collection_name=existing_collection_name,
            allowed_input_dirs=get_generation_input_allowed_dirs(),
            user_id=current_user.get("user_id"),
        )
        validate_generation_resource_config(config)
        response = TaskResponse(
            task_id=task_id,
            task_name=task["task_name"],
            status=task["status"],
            generation_mode=task.get("generation_mode", "doc_to_training"),
            pos_neg_method=task.get("pos_neg_method"),
            progress=task["progress"],
            total_docs=task["total_docs"],
            processed_docs=task["processed_docs"],
            output_sample_count=task["output_sample_count"],
        )
        _clear_generation_cancellation(task_id)
        background_tasks.add_task(
            background_task_admission_service.run_async,
            execution_lease,
            _run_generation_task,
            task_id=task_id,
            config=config,
            expected_run_token=scheduled_run_token,
        )
    except Exception:
        execution_lease.release()
        try:
            failed = generation_task_service.update_status(
                task_id,
                GenerationStatus.FAILED,
                "Task could not be scheduled for background execution",
                expected_run_token=scheduled_run_token,
            )
            if not failed:
                logger.info(
                    "Generation task %s changed state before scheduling failure "
                    "could be persisted",
                    task_id,
                )
        except Exception:
            logger.exception(
                "Failed to finalize unscheduled generation task %s",
                task_id,
            )
        raise

    return response


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = None,
    generation_mode: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """列出所有生成任务"""
    user_id = None if _is_anonymous_user(current_user) else current_user.get("user_id")

    tasks, total = generation_task_service.get_all_tasks(
        status=status,
        generation_mode=generation_mode,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )

    task_responses = [
        TaskResponse(
            task_id=task["task_id"],
            task_name=task["task_name"],
            status=task["status"],
            generation_mode=task.get("generation_mode", "doc_to_training"),
            pos_neg_method=task.get("pos_neg_method"),
            progress=task["progress"],
            total_docs=task["total_docs"],
            processed_docs=task["processed_docs"],
            output_sample_count=task["output_sample_count"],
            error_message=public_task_error_message(task["error_message"]),
            created_at=(
                task["created_at"]
                if isinstance(task.get("created_at"), str)
                else (task["created_at"].isoformat() if task.get("created_at") else None)
            ),
        )
        for task in tasks
    ]
    return TaskListResponse(
        tasks=task_responses,
        total=total,
        stats=TaskStatsResponse(**generation_task_service.get_task_stats(user_id=user_id)),
        limit=limit,
        offset=offset,
    )


@router.get("/tasks/{task_id}", response_model=TaskDetailResponse)
async def get_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取任务详情（完整配置）"""
    task = _verify_task_access(task_id, current_user)
    return TaskDetailResponse(
        task_id=task["task_id"],
        task_name=task["task_name"],
        description=task.get("description"),
        status=task["status"],
        progress=task["progress"],
        total_docs=task["total_docs"],
        processed_docs=task["processed_docs"],
        output_sample_count=task["output_sample_count"],
        error_message=public_task_error_message(task.get("error_message")),
        # 输入配置
        input_path=task["input_path"],
        input_format=task["input_format"],
        content_field=task.get("content_field"),
        generation_mode=task["generation_mode"],
        pos_neg_method=task.get("pos_neg_method"),
        # 输出配置
        output_path=task.get("output_path"),
        output_format=task["output_format"],
        output_dataset_id=task.get("output_dataset_id"),
        auto_register_dataset=task.get("auto_register_dataset", True),
        # 模型配置
        llm_config=task.get("llm_config"),
        eval_llm_config=task.get("eval_llm_config"),
        embedding_config=task.get("embedding_config"),
        rerank_config=task.get("rerank_config"),
        # 处理配置
        worker_config=task.get("worker_config"),
        steps_config=task.get("steps_config"),
        post_process_config=task.get("post_process_config"),
        # Training 模式配置
        source_dataset_id=task.get("source_dataset_id"),
        embedding_config_id=task.get("embedding_config_id"),
        milvus_collection=task.get("milvus_collection"),
        filter_stats=task.get("filter_stats"),
        # Doc-to-Training 中间产物
        qa_output_path=task.get("qa_output_path"),
        qa_dataset_id=task.get("qa_dataset_id"),
        qa_filtered_path=task.get("qa_filtered_path"),
        qa_filtered_dataset_id=task.get("qa_filtered_dataset_id"),
        # 深度评估数据集
        deep_eval_path=task.get("deep_eval_path"),
        deep_eval_dataset_id=task.get("deep_eval_dataset_id"),
        # 阶段状态视图
        stages=_build_task_stages(task),
        # 时间戳
        created_at=task.get("created_at"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
    )


@router.get("/tasks/{task_id}/progress", response_model=TaskProgressResponse)
async def get_task_progress(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取任务进度"""
    task = _verify_task_access(task_id, current_user)
    return TaskProgressResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=task["progress"],
        total_docs=task["total_docs"],
        processed_docs=task["processed_docs"],
        output_sample_count=task["output_sample_count"],
    )


@router.post("/tasks/{task_id}/stop")
async def stop_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """停止任务"""
    _verify_task_access(task_id, current_user)
    task = generation_task_service.get_task_raw(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Generation task not found")
    if task["status"] == GenerationStatus.STOPPED:
        return {"status": "stopped"}
    if task["status"] == GenerationStatus.STOPPING:
        return {"status": "stopping"}
    if task["status"] not in {
        GenerationStatus.RUNNING,
        GenerationStatus.PENDING,
    }:
        raise HTTPException(
            status_code=409,
            detail=f"Generation task is already {task['status']}",
        )

    run_token = task.get("run_token")
    pipeline, pipeline_token = _get_running_pipeline(task_id)
    owned_pipeline = pipeline if pipeline_token == run_token else None
    cancellation_request = _request_generation_cancellation(
        task_id,
        owner=owned_pipeline,
        run_token=run_token,
    )
    requested_status = (
        GenerationStatus.STOPPED
        if task["status"] == GenerationStatus.PENDING
        else GenerationStatus.STOPPING
    )
    try:
        stopped = generation_task_service.update_status(
            task_id,
            requested_status,
            expected_run_token=run_token,
        )
    except Exception:
        _clear_generation_cancellation(task_id, cancellation_request)
        raise

    if not stopped:
        _clear_generation_cancellation(task_id, cancellation_request)
        current = generation_task_service.get_task_raw(task_id)
        current_status = current.get("status") if current else "missing"
        if current and current.get("run_token") == run_token:
            if current_status == GenerationStatus.STOPPED:
                return {"status": "stopped"}
            if current_status == GenerationStatus.STOPPING:
                return {"status": "stopping"}
        raise HTTPException(
            status_code=409,
            detail=f"Generation task is now {current_status}; stop was not applied",
        )

    if owned_pipeline:
        try:
            owned_pipeline.stop()
        except Exception as exc:
            logger.warning("Failed to stop generation pipeline %s: %s", task_id, exc)
    if requested_status == GenerationStatus.STOPPED:
        _finalize_stopped_generation(task_id, "Generation stopped by user")
        return {"status": "stopped"}
    return {"status": "stopping"}


def _resolve_generation_delete_path(path: str) -> Path:
    """Resolve and validate one managed local cleanup target without deleting it."""
    if not path:
        raise ValueError("Generation storage cleanup requires a path")

    settings = get_settings()
    trusted_roots = {
        Path(settings.datasets_dir).resolve(),
        Path(settings.local_cache_dir).resolve(),
        Path(os.environ.get("GENERATION_OUTPUT_DIR", str(settings.datasets_dir))).resolve(),
        Path(os.environ.get("SYNC_DATA_DIR", "/app/data/sync")).resolve(),
    }

    try:
        target = Path(path).resolve()
    except Exception as exc:
        raise ValueError("Generation storage cleanup refused an invalid path") from exc

    rel = None
    for root in trusted_roots:
        try:
            rel = target.relative_to(root)
            break
        except ValueError:
            continue

    if rel is None:
        raise ValueError("Generation storage cleanup refused an unmanaged path")
    if not rel.parts:
        raise ValueError("Generation storage cleanup refused a managed root")
    if target.is_dir() and len(rel.parts) < 2:
        raise ValueError("Generation storage cleanup refused a broad directory")
    return target


def _safe_delete_path(path: str):
    """Delete generation artifacts only under trusted managed roots."""
    import shutil

    target = _resolve_generation_delete_path(path)

    try:
        if target.is_file():
            os.remove(target)
            logger.info(f"Deleted file: {target}")
        elif target.is_dir():
            shutil.rmtree(target)
            logger.info(f"Deleted directory: {target}")
    except OSError as exc:
        raise RuntimeError("Generation storage cleanup failed") from exc


def _resolve_generation_s3_target(storage_uri: str):
    """Resolve one managed S3 target without mutating remote storage."""
    if not storage_uri or not storage_uri.startswith("s3://"):
        raise ValueError("Generation storage cleanup requires an S3 URI")
    from ...storage.object_store import get_object_store, uri_to_key

    store = get_object_store()
    object_key = uri_to_key(storage_uri, expected_bucket=store.bucket)
    return store, object_key


def _safe_delete_s3_object(storage_uri: str):
    """Delete an S3 object by URI."""
    try:
        store, object_key = _resolve_generation_s3_target(storage_uri)
        store.delete_object(object_key)
    except Exception as exc:
        raise RuntimeError("Generation storage cleanup failed") from exc


def _validate_generation_cleanup_manifest(
    local_paths: set[str],
    s3_uris: set[str],
) -> None:
    """Validate every storage target before the first destructive call."""
    for storage_uri in sorted(s3_uris):
        _resolve_generation_s3_target(storage_uri)
    for storage_path in sorted(local_paths):
        _resolve_generation_delete_path(storage_path)


def _restore_generation_dataset_fences(
    dataset_records: Dict[str, Dict[str, Any]],
    original_statuses: Dict[str, str],
    deletion_owner: str,
    owner_user_id: Optional[str],
) -> bool:
    restored = True
    for dataset_id in reversed(list(dataset_records)):
        original_status = original_statuses[dataset_id]
        if original_status == "deleting":
            continue
        restored = (
            dataset_service.restore_from_deleting(
                dataset_id,
                deletion_owner=deletion_owner,
                status=original_status,
                user_id=owner_user_id,
            )
            and restored
        )
    return restored


_GENERATION_DATASET_DELETE_PAGE_SIZE = 500


def _list_generation_output_datasets(
    task_id: str,
) -> List[Dict[str, Any]]:
    """Collect a complete, stable snapshot of one task's output datasets."""
    datasets: List[Dict[str, Any]] = []
    seen_ids = set()
    expected_total: Optional[int] = None
    offset = 0
    while True:
        page, total = dataset_service.list_datasets(
            source_task_type="generation",
            source_task_id=task_id,
            limit=_GENERATION_DATASET_DELETE_PAGE_SIZE,
            offset=offset,
        )
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise RuntimeError("Generation dataset listing returned an invalid total")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise RuntimeError("Generation dataset listing changed during deletion")
        if not page:
            if len(datasets) != total:
                raise RuntimeError("Generation dataset listing ended before completion")
            return datasets

        page_ids = [dataset.get("dataset_id") for dataset in page]
        if any(not isinstance(dataset_id, str) or not dataset_id for dataset_id in page_ids):
            raise RuntimeError("Generation dataset listing returned an invalid ID")
        if seen_ids.intersection(page_ids) or len(set(page_ids)) != len(page_ids):
            raise RuntimeError("Generation dataset listing did not advance")
        seen_ids.update(page_ids)
        datasets.extend(page)
        offset += len(page)
        if len(datasets) == total:
            return datasets
        if len(datasets) > total:
            raise RuntimeError("Generation dataset listing exceeded its total")


def _cascade_delete_generation_outputs(
    task_id: str,
    task: Dict[str, Any],
    on_preflight_failure: Optional[Callable[[], None]] = None,
) -> List[str]:
    owner_user_id = task.get("user_id")
    deletion_owner = f"generation:{task_id}"
    dataset_records: Dict[str, Dict[str, Any]] = {}
    original_statuses: Dict[str, str] = {}
    destructive_cleanup_started = False

    def _restore_pre_destructive_state() -> None:
        if destructive_cleanup_started:
            return
        if dataset_records and not _restore_generation_dataset_fences(
            dataset_records,
            original_statuses,
            deletion_owner,
            owner_user_id,
        ):
            raise RuntimeError(
                "Could not restore generation dataset deletion state"
            )
        has_existing_dataset_intent = any(
            original_statuses.get(dataset_id) == "deleting"
            for dataset_id in dataset_records
        )
        if on_preflight_failure is not None and not has_existing_dataset_intent:
            on_preflight_failure()

    try:
        dataset_ids = {
            edge.get("to_dataset_id")
            for edge in dataset_lineage_service.get_edges_by_task(task_id)
            if edge.get("to_dataset_id")
        }
        datasets = _list_generation_output_datasets(task_id)
        dataset_ids.update(
            dataset["dataset_id"]
            for dataset in datasets
            if dataset.get("dataset_id")
        )

        # Validate the complete provenance snapshot before fencing anything.
        # Otherwise a late foreign-lineage record could strand earlier rows in
        # deleting state even though destructive cleanup never started.
        validated_datasets: Dict[str, Dict[str, Any]] = {}
        for dataset_id in sorted(dataset_ids):
            dataset = dataset_service.get_dataset(dataset_id)
            if not dataset:
                continue
            if owner_user_id and dataset.get("user_id") != owner_user_id:
                raise PermissionError(
                    "Generation output dataset owner changed during deletion"
                )
            if (
                dataset.get("source_task_type") != "generation"
                or dataset.get("source_task_id") != task_id
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Dataset is not owned by the generation task",
                )
            original_statuses[dataset_id] = dataset.get("status") or "ready"
            validated_datasets[dataset_id] = dataset

        # Fence every validated output durably before checking consumers.
        for dataset_id, dataset in validated_datasets.items():
            if not dataset_service.mark_deleting(
                dataset_id,
                deletion_owner=deletion_owner,
                user_id=owner_user_id,
            ):
                raise RuntimeError(
                    "Generation output dataset disappeared during deletion"
                )
            dataset_records[dataset_id] = dataset

        dataset_paths = sorted(
            {
                reference
                for dataset in dataset_records.values()
                for reference in (
                    dataset.get("storage_path"),
                    dataset.get("storage_uri"),
                )
                if isinstance(reference, str) and reference
            }
        )
        active_generation_consumers = (
            generation_task_service.list_active_dataset_consumers(
                list(dataset_records)
            )
        )
        active_training_consumers = (
            training_task_service.list_active_dataset_consumers(dataset_paths)
        )
        active_evaluation_consumers = (
            evaluation_task_service.list_active_dataset_consumers(
                list(dataset_records),
                dataset_paths,
            )
        )
        milvus_links = milvus_collection_service.list_dataset_links(
            dataset_records
        )
        if (
            active_generation_consumers
            or active_training_consumers
            or active_evaluation_consumers
            or milvus_links
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot cascade delete datasets with active consumers or "
                    "Milvus links. "
                    f"generation={active_generation_consumers[:5]}, "
                    f"training={active_training_consumers[:5]}, "
                    f"evaluation={active_evaluation_consumers[:5]}, "
                    f"milvus_link_count={len(milvus_links)}"
                ),
            )

        local_paths = set()
        s3_uris = set()
        for dataset_id, dataset in dataset_records.items():
            storage_path = dataset.get("storage_path")
            storage_uri = dataset.get("storage_uri")
            if isinstance(storage_path, str) and storage_path:
                local_paths.add(storage_path)
            if isinstance(storage_uri, str) and storage_uri:
                if not storage_uri.startswith("s3://"):
                    raise ValueError("Generation dataset has an invalid storage URI")
                s3_uris.add(storage_uri)
            for asset in dataset_asset_service.list_assets(dataset_id=dataset_id):
                reference = asset.get("storage_uri")
                if not isinstance(reference, str) or not reference:
                    continue
                if reference.startswith("s3://"):
                    s3_uris.add(reference)
                else:
                    local_paths.add(reference)

        # Include task artifacts that do not belong to a different dataset.
        for path_key in (
            "output_path",
            "qa_output_path",
            "qa_filtered_path",
            "deep_eval_path",
        ):
            path = task.get(path_key)
            if not path:
                continue
            referencing_dataset = dataset_service.get_dataset_by_storage_path(
                path,
                user_id=None,
            )
            if (
                referencing_dataset
                and referencing_dataset.get("dataset_id") not in dataset_records
            ):
                continue
            local_paths.add(path)

        external_storage_consumers = (
            dataset_service.list_external_storage_reference_consumers(
                local_paths | s3_uris,
                exclude_dataset_ids=dataset_records,
            )
        )
        if external_storage_consumers:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot cascade delete storage shared by another dataset "
                    "or dataset asset. "
                    f"external_reference_count={len(external_storage_consumers)}"
                ),
            )
        generation_artifact_consumers = (
            generation_task_service.list_artifact_reference_consumers(
                local_paths | s3_uris,
                exclude_task_ids=(task_id,),
            )
        )
        if generation_artifact_consumers:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot cascade delete storage referenced by another "
                    "generation task. "
                    f"generation_reference_count={len(generation_artifact_consumers)}"
                ),
            )
        manifest_paths = sorted(local_paths | s3_uris)
        active_manifest_training_consumers = (
            training_task_service.list_active_dataset_consumers(manifest_paths)
        )
        active_manifest_evaluation_consumers = (
            evaluation_task_service.list_active_dataset_consumers(
                list(dataset_records),
                manifest_paths,
            )
        )
        if (
            active_manifest_training_consumers
            or active_manifest_evaluation_consumers
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot cascade delete storage used by an active task. "
                    f"training={active_manifest_training_consumers[:5]}, "
                    f"evaluation={active_manifest_evaluation_consumers[:5]}"
                ),
            )

        _validate_generation_cleanup_manifest(local_paths, s3_uris)
        destructive_cleanup_started = True
        for storage_uri in sorted(s3_uris):
            _safe_delete_s3_object(storage_uri)
        for storage_path in sorted(local_paths):
            _safe_delete_path(storage_path)

        if not generation_task_service.delete_task_with_datasets(
            task_id,
            sorted(dataset_records),
            deletion_owner=deletion_owner,
            user_id=owner_user_id,
        ):
            raise RuntimeError("Generation task disappeared during cascade deletion")
        return sorted(dataset_records)
    except DatasetDeletionOwnerConflictError as exc:
        if not destructive_cleanup_started:
            _restore_pre_destructive_state()
        raise HTTPException(
            status_code=409,
            detail="Dataset is owned by another deletion operation",
        ) from exc
    except PermissionError as exc:
        if not destructive_cleanup_started:
            _restore_pre_destructive_state()
        raise HTTPException(
            status_code=409,
            detail="Generation output dataset ownership changed",
        ) from exc
    except HTTPException:
        if not destructive_cleanup_started:
            _restore_pre_destructive_state()
        raise
    except Exception as exc:
        if not destructive_cleanup_started:
            try:
                _restore_pre_destructive_state()
            except Exception as rollback_exc:
                logger.error(
                    "Generation cascade preflight rollback failed for %s "
                    "(error_type=%s)",
                    task_id,
                    type(rollback_exc).__name__,
                )
        logger.error(
            "Generation cascade cleanup failed for %s (error_type=%s)",
            task_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Generation cascade cleanup failed; task and dataset metadata "
                "were retained"
            ),
        ) from exc


async def _delete_task_impl(
    task_id: str,
    cascade: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user),
    *,
    task_snapshot: Optional[Dict[str, Any]] = None,
    on_preflight_failure: Optional[Callable[[], None]] = None,
):
    """删除任务及其产出文件。cascade=true 时同时删除关联数据集、血缘、资产。"""
    task = task_snapshot or _verify_task_access(task_id, current_user)

    if task["status"] in {
        GenerationStatus.PENDING,
        GenerationStatus.RUNNING,
        GenerationStatus.STOPPING,
        GenerationStatus.PUBLISHING,
        GenerationStatus.RECOVERING,
        GenerationStatus.RESTARTING,
    }:
        raise HTTPException(status_code=400, detail="Cannot delete running task")
    expected_intent = (
        GenerationStatus.DELETING_CASCADE
        if cascade
        else GenerationStatus.DELETING
    )
    if task["status"] in {
        GenerationStatus.DELETING,
        GenerationStatus.DELETING_CASCADE,
    } and task["status"] != expected_intent:
        raise HTTPException(
            status_code=409,
            detail="Generation task deletion mode cannot change after cleanup starts",
        )
    try:
        _reject_pending_sync_tracking(task_id)
    except Exception:
        if on_preflight_failure is not None:
            on_preflight_failure()
        raise
    deleted_datasets: List[str] = []

    if cascade:
        deleted_datasets = _cascade_delete_generation_outputs(
            task_id,
            task,
            on_preflight_failure=on_preflight_failure,
        )
        return {
            "status": "deleted",
            "deleted_datasets": deleted_datasets,
        }

    # Non-cascade cleanup still needs a complete validated manifest before the
    # first unlink; otherwise a later read/validation failure can strand a
    # terminal task after only some artifacts were removed.
    local_paths: set[str] = set()
    try:
        for path_key in (
            "output_path",
            "qa_output_path",
            "qa_filtered_path",
            "deep_eval_path",
        ):
            path = task.get(path_key)
            if not path:
                continue
            dataset = dataset_service.get_dataset_by_storage_path(
                path,
                user_id=None,
            )
            if dataset:
                logger.info(
                    "Skip deleting %s because dataset %s still references it",
                    path,
                    dataset.get("dataset_id"),
                )
                continue
            external_storage_consumers = (
                dataset_service.list_external_storage_reference_consumers(
                    [path],
                )
            )
            if external_storage_consumers:
                logger.info(
                    "Skip deleting %s because a dataset asset still references it",
                    path,
                )
                continue
            generation_artifact_consumers = (
                generation_task_service.list_artifact_reference_consumers(
                    [path],
                    exclude_task_ids=(task_id,),
                )
            )
            if generation_artifact_consumers:
                logger.info(
                    "Skip deleting %s because another generation task "
                    "still references it",
                    path,
                )
                continue
            active_training_consumers = (
                training_task_service.list_active_dataset_consumers([path])
            )
            active_evaluation_consumers = (
                evaluation_task_service.list_active_dataset_consumers([], [path])
            )
            if active_training_consumers or active_evaluation_consumers:
                logger.info(
                    "Skip deleting %s because an active training or evaluation "
                    "task still references it",
                    path,
                )
                continue
            local_paths.add(path)
        _validate_generation_cleanup_manifest(local_paths, set())
    except Exception:
        if on_preflight_failure is not None:
            on_preflight_failure()
        raise

    for path in sorted(local_paths):
        _safe_delete_path(path)

    delete_kwargs: Dict[str, Any] = {}
    if task_snapshot is not None:
        delete_kwargs = {
            "expected_status": GenerationStatus.DELETING,
            "user_id": task.get("user_id"),
        }
    success = generation_task_service.delete_task(task_id, **delete_kwargs)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete task")

    return {"status": "deleted"}


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    cascade: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a terminal task while atomically blocking concurrent restart."""
    task = _verify_task_access(task_id, current_user)
    if task["status"] in {
        GenerationStatus.PENDING,
        GenerationStatus.RUNNING,
        GenerationStatus.STOPPING,
        GenerationStatus.PUBLISHING,
        GenerationStatus.RECOVERING,
        GenerationStatus.RESTARTING,
    }:
        raise HTTPException(status_code=400, detail="Cannot delete running task")
    requested_status = (
        GenerationStatus.DELETING_CASCADE
        if cascade
        else GenerationStatus.DELETING
    )
    if task["status"] in {
        GenerationStatus.DELETING,
        GenerationStatus.DELETING_CASCADE,
    } and task["status"] != requested_status:
        raise HTTPException(
            status_code=409,
            detail="Generation task deletion mode cannot change after cleanup starts",
        )
    try:
        deletion_guard = background_task_admission_service.begin_deletion(
            "generation",
            task_id,
        )
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(
            status_code=409,
            detail="Generation worker is still shutting down; retry deletion shortly",
        ) from exc

    try:
        try:
            intent = generation_task_service.begin_task_deletion(
                task_id,
                cascade=cascade,
                user_id=task.get("user_id"),
            )
        except PermissionError as exc:
            raise HTTPException(
                status_code=409,
                detail="Generation task ownership changed during deletion",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if intent is None:
            raise HTTPException(status_code=404, detail="Generation task not found")

        rollback_finished = False

        def restore_new_parent_intent() -> None:
            nonlocal rollback_finished
            if rollback_finished or not intent.newly_started:
                return
            if not intent.previous_status:
                raise RuntimeError(
                    "Generation deletion rollback is missing its prior status"
                )
            if not generation_task_service.cancel_task_deletion(
                task_id,
                cascade=cascade,
                previous_status=intent.previous_status,
                user_id=task.get("user_id"),
            ):
                raise RuntimeError(
                    "Could not restore generation pre-destructive deletion state"
                )
            rollback_finished = True

        return await _delete_task_impl(
            task_id,
            cascade,
            current_user,
            task_snapshot=intent.task,
            on_preflight_failure=restore_new_parent_intent,
        )
    finally:
        deletion_guard.release()


class PreviewResponse(BaseModel):
    """预览数据响应"""
    total: int
    samples: List[Dict[str, Any]]


_PREVIEW_SOURCE_MAP = {
    "output": "output_path",
    "qa_full": "qa_output_path",
    "qa_filtered": "qa_filtered_path",
    "deep_eval": "deep_eval_path",
}


def _infer_stage_key_from_dataset(task: Dict[str, Any], dataset: Dict[str, Any]) -> str:
    """Infer stage key for a dataset artifact with backward compatibility."""
    stage_alias = {
        "qa_filtered": "qa_filter",
        "deep_eval": "deep_eval_export",
    }
    extra_metadata = dataset.get("extra_metadata") or {}
    stage_key = extra_metadata.get("stage")
    if isinstance(stage_key, str) and stage_key.strip():
        return stage_alias.get(stage_key.strip(), stage_key.strip())

    dataset_id = dataset.get("dataset_id")
    if dataset_id and dataset_id == task.get("qa_dataset_id"):
        return "qa_extraction"
    if dataset_id and dataset_id == task.get("qa_filtered_dataset_id"):
        return "qa_filter"
    if dataset_id and dataset_id == task.get("deep_eval_dataset_id"):
        return "deep_eval_export"
    if dataset_id and dataset_id == task.get("output_dataset_id"):
        if task.get("generation_mode") in ("doc_to_eval", "qa_to_eval"):
            return "deep_eval_export"
        return "pos_neg_generation"

    return "legacy_pipeline"


def _infer_artifact_role(task: Dict[str, Any], dataset: Dict[str, Any], stage_key: str) -> str:
    """Infer artifact role for display."""
    extra_metadata = dataset.get("extra_metadata") or {}
    explicit_role = extra_metadata.get("artifact_role")
    if isinstance(explicit_role, str) and explicit_role.strip():
        return explicit_role.strip()

    dataset_id = dataset.get("dataset_id")
    if dataset_id and dataset_id == task.get("output_dataset_id"):
        return "output"
    if stage_key in ("qa_extraction", "qa_filter", "retrieval_build"):
        return "intermediate"
    if stage_key == "deep_eval_export":
        return "eval"
    return "intermediate"


@router.get("/tasks/{task_id}/artifacts", response_model=TaskArtifactsResponse)
async def list_task_artifacts(
    task_id: str,
    include_empty: bool = True,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """按阶段聚合返回 generation 任务产物（数据集）。"""
    task = _verify_task_access(task_id, current_user)

    # 1) 收集该任务自动注册的数据集产物
    datasets: List[Dict[str, Any]] = []
    offset = 0
    limit = 200
    total = 0
    while True:
        page, total = dataset_service.list_datasets(
            source_task_type="generation",
            source_task_id=task_id,
            user_id=task.get("user_id"),
            limit=limit,
            offset=offset,
        )
        if not page:
            break
        datasets.extend(page)
        offset += len(page)
        if offset >= total:
            break

    dataset_by_id = {
        ds.get("dataset_id"): ds
        for ds in datasets
        if ds.get("dataset_id")
    }

    # 2) 回填任务详情里显式记录的 dataset_id（兼容历史/异常数据）
    for explicit_id in (
        task.get("qa_dataset_id"),
        task.get("qa_filtered_dataset_id"),
        task.get("deep_eval_dataset_id"),
        task.get("output_dataset_id"),
    ):
        if not explicit_id or explicit_id in dataset_by_id:
            continue
        ds = dataset_service.get_dataset(explicit_id)
        if not ds:
            continue
        if ds.get("source_task_type") == "generation" and ds.get("source_task_id") == task_id:
            dataset_by_id[explicit_id] = ds

    stage_view = _build_task_stages(task)
    stage_index = {
        s.get("stage"): {
            "name": s.get("label") or s.get("stage"),
            "order": idx,
            "status": s.get("status") or "pending",
        }
        for idx, s in enumerate(stage_view)
    }

    grouped: Dict[str, TaskStageArtifactsResponse] = {}
    for ds in dataset_by_id.values():
        stage_key = _infer_stage_key_from_dataset(task, ds)
        stage_meta = stage_index.get(stage_key, {})
        stage_name = str(stage_meta.get("name") or stage_key)
        step_order = int(stage_meta.get("order", 999))
        stage_status = str(stage_meta.get("status") or "completed")

        if stage_key not in grouped:
            grouped[stage_key] = TaskStageArtifactsResponse(
                stage_key=stage_key,
                stage_name=stage_name,
                step_order=step_order,
                stage_status=stage_status,
                artifacts=[],
            )

        grouped[stage_key].artifacts.append(
            TaskArtifactItemResponse(
                dataset_id=ds["dataset_id"],
                dataset_name=ds.get("dataset_name") or ds["dataset_id"],
                stage_key=stage_key,
                stage_name=stage_name,
                artifact_role=_infer_artifact_role(task, ds, stage_key),
                dataset_type=ds.get("dataset_type") or "custom",
                usage=ds.get("usage") or "raw",
                file_format=ds.get("file_format"),
                num_rows=ds.get("num_rows"),
                file_size=ds.get("file_size"),
                status=ds.get("status") or "ready",
                tags=ds.get("tags"),
                created_at=ds.get("created_at"),
            )
        )

    if include_empty:
        for stage_key, meta in stage_index.items():
            if stage_key in grouped:
                continue
            grouped[stage_key] = TaskStageArtifactsResponse(
                stage_key=stage_key,
                stage_name=str(meta.get("name") or stage_key),
                step_order=int(meta.get("order", 999)),
                stage_status=str(meta.get("status") or "pending"),
                artifacts=[],
            )

    stages = sorted(grouped.values(), key=lambda s: (s.step_order, s.stage_name))
    for stage in stages:
        stage.artifacts.sort(
            key=lambda a: (a.created_at or "", a.dataset_id),
            reverse=True,
        )

    total_artifacts = sum(len(stage.artifacts) for stage in stages)
    return TaskArtifactsResponse(
        task_id=task_id,
        stages=stages,
        total_artifacts=total_artifacts,
    )


@router.get("/tasks/{task_id}/preview", response_model=PreviewResponse)
async def preview_output(
    task_id: str,
    limit: int = Query(default=10, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    source: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    预览任��输出数据

    从输出文件读取样本数据供前端展示。
    """
    import os
    import json

    task = _verify_task_access(task_id, current_user)

    def _has_data(path: str) -> bool:
        return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0

    if source and source in _PREVIEW_SOURCE_MAP:
        # Explicit source requested
        output_path = task.get(_PREVIEW_SOURCE_MAP[source]) or ""
    else:
        # Auto-fallback: output → qa_filtered → qa_full
        output_path = task.get("output_path") or ""
        if not _has_data(output_path):
            for fallback_key in ("qa_filtered_path", "qa_output_path"):
                fallback = task.get(fallback_key)
                if _has_data(fallback):
                    output_path = fallback
                    break

    if not output_path or not os.path.exists(output_path):
        return PreviewResponse(total=0, samples=[])

    samples = []
    total = 0

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            # 先统计总行数并读取指定范围
            for i, line in enumerate(f):
                total += 1
                if offset <= i < offset + limit:
                    try:
                        samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to read output file {output_path}: {e}")

    return PreviewResponse(total=total, samples=samples)


@router.post("/tasks/{task_id}/restart")
async def restart_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    force: bool = False,
):
    """
    重启失败、已停止或已完成的任务

    - force=false（默认）：断点续传，复用已有的中间产物（QA 抽取等）
    - force=true：从头重新运行，忽略所有检查点
    """
    task = _verify_task_access(task_id, current_user)
    raw_task = generation_task_service.get_task_raw(task_id)
    if raw_task is None:
        raise HTTPException(status_code=404, detail="Generation task not found")
    previous_run_token = raw_task.get("run_token")

    if task["status"] not in {GenerationStatus.FAILED, GenerationStatus.STOPPED, GenerationStatus.COMPLETED}:
        raise HTTPException(
            status_code=400,
            detail=f"只能重启失败、已停止或已完成的任务，当前状态: {task['status']}"
        )
    _reject_pending_sync_tracking(task_id, reject_failed=True)

    try:
        verified_input_path = _resolve_generation_task_input(task)
    except UnsupportedEvaluationDatasetStorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResourceProvenanceError as exc:
        raise HTTPException(
            status_code=403,
            detail="Dataset does not have verifiable API-managed provenance",
        ) from exc
    task = dict(task)
    task["input_path"] = verified_input_path

    try:
        _require_task_collection_ownership(task)
    except ResourceProvenanceError as exc:
        raise HTTPException(
            status_code=403,
            detail="Milvus collection does not have verifiable tenant ownership",
        ) from exc

    # 确保旧 pipeline 已停止（避免新旧 pipeline 交替更新进度导致跳动）
    current = generation_task_service.get_task_raw(task_id)
    if (
        not current
        or current.get("status")
        not in {
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
            GenerationStatus.COMPLETED,
        }
        or current.get("run_token") != previous_run_token
    ):
        raise HTTPException(
            status_code=409,
            detail="Generation task lifecycle changed before restart",
        )
    if previous_run_token and generation_publication_service.has_staging_products(
        task_id,
        expected_run_token=previous_run_token,
    ):
        cleaned = generation_publication_service.compensate_attempt(
            task_id=task_id,
            expected_run_token=previous_run_token,
            user_id=task.get("user_id"),
        )
        if (
            not cleaned
            or generation_publication_service.has_staging_products(
                task_id,
                expected_run_token=previous_run_token,
            )
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Generation publication staging still requires repair "
                    "before restart"
                ),
            )
    old_pipeline, old_pipeline_token = _get_running_pipeline(task_id)
    if old_pipeline_token == previous_run_token:
        _remove_running_pipeline(
            task_id,
            pipeline=old_pipeline,
            run_token=old_pipeline_token,
        )
        if old_pipeline:
            old_pipeline.stop()
    with _generation_state_lock:
        previous_cancellation = _generation_cancellation_requests.get(task_id)

    # ── 断点续传：优先从统一阶段视图读取 checkpoint ──
    resume_qa_path = None
    resume_qa_filtered_path = None
    skip_filter = False
    previous_filter_stats = None

    if force:
        logger.info("[restart] force=True, skipping all checkpoints for task %s", task_id)
    else:
        stage_map = {s.get("stage"): s for s in _build_task_stages(task)}

        def _stage_checkpoint(stage_name: str) -> Optional[str]:
            stage = stage_map.get(stage_name) or {}
            path = stage.get("checkpoint_path")
            if not path:
                return None
            if not os.path.exists(path):
                return None
            if stage.get("resume_ready") is False:
                return None
            return path

        mode = task.get("generation_mode")
        if mode == "doc_to_training":
            # Prefer later phase checkpoint when available.
            resume_qa_filtered_path = _stage_checkpoint("qa_filter")
            if resume_qa_filtered_path:
                logger.info(
                    "[restart] Use qa_filter checkpoint: %s (skip QA extraction + filtering)",
                    resume_qa_filtered_path,
                )
            else:
                resume_qa_path = _stage_checkpoint("qa_extraction")
                if resume_qa_path:
                    logger.info(
                        "[restart] Use qa_extraction checkpoint: %s (skip QA extraction)",
                        resume_qa_path,
                    )
        elif mode == "qa_to_training":
            resume_qa_filtered_path = _stage_checkpoint("qa_filter")
            if resume_qa_filtered_path:
                logger.info(
                    "[restart] Use qa_filter checkpoint for qa_to_training: %s",
                    resume_qa_filtered_path,
                )
        elif mode == "doc_to_eval":
            resume_qa_path = _stage_checkpoint("qa_extraction")
            if resume_qa_path:
                logger.info(
                    "[restart] Use qa_extraction checkpoint for doc_to_eval: %s",
                    resume_qa_path,
                )

        # Backward-compatible fallback for older tasks that do not expose stage checkpoints.
        if not resume_qa_filtered_path and mode in ("doc_to_training", "qa_to_training"):
            legacy_filtered = task.get("qa_filtered_path")
            if legacy_filtered and os.path.exists(legacy_filtered):
                resume_qa_filtered_path = legacy_filtered
        if not resume_qa_path and mode in ("doc_to_training", "doc_to_eval"):
            legacy_qa = task.get("qa_output_path")
            if legacy_qa and os.path.exists(legacy_qa):
                resume_qa_path = legacy_qa

    # 去重检查（训练模式重启时，force 模式跳过）
    if not force and task["generation_mode"] in ("qa_to_training", "doc_to_training") and task.get("embedding_config_id") and task.get("source_dataset_id"):
        existing_task = generation_task_service.find_completed_task(
            source_dataset_id=task.get("source_dataset_id"),
            embedding_config_id=task.get("embedding_config_id"),
            user_id=task.get("user_id"),
        )
        if existing_task:
            previous_filter_stats = existing_task.get("filter_stats")
            prev_threshold = None
            prev_dedup_enabled = None
            if previous_filter_stats:
                prev_threshold = previous_filter_stats.get("threshold")
                prev_dedup_enabled = previous_filter_stats.get("dedup_enabled")

            current_dedup_enabled = bool((task.get("post_process_config") or {}).get("dedup", {}).get("enabled"))
            threshold_match = prev_threshold is not None and abs(prev_threshold - task.get("similarity_threshold", 0.85)) < 1e-6
            dedup_match = prev_dedup_enabled == current_dedup_enabled if prev_dedup_enabled is not None else True

            if threshold_match and dedup_match:
                skip_filter = True

    # 重新构建配置
    resume_output_candidate = None if force else task.get("output_path")
    resume_output_path = (
        resume_output_candidate
        if resume_output_candidate and os.path.isfile(resume_output_candidate)
        else None
    )
    planned_run_token = str(uuid4())
    base_output_path = os.path.join(
        GENERATION_OUTPUT_DIR,
        f"generated_{task_id}.jsonl",
    )
    output_path = str(
        resolve_generation_attempt_output_path(
            base_output_path,
            task_id,
            planned_run_token,
        )
    )

    # 兼容旧任务：优先从 embedding_config JSON 读取，回退到顶层列
    _emb_cfg = task.get("embedding_config") or {}
    _compat_embedding_config = _emb_cfg.copy() if _emb_cfg else {}
    if _compat_embedding_config:
        _compat_embedding_config.setdefault("similarity_threshold", task.get("similarity_threshold", 0.85))
        _compat_embedding_config.setdefault("retrieval_top_k", task.get("retrieval_top_k", 10))

    llm_config = task.get("llm_config") or {}
    llm_concurrency = llm_config.get("concurrency")
    if llm_concurrency is None:
        llm_concurrency = (task.get("worker_config") or {}).get("concurrency", 10)

    config = PipelineConfig(
        input_path=task["input_path"],
        input_format=task["input_format"],
        content_field=task.get("content_field"),
        generation_mode=task["generation_mode"],
        pos_neg_method=task.get("pos_neg_method", "retrieval"),
        output_path=output_path,
        output_format=task["output_format"],
        llm_config=llm_config,
        eval_llm_config=task.get("eval_llm_config"),
        llm_concurrency=llm_concurrency,
        embedding_concurrency=task.get("embedding_config", {}).get("concurrency", 20) if task.get("embedding_config") else 20,
        timeout_per_doc=task["worker_config"].get("timeout_per_doc", 300),
        steps=task["steps_config"],
        post_process=task.get("post_process_config"),
        custom_prompts=task.get("custom_prompts"),
        embedding_config=_compat_embedding_config or None,
        rerank_config=task.get("rerank_config"),
        source_dataset_id=task.get("source_dataset_id"),
        skip_filter=skip_filter,
        previous_filter_stats=previous_filter_stats,
        resume_qa_path=resume_qa_path,
        resume_qa_filtered_path=resume_qa_filtered_path,
        resume_output_path=resume_output_path,
        task_id=task_id,
        existing_collection_name=task.get("milvus_collection"),
        allowed_input_dirs=get_generation_input_allowed_dirs(),
        user_id=task.get("user_id"),
        run_token=planned_run_token,
    )
    try:
        validate_generation_resource_config(config)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Stored generation configuration exceeds current resource limits",
        ) from exc

    try:
        claimed, execution_lease = background_task_admission_service.admit_execution(
            "generation",
            task_id,
            current_user.get("user_id"),
            generation_task_service.begin_restart,
            task_id,
            require_source_dataset=bool(task.get("source_dataset_id")),
            expected_run_token=previous_run_token,
            new_run_token=planned_run_token,
        )
    except BackgroundTaskCapacityExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except BackgroundTaskAlreadyExecuting as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not claimed:
        raise HTTPException(
            status_code=409,
            detail="Generation task was already restarted by another request",
        )
    scheduled_run_token = str(claimed)
    if scheduled_run_token != planned_run_token or execution_lease is None:
        if execution_lease is not None:
            execution_lease.release()
        raise HTTPException(
            status_code=409,
            detail="Generation task lifecycle changed during restart reservation",
        )

    attempt_dir = _generation_attempt_directory(task_id, scheduled_run_token)
    try:
        if resume_output_path:
            config.resume_output_path = _copy_restart_checkpoint(
                resume_output_path,
                Path(config.output_path),
                expected_directory=attempt_dir,
            )
        else:
            config.resume_output_path = None
        if resume_qa_path:
            config.resume_qa_path = _copy_restart_checkpoint(
                resume_qa_path,
                attempt_dir / f"qa_extracted_{task_id}.jsonl",
                expected_directory=attempt_dir,
            )
        else:
            config.resume_qa_path = None
        if resume_qa_filtered_path:
            config.resume_qa_filtered_path = _copy_restart_checkpoint(
                resume_qa_filtered_path,
                attempt_dir / f"qa_filtered_{task_id}.jsonl",
                expected_directory=attempt_dir,
            )
        else:
            config.resume_qa_filtered_path = None
    except Exception as exc:
        try:
            _cleanup_generation_attempt_directory(task_id, scheduled_run_token)
        except Exception:
            execution_lease.release()
            logger.exception(
                "Restart checkpoint cleanup remains pending for generation %s",
                task_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Generation restart cleanup requires startup recovery",
            ) from exc
        generation_task_service.fail_restart(
            task_id,
            expected_run_token=scheduled_run_token,
            error_message="Generation restart checkpoint copy failed",
        )
        execution_lease.release()
        raise HTTPException(
            status_code=409,
            detail="Generation restart checkpoint is unavailable",
        ) from exc

    if not generation_task_service.finish_restart(
        task_id,
        expected_run_token=scheduled_run_token,
        output_path=config.output_path,
    ):
        try:
            _cleanup_generation_attempt_directory(task_id, scheduled_run_token)
        except Exception as exc:
            execution_lease.release()
            raise HTTPException(
                status_code=500,
                detail="Generation restart cleanup requires startup recovery",
            ) from exc
        generation_task_service.fail_restart(
            task_id,
            expected_run_token=scheduled_run_token,
            error_message="Generation restart lost ownership before scheduling",
        )
        execution_lease.release()
        raise HTTPException(
            status_code=409,
            detail="Generation task lost attempt ownership before scheduling",
        )
    if previous_cancellation is not None:
        _clear_generation_cancellation(task_id, previous_cancellation)
    try:

        # 启动后台任务
        background_tasks.add_task(
            background_task_admission_service.run_async,
            execution_lease,
            _run_generation_task,
            task_id=task_id,
            config=config,
            expected_run_token=scheduled_run_token,
        )
    except Exception:
        execution_lease.release()
        try:
            failed = generation_task_service.update_status(
                task_id,
                GenerationStatus.FAILED,
                "Task could not be scheduled for background execution",
                expected_run_token=scheduled_run_token,
            )
            if not failed:
                logger.info(
                    "Restarted generation task %s changed state before scheduling "
                    "failure could be persisted",
                    task_id,
                )
        except Exception:
            logger.exception(
                "Failed to finalize unscheduled generation task %s",
                task_id,
            )
        raise

    return {"status": "restarted", "task_id": task_id}


@router.get("/formats")
async def list_formats(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取支持的输出格式"""
    return {
        "output_formats": [
            {"name": "universal", "description": "通用格式，包含 query, positives, negatives"},
            {"name": "triplet", "description": "三元组格式，每行一个 (query, positive, negative)"},
            {"name": "pair", "description": "对格式，每行一个 (query, text, label)"},
        ],
        "source_types": list(SOURCE_TYPES.keys()),
        "length_types": list(LENGTH_TYPES.keys()),
    }


# ===== Background Task =====

# 输出格式到数据集类型的映射
OUTPUT_FORMAT_TO_DATASET_TYPE = {
    "universal": "embedding_universal",
    "triplet": "embedding_triplet",
    "pair": "embedding_pair",
}


def _get_file_stats(file_path: str) -> Dict[str, Any]:
    """获取输出文件的统计信息"""
    import os
    stats = {"file_size": 0, "num_rows": 0}
    try:
        if os.path.exists(file_path):
            stats["file_size"] = os.path.getsize(file_path)
            # 统计 JSONL 行数
            with open(file_path, "r", encoding="utf-8") as f:
                stats["num_rows"] = sum(1 for _ in f)
    except Exception as e:
        logger.warning(f"Failed to get file stats for {file_path}: {e}")
    return stats




def _stage_generation_dataset(
    task: Dict[str, Any],
    storage_path: str,
    *,
    run_token: str,
    name_suffix: str,
    dataset_type: str,
    usage: str,
    relation_type: str,
    num_rows: int,
    file_format: str = "jsonl",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Stage one exact-attempt dataset without making it consumable."""
    task_id = task.get("task_id")
    if not task_id or not run_token:
        return None
    file_stats = _get_file_stats(storage_path)
    dataset_id = str(
        uuid5(
            NAMESPACE_URL,
            "\x1f".join(
                (
                    "train-factory-generation",
                    task_id,
                    run_token,
                    relation_type,
                    storage_path,
                )
            ),
        )
    )
    attempt_suffix = generation_attempt_key(task_id, run_token)[:10]
    task_name = str(task.get("task_name") or task_id[:8])
    dataset_name = f"{task_name[:220]}_{name_suffix}_{attempt_suffix}"
    metadata = dict(extra_metadata or {})
    metadata.update(
        {
            "source_task_id": task_id,
            "generation_mode": task.get("generation_mode"),
        }
    )
    # The immutable token lives only in the hidden owner column.  Do not expose
    # it through public extra_metadata after activation.
    metadata.pop("generation_run_token", None)
    return generation_publication_service.stage_dataset(
        task_id=task_id,
        expected_run_token=run_token,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        storage_path=storage_path,
        dataset_type=dataset_type,
        usage=usage,
        model_type=["embedding", "rerank"],
        source_dataset_id=task.get("source_dataset_id"),
        relation_type=relation_type,
        file_format=file_format,
        num_rows=file_stats.get("num_rows") or num_rows,
        file_size=file_stats.get("file_size"),
        user_id=task.get("user_id"),
        description=f"Generation product for task {task_id}",
        extra_metadata=metadata,
    )


def _auto_register_dataset(
    task: Dict[str, Any],
    output_path: str,
    output_samples: int,
    *,
    run_token: str,
) -> Optional[str]:
    generation_mode = task.get("generation_mode", "doc_to_training")
    dataset_type = (
        "qa_pair"
        if generation_mode == "qa_extraction"
        else OUTPUT_FORMAT_TO_DATASET_TYPE.get(
            task.get("output_format", "universal"),
            "custom",
        )
    )
    return _stage_generation_dataset(
        task,
        output_path,
        run_token=run_token,
        name_suffix="output",
        dataset_type=dataset_type,
        usage="raw" if generation_mode == "qa_extraction" else "train",
        relation_type="training_generated",
        num_rows=output_samples,
        extra_metadata={
            "output_format": task.get("output_format", "universal"),
            "milvus_collection": task.get("milvus_collection"),
        },
    )


def _register_qa_dataset(
    task: Dict[str, Any],
    qa_output_path: str,
    qa_total: int,
    *,
    run_token: str,
) -> Optional[str]:
    return _stage_generation_dataset(
        task,
        qa_output_path,
        run_token=run_token,
        name_suffix="qa",
        dataset_type="qa_pair",
        usage="raw",
        relation_type="qa_extracted",
        num_rows=qa_total,
        extra_metadata={"stage": "qa_extraction"},
    )


def _register_qa_filtered_dataset(
    task: Dict[str, Any],
    qa_filtered_path: str,
    qa_filtered_count: int,
    filter_stats: Dict[str, Any],
    *,
    run_token: str,
) -> Optional[str]:
    return _stage_generation_dataset(
        task,
        qa_filtered_path,
        run_token=run_token,
        name_suffix="qa_filtered",
        dataset_type="qa_pair",
        usage="raw",
        relation_type="qa_filtered",
        num_rows=qa_filtered_count,
        extra_metadata={
            "stage": "qa_filtered",
            "filter_stats": filter_stats,
        },
    )


def _register_deep_eval_dataset(
    task: Dict[str, Any],
    deep_eval_path: str,
    deep_eval_count: int,
    *,
    run_token: str,
) -> Optional[str]:
    return _stage_generation_dataset(
        task,
        deep_eval_path,
        run_token=run_token,
        name_suffix="deep_eval",
        dataset_type="custom",
        usage="eval",
        relation_type="deep_eval_generated",
        num_rows=deep_eval_count,
        extra_metadata={
            "stage": "deep_eval",
            "fields": [
                "query",
                "expected_output",
                "retrieval_context",
                "source_chunk_id",
                "source_chunk_content",
            ],
        },
    )


def _publication_artifacts_belong_to_attempt(
    task: Dict[str, Any],
    run_token: str,
    result: Any,
) -> bool:
    """Fail closed unless every product stays beneath the current attempt."""
    task_id = task.get("task_id")
    current_output_path = task.get("output_path")
    if not task_id or not current_output_path:
        return False
    expected_directory = Path(current_output_path).resolve().parent
    if (
        expected_directory.name != generation_attempt_key(task_id, run_token)
        or expected_directory.parent.name != f"generation_{task_id}"
    ):
        return False

    details = result.details or {}
    artifact_paths = (
        result.output_path,
        details.get("qa_output_path"),
        details.get("qa_filtered_path"),
        details.get("deep_eval_path"),
    )
    for artifact_path in artifact_paths:
        if not artifact_path:
            continue
        try:
            resolved = Path(artifact_path).resolve()
            relative = resolved.relative_to(expected_directory)
        except (OSError, ValueError):
            return False
        if not relative.parts:
            return False
    return True


def _publish_generation_result(
    task_id: str,
    run_token: str,
    result: Any,
) -> bool:
    """Stage every product, then atomically expose the complete attempt."""
    task = generation_task_service.get_task(task_id)
    if task is None:
        return False
    details = result.details or {}
    if not _publication_artifacts_belong_to_attempt(task, run_token, result):
        logger.warning(
            "Generation task %s produced an artifact outside its current attempt",
            task_id,
        )
        return False
    auto_register = bool(task.get("auto_register_dataset"))
    dataset_bindings: Dict[str, Optional[str]] = {}

    qa_output_path = details.get("qa_output_path")
    qa_dataset_id = None
    if qa_output_path and auto_register:
        qa_dataset_id = _register_qa_dataset(
            task,
            qa_output_path,
            details.get("qa_total", 0),
            run_token=run_token,
        )
        if qa_dataset_id is None:
            return False
        dataset_bindings["qa_dataset_id"] = qa_dataset_id

    qa_filtered_path = details.get("qa_filtered_path")
    qa_filtered_dataset_id = None
    if qa_filtered_path and auto_register:
        qa_filtered_dataset_id = _register_qa_filtered_dataset(
            task,
            qa_filtered_path,
            details.get("qa_filtered_count", 0),
            details.get("filter_stats") or {},
            run_token=run_token,
        )
        if qa_filtered_dataset_id is None:
            return False
        dataset_bindings["qa_filtered_dataset_id"] = (
            qa_filtered_dataset_id
        )

    deep_eval_path = details.get("deep_eval_path")
    deep_eval_dataset_id = None
    if deep_eval_path and auto_register:
        deep_eval_dataset_id = _register_deep_eval_dataset(
            task,
            deep_eval_path,
            details.get("deep_eval_count", 0),
            run_token=run_token,
        )
        if deep_eval_dataset_id is None:
            return False
        dataset_bindings["deep_eval_dataset_id"] = deep_eval_dataset_id

    generation_mode = task.get("generation_mode", "")
    is_eval_mode = generation_mode in {"doc_to_eval", "qa_to_eval"}
    output_dataset_id = None
    if result.output_path and auto_register and not is_eval_mode:
        output_dataset_id = _auto_register_dataset(
            task,
            result.output_path,
            result.output_samples,
            run_token=run_token,
        )
        if output_dataset_id is None:
            return False
        dataset_bindings["output_dataset_id"] = output_dataset_id

    filter_stats = details.get("filter_stats") or {}
    milvus_collection = (
        filter_stats.get("milvus_collection")
        or task.get("milvus_collection")
    )
    milvus_registration = None
    source_dataset_id = task.get("source_dataset_id")
    if milvus_collection and source_dataset_id:
        source_dataset = dataset_service.get_dataset(source_dataset_id)
        embedding = task.get("embedding_config") or {}
        milvus_registration = {
            "collection_name": milvus_collection,
            "dataset_id": source_dataset_id,
            "dataset_name": (
                source_dataset.get("dataset_name") if source_dataset else None
            ),
            "user_id": task.get("user_id"),
            "embedding_config_id": task.get("embedding_config_id"),
            "embedding_model": embedding.get("model"),
            "embedding_endpoint": embedding.get("endpoint"),
            "dim": embedding.get("dim"),
            "metric_type": embedding.get("metric_type"),
            "hybrid_enabled": embedding.get("hybrid_enabled", False),
        }

    final_processed = max(
        int(result.processed_docs or 0),
        int(result.total_docs or 0),
    )
    artifact_updates: Dict[str, Any] = {
        "output_sample_count": int(result.output_samples or 0),
        "processed_docs": final_processed,
        "total_docs": int(result.total_docs or 0),
        "progress": 100.0 if result.total_docs else task.get("progress", 0.0),
    }
    if result.output_path:
        artifact_updates["output_path"] = result.output_path
    if qa_output_path:
        artifact_updates["qa_output_path"] = qa_output_path
    if qa_filtered_path:
        artifact_updates["qa_filtered_path"] = qa_filtered_path
    if deep_eval_path:
        artifact_updates["deep_eval_path"] = deep_eval_path
    if filter_stats:
        artifact_updates["filter_stats"] = filter_stats
    if milvus_collection:
        artifact_updates["milvus_collection"] = milvus_collection

    completed = generation_task_service.complete_publication(
        task_id,
        expected_run_token=run_token,
        dataset_bindings=dataset_bindings,
        milvus_registration=milvus_registration,
        artifact_updates=artifact_updates,
    )
    if not completed:
        return False
    tracking_finalized = _finalize_sync_generation_tracking(
        task_id,
        generation_mode=generation_mode,
        output_dataset_id=output_dataset_id,
        output_sample_count=result.output_samples or 0,
        qa_dataset_id=qa_dataset_id,
        deep_eval_dataset_id=deep_eval_dataset_id,
        user_id=task.get("user_id"),
    )
    if not tracking_finalized:
        logger.warning(
            "[sync] Successful generation %s is awaiting runtime tracking "
            "reconciliation",
            task_id,
        )
    return True


async def _run_generation_task(
    task_id: str,
    config: PipelineConfig,
    *,
    expected_run_token: Any = _RUN_TOKEN_UNSET,
    sync_handoff: Optional[Dict[str, Any]] = None,
):
    """运行生成任务"""
    _max_progress = 0.0
    pipeline: Optional[DatasetGenerationPipeline] = None
    run_token: Optional[str] = None

    def progress_callback(processed: int, total: int):
        nonlocal _max_progress
        if (
            pipeline is None
            or run_token is None
            or not _is_current_pipeline(task_id, pipeline, run_token)
            or _is_generation_cancellation_requested(
                task_id,
                pipeline,
                run_token=run_token,
            )
        ):
            return
        # 单调递增保护：进度百分比只增不减，避免阶段切换时前端进度条回退
        if total > 0:
            pct = (processed / total) * 100
            if pct < _max_progress:
                return  # 忽略回退的进度更新
            _max_progress = pct
        generation_task_service.update_progress(
            task_id,
            processed,
            total,
            expected_status=GenerationStatus.RUNNING,
            expected_run_token=run_token,
        )

    def _retry_failed_sync_tracking(reason: str) -> None:
        _finalize_sync_generation_failure_tracking(task_id, reason)

    def _finalize_unlaunched_sync_handoff(reason: str) -> bool:
        nonlocal sync_handoff
        if sync_handoff is None:
            return False
        from ...sync.sync_manager import SyncManager

        handoff = sync_handoff
        sync_handoff = None
        SyncManager._finalize_unlaunched_generation(handoff, reason)
        return True

    try:
        task = generation_task_service.get_task_raw(task_id)
        if expected_run_token is _RUN_TOKEN_UNSET:
            expected_run_token = task.get("run_token") if task else None
        elif task and task.get("run_token") != expected_run_token:
            logger.info(
                "Generation worker %s belongs to an obsolete attempt",
                task_id,
            )
            _finalize_unlaunched_sync_handoff(
                "Sync generation handoff belongs to an obsolete attempt"
            )
            return
        if not task:
            raise RuntimeError(f"Generation task not found: {task_id}")
        if task and task.get("status") == GenerationStatus.STOPPED:
            logger.info(f"Task {task_id} already stopped before start")
            if not _finalize_unlaunched_sync_handoff(
                "Generation stopped before worker startup"
            ):
                _stop_generation_execution(
                    task_id,
                    "Generation stopped before worker startup",
                    run_token=expected_run_token,
                )
            _clear_generation_cancellation(
                task_id,
                run_token=expected_run_token,
            )
            return

        validate_generation_resource_config(config)
        config.existing_collection_name = _require_task_collection_ownership(
            task,
            config.existing_collection_name,
        )
        resolved_model_configs = _revalidate_task_model_config_ownership(task)
        _refresh_owned_pipeline_model_configs(config, resolved_model_configs)
        config.user_id = task.get("user_id")
        config.allowed_input_dirs = get_generation_input_allowed_dirs()
        config.input_path = _resolve_generation_input_path(
            _resolve_generation_task_input(task),
            allowed_dirs=config.allowed_input_dirs,
        )

        if _is_generation_stopped(task_id, run_token=expected_run_token):
            if not _finalize_unlaunched_sync_handoff(
                "Generation stopped during worker validation"
            ):
                _stop_generation_execution(
                    task_id,
                    "Generation stopped during worker validation",
                    run_token=expected_run_token,
                )
            _clear_generation_cancellation(
                task_id,
                run_token=expected_run_token,
            )
            return

        run_token = generation_task_service.claim_running(
            task_id,
            expected_run_token=expected_run_token,
        )
        if not run_token:
            current = generation_task_service.get_task_raw(task_id)
            if _is_generation_cancellation_requested(
                task_id,
                run_token=expected_run_token,
            ) or (
                current
                and current.get("status") == GenerationStatus.STOPPED
                and current.get("run_token") == expected_run_token
            ):
                if not _finalize_unlaunched_sync_handoff(
                    "Generation stopped before worker claim"
                ):
                    _stop_generation_execution(
                        task_id,
                        "Generation stopped before worker claim",
                        run_token=expected_run_token,
                    )
                _clear_generation_cancellation(
                    task_id,
                    run_token=expected_run_token,
                )
            else:
                logger.warning(
                    "Generation worker %s did not acquire the pending task; "
                    "current status is %s",
                    task_id,
                    current.get("status") if current else "missing",
                )
            return

        sync_handoff = None

        if _is_generation_stopped(task_id, run_token=run_token):
            _stop_generation_execution(
                task_id,
                "Generation stopped after worker claim",
                run_token=run_token,
            )
            _clear_generation_cancellation(task_id, run_token=run_token)
            return

        def _owns_running_pipeline() -> bool:
            if (
                pipeline is None
                or run_token is None
                or not _is_current_pipeline(task_id, pipeline, run_token)
            ):
                return False
            current = generation_task_service.get_task_raw(task_id)
            return bool(
                current
                and current.get("status") == GenerationStatus.RUNNING
                and current.get("run_token") == run_token
                and not _is_generation_cancellation_requested(
                    task_id,
                    pipeline,
                    run_token=run_token,
                )
            )

        async def _on_qa_complete(qa_path: str, _qa_count: int):
            """Phase 1 完成后立即注册 QA 数据集（consumer 仍在处理后续阶段）"""
            if not _owns_running_pipeline():
                logger.info(
                    "Skipping QA checkpoint for inactive generation attempt %s",
                    task_id,
                )
                return
            generation_task_service.set_qa_output(
                task_id,
                qa_output_path=qa_path,
                expected_status=GenerationStatus.RUNNING,
                expected_run_token=run_token,
            )

        async def _persist_qa_phase(
            qa_path: str,
            qa_filtered_path: Optional[str],
        ) -> bool:
            if not _owns_running_pipeline():
                return False
            return generation_task_service.set_qa_output(
                task_id,
                qa_output_path=qa_path or "",
                qa_filtered_path=qa_filtered_path,
                expected_status=GenerationStatus.RUNNING,
                expected_run_token=run_token,
            )

        config.run_token = run_token
        if config.output_path:
            config.output_path = str(
                resolve_generation_attempt_output_path(
                    config.output_path,
                    task_id,
                    run_token,
                )
            )
        config.attempt_is_valid = _owns_running_pipeline
        pipeline = DatasetGenerationPipeline(
            config=config,
            progress_callback=progress_callback,
            qa_phase_callback=_persist_qa_phase,
            on_qa_complete=_on_qa_complete,
        )

        # 保存 pipeline 引用以便停止
        current = generation_task_service.get_task_raw(task_id)
        owns_claim = bool(
            run_token
            and current
            and current.get("status") == GenerationStatus.RUNNING
            and current.get("run_token") == run_token
            and not _is_generation_cancellation_requested(
                task_id,
                run_token=run_token,
            )
        )
        if owns_claim:
            _install_running_pipeline(task_id, pipeline, run_token)

        if not owns_claim:
            pipeline.stop()
            _stop_generation_execution(
                task_id,
                "Generation stopped before pipeline execution",
                pipeline,
                run_token=run_token,
            )
            _clear_generation_cancellation(task_id, run_token=run_token)
            return

        if _is_generation_stopped(
            task_id,
            run_token=run_token,
            owner=pipeline,
        ):
            _stop_generation_execution(
                task_id,
                "Generation stopped before pipeline execution",
                pipeline,
                run_token=run_token,
            )
            return

        result = await pipeline.run()

        if not _is_current_pipeline(task_id, pipeline, run_token):
            logger.info(
                "Generation task %s moved to a newer pipeline attempt",
                task_id,
            )
            return

        if _is_generation_stopped(
            task_id,
            run_token=run_token,
            owner=pipeline,
        ):
            _stop_generation_execution(
                task_id,
                "Generation stopped during pipeline execution",
                pipeline,
                run_token=run_token,
            )
            return

        if result.success:
            current = generation_task_service.get_task_raw(task_id)
            if not (
                _is_current_pipeline(task_id, pipeline, run_token)
                and current
                and current.get("status") == GenerationStatus.RUNNING
                and current.get("run_token") == run_token
                and not _is_generation_cancellation_requested(
                    task_id,
                    pipeline,
                    run_token=run_token,
                )
            ):
                logger.info(
                    "Generation task %s lost ownership before publication",
                    task_id,
                )
                return
            if not generation_task_service.update_status(
                task_id,
                GenerationStatus.PUBLISHING,
                expected_run_token=run_token,
            ):
                current = generation_task_service.get_task_raw(task_id)
                if (
                    current
                    and current.get("status") == GenerationStatus.STOPPED
                    and current.get("run_token") == run_token
                ):
                    _finalize_stopped_generation(
                        task_id,
                        "Generation stopped before publication",
                    )
                logger.info(
                    "Generation task %s lost the publication claim",
                    task_id,
                )
                return
            # 对齐最终进度：成功完成时以 pipeline 结果回填一次终态进度，
            # 避免阶段切换/过滤口径导致 completed 但进度<100。
            if result.total_docs > 0:
                final_processed = max(result.processed_docs, result.total_docs)
                if not generation_task_service.update_progress(
                    task_id,
                    final_processed,
                    result.total_docs,
                    result.output_samples,
                    expected_status=GenerationStatus.PUBLISHING,
                    expected_run_token=run_token,
                ):
                    raise RuntimeError("Generation publication lost progress ownership")

            # 保存 filter_stats
            filter_stats = result.details.get("filter_stats") if result.details else None
            if filter_stats:
                if not generation_task_service.set_filter_results(
                    task_id,
                    milvus_collection=filter_stats.get("milvus_collection"),
                    filter_stats=filter_stats,
                    expected_status=GenerationStatus.PUBLISHING,
                    expected_run_token=run_token,
                ):
                    raise RuntimeError("Generation publication lost filter ownership")

            if not _publish_generation_result(task_id, run_token, result):
                raise RuntimeError(
                    "Generation publication transaction did not complete"
                )
            return

        else:
            # 检查是否是用户主动停止
            task = generation_task_service.get_task_raw(task_id)
            is_stopped = bool(
                task
                and task.get("status") == GenerationStatus.STOPPED
                and task.get("run_token") == run_token
            )
            failed = False
            if not is_stopped:
                failed = generation_task_service.update_status(
                    task_id,
                    GenerationStatus.FAILED,
                    result.error,
                    expected_run_token=run_token,
                )

            # FAILED: rollback batches + restore pending counters.
            # STOPPED: only restore sync task status to IDLE (no auto-retry).
            if failed:
                _retry_failed_sync_tracking(
                    result.error or "Generation produced no output"
                )
            else:
                current = generation_task_service.get_task_raw(task_id)
                is_stopped = bool(
                    current
                    and current.get("status") == GenerationStatus.STOPPED
                    and current.get("run_token") == run_token
                )
            if is_stopped:
                _finalize_stopped_generation(
                    task_id,
                    result.error or "Generation stopped by user",
                )

    except Exception as e:
        attempt_token = (
            run_token if run_token is not None else expected_run_token
        )
        if _is_generation_stopped(task_id, run_token=attempt_token):
            logger.info("Generation task %s stopped while worker exited", task_id)
            _stop_generation_execution(
                task_id,
                "Generation stopped while the worker was unwinding",
                pipeline,
                run_token=attempt_token,
            )
            return

        logger.exception(f"Task {task_id} failed")
        current_attempt = generation_task_service.get_task_raw(task_id)
        if (
            run_token
            and current_attempt
            and current_attempt.get("status") == GenerationStatus.PUBLISHING
            and current_attempt.get("run_token") == run_token
        ):
            compensated = generation_publication_service.compensate_attempt(
                task_id=task_id,
                expected_run_token=run_token,
                user_id=current_attempt.get("user_id"),
            )
            if not compensated:
                logger.error(
                    "Generation publication cleanup for %s requires repair; "
                    "leaving the exact attempt PUBLISHING",
                    task_id,
                )
                return
        status_set = False
        try:
            status_set = generation_task_service.update_status(
                task_id,
                GenerationStatus.FAILED,
                str(e),
                expected_run_token=attempt_token,
            )
        except Exception as status_err:
            # Task may already be in a terminal state (COMPLETED/STOPPED); an
            # invalid transition here must not abort processing. If the task did
            # not actually transition to FAILED, skip the failure callback below
            # so a manually STOPPED/COMPLETED task is not treated as failed.
            logger.warning(f"Task {task_id}: could not set FAILED status: {status_err}")

        # Notify sync system so it can restore IDLE status — only when the task
        # genuinely transitioned to FAILED. A STOPPED/COMPLETED task is handled by
        # its own branch and must not trigger the failure/auto-retry path here.
        if status_set:
            _retry_failed_sync_tracking(str(e))
        else:
            current = generation_task_service.get_task_raw(task_id)
            if (
                current
                and current.get("status") == GenerationStatus.STOPPED
                and current.get("run_token") == attempt_token
            ):
                _finalize_stopped_generation(
                    task_id,
                    "Generation stopped while worker failure was being persisted",
                )
    finally:
        if pipeline is not None and run_token is not None:
            removed = _remove_running_pipeline(
                task_id,
                pipeline=pipeline,
                run_token=run_token,
            )
            if removed is pipeline:
                _clear_generation_cancellation(
                    task_id,
                    pipeline,
                    run_token=run_token,
                )
