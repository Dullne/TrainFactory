"""
Sync worker: executes one sync cycle for a given task.

Phase 1: Incremental fetch from external API → save batch JSONL
Phase 2: Check Level 1 threshold → trigger generation task
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..core.time_utils import sync_normalize_naive, sync_now_naive
from ..config.settings import get_settings
from ..enums.sync_status import (
    BatchStatus,
    SyncGenerationStatus,
    SyncStatus,
    SyncTrainingStatus,
)
from ..storage.services.outbound_endpoint_policy import validate_user_outbound_url
from ..storage.entities.external_sync_entity import _is_sensitive_config_key
from .external_client import ExternalApiClient

logger = logging.getLogger(__name__)

# Output directory for sync data
SYNC_DATA_DIR = os.environ.get("SYNC_DATA_DIR", "/app/data/sync")

_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_WINDOWS_RESERVED_PATH_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_SYNC_STORAGE_RESERVATION_LOCK = threading.Lock()
_SYNC_STORAGE_RESERVATIONS: Dict[str, tuple[str, int]] = {}
_REGISTERED_RECOVERY_CURSOR_LOCK = threading.Lock()
_REGISTERED_RECOVERY_OFFSETS: Dict[str, int] = {}


class SyncResourceLimitExceeded(ValueError):
    """Raised when external sync resource limits would be exceeded."""


def _utcnow_naive() -> datetime:
    """Return sync-mode timestamp as naive datetime for DB and naming compatibility."""
    return sync_now_naive()


def _require_path_component(value: Any, label: str) -> str:
    """Validate an identifier before using it as a single path component."""
    if not isinstance(value, str):
        raise ValueError(f"{label} path component must be a string")
    component = value.strip()
    if (
        not component
        or component != value
        or component in {".", ".."}
        or not _SAFE_PATH_COMPONENT.fullmatch(component)
        or component.endswith(".")
        or component.split(".", 1)[0].upper() in _WINDOWS_RESERVED_PATH_NAMES
    ):
        raise ValueError(f"{label} path component is invalid")
    return component


def _sync_storage_user_component(user_id: Any) -> str:
    """Map only the no-auth database owner to a safe, stable directory.

    This is a filesystem identity, never a replacement database owner. Keep
    task/batch ownership comparisons exact, including historical empty owners.
    """
    if user_id == "" and not get_settings().auth_enabled:
        return "anonymous"
    return _require_path_component(user_id, "User ID")


def _managed_directory(
    *components: str,
    create: bool,
) -> Path:
    """Resolve a directory below the trusted sync root without following links."""
    root = Path(os.path.abspath(SYNC_DATA_DIR))
    if create:
        root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve(strict=True)
    if not root_resolved.is_dir():
        raise ValueError("Sync data root must be a directory")

    current = root
    for component in components:
        current = current / component
        if create:
            current.mkdir(exist_ok=True)
        if current.is_symlink():
            raise ValueError("Managed sync directories must not be symbolic links")
        resolved = current.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("Managed sync path must be a directory")
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError("Managed sync directory escapes the sync root") from exc
    return current


def _sync_directory_size(root: Path, excluded_paths: set[str]) -> int:
    """Measure regular files without following links; fail closed on scan races."""
    try:
        if root.is_symlink() or not root.is_dir():
            raise OSError("invalid sync storage directory")
        total = 0
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    normalized = os.path.normcase(os.path.abspath(entry.path))
                    if normalized in excluded_paths:
                        continue
                    total += entry.stat(follow_symlinks=False).st_size
        return total
    except OSError as exc:
        raise SyncResourceLimitExceeded(
            "Unable to verify external sync storage quota"
        ) from exc


def _normalized_path_is_within(path: str, directory: Path) -> bool:
    normalized_directory = os.path.normcase(os.path.abspath(directory))
    try:
        return os.path.commonpath((path, normalized_directory)) == normalized_directory
    except ValueError:
        return False


@contextmanager
def _reserve_sync_storage(user_id: str, target_path: Path, byte_count: int):
    """Reserve sync-root capacity while a file is being created."""
    user_id = _sync_storage_user_component(user_id)
    requested = int(byte_count)
    if requested < 0:
        raise SyncResourceLimitExceeded("External sync storage request is invalid")

    root = _managed_directory(create=True)
    user_root = _managed_directory(user_id, create=True)
    user_roots = [user_root]
    legacy_user_component = user_id[:16]
    if legacy_user_component != user_id:
        legacy_user_path = root / legacy_user_component
        if os.path.lexists(legacy_user_path):
            user_roots.append(
                _managed_directory(legacy_user_component, create=False)
            )
    normalized_target = os.path.normcase(os.path.abspath(target_path))
    settings = get_settings()

    with _SYNC_STORAGE_RESERVATION_LOCK:
        if normalized_target in _SYNC_STORAGE_RESERVATIONS:
            raise SyncResourceLimitExceeded(
                "External sync storage target is already reserved"
            )
        excluded = set(_SYNC_STORAGE_RESERVATIONS)
        global_usage = _sync_directory_size(root, excluded)
        user_usage = sum(
            _sync_directory_size(storage_root, excluded)
            for storage_root in user_roots
        )
        reserved_global = sum(
            size for _reserved_user, size in _SYNC_STORAGE_RESERVATIONS.values()
        )
        reserved_user = sum(
            size
            for reserved_path, (reserved_user_id, size) in (
                _SYNC_STORAGE_RESERVATIONS.items()
            )
            if reserved_user_id == user_id
            or any(
                _normalized_path_is_within(reserved_path, storage_root)
                for storage_root in user_roots
            )
        )
        if user_id == "anonymous" and not settings.auth_enabled:
            # Historical empty-owner batches live directly under the sync root.
            # Reuse its existing accounting conservatively rather than omit
            # those files from the development identity's quota or claim them.
            user_usage = global_usage
            reserved_user = reserved_global
        if (
            global_usage + reserved_global + requested
            > settings.sync_storage_max_bytes_global
        ):
            raise SyncResourceLimitExceeded(
                "Global sync storage quota exceeded"
            )
        if (
            user_usage + reserved_user + requested
            > settings.sync_storage_max_bytes_per_user
        ):
            raise SyncResourceLimitExceeded(
                "Per-user sync storage quota exceeded"
            )
        _SYNC_STORAGE_RESERVATIONS[normalized_target] = (user_id, requested)

    try:
        yield
    finally:
        with _SYNC_STORAGE_RESERVATION_LOCK:
            _SYNC_STORAGE_RESERVATIONS.pop(normalized_target, None)


def _resolve_managed_batch_path(
    task_id: str,
    user_id: str,
    batch: Dict[str, Any],
) -> Path:
    """Resolve one batch file and prove it belongs to this sync task."""
    if batch.get("task_id") != task_id or batch.get("user_id") != user_id:
        raise ValueError("Batch does not belong to the requested sync task")

    task_id = _require_path_component(task_id, "Task ID")
    user_component = _sync_storage_user_component(user_id)

    raw_storage_path = batch.get("storage_path")
    if not isinstance(raw_storage_path, str) or not raw_storage_path.strip():
        raise ValueError("Batch storage path is invalid")

    root = Path(os.path.abspath(SYNC_DATA_DIR))
    raw_path = Path(os.path.abspath(raw_storage_path))
    candidate_components = (
        (user_component, task_id),
        (user_id, task_id),
        # Compatibility for batches written before full tenant scoping.
        (user_id[:16], task_id[:8]),
    )
    managed_components = next(
        (
            components
            for components in candidate_components
            if raw_path.parent == root.joinpath(*components)
        ),
        None,
    )
    if managed_components is None:
        raise ValueError("Batch storage path is outside the managed sync directory")

    managed_parent = _managed_directory(*managed_components, create=False)
    if raw_path.is_symlink():
        raise ValueError("Batch storage path must not be a symbolic link")
    resolved_path = raw_path.resolve(strict=True)
    if not resolved_path.is_file():
        raise ValueError("Batch storage path must be a regular file")
    if not (
        resolved_path.name.startswith("batch_")
        and resolved_path.suffix.lower() == ".jsonl"
    ):
        raise ValueError("Batch storage path is not a managed batch file")

    if resolved_path.parent != managed_parent.resolve(strict=True):
        raise ValueError("Batch storage path is outside the managed sync directory")
    return resolved_path


def _delete_managed_merged_file(
    task_id: str,
    user_id: str,
    merged_path: str,
) -> bool:
    """Delete one generation input after proving its full sync-task scope."""
    task_id = _require_path_component(task_id, "Task ID")
    user_id = _sync_storage_user_component(user_id)
    if not isinstance(merged_path, str) or not merged_path.strip():
        raise ValueError("Merged sync path is invalid")
    root = Path(os.path.abspath(SYNC_DATA_DIR))
    raw_path = Path(os.path.abspath(merged_path))
    expected_parent = root / user_id / task_id / "merged"
    if raw_path.parent != expected_parent:
        raise ValueError("Merged sync path is outside the managed task directory")
    if not (
        raw_path.name.startswith("merged_")
        and raw_path.suffix.lower() == ".jsonl"
    ):
        raise ValueError("Merged sync filename is invalid")
    if not os.path.lexists(expected_parent):
        return False
    _managed_directory(user_id, task_id, "merged", create=False)
    if raw_path.is_symlink():
        raise ValueError("Merged sync path must not be a symbolic link")
    try:
        resolved = raw_path.resolve(strict=True)
    except FileNotFoundError:
        return False
    if not resolved.is_file():
        raise ValueError("Merged sync path must be a regular file")
    resolved.unlink()
    return True


def _delete_managed_batch_files(
    task_id: str,
    user_id: str,
    batches: List[Dict[str, Any]],
) -> tuple[int, int]:
    """Delete invalidated batch files without allowing paths outside task scope."""
    task_id = _require_path_component(task_id, "Task ID")
    _sync_storage_user_component(user_id)
    deleted = 0
    failed = 0
    for batch in batches:
        try:
            batch_path = _resolve_managed_batch_path(task_id, user_id, batch)
            batch_path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            failed += 1
            batch_id = str(batch.get("batch_id") or "unknown")[:8]
            logger.exception(
                "[sync:%s] Failed to delete invalidated batch %s",
                task_id[:8],
                batch_id,
            )
    return deleted, failed


def _resolve_sync_milvus_connection_config(
    generation_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Use only operator-owned Milvus connection settings under authentication."""
    if get_settings().auth_enabled:
        return {}

    raw = generation_config.get("milvus_config") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        "host": raw.get("host") or os.environ.get("MILVUS_HOST", ""),
        "port": raw.get("port", 19530),
        "token": raw.get("token"),
    }




async def run_once(config: Dict[str, Any]):
    """Execute one sync cycle for a task."""
    task_id = config["task_id"]
    tag = task_id[:8]

    from ..storage.services.external_sync_service import external_sync_service

    api_url = validate_user_outbound_url(
        config["external_api_url"],
        config.get("user_id"),
    )

    # 先推进历史 REGISTERED 批次，确保 Stage-1 入库失败可重试。
    await _promote_registered_batches(config)

    # === Phase 1: Incremental fetch ===
    client = ExternalApiClient(
        api_url=api_url,
        auth_config=config["external_auth_config"],
        user_id=config.get("user_id"),
    )

    since_str = config.get("last_sync_at")
    previous_cursor: Optional[datetime] = None
    since: Optional[datetime] = None
    if since_str:
        if isinstance(since_str, str):
            since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
        else:
            since = since_str
        previous_cursor = _normalize_utc_naive(since)
        # Roll back N seconds to catch boundary records that may have been
        # missed due to same-timestamp splits. Dedup via boundary_ids below
        # ensures no duplicates regardless of whether API uses > or >=.
        rollback = config.get("boundary_rollback_seconds", 5)
        since = previous_cursor - timedelta(seconds=rollback)
    # else: since=None → first sync, fetch all data from the beginning

    external_sync_service.update_task(
        task_id, status=SyncStatus.SYNCING, error_message=None,
    )

    try:
        items = await client.fetch_all_since(since)
        items = _validate_and_deduplicate_items(items)
    except Exception as e:
        logger.warning(f"[sync:{tag}] Fetch or validation failed: {e}")
        external_sync_service.update_task(
            task_id,
            status=SyncStatus.ERROR,
            error_message=f"Fetch or validation failed: {e}",
        )
        _update_api_config_status(config, "error")
        latest = external_sync_service.get_task_raw(task_id)
        result = await _check_thresholds(latest) if latest else None
        # Existing batches may still produce a generation handoff. Preserve it
        # while reporting the source failure to a manual caller.
        return {**(result or {}), "fetch_failed": True}

    # The source is considered healthy only after its full page validates.
    _update_api_config_status(config, "active")

    if not items:
        external_sync_service.update_task(task_id, status=SyncStatus.IDLE)
        logger.debug(f"[sync:{tag}] No new data since {since}")
        latest = external_sync_service.get_task_raw(task_id)
        if latest:
            return await _check_thresholds(latest)
        return None

    validated_items = list(items)

    # Deduplicate records already seen in the previous rollback boundary.
    boundary_keys = set(config.get("last_sync_boundary_ids") or [])
    if boundary_keys:
        before_count = len(items)
        items = [
            item for item in items if not _item_matches_boundary(item, boundary_keys)
        ]
        if before_count != len(items):
            logger.info(f"[sync:{tag}] Dedup: {before_count} → {len(items)} (removed {before_count - len(items)} duplicates)")

    if not items:
        external_sync_service.update_task(task_id, status=SyncStatus.IDLE)
        logger.debug(f"[sync:{tag}] All fetched records were duplicates, skipping")
        latest = external_sync_service.get_task_raw(task_id)
        if latest:
            return await _check_thresholds(latest)
        return None

    # Secondary time-based dedup: discard records older than our query window.
    # This catches external APIs that ignore the 'since' parameter and return
    # all historical records regardless — boundary_keys alone only covers the
    # last few seconds, so older duplicates would slip through.
    if since and items:
        kept, dropped_items = [], []
        for item in items:
            if _item_is_after(item, since):
                kept.append(item)
            else:
                dropped_items.append(item)
        items = kept
        if dropped_items:
            # 这些记录早于水位线（created_at < since）。若外部源存在可见性
            # 延迟，它们可能是晚到数据——水位推进后无法自动补偿重查（多数
            # 外部 API 不支持按 id 查询）。记录被丢弃样本的 external_id 供
            # 诊断，并提示增大 boundary_rollback_seconds 覆盖可见性延迟。
            dropped_ids = [str(_item_dedup_key(item)) for item in dropped_items][:5]
            logger.warning(
                f"[sync:{tag}] Time filter removed {len(dropped_items)} records older "
                f"than {since} (samples: {dropped_ids}). Late-arriving data older than "
                f"the rollback window ({config.get('boundary_rollback_seconds', 5)}s) is "
                f"not re-fetched; increase boundary_rollback_seconds if the source has "
                f"visibility lag"
            )

    if not items:
        external_sync_service.update_task(task_id, status=SyncStatus.IDLE)
        logger.debug(f"[sync:{tag}] No records after time filter, skipping")
        latest = external_sync_service.get_task_raw(task_id)
        if latest:
            return await _check_thresholds(latest)
        return None

    logger.info(f"[sync:{tag}] Fetched {len(items)} new records")

    # The high-water mark is monotonic even when the source returns late data.
    fetched_max_time = _get_max_time(items)
    until_time = fetched_max_time
    if previous_cursor is not None and (
        until_time is None or previous_cursor > until_time
    ):
        until_time = previous_cursor

    # Rebuild from the original validated page and retain the previous boundary
    # while it can still be returned by the next rollback query.
    rollback = config.get("boundary_rollback_seconds", 5)
    new_boundary_ids = _merge_boundary_ids(
        config.get("last_sync_boundary_ids") or [],
        validated_items,
        previous_cursor=previous_cursor,
        next_cursor=until_time,
        rollback_seconds=rollback,
    )

    # Validate cursor metadata before writing a file that is not yet tracked.
    batch_path = _save_batch(config, items)

    try:
        batch, is_new = external_sync_service.create_batch(
            task_id=task_id,
            user_id=config["user_id"],
            record_count=len(items),
            storage_path=batch_path,
            since_time=since,
            until_time=until_time,
            initial_status=BatchStatus.REGISTERED,
        )
    except Exception:
        try:
            os.remove(batch_path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception(
                "[sync:%s] Failed to delete untracked batch after admission failure",
                tag,
            )
        raise

    if not is_new:
        # Duplicate batch: 优先推进已有批次，确保重复重试不会丢 Stage-1 进度。
        logger.warning(f"[sync:{tag}] Duplicate batch detected: {batch.get('batch_id', '')[:8]}")
        existing_path = (batch or {}).get("storage_path")
        if batch_path and existing_path and batch_path != existing_path:
            try:
                os.remove(batch_path)
            except FileNotFoundError:
                pass
            batch_path = ""
        external_sync_service.update_task(
            task_id,
            last_sync_at=until_time or _utcnow_naive(),
            last_sync_boundary_ids=new_boundary_ids,
            status=SyncStatus.IDLE,
        )
        if batch.get("status") == BatchStatus.REGISTERED:
            await _promote_batch_to_fetched(config, batch)
        else:
            # 历史兼容：已是 FETCHED 但计数丢失时回填。
            backfill = external_sync_service.reconcile_pending_records(task_id)
            if backfill > 0:
                logger.warning(
                    f"[sync:{tag}] Backfilled pending counters by {backfill} records after duplicate retry"
                )
    else:
        # 新批次先 REGISTERED，Stage-1 入库成功后才转 FETCHED 并计数。
        await _promote_batch_to_fetched(config, batch)
        external_sync_service.update_task(
            task_id,
            last_sync_at=until_time or _utcnow_naive(),
            last_sync_boundary_ids=new_boundary_ids,
            status=SyncStatus.IDLE,
        )

    latest = external_sync_service.get_task_raw(task_id)
    if not latest:
        return None
    return await _check_thresholds(latest)


async def _check_thresholds(config: Dict[str, Any]) -> Optional[dict]:
    """统一的阈值检查：先 Level-1 生成，再 Level-2 训练。

    Returns:
        Generation task info dict when a generation task was created,
        None otherwise.  The caller on the main event loop is responsible
        for actually launching the pipeline.
    """
    task_id = config["task_id"]
    tag = task_id[:8]
    generation_threshold = int(config.get("generation_threshold", 0) or 0)
    if generation_threshold > 0 and int(config.get("pending_record_count", 0) or 0) >= generation_threshold:
        logger.info(
            f"[sync:{tag}] Level 1 threshold reached: "
            f"{config['pending_record_count']} >= {generation_threshold}"
        )
        return await _trigger_generation(config)  # Level 2 由 generation 回调或后续周期处理

    from ..storage.services.external_sync_service import external_sync_service
    targets = external_sync_service.list_training_targets_raw(
        task_id,
        is_active=None,
    )
    active_targets = [target for target in targets if target.get("is_active", True)]

    if not targets:
        training_threshold = int(config.get("training_threshold", 0) or 0)
        if training_threshold > 0 and int(config.get("pending_training_samples", 0) or 0) >= training_threshold:
            status = config.get("status", SyncStatus.IDLE)
            if status not in (SyncStatus.TRAINING, SyncStatus.LOADING_ADAPTER, SyncStatus.GENERATING):
                logger.info(
                    f"[sync:{tag}] Level 2 threshold reached (sync cycle check): "
                    f"{config['pending_training_samples']} >= {training_threshold}"
                )
                from .level2_handler import _trigger_training
                await asyncio.to_thread(_trigger_training, config)

    # Level 2 (multi-target): Check individual training targets
    if active_targets:
        for target in active_targets:
            t_threshold = int(target.get("training_threshold", 0) or 0)
            t_pending = int(target.get("pending_training_samples", 0) or 0)
            if t_threshold > 0 and t_pending >= t_threshold:
                status = config.get("status", SyncStatus.IDLE)
                if status not in (SyncStatus.TRAINING, SyncStatus.LOADING_ADAPTER, SyncStatus.GENERATING):
                    logger.info(
                        f"[sync:{tag}] Level 2 target threshold reached for "
                        f"'{target['target_name']}' (sync cycle): {t_pending} >= {t_threshold}"
                    )
                    from .level2_handler import _schedule_next_training
                    await asyncio.to_thread(_schedule_next_training, task_id)
                    break  # Only schedule one at a time


async def _promote_registered_batches(config: Dict[str, Any]) -> int:
    """重试推进 REGISTERED 批次到 FETCHED，返回新增 pending 记录数。"""
    from ..storage.services.external_sync_service import external_sync_service

    task_id = config["task_id"]
    promoted_records = 0

    # 懒加载：有 REGISTERED 批次时预加载一次历史文档，避免逐批次重复加载
    historical_docs_cache: Optional[list] = None

    # Bound each cycle while rotating through failures so one bad page cannot
    # permanently starve the rest of the queue.
    max_batches_per_cycle = 50
    with _REGISTERED_RECOVERY_CURSOR_LOCK:
        offset = _REGISTERED_RECOVERY_OFFSETS.get(task_id, 0)
    registered_batches, total = external_sync_service.list_batches(
        task_id=task_id,
        status=BatchStatus.REGISTERED,
        limit=max_batches_per_cycle,
        offset=offset,
        oldest_first=True,
    )
    if not registered_batches and total > 0 and offset > 0:
        offset = 0
        registered_batches, total = external_sync_service.list_batches(
            task_id=task_id,
            status=BatchStatus.REGISTERED,
            limit=max_batches_per_cycle,
            offset=0,
            oldest_first=True,
        )
    with _REGISTERED_RECOVERY_CURSOR_LOCK:
        next_offset = offset + len(registered_batches)
        if total <= 0 or next_offset >= total:
            _REGISTERED_RECOVERY_OFFSETS.pop(task_id, None)
        else:
            _REGISTERED_RECOVERY_OFFSETS[task_id] = next_offset

    if registered_batches:
        for batch in registered_batches:
            if await _promote_batch_to_fetched(
                config,
                batch,
                historical_docs_cache,
            ):
                promoted_records += int(batch.get("record_count") or 0)
                if historical_docs_cache is None:
                    historical_docs_cache = []

    if promoted_records > 0:
        logger.info(
            f"[sync:{task_id[:8]}] Promoted registered batches: +{promoted_records} pending records"
        )
    return promoted_records


async def _promote_batch_to_fetched(
    config: Dict[str, Any],
    batch: Dict[str, Any],
    historical_docs_cache: Optional[list] = None,
) -> bool:
    """将单个 REGISTERED 批次完成 Stage-1 入库并提升为 FETCHED。"""
    from ..storage.services.external_sync_service import external_sync_service

    if not batch or batch.get("status") != BatchStatus.REGISTERED:
        return batch is not None and batch.get("status") == BatchStatus.FETCHED

    task_id = config["task_id"]
    tag = task_id[:8]
    batch_id = batch.get("batch_id")
    batch_path = batch.get("storage_path")
    if not batch_id or not batch_path:
        return False

    try:
        pre_index_result = await _pre_index_to_milvus(
            config, batch_path, historical_docs_cache=historical_docs_cache,
        )
        ready = bool(pre_index_result.get("ready_for_generation", True))
    except Exception as e:
        logger.warning(
            f"[sync:{tag}] Stage-1 pre-index failed for batch {batch_id[:8]}: {e}"
        )
        ready = False

    if not ready:
        return False

    return external_sync_service.promote_batch_to_fetched(
        task_id,
        batch_id,
        config["user_id"],
    )


def _serialize_batch_item(item: Dict[str, Any]) -> str:
    metadata = {
        "external_id": item.get("id", ""),
        "source": item.get("source", ""),
        "doc_id": item.get("doc_id", ""),
        "session_id": item.get("session_id", ""),
        "created_at": item.get("created_at", ""),
    }
    stable_chunk_id = _build_stable_chunk_id(metadata) or (
        f"v2:{_item_dedup_key(item)}"
    )
    if stable_chunk_id:
        metadata["chunk_id"] = stable_chunk_id
    return json.dumps(
        {"content": item.get("text", ""), "metadata": metadata},
        ensure_ascii=False,
    ) + "\n"


def _validate_and_deduplicate_items(items: Any) -> List[Dict[str, Any]]:
    """Validate a fetched page atomically and keep the first copy of each item."""
    if not isinstance(items, list):
        raise ValueError("External sync response items must be a list")

    settings = get_settings()
    now = _utcnow_naive()
    max_future = now + timedelta(
        seconds=int(settings.sync_max_future_skew_seconds)
    )
    max_records = int(settings.sync_pending_max_records_per_task)
    max_cycle_bytes = int(settings.sync_generation_max_input_bytes)
    max_record_bytes = int(settings.sync_max_record_bytes)
    unique_items: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    total_bytes = 0

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("External sync records must be objects")
        text_value = item.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError("External sync record text must be a non-empty string")
        created_at = _parse_created_at_utc_naive(item.get("created_at"))
        if created_at is None:
            raise ValueError("External sync record created_at is invalid")
        if created_at > max_future:
            raise ValueError("External sync record created_at exceeds future skew limit")

        encoded_bytes = len(_serialize_batch_item(item).encode("utf-8"))
        if encoded_bytes > max_record_bytes:
            raise SyncResourceLimitExceeded(
                "External sync JSONL record byte limit exceeded"
            )

        key = _item_dedup_key(item)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_items.append(item)
        total_bytes += encoded_bytes
        if len(unique_items) > max_records:
            raise SyncResourceLimitExceeded(
                "External sync cycle record limit exceeded"
            )
        if total_bytes > max_cycle_bytes:
            raise SyncResourceLimitExceeded(
                "External sync cycle byte limit exceeded"
            )

    return unique_items


def _save_batch(config: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
    """Save external API items as a JSONL file for the generation pipeline.

    Converts external format:
      {"source":"chunk","id":"...","text":"...","session_id":"...","doc_id":"...","created_at":"..."}
    To pipeline document format:
      {"content":"...","metadata":{"external_id":"...","source":"...","doc_id":"...","session_id":"..."}}
    """
    task_id = _require_path_component(config["task_id"], "Task ID")
    user_id = _sync_storage_user_component(config["user_id"])

    batch_dir = _managed_directory(user_id, task_id, create=True)

    batch_file = batch_dir / f"batch_{uuid.uuid4().hex}.jsonl"
    settings = get_settings()
    if len(items) > settings.sync_pending_max_records_per_task:
        raise SyncResourceLimitExceeded(
            "External sync pending record quota exceeded"
        )
    payload_bytes = 0
    for item in items:
        record_bytes = len(_serialize_batch_item(item).encode("utf-8"))
        if record_bytes > settings.sync_max_record_bytes:
            raise SyncResourceLimitExceeded(
                "External sync JSONL record byte limit exceeded"
            )
        payload_bytes += record_bytes

    created = False
    try:
        with _reserve_sync_storage(user_id, batch_file, payload_bytes):
            with open(batch_file, "x", encoding="utf-8", newline="") as f:
                created = True
                for item in items:
                    f.write(_serialize_batch_item(item))
    except Exception:
        if created:
            try:
                os.remove(batch_file)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception(
                    "[sync:%s] Failed to delete partial batch file %s",
                    task_id[:8],
                    batch_file,
                )
        raise

    logger.info(f"[sync:{task_id[:8]}] Saved batch: {batch_file} ({len(items)} records)")
    return str(batch_file)


def _update_api_config_status(config: Dict[str, Any], status: str):
    """Update the referenced external API config's status if applicable."""
    api_config_id = config.get("external_api_config_id")
    if not api_config_id:
        return
    try:
        from ..storage.services.external_api_config_service import external_api_config_service
        external_api_config_service.update_config(api_config_id, status=status)
    except Exception as e:
        logger.debug(f"[sync] Failed to update API config status: {e}")


def _get_max_time(items: List[Dict[str, Any]]) -> Optional[datetime]:
    """Get the latest created_at from items."""
    max_time = None
    for item in items:
        t = _parse_created_at_utc_naive(item.get("created_at"))
        if t is None:
            continue
        if max_time is None or t > max_time:
            max_time = t
    return max_time


def _item_dedup_key(item: Dict[str, Any]) -> str:
    """Build an unambiguous private identity for boundary deduplication."""
    business_keys = [
        str(item.get("id") or ""),
        str(item.get("session_id") or ""),
        str(item.get("doc_id") or ""),
    ]
    identity: Any = {"business_keys": business_keys}
    if not (business_keys[0] or business_keys[2]):
        identity = {"record": item}
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_item_dedup_key(item: Dict[str, Any]) -> str:
    """Reproduce boundary keys persisted before the hashed-key migration."""
    return (
        f"{item.get('id', '')}:"
        f"{item.get('session_id', '')}:"
        f"{item.get('doc_id', '')}"
    )


def _item_matches_boundary(item: Dict[str, Any], boundary_keys: set[str]) -> bool:
    """Match both current hashes and transient legacy boundary keys."""
    return (
        _item_dedup_key(item) in boundary_keys
        or _legacy_item_dedup_key(item) in boundary_keys
    )


def _build_stable_chunk_id(metadata: Dict[str, Any]) -> str:
    """Build a versioned, unambiguous identifier from all business key parts."""
    external_id = str(metadata.get("external_id") or "").strip()
    session_id = str(metadata.get("session_id") or "").strip()
    doc_id = str(metadata.get("doc_id") or "").strip()

    # A session identifies a namespace, not an individual record. Records with
    # no external/doc ID fall back to the full canonical item hash at ingress.
    if not (external_id or doc_id):
        return ""

    encoded = json.dumps(
        {
            "external_id": external_id,
            "session_id": session_id,
            "doc_id": doc_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"v2:{hashlib.sha256(encoded).hexdigest()}"


def _item_is_after(item: Dict[str, Any], threshold: datetime) -> bool:
    """Check if a validated item's created_at is at or after the threshold."""
    created_at = item.get("created_at")
    if not created_at:
        return False
    t = _parse_created_at_utc_naive(created_at)
    if t is None:
        return False
    try:
        thr = _normalize_utc_naive(threshold)
        return t >= thr
    except (ValueError, TypeError):
        return False






def _get_boundary_ids(items: List[Dict[str, Any]], max_time: Optional[datetime],
                      rollback_seconds: int = 5) -> List[str]:
    """Get composite dedup keys of records within the rollback window for dedup on next sync.

    Since we roll back N seconds on the next request, we need to track all record
    keys with created_at >= (max_time - N) to avoid duplicates.
    Uses composite key (id:session_id:doc_id) instead of just id.
    """
    if not max_time:
        return []
    settings = get_settings()
    cutoff = max_time - timedelta(seconds=rollback_seconds)
    boundary_keys = []
    encoded_bytes = 2
    for item in items:
        t = _parse_created_at_utc_naive(item.get("created_at"))
        if t is not None and t >= cutoff:
            key = _item_dedup_key(item)
            key_bytes = len(json.dumps(key).encode("utf-8"))
            projected_bytes = encoded_bytes + key_bytes
            if boundary_keys:
                projected_bytes += 2
            if (
                len(boundary_keys) >= settings.sync_boundary_max_ids
                or projected_bytes > settings.sync_boundary_max_bytes
            ):
                raise SyncResourceLimitExceeded(
                    "External sync boundary deduplication limit exceeded"
                )
            boundary_keys.append(key)
            encoded_bytes = projected_bytes
    return boundary_keys


def _merge_boundary_ids(
    previous_keys: List[str],
    fetched_items: List[Dict[str, Any]],
    *,
    previous_cursor: Optional[datetime],
    next_cursor: Optional[datetime],
    rollback_seconds: int,
) -> List[str]:
    """Build the next bounded boundary without dropping still-queryable keys."""
    if next_cursor is None:
        return []

    fresh_keys = _get_boundary_ids(
        fetched_items,
        next_cursor,
        rollback_seconds,
    )
    retain_previous = (
        previous_cursor is not None
        and previous_cursor
        >= next_cursor - timedelta(seconds=rollback_seconds)
    )
    candidates: List[str] = []
    if retain_previous:
        # Never re-persist legacy plaintext identities during the hash migration.
        candidates.extend(
            key
            for key in previous_keys
            if isinstance(key, str)
            and len(key) == 64
            and all(char in "0123456789abcdefABCDEF" for char in key)
        )
    candidates.extend(fresh_keys)

    settings = get_settings()
    result: List[str] = []
    seen: set[str] = set()
    encoded_bytes = 2
    for key in candidates:
        if key in seen:
            continue
        key_bytes = len(json.dumps(key).encode("utf-8"))
        projected_bytes = encoded_bytes + key_bytes + (2 if result else 0)
        if (
            len(result) >= settings.sync_boundary_max_ids
            or projected_bytes > settings.sync_boundary_max_bytes
        ):
            raise SyncResourceLimitExceeded(
                "External sync boundary deduplication limit exceeded"
            )
        seen.add(key)
        result.append(key)
        encoded_bytes = projected_bytes
    return result


def _normalize_utc_naive(dt: datetime) -> datetime:
    """Normalize datetime to sync-mode naive form for stable comparisons."""
    return sync_normalize_naive(dt)


def _parse_created_at_utc_naive(value: Any) -> Optional[datetime]:
    """Parse created_at value and normalize to UTC naive datetime."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _normalize_utc_naive(parsed)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Milvus pre-index helpers (Stage 1 → vector DB sync)
# ---------------------------------------------------------------------------

def _read_documents_from_batch(
    batch_path: str,
    *,
    max_docs: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> tuple[list, int, bool]:
    """从 batch JSONL 加载 Document 对象供 pre_index_all_chunks 使用。"""
    from ..generation.steps.base import Document

    documents = []
    bytes_read = 0
    limit_reached = False
    try:
        with open(batch_path, "rb") as batch_file:
            line_num = 0
            while True:
                if max_bytes is None:
                    raw_line = batch_file.readline()
                else:
                    remaining_bytes = max(max_bytes - bytes_read, 0)
                    raw_line = batch_file.readline(remaining_bytes + 1)
                if not raw_line:
                    break
                if max_bytes is not None and len(raw_line) > remaining_bytes:
                    limit_reached = True
                    break
                line_num += 1
                bytes_read += len(raw_line)
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    content = obj.get("content", "")
                    if not content:
                        continue
                    metadata = obj.get("metadata", {})
                    doc_id = (
                        _build_stable_chunk_id(metadata)
                        or metadata.get("chunk_id")
                        or metadata.get("source_doc_id")
                        or "v2:" + hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest()
                    )
                    documents.append(
                        Document(content=content, metadata=metadata, doc_id=doc_id)
                    )
                    if max_docs is not None and len(documents) >= max_docs:
                        limit_reached = True
                        break
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return documents, bytes_read, limit_reached


def _load_documents_from_batch(batch_path: str) -> list:
    """Load one batch for current-batch Stage-1 processing."""
    documents, _bytes_read, _limit_reached = _read_documents_from_batch(batch_path)
    return documents


def _iter_document_windows_from_batch(
    batch_path: str,
    *,
    max_docs: int,
    max_bytes: int,
) -> Iterator[List[Any]]:
    """Yield every JSONL document in bounded in-memory windows."""
    from ..generation.steps.base import Document

    max_line_bytes = min(
        max_bytes,
        int(getattr(get_settings(), "sync_max_record_bytes", max_bytes)),
    )
    with open(batch_path, "rb") as batch_file:
        line_num = 0
        reached_eof = False
        while not reached_eof:
            documents: List[Any] = []
            window_bytes = 0
            while len(documents) < max_docs and window_bytes < max_bytes:
                remaining_bytes = max_bytes - window_bytes
                allowed_bytes = min(remaining_bytes, max_line_bytes)
                line_start = batch_file.tell()
                raw_line = batch_file.readline(allowed_bytes + 1)
                if not raw_line:
                    reached_eof = True
                    break
                if len(raw_line) > allowed_bytes:
                    batch_file.seek(line_start)
                    if documents or window_bytes:
                        break
                    raise SyncResourceLimitExceeded(
                        "External sync JSONL record byte limit exceeded"
                    )

                line_num += 1
                window_bytes += len(raw_line)
                line = raw_line.decode("utf-8").strip()
                if not line:
                    raise ValueError(
                        f"External sync JSONL record {line_num} is empty"
                    )
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"External sync JSONL record {line_num} is invalid"
                    ) from exc
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"External sync JSONL record {line_num} must be an object"
                    )
                content = obj.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"External sync JSONL record {line_num} has empty content"
                    )
                metadata = obj.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"External sync JSONL record {line_num} has invalid metadata"
                    )
                doc_id = (
                    _build_stable_chunk_id(metadata)
                    or metadata.get("chunk_id")
                    or metadata.get("source_doc_id")
                    or "v2:" + hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest()
                )
                documents.append(
                    Document(content=content, metadata=metadata, doc_id=doc_id)
                )

            if documents:
                yield documents


def _iter_historical_document_windows(
    config: Dict[str, Any],
) -> Iterator[List[Any]]:
    """Stream every historical batch through bounded document windows."""
    from ..storage.services.external_sync_service import external_sync_service

    task_id = config["task_id"]
    settings = get_settings()
    offset = 0
    limit = 200
    while True:
        batches, total = external_sync_service.list_batches(
            task_id=task_id,
            limit=limit,
            offset=offset,
        )
        if not batches:
            break
        for batch in batches:
            batch_path = batch.get("storage_path")
            if not batch_path:
                raise FileNotFoundError("Tracked external sync batch has no storage path")
            path = Path(str(batch_path))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Tracked external sync batch file is missing: {batch.get('batch_id', '')}"
                )
            if path.stat().st_size == 0:
                raise ValueError("Tracked external sync batch file is empty")
            yield from _iter_document_windows_from_batch(
                str(path),
                max_docs=settings.sync_historical_max_docs,
                max_bytes=settings.sync_historical_max_bytes,
            )
        offset += len(batches)
        if offset >= total:
            break


def _load_all_historical_documents(config: Dict[str, Any]) -> list:
    """Load documents from ALL historical batches for adapter backfill.

    When a new adapter is loaded (new training) or re-loaded (previously unloaded),
    it needs all historical data indexed into its Milvus collection.

    - New adapter (collection doesn't exist): pre_index_all_chunks indexes everything.
    - Re-loaded adapter (collection preserved): pre_index_all_chunks skips existing
      chunks via Milvus cache, only indexes data added while the adapter was offline.

    Dedup is performed by doc_id to avoid duplicate Documents across overlapping batches.
    """
    from ..storage.services.external_sync_service import external_sync_service

    task_id = config["task_id"]
    tag = task_id[:8]
    all_docs: list = []
    seen_doc_ids: set = set()
    offset = 0
    limit = 200
    batch_count = 0
    # 文档上限，防止长期运行的 sync 任务加载百万级文档导致 OOM（#5）
    settings = get_settings()
    max_docs = settings.sync_historical_max_docs
    max_bytes = settings.sync_historical_max_bytes
    input_bytes = 0
    capped = False

    while True:
        batches, total = external_sync_service.list_batches(
            task_id=task_id, limit=limit, offset=offset,
        )
        if not batches:
            break
        for batch in batches:
            batch_path = batch.get("storage_path")
            if not batch_path or not os.path.exists(batch_path):
                continue
            docs, batch_bytes, batch_capped = _read_documents_from_batch(
                batch_path,
                max_docs=max_docs - len(all_docs),
                max_bytes=max_bytes - input_bytes,
            )
            input_bytes += batch_bytes
            batch_count += 1
            for doc in docs:
                doc_id = getattr(doc, "doc_id", "")
                # 跳过空 doc_id 的文档（#11: 避免去重失效）
                if not doc_id:
                    continue
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                all_docs.append(doc)
                if len(all_docs) >= max_docs:
                    logger.warning(
                        f"[sync:{tag}] Historical docs capped at {max_docs}, "
                        f"some older data may not be indexed for new adapters"
                    )
                    capped = True
                    break
            if batch_capped or len(all_docs) >= max_docs or input_bytes >= max_bytes:
                capped = True
                break
        if capped:
            break
        offset += len(batches)
        if offset >= total:
            break

    if capped:
        logger.warning(
            "[sync:%s] Historical backfill reached its bounded input limit",
            tag,
        )

    if all_docs:
        logger.info(
            f"[sync:{tag}] Loaded {len(all_docs)} historical documents "
            f"from {batch_count} batches for adapter backfill"
        )
    return all_docs


def _get_training_target_adapter_targets(
    config: Dict[str, Any],
    emb_hash: str,
    task_hash: str,
    milvus_client_class: Any,
) -> List[Dict[str, Any]]:
    """Resolve loaded adapters from each target's own healthy runtime binding."""
    from ..deployment.adapter_service import AdapterService
    from ..deployment.deployment_service import deployment_service
    from ..storage.services.external_sync_service import external_sync_service

    task_id = config["task_id"]
    loaded_training_ids: Dict[Optional[str], set[str]] = {}
    training_offset = 0
    training_page_size = 1000
    while True:
        training_page, training_total = external_sync_service.list_trainings(
            task_id=task_id,
            status=SyncTrainingStatus.ADAPTER_LOADED,
            limit=training_page_size,
            offset=training_offset,
        )
        for training in training_page:
            training_task_id = training.get("training_task_id")
            if training_task_id:
                loaded_training_ids.setdefault(training.get("target_id"), set()).add(
                    training_task_id
                )
        training_offset += len(training_page)
        if not training_page or training_offset >= training_total:
            break

    adapter_service = AdapterService()
    task_api_config_id = (config.get("external_api_config_id") or "").strip()
    seen_runtime_adapters: set[tuple[str, Optional[str], str]] = set()
    adapter_targets: List[Dict[str, Any]] = []
    for target in external_sync_service.list_training_targets_raw(
        task_id,
        is_active=None,
    ):
        desired_adapter_id = target.get("current_adapter_id")
        desired_adapter_name = target.get("current_adapter_name")
        if not desired_adapter_id and not desired_adapter_name:
            continue
        deployment_id = (
            target.get("base_deployment_id") or config.get("base_deployment_id")
        )
        if not deployment_id:
            continue
        deployment_replica_id = (
            target.get("base_deployment_replica_id")
            if target.get("base_deployment_id")
            else config.get("base_deployment_replica_id")
        )
        try:
            deployment, replica = deployment_service.resolve_replica_selection(
                deployment_id,
                deployment_replica_id,
                user_id=config.get("user_id"),
                require_healthy=True,
            )
            if replica is None and deployment.get("status") != "running":
                raise ValueError("deployment is not running")
            deployment_api_config_id = _get_deployment_external_api_config_id(
                deployment
            )
            if (
                task_api_config_id
                and deployment_api_config_id != task_api_config_id
            ):
                raise ValueError("deployment api_config mismatch")
            resolved_replica_id = (
                replica.get("replica_id") if replica is not None else None
            )
            expected_training_ids = loaded_training_ids.get(
                target.get("target_id"), set()
            )
            for adapter in adapter_service.list_loaded_adapters(
                deployment_id,
                deployment_replica_id=resolved_replica_id,
                user_id=config.get("user_id"),
            ):
                adapter_id = adapter.get("adapter_id")
                adapter_name = adapter.get("adapter_name")
                if (
                    adapter.get("status") != "loaded"
                    or adapter.get("source_task_id") not in expected_training_ids
                    or not adapter_id
                    or not adapter_name
                ):
                    continue
                if desired_adapter_id:
                    if adapter_id != desired_adapter_id:
                        continue
                elif adapter_name != desired_adapter_name:
                    continue
                runtime_key = (deployment_id, resolved_replica_id, adapter_id)
                if runtime_key in seen_runtime_adapters:
                    break
                seen_runtime_adapters.add(runtime_key)
                fingerprint = json.dumps(
                    {
                        "base": emb_hash,
                        "deployment_id": deployment_id,
                        "deployment_replica_id": resolved_replica_id,
                        "adapter_id": adapter_id,
                        "adapter_name": adapter_name,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                adapter_hash = hashlib.sha256(fingerprint).hexdigest()[:16]
                adapter_targets.append(
                    {
                        "type": "adapter",
                        "target_id": target.get("target_id"),
                        "deployment_id": deployment_id,
                        "deployment_replica_id": resolved_replica_id,
                        "adapter_id": adapter_id,
                        "collection_name": milvus_client_class.sanitize_collection_name(
                            f"tf_sync_v3_{task_hash}_a{adapter_id[:8]}_{adapter_hash}"
                        ),
                        "fingerprint": adapter_hash,
                        "model_name": adapter_name,
                        "label": adapter_name,
                    }
                )
                break
        except Exception as exc:
            logger.warning(
                "[sync:%s] Skip adapter target %s: %s",
                task_id[:8],
                target.get("target_id"),
                exc,
            )
    return adapter_targets


def _get_sync_targets(
    config: Dict[str, Any], embedding_config_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """获取所有需要同步的 Milvus 目标（基座模型 + 已加载 adapters）。

    返回列表，每项包含:
      - collection_name: Milvus collection 名
      - model_name: 用于 EmbeddingClient 的 model 参数
      - label: 日志标签（如 "base" 或 adapter_name）
    """
    from ..generation.clients.milvus_client import MilvusClient

    targets: List[Dict[str, Any]] = []
    task_id = config["task_id"]
    base_model = embedding_config_data.get("model", "")
    if not base_model:
        return targets

    # 目标 1: 基座模型
    def fingerprint_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): fingerprint_value(child)
                for key, child in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
                if not _is_sensitive_config_key(key)
            }
        if isinstance(value, (list, tuple)):
            return [fingerprint_value(child) for child in value]
        return value

    fingerprint_payload = {
        "version": 3,
        "source": {
            "external_api_config_id": config.get("external_api_config_id") or "",
            "external_api_url": config.get("external_api_url") or "",
        },
        "embedding": fingerprint_value(embedding_config_data),
    }
    encoded_fingerprint = json.dumps(
        fingerprint_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    emb_hash = hashlib.sha256(encoded_fingerprint).hexdigest()[:16]
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    base_collection = MilvusClient.sanitize_collection_name(
        f"tf_sync_v3_{task_hash}_{emb_hash}"
    )
    targets.append({
        "type": "base",
        "collection_name": base_collection,
        "fingerprint": emb_hash,
        "model_name": base_model,
        "label": "base",
    })

    try:
        targets.extend(
            _get_training_target_adapter_targets(
                config,
                emb_hash,
                task_hash,
                MilvusClient,
            )
        )
    except Exception as exc:
        logger.warning(
            "[sync:%s] Failed to query training target adapters: %s",
            task_id[:8],
            exc,
            exc_info=True,
        )

    # 目标 2+: 部署上已加载的 adapters
    deployment_id = config.get("base_deployment_id")
    if deployment_id:
        try:
            from ..deployment.deployment_service import deployment_service
            from ..deployment.adapter_service import AdapterService
            from ..storage.services.external_sync_service import external_sync_service

            deployment, replica = deployment_service.resolve_replica_selection(
                deployment_id,
                config.get("base_deployment_replica_id"),
                user_id=config.get("user_id"),
                require_healthy=True,
            )
            resolved_replica_id = (
                replica.get("replica_id") if replica is not None else None
            )
            if replica is None and deployment.get("status") != "running":
                logger.info(
                    f"[sync:{task_id[:8]}] Deployment not running ({deployment.get('status')}), "
                    "skip adapter targets"
                )
                return targets

            # 租户隔离：sync 任务与 deployment 绑定的 external_api_config_id 必须一致。
            task_api_cfg = (config.get("external_api_config_id") or "").strip()
            dep_api_cfg = _get_deployment_external_api_config_id(deployment)
            if task_api_cfg and task_api_cfg != dep_api_cfg:
                logger.warning(
                    f"[sync:{task_id[:8]}] Deployment api_config mismatch, "
                    f"skip adapter targets (deployment={dep_api_cfg}, task={task_api_cfg})"
                )
                return targets

            # 只同步“训练完成且已加载”的 adapter：
            # 1) 来自 sync 训练追踪且状态 adapter_loaded
            # 2) 当前 deployment 上实际处于 loaded
            loaded_training_ids: set[str] = set()
            training_offset = 0
            training_page_size = 1000
            while True:
                training_page, training_total = external_sync_service.list_trainings(
                    task_id=task_id,
                    status=SyncTrainingStatus.ADAPTER_LOADED,
                    limit=training_page_size,
                    offset=training_offset,
                )
                loaded_training_ids.update(
                    training["training_task_id"]
                    for training in training_page
                    if training.get("training_task_id")
                    and training.get("target_id") is None
                )
                training_offset += len(training_page)
                if not training_page or training_offset >= training_total:
                    break

            adapter_service = AdapterService()
            adapters = adapter_service.list_loaded_adapters(
                deployment_id,
                deployment_replica_id=resolved_replica_id,
                user_id=config.get("user_id"),
            )
            for adapter in adapters:
                if adapter.get("status") != "loaded":
                    continue
                source_task_id = adapter.get("source_task_id")
                # 仅同步本 sync 任务训练产出的、且已标记 adapter_loaded 的 adapter。
                if source_task_id not in loaded_training_ids:
                    continue
                adapter_id = adapter.get("adapter_id")
                adapter_name = adapter.get("adapter_name")
                if not adapter_id or not adapter_name:
                    continue
                adapter_fingerprint = json.dumps(
                    {
                        "base": emb_hash,
                        "deployment_id": deployment_id,
                        "deployment_replica_id": resolved_replica_id,
                        "adapter_id": adapter_id,
                        "adapter_name": adapter_name,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                a_hash = hashlib.sha256(adapter_fingerprint).hexdigest()[:16]
                a_collection = MilvusClient.sanitize_collection_name(
                    f"tf_sync_v3_{task_hash}_a{adapter_id[:8]}_{a_hash}"
                )
                targets.append({
                    "type": "adapter",
                    "target_id": None,
                    "deployment_id": deployment_id,
                    "deployment_replica_id": resolved_replica_id,
                    "adapter_id": adapter_id,
                    "collection_name": a_collection,
                    "fingerprint": a_hash,
                    "model_name": adapter_name,
                    "label": adapter_name,
                })
        except Exception as e:
            logger.warning(
                f"[sync:{task_id[:8]}] Failed to query loaded adapters: {e}",
                exc_info=True,
            )

    return targets


def _get_deployment_external_api_config_id(dep: Dict[str, Any]) -> str:
    """读取 deployment 绑定的 external_api_config_id（兼容新旧字段位置）。"""
    direct = (dep.get("external_api_config_id") or "").strip()
    if direct:
        return direct
    cfg = dep.get("config") or {}
    raw = cfg.get("external_api_config_id")
    return raw.strip() if isinstance(raw, str) else ""


def _split_docs_for_targets(
    config: Dict[str, Any],
    batch_documents: List[Any],
    targets: List[Dict[str, Any]],
    historical_docs: Optional[List[Any]] = None,
    adapter_assigned_counts: Optional[List[int]] = None,
) -> Dict[str, List[Any]]:
    """按目标集合拆分文档。

    规则：
    - base 目标始终使用当前批次文档（增量入库）。
    - adapter 目标默认复制同一份文档。
    - 当 generation_config.adapter_request_balancing=true 时，
      adapter 目标按权重切分（总量守恒），用于多 adapter 的负载分发。
    """
    docs_by_collection: Dict[str, List[Any]] = {}
    if not targets:
        return docs_by_collection

    gen_cfg = config.get("generation_config") or {}
    balancing_enabled = bool(gen_cfg.get("adapter_request_balancing"))
    raw_weights = gen_cfg.get("adapter_request_weights") or {}

    adapter_targets: List[Dict[str, Any]] = []
    for target in targets:
        collection = target.get("collection_name")
        if not collection:
            continue
        if target.get("type") == "adapter":
            adapter_targets.append(target)
        else:
            docs_by_collection[collection] = list(batch_documents)

    if not adapter_targets:
        return docs_by_collection

    source_docs = list(historical_docs if historical_docs is not None else batch_documents)
    if not source_docs:
        for target in adapter_targets:
            collection = target.get("collection_name")
            if collection:
                docs_by_collection[collection] = []
        return docs_by_collection

    if not balancing_enabled:
        for target in adapter_targets:
            collection = target.get("collection_name")
            if collection:
                docs_by_collection[collection] = list(source_docs)
        return docs_by_collection

    # 基于权重的确定性最小负载分配：每条文档只分配给一个 adapter，避免重复计算。
    weights: List[float] = []
    for target in adapter_targets:
        candidates = [
            target.get("adapter_id"),
            target.get("adapter_name"),
            target.get("label"),
            target.get("target_id"),
        ]
        weight_val: Optional[float] = None
        for key in candidates:
            if key and key in raw_weights:
                try:
                    parsed = float(raw_weights[key])
                except (TypeError, ValueError):
                    parsed = 1.0
                weight_val = parsed
                break
        if weight_val is None:
            weight_val = 1.0
        if weight_val <= 0:
            weight_val = 1.0
        weights.append(weight_val)

    adapter_slices: List[List[Any]] = [[] for _ in adapter_targets]
    assigned_counts = adapter_assigned_counts
    if assigned_counts is None:
        assigned_counts = [0 for _ in adapter_targets]
    elif len(assigned_counts) != len(adapter_targets):
        raise ValueError("Adapter balancing state does not match target count")
    for doc in source_docs:
        best_idx = 0
        best_score = float("inf")
        for idx, weight in enumerate(weights):
            score = assigned_counts[idx] / weight
            if score < best_score:
                best_score = score
                best_idx = idx
        adapter_slices[best_idx].append(doc)
        assigned_counts[best_idx] += 1

    for idx, target in enumerate(adapter_targets):
        collection = target.get("collection_name")
        if collection:
            docs_by_collection[collection] = adapter_slices[idx]

    return docs_by_collection


async def _pre_index_to_milvus(
    config: Dict[str, Any],
    batch_path: str,
    *,
    historical_docs_cache: Optional[list] = None,
) -> Dict[str, Any]:
    """Stage 1 数据入库：将新拉取的文档 embed 并入库到所有目标 Milvus collection。

    遍历基座模型 + 所有已加载 adapter，每个目标独立 embed + 入库。
    每个 adapter 产生不同的向量，因此需要独立的 EmbeddingClient。
    失败仅打 warning，不影响主流程。

    Args:
        config: 同步任务配置
        batch_path: 当前批次 JSONL 文件路径
        historical_docs_cache: 预加载的历史文档缓存，避免多批次时重复加载。
            为 None 时自动加载；为空列表 [] 时跳过 adapter 回填。
    """
    from ..storage.services.model_config_service import model_config_service
    from ..storage.services.external_sync_service import external_sync_service
    from ..storage.services.milvus_collection_service import (
        milvus_collection_service,
    )
    from ..generation.steps.embedding_filter_step import EmbeddingFilterStep
    from ..generation.clients.embedding_client import EmbeddingClient, EmbeddingConfig
    from ..generation.clients.milvus_client import MilvusClient, MilvusConfig

    gen_cfg = config.get("generation_config") or {}
    embedding_config_data = _resolve_config_from_gen_cfg(
        gen_cfg,
        "embedding_config",
        "embedding",
        model_config_service,
        expected_user_id=config.get("user_id"),
    )
    if not embedding_config_data:
        # 无 embedding 配置：视为 Stage-1 可通过（不阻塞后续）。
        return {
            "ready_for_generation": True,
            "indexed_records": 0,
            "successful_targets": [],
            "failed_targets": [],
        }

    targets = _get_sync_targets(config, embedding_config_data)
    if not targets:
        return {
            "ready_for_generation": False,
            "indexed_records": 0,
            "successful_targets": [],
            "failed_targets": [],
        }

    tag = config["task_id"][:8]

    base_collection = targets[0]["collection_name"]
    base_requires_backfill = config.get("milvus_collection_name") != base_collection

    for target in targets:
        milvus_collection_service.register_collection(
            collection_name=target["collection_name"],
            embedding_config_id=embedding_config_data.get("config_id"),
            embedding_model=target.get("model_name"),
            embedding_endpoint=embedding_config_data.get("endpoint"),
            dim=int(embedding_config_data.get("dimension") or 1024),
            user_id=config.get("user_id"),
            sync_task_id=config["task_id"],
        )

    # 共享一个 MilvusClient 连接（支持多 collection 操作）
    milvus_client = MilvusClient(
        MilvusConfig(**_resolve_sync_milvus_connection_config(gen_cfg))
    )
    milvus_client.connect()

    base_targets = [t for t in targets if t.get("type") != "adapter"]
    adapter_targets = [t for t in targets if t.get("type") == "adapter"]
    settings = get_settings()
    window_max_docs = int(settings.sync_historical_max_docs)
    window_max_bytes = int(settings.sync_historical_max_bytes)
    successful_targets: List[str] = []
    failed_targets: List[str] = []
    indexed_records = 0
    base_ok = bool(base_targets)
    adapter_ok = True

    def record_once(values: List[str], value: str) -> None:
        if value not in values:
            values.append(value)

    async def index_window(target: Dict[str, Any], documents: List[Any]) -> int:
        # Use a fresh client/filter so per-window caches remain bounded.
        emb_client = EmbeddingClient(EmbeddingConfig(
            endpoint=embedding_config_data.get("endpoint", ""),
            model=target["model_name"],
            api_key=embedding_config_data.get("api_key"),
            batch_size=embedding_config_data.get("batch_size", 32),
            concurrency=embedding_config_data.get("concurrency", 20),
            user_id=config.get("user_id"),
        ))
        filter_step = EmbeddingFilterStep(
            embedding_client=emb_client,
            milvus_client=milvus_client,
            threshold=1.0,
            collection_name=target["collection_name"],
            default_metadata={"sync_task_id": config["task_id"]},
        )
        async with emb_client:
            return await filter_step.pre_index_all_chunks(documents)

    async def index_target_window(target: Dict[str, Any], documents: List[Any]) -> bool:
        if not documents:
            return True
        try:
            inserted = await index_window(target, documents)
            logger.info(
                "[sync:%s] Pre-index [%s] window complete: %s new chunks -> '%s'",
                tag,
                target["label"],
                inserted,
                target["collection_name"],
            )
            return True
        except Exception as exc:
            logger.warning(
                "[sync:%s] Pre-index [%s] window failed: %s",
                tag,
                target["label"],
                exc,
            )
            record_once(failed_targets, target["label"])
            return False

    client_closed = False

    def close_milvus_client_once() -> None:
        nonlocal client_closed
        if client_closed:
            return
        client_closed = True
        try:
            milvus_client.close()
        except Exception:
            pass

    with ExitStack() as guard_stack:
        # The first callback covers failures while entering the guard. Once the
        # guard is active, the second callback closes the client before the
        # collection locks are released; the first callback then becomes a no-op.
        guard_stack.callback(close_milvus_client_once)
        guard_stack.enter_context(
            milvus_collection_service.consumption_guard(
                [target["collection_name"] for target in targets],
                user_id=config.get("user_id"),
            )
        )
        guard_stack.callback(close_milvus_client_once)

        saw_batch_documents = False
        for documents in _iter_document_windows_from_batch(
            batch_path,
            max_docs=window_max_docs,
            max_bytes=window_max_bytes,
        ):
            saw_batch_documents = True
            indexed_records += len(documents)
            for target in base_targets:
                if not await index_target_window(target, documents):
                    base_ok = False
                    break
                record_once(successful_targets, target["label"])
            if not base_ok:
                break
        if not saw_batch_documents:
            base_ok = False

        if base_ok and base_requires_backfill:
            for documents in _iter_historical_document_windows(config):
                for target in base_targets:
                    if not await index_target_window(target, documents):
                        base_ok = False
                        break
                if not base_ok:
                    break

        # [] means an earlier REGISTERED batch completed the full adapter
        # backfill in this recovery cycle.
        if base_ok and adapter_targets and historical_docs_cache != []:
            adapter_counts = [0 for _ in adapter_targets]
            history_seen = False
            if historical_docs_cache is None:
                historical_windows: Iterator[List[Any]] = (
                    _iter_historical_document_windows(config)
                )
            else:
                historical_windows = iter([historical_docs_cache])

            for documents in historical_windows:
                if not documents:
                    continue
                history_seen = True
                docs_by_collection = _split_docs_for_targets(
                    config=config,
                    batch_documents=[],
                    targets=adapter_targets,
                    historical_docs=documents,
                    adapter_assigned_counts=adapter_counts,
                )
                for target in adapter_targets:
                    target_documents = docs_by_collection.get(
                        target["collection_name"], []
                    )
                    if not await index_target_window(target, target_documents):
                        adapter_ok = False
                        break
                    if target_documents:
                        record_once(successful_targets, target["label"])
                if not adapter_ok:
                    break

            if adapter_ok and not history_seen:
                for documents in _iter_document_windows_from_batch(
                    batch_path,
                    max_docs=window_max_docs,
                    max_bytes=window_max_bytes,
                ):
                    docs_by_collection = _split_docs_for_targets(
                        config=config,
                        batch_documents=[],
                        targets=adapter_targets,
                        historical_docs=documents,
                        adapter_assigned_counts=adapter_counts,
                    )
                    for target in adapter_targets:
                        target_documents = docs_by_collection.get(
                            target["collection_name"], []
                        )
                        if not await index_target_window(target, target_documents):
                            adapter_ok = False
                            break
                        if target_documents:
                            record_once(successful_targets, target["label"])
                    if not adapter_ok:
                        break
    if base_ok and base_requires_backfill:
        external_sync_service.update_task(
            config["task_id"], milvus_collection_name=base_collection,
        )
        config["milvus_collection_name"] = base_collection

    return {
        "ready_for_generation": base_ok and adapter_ok,
        "indexed_records": indexed_records if base_ok else 0,
        "successful_targets": successful_targets,
        "failed_targets": failed_targets,
    }


# ---------------------------------------------------------------------------
# Batch management
# ---------------------------------------------------------------------------

def _iter_normalized_jsonl_bytes(path: Path) -> Iterator[bytes]:
    """Yield non-empty JSONL records without ever allocating an oversized line."""
    max_record_bytes = int(get_settings().sync_max_record_bytes)
    with open(path, "rb") as input_file:
        while True:
            raw_line = input_file.readline(max_record_bytes + 1)
            if not raw_line:
                return
            if len(raw_line) > max_record_bytes:
                raise SyncResourceLimitExceeded(
                    "External sync JSONL record byte limit exceeded"
                )
            normalized = raw_line.strip()
            if normalized:
                yield normalized


def _normalized_jsonl_stats(path: Path) -> tuple[int, int]:
    records = 0
    byte_count = 0
    for normalized in _iter_normalized_jsonl_bytes(path):
        records += 1
        byte_count += len(normalized) + 1
    return records, byte_count


def _select_generation_batches(
    config: Dict[str, Any],
    batches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select the oldest FIFO prefix that fits one bounded generation."""
    task_id = _require_path_component(config["task_id"], "Task ID")
    user_id = config["user_id"]
    _sync_storage_user_component(user_id)
    settings = get_settings()
    selected: List[Dict[str, Any]] = []
    selected_records = 0
    selected_bytes = 0

    for batch in batches:
        path = _resolve_managed_batch_path(task_id, user_id, batch)
        actual_records, actual_bytes = _normalized_jsonl_stats(path)
        projected_records = selected_records + max(
            int(batch.get("record_count") or actual_records), 0
        )
        projected_bytes = selected_bytes + actual_bytes
        would_exceed = (
            len(selected) + 1 > settings.sync_pending_max_batches_per_task
            or projected_records > settings.sync_pending_max_records_per_task
            or projected_bytes > settings.sync_generation_max_input_bytes
        )
        if would_exceed:
            if selected:
                break
            raise SyncResourceLimitExceeded(
                "Oldest external sync batch exceeds generation input limits"
            )
        selected.append(batch)
        selected_records = projected_records
        selected_bytes = projected_bytes

    return selected


def _merge_batches(config: Dict[str, Any], batches: List[Dict[str, Any]]) -> str:
    """Merge multiple batch files into a single input file for generation."""
    task_id = _require_path_component(config["task_id"], "Task ID")
    user_id = config["user_id"]
    user_component = _sync_storage_user_component(user_id)
    settings = get_settings()
    if len(batches) > settings.sync_pending_max_batches_per_task:
        raise SyncResourceLimitExceeded(
            "External sync pending batch quota exceeded"
        )
    declared_records = sum(
        max(int(batch.get("record_count") or 0), 0) for batch in batches
    )
    if declared_records > settings.sync_pending_max_records_per_task:
        raise SyncResourceLimitExceeded(
            "External sync pending record quota exceeded"
        )

    attempt_id = _require_path_component(
        config.get("_generation_attempt_id") or uuid.uuid4().hex,
        "Generation attempt ID",
    )
    source_paths = [
        _resolve_managed_batch_path(task_id, user_id, batch) for batch in batches
    ]
    output_bytes = 0
    actual_records = 0
    for batch_path in source_paths:
        batch_records, batch_bytes = _normalized_jsonl_stats(batch_path)
        actual_records += batch_records
        output_bytes += batch_bytes
        if actual_records > settings.sync_pending_max_records_per_task:
            raise SyncResourceLimitExceeded(
                "External sync pending record quota exceeded"
            )
        if output_bytes > settings.sync_generation_max_input_bytes:
            raise SyncResourceLimitExceeded(
                "External sync generation input byte limit exceeded"
            )

    merge_dir = _managed_directory(user_component, task_id, "merged", create=True)
    merged_file = merge_dir / f"merged_{attempt_id}.jsonl"

    total_lines = 0
    created = False
    try:
        with _reserve_sync_storage(user_id, merged_file, output_bytes):
            with open(merged_file, "xb") as out_f:
                created = True
                for batch_path in source_paths:
                    for normalized in _iter_normalized_jsonl_bytes(batch_path):
                        out_f.write(normalized + b"\n")
                        total_lines += 1
    except Exception:
        if created:
            try:
                os.remove(merged_file)
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception(
                    "[sync:%s] Failed to delete partial merge file %s",
                    task_id[:8],
                    merged_file,
                )
        raise

    logger.info(f"[sync:{task_id[:8]}] Merged {len(batches)} batches → {merged_file} ({total_lines} records)")
    return str(merged_file)


def _restore_stopped_queued_batches(task_id: str) -> int:
    """Reconcile terminal generation tasks before attempting a new claim.

    User-stopped generation tasks should not auto-retry in the next cycle, but
    these queued batches should be recoverable for a future manual trigger.
    """
    from ..storage.services.external_sync_service import external_sync_service
    from ..storage.services.generation_task_service import generation_task_service
    from ..storage.entities.generation_task_entity import GenerationStatus
    from ..api.routes.generation_routes import _finalize_sync_generation_tracking

    restored = 0
    limit = 200
    offset = 0
    generation_snapshots: List[Dict[str, Any]] = []

    # Snapshot tracking rows first, then mutate statuses.
    # Mutating during offset pagination can skip remaining rows.
    while True:
        generations, total = external_sync_service.list_generations(
            task_id=task_id,
            status=SyncGenerationStatus.PENDING,
            limit=limit,
            offset=offset,
        )
        if not generations:
            break

        generation_snapshots.extend(generations)
        offset += limit
        if offset >= total:
            break

    generation_task_ids = list(
        dict.fromkeys(
            batch.get("generation_task_id")
            for batch in generation_snapshots
            if batch.get("generation_task_id")
        )
    )
    for generation_task_id in generation_task_ids:
        if not generation_task_id:
            continue

        gen_task = generation_task_service.get_task(generation_task_id)
        if gen_task and gen_task.get("status") == GenerationStatus.COMPLETED:
            _finalize_sync_generation_tracking(
                generation_task_id,
                generation_mode=gen_task.get("generation_mode", ""),
                output_dataset_id=gen_task.get("output_dataset_id"),
                output_sample_count=gen_task.get("output_sample_count", 0),
                qa_dataset_id=gen_task.get("qa_dataset_id"),
                deep_eval_dataset_id=gen_task.get("deep_eval_dataset_id"),
                user_id=gen_task.get("user_id"),
            )
        elif gen_task is None or gen_task.get("status") in {
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        }:
            if gen_task is None:
                reason = "Generation task missing; batches restored for retry"
            elif gen_task.get("status") == GenerationStatus.STOPPED:
                reason = "Stopped generation batches restored for retry"
            else:
                reason = "Failed generation batches restored for retry"
            recovery = external_sync_service.fail_generation_and_restore_batches(
                generation_task_id, reason
            )
            if recovery.get("recovered"):
                restored += int(recovery.get("restored_batch_count") or 0)

    return restored




async def _trigger_generation(config: Dict[str, Any]):
    """Level 1: Create a generation task from accumulated batches."""
    from ..storage.services.external_sync_service import external_sync_service
    from ..storage.services.dataset_service import dataset_service
    from ..storage.services.dataset_lineage_service import dataset_lineage_service
    from ..storage.services.dataset_asset_service import dataset_asset_service
    from ..storage.services.generation_task_service import generation_task_service
    from ..storage.services.milvus_collection_service import (
        milvus_collection_service,
    )
    from ..storage.entities.generation_task_entity import GenerationStatus
    from ..storage.services.model_config_service import model_config_service
    from ..generation.pipeline import (
        PipelineConfig,
        resolve_generation_attempt_output_path,
        validate_generation_resource_config,
    )
    from ..storage.services.background_task_admission_service import (
        BackgroundTaskAlreadyExecuting,
        background_task_admission_service,
    )

    task_id = config["task_id"]
    tag = task_id[:8]

    # Recover batches from user-stopped generation tasks first, then
    # uniformly read current FETCHED batches. This avoids leaving old
    # queued batches stranded when new fetched batches already exist.
    restored = _restore_stopped_queued_batches(task_id)
    if restored > 0:
        logger.info(
            f"[sync:{tag}] Restored {restored} queued batches from STOPPED generation tasks"
        )

    if external_sync_service.has_pending_generation(task_id):
        raise BackgroundTaskAlreadyExecuting(
            "A generation is already pending for this sync task"
        )
    external_sync_service.update_task(task_id, status=SyncStatus.GENERATING)

    # Collect pending batches
    pending_batches = external_sync_service.get_pending_batches(task_id)

    if not pending_batches:
        if external_sync_service.has_pending_generation(task_id):
            raise BackgroundTaskAlreadyExecuting(
                "A generation is already pending for this sync task"
            )
        # Check if all generation outputs are disabled — if so, reset batches
        # so the user can regenerate from scratch with updated config.
        if not external_sync_service.has_enabled_completed_generation(task_id):
            reset_count = external_sync_service.reset_completed_batches(task_id)
            if reset_count > 0:
                logger.info(f"[sync:{tag}] All generation outputs disabled, reset {reset_count} batches for regeneration")
                pending_batches = external_sync_service.get_pending_batches(task_id)

        if not pending_batches:
            if not external_sync_service.reset_generating_task_if_no_pending_generation(
                task_id
            ):
                raise BackgroundTaskAlreadyExecuting(
                    "A generation is already pending for this sync task"
                )
            return

    pending_batches = _select_generation_batches(config, pending_batches)

    # Avoid expensive merge/registration work when capacity is already full.
    # The later admit_execution call remains authoritative against races.
    background_task_admission_service.assert_capacity_available(config.get("user_id"))

    # Resolve and validate settings that do not depend on staged data before
    # creating any files or database records.
    gen_cfg = config.get("generation_config")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
    generation_mode = config.get("generation_mode", "doc_to_training")

    llm_config_data = _resolve_config_from_gen_cfg(
        gen_cfg,
        "llm_config",
        "llm",
        model_config_service,
        expected_user_id=config.get("user_id"),
    )
    embedding_config_data = _resolve_config_from_gen_cfg(
        gen_cfg,
        "embedding_config",
        "embedding",
        model_config_service,
        expected_user_id=config.get("user_id"),
    )
    rerank_config_data = _resolve_config_from_gen_cfg(
        gen_cfg,
        "rerank_config",
        "rerank",
        model_config_service,
        expected_user_id=config.get("user_id"),
    )

    steps_config = gen_cfg.get("steps_config", {})
    worker_config = gen_cfg.get(
        "worker_config",
        {"concurrency": 10, "timeout_per_doc": 300},
    )
    post_process_config = gen_cfg.get("post_process_config")
    output_format = gen_cfg.get("output_format", "universal")
    pos_neg_method = gen_cfg.get("pos_neg_method", "retrieval")

    embedding_config_id = None
    if embedding_config_data and gen_cfg.get("embedding_config", {}).get("config_id"):
        embedding_config_id = gen_cfg["embedding_config"]["config_id"]

    generation_output_dir = os.environ.get(
        "GENERATION_OUTPUT_DIR",
        "/app/data/datasets",
    )
    os.makedirs(generation_output_dir, exist_ok=True)

    attempt_id = uuid.uuid4().hex
    planned_gen_task_id = str(uuid.uuid4())
    gen_task_id = planned_gen_task_id
    raw_dataset_id = str(uuid.uuid4())
    task_component = _require_path_component(task_id, "Task ID")
    user_component = _sync_storage_user_component(config["user_id"])
    merged_path = str(
        Path(os.path.abspath(SYNC_DATA_DIR))
        / user_component
        / task_component
        / "merged"
        / f"merged_{attempt_id}.jsonl"
    )
    execution_lease = None
    failure_reason = "Sync generation could not be scheduled"
    tracking_attempted = False
    generation_task_attempted = False
    generation_run_token = None

    def rollback_generation_setup(*, cleanup_staging: bool) -> None:
        nonlocal generation_run_token
        if execution_lease is not None:
            try:
                execution_lease.release()
            except Exception:
                logger.exception(
                    "[sync:%s] Failed to release generation execution lease",
                    tag,
                )

        setup_reconciled = True
        if tracking_attempted:
            try:
                recovery = external_sync_service.fail_generation_and_restore_batches(
                    gen_task_id,
                    failure_reason,
                )
                setup_reconciled = bool(
                    recovery
                    and (
                        recovery.get("recovered")
                        or recovery.get("already_recovered")
                        or recovery.get("reconciled")
                        or not recovery.get("tracking_found")
                    )
                )
            except Exception:
                setup_reconciled = False
                logger.exception(
                    "[sync:%s] Failed to restore sync state for generation %s",
                    tag,
                    gen_task_id,
                )

        cleanup_succeeded = True
        if cleanup_staging and setup_reconciled:
            if raw_dataset_id:
                staged_dataset = dataset_service.get_dataset(raw_dataset_id)
                deletion_owner = f"generation:{gen_task_id}"
                try:
                    if staged_dataset:
                        if not dataset_service.mark_deleting(
                            raw_dataset_id,
                            deletion_owner=deletion_owner,
                            user_id=config.get("user_id"),
                        ):
                            raise RuntimeError(
                                "Staged dataset disappeared before deletion"
                            )
                    dataset_lineage_service.delete_edges_for_dataset(
                        raw_dataset_id
                    )
                    dataset_asset_service.delete_assets_for_dataset(
                        raw_dataset_id
                    )
                    if staged_dataset:
                        if not dataset_service.delete_dataset(
                            raw_dataset_id,
                            deletion_owner=deletion_owner,
                            user_id=config.get("user_id"),
                        ):
                            raise RuntimeError(
                                "Staged dataset disappeared during deletion"
                            )
                except Exception:
                    cleanup_succeeded = False
                    logger.exception(
                        "[sync:%s] Failed to delete staged dataset %s",
                        tag,
                        raw_dataset_id,
                    )

            if merged_path and cleanup_succeeded:
                try:
                    _delete_managed_merged_file(
                        task_component,
                        user_component,
                        merged_path,
                    )
                except Exception:
                    cleanup_succeeded = False
                    logger.exception(
                        "[sync:%s] Failed to delete staged merge file %s",
                        tag,
                        merged_path,
                    )

        if generation_task_attempted and not generation_run_token:
            try:
                raw_generation_task = generation_task_service.get_task_raw(
                    gen_task_id
                )
                generation_run_token = (
                    raw_generation_task.get("run_token")
                    if raw_generation_task
                    else None
                )
            except Exception:
                logger.exception(
                    "[sync:%s] Failed to recover generation attempt token for %s",
                    tag,
                    gen_task_id,
                )

        if (
            generation_task_attempted
            and generation_run_token
            and setup_reconciled
            and cleanup_succeeded
        ):
            should_finalize_failed = True
            try:
                current_generation_task = generation_task_service.get_task_raw(
                    gen_task_id
                )
                if current_generation_task and (
                    current_generation_task.get("status")
                    != GenerationStatus.PENDING
                    or current_generation_task.get("run_token")
                    != generation_run_token
                ):
                    should_finalize_failed = False
            except Exception:
                logger.exception(
                    "[sync:%s] Failed to recheck generation ownership for %s",
                    tag,
                    gen_task_id,
                )
            if not should_finalize_failed:
                return
            try:
                generation_task_service.update_status(
                    gen_task_id,
                    GenerationStatus.FAILED,
                    failure_reason,
                    expected_run_token=generation_run_token,
                )
            except Exception:
                logger.exception(
                    "[sync:%s] Failed to finalize unscheduled generation task %s",
                    tag,
                    gen_task_id,
                )

    total_records = sum(b["record_count"] for b in pending_batches)
    output_path = os.path.join(
        generation_output_dir,
        f"generated_{gen_task_id}.jsonl",
    )
    _llm_cfg = llm_config_data or {}
    pipeline_config = PipelineConfig(
        input_path=merged_path,
        input_format="jsonl",
        content_field="content",
        generation_mode=generation_mode,
        pos_neg_method=pos_neg_method,
        output_path=output_path,
        output_format=output_format,
        llm_config=_llm_cfg,
        llm_concurrency=_llm_cfg.get("concurrency", 10),
        embedding_concurrency=(
            embedding_config_data.get("concurrency", 20)
            if embedding_config_data
            else 20
        ),
        timeout_per_doc=worker_config.get("timeout_per_doc", 300),
        steps=steps_config,
        post_process=post_process_config,
        embedding_config=embedding_config_data,
        rerank_config=rerank_config_data,
        source_dataset_id=raw_dataset_id,
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
        task_id=gen_task_id,
        existing_collection_name=config.get("milvus_collection_name"),
        user_id=config.get("user_id"),
    )
    validate_generation_resource_config(pipeline_config)

    def create_admitted_generation_task(**kwargs):
        nonlocal generation_task_attempted
        generation_task_attempted = True
        created_task = generation_task_service.create_task(
            task_id=planned_gen_task_id,
            **kwargs,
        )
        if (
            not isinstance(created_task, dict)
            or created_task.get("task_id") != planned_gen_task_id
        ):
            raise RuntimeError("Generation task was created with an unexpected ID")
        return created_task

    try:
        existing_collection_name = config.get("milvus_collection_name")
        if existing_collection_name:
            milvus_collection_service.register_collection(
                collection_name=existing_collection_name,
                embedding_config_id=embedding_config_id,
                embedding_model=(embedding_config_data or {}).get("model"),
                embedding_endpoint=(embedding_config_data or {}).get("endpoint"),
                dim=int(
                    (embedding_config_data or {}).get("dimension")
                    or (embedding_config_data or {}).get("dim")
                    or 1024
                ),
                user_id=config.get("user_id"),
                sync_task_id=task_id,
            )
        batch_ids = [b["batch_id"] for b in pending_batches]
        tracking_attempted = True
        external_sync_service.create_generation_and_claim_batches(
            task_id=task_id,
            generation_task_id=gen_task_id,
            user_id=config["user_id"],
            input_batch_ids=batch_ids,
            input_record_count=total_records,
            dataset_id=raw_dataset_id,
        )

        gen_task, execution_lease = background_task_admission_service.admit_execution(
            "generation",
            None,
            config["user_id"],
            create_admitted_generation_task,
            task_name=f"sync-{tag}-gen-{_utcnow_naive():%Y%m%d%H%M}",
            input_path=merged_path,
            input_format="jsonl",
            output_format=output_format,
            generation_mode=generation_mode,
            pos_neg_method=pos_neg_method,
            llm_config=llm_config_data or {},
            embedding_config=embedding_config_data,
            rerank_config=rerank_config_data,
            worker_config=worker_config,
            steps_config=steps_config,
            post_process_config=post_process_config,
            content_field="content",
            auto_register_dataset=True,
            user_id=config["user_id"],
            source_dataset_id=raw_dataset_id,
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
        )
        if gen_task.get("task_id") != gen_task_id:
            raise RuntimeError("Generation admission returned an unexpected task ID")
        raw_gen_task = generation_task_service.get_task_raw(gen_task_id)
        generation_run_token = (
            raw_gen_task.get("run_token") if raw_gen_task else None
        )
        if not generation_run_token:
            raise RuntimeError("Generation task was created without an attempt token")
        output_path = str(
            resolve_generation_attempt_output_path(
                output_path,
                gen_task_id,
                generation_run_token,
            )
        )
        pipeline_config.run_token = generation_run_token
        pipeline_config.output_path = output_path
        if not generation_task_service.set_output(
            gen_task_id,
            output_path,
            0,
            expected_status=GenerationStatus.PENDING,
            expected_run_token=generation_run_token,
        ):
            raise RuntimeError("Generation task lost ownership during sync handoff")

        merge_config = dict(config)
        merge_config.update(
            _generation_attempt_id=attempt_id,
            _generation_merged_path=merged_path,
        )
        actual_merged_path = _merge_batches(merge_config, pending_batches)
        if os.path.abspath(actual_merged_path) != os.path.abspath(merged_path):
            raise RuntimeError("Generation merge returned an unexpected path")

        ds_name = f"sync-{task_id}-raw-{attempt_id}"
        raw_dataset = dataset_service.create_dataset(
            dataset_id=raw_dataset_id,
            dataset_name=ds_name,
            storage_path=merged_path,
            dataset_type="custom",
            usage="raw",
            source_type="generated",
            file_format="jsonl",
            num_rows=total_records,
            user_id=config["user_id"],
            source_task_type="sync",
            source_task_id=task_id,
            extra_metadata={"sync_task_id": task_id, "content_field": "content"},
        )
        if raw_dataset.get("dataset_id") != raw_dataset_id:
            raise RuntimeError("Staged dataset was created with an unexpected ID")
        dataset_lineage_service.create_edge(
            from_dataset_id=None,
            to_dataset_id=raw_dataset_id,
            relation_type="sync_fetched",
            op_task_type="sync",
            op_task_id=task_id,
            op_params={"record_count": total_records},
        )
        byte_size = (
            os.path.getsize(merged_path) if os.path.exists(merged_path) else None
        )
        dataset_asset_service.create_asset(
            dataset_id=raw_dataset_id,
            storage_uri=merged_path,
            asset_type="data",
            file_format="jsonl",
            row_count=total_records,
            byte_size=byte_size,
        )

        handoff_task = generation_task_service.get_task_raw(gen_task_id)
        if not (
            handoff_task
            and handoff_task.get("status") == GenerationStatus.PENDING
            and handoff_task.get("run_token") == generation_run_token
        ):
            raise RuntimeError(
                "Generation task lost ownership during sync staging"
            )

        # The caller runs the coroutine on the main event loop; this function's
        # thread-local loop is closed immediately after returning.
        logger.info(f"[sync:{tag}] Generation task created: {gen_task_id}")
        return {
            "gen_task_id": gen_task_id,
            "generation_run_token": generation_run_token,
            "pipeline_config": pipeline_config,
            "execution_lease": execution_lease,
            "sync_task_id": task_id,
            "user_id": config["user_id"],
            "raw_dataset_id": raw_dataset_id,
            "merged_path": merged_path,
        }
    except Exception:
        rollback_generation_setup(cleanup_staging=True)
        raise


def _resolve_config_from_gen_cfg(
    gen_cfg: Dict[str, Any],
    cfg_key: str,
    model_type: str,
    model_config_service,
    *,
    expected_user_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Resolve a model config from generation_config.

    Supports both config_id reference and direct endpoint/model specification.
    """
    sub_cfg = gen_cfg.get(cfg_key)
    if not sub_cfg:
        return None

    if isinstance(sub_cfg, dict) and sub_cfg.get("config_id"):
        config_id = str(sub_cfg["config_id"]).strip()
        config = model_config_service.get_config(config_id)
        if not config:
            raise ValueError(f"Model config {config_id} not found for {cfg_key}")
        if not expected_user_id or config.get("user_id") != expected_user_id:
            raise PermissionError(
                f"Model config {config_id} is not owned by sync task user"
            )

        endpoint = config.get("api_endpoint")
        if not endpoint:
            raise ValueError(f"Model config {config_id} has no API endpoint")
        endpoint = validate_user_outbound_url(endpoint, expected_user_id)

        result = {
            "config_id": config_id,
            "endpoint": endpoint,
            "model": config.get("model_name"),
            "api_key": config.get("api_key"),
        }
        # Copy extra params from sub_cfg
        for k in ("temperature", "max_tokens", "timeout", "max_retries", "concurrency",
                   "batch_size", "similarity_threshold", "retrieval_top_k",
                   "top_k", "rerank_threshold"):
            if k in sub_cfg:
                result[k] = sub_cfg[k]
        return result

    # Direct config (endpoint + model already specified)
    if isinstance(sub_cfg, dict) and (sub_cfg.get("endpoint") or sub_cfg.get("model")):
        result = dict(sub_cfg)
        if result.get("endpoint"):
            result["endpoint"] = validate_user_outbound_url(
                result["endpoint"],
                expected_user_id,
            )
        return result

    return None
