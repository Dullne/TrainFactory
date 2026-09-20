"""
Sync manager: background loop that manages per-task sync workers.

Started/stopped in FastAPI lifespan. Each active sync task gets
its own asyncio task that periodically calls sync_worker.run_once().

Worker cycles run in a dedicated ThreadPoolExecutor to avoid blocking
uvicorn's event loop with synchronous DB/IO/Milvus operations.
"""

import asyncio
import concurrent.futures
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional

from ..enums.sync_status import SyncStatus

logger = logging.getLogger(__name__)

_MAX_SYNC_WORKERS = int(os.environ.get("SYNC_MAX_WORKERS", "4"))


class SyncTaskBusyError(Exception):
    """Raised when a manual sync-now/trigger hits a task already generating/training."""


class SyncGenerationReconciliationError(RuntimeError):
    """Keep sync state when generation ownership cannot be proven safely."""


def _reconcile_terminal_sync_generation(task_id: str) -> int:
    """Reconcile terminal generation tracking during normal worker operation."""
    from .sync_worker import _restore_stopped_queued_batches

    return _restore_stopped_queued_batches(task_id)


def _reconcile_sync_generation_state(
    external_sync_service,
    task_id: str,
) -> Optional[dict]:
    """Reconcile tracking, atomically reset an orphan, then read final state."""
    try:
        _reconcile_terminal_sync_generation(task_id)
        external_sync_service.reset_generating_task_if_no_pending_generation(task_id)
        return external_sync_service.get_task_raw(task_id)
    except Exception as exc:
        logger.warning(
            "[sync:%s] Generation reconciliation failed (%s)",
            task_id[:8],
            type(exc).__name__,
        )
        raise SyncGenerationReconciliationError(
            "Sync generation state could not be reconciled"
        ) from None


def _run_sync_cycle(config: dict) -> Optional[dict]:
    """Run one sync cycle in a dedicated thread with its own event loop.

    This isolates all synchronous DB/IO/Milvus calls from uvicorn's
    event loop, preventing sync workers from blocking HTTP requests.

    Returns:
        Optional dict with ``gen_task_id`` and ``pipeline_config`` when
        a generation task was created and needs to be launched on the main
        event loop.  ``None`` otherwise.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from . import sync_worker

        return loop.run_until_complete(sync_worker.run_once(config))
    finally:
        loop.close()


def _run_generation_cycle(config: dict) -> Optional[dict]:
    """Run generation trigger in a dedicated thread with its own event loop.

    Returns generation task info (same as _run_sync_cycle) or None.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from . import sync_worker

        return loop.run_until_complete(sync_worker._trigger_generation(config))
    finally:
        loop.close()


class SyncManager:
    """Manages background sync workers for all active tasks."""

    def __init__(self):
        self._workers: Dict[str, asyncio.Task] = {}
        self._running = False
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Per-task locks serialize concurrent sync cycles (periodic worker vs
        # sync-now API) for the same task so two cycles can't interleave and
        # double-count the same records.
        self._task_locks: Dict[str, asyncio.Lock] = {}
        self._manual_admission_lock = threading.Lock()
        self._manual_task_ids: set[str] = set()

    def _get_task_lock(self, config_id: str) -> asyncio.Lock:
        lock = self._task_locks.get(config_id)
        if lock is None:
            lock = asyncio.Lock()
            self._task_locks[config_id] = lock
        return lock

    def _claim_manual_task(self, config_id: str) -> None:
        with self._manual_admission_lock:
            if config_id in self._manual_task_ids:
                raise SyncTaskBusyError(
                    f"Task {config_id} already has a manual operation in progress"
                )
            self._manual_task_ids.add(config_id)

    def _release_manual_task(self, config_id: str) -> None:
        with self._manual_admission_lock:
            self._manual_task_ids.discard(config_id)

    async def _refresh_busy_state(
        self,
        loop: asyncio.AbstractEventLoop,
        external_sync_service,
        task_id: str,
        config: dict,
    ) -> Optional[dict]:
        """Refresh generating state while the caller holds the per-task lock."""
        if config.get("status") != SyncStatus.GENERATING:
            return config
        future = loop.run_in_executor(
            self._executor,
            _reconcile_sync_generation_state,
            external_sync_service,
            task_id,
        )
        try:
            refreshed = await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(future)
            except Exception:
                pass
            raise
        if refreshed:
            _require_not_deleting(refreshed)
        return refreshed

    @asynccontextmanager
    async def task_operation_lock(self, config_id: str):
        """Serialize external lifecycle operations with every task worker path."""
        async with self._get_task_lock(config_id):
            yield

    async def _run_cycle_with_handoff(
        self,
        loop: asyncio.AbstractEventLoop,
        operation,
        config: dict,
    ) -> Optional[dict]:
        """Finish an executor cycle after cancellation and hand off any job."""
        future = loop.run_in_executor(self._executor, operation, config)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                result = await asyncio.shield(future)
                if result and "gen_task_id" in result:
                    self._finalize_unlaunched_generation(
                        result,
                        "Sync cycle was cancelled before generation handoff",
                    )
            except Exception:
                logger.exception("Cancelled sync cycle failed after cancellation")
            raise

    @property
    def running(self) -> bool:
        return self._running

    async def start(self):
        """Start sync manager: load active tasks and create workers."""
        from ..storage.services.external_sync_service import external_sync_service

        self._running = True
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_SYNC_WORKERS, thread_name_prefix="sync-worker"
        )
        try:
            configs = external_sync_service.list_active_tasks()
            for cfg in configs:
                self._start_worker(cfg["task_id"])
            logger.info(f"SyncManager started with {len(configs)} active task(s)")
        except Exception as e:
            logger.warning(f"SyncManager start failed: {e}")

    async def stop(self):
        """Stop all workers gracefully."""
        self._running = False
        tasks = list(self._workers.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("SyncManager stopped")

    def start_worker(self, config_id: str):
        """Start a worker for a specific config (called when config is activated)."""
        if not self._running:
            logger.warning("SyncManager not running, cannot start worker")
            return
        self._start_worker(config_id)

    async def stop_worker(self, config_id: str):
        """Stop a worker and wait until its in-flight cycle is quiescent."""
        task = self._workers.get(config_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            logger.info(f"[sync:{config_id[:8]}] Worker stopped")
        if self._workers.get(config_id) is task:
            self._workers.pop(config_id, None)

    async def restart_worker(self, config_id: str):
        """Restart a worker after its previous cycle is quiescent."""
        await self.stop_worker(config_id)
        self._start_worker(config_id)

    def get_worker_status(self, config_id: str) -> str:
        """Get worker status: running, stopped, or not_found."""
        task = self._workers.get(config_id)
        if task is None:
            return "not_found"
        if task.done():
            return "stopped"
        return "running"

    def _start_worker(self, config_id: str):
        """Internal: create asyncio task for a config."""
        if config_id in self._workers and not self._workers[config_id].done():
            return  # Already running
        self._workers[config_id] = asyncio.create_task(
            self._run_worker(config_id),
            name=f"sync-worker-{config_id[:8]}",
        )
        logger.info(f"[sync:{config_id[:8]}] Worker started")

    async def _run_worker(self, config_id: str):
        """Worker loop: periodically run sync for one task.

        Each cycle runs in a thread pool thread with its own event loop,
        so synchronous calls (DB, file IO, Milvus) never block uvicorn.
        """
        from ..storage.services.external_sync_service import external_sync_service

        tag = config_id[:8]
        loop = asyncio.get_event_loop()

        while self._running:
            interval = 300  # default
            try:
                config = await loop.run_in_executor(
                    self._executor, external_sync_service.get_task_raw, config_id
                )
                if not config or not config.get("is_active"):
                    logger.info(f"[sync:{tag}] Task inactive or deleted, stopping worker")
                    break

                interval = config.get("sync_interval_seconds", 300)

                gen_result = None

                async with self._get_task_lock(config_id):
                    # Refresh after acquiring the lock. Another periodic/manual
                    # cycle may have advanced the cursor while this worker waited.
                    config = await loop.run_in_executor(
                        self._executor,
                        external_sync_service.get_task_raw,
                        config_id,
                    )
                    if not config or not config.get("is_active"):
                        logger.info(f"[sync:{tag}] Task inactive or deleted, stopping worker")
                        break
                    interval = config.get("sync_interval_seconds", 300)

                    config = await self._refresh_busy_state(
                        loop,
                        external_sync_service,
                        config_id,
                        config,
                    )
                    if not config or not config.get("is_active"):
                        logger.info(f"[sync:{tag}] Task inactive or deleted, stopping worker")
                        break
                    status = config.get("status", SyncStatus.IDLE)
                    if status in (
                        SyncStatus.DELETING,
                        SyncStatus.DELETING_CASCADE,
                    ):
                        logger.info(
                            "[sync:%s] Task is being deleted, stopping worker",
                            tag,
                        )
                        break
                    if status in (
                        SyncStatus.GENERATING,
                        SyncStatus.TRAINING,
                        SyncStatus.LOADING_ADAPTER,
                    ):
                        logger.debug(
                            f"[sync:{tag}] Status is '{status}', skipping this cycle"
                        )
                    else:
                        run_cycle = True
                        if status == SyncStatus.ERROR:
                            logger.info(f"[sync:{tag}] Recovering from error state")
                            recovered = await loop.run_in_executor(
                                self._executor,
                                _update_task_sync,
                                external_sync_service,
                                config_id,
                            )
                            if not recovered:
                                run_cycle = False
                                logger.warning(
                                    "[sync:%s] Preserving error state while generation "
                                    "recovery is unresolved",
                                    tag,
                                )
                            else:
                                config = await loop.run_in_executor(
                                    self._executor,
                                    external_sync_service.get_task_raw,
                                    config_id,
                                )
                                if not config:
                                    break
                        if run_cycle:
                            gen_result = await self._run_cycle_with_handoff(
                                loop,
                                _run_sync_cycle,
                                config,
                            )

                # Launch generation pipeline on the MAIN event loop.
                # The sync cycle (in the thread) only creates the DB record
                # and returns the pipeline config; the actual pipeline must
                # run here so its async coroutines (httpx, etc.) work correctly.
                if gen_result and "gen_task_id" in gen_result:
                    self._launch_generation(gen_result)

            except asyncio.CancelledError:
                logger.info(f"[sync:{tag}] Worker cancelled")
                break
            except SyncGenerationReconciliationError:
                logger.warning(
                    "[sync:%s] Generation reconciliation unavailable; retrying next cycle",
                    tag,
                )
            except Exception as e:
                logger.exception(f"[sync:{tag}] Worker error: {e}")
                try:
                    error_msg = str(e)
                    await loop.run_in_executor(
                        self._executor,
                        _set_task_error,
                        external_sync_service,
                        config_id,
                        error_msg,
                    )
                except Exception:
                    pass

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

        # Cleanup
        current_task = asyncio.current_task()
        if self._workers.get(config_id) is current_task:
            self._workers.pop(config_id, None)
        logger.info(f"[sync:{tag}] Worker exited")

    async def run_once(self, config_id: str):
        """Manually trigger one sync cycle with immediate per-task admission."""
        self._claim_manual_task(config_id)
        try:
            await self._run_once_claimed(config_id)
        finally:
            self._release_manual_task(config_id)

    async def _run_once_claimed(self, config_id: str):
        """Manually trigger one sync cycle (bypass interval wait)."""
        from ..storage.services.external_sync_service import external_sync_service

        loop = asyncio.get_event_loop()
        try:
            async with self._get_task_lock(config_id):
                config = await loop.run_in_executor(
                    self._executor,
                    external_sync_service.get_task_raw,
                    config_id,
                )
                if not config:
                    raise ValueError(f"Task {config_id} not found")
                config = await self._refresh_busy_state(
                    loop,
                    external_sync_service,
                    config_id,
                    config,
                )
                if not config:
                    raise ValueError(f"Task {config_id} not found")
                _require_not_deleting(config)
                if config.get("status") in (
                    SyncStatus.GENERATING,
                    SyncStatus.TRAINING,
                    SyncStatus.LOADING_ADAPTER,
                ):
                    # 与周期 worker 的跳过逻辑对齐：手动触发遇到忙状态返回
                    # 明确的"任务忙"（409），而不是让状态机校验抛 ValueError -> 500
                    raise SyncTaskBusyError(
                        f"Task {config_id} is busy (status={config.get('status')})"
                    )
                gen_result = await self._run_cycle_with_handoff(
                    loop,
                    _run_sync_cycle,
                    config,
                )
        except (SyncTaskBusyError, SyncGenerationReconciliationError):
            raise
        except Exception:
            # Recover stuck GENERATING status — _run_worker has its own
            # error handler but run_once (sync-now API) does not.
            try:
                latest = external_sync_service.get_task_raw(config_id)
                if latest and latest.get("status") == SyncStatus.GENERATING:
                    _set_task_error(
                        external_sync_service,
                        config_id,
                        "sync-now cycle failed",
                    )
            except Exception:
                pass
            raise
        if gen_result and "gen_task_id" in gen_result:
            self._launch_generation(gen_result)

    def _launch_generation(self, gen_result: dict):
        """Schedule a generation worker from the main event loop.

        Called from _run_worker (which is already on the main loop) after
        _run_sync_cycle returns generation task info.
        """
        from ..api.routes.generation_routes import _run_generation_task
        from ..storage.services.background_task_admission_service import (
            background_task_admission_service,
        )
        from ..storage.entities.generation_task_entity import GenerationStatus
        from ..storage.services.generation_task_service import (
            generation_task_service,
        )

        gen_task_id = gen_result["gen_task_id"]
        generation_run_token = gen_result.get("generation_run_token")
        if not generation_run_token:
            self._finalize_unlaunched_generation(
                gen_result,
                "Sync generation handoff was missing its attempt token",
            )
            raise RuntimeError(
                "Sync generation result is missing its generation run token"
            )
        pipeline_config = gen_result["pipeline_config"]
        execution_lease = gen_result.get("execution_lease")
        if execution_lease is None:
            self._finalize_unlaunched_generation(
                gen_result,
                "Sync generation handoff was missing its execution lease",
            )
            raise RuntimeError(
                "Sync generation result is missing its background execution lease"
            )
        current_task = generation_task_service.get_task_raw(gen_task_id)
        if not (
            current_task
            and current_task.get("status") == GenerationStatus.PENDING
            and current_task.get("run_token") == generation_run_token
        ):
            self._finalize_unlaunched_generation(
                gen_result,
                "Sync generation stopped before manager launch",
            )
            return
        logger.info(f"Launching generation pipeline {gen_task_id[:8]} on a worker event loop")
        coroutine = background_task_admission_service.run_async(
            execution_lease,
            _run_generation_task,
            gen_task_id,
            pipeline_config,
            expected_run_token=generation_run_token,
            sync_handoff=gen_result,
        )
        try:
            asyncio.create_task(
                coroutine,
                name=f"gen-pipeline-{gen_task_id[:8]}",
            )
        except Exception:
            coroutine.close()
            self._finalize_unlaunched_generation(
                gen_result,
                "Sync generation could not be launched on the main event loop",
            )
            raise

    @staticmethod
    def _finalize_unlaunched_generation(gen_result: dict, reason: str) -> None:
        """Release a handoff and make its persisted task retryable."""
        from ..storage.entities.generation_task_entity import GenerationStatus
        from ..storage.services.external_sync_service import external_sync_service
        from ..storage.services.generation_task_service import generation_task_service

        execution_lease = gen_result.get("execution_lease")
        if execution_lease is not None:
            execution_lease.release()
        gen_task_id = gen_result.get("gen_task_id")
        if not gen_task_id:
            return
        generation_run_token = gen_result.get("generation_run_token")
        if not generation_run_token:
            raw_task = generation_task_service.get_task_raw(gen_task_id)
            generation_run_token = (
                raw_task.get("run_token") if raw_task else None
            )
        if not generation_run_token:
            logger.warning(
                "Cannot fence unlaunched generation cleanup for %s",
                gen_task_id,
            )
            return

        try:
            recovery = external_sync_service.fail_generation_and_restore_batches(
                gen_task_id,
                reason,
            )
        except Exception:
            logger.exception(
                "Failed to restore sync state for unlaunched generation %s",
                gen_task_id,
            )
            return
        if not (
            recovery.get("recovered")
            or recovery.get("already_recovered")
            or recovery.get("reconciled")
        ):
            logger.warning(
                "Could not restore sync state for unlaunched generation %s",
                gen_task_id,
            )
            return

        try:
            SyncManager._cleanup_unlaunched_generation_staging(gen_result)
        except Exception:
            logger.exception(
                "Failed to clean staging for unlaunched generation %s",
                gen_task_id,
            )
            return

        try:
            generation_task_service.update_status(
                gen_task_id,
                GenerationStatus.FAILED,
                reason,
                expected_run_token=generation_run_token,
            )
        except Exception:
            logger.exception(
                "Failed to finalize unlaunched generation task %s",
                gen_task_id,
            )

    @staticmethod
    def _cleanup_unlaunched_generation_staging(gen_result: dict) -> None:
        """Remove sync staging after its claimed batches were restored."""
        from ..storage.services.dataset_asset_service import dataset_asset_service
        from ..storage.services.dataset_lineage_service import dataset_lineage_service
        from ..storage.services.dataset_service import dataset_service
        from .sync_worker import _delete_managed_merged_file

        task_id = gen_result.get("sync_task_id")
        user_id = gen_result.get("user_id")
        raw_dataset_id = gen_result.get("raw_dataset_id")
        merged_path = gen_result.get("merged_path")
        if not (task_id and user_id):
            return

        if raw_dataset_id:
            dataset = dataset_service.get_dataset(raw_dataset_id)
            if dataset and not (
                dataset.get("user_id") == user_id
                and dataset.get("source_task_type") == "sync"
                and dataset.get("source_task_id") == task_id
            ):
                raise ValueError("Unlaunched generation staging ownership mismatch")
            if dataset:
                deletion_owner = f"generation:{gen_result['gen_task_id']}"
                if not dataset_service.mark_deleting(
                    raw_dataset_id,
                    deletion_owner=deletion_owner,
                    user_id=user_id,
                ):
                    raise RuntimeError(
                        "Unlaunched generation staging dataset disappeared"
                    )
            dataset_lineage_service.delete_edges_for_dataset(raw_dataset_id)
            dataset_asset_service.delete_assets_for_dataset(raw_dataset_id)
            if dataset:
                if not dataset_service.delete_dataset(
                    raw_dataset_id,
                    deletion_owner=deletion_owner,
                    user_id=user_id,
                ):
                    raise RuntimeError(
                        "Unlaunched generation staging dataset disappeared"
                    )

        if merged_path:
            _delete_managed_merged_file(task_id, user_id, merged_path)

    async def trigger_generation(self, config_id: str):
        """Manually trigger generation with immediate per-task admission."""
        self._claim_manual_task(config_id)
        try:
            await self._trigger_generation_claimed(config_id)
        finally:
            self._release_manual_task(config_id)

    async def _trigger_generation_claimed(self, config_id: str):
        """Manually trigger generation (ignore threshold)."""
        from ..storage.services.external_sync_service import external_sync_service

        loop = asyncio.get_event_loop()
        try:
            async with self._get_task_lock(config_id):
                config = await loop.run_in_executor(
                    self._executor,
                    external_sync_service.get_task_raw,
                    config_id,
                )
                if not config:
                    raise ValueError(f"Task {config_id} not found")
                config = await self._refresh_busy_state(
                    loop,
                    external_sync_service,
                    config_id,
                    config,
                )
                if not config:
                    raise ValueError(f"Task {config_id} not found")
                _require_not_deleting(config)
                if config.get("status") in (
                    SyncStatus.GENERATING,
                    SyncStatus.TRAINING,
                    SyncStatus.LOADING_ADAPTER,
                ):
                    raise SyncTaskBusyError(
                        f"Task {config_id} is busy (status={config.get('status')})"
                    )
                gen_result = await self._run_cycle_with_handoff(
                    loop,
                    _run_generation_cycle,
                    config,
                )
        except (SyncTaskBusyError, SyncGenerationReconciliationError):
            raise
        except Exception:
            try:
                latest = external_sync_service.get_task_raw(config_id)
                if latest and latest.get("status") == SyncStatus.GENERATING:
                    _set_task_error(
                        external_sync_service,
                        config_id,
                        "trigger-generation failed",
                    )
            except Exception:
                pass
            raise
        if gen_result and "gen_task_id" in gen_result:
            self._launch_generation(gen_result)

    async def trigger_training(self, config_id: str):
        """Manually trigger training (ignore threshold)."""
        from ..storage.services.external_sync_service import external_sync_service
        from .level2_handler import _trigger_training, _trigger_training_for_target

        loop = asyncio.get_event_loop()
        async with self._get_task_lock(config_id):
            config = await loop.run_in_executor(
                self._executor,
                external_sync_service.get_task_raw,
                config_id,
            )
            if not config:
                raise ValueError(f"Task {config_id} not found")
            _require_not_deleting(config)
            if config.get("status") in (
                SyncStatus.GENERATING,
                SyncStatus.TRAINING,
                SyncStatus.LOADING_ADAPTER,
            ):
                raise SyncTaskBusyError(
                    f"Task {config_id} is busy (status={config.get('status')})"
                )

            targets = await loop.run_in_executor(
                self._executor,
                external_sync_service.list_training_targets_raw,
                config_id,
                True,
            )
            if targets:
                targets.sort(
                    key=lambda target: (
                        target.get("priority", 0),
                        target.get("sort_order", 0),
                    )
                )
                await loop.run_in_executor(
                    self._executor,
                    _trigger_training_for_target,
                    config_id,
                    targets[0],
                )
                return
            await loop.run_in_executor(self._executor, _trigger_training, config)


def _require_not_deleting(config: dict) -> None:
    if config.get("status") in (
        SyncStatus.DELETING,
        SyncStatus.DELETING_CASCADE,
    ):
        raise RuntimeError("Sync task is being deleted")


def _update_task_sync(service, config_id: str):
    """Helper: reset task from error to idle (runs in thread pool)."""
    try:
        return bool(
            service.transition_task_if_no_pending_generation(
                config_id,
                SyncStatus.IDLE,
                error_message=None,
                expected_status=SyncStatus.ERROR,
            )
        )
    except Exception:
        logger.exception(
            "Could not atomically recover sync task %s; preserving state",
            config_id,
        )
        return False


def _set_task_error(service, config_id: str, error_msg: str):
    """Helper: set task to error state (runs in thread pool)."""
    try:
        return bool(
            service.transition_task_if_no_pending_generation(
                config_id,
                SyncStatus.ERROR,
                error_message=error_msg,
            )
        )
    except Exception:
        logger.exception(
            "Could not atomically fail sync task %s; preserving state",
            config_id,
        )
        return False


# Global singleton
sync_manager = SyncManager()
