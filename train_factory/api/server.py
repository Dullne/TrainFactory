"""
FastAPI server for TrainFactory.

Provides REST API for training management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ..auth.bootstrap import bootstrap_default_admin
from ..config import settings
from ..config.settings import validate_api_worker_count
from ..core.ssrf import SSRFError
from ..core.process_identity import terminate_process_if_matches
from ..storage import get_session, init_db
from ..storage.services.dataset_service import DatasetConsumptionUnavailableError
from ..enums import TrainingStatus
from .routes.training_routes import router as training_router
from .routes.registry_routes import router as registry_router
from .routes.deployment_routes import router as deployment_router
from .routes.model_config_routes import router as model_config_router
from .routes.dataset_routes import router as dataset_router
from .routes.resource_routes import router as resource_router
from .routes.evaluation_routes import router as evaluation_router
from .routes.auth_routes import get_rate_limit_key
from .routes.auth_routes import limiter as auth_limiter
from .routes.auth_routes import router as auth_router
from .routes.adapter_routes import router as adapter_router
from .routes.deep_evaluation_routes import router as deep_evaluation_router
from .routes.generation_routes import router as generation_router
from .routes.milvus_routes import router as milvus_router
from .routes.sync_routes import router as sync_router
from .routes.external_api_config_routes import router as ext_api_config_router

logger = logging.getLogger(__name__)

_SYNC_RECONCILIATION_PAGE_SIZE = 1000


def cleanup_old_audit_logs() -> int:
    """Apply audit retention without allowing cleanup failures to block startup."""
    try:
        from ..storage.services.audit_log_service import audit_log_service

        return audit_log_service.cleanup_old_logs(settings.audit_log_retention_days)
    except Exception as exc:
        logger.warning("Failed to clean up old audit logs: %s", exc)
        return 0


async def _audit_log_cleanup_loop() -> None:
    """Enforce audit retention for long-running API processes."""
    interval_seconds = settings.audit_log_cleanup_interval_hours * 60 * 60
    while True:
        await asyncio.to_thread(cleanup_old_audit_logs)
        await asyncio.sleep(interval_seconds)


def _collect_task_pages(get_all_tasks, status=None, page_size: int = 1000) -> list:
    """Read a stable task snapshot before any status mutations change paging."""
    tasks = []
    offset = 0
    while True:
        query = {"limit": page_size, "offset": offset}
        if status is not None:
            query["status"] = status
        page, total = get_all_tasks(**query)
        tasks.extend(page)
        offset += len(page)
        if not page or offset >= total:
            return tasks


def cleanup_orphan_tasks():
    """
    Clean up orphan running tasks on startup.

    If the server crashed while tasks were running, those tasks will still be
    marked as 'running' in the database but have no active process.
    This function marks them as 'failed' to prevent ghost running tasks.
    """
    try:
        from ..storage.services.training_task_service import training_task_service
        from ..core.gpu_resource_manager import gpu_resource_manager

        # Get all tasks marked as active or pending. These tasks no longer have a
        # live worker after restart, so they must be finalized to avoid ghost states.
        active_statuses = [
            TrainingStatus.PENDING.value,
            TrainingStatus.PREPARING.value,
            TrainingStatus.RUNNING.value,
            TrainingStatus.EVALUATING.value,
        ]

        orphan_tasks_by_id = {}
        for status in active_statuses:
            for task in _collect_task_pages(
                training_task_service.get_all_tasks,
                status,
            ):
                orphan_tasks_by_id[task["task_id"]] = task
        for task in _collect_task_pages(
            training_task_service.get_tasks_with_process_info,
        ):
            orphan_tasks_by_id[task["task_id"]] = task

        orphan_count = 0
        for task in orphan_tasks_by_id.values():
            task_id = task['task_id']
            # 清理可能仍存活的训练子进程：API 重启（SIGKILL/OOM）后 spawn
            # 子进程不随父死亡，会继续占用 GPU——SIGKILL 孤儿进程，防新任务
            # 被分配到同一张卡（同卡双训练 OOM）。
            pid = task.get("process_pid")
            process_create_time = task.get("process_create_time")
            if pid is not None and not terminate_process_if_matches(
                pid,
                process_create_time,
            ):
                logger.error(
                    "Preserving training task %s because process identity or exit "
                    "could not be confirmed (pid=%s)",
                    task_id,
                    pid,
                )
                continue

            if task.get("status") in active_statuses:
                try:
                    updated = training_task_service.update_task_status(
                        task_id,
                        TrainingStatus.FAILED.value,
                        error_message="Task interrupted by server restart",
                        run_token=task.get("run_token"),
                    )
                    if not updated:
                        logger.warning(
                            "Could not finalize orphan task %s; preserving its resources",
                            task_id,
                        )
                        continue
                    orphan_count += 1
                    logger.warning("Marked orphan task as failed: %s", task_id)
                except Exception as task_error:
                    logger.warning(
                        "Could not finalize orphan task %s (error_type=%s)",
                        task_id,
                        type(task_error).__name__,
                    )
                    continue

            try:
                cleared = training_task_service.update_process_info(
                    task_id,
                    process_pid=None,
                    process_status=None,
                    process_create_time=None,
                    run_token=task.get("run_token"),
                    expected_process_pid=pid,
                    expected_process_create_time=process_create_time,
                )
                if not cleared:
                    logger.warning(
                        "Could not clear orphan process state for %s; preserving its GPU lease",
                        task_id,
                    )
                    continue
            except Exception as process_error:
                logger.warning(
                    "Could not clear orphan process state for %s (error_type=%s)",
                    task_id,
                    type(process_error).__name__,
                )
                continue

            try:
                run_token = task.get("run_token")
                lease_id = (
                    f"training:{task_id}:{run_token}" if run_token else task_id
                )
                gpu_resource_manager.release_gpus_for_task(lease_id)
            except Exception as gpu_error:
                logger.warning(
                    "Could not release orphan task GPU %s (error_type=%s)",
                    task_id,
                    type(gpu_error).__name__,
                )

        if orphan_count > 0:
            logger.info(f"Cleaned up {orphan_count} orphan tasks on startup")

        # Also cleanup any stale GPU allocations older than 24 hours
        stale_count = gpu_resource_manager.cleanup_stale_allocations(max_age_hours=24)
        if stale_count > 0:
            logger.info(f"Cleaned up {stale_count} stale GPU allocations")

    except Exception as e:
        logger.error(f"Failed to cleanup orphan tasks: {e}")


def cleanup_orphan_evaluation_tasks():
    """
    Clean up orphan running evaluation tasks on startup.

    If the server crashed while evaluation tasks were running, those tasks will
    still be marked as 'running' in the database but have no active process.
    This function marks them as 'interrupted' so users can resume them.
    """
    try:
        from ..storage.services.evaluation_task_service import evaluation_task_service

        orphan_tasks = []
        for active_status in ("pending", "running"):
            orphan_tasks.extend(
                _collect_task_pages(
                    evaluation_task_service.get_all_tasks,
                    active_status,
                )
            )

        orphan_count = 0
        for task in orphan_tasks:
            task_id = task['task_id']
            # Mark as failed with specific message
            # Use update_status directly to go running→failed (valid transition)
            try:
                updated = evaluation_task_service.update_status(
                    task_id,
                    status="failed",
                    error_message=(
                        "Task interrupted by server restart. Use 'Resume' to continue."
                    ),
                )
                if not updated:
                    logger.warning(
                        "Could not finalize orphan evaluation task %s",
                        task_id,
                    )
                    continue
                orphan_count += 1
            except Exception as task_error:
                logger.warning(
                    "Could not finalize orphan evaluation task %s (error_type=%s)",
                    task_id,
                    type(task_error).__name__,
                )

        if orphan_count > 0:
            logger.info(f"Cleaned up {orphan_count} orphan evaluation tasks on startup")

    except Exception as e:
        logger.error(f"Failed to cleanup orphan evaluation tasks: {e}")


def cleanup_orphan_deep_evaluation_tasks():
    """
    Clean up orphan running deep evaluation tasks on startup.

    Marks running tasks as failed so users can resume them.
    """
    try:
        from ..storage.services.deep_evaluation_task_service import deep_evaluation_task_service

        orphan_tasks = []
        for active_status in ("pending", "running"):
            orphan_tasks.extend(
                _collect_task_pages(
                    deep_evaluation_task_service.get_all_tasks,
                    active_status,
                )
            )

        orphan_count = 0
        for task in orphan_tasks:
            task_id = task["task_id"]
            try:
                updated = deep_evaluation_task_service.update_status(
                    task_id,
                    status="failed",
                    error_message=(
                        "Task interrupted by server restart. Use 'Resume' to continue."
                    ),
                )
                if not updated:
                    logger.warning(
                        "Could not finalize orphan deep evaluation task %s",
                        task_id,
                    )
                    continue
                orphan_count += 1
            except Exception as task_error:
                logger.warning(
                    "Could not finalize orphan deep evaluation task %s "
                    "(error_type=%s)",
                    task_id,
                    type(task_error).__name__,
                )

        if orphan_count > 0:
            logger.info(f"Cleaned up {orphan_count} orphan deep evaluation tasks on startup")

    except Exception as e:
        logger.error(f"Failed to cleanup orphan deep evaluation tasks: {e}")


def _iter_pending_sync_generations(external_sync_service):
    """Stream all pending sync generations with bounded keyset pages."""
    after_id = None
    while True:
        page, _total = external_sync_service.list_pending_generations(
            limit=_SYNC_RECONCILIATION_PAGE_SIZE,
            after_id=after_id,
        )
        if not page:
            return
        page_ids = [row.get("id") for row in page]
        if any(not isinstance(row_id, int) for row_id in page_ids):
            raise ValueError("Pending sync generation page has invalid identifiers")
        if after_id is not None and page_ids[0] <= after_id:
            raise ValueError("Pending sync generation page did not advance")
        if page_ids != sorted(page_ids) or len(set(page_ids)) != len(page_ids):
            raise ValueError("Pending sync generation page ordering is invalid")
        yield from page
        after_id = page_ids[-1]


def _iter_pending_sync_trainings(external_sync_service):
    """Stream all pending sync trainings with bounded keyset pages."""
    after_id = None
    while True:
        page, _total = external_sync_service.list_pending_trainings(
            limit=_SYNC_RECONCILIATION_PAGE_SIZE,
            after_id=after_id,
        )
        if not page:
            return
        page_ids = [row.get("id") for row in page]
        if any(not isinstance(row_id, int) for row_id in page_ids):
            raise ValueError("Pending sync training page has invalid identifiers")
        if after_id is not None and page_ids[0] <= after_id:
            raise ValueError("Pending sync training page did not advance")
        if page_ids != sorted(page_ids) or len(set(page_ids)) != len(page_ids):
            raise ValueError("Pending sync training page ordering is invalid")
        yield from page
        after_id = page_ids[-1]


def cleanup_orphan_generation_tasks():
    """
    Clean up orphan running generation tasks on startup.

    If the server crashed while generation tasks were running, those tasks will
    still be marked as 'running' in the database but have no active process.
    This function marks them as 'stopped' so users can resume them.
    """
    try:
        from ..storage.services.generation_task_service import (
            GENERATION_RESTART_RECOVERY_MARKER,
            generation_task_service,
        )
        from ..storage.services.external_sync_service import external_sync_service
        from ..storage.entities.generation_task_entity import GenerationStatus
        from ..storage.services.generation_publication_service import (
            generation_publication_service,
        )
        from ..sync.sync_manager import SyncManager
        from .routes.generation_routes import (
            _cleanup_generation_attempt_directory,
            _finalize_sync_generation_failure_tracking,
            _finalize_sync_generation_tracking,
        )

        for tracking in _iter_pending_sync_generations(external_sync_service):
            generation_task_id = tracking.get("generation_task_id")
            if not generation_task_id:
                continue
            try:
                generation_task = generation_task_service.get_task(
                    generation_task_id
                )
                if generation_task is None:
                    _finalize_sync_generation_failure_tracking(
                        generation_task_id,
                        "Generation task missing during startup reconciliation",
                    )
                    continue
                generation_status = generation_task.get("status")
                if generation_status == GenerationStatus.COMPLETED:
                    _finalize_sync_generation_tracking(
                        generation_task_id,
                        generation_mode=generation_task.get("generation_mode", ""),
                        output_dataset_id=generation_task.get("output_dataset_id"),
                        output_sample_count=generation_task.get(
                            "output_sample_count", 0
                        ),
                        qa_dataset_id=generation_task.get("qa_dataset_id"),
                        deep_eval_dataset_id=generation_task.get(
                            "deep_eval_dataset_id"
                        ),
                        user_id=generation_task.get("user_id")
                        or tracking.get("user_id"),
                    )
                elif generation_status in {
                    GenerationStatus.FAILED,
                    GenerationStatus.STOPPED,
                }:
                    _finalize_sync_generation_failure_tracking(
                        generation_task_id,
                        "Terminal generation reconciled during server startup",
                    )
            except Exception as reconcile_error:
                logger.warning(
                    "Could not reconcile sync generation %s (error_type=%s)",
                    generation_task_id,
                    type(reconcile_error).__name__,
                )

        orphan_tasks = []
        for active_status in (
            GenerationStatus.PENDING,
            GenerationStatus.RUNNING,
            GenerationStatus.STOPPING,
            GenerationStatus.PUBLISHING,
            GenerationStatus.RECOVERING,
            GenerationStatus.RESTARTING,
        ):
            orphan_tasks.extend(
                _collect_task_pages(
                    generation_task_service.get_all_tasks,
                    active_status,
                )
            )

        orphan_count = 0
        for task in orphan_tasks:
            task_id = task.get("task_id") or task.get("id", "")
            try:
                raw_task = generation_task_service.get_task_raw(task_id)
                if raw_task is None:
                    continue
                source_status = raw_task.get("status")
                run_token = raw_task.get("run_token")
                restart_recovery = bool(
                    source_status == GenerationStatus.RESTARTING
                    or (
                        source_status == GenerationStatus.RECOVERING
                        and raw_task.get("error_message")
                        == GENERATION_RESTART_RECOVERY_MARKER
                    )
                )
                if source_status == GenerationStatus.RECOVERING:
                    if not run_token:
                        logger.warning(
                            "Orphan generation recovery %s has no owner token",
                            task_id,
                        )
                        continue
                    recovery_token = run_token
                else:
                    claim = generation_task_service.claim_orphan_recovery(
                        task_id,
                        expected_status=source_status,
                        expected_run_token=run_token,
                    )
                    if claim is None:
                        continue
                    recovery_token = claim["run_token"]
                if restart_recovery:
                    _cleanup_generation_attempt_directory(
                        task_id,
                        recovery_token,
                    )
                    if not generation_task_service.finish_orphan_recovery(
                        task_id,
                        expected_run_token=recovery_token,
                        terminal_status=GenerationStatus.FAILED,
                        error_message=(
                            "Generation restart was interrupted before checkpoint "
                            "preparation completed. Restart the task to retry safely."
                        ),
                    ):
                        logger.warning(
                            "Could not finalize orphan generation restart %s",
                            task_id,
                        )
                        continue
                    orphan_count += 1
                    continue
                recovery = external_sync_service.fail_generation_and_restore_batches(
                    task_id,
                    "Task interrupted by server restart",
                )
                recovery_reconciled = bool(
                    recovery.get("recovered")
                    or recovery.get("already_recovered")
                    or recovery.get("reconciled")
                )
                if recovery.get("tracking_found") and not recovery_reconciled:
                    logger.warning(
                        "Could not safely recover sync tracking for generation %s",
                        task_id,
                    )
                    continue
                if recovery.get("tracking_found"):
                    SyncManager._cleanup_unlaunched_generation_staging(
                        {
                            "gen_task_id": task_id,
                            "sync_task_id": recovery.get("task_id"),
                            "user_id": recovery.get("user_id")
                            or task.get("user_id"),
                            "raw_dataset_id": task.get("source_dataset_id"),
                            "merged_path": task.get("input_path"),
                        }
                    )
                sync_owned = bool(recovery.get("tracking_found"))
                terminal_status = (
                    GenerationStatus.FAILED
                    if sync_owned
                    or source_status
                    in {
                        GenerationStatus.PUBLISHING,
                        GenerationStatus.RECOVERING,
                    }
                    else GenerationStatus.STOPPED
                )
                if source_status in {
                    GenerationStatus.PUBLISHING,
                    GenerationStatus.RECOVERING,
                }:
                    terminal_error = (
                        "Dataset publication was interrupted by server restart. "
                        "Restart the task to retry safely."
                    )
                elif sync_owned:
                    terminal_error = (
                        "Sync generation setup was recovered. Trigger a new generation "
                        "from the sync task."
                    )
                else:
                    terminal_error = (
                        "Task interrupted by server restart. Use 'Restart' to resume."
                    )
                if source_status in {
                    GenerationStatus.PUBLISHING,
                    GenerationStatus.RECOVERING,
                }:
                    compensated = generation_publication_service.compensate_attempt(
                        task_id=task_id,
                        expected_run_token=run_token,
                        recovery_run_token=recovery_token,
                        user_id=raw_task.get("user_id"),
                    )
                    if not compensated:
                        logger.warning(
                            "Could not repair orphan generation publication %s",
                            task_id,
                        )
                        continue
                updated = generation_task_service.finish_orphan_recovery(
                    task_id,
                    expected_run_token=recovery_token,
                    terminal_status=terminal_status,
                    error_message=terminal_error,
                )
                if not updated:
                    logger.warning(
                        "Could not finalize orphan generation task %s",
                        task_id,
                    )
                    continue
                if recovery.get("recovered"):
                    logger.info(
                        "Recovered sync generation %s (%s batches, %s records)",
                        task_id,
                        recovery.get("restored_batch_count", 0),
                        recovery.get("restored_record_count", 0),
                    )
                orphan_count += 1
            except Exception as task_error:
                logger.warning(
                    "Could not finalize orphan generation task %s (error_type=%s)",
                    task_id,
                    type(task_error).__name__,
                )

        if orphan_count > 0:
            logger.info(f"Cleaned up {orphan_count} orphan generation tasks on startup")

    except Exception as e:
        logger.error(f"Failed to cleanup orphan generation tasks: {e}")


def cleanup_orphan_sync_trainings():
    """Reconcile durable sync training claims left pending at startup."""
    try:
        from ..storage.services.external_sync_service import external_sync_service
        from ..storage.services.training_task_service import training_task_service
        from ..sync.post_training_handler import on_training_completed

        failed_statuses = {
            TrainingStatus.FAILED.value,
            TrainingStatus.STOPPED.value,
            TrainingStatus.CANCELLED.value,
        }
        active_statuses = {
            TrainingStatus.PENDING.value,
            TrainingStatus.PREPARING.value,
            TrainingStatus.RUNNING.value,
            TrainingStatus.EVALUATING.value,
        }

        for tracking in _iter_pending_sync_trainings(external_sync_service):
            training_task_id = tracking.get("training_task_id")
            if not training_task_id:
                logger.warning("Pending sync training has no training task ID")
                continue
            try:
                training_task = training_task_service.get_task(training_task_id)
                if training_task is None:
                    external_sync_service.fail_training_and_restore_claim(
                        training_task_id,
                        "Training task missing during startup reconciliation",
                    )
                    continue

                status = training_task.get("status")
                if status == TrainingStatus.SUCCEEDED.value:
                    final_model_path = training_task.get("final_model_path")
                    if not final_model_path:
                        external_sync_service.fail_training_and_restore_claim(
                            training_task_id,
                            "Succeeded training has no output model path",
                        )
                        continue
                    on_training_completed(
                        training_task_id=training_task_id,
                        final_model_path=final_model_path,
                        model_registry_id=training_task.get(
                            "trained_model_registry_id"
                        ),
                    )
                    continue

                if status in active_statuses:
                    reason = "Training task interrupted by server restart"
                elif status in failed_statuses:
                    reason = "Terminal training reconciled during server startup"
                else:
                    reason = (
                        "Training task has an unsupported terminal status during "
                        "startup reconciliation"
                    )
                external_sync_service.fail_training_and_restore_claim(
                    training_task_id,
                    reason,
                )
            except Exception as reconcile_error:
                logger.warning(
                    "Could not reconcile sync training %s (error_type=%s)",
                    training_task_id,
                    type(reconcile_error).__name__,
                )
    except Exception as exc:
        logger.error(
            "Failed to cleanup orphan sync trainings (error_type=%s)",
            type(exc).__name__,
        )


def cleanup_orphan_datasets():
    """
    Clean up orphan dataset downloads stuck in 'downloading' on startup.

    Download progress is tracked only in-process, so after a restart a row left
    in 'downloading' can never complete and its progress endpoint returns 404.
    Transition such rows to 'error' so they are visibly failed and re-startable.
    """
    try:
        from ..storage.services.dataset_download_service import (
            dataset_download_service,
        )
        from ..storage.services.model_download_service import model_download_service

        model_count = model_download_service.cleanup_interrupted_downloads()
        dataset_count = dataset_download_service.cleanup_interrupted_downloads()
        if model_count or dataset_count:
            logger.info(
                "Cleaned up %s orphan model and %s orphan dataset downloads on startup",
                model_count,
                dataset_count,
            )

    except Exception as e:
        logger.error(f"Failed to cleanup orphan datasets: {e}")


def cleanup_orphan_sync_tasks():
    """
    Clean up orphan sync tasks stuck in transient states on startup.

    If the server crashed while sync tasks were in 'syncing', 'generating',
    'training', or 'loading_adapter' status, they will be stuck forever because
    the sync worker skips tasks in those states. Reset them to 'idle' so the
    worker can pick them up again.
    """
    try:
        from ..storage.services.external_sync_service import external_sync_service

        stuck_statuses = {"syncing", "generating", "training", "loading_adapter"}
        tasks = _collect_task_pages(external_sync_service.list_tasks)

        orphan_count = 0
        for task in tasks:
            if task.get("status") in stuck_statuses:
                task_id = task["task_id"]
                try:
                    if task.get("status") == "generating":
                        updated = external_sync_service.reset_generating_task_if_no_pending_generation(
                            task_id
                        )
                        if not updated:
                            logger.warning(
                                "Preserving sync task %s with an unreconciled generation",
                                task_id,
                            )
                            continue
                    else:
                        updated = external_sync_service.update_task(
                            task_id,
                            status="idle",
                            error_message=None,
                        )
                    if not updated:
                        logger.warning(
                            "Could not reset stuck sync task %s",
                            task_id,
                        )
                        continue
                    orphan_count += 1
                    logger.warning(
                        "Reset stuck sync task to idle: %s (was %s)",
                        task_id,
                        task["status"],
                    )
                except Exception as task_error:
                    logger.warning(
                        "Could not reset stuck sync task %s (error_type=%s)",
                        task_id,
                        type(task_error).__name__,
                    )

        if orphan_count > 0:
            logger.info(f"Cleaned up {orphan_count} stuck sync tasks on startup")

    except Exception as e:
        logger.error(f"Failed to cleanup orphan sync tasks: {e}")


def cleanup_orphan_containers():
    """
    Clean up orphan Docker containers on startup.

    Removes managed inference containers without a corresponding active deployment.
    """
    try:
        from sqlmodel import select
        from ..storage.database import get_session
        from ..storage.entities.deployment_entity import DeploymentDB
        from ..deployment.docker_deployer import docker_deployer

        # Get all container names from active deployments
        with get_session() as session:
            statement = select(DeploymentDB).where(
                DeploymentDB.deploy_mode == "container",
                DeploymentDB.container_name.isnot(None),
                DeploymentDB.status.in_(["pending", "starting", "running", "restarting"])
            )
            deployments = session.exec(statement).all()
            valid_containers = {d.container_name for d in deployments if d.container_name}

        # Cleanup orphan containers
        removed_count = docker_deployer.cleanup_orphan_containers(valid_containers)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} orphan Docker containers on startup")

    except Exception as e:
        logger.error(f"Failed to cleanup orphan containers: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting TrainFactory API server...")
    init_db()
    bootstrap_default_admin(settings)

    # Reconfigure logging after database migration and administrator bootstrap.
    # alembic fileConfig() disables existing loggers and sets root to WARN
    _root = logging.getLogger()
    _log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    _root.setLevel(_log_level)
    # Re-enable all app loggers disabled by alembic's fileConfig(disable_existing_loggers=True)
    for _name in logging.Logger.manager.loggerDict:
        _lg = logging.getLogger(_name)
        if hasattr(_lg, 'disabled'):
            _lg.disabled = False
    logger.info(
        "Database initialized, administrator bootstrap checked, logging configured "
        "(level=%s)",
        settings.log_level,
    )

    audit_cleanup_task = asyncio.create_task(_audit_log_cleanup_loop())
    sync_manager = None
    try:
        # Cleanup orphan tasks from previous runs
        cleanup_orphan_tasks()
        cleanup_orphan_sync_trainings()
        cleanup_orphan_evaluation_tasks()
        cleanup_orphan_deep_evaluation_tasks()
        cleanup_orphan_generation_tasks()
        cleanup_orphan_datasets()
        cleanup_orphan_containers()
        from .routes.sync_routes import resume_pending_sync_deletions

        resumed_sync_deletions, failed_sync_deletions = (
            await resume_pending_sync_deletions()
        )
        if resumed_sync_deletions or failed_sync_deletions:
            logger.info(
                "Sync deletion recovery completed: resumed=%s failed=%s",
                resumed_sync_deletions,
                failed_sync_deletions,
            )
        cleanup_orphan_sync_tasks()

        # Initialize the optional object store only when S3 storage is selected.
        if settings.storage_backend == "s3":
            try:
                from ..storage.object_store import get_object_store
                get_object_store()
                logger.info("MinIO object store initialized")
            except Exception as e:
                logger.warning(f"MinIO object store initialization failed (non-fatal): {e}")

        # Start sync manager (background workers for external data sync).
        from ..sync.sync_manager import sync_manager as runtime_sync_manager
        sync_manager = runtime_sync_manager
        await sync_manager.start()
        yield
    finally:
        audit_cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await audit_cleanup_task
        if sync_manager is not None:
            try:
                await sync_manager.stop()
            except Exception as exc:
                logger.error("Failed to stop sync manager cleanly: %s", exc)
        logger.info("Shutting down TrainFactory API server...")


def _get_action_from_method(method: str, path: str) -> str:
    """Determine audit action from HTTP method and path."""
    from ..storage.entities.audit_log_entity import AuditAction

    if "/login" in path:
        return AuditAction.LOGIN
    if "/logout" in path:
        return AuditAction.LOGOUT
    if "/start" in path:
        return AuditAction.START
    if "/stop" in path:
        return AuditAction.STOP
    if "/deploy" in path or "/container" in path:
        return AuditAction.DEPLOY
    if "/download" in path:
        return AuditAction.DOWNLOAD

    method_action_map = {
        "POST": AuditAction.CREATE,
        "PUT": AuditAction.UPDATE,
        "PATCH": AuditAction.UPDATE,
        "DELETE": AuditAction.DELETE,
        "GET": AuditAction.READ,
    }
    return method_action_map.get(method, AuditAction.READ)


def _get_resource_type_from_path(path: str) -> str:
    """Determine resource type from API path."""
    from ..storage.entities.audit_log_entity import AuditResource

    resource_prefixes = (
        ("/api/sync/api-configs", AuditResource.EXTERNAL_API_CONFIG),
        ("/api/deep-evaluation", AuditResource.DEEP_EVALUATION_TASK),
        ("/api/generation", AuditResource.GENERATION_TASK),
        ("/api/evaluations", AuditResource.EVALUATION_TASK),
        ("/api/sync", AuditResource.SYNC_TASK),
        ("/api/train", AuditResource.TRAINING_TASK),
        ("/api/models", AuditResource.MODEL),
        ("/api/deployments", AuditResource.DEPLOYMENT),
        ("/api/datasets", AuditResource.DATASET),
        ("/api/configs", AuditResource.MODEL_CONFIG),
        ("/api/auth", AuditResource.USER),
    )
    for prefix, resource_type in resource_prefixes:
        if path == prefix or path.startswith(f"{prefix}/"):
            return resource_type
    return "unknown"


def _should_audit_request(method: str, path: str) -> bool:
    """Keep security-relevant audit events without logging high-rate probes."""
    if path in {"/health", "/", "/docs", "/openapi.json", "/redoc"}:
        return False
    if method == "GET":
        return path.startswith("/api/auth/") and path not in {
            "/api/auth/config",
            "/api/auth/me",
        }
    return method in {"POST", "PUT", "PATCH", "DELETE"}


async def _resolve_audit_identity(request: Request) -> dict:
    """Resolve the request identity without allowing audit work to affect it."""
    from ..auth import dependencies

    anonymous = {"user_id": "anonymous", "username": None}
    if not settings.auth_enabled:
        return anonymous

    token = dependencies.get_access_token_from_http_request(request)
    if not token:
        return anonymous

    try:
        user = await dependencies.resolve_current_user(token)
    except Exception as exc:
        logger.debug("Audit identity resolution failed: %s", exc)
        return anonymous
    return {"user_id": user["user_id"], "username": user["username"]}


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="TrainFactory API",
        description="A standalone training framework for Embedding and Reranker models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Set up rate limiting
    app.state.limiter = auth_limiter
    app.add_middleware(SlowAPIMiddleware)
    if settings.rate_limit_enabled:
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Map SSRFError -> HTTP response. Raised by core.ssrf.validate_outbound_url
    # from any user-controlled outbound URL endpoint (model-config, deep-eval,
    # external-api-config, deployment discover-models).
    def _ssrf_error_handler(request: Request, exc: SSRFError):
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail}
        )

    app.add_exception_handler(SSRFError, _ssrf_error_handler)

    async def _integrity_error_handler(request: Request, exc: IntegrityError):
        # Constraint details can include SQL text and submitted values. Record only
        # request context, then return one stable conflict response for all routes.
        logger.info(
            "Database integrity conflict for %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=409,
            content={"detail": "Resource conflicts with existing data"},
        )

    app.add_exception_handler(IntegrityError, _integrity_error_handler)

    async def _dataset_consumption_handler(
        request: Request,
        exc: DatasetConsumptionUnavailableError,
    ):
        return JSONResponse(
            status_code=409,
            content={"detail": "Dataset is unavailable for task execution"},
        )

    app.add_exception_handler(
        DatasetConsumptionUnavailableError,
        _dataset_consumption_handler,
    )

    # Add CORS middleware with configurable origins
    allowed_origins = [
        origin.strip()
        for origin in settings.allowed_origins.split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        max_age=600,  # Cache preflight requests for 10 minutes
    )

    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Add audit logging middleware
    @app.middleware("http")
    async def audit_logging_middleware(request: Request, call_next):
        """Log API requests for audit trail."""
        from ..storage.services.audit_log_service import audit_log_service

        method = request.method
        path = request.url.path
        if not _should_audit_request(method, path):
            return await call_next(request)

        # Capture the identity while the request token is still current. Endpoints
        # such as password changes may revoke it before producing a response.
        identity = await _resolve_audit_identity(request)

        # Execute the request
        response = await call_next(request)

        # Successful login and registration requests have no inbound token. The
        # auth route records the newly issued, signed identity on request.state.
        identity = getattr(request.state, "audit_identity", identity)

        # Determine action and resource type from path
        action = _get_action_from_method(method, path)
        resource_type = _get_resource_type_from_path(path)

        # Get client info
        ip_address = get_rate_limit_key(request)
        user_agent = request.headers.get("user-agent", "")[:500]

        # Log the request (async in background to not block response)
        try:
            audit_log_service.log(
                user_id=identity["user_id"],
                username=identity["username"],
                action=action,
                resource_type=resource_type,
                method=method,
                endpoint=path,
                status_code=response.status_code,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        except Exception as e:
            logger.debug(f"Audit logging failed: {e}")

        return response

    # Include routers
    app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
    app.include_router(training_router, prefix="/api", tags=["training"])
    app.include_router(registry_router, prefix="/api", tags=["registry"])
    app.include_router(deployment_router, prefix="/api", tags=["deployment"])
    app.include_router(model_config_router, prefix="/api", tags=["model-config"])
    app.include_router(dataset_router, prefix="/api", tags=["dataset"])
    app.include_router(resource_router, prefix="/api", tags=["resources"])
    app.include_router(evaluation_router, prefix="/api", tags=["evaluations"])
    app.include_router(adapter_router, prefix="/api", tags=["adapters"])
    app.include_router(deep_evaluation_router, prefix="/api/deep-evaluation", tags=["deep-evaluation"])
    app.include_router(generation_router, prefix="/api/generation", tags=["generation"])
    app.include_router(milvus_router, prefix="/api/milvus", tags=["milvus"])
    app.include_router(sync_router, prefix="/api/sync", tags=["sync"])
    app.include_router(ext_api_config_router, prefix="/api/sync", tags=["sync"])

    # Global exception handler to prevent error message leaking
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Global exception handler to sanitize error messages.

        In production, internal errors should not expose stack traces or sensitive details.
        """
        # Exception messages from SDKs and databases can contain credentials or
        # request payload fragments. Keep only stable request context in logs.
        logger.error(
            "Unhandled exception (error_type=%s) for %s %s",
            type(exc).__name__,
            request.method,
            request.url.path,
            stack_info=settings.debug,
        )

        # Return sanitized error response
        if settings.debug:
            # In debug mode, include error details
            return JSONResponse(
                status_code=500,
                content={
                    "detail": str(exc),
                    "type": type(exc).__name__,
                }
            )
        else:
            # In production, return generic message
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

    @app.get("/health")
    def health_check():
        """Report readiness only while the primary database is reachable."""
        try:
            with get_session() as session:
                session.exec(text("SELECT 1"))
        except Exception as exc:
            logger.warning(
                "Database readiness check failed (%s)",
                type(exc).__name__,
            )
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "version": "0.1.0"},
            )
        return {"status": "healthy", "version": "0.1.0"}

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "TrainFactory API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


def run_server(host: str | None = None, port: int | None = None, workers: int | None = None):
    """Run the API server."""
    import uvicorn

    resolved_host = host or settings.api_host
    resolved_port = port or settings.api_port
    resolved_workers = settings.api_workers if workers is None else workers
    validate_api_worker_count(resolved_workers)

    app = create_app()
    uvicorn.run(
        app,
        host=resolved_host,
        port=resolved_port,
        workers=resolved_workers,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
