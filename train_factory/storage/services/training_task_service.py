"""
Training task service for database operations.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from train_factory.core.time_utils import now_naive
from train_factory.utils.strict_json import sanitize_json_value
from train_factory.utils.path_utils import (
    artifact_path_uses_root,
    canonicalize_artifact_path,
)

from sqlalchemy import case, or_, update
from sqlmodel import select, func

from ..database import get_session
from ..entities.model_registry_entity import ModelRegistryDB
from ..entities.training_task_entity import TrainingTaskDB
from .dataset_service import (
    DatasetConsumptionUnavailableError,
    DatasetDeletionInProgressError,
    lock_datasets_for_consumption,
    storage_reference_sets_overlap,
)
from .background_task_admission_service import background_task_admission_service
from .model_registry_service import ModelDeletionInProgressError
from .model_artifact_membership_service import lock_model_artifact_membership

logger = logging.getLogger(__name__)

_UNSET = object()


def _persisted_training_dataset_paths(task: TrainingTaskDB) -> set[str]:
    """Collect every dataset path used by a persisted training task."""
    paths = set()
    if isinstance(task.train_dataset_path, str) and task.train_dataset_path.strip():
        paths.add(task.train_dataset_path.strip())
    params = task.training_params or {}
    if isinstance(params, dict):
        direct_path = params.get("train_dataset_path")
        if isinstance(direct_path, str) and direct_path.strip():
            paths.add(direct_path.strip())
        configs = params.get("dataset_configs")
        if isinstance(configs, list):
            for config in configs:
                if not isinstance(config, dict):
                    continue
                path = config.get("path")
                if isinstance(path, str) and path.strip():
                    paths.add(path.strip())
    return paths


def _path_uses_artifact_root(candidate: Optional[str], root: Optional[str]) -> bool:
    return artifact_path_uses_root(candidate, root)


def _persisted_training_model_paths(task: TrainingTaskDB) -> set[str]:
    """Collect base and guide model artifacts used by one training task."""
    paths = set()

    def add_path(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            paths.add(value.strip())

    add_path(task.base_model_path)
    if isinstance(task.loss_config, dict):
        add_path(task.loss_config.get("guide_model"))
    params = task.training_params or {}
    if isinstance(params, dict):
        add_path(params.get("base_model_path"))
        nested_loss = params.get("loss_config")
        if isinstance(nested_loss, dict):
            add_path(nested_loss.get("guide_model"))
    return paths


def _lock_registry_models_for_training(
    session,
    task: TrainingTaskDB,
    *,
    membership_gate_locked: bool = False,
) -> List[ModelRegistryDB]:
    """Lock the membership gate, then matching models in stable ID order."""
    if not membership_gate_locked:
        lock_model_artifact_membership(session)
    paths = _persisted_training_model_paths(task)
    candidates = list(session.exec(select(ModelRegistryDB)).all())
    matching_ids = sorted(
        model.model_id
        for model in candidates
        if any(
            _path_uses_artifact_root(path, model.model_path) for path in paths
        )
    )
    if not matching_ids:
        return []
    locked = list(
        session.exec(
            select(ModelRegistryDB)
            .where(ModelRegistryDB.model_id.in_(matching_ids))
            .order_by(ModelRegistryDB.model_id)
            .with_for_update()
        ).all()
    )
    if {model.model_id for model in locked} != set(matching_ids):
        raise ModelDeletionInProgressError(
            "Registry model membership changed while acquiring locks; retry"
        )
    if any(
        not any(
            _path_uses_artifact_root(path, model.model_path) for path in paths
        )
        for model in locked
    ):
        raise ModelDeletionInProgressError(
            "Registry model paths changed while acquiring locks; retry"
        )
    return locked


def _training_model_reference_signature(task: TrainingTaskDB) -> tuple[str, ...]:
    return tuple(
        sorted(
            canonical
            for path in _persisted_training_model_paths(task)
            if (canonical := canonicalize_artifact_path(path)) is not None
        )
    )


def _require_training_reference_signature(
    task: TrainingTaskDB,
    expected: tuple[str, ...],
) -> None:
    if _training_model_reference_signature(task) != expected:
        raise ModelDeletionInProgressError(
            "Training model references changed while acquiring locks; retry"
        )


def _require_training_model_paths_available(
    models: List[ModelRegistryDB],
    task: TrainingTaskDB,
) -> None:
    paths = _persisted_training_model_paths(task)
    for model in models:
        if not any(
            _path_uses_artifact_root(path, model.model_path) for path in paths
        ):
            continue
        if model.status == "deleting":
            raise ModelDeletionInProgressError(
                f"Model is being deleted: {model.model_id}"
            )


class TrainingTaskService:
    """Service for training task database operations."""

    def create_task(
        self,
        task_name: Optional[str] = None,
        model_path: Optional[str] = None,
        train_dataset_path: Optional[str] = None,
        training_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        # New unified type system
        model_type: str = "embedding",
        training_method: str = "sft",
        model_architecture: Optional[str] = None,
        # RL & Two-stage training
        rl_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None,
        sft_checkpoint_path: Optional[str] = None,
        # Output directory
        output_dir: Optional[str] = None,
        # Per-attempt execution identity
        run_token: Optional[str] = None,
        # Optional caller-planned identity for durable cross-table workflows
        task_id: Optional[str] = None,
        require_managed_datasets: bool = False,
    ) -> Dict[str, Any]:
        """Create a new training task. Returns dict with task info."""
        is_lora = bool(training_params.get("use_lora")) if training_params else False
        with get_session() as session:
            task = TrainingTaskDB(
                task_name=task_name,
                base_model_path=model_path,
                train_dataset_path=train_dataset_path,
                training_params=training_params,
                description=description,
                user_id=user_id,
                status="pending",
                is_lora=is_lora,
                # New unified type system
                model_type=model_type,
                training_method=training_method,
                model_architecture=model_architecture or ("encoder" if model_type in ["embedding", "reranker"] else "decoder"),
                # RL & Two-stage training
                rl_config=rl_config,
                loss_config=loss_config,
                parent_task_id=parent_task_id,
                sft_checkpoint_path=sft_checkpoint_path,
                # Output directory
                output_dir=output_dir,
                run_token=run_token,
            )
            if task_id:
                task.task_id = task_id
            registry_models = _lock_registry_models_for_training(session, task)
            _require_training_model_paths_available(registry_models, task)
            lock_datasets_for_consumption(
                session,
                storage_refs=_persisted_training_dataset_paths(task),
                require_all_storage_refs=require_managed_datasets,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            logger.info(f"Created training task: {task.task_id}")

            # Log event
            try:
                from .training_task_event_service import training_task_event_service

                training_task_event_service.log_event(
                    task_id=task.task_id,
                    event_type="task_created",
                    user_id=task.user_id,
                    payload={
                        "model_type": task.model_type,
                        "training_method": task.training_method,
                        "status": task.status,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log task_created event for task {task.task_id}: {e}")
            # Return dict to avoid DetachedInstanceError
            return {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "model_type": task.model_type,
                "training_method": task.training_method,
                "model_architecture": task.model_architecture,
                "user_id": task.user_id,
                "status": task.status,
                "created_at": task.created_at,
                "run_token": task.run_token,
            }

    def _task_to_dict(self, task: TrainingTaskDB) -> Dict[str, Any]:
        """Convert task ORM object to dictionary."""
        return {
            "id": task.id,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "description": task.description,
            # New unified type system
            "model_type": task.model_type,
            "training_method": task.training_method,
            "model_architecture": task.model_architecture,
            "user_id": task.user_id,
            "base_model_path": task.base_model_path,
            "final_model_path": task.final_model_path,
            "train_dataset_path": task.train_dataset_path,
            "output_dir": task.output_dir,
            "device": task.device,
            "is_lora": task.is_lora,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "training_params": sanitize_json_value(task.training_params),
            "final_metrics": sanitize_json_value(task.final_metrics),
            # RL & Two-stage training
            "rl_config": task.rl_config,
            "loss_config": task.loss_config,
            "parent_task_id": task.parent_task_id,
            "sft_checkpoint_path": task.sft_checkpoint_path,
            "trained_model_registry_id": task.trained_model_registry_id,
            "process_pid": task.process_pid,
            "process_status": task.process_status,
            "process_create_time": task.process_create_time,
            "run_token": task.run_token,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
        }

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by task_id. Returns dict or None."""
        with get_session() as session:
            statement = select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            task = session.exec(statement).first()
            if task:
                return self._task_to_dict(task)
            return None

    def list_active_dataset_consumers(
        self,
        dataset_paths: List[str],
        *,
        exclude_task_ids: tuple[str, ...] = (),
    ) -> List[str]:
        """Return active training tasks that still consume these paths."""
        if not dataset_paths:
            return []
        requested_paths = {
            path.strip()
            for path in dataset_paths
            if isinstance(path, str) and path.strip()
        }
        if not requested_paths:
            return []
        executing_task_ids = (
            background_task_admission_service.get_executing_task_ids("training")
        )
        with get_session() as session:
            query = select(TrainingTaskDB).where(
                or_(
                    TrainingTaskDB.status.in_(
                        ("pending", "preparing", "running", "evaluating")
                    ),
                    TrainingTaskDB.task_id.in_(executing_task_ids),
                    TrainingTaskDB.process_pid.is_not(None),
                    TrainingTaskDB.process_status.is_not(None),
                    TrainingTaskDB.process_create_time.is_not(None),
                ),
            )
            if exclude_task_ids:
                query = query.where(TrainingTaskDB.task_id.notin_(exclude_task_ids))
            return [
                task.task_id
                for task in session.exec(query).all()
                if storage_reference_sets_overlap(
                    _persisted_training_dataset_paths(task),
                    requested_paths,
                )
            ]

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        training_method: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get all tasks with optional filters.

        Returns:
            Tuple of (tasks, total_count)
        """
        with get_session() as session:
            # Build filter conditions
            conditions = []
            if status:
                conditions.append(TrainingTaskDB.status == status)
            if model_type:
                conditions.append(TrainingTaskDB.model_type == model_type)
            if training_method:
                conditions.append(TrainingTaskDB.training_method == training_method)
            if user_id:
                conditions.append(TrainingTaskDB.user_id == user_id)

            # Get total count
            count_stmt = select(func.count()).select_from(TrainingTaskDB)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Get paginated data
            statement = select(TrainingTaskDB)
            for cond in conditions:
                statement = statement.where(cond)
            statement = statement.order_by(
                TrainingTaskDB.created_at.desc(),
                TrainingTaskDB.id.desc(),
            )
            statement = statement.offset(offset).limit(limit)
            tasks = session.exec(statement).all()
            return [self._task_to_dict(task) for task in tasks], total

    def get_tasks_with_process_info(
        self,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return tasks carrying durable subprocess ownership evidence."""
        condition = or_(
            TrainingTaskDB.process_pid.is_not(None),
            TrainingTaskDB.process_status.is_not(None),
            TrainingTaskDB.process_create_time.is_not(None),
        )
        with get_session() as session:
            total = session.exec(
                select(func.count()).select_from(TrainingTaskDB).where(condition)
            ).one()
            tasks = session.exec(
                select(TrainingTaskDB)
                .where(condition)
                .order_by(TrainingTaskDB.id)
                .offset(offset)
                .limit(limit)
            ).all()
            return [self._task_to_dict(task) for task in tasks], total

    def update_task_status(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None,
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Atomically update task status without overwriting a concurrent transition."""
        with get_session() as session:
            registry_models: List[ModelRegistryDB] = []
            expected_reference_signature: tuple[str, ...] | None = None
            if status in {"pending", "preparing", "running", "evaluating"}:
                lock_model_artifact_membership(session)
                candidate_statement = select(TrainingTaskDB).where(
                    TrainingTaskDB.task_id == task_id
                )
                if run_token is not None:
                    candidate_statement = candidate_statement.where(
                        TrainingTaskDB.run_token == run_token
                    )
                candidate_task = session.exec(candidate_statement).first()
                if candidate_task is None:
                    return False
                expected_reference_signature = (
                    _training_model_reference_signature(candidate_task)
                )
                registry_models = _lock_registry_models_for_training(
                    session,
                    candidate_task,
                    membership_gate_locked=True,
                )
            statement = (
                select(TrainingTaskDB)
                .where(TrainingTaskDB.task_id == task_id)
                .with_for_update()
            )
            if run_token is not None:
                statement = statement.where(TrainingTaskDB.run_token == run_token)
            task = session.exec(statement).first()
            if not task:
                return False
            if expected_reference_signature is not None:
                _require_training_reference_signature(
                    task,
                    expected_reference_signature,
                )
                _require_training_model_paths_available(
                    registry_models,
                    task,
                )

            valid_statuses = set(TrainingTaskDB.VALID_TRANSITIONS)
            if status not in valid_statuses:
                raise ValueError(
                    f"Invalid status: {status}. Must be one of: {', '.join(sorted(valid_statuses))}"
                )

            old_status = task.status
            allowed = TrainingTaskDB.VALID_TRANSITIONS.get(old_status, set())
            if status != old_status and status not in allowed:
                raise ValueError(
                    f"Invalid status transition: {old_status} -> {status}. "
                    f"Allowed: {allowed or 'none'}"
                )

            now = now_naive()
            values: Dict[str, Any] = {
                "status": status,
                "updated_at": now,
            }
            if error_message:
                values["error_message"] = error_message
            if status == "running" and task.started_at is None:
                values["started_at"] = now
            if status in {"succeeded", "failed", "stopped", "cancelled"}:
                values["completed_at"] = now

            conditions = [
                TrainingTaskDB.task_id == task_id,
                TrainingTaskDB.status == old_status,
            ]
            if run_token is not None:
                conditions.append(TrainingTaskDB.run_token == run_token)
            result = session.exec(
                update(TrainingTaskDB).where(*conditions).values(**values)
            )
            session.commit()
            if result.rowcount != 1:
                logger.info(
                    "Skipped stale training task transition %s -> %s for %s",
                    old_status,
                    status,
                    task_id,
                )
                return False

            logger.info(f"Updated task {task_id} status to {status}")

            if old_status != status:
                try:
                    from .training_task_event_service import training_task_event_service

                    training_task_event_service.log_event(
                        task_id=task.task_id,
                        event_type="status_changed",
                        user_id=task.user_id,
                        payload={"from": old_status, "to": status, "error_message": error_message},
                    )
                except Exception as e:
                    logger.warning(f"Failed to log status_changed event for task {task_id}: {e}")
            return True

    def update_task_progress(
        self,
        task_id: str,
        progress: float,
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Update task progress."""
        with get_session() as session:
            conditions = [TrainingTaskDB.task_id == task_id]
            if run_token is not None:
                conditions.extend(
                    [
                        TrainingTaskDB.run_token == run_token,
                        TrainingTaskDB.status.in_(
                            ("preparing", "running", "evaluating")
                        ),
                    ]
                )
            statement = select(TrainingTaskDB).where(*conditions)
            task = session.exec(statement).first()
            if task:
                previous_progress = task.progress or 0.0
                next_progress = min(max(progress, 0.0), 100.0)
                result = session.exec(
                    update(TrainingTaskDB)
                    .where(*conditions)
                    .values(progress=next_progress, updated_at=now_naive())
                )
                session.commit()
                if result.rowcount != 1:
                    return False
                should_log = next_progress >= 100 or abs(next_progress - previous_progress) >= 1.0
                if should_log:
                    try:
                        from .training_task_event_service import training_task_event_service

                        training_task_event_service.log_event(
                            task_id=task.task_id,
                            event_type="progress_updated",
                            user_id=task.user_id,
                            payload={"progress": next_progress},
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log progress event for task {task_id}: {e}")
                return True
            return False

    def update_task_result(
        self,
        task_id: str,
        final_model_path: Optional[str] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update task result.

        DEPRECATED：无调用者（complete_task 已覆盖并发安全的完成路径），
        且缺少状态校验/条件更新——启用前需并入 complete_task 的 CAS 语义。
        """
        with get_session() as session:
            statement = select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            task = session.exec(statement).first()
            if task:
                if final_model_path:
                    task.final_model_path = final_model_path
                if final_metrics:
                    task.final_metrics = sanitize_json_value(final_metrics)
                if error_message:
                    task.error_message = error_message
                task.updated_at = now_naive()
                session.add(task)
                session.commit()
                logger.info(f"Updated task {task_id} result")
                try:
                    from .training_task_event_service import training_task_event_service

                    training_task_event_service.log_event(
                        task_id=task.task_id,
                        event_type="result_updated",
                        user_id=task.user_id,
                        payload={
                            "final_model_path": final_model_path,
                            "final_metrics": sanitize_json_value(final_metrics),
                            "error_message": error_message,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to log result update for task {task_id}: {e}")
                return True
            return False

    def update_task_trained_model_registry_id(self, task_id: str, model_id: str) -> bool:
        """Update trained model registry ID for a task."""
        with get_session() as session:
            statement = select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            task = session.exec(statement).first()
            if not task:
                return False

            task.trained_model_registry_id = model_id
            task.updated_at = now_naive()
            session.add(task)
            session.commit()
            logger.info(f"Updated task {task_id} trained_model_registry_id to {model_id}")
            try:
                from .training_task_event_service import training_task_event_service

                training_task_event_service.log_event(
                    task_id=task.task_id,
                    event_type="model_registered",
                    user_id=task.user_id,
                    payload={"model_id": model_id},
                )
            except Exception as e:
                logger.warning(f"Failed to log model_registered for task {task_id}: {e}")
            return True

    def complete_task(
        self,
        task_id: str,
        status: str,
        final_model_path: Optional[str] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """
        Atomically complete a task by updating status and result in a single transaction.

        This ensures that the task status and result are always consistent.

        Args:
            task_id: The task ID
            status: New status (succeeded, failed, stopped)
            final_model_path: Path to the trained model
            final_metrics: Training metrics
            error_message: Error message if failed

        Returns:
            True if updated successfully, False if task not found
        """
        terminal_statuses = {"succeeded", "failed", "stopped", "cancelled"}
        if status not in terminal_statuses:
            raise ValueError(
                f"Completion status must be one of: {', '.join(sorted(terminal_statuses))}"
            )

        allowed_sources = tuple(
            source
            for source, targets in TrainingTaskDB.VALID_TRANSITIONS.items()
            if status in targets
        )
        now = now_naive()
        with get_session() as session:
            statement = select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            if run_token is not None:
                statement = statement.where(TrainingTaskDB.run_token == run_token)
            task = session.exec(statement).first()
            if not task:
                logger.warning(f"Task not found for completion: {task_id}")
                return False

            old_status = task.status
            values: Dict[str, Any] = {
                "status": status,
                "updated_at": now,
                "completed_at": now,
            }
            if error_message:
                values["error_message"] = error_message
            elif status == "succeeded":
                values["error_message"] = None
            if final_model_path:
                values["final_model_path"] = final_model_path
            sanitized_final_metrics = sanitize_json_value(final_metrics)
            if final_metrics:
                values["final_metrics"] = sanitized_final_metrics

            conditions = [
                TrainingTaskDB.task_id == task_id,
                TrainingTaskDB.status.in_(allowed_sources),
            ]
            if run_token is not None:
                conditions.append(TrainingTaskDB.run_token == run_token)
            result = session.exec(
                update(TrainingTaskDB).where(*conditions).values(**values)
            )
            session.commit()
            if result.rowcount != 1:
                logger.info(
                    "Skipped completion for inactive training task %s (status=%s)",
                    task_id,
                    old_status,
                )
                return False

            logger.info(f"Task {task_id} completed atomically with status={status}")

            if old_status != status:
                try:
                    from .training_task_event_service import training_task_event_service

                    training_task_event_service.log_event(
                        task_id=task.task_id,
                        event_type="task_completed",
                        user_id=task.user_id,
                        payload={
                            "from": old_status,
                            "to": status,
                            "final_model_path": final_model_path,
                            "final_metrics": sanitized_final_metrics,
                            "error_message": error_message,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to log task_completed event for task {task_id}: {e}")
            return True

    def update_process_info(
        self,
        task_id: str,
        process_pid: Any = _UNSET,
        process_status: Any = _UNSET,
        process_create_time: Any = _UNSET,
        *,
        run_token: Optional[str] = None,
        expected_process_pid: Any = _UNSET,
        expected_process_create_time: Any = _UNSET,
    ) -> bool:
        """Update process information, optionally fenced by prior identity."""
        with get_session() as session:
            conditions = [TrainingTaskDB.task_id == task_id]
            if run_token is not None:
                conditions.append(TrainingTaskDB.run_token == run_token)
            if expected_process_pid is not _UNSET:
                if expected_process_pid is None:
                    conditions.append(TrainingTaskDB.process_pid.is_(None))
                else:
                    conditions.append(
                        TrainingTaskDB.process_pid == expected_process_pid
                    )
            if expected_process_create_time is not _UNSET:
                if expected_process_create_time is None:
                    conditions.append(
                        TrainingTaskDB.process_create_time.is_(None)
                    )
                else:
                    conditions.append(
                        TrainingTaskDB.process_create_time
                        == expected_process_create_time
                    )
            values: Dict[str, Any] = {"updated_at": now_naive()}
            if process_pid is not _UNSET:
                values["process_pid"] = process_pid
            if process_status is not _UNSET:
                values["process_status"] = process_status
            if process_create_time is not _UNSET:
                values["process_create_time"] = process_create_time
            result = session.exec(
                update(TrainingTaskDB).where(*conditions).values(**values)
            )
            session.commit()
            return result.rowcount == 1

    def clear_stale_process_info(self) -> int:
        """Clear process leases left behind when the API process restarts."""
        with get_session() as session:
            result = session.exec(
                update(TrainingTaskDB)
                .where(
                    or_(
                        TrainingTaskDB.process_pid.is_not(None),
                        TrainingTaskDB.process_status.is_not(None),
                        TrainingTaskDB.process_create_time.is_not(None),
                    )
                )
                .values(
                    process_pid=None,
                    process_status=None,
                    process_create_time=None,
                    updated_at=now_naive(),
                )
            )
            session.commit()
            return int(result.rowcount or 0)

    def list_artifact_consumers(
        self,
        task_id: str,
        output_dir: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Return training tasks that still reference a task's output artifacts."""
        with get_session() as session:
            candidates = session.exec(
                select(TrainingTaskDB).where(
                    TrainingTaskDB.task_id != task_id,
                    or_(
                        TrainingTaskDB.parent_task_id == task_id,
                        TrainingTaskDB.sft_checkpoint_path.is_not(None),
                    ),
                )
            ).all()
            return [
                candidate.to_dict()
                for candidate in candidates
                if candidate.parent_task_id == task_id
                or _path_uses_artifact_root(
                    candidate.sft_checkpoint_path,
                    output_dir,
                )
            ]

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        with get_session() as session:
            statement = select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            task = session.exec(statement).first()
            if task:
                user_id = task.user_id
                status = task.status
                session.delete(task)
                session.commit()
                logger.info(f"Deleted task {task_id}")
                try:
                    from .training_task_event_service import training_task_event_service

                    training_task_event_service.log_event(
                        task_id=task_id,
                        event_type="task_deleted",
                        user_id=user_id,
                        payload={"status": status},
                    )
                except Exception as e:
                    logger.warning(f"Failed to log task_deleted event for task {task_id}: {e}")

                # Cascade: clean up sync training tracking if this task was triggered by sync
                try:
                    from .external_sync_service import external_sync_service
                    external_sync_service.delete_training_tracking(task_id)
                except Exception as e:
                    logger.warning(f"Failed to clean sync tracking for training task {task_id}: {e}")

                return True
            return False

    def update_task_metrics(
        self,
        task_id: str,
        metrics: Dict[str, Any],
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Update real-time training metrics.

        Stores metrics like current_step, train_loss, etc. in training_params['_runtime_metrics'].

        Args:
            task_id: The task ID
            metrics: Dict with keys like current_step, total_steps, train_loss, etc.

        Returns:
            True if updated successfully
        """
        with get_session() as session:
            conditions = [TrainingTaskDB.task_id == task_id]
            if run_token is not None:
                conditions.extend(
                    [
                        TrainingTaskDB.run_token == run_token,
                        TrainingTaskDB.status.in_(
                            ("preparing", "running", "evaluating")
                        ),
                    ]
                )
            # 行锁：_runtime_metrics 是读-改-写整列合并，并发写者会互相覆盖
            # （对齐 evaluation_task_service.update_model_progress 的做法）
            statement = select(TrainingTaskDB).where(*conditions).with_for_update()
            task = session.exec(statement).first()
            if not task:
                return False

            # Get existing training_params or create new dict
            params = sanitize_json_value(dict(task.training_params or {}))
            sanitized_metrics = sanitize_json_value(metrics)

            # Merge metrics to avoid overwriting train/eval loss from alternating logs
            existing_metrics = dict(params.get('_runtime_metrics') or {})
            params['_runtime_metrics'] = {
                **existing_metrics,
                **sanitized_metrics,
                'updated_at': now_naive().isoformat(),
            }

            result = session.exec(
                update(TrainingTaskDB)
                .where(*conditions)
                .values(training_params=params, updated_at=now_naive())
            )
            session.commit()
            if result.rowcount != 1:
                return False
            try:
                from .training_task_event_service import training_task_event_service

                training_task_event_service.log_event(
                    task_id=task.task_id,
                    event_type="metrics_updated",
                    user_id=task.user_id,
                    payload=sanitized_metrics,
                )
            except Exception as e:
                logger.warning(f"Failed to log metrics update for task {task_id}: {e}")
            return True

    def get_task_metrics(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get real-time training metrics for a task.

        Returns:
            Dict with runtime metrics or None if not found
        """
        with get_session() as session:
            statement = select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            task = session.exec(statement).first()
            if task and task.training_params:
                return sanitize_json_value(
                    task.training_params.get('_runtime_metrics')
                )
        return None

    def update_task_output_dir(
        self,
        task_id: str,
        output_dir: str,
        training_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update task output directory and optionally refresh training_params."""
        with get_session() as session:
            registry_models: List[ModelRegistryDB] = []
            expected_reference_signature: tuple[str, ...] | None = None
            if training_params is not None:
                lock_model_artifact_membership(session)
                candidate_task = session.exec(
                    select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
                ).first()
                if candidate_task is None:
                    return False
                expected_reference_signature = (
                    _training_model_reference_signature(candidate_task)
                )
                prospective_candidate = TrainingTaskDB(
                    **candidate_task.model_dump()
                )
                prospective_params = dict(training_params)
                prospective_params["output_dir"] = output_dir
                prospective_candidate.training_params = prospective_params
                registry_models = _lock_registry_models_for_training(
                    session,
                    prospective_candidate,
                    membership_gate_locked=True,
                )
            statement = (
                select(TrainingTaskDB)
                .where(TrainingTaskDB.task_id == task_id)
                .with_for_update()
            )
            task = session.exec(statement).first()
            if not task:
                return False
            if expected_reference_signature is not None:
                _require_training_reference_signature(
                    task,
                    expected_reference_signature,
                )

            task.output_dir = output_dir
            if training_params is not None:
                training_params = dict(training_params)
                training_params["output_dir"] = output_dir
                task.training_params = training_params

            if expected_reference_signature is not None:
                _require_training_model_paths_available(registry_models, task)

            task.updated_at = now_naive()
            session.add(task)
            session.commit()
            logger.info(f"Updated task {task_id} output_dir to {output_dir}")
            try:
                from .training_task_event_service import training_task_event_service

                training_task_event_service.log_event(
                    task_id=task.task_id,
                    event_type="output_dir_updated",
                    user_id=task.user_id,
                    payload={"output_dir": output_dir},
                )
            except Exception as e:
                logger.warning(f"Failed to log output_dir update for task {task_id}: {e}")
            return True

    def update_task_execution_config(
        self,
        task_id: str,
        *,
        model_path: Optional[str] = None,
        train_dataset_path: Optional[str] = None,
        training_params: Optional[Dict[str, Any]] = None,
        sft_checkpoint_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        run_token: Optional[str] = None,
    ) -> bool:
        """Persist normalized execution config for an existing task."""
        with get_session() as session:
            conditions = [TrainingTaskDB.task_id == task_id]
            if run_token is not None:
                conditions.extend(
                    [
                        TrainingTaskDB.run_token == run_token,
                        TrainingTaskDB.status.in_(("preparing", "running")),
                    ]
                )
            registry_models: List[ModelRegistryDB] = []
            expected_reference_signature: tuple[str, ...] | None = None
            membership_change = model_path is not None or training_params is not None
            if membership_change:
                lock_model_artifact_membership(session)
                candidate_task = session.exec(
                    select(TrainingTaskDB).where(*conditions)
                ).first()
                if candidate_task is None:
                    return False
                expected_reference_signature = (
                    _training_model_reference_signature(candidate_task)
                )
                prospective_candidate = TrainingTaskDB(
                    **candidate_task.model_dump()
                )
                if model_path is not None:
                    prospective_candidate.base_model_path = model_path
                if training_params is not None:
                    prospective_params = dict(training_params)
                    prospective_params.pop("_run_token", None)
                    prospective_candidate.training_params = prospective_params
                registry_models = _lock_registry_models_for_training(
                    session,
                    prospective_candidate,
                    membership_gate_locked=True,
                )
            statement = (
                select(TrainingTaskDB)
                .where(*conditions)
                .with_for_update()
            )
            task = session.exec(statement).first()
            if not task:
                return False
            if expected_reference_signature is not None:
                _require_training_reference_signature(
                    task,
                    expected_reference_signature,
                )

            # Construct a detached value object. ``model_copy`` preserves the
            # SQLAlchemy instrumentation state of ORM-backed SQLModel rows and
            # mutating that copy can dereference the attached parent instance.
            prospective_task = TrainingTaskDB(**task.model_dump())
            if model_path is not None:
                prospective_task.base_model_path = model_path
            if training_params is not None:
                prospective_params = dict(training_params)
                prospective_params.pop("_run_token", None)
                prospective_task.training_params = prospective_params
            if membership_change:
                _require_training_model_paths_available(
                    registry_models,
                    prospective_task,
                )

            values: Dict[str, Any] = {"updated_at": now_naive()}
            if model_path is not None:
                values["base_model_path"] = model_path
            if train_dataset_path is not None:
                values["train_dataset_path"] = train_dataset_path
            if sft_checkpoint_path is not None:
                values["sft_checkpoint_path"] = sft_checkpoint_path
            if output_dir is not None:
                values["output_dir"] = output_dir
            if device is not None:
                values["device"] = device
            if training_params is not None:
                persisted_params = dict(training_params)
                persisted_params.pop("_run_token", None)
                values["training_params"] = persisted_params

            result = session.exec(
                update(TrainingTaskDB).where(*conditions).values(**values)
            )
            session.commit()
            if result.rowcount != 1:
                return False
            logger.info(f"Updated task {task_id} execution config")
            return True

    def get_task_stats(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """Get task statistics by status.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Dict with status counts: {total, pending, running, succeeded, failed, stopped}
        """
        with get_session() as session:
            stats: Dict[str, int] = {
                "total": 0,
                "pending": 0,
                "running": 0,
                "succeeded": 0,
                "failed": 0,
                "stopped": 0,
            }

            query = select(TrainingTaskDB.status, func.count()).group_by(TrainingTaskDB.status)
            if user_id:
                query = query.where(TrainingTaskDB.user_id == user_id)
            rows = session.exec(query).all()

            for status, count in rows:
                normalized = status
                if status in ["preparing", "evaluating"]:
                    normalized = "running"
                elif status == "cancelled":
                    normalized = "stopped"

                if normalized not in stats:
                    stats[normalized] = 0
                stats[normalized] += count

            stats["total"] = sum(
                stats.get(k, 0) for k in ["pending", "running", "succeeded", "failed", "stopped"]
            )

            return stats

    def claim_preparing(self, task_id: str, run_token: str) -> bool:
        """Atomically claim a pending task before allocating resources."""
        if not run_token:
            return False
        with get_session() as session:
            lock_model_artifact_membership(session)
            candidate_task = session.exec(
                select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            ).first()
            if candidate_task is None:
                return False
            expected_reference_signature = _training_model_reference_signature(
                candidate_task
            )
            registry_models = _lock_registry_models_for_training(
                session,
                candidate_task,
                membership_gate_locked=True,
            )
            task = session.exec(
                select(TrainingTaskDB)
                .where(TrainingTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False
            try:
                _require_training_reference_signature(
                    task,
                    expected_reference_signature,
                )
                _require_training_model_paths_available(
                    registry_models,
                    task,
                )
                lock_datasets_for_consumption(
                    session,
                    storage_refs=_persisted_training_dataset_paths(task),
                )
            except (
                DatasetDeletionInProgressError,
                ModelDeletionInProgressError,
            ):
                return False
            result = session.exec(
                update(TrainingTaskDB)
                .where(
                    TrainingTaskDB.task_id == task_id,
                    TrainingTaskDB.status == "pending",
                    or_(
                        TrainingTaskDB.run_token == run_token,
                        TrainingTaskDB.run_token.is_(None),
                    ),
                )
                .values(
                    status="preparing",
                    run_token=run_token,
                    updated_at=now_naive(),
                )
            )
            session.commit()
            claimed = result.rowcount == 1
            if claimed:
                logger.info("Claimed training task %s for preparation", task_id)
            return claimed

    def stop_if_active(self, task_id: str) -> bool:
        """Atomically stop a task only while it remains active."""
        now = now_naive()
        with get_session() as session:
            result = session.exec(
                update(TrainingTaskDB)
                .where(
                    TrainingTaskDB.task_id == task_id,
                    TrainingTaskDB.status.in_(
                        ("pending", "preparing", "running", "evaluating")
                    ),
                )
                .values(
                    status="stopped",
                    process_status=case(
                        (
                            or_(
                                TrainingTaskDB.process_pid.is_not(None),
                                TrainingTaskDB.process_status.is_not(None),
                                TrainingTaskDB.process_create_time.is_not(None),
                            ),
                            "stopping",
                        ),
                        else_=None,
                    ),
                    updated_at=now,
                    completed_at=now,
                )
            )
            session.commit()
            stopped = result.rowcount == 1
            if stopped:
                logger.info("Stopped active training task %s", task_id)
            return stopped

    def reset_for_resume(
        self,
        task_id: str,
        run_token: str,
        *,
        require_managed_datasets: bool = False,
    ) -> bool:
        """Reset task state for resuming training from a checkpoint.

        Preserves output_dir, training_params, and final_metrics so
        the trainer can pick up from where it left off.
        """
        if not run_token:
            return False
        with get_session() as session:
            lock_model_artifact_membership(session)
            candidate_task = session.exec(
                select(TrainingTaskDB).where(TrainingTaskDB.task_id == task_id)
            ).first()
            if candidate_task is None:
                return False
            expected_reference_signature = _training_model_reference_signature(
                candidate_task
            )
            registry_models = _lock_registry_models_for_training(
                session,
                candidate_task,
                membership_gate_locked=True,
            )
            task = session.exec(
                select(TrainingTaskDB)
                .where(TrainingTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False
            try:
                _require_training_reference_signature(
                    task,
                    expected_reference_signature,
                )
                _require_training_model_paths_available(
                    registry_models,
                    task,
                )
                lock_datasets_for_consumption(
                    session,
                    storage_refs=_persisted_training_dataset_paths(task),
                    require_all_storage_refs=require_managed_datasets,
                )
            except (
                DatasetConsumptionUnavailableError,
                ModelDeletionInProgressError,
            ):
                return False
            result = session.exec(
                update(TrainingTaskDB)
                .where(
                    TrainingTaskDB.task_id == task_id,
                    TrainingTaskDB.status.in_(("failed", "stopped")),
                    TrainingTaskDB.process_pid.is_(None),
                    TrainingTaskDB.process_status.is_(None),
                    TrainingTaskDB.process_create_time.is_(None),
                )
                .values(
                    status="pending",
                    error_message=None,
                    progress=0.0,
                    process_pid=None,
                    process_status=None,
                    process_create_time=None,
                    run_token=run_token,
                    started_at=None,
                    completed_at=None,
                    updated_at=now_naive(),
                )
            )
            session.commit()
            return result.rowcount == 1

    def find_latest_checkpoint(self, task_id: str) -> Optional[str]:
        """Find the latest HuggingFace checkpoint in the task's output directory.

        Returns the path to the most recent checkpoint-* directory, or None
        if no checkpoints exist.
        """
        import os
        import glob as glob_mod

        task = self.get_task(task_id)
        if not task:
            return None
        output_dir = task.get("output_dir")
        if not output_dir or not os.path.isdir(output_dir):
            return None
        # HuggingFace checkpoint directory pattern: checkpoint-{step}
        checkpoints = sorted(
            glob_mod.glob(os.path.join(output_dir, "checkpoint-*")),
            key=os.path.getmtime,
        )
        return checkpoints[-1] if checkpoints else None


# Global service instance
training_task_service = TrainingTaskService()
