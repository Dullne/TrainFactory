"""
Evaluation task service for database operations.
"""

import logging
from copy import deepcopy
from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4
from train_factory.core.time_utils import now_naive

from sqlmodel import select, func
from sqlalchemy import or_, update
from sqlalchemy.orm.attributes import flag_modified

from ..database import get_session
from ..entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationStatus,
    EvaluationTaskDB,
)
from .dataset_service import (
    DatasetConsumptionUnavailableError,
    lock_datasets_for_consumption,
    storage_reference_sets_overlap,
)
from .background_task_admission_service import background_task_admission_service
from .runtime_dependency_service import (
    RuntimeDependencyUnavailableError,
    evaluation_runtime_dependency_references,
    lock_runtime_dependencies,
    lock_runtime_task_for_transition,
)

logger = logging.getLogger(__name__)

_RUN_TOKEN_UNSET = object()


def _is_successful_result(value: Any) -> bool:
    return isinstance(value, dict) and "error" not in value


def _merge_evaluation_results(existing: Any, incoming: Any) -> Dict[str, Any]:
    """Merge a result snapshot without allowing an error to erase success."""
    merged = deepcopy(existing) if isinstance(existing, dict) else {}
    if incoming is None:
        return merged
    if not isinstance(incoming, dict):
        raise ValueError("Evaluation results must be an object")

    for model_key, incoming_model_results in incoming.items():
        if not isinstance(incoming_model_results, dict):
            merged[model_key] = deepcopy(incoming_model_results)
            continue
        existing_model_results = merged.get(model_key)
        if not isinstance(existing_model_results, dict):
            existing_model_results = {}
            merged[model_key] = existing_model_results
        for dataset_key, incoming_result in incoming_model_results.items():
            existing_result = existing_model_results.get(dataset_key)
            if _is_successful_result(existing_result) and not _is_successful_result(
                incoming_result
            ):
                continue
            existing_model_results[dataset_key] = deepcopy(incoming_result)
    return merged


def _overall_progress(model_progress: Any) -> float:
    if not isinstance(model_progress, dict):
        return 0.0
    pair_progress = []
    for model_data in model_progress.values():
        if not isinstance(model_data, dict):
            continue
        for dataset_data in model_data.values():
            if isinstance(dataset_data, dict):
                value = dataset_data.get("progress", 0)
                if isinstance(value, (int, float)):
                    pair_progress.append(float(value))
    return sum(pair_progress) / len(pair_progress) if pair_progress else 0.0


def _reconcile_model_progress(results: Any, model_progress: Any) -> Dict[str, Any]:
    """Derive resumable progress solely from durable successful results."""
    result_matrix = results if isinstance(results, dict) else {}
    progress_matrix = model_progress if isinstance(model_progress, dict) else {}
    reconciled: Dict[str, Dict[str, Any]] = {}
    for model_key in set(progress_matrix) | set(result_matrix):
        stored_progress = progress_matrix.get(model_key, {})
        stored_results = result_matrix.get(model_key, {})
        if not isinstance(stored_progress, dict):
            stored_progress = {}
        if not isinstance(stored_results, dict):
            stored_results = {}
        dataset_keys = set(stored_progress) | {
            key for key in stored_results if key != "_error"
        }
        reconciled[model_key] = {}
        for dataset_key in dataset_keys:
            completed = _is_successful_result(stored_results.get(dataset_key))
            reconciled[model_key][dataset_key] = {
                "progress": 100 if completed else 0,
                "status": "completed" if completed else "pending",
            }
    return reconciled


def _evaluation_dataset_references(
    configs: Any,
) -> tuple[set[str], set[str]]:
    """Collect only local/registered references, never MTEB payload fields."""
    dataset_ids = set()
    dataset_paths = set()
    if isinstance(configs, list):
        for config in configs:
            if not isinstance(config, dict):
                continue
            dataset_type = config.get("type")
            if isinstance(dataset_type, str):
                normalized_type = dataset_type.strip().lower()
                if normalized_type == "mteb":
                    continue
            dataset_id = config.get("dataset_id")
            if isinstance(dataset_id, str) and dataset_id.strip():
                dataset_ids.add(dataset_id.strip())
            path = config.get("path")
            if isinstance(path, str) and path.strip():
                dataset_paths.add(path.strip())
    return dataset_ids, dataset_paths


def _persisted_evaluation_dataset_references(
    task: EvaluationTaskDB,
) -> tuple[set[str], set[str]]:
    return _evaluation_dataset_references(task.dataset_configs or [])


class EvaluationTaskService:
    """Service for evaluation task database operations."""

    def create_task(
        self,
        task_name: Optional[str] = None,
        description: Optional[str] = None,
        eval_type: str = "single",
        model_configs: Optional[List[Dict[str, Any]]] = None,
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        max_samples: Optional[int] = None,
        batch_size: int = 50,
        workers: int = 8,
        model_workers: int = 2,
        user_id: Optional[str] = None,
        require_managed_datasets: bool = False,
    ) -> Dict[str, Any]:
        """Create a new evaluation task. Returns dict with task info."""
        with get_session() as session:
            task = EvaluationTaskDB(
                task_name=task_name,
                description=description,
                eval_type=eval_type,
                eval_framework=EvaluationFramework.MTEB,
                model_configs=model_configs,
                dataset_configs=dataset_configs,
                max_samples=max_samples,
                batch_size=batch_size,
                workers=workers,
                model_workers=model_workers,
                user_id=user_id,
                status="pending",
                run_token=str(uuid4()),
            )
            lock_runtime_dependencies(
                session,
                evaluation_runtime_dependency_references(task),
            )
            dataset_ids, dataset_paths = _persisted_evaluation_dataset_references(task)
            lock_datasets_for_consumption(
                session,
                dataset_ids=dataset_ids,
                storage_refs=dataset_paths,
                require_all_dataset_ids=True,
                require_all_storage_refs=require_managed_datasets,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            logger.info(f"Created evaluation task: {task.task_id}")
            return task.to_dict()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation task by task_id."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return None
            if task.eval_framework != EvaluationFramework.MTEB:
                return None
            return task.to_dict()

    def list_active_dataset_consumers(
        self,
        dataset_ids: List[str],
        dataset_paths: List[str],
    ) -> List[str]:
        """Return active MTEB or DeepEval tasks using any candidate dataset."""
        requested_ids = {
            value.strip()
            for value in dataset_ids
            if isinstance(value, str) and value.strip()
        }
        requested_paths = {
            value.strip()
            for value in dataset_paths
            if isinstance(value, str) and value.strip()
        }
        if not requested_ids and not requested_paths:
            return []
        executing_task_ids = (
            background_task_admission_service.get_executing_task_ids("evaluation")
        )
        with get_session() as session:
            tasks = session.exec(
                select(EvaluationTaskDB).where(
                    or_(
                        EvaluationTaskDB.status.in_(
                            (EvaluationStatus.PENDING, EvaluationStatus.RUNNING)
                        ),
                        EvaluationTaskDB.task_id.in_(executing_task_ids),
                    )
                )
            ).all()
            consumers = []
            for task in tasks:
                task_ids, task_paths = _persisted_evaluation_dataset_references(task)
                if task_ids & requested_ids or storage_reference_sets_overlap(
                    task_paths,
                    requested_paths,
                ):
                    consumers.append(task.task_id)
            return consumers

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get all evaluation tasks with optional filters."""
        with get_session() as session:
            query = select(EvaluationTaskDB).where(
                EvaluationTaskDB.eval_framework == EvaluationFramework.MTEB
            )
            count_query = select(func.count()).select_from(EvaluationTaskDB).where(
                EvaluationTaskDB.eval_framework == EvaluationFramework.MTEB
            )

            if status:
                query = query.where(EvaluationTaskDB.status == status)
                count_query = count_query.where(EvaluationTaskDB.status == status)
            if user_id:
                query = query.where(EvaluationTaskDB.user_id == user_id)
                count_query = count_query.where(EvaluationTaskDB.user_id == user_id)

            total = session.exec(count_query).one()

            query = query.order_by(EvaluationTaskDB.created_at.desc())
            query = query.offset(offset).limit(limit)

            tasks = session.exec(query).all()
            return [t.to_dict() for t in tasks], total

    def claim_running(self, task_id: str) -> bool:
        """Atomically claim a pending MTEB task for worker execution."""
        now = now_naive()
        legacy_run_token = str(uuid4())
        with get_session() as session:
            conditions = (
                EvaluationTaskDB.task_id == task_id,
                EvaluationTaskDB.eval_framework == EvaluationFramework.MTEB,
                EvaluationTaskDB.status == EvaluationStatus.PENDING,
            )
            try:
                task = lock_runtime_task_for_transition(
                    session,
                    entity=EvaluationTaskDB,
                    conditions=conditions,
                    reference_parser=evaluation_runtime_dependency_references,
                )
            except RuntimeDependencyUnavailableError:
                session.rollback()
                return False
            if task is None:
                return False
            result = session.exec(
                update(EvaluationTaskDB)
                .where(*conditions)
                .values(
                    status=EvaluationStatus.RUNNING,
                    run_token=func.coalesce(
                        EvaluationTaskDB.run_token,
                        legacy_run_token,
                    ),
                    started_at=func.coalesce(EvaluationTaskDB.started_at, now),
                    completed_at=None,
                    updated_at=now,
                )
            )
            session.commit()
            return result.rowcount == 1

    def cancel_task(self, task_id: str) -> bool:
        """Atomically cancel an active MTEB task."""
        now = now_naive()
        with get_session() as session:
            result = session.exec(
                update(EvaluationTaskDB)
                .where(
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.MTEB,
                    EvaluationTaskDB.status.in_(
                        (EvaluationStatus.PENDING, EvaluationStatus.RUNNING)
                    ),
                )
                .values(
                    status=EvaluationStatus.CANCELLED,
                    completed_at=now,
                    updated_at=now,
                )
            )
            session.commit()
            return result.rowcount == 1

    def update_status(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None,
        *,
        run_token: Any = _RUN_TOKEN_UNSET,
    ) -> bool:
        """Update evaluation task status."""
        with get_session() as session:
            if status == EvaluationStatus.RUNNING:
                now = now_naive()
                conditions = [
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.MTEB,
                    EvaluationTaskDB.status.in_(
                        (EvaluationStatus.PENDING, EvaluationStatus.RUNNING)
                    ),
                ]
                if run_token is not _RUN_TOKEN_UNSET:
                    conditions.append(EvaluationTaskDB.run_token == run_token)
                try:
                    task = lock_runtime_task_for_transition(
                        session,
                        entity=EvaluationTaskDB,
                        conditions=tuple(conditions),
                        reference_parser=(
                            evaluation_runtime_dependency_references
                        ),
                    )
                except RuntimeDependencyUnavailableError:
                    session.rollback()
                    return False
                if task is None:
                    return False
                result = session.exec(
                    update(EvaluationTaskDB)
                    .where(*conditions)
                    .values(
                        status=EvaluationStatus.RUNNING,
                        started_at=func.coalesce(EvaluationTaskDB.started_at, now),
                        updated_at=now,
                    )
                )
                session.commit()
                return result.rowcount == 1

            conditions = [
                EvaluationTaskDB.task_id == task_id,
                EvaluationTaskDB.eval_framework == EvaluationFramework.MTEB,
            ]
            if run_token is not _RUN_TOKEN_UNSET:
                conditions.append(EvaluationTaskDB.run_token == run_token)
            task = session.exec(
                select(EvaluationTaskDB)
                .where(*conditions)
                .with_for_update()
            ).first()
            if not task:
                return False
            task.update_status(status, error_message)
            session.add(task)
            session.commit()
            return True

    def update_progress(
        self,
        task_id: str,
        progress: float,
        current_model: Optional[str] = None,
        current_dataset: Optional[str] = None,
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Update evaluation task progress."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False
            if task.eval_framework != EvaluationFramework.MTEB:
                return False
            if (
                task.status != EvaluationStatus.RUNNING
                or task.run_token != run_token
            ):
                return False
            task.update_progress(progress, current_model, current_dataset)
            session.add(task)
            session.commit()
            return True

    def update_model_progress(
        self,
        task_id: str,
        model_name: str,
        dataset_name: str,
        progress: float,
        status: str = "running",
        *,
        current_model: Optional[str] = None,
        current_dataset: Optional[str] = None,
        run_token: Optional[str] = None,
    ) -> bool:
        """Update progress for a specific model-dataset pair."""
        with get_session() as session:
            # Use FOR UPDATE to lock the row and prevent concurrent modifications
            task = session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False
            if task.eval_framework != EvaluationFramework.MTEB:
                return False
            if (
                task.status != EvaluationStatus.RUNNING
                or task.run_token != run_token
            ):
                return False
            durable_result = (
                task.results.get(model_name, {}).get(dataset_name)
                if isinstance(task.results, dict)
                and isinstance(task.results.get(model_name), dict)
                else None
            )
            if _is_successful_result(durable_result):
                progress = 100
                status = "completed"
            task.update_model_progress(
                model_name,
                dataset_name,
                progress,
                status,
                current_model=current_model,
                current_dataset=current_dataset,
            )
            # Mark JSON field as modified for SQLAlchemy to detect the change
            flag_modified(task, "model_progress")
            task.progress = _overall_progress(task.model_progress)
            session.add(task)
            session.commit()
            return True

    def init_model_progress(
        self,
        task_id: str,
        model_names: List[str],
        dataset_names: List[str],
        preserve_completed: bool = False,
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Initialize model_progress matrix with all pending entries.

        Args:
            preserve_completed: If True, keep existing completed entries
                instead of resetting them to pending.
        """
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False
            if task.eval_framework != EvaluationFramework.MTEB:
                return False
            if (
                task.status != EvaluationStatus.RUNNING
                or task.run_token != run_token
            ):
                return False
            existing_results = task.results if isinstance(task.results, dict) else {}
            new_progress = {}
            for model in model_names:
                new_progress[model] = {}
                for dataset in dataset_names:
                    result = existing_results.get(model, {}).get(dataset)
                    completed = preserve_completed and _is_successful_result(result)
                    new_progress[model][dataset] = {
                        "progress": 100 if completed else 0,
                        "status": "completed" if completed else "pending",
                    }
            task.model_progress = new_progress
            # 重新计算 overall progress
            task.progress = _overall_progress(new_progress)
            flag_modified(task, "model_progress")
            session.add(task)
            session.commit()
            return True

    def reset_for_resume(
        self,
        task_id: str,
        *,
        require_managed_datasets: bool = False,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        results: Optional[Dict[str, Any]] = None,
        model_progress: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Atomically persist a resume snapshot and move it back to pending."""
        with get_session() as session:
            conditions = (
                EvaluationTaskDB.task_id == task_id,
                EvaluationTaskDB.eval_framework == EvaluationFramework.MTEB,
                EvaluationTaskDB.status.in_(
                    (
                        EvaluationStatus.FAILED,
                        EvaluationStatus.CANCELLED,
                    )
                ),
            )

            def resume_references(candidate: EvaluationTaskDB):
                return evaluation_runtime_dependency_references(
                    {
                        "eval_framework": EvaluationFramework.MTEB,
                        "model_configs": (
                            candidate.model_configs
                            if model_configs is None
                            else model_configs
                        ),
                    }
                )

            try:
                task = lock_runtime_task_for_transition(
                    session,
                    entity=EvaluationTaskDB,
                    conditions=conditions,
                    reference_parser=evaluation_runtime_dependency_references,
                    dependency_reference_builder=resume_references,
                )
            except RuntimeDependencyUnavailableError:
                session.rollback()
                return False
            if task is None:
                return False
            snapshot_model_configs = deepcopy(
                task.model_configs if model_configs is None else model_configs
            )
            snapshot_dataset_configs = deepcopy(
                task.dataset_configs if dataset_configs is None else dataset_configs
            )
            snapshot_results = deepcopy(task.results if results is None else results)
            snapshot_model_progress = deepcopy(
                task.model_progress if model_progress is None else model_progress
            )
            dataset_ids, dataset_paths = _evaluation_dataset_references(
                snapshot_dataset_configs
            )
            try:
                lock_datasets_for_consumption(
                    session,
                    dataset_ids=dataset_ids,
                    storage_refs=dataset_paths,
                    require_all_dataset_ids=True,
                    require_all_storage_refs=require_managed_datasets,
                )
            except DatasetConsumptionUnavailableError:
                return False
            # A cancelled worker may commit after the route reads its resume
            # snapshot. Preserve only canonical successes present in the
            # supplied matrix, without reintroducing raw legacy aliases.
            if isinstance(snapshot_model_progress, dict):
                late_successes: Dict[str, Dict[str, Any]] = {}
                current_results = task.results if isinstance(task.results, dict) else {}
                for model_key, datasets in snapshot_model_progress.items():
                    if not isinstance(datasets, dict):
                        continue
                    current_model_results = current_results.get(model_key, {})
                    if not isinstance(current_model_results, dict):
                        continue
                    for dataset_key in datasets:
                        candidate = current_model_results.get(dataset_key)
                        if _is_successful_result(candidate):
                            late_successes.setdefault(model_key, {})[
                                dataset_key
                            ] = candidate
                if late_successes:
                    snapshot_results = _merge_evaluation_results(
                        snapshot_results,
                        late_successes,
                    )
            model_progress_to_persist = _reconcile_model_progress(
                snapshot_results,
                snapshot_model_progress,
            )
            # 保留已完成条目，仅重置未完成的，从已完成条目计算初始进度
            if model_progress_to_persist:
                total_pairs = 0
                completed_progress = 0.0
                for model_data in model_progress_to_persist.values():
                    for ds_data in model_data.values():
                        total_pairs += 1
                        if ds_data.get("status") == "completed":
                            completed_progress += ds_data.get("progress", 100)
                        else:
                            ds_data["progress"] = 0
                            ds_data["status"] = "pending"
                progress = completed_progress / total_pairs if total_pairs > 0 else 0.0
            else:
                progress = 0.0

            task.status = EvaluationStatus.PENDING
            task.run_token = str(uuid4())
            task.error_message = None
            task.current_model = None
            task.current_dataset = None
            task.started_at = None
            task.completed_at = None
            task.updated_at = now_naive()
            task.model_configs = snapshot_model_configs
            task.dataset_configs = snapshot_dataset_configs
            task.results = snapshot_results
            task.model_progress = model_progress_to_persist
            task.progress = progress
            session.add(task)
            session.commit()
            return True

    def complete_task(
        self,
        task_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None,
        report_path: Optional[str] = None,
        error_message: Optional[str] = None,
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Atomically complete a running MTEB task with its final results."""
        if status not in {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED}:
            raise ValueError("MTEB completion status must be succeeded or failed")

        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if (
                not task
                or task.eval_framework != EvaluationFramework.MTEB
                or task.status != EvaluationStatus.RUNNING
                or task.run_token != run_token
            ):
                return False

            now = now_naive()
            task.status = status
            task.updated_at = now
            task.completed_at = now
            if results is not None:
                task.results = _merge_evaluation_results(task.results, results)
                flag_modified(task, "results")
            if report_path is not None:
                task.results_path = report_path
            if error_message is not None:
                task.error_message = error_message
            elif status == EvaluationStatus.SUCCEEDED:
                task.error_message = None
                task.progress = 100.0
            session.add(task)
            session.commit()
            return True

    def save_model_dataset_result(
        self,
        task_id: str,
        model_name: str,
        dataset_name: str,
        result: Dict[str, Any],
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Commit one successful result and completed progress in one transaction."""
        if not _is_successful_result(result):
            raise ValueError("Only successful pair results can be durably committed")

        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if (
                not task
                or task.eval_framework != EvaluationFramework.MTEB
                or task.status != EvaluationStatus.RUNNING
                or task.run_token != run_token
            ):
                return False

            durable_results = (
                deepcopy(task.results) if isinstance(task.results, dict) else {}
            )
            model_results = durable_results.setdefault(model_name, {})
            if not isinstance(model_results, dict):
                model_results = {}
                durable_results[model_name] = model_results
            model_results[dataset_name] = deepcopy(result)

            durable_progress = (
                deepcopy(task.model_progress)
                if isinstance(task.model_progress, dict)
                else {}
            )
            model_progress = durable_progress.setdefault(model_name, {})
            if not isinstance(model_progress, dict):
                model_progress = {}
                durable_progress[model_name] = model_progress
            model_progress[dataset_name] = {
                "progress": 100,
                "status": "completed",
            }

            task.results = durable_results
            task.model_progress = durable_progress
            task.progress = _overall_progress(durable_progress)
            task.updated_at = now_naive()
            flag_modified(task, "results")
            flag_modified(task, "model_progress")
            session.add(task)
            session.commit()
            return True

    def save_partial_results(
        self,
        task_id: str,
        results: Optional[Dict[str, Any]] = None,
        *,
        run_token: Optional[str] = None,
    ) -> bool:
        """Save partial results without changing task status (e.g., after cancellation)."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if (
                not task
                or task.eval_framework != EvaluationFramework.MTEB
                or task.status
                not in {EvaluationStatus.RUNNING, EvaluationStatus.CANCELLED}
                or task.run_token != run_token
            ):
                return False
            if results is not None:
                task.results = _merge_evaluation_results(task.results, results)
                flag_modified(task, "results")
                task.updated_at = now_naive()
            session.add(task)
            session.commit()
            return True

    def delete_task(self, task_id: str) -> bool:
        """Delete an evaluation task."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return False
            if task.eval_framework != EvaluationFramework.MTEB:
                return False
            session.delete(task)
            session.commit()
            logger.info(f"Deleted evaluation task: {task_id}")
            return True


# Global service instance
evaluation_task_service = EvaluationTaskService()
