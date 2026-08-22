"""
Deep evaluation task service for database operations.

This service stores DeepEval tasks in the unified EvaluationTaskDB table
with eval_framework='deepeval'.
"""

import logging
from copy import deepcopy
from typing import Optional, Dict, Any, List, Tuple

from sqlmodel import select, func
from sqlalchemy import or_, update
from sqlalchemy.orm.attributes import flag_modified

from ...core.time_utils import now_naive
from ..database import get_session
from ..entities.evaluation_task_entity import (
    EvaluationTaskDB,
    EvaluationFramework,
    EvaluationStatus,
)
from .dataset_service import (
    DatasetConsumptionUnavailableError,
    lock_datasets_for_consumption,
)
from .background_task_admission_service import background_task_admission_service
from .evaluation_task_service import _persisted_evaluation_dataset_references
from .milvus_collection_service import (
    MilvusCollectionUnavailableError,
    lock_collection_for_consumption,
)

logger = logging.getLogger(__name__)


def _persisted_evaluation_collection(
    worker_groups: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not isinstance(worker_groups, dict):
        return None
    if worker_groups.get("retrieval_mode") != "online":
        return None
    collection_name = worker_groups.get("milvus_collection")
    if not isinstance(collection_name, str) or not collection_name.strip():
        return None
    return collection_name.strip()


class DeepEvaluationTaskService:
    """Service for deep evaluation task database operations."""

    def create_task(
        self,
        task_name: Optional[str] = None,
        description: Optional[str] = None,
        eval_type: str = "multi",
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        max_samples: Optional[int] = None,
        field_mapping: Optional[Dict[str, Any]] = None,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
        worker_groups: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        require_managed_datasets: bool = False,
    ) -> Dict[str, Any]:
        """Create a new deep evaluation task."""
        with get_session() as session:
            total_workers = 0
            if worker_groups:
                for group in worker_groups.values():
                    try:
                        total_workers += int(group.get("workers", 0))
                    except Exception:
                        continue

            task = EvaluationTaskDB(
                task_name=task_name,
                description=description,
                eval_framework=EvaluationFramework.DEEPEVAL,
                eval_type=eval_type,
                dataset_configs=dataset_configs,
                model_configs=model_configs,
                field_mapping=field_mapping,
                metrics=metrics or ["mrr", "ndcg@10"],
                llm_config=llm_config,
                worker_groups=worker_groups,
                max_samples=max_samples,
                workers=total_workers or 1,
                status=EvaluationStatus.PENDING,
                user_id=user_id,
            )
            dataset_ids, dataset_paths = _persisted_evaluation_dataset_references(task)
            lock_datasets_for_consumption(
                session,
                dataset_ids=dataset_ids,
                storage_refs=dataset_paths,
                require_all_dataset_ids=True,
                require_all_storage_refs=require_managed_datasets,
            )
            collection_name = _persisted_evaluation_collection(worker_groups)
            if collection_name:
                lock_collection_for_consumption(
                    session,
                    collection_name,
                    user_id=user_id,
                )
            session.add(task)
            session.commit()
            session.refresh(task)
            logger.info("Created deep evaluation task: %s (type=%s)", task.task_id, eval_type)
            return self._to_task_dict(task)

    def get_task(self, task_id: str, include_secrets: bool = False) -> Optional[Dict[str, Any]]:
        """Get deep evaluation task by task_id."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
                )
            ).first()
            return self._to_task_dict(task, mask_api_key=not include_secrets) if task else None

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        eval_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get all deep evaluation tasks with optional filters."""
        with get_session() as session:
            # Only get deepeval tasks
            query = select(EvaluationTaskDB).where(
                EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL
            )
            count_query = select(func.count()).select_from(EvaluationTaskDB).where(
                EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL
            )

            if status:
                query = query.where(EvaluationTaskDB.status == status)
                count_query = count_query.where(EvaluationTaskDB.status == status)
            if eval_type:
                query = query.where(EvaluationTaskDB.eval_type == eval_type)
                count_query = count_query.where(EvaluationTaskDB.eval_type == eval_type)
            if user_id:
                query = query.where(EvaluationTaskDB.user_id == user_id)
                count_query = count_query.where(EvaluationTaskDB.user_id == user_id)

            total = session.exec(count_query).one()
            query = query.order_by(EvaluationTaskDB.created_at.desc())
            query = query.offset(offset).limit(limit)

            tasks = session.exec(query).all()
            return [self._to_task_dict(t) for t in tasks], total

    def list_active_collection_consumers(
        self,
        collection_names: List[str],
        *,
        exclude_task_ids: tuple[str, ...] = (),
    ) -> List[str]:
        """Return active online DeepEval tasks using these collections."""
        normalized_names = {
            name.strip()
            for name in collection_names
            if isinstance(name, str) and name.strip()
        }
        if not normalized_names:
            return []
        executing_task_ids = (
            background_task_admission_service.get_executing_task_ids("evaluation")
        )
        with get_session() as session:
            query = select(EvaluationTaskDB).where(
                EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
                or_(
                    EvaluationTaskDB.status.in_(
                        (EvaluationStatus.PENDING, EvaluationStatus.RUNNING)
                    ),
                    EvaluationTaskDB.task_id.in_(executing_task_ids),
                ),
            )
            if exclude_task_ids:
                query = query.where(
                    EvaluationTaskDB.task_id.notin_(exclude_task_ids)
                )
            tasks = session.exec(
                query.order_by(EvaluationTaskDB.task_id)
            ).all()
            return [
                task.task_id
                for task in tasks
                if _persisted_evaluation_collection(task.worker_groups)
                in normalized_names
            ]

    def claim_running(self, task_id: str) -> bool:
        """Atomically claim a pending DeepEval task for worker execution."""
        now = now_naive()
        with get_session() as session:
            result = session.exec(
                update(EvaluationTaskDB)
                .where(
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
                    EvaluationTaskDB.status == EvaluationStatus.PENDING,
                )
                .values(
                    status=EvaluationStatus.RUNNING,
                    started_at=func.coalesce(EvaluationTaskDB.started_at, now),
                    completed_at=None,
                    updated_at=now,
                )
            )
            session.commit()
            return result.rowcount == 1

    def cancel_task(self, task_id: str) -> bool:
        """Atomically cancel an active DeepEval task."""
        now = now_naive()
        with get_session() as session:
            result = session.exec(
                update(EvaluationTaskDB)
                .where(
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
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
        self, task_id: str, status: str, error_message: Optional[str] = None
    ) -> bool:
        """Update task status."""
        with get_session() as session:
            if status == EvaluationStatus.RUNNING:
                now = now_naive()
                result = session.exec(
                    update(EvaluationTaskDB)
                    .where(
                        EvaluationTaskDB.task_id == task_id,
                        EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
                        EvaluationTaskDB.status.in_(
                            (EvaluationStatus.PENDING, EvaluationStatus.RUNNING)
                        ),
                    )
                    .values(
                        status=EvaluationStatus.RUNNING,
                        started_at=func.coalesce(EvaluationTaskDB.started_at, now),
                        updated_at=now,
                    )
                )
                session.commit()
                return result.rowcount == 1

            task = session.exec(
                select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task_id)
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
        processed: int,
        total: Optional[int] = None,
    ) -> bool:
        """Update task progress."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return False
            task.update_progress(
                processed_samples=processed,
                total_samples=total,
            )
            session.add(task)
            session.commit()
            return True

    def update_results(
        self,
        task_id: str,
        results_summary: Optional[Dict[str, Any]] = None,
        results_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update results for a task."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return False
            if results_summary is not None:
                task.results = results_summary
                flag_modified(task, "results")
            if results_path is not None:
                task.results_path = results_path
            if error_message:
                task.error_message = error_message
            session.add(task)
            session.commit()
            return True

    def update_model_progress(
        self,
        task_id: str,
        group_name: str,
        dataset_name: str,
        progress: float,
        status: str = "running",
    ) -> bool:
        """Update progress for a specific model group and dataset pair.

        Args:
            task_id: Task ID
            group_name: Model group name (e.g., "vLLM 本地", "Xinference")
            dataset_name: Dataset name or ID
            progress: Progress percentage (0-100)
            status: Status string (pending, running, completed, failed)

        Returns:
            True if update succeeded, False otherwise
        """
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB)
                .where(EvaluationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False

            task.update_model_progress(group_name, dataset_name, progress, status)
            flag_modified(task, "model_progress")

            # Calculate overall progress from model_progress matrix
            if task.model_progress:
                total_pairs = 0
                total_progress = 0.0
                for group_data in task.model_progress.values():
                    for ds_data in group_data.values():
                        total_pairs += 1
                        total_progress += ds_data.get("progress", 0)
                if total_pairs > 0:
                    task.progress = total_progress / total_pairs

            session.add(task)
            session.commit()
            return True

    def init_model_progress(
        self,
        task_id: str,
        group_names: List[str],
        dataset_names: List[str],
        preserve_completed: bool = False,
    ) -> bool:
        """Initialize model_progress matrix for all group-dataset pairs.

        Args:
            task_id: Task ID
            group_names: List of model group names
            dataset_names: List of dataset names
            preserve_completed: If True, keep existing completed entries
                instead of resetting them to pending.

        Returns:
            True if initialization succeeded, False otherwise
        """
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return False

            existing = (task.model_progress or {}) if preserve_completed else {}
            model_progress = {}
            for group_name in group_names:
                model_progress[group_name] = {}
                for dataset_name in dataset_names:
                    old = existing.get(group_name, {}).get(dataset_name)
                    if preserve_completed and old and old.get("status") == "completed":
                        model_progress[group_name][dataset_name] = old
                    else:
                        model_progress[group_name][dataset_name] = {
                            "progress": 0,
                            "status": "pending",
                        }

            task.model_progress = model_progress
            total_pairs = sum(len(d) for d in model_progress.values())
            total_progress = sum(
                ds.get("progress", 0) for gd in model_progress.values() for ds in gd.values()
            )
            task.progress = total_progress / total_pairs if total_pairs > 0 else 0.0
            flag_modified(task, "model_progress")
            session.add(task)
            session.commit()
            return True

    def reset_for_resume(
        self,
        task_id: str,
        *,
        require_managed_datasets: bool = False,
    ) -> bool:
        """Reset task state for resuming execution.

        Preserves existing results, results_path and completed model_progress
        entries so that the runner can skip already-completed model groups.
        """
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
                )
            ).first()
            if not task:
                return False
            dataset_ids, dataset_paths = _persisted_evaluation_dataset_references(task)
            try:
                lock_datasets_for_consumption(
                    session,
                    dataset_ids=dataset_ids,
                    storage_refs=dataset_paths,
                    require_all_dataset_ids=True,
                    require_all_storage_refs=require_managed_datasets,
                )
                collection_name = _persisted_evaluation_collection(
                    task.worker_groups
                )
                if collection_name:
                    lock_collection_for_consumption(
                        session,
                        collection_name,
                        user_id=task.user_id,
                    )
            except (
                DatasetConsumptionUnavailableError,
                MilvusCollectionUnavailableError,
            ):
                return False

            model_progress = deepcopy(task.model_progress)
            # 保留 results / results_path，用于 resume 时跳过已完成的模型组
            # 保留 model_progress 中已完成的组，重置其他
            if model_progress:
                for group_data in model_progress.values():
                    for ds_data in group_data.values():
                        if ds_data.get("status") != "completed":
                            ds_data["progress"] = 0
                            ds_data["status"] = "pending"
                total_pairs = sum(len(d) for d in model_progress.values())
                completed_progress = sum(
                    ds.get("progress", 0)
                    for gd in model_progress.values()
                    for ds in gd.values()
                    if ds.get("status") == "completed"
                )
                progress = completed_progress / total_pairs if total_pairs > 0 else 0.0
            else:
                progress = 0.0

            result = session.exec(
                update(EvaluationTaskDB)
                .where(
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
                    EvaluationTaskDB.status.in_(
                        (EvaluationStatus.FAILED, EvaluationStatus.CANCELLED)
                    ),
                )
                .values(
                    status=EvaluationStatus.PENDING,
                    error_message=None,
                    started_at=None,
                    completed_at=None,
                    updated_at=now_naive(),
                    model_progress=model_progress,
                    progress=progress,
                )
            )
            session.commit()
            return result.rowcount == 1

    def complete_task(
        self,
        task_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None,
        report_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Atomically complete a running DeepEval task with its final results."""
        if status not in {EvaluationStatus.COMPLETED, EvaluationStatus.FAILED}:
            raise ValueError("DeepEval completion status must be completed or failed")

        now = now_naive()
        values: Dict[str, Any] = {
            "status": status,
            "updated_at": now,
            "completed_at": now,
        }
        if results is not None:
            values["results"] = results
        if report_path is not None:
            values["results_path"] = report_path
        if error_message is not None:
            values["error_message"] = error_message
        elif status == EvaluationStatus.COMPLETED:
            values["error_message"] = None
            values["progress"] = 100.0

        with get_session() as session:
            result = session.exec(
                update(EvaluationTaskDB)
                .where(
                    EvaluationTaskDB.task_id == task_id,
                    EvaluationTaskDB.eval_framework == EvaluationFramework.DEEPEVAL,
                    EvaluationTaskDB.status == EvaluationStatus.RUNNING,
                )
                .values(**values)
            )
            session.commit()
            return result.rowcount == 1

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by task_id."""
        with get_session() as session:
            task = session.exec(
                select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return False
            session.delete(task)
            session.commit()
            logger.info("Deleted deep evaluation task: %s", task_id)
            return True

    def _to_task_dict(self, task: EvaluationTaskDB, mask_api_key: bool = True) -> Dict[str, Any]:
        """Convert unified entity to deep evaluation API response."""
        def mask_api_keys(value: Any) -> Any:
            if isinstance(value, dict):
                masked = {}
                for key, item in value.items():
                    if key == "api_key":
                        masked[key] = "***" if item else None
                    else:
                        masked[key] = mask_api_keys(item)
                return masked
            if isinstance(value, list):
                return [mask_api_keys(item) for item in value]
            return value

        model_configs = task.model_configs or []
        if mask_api_key and model_configs:
            model_configs = mask_api_keys(model_configs)

        llm_config = task.llm_config
        if mask_api_key and llm_config:
            llm_config = mask_api_keys(llm_config)

        worker_groups = task.worker_groups
        if mask_api_key and worker_groups:
            worker_groups = mask_api_keys(worker_groups)

        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "description": task.description,
            "eval_type": task.eval_type,
            "model_configs": model_configs,
            "dataset_configs": task.dataset_configs or [],
            "field_mapping": task.field_mapping,
            "metrics": task.metrics,
            "worker_groups": worker_groups,
            "llm_config": llm_config,
            "max_samples": task.max_samples,
            "status": task.status,
            "progress": task.progress,
            "total_samples": task.total_samples,
            "processed_samples": task.processed_samples,
            "model_progress": task.model_progress,
            "results_summary": task.results,
            "results_path": task.results_path,
            "error_message": task.error_message,
            "user_id": task.user_id,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }


deep_evaluation_task_service = DeepEvaluationTaskService()
