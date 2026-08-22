"""
Generation task service for database operations.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from sqlmodel import select, func
from sqlalchemy import and_, delete, or_, update

from ...core.time_utils import now_naive
from ..database import get_session
from ..entities.dataset_asset_entity import DatasetAssetDB
from ..entities.dataset_entity import DatasetDB
from ..entities.dataset_lineage_entity import DatasetLineageEdgeDB
from ..entities.external_sync_entity import ExternalSyncGenerationDB
from ..entities.generation_task_entity import GenerationTaskDB, GenerationStatus
from .dataset_service import (
    DatasetConsumptionUnavailableError,
    DatasetDeletionOwnerConflictError,
    _storage_reference_identity,
    _storage_reference_identities_overlap,
    lock_datasets_for_consumption,
    require_dataset_deletion_owner,
)
from .background_task_admission_service import background_task_admission_service
from .milvus_collection_service import (
    MilvusCollectionUnavailableError,
    lock_collection_for_consumption,
)

logger = logging.getLogger(__name__)
_RUN_TOKEN_UNSET = object()
GENERATION_RESTART_RECOVERY_MARKER = (
    "Generation restart recovery owns checkpoint cleanup"
)


@dataclass(frozen=True)
class GenerationTaskDeletionIntent:
    """Describe a durable generation-task deletion transition."""

    task: Dict[str, Any]
    newly_started: bool
    previous_status: Optional[str]


class GenerationTaskService:
    """Service for generation task database operations."""

    def create_task(
        self,
        task_name: str,
        input_path: str,
        llm_config: Dict[str, Any],
        steps_config: Dict[str, Any],
        input_format: str = "auto",
        output_format: str = "universal",
        generation_mode: str = "doc_to_training",
        pos_neg_method: str = "retrieval",
        output_path: Optional[str] = None,
        eval_llm_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        rerank_config: Optional[Dict[str, Any]] = None,
        worker_config: Optional[Dict[str, Any]] = None,
        post_process_config: Optional[Dict[str, Any]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        content_field: Optional[str] = None,
        auto_register_dataset: bool = True,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        # QA-to-Training specific
        source_dataset_id: Optional[str] = None,
        embedding_config_id: Optional[str] = None,
        similarity_threshold: float = 0.85,
        retrieval_top_k: int = 10,
        milvus_collection: Optional[str] = None,
        task_id: Optional[str] = None,
        require_source_dataset: bool = False,
    ) -> Dict[str, Any]:
        """Create a new generation task."""
        with get_session() as session:
            lock_datasets_for_consumption(
                session,
                dataset_ids=(source_dataset_id,) if source_dataset_id else (),
                require_all_dataset_ids=require_source_dataset,
            )
            if milvus_collection:
                lock_collection_for_consumption(
                    session,
                    milvus_collection,
                    user_id=user_id,
                )
            task = GenerationTaskDB(
                task_id=task_id or str(uuid.uuid4()),
                task_name=task_name,
                description=description,
                input_path=input_path,
                input_format=input_format,
                content_field=content_field,
                generation_mode=generation_mode,
                pos_neg_method=pos_neg_method,
                output_path=output_path,
                output_format=output_format,
                llm_config=llm_config,
                eval_llm_config=eval_llm_config,
                embedding_config=embedding_config,
                rerank_config=rerank_config,
                worker_config=worker_config or {"concurrency": 10, "timeout_per_doc": 300},
                steps_config=steps_config,
                post_process_config=post_process_config,
                custom_prompts=custom_prompts,
                auto_register_dataset=auto_register_dataset,
                source_dataset_id=source_dataset_id,
                embedding_config_id=embedding_config_id,
                milvus_collection=milvus_collection,
                similarity_threshold=similarity_threshold,
                retrieval_top_k=retrieval_top_k,
                status=GenerationStatus.PENDING,
                run_token=str(uuid.uuid4()),
                user_id=user_id,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            logger.info("Created generation task: %s", task.task_id)
            return task.to_dict()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get generation task by task_id."""
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
            ).first()
            return task.to_dict() if task else None

    def get_task_raw(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get private attempt state as a detached-safe dict."""
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return None
            raw_task = task.to_dict()
            raw_task["run_token"] = task.run_token
            return raw_task

    def list_active_dataset_consumers(
        self,
        dataset_ids: List[str],
        *,
        exclude_task_ids: tuple[str, ...] = (),
    ) -> List[str]:
        """Return active generation tasks that still consume these datasets."""
        if not dataset_ids:
            return []
        executing_task_ids = (
            background_task_admission_service.get_executing_task_ids("generation")
        )
        with get_session() as session:
            query = select(GenerationTaskDB.task_id).where(
                or_(
                    GenerationTaskDB.source_dataset_id.in_(dataset_ids),
                    and_(
                        GenerationTaskDB.status == GenerationStatus.RESTARTING,
                        or_(
                            GenerationTaskDB.output_dataset_id.in_(dataset_ids),
                            GenerationTaskDB.qa_dataset_id.in_(dataset_ids),
                            GenerationTaskDB.qa_filtered_dataset_id.in_(dataset_ids),
                            GenerationTaskDB.deep_eval_dataset_id.in_(dataset_ids),
                        ),
                    ),
                ),
                or_(
                    GenerationTaskDB.status.in_(
                        (
                            GenerationStatus.PENDING,
                            GenerationStatus.RUNNING,
                            GenerationStatus.STOPPING,
                            GenerationStatus.PUBLISHING,
                            GenerationStatus.RECOVERING,
                            GenerationStatus.RESTARTING,
                        )
                    ),
                    GenerationTaskDB.task_id.in_(executing_task_ids),
                ),
            )
            if exclude_task_ids:
                query = query.where(
                    GenerationTaskDB.task_id.notin_(exclude_task_ids)
                )
            return list(session.exec(query).all())

    def list_artifact_reference_consumers(
        self,
        storage_references: List[str] | set[str],
        *,
        exclude_task_ids: tuple[str, ...] = (),
    ) -> List[Dict[str, str]]:
        """Find all-tenant generation tasks that still reference storage targets."""
        target_identities = {
            identity
            for reference in storage_references
            if (identity := _storage_reference_identity(reference)) is not None
        }
        if not target_identities:
            return []

        artifact_fields = (
            "input_path",
            "output_path",
            "qa_output_path",
            "qa_filtered_path",
            "deep_eval_path",
        )
        with get_session() as session:
            query = select(
                GenerationTaskDB.task_id,
                GenerationTaskDB.input_path,
                GenerationTaskDB.output_path,
                GenerationTaskDB.qa_output_path,
                GenerationTaskDB.qa_filtered_path,
                GenerationTaskDB.deep_eval_path,
            )
            if exclude_task_ids:
                query = query.where(
                    GenerationTaskDB.task_id.notin_(exclude_task_ids)
                )
            rows = session.exec(query).all()

        consumers: List[Dict[str, str]] = []
        for row in rows:
            task_id = str(row[0])
            for field, reference in zip(artifact_fields, row[1:]):
                reference_identity = _storage_reference_identity(reference)
                if any(
                    _storage_reference_identities_overlap(
                        reference_identity,
                        target_identity,
                    )
                    for target_identity in target_identities
                ):
                    consumers.append(
                        {
                            "task_id": task_id,
                            "field": field,
                            "reference": str(reference),
                        }
                    )
        return sorted(
            consumers,
            key=lambda item: (
                item["task_id"],
                item["field"],
                item["reference"],
            ),
        )

    def list_active_collection_consumers(
        self,
        collection_names: List[str],
        *,
        exclude_task_ids: tuple[str, ...] = (),
    ) -> List[str]:
        """Return active generation tasks that consume these collections."""
        normalized_names = sorted(
            {
                name.strip()
                for name in collection_names
                if isinstance(name, str) and name.strip()
            }
        )
        if not normalized_names:
            return []
        executing_task_ids = (
            background_task_admission_service.get_executing_task_ids("generation")
        )
        with get_session() as session:
            query = select(GenerationTaskDB.task_id).where(
                GenerationTaskDB.milvus_collection.in_(normalized_names),
                or_(
                    GenerationTaskDB.status.in_(
                        (
                            GenerationStatus.PENDING,
                            GenerationStatus.RUNNING,
                            GenerationStatus.STOPPING,
                            GenerationStatus.PUBLISHING,
                            GenerationStatus.RECOVERING,
                            GenerationStatus.RESTARTING,
                        )
                    ),
                    GenerationTaskDB.task_id.in_(executing_task_ids),
                ),
            )
            if exclude_task_ids:
                query = query.where(
                    GenerationTaskDB.task_id.notin_(exclude_task_ids)
                )
            query = query.order_by(GenerationTaskDB.task_id)
            return list(session.exec(query).all())

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        generation_mode: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get all generation tasks with optional filters."""
        with get_session() as session:
            query = select(GenerationTaskDB)
            count_query = select(func.count()).select_from(GenerationTaskDB)

            if status:
                query = query.where(GenerationTaskDB.status == status)
                count_query = count_query.where(GenerationTaskDB.status == status)
            if generation_mode:
                query = query.where(GenerationTaskDB.generation_mode == generation_mode)
                count_query = count_query.where(GenerationTaskDB.generation_mode == generation_mode)
            if user_id:
                query = query.where(GenerationTaskDB.user_id == user_id)
                count_query = count_query.where(GenerationTaskDB.user_id == user_id)

            total = session.exec(count_query).one()
            query = query.order_by(GenerationTaskDB.created_at.desc())
            query = query.offset(offset).limit(limit)

            tasks = session.exec(query).all()
            return [t.to_dict() for t in tasks], total

    def get_task_stats(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """Return unfiltered status totals for a user's generation tasks."""
        stats = {
            "total": 0,
            "pending": 0,
            "running": 0,
            "stopping": 0,
            "publishing": 0,
            "recovering": 0,
            "restarting": 0,
            "completed": 0,
            "failed": 0,
            "stopped": 0,
        }
        with get_session() as session:
            query = select(GenerationTaskDB.status, func.count()).group_by(
                GenerationTaskDB.status
            )
            if user_id:
                query = query.where(GenerationTaskDB.user_id == user_id)

            for status, count in session.exec(query).all():
                count_value = int(count)
                if status in stats:
                    stats[status] = count_value
                stats["total"] += count_value
        return stats

    def claim_running(
        self,
        task_id: str,
        *,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> Optional[str]:
        """Atomically claim a pending task and return its attempt token."""
        legacy_run_token = str(uuid.uuid4())
        with get_session() as session:
            query = update(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id,
                GenerationTaskDB.status == GenerationStatus.PENDING,
            )
            if expected_run_token is not _RUN_TOKEN_UNSET:
                query = query.where(
                    GenerationTaskDB.run_token == expected_run_token
                )
            result = session.exec(
                query.values(
                    status=GenerationStatus.RUNNING,
                    run_token=func.coalesce(
                        GenerationTaskDB.run_token,
                        legacy_run_token,
                    ),
                    started_at=now_naive(),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                return None
            run_token = session.exec(
                select(GenerationTaskDB.run_token).where(
                    GenerationTaskDB.task_id == task_id
                )
            ).one()
            session.commit()
            if run_token:
                logger.info("Claimed generation task %s for execution", task_id)
            return run_token

    def claim_orphan_recovery(
        self,
        task_id: str,
        *,
        expected_status: str,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> Optional[Dict[str, str]]:
        """Atomically own recovery of one durable orphaned attempt."""
        recoverable_statuses = {
            GenerationStatus.PENDING,
            GenerationStatus.RUNNING,
            GenerationStatus.STOPPING,
            GenerationStatus.PUBLISHING,
            GenerationStatus.RESTARTING,
        }
        if expected_status not in recoverable_statuses:
            raise ValueError(f"Invalid orphan status: {expected_status}")
        legacy_recovery_token = str(uuid.uuid4())
        with get_session() as session:
            query = update(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id,
                GenerationTaskDB.status == expected_status,
            )
            if expected_run_token is not _RUN_TOKEN_UNSET:
                query = query.where(
                    GenerationTaskDB.run_token == expected_run_token
                )
            recovery_values: Dict[str, Any] = {
                "status": GenerationStatus.RECOVERING,
                "run_token": func.coalesce(
                    GenerationTaskDB.run_token,
                    legacy_recovery_token,
                ),
                "completed_at": None,
            }
            if expected_status == GenerationStatus.RESTARTING:
                recovery_values["error_message"] = (
                    GENERATION_RESTART_RECOVERY_MARKER
                )
            result = session.exec(query.values(**recovery_values))
            if result.rowcount != 1:
                session.rollback()
                return None
            recovery_token = session.exec(
                select(GenerationTaskDB.run_token).where(
                    GenerationTaskDB.task_id == task_id
                )
            ).one()
            session.commit()
        if not recovery_token:
            return None
        return {
            "run_token": recovery_token,
            "source_status": expected_status,
        }

    def finish_orphan_recovery(
        self,
        task_id: str,
        *,
        expected_run_token: str,
        terminal_status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """Finalize a recovery claim without exposing an intermediate race."""
        if terminal_status not in {
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        }:
            raise ValueError("Orphan recovery must finish failed or stopped")
        with get_session() as session:
            result = session.exec(
                update(GenerationTaskDB)
                .where(
                    GenerationTaskDB.task_id == task_id,
                    GenerationTaskDB.status == GenerationStatus.RECOVERING,
                    GenerationTaskDB.run_token == expected_run_token,
                )
                .values(
                    status=terminal_status,
                    error_message=error_message,
                    completed_at=now_naive(),
                )
            )
            session.commit()
            return result.rowcount == 1

    def begin_restart(
        self,
        task_id: str,
        *,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
        new_run_token: str,
        require_source_dataset: bool = False,
    ) -> Optional[str]:
        """Reserve one restart attempt while retaining its source checkpoints."""
        try:
            if str(uuid.UUID(new_run_token)) != new_run_token:
                return None
        except (AttributeError, TypeError, ValueError):
            return None

        terminal_statuses = {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        }
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB)
                .where(GenerationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if task is None or task.status not in terminal_statuses:
                return None
            if (
                expected_run_token is not _RUN_TOKEN_UNSET
                and task.run_token != expected_run_token
            ):
                return None
            if task.run_token == new_run_token:
                return None

            artifact_dataset_ids = {
                dataset_id
                for dataset_id in (
                    task.output_dataset_id,
                    task.qa_dataset_id,
                    task.qa_filtered_dataset_id,
                    task.deep_eval_dataset_id,
                )
                if dataset_id
            }
            referenced_dataset_ids = set(artifact_dataset_ids)
            if task.source_dataset_id:
                referenced_dataset_ids.add(task.source_dataset_id)
            lock_conditions = [
                and_(
                    DatasetDB.source_task_type == "generation",
                    DatasetDB.source_task_id == task_id,
                )
            ]
            if referenced_dataset_ids:
                lock_conditions.append(
                    DatasetDB.dataset_id.in_(referenced_dataset_ids)
                )
            locked_datasets = list(
                session.exec(
                    select(DatasetDB)
                    .where(or_(*lock_conditions))
                    .order_by(DatasetDB.dataset_id)
                    .with_for_update()
                ).all()
            )
            locked_ids = {dataset.dataset_id for dataset in locked_datasets}
            if not artifact_dataset_ids.issubset(locked_ids):
                return None
            if (
                require_source_dataset
                and task.source_dataset_id
                and task.source_dataset_id not in locked_ids
            ):
                return None
            if any(
                dataset.status in {"staging", "deleting"}
                or dataset.deletion_owner is not None
                for dataset in locked_datasets
            ):
                return None
            try:
                if task.milvus_collection:
                    lock_collection_for_consumption(
                        session,
                        task.milvus_collection,
                        user_id=task.user_id,
                    )
            except (
                DatasetConsumptionUnavailableError,
                MilvusCollectionUnavailableError,
            ):
                return None

            task.status = GenerationStatus.RESTARTING
            task.run_token = new_run_token
            task.completed_at = None
            session.add(task)
            session.commit()
            logger.info("Reserved generation task %s restart attempt", task_id)
            return new_run_token

    def finish_restart(
        self,
        task_id: str,
        *,
        expected_run_token: str,
        output_path: str,
    ) -> bool:
        """Expose a prepared restart attempt after all checkpoint copies finish."""
        if not output_path:
            return False
        with get_session() as session:
            result = session.exec(
                update(GenerationTaskDB)
                .where(
                    GenerationTaskDB.task_id == task_id,
                    GenerationTaskDB.status == GenerationStatus.RESTARTING,
                    GenerationTaskDB.run_token == expected_run_token,
                )
                .values(
                    status=GenerationStatus.PENDING,
                    progress=0.0,
                    total_docs=0,
                    processed_docs=0,
                    output_sample_count=0,
                    output_path=output_path,
                    output_dataset_id=None,
                    filter_stats=None,
                    qa_output_path=None,
                    qa_dataset_id=None,
                    qa_filtered_path=None,
                    qa_filtered_dataset_id=None,
                    deep_eval_path=None,
                    deep_eval_dataset_id=None,
                    error_message=None,
                    started_at=None,
                    completed_at=None,
                )
            )
            session.commit()
            return result.rowcount == 1

    def fail_restart(
        self,
        task_id: str,
        *,
        expected_run_token: str,
        error_message: str,
    ) -> bool:
        """Fail one restart reservation without dropping its old artifact refs."""
        with get_session() as session:
            result = session.exec(
                update(GenerationTaskDB)
                .where(
                    GenerationTaskDB.task_id == task_id,
                    GenerationTaskDB.status == GenerationStatus.RESTARTING,
                    GenerationTaskDB.run_token == expected_run_token,
                )
                .values(
                    status=GenerationStatus.FAILED,
                    error_message=error_message,
                    completed_at=now_naive(),
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
        require_source_dataset: bool = False,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> bool:
        """Atomically move a task between valid lifecycle states."""
        allowed_sources = {
            GenerationStatus.PENDING: (
                GenerationStatus.COMPLETED,
                GenerationStatus.FAILED,
                GenerationStatus.STOPPED,
            ),
            GenerationStatus.RUNNING: (GenerationStatus.PENDING,),
            GenerationStatus.STOPPING: (GenerationStatus.RUNNING,),
            GenerationStatus.PUBLISHING: (GenerationStatus.RUNNING,),
            GenerationStatus.RECOVERING: (),
            GenerationStatus.COMPLETED: (GenerationStatus.PUBLISHING,),
            GenerationStatus.FAILED: (
                GenerationStatus.PENDING,
                GenerationStatus.RUNNING,
                GenerationStatus.STOPPING,
                GenerationStatus.PUBLISHING,
            ),
            GenerationStatus.STOPPED: (
                GenerationStatus.PENDING,
                GenerationStatus.STOPPING,
                GenerationStatus.RECOVERING,
            ),
            GenerationStatus.DELETING: (
                GenerationStatus.COMPLETED,
                GenerationStatus.FAILED,
                GenerationStatus.STOPPED,
            ),
            GenerationStatus.DELETING_CASCADE: (
                GenerationStatus.COMPLETED,
                GenerationStatus.FAILED,
                GenerationStatus.STOPPED,
            ),
        }
        if status not in allowed_sources:
            raise ValueError(f"Invalid status: {status}")

        with get_session() as session:
            if status == GenerationStatus.PENDING:
                task = session.exec(
                    select(GenerationTaskDB)
                    .where(GenerationTaskDB.task_id == task_id)
                    .with_for_update()
                ).first()
                if not task:
                    return False
                fenced_output = session.exec(
                    select(DatasetDB.dataset_id)
                    .where(
                        DatasetDB.source_task_type == "generation",
                        DatasetDB.source_task_id == task_id,
                        or_(
                            DatasetDB.status == "staging",
                            DatasetDB.status == "deleting",
                            DatasetDB.deletion_owner.is_not(None),
                        ),
                    )
                    .limit(1)
                    .with_for_update()
                ).first()
                if fenced_output is not None:
                    return False
                try:
                    lock_datasets_for_consumption(
                        session,
                        dataset_ids=(task.source_dataset_id,)
                        if task.source_dataset_id
                        else (),
                        require_all_dataset_ids=require_source_dataset,
                    )
                    if task.milvus_collection:
                        lock_collection_for_consumption(
                            session,
                            task.milvus_collection,
                            user_id=task.user_id,
                        )
                except (
                    DatasetConsumptionUnavailableError,
                    MilvusCollectionUnavailableError,
                ):
                    return False
                values: Dict[str, Any] = {
                    "status": GenerationStatus.PENDING,
                    "run_token": str(uuid.uuid4()),
                    "progress": 0.0,
                    "total_docs": 0,
                    "processed_docs": 0,
                    "output_sample_count": 0,
                    "output_path": None,
                    "output_dataset_id": None,
                    "filter_stats": None,
                    "qa_output_path": None,
                    "qa_dataset_id": None,
                    "qa_filtered_path": None,
                    "qa_filtered_dataset_id": None,
                    "deep_eval_path": None,
                    "deep_eval_dataset_id": None,
                    "error_message": None,
                    "started_at": None,
                    "completed_at": None,
                }
            else:
                now = now_naive()
                values = {"status": status}
                if status == GenerationStatus.RUNNING:
                    values.update(started_at=now, completed_at=None)
                elif status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.STOPPED,
                }:
                    values["completed_at"] = now
                if error_message is not None:
                    values["error_message"] = error_message
                elif status == GenerationStatus.COMPLETED:
                    values["error_message"] = None

            query = update(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id,
                GenerationTaskDB.status.in_(allowed_sources[status]),
            )
            if expected_run_token is not _RUN_TOKEN_UNSET:
                query = query.where(
                    GenerationTaskDB.run_token == expected_run_token
                )
            result = session.exec(query.values(**values))
            session.commit()
            if result.rowcount != 1:
                logger.warning(
                    "Generation task %s was not eligible for transition to %s",
                    task_id,
                    status,
                )
                return False
            logger.info("Updated generation task %s status to %s", task_id, status)
            return True

    def complete_publication(
        self,
        task_id: str,
        *,
        expected_run_token: str,
        dataset_bindings: Dict[str, Optional[str]],
        milvus_registration: Optional[Dict[str, Any]] = None,
        artifact_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Delegate the all-or-nothing publication transaction."""
        from .generation_publication_service import GenerationPublicationService

        return GenerationPublicationService(
            session_factory=get_session
        ).complete_publication(
            task_id,
            expected_run_token=expected_run_token,
            dataset_bindings=dataset_bindings,
            milvus_registration=milvus_registration,
            artifact_updates=artifact_updates,
        )

    def update_progress(
        self,
        task_id: str,
        processed_docs: int,
        total_docs: Optional[int] = None,
        output_sample_count: Optional[int] = None,
        *,
        expected_status: Optional[str] = None,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> bool:
        """Update task progress."""
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return False
            values: Dict[str, Any] = {"processed_docs": processed_docs}
            effective_total = task.total_docs
            if total_docs is not None:
                effective_total = total_docs
                values["total_docs"] = total_docs
            if output_sample_count is not None:
                values["output_sample_count"] = output_sample_count
            if effective_total > 0:
                values["progress"] = (processed_docs / effective_total) * 100
            query = update(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id
            )
            if expected_status is not None:
                query = query.where(GenerationTaskDB.status == expected_status)
            if expected_run_token is not _RUN_TOKEN_UNSET:
                query = query.where(
                    GenerationTaskDB.run_token == expected_run_token
                )
            result = session.exec(query.values(**values))
            session.commit()
            return result.rowcount == 1

    def reset_progress(self, task_id: str) -> bool:
        """Reset task progress for restart."""
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
            ).first()
            if not task:
                return False

            task.progress = 0.0
            task.processed_docs = 0
            task.output_sample_count = 0
            task.error_message = None
            task.started_at = None
            task.completed_at = None
            session.add(task)
            session.commit()
            logger.info("Reset progress for generation task: %s", task_id)
            return True

    def set_output(
        self,
        task_id: str,
        output_path: str,
        output_sample_count: int,
        output_dataset_id: Optional[str] = None,
        *,
        expected_status: Optional[str] = None,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> bool:
        """Set task output information."""
        values: Dict[str, Any] = {
            "output_path": output_path,
            "output_sample_count": output_sample_count,
        }
        if output_dataset_id:
            values["output_dataset_id"] = output_dataset_id
        with get_session() as session:
            query = update(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id
            )
            if expected_status is not None:
                query = query.where(GenerationTaskDB.status == expected_status)
            if expected_run_token is not _RUN_TOKEN_UNSET:
                query = query.where(
                    GenerationTaskDB.run_token == expected_run_token
                )
            result = session.exec(query.values(**values))
            session.commit()
            if result.rowcount != 1:
                return False
            logger.info("Set generation task %s output: %s (%d samples)", task_id, output_path, output_sample_count)
            return True

    def delete_task(
        self,
        task_id: str,
        *,
        expected_status: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a generation task."""
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB)
                .where(GenerationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False

            if expected_status is not None and task.status != expected_status:
                return False
            if user_id and task.user_id != user_id:
                raise PermissionError(
                    "Generation task owner changed during deletion"
                )

            # Don't allow deleting running tasks
            if task.status in {
                GenerationStatus.PENDING,
                GenerationStatus.RUNNING,
                GenerationStatus.STOPPING,
                GenerationStatus.PUBLISHING,
                GenerationStatus.RECOVERING,
                GenerationStatus.RESTARTING,
            }:
                return False

            session.exec(
                delete(ExternalSyncGenerationDB).where(
                    ExternalSyncGenerationDB.generation_task_id == task_id
                )
            )
            session.delete(task)
            session.commit()
            logger.info("Deleted generation task: %s", task_id)

            return True

    def begin_task_deletion(
        self,
        task_id: str,
        *,
        cascade: bool,
        user_id: Optional[str],
    ) -> Optional[GenerationTaskDeletionIntent]:
        """Persist a generation deletion intent before storage mutation."""
        requested_status = (
            GenerationStatus.DELETING_CASCADE
            if cascade
            else GenerationStatus.DELETING
        )
        deleting_statuses = {
            GenerationStatus.DELETING,
            GenerationStatus.DELETING_CASCADE,
        }
        terminal_statuses = {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        }
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB)
                .where(GenerationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if task is None:
                return None
            if user_id and task.user_id != user_id:
                raise PermissionError(
                    "Generation task owner changed during deletion"
                )
            if task.status in deleting_statuses:
                if task.status != requested_status:
                    raise ValueError(
                        "Generation task deletion mode cannot change after "
                        "cleanup starts"
                    )
                return GenerationTaskDeletionIntent(
                    task=task.to_dict(),
                    newly_started=False,
                    previous_status=None,
                )
            if task.status not in terminal_statuses:
                raise ValueError("Cannot delete a running generation task")

            previous_status = task.status
            task.status = requested_status
            session.add(task)
            session.commit()
            session.refresh(task)
            return GenerationTaskDeletionIntent(
                task=task.to_dict(),
                newly_started=True,
                previous_status=previous_status,
            )

    def cancel_task_deletion(
        self,
        task_id: str,
        *,
        cascade: bool,
        previous_status: str,
        user_id: Optional[str],
    ) -> bool:
        """Restore a newly-created intent before any storage mutation."""
        requested_status = (
            GenerationStatus.DELETING_CASCADE
            if cascade
            else GenerationStatus.DELETING
        )
        terminal_statuses = {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        }
        if previous_status not in terminal_statuses:
            raise ValueError("Invalid generation deletion rollback status")
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB)
                .where(GenerationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if task is None or task.status != requested_status:
                return False
            if user_id and task.user_id != user_id:
                raise PermissionError(
                    "Generation task owner changed during deletion rollback"
                )
            task.status = previous_status
            session.add(task)
            session.commit()
            return True

    def delete_task_with_datasets(
        self,
        task_id: str,
        dataset_ids: List[str],
        *,
        deletion_owner: str,
        user_id: Optional[str],
    ) -> bool:
        """Atomically delete one terminal task and its fenced output datasets."""
        normalized_deletion_owner = require_dataset_deletion_owner(deletion_owner)
        normalized_dataset_ids = list(dict.fromkeys(dataset_ids))
        with get_session() as session:
            task = session.exec(
                select(GenerationTaskDB)
                .where(GenerationTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not task:
                return False
            if task.status != GenerationStatus.DELETING_CASCADE:
                return False
            if user_id and task.user_id != user_id:
                raise PermissionError("Generation task owner changed during deletion")

            datasets = []
            if normalized_dataset_ids:
                datasets = list(
                    session.exec(
                        select(DatasetDB)
                        .where(DatasetDB.dataset_id.in_(normalized_dataset_ids))
                        .with_for_update()
                    ).all()
                )
                datasets_by_id = {dataset.dataset_id: dataset for dataset in datasets}
                missing = set(normalized_dataset_ids) - set(datasets_by_id)
                if missing:
                    raise RuntimeError(
                        "Generation output dataset disappeared during deletion"
                    )
                for dataset_id in normalized_dataset_ids:
                    dataset = datasets_by_id[dataset_id]
                    if user_id and dataset.user_id != user_id:
                        raise PermissionError(
                            "Generation output dataset owner changed during deletion"
                        )
                    if (
                        dataset.source_task_type != "generation"
                        or dataset.source_task_id != task_id
                    ):
                        raise PermissionError(
                            "Dataset is not owned by the generation task"
                        )
                    if dataset.status != "deleting":
                        raise RuntimeError(
                            "Generation output dataset is not fenced for deletion"
                        )
                    if dataset.deletion_owner != normalized_deletion_owner:
                        raise DatasetDeletionOwnerConflictError(
                            "Dataset is owned by another deletion operation"
                        )

                session.exec(
                    delete(DatasetAssetDB).where(
                        DatasetAssetDB.dataset_id.in_(normalized_dataset_ids)
                    )
                )
                session.exec(
                    delete(DatasetLineageEdgeDB).where(
                        or_(
                            DatasetLineageEdgeDB.from_dataset_id.in_(
                                normalized_dataset_ids
                            ),
                            DatasetLineageEdgeDB.to_dataset_id.in_(
                                normalized_dataset_ids
                            ),
                        )
                    )
                )
                for dataset in datasets:
                    session.delete(dataset)

            session.exec(
                delete(ExternalSyncGenerationDB).where(
                    ExternalSyncGenerationDB.generation_task_id == task_id
                )
            )
            session.delete(task)
            session.commit()
            logger.info(
                "Deleted generation task %s with %d output datasets",
                task_id,
                len(normalized_dataset_ids),
            )
        return True

    def find_completed_task(
        self,
        source_dataset_id: str,
        embedding_config_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a completed task with same source dataset + embedding config (for dedup)."""
        with get_session() as session:
            query = select(GenerationTaskDB).where(
                GenerationTaskDB.source_dataset_id == source_dataset_id,
                GenerationTaskDB.embedding_config_id == embedding_config_id,
                GenerationTaskDB.status == GenerationStatus.COMPLETED,
            )
            if user_id:
                query = query.where(GenerationTaskDB.user_id == user_id)
            query = query.order_by(GenerationTaskDB.completed_at.desc())
            task = session.exec(query).first()
            return task.to_dict() if task else None

    def set_filter_results(
        self,
        task_id: str,
        milvus_collection: Optional[str] = None,
        filter_stats: Optional[Dict[str, Any]] = None,
        *,
        expected_status: Optional[str] = None,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> bool:
        """Save embedding filter results for a doc_to_training / qa_to_training task."""
        values: Dict[str, Any] = {}
        if milvus_collection:
            values["milvus_collection"] = milvus_collection
        if filter_stats:
            values["filter_stats"] = filter_stats
        if not values:
            return True
        with get_session() as session:
            query = update(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id
            )
            if expected_status is not None:
                query = query.where(GenerationTaskDB.status == expected_status)
            if expected_run_token is not _RUN_TOKEN_UNSET:
                query = query.where(
                    GenerationTaskDB.run_token == expected_run_token
                )
            result = session.exec(query.values(**values))
            session.commit()
            if result.rowcount != 1:
                return False
            logger.info("Set filter results for task %s: collection=%s", task_id, milvus_collection)
            return True

    def set_deep_eval_output(
        self,
        task_id: str,
        deep_eval_path: str,
        deep_eval_dataset_id: Optional[str] = None,
        *,
        expected_status: Optional[str] = None,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> bool:
        """Save deep evaluation dataset output for a generation task."""
        values: Dict[str, Any] = {"deep_eval_path": deep_eval_path}
        if deep_eval_dataset_id:
            values["deep_eval_dataset_id"] = deep_eval_dataset_id
        with get_session() as session:
            query = update(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id
            )
            if expected_status is not None:
                query = query.where(GenerationTaskDB.status == expected_status)
            if expected_run_token is not _RUN_TOKEN_UNSET:
                query = query.where(
                    GenerationTaskDB.run_token == expected_run_token
                )
            result = session.exec(query.values(**values))
            session.commit()
            if result.rowcount != 1:
                return False
            logger.info(
                "Set deep eval output for task %s: path=%s (dataset=%s)",
                task_id, deep_eval_path, deep_eval_dataset_id,
            )
            return True

    def set_qa_output(
        self,
        task_id: str,
        qa_output_path: str,
        qa_dataset_id: Optional[str] = None,
        qa_filtered_path: Optional[str] = None,
        qa_filtered_dataset_id: Optional[str] = None,
        *,
        expected_status: str = GenerationStatus.RUNNING,
        expected_run_token: Any = _RUN_TOKEN_UNSET,
    ) -> bool:
        """Save QA output only while the expected task state still owns it."""
        values: Dict[str, Any] = {"qa_output_path": qa_output_path}
        if qa_dataset_id:
            values["qa_dataset_id"] = qa_dataset_id
        if qa_filtered_path:
            values["qa_filtered_path"] = qa_filtered_path
        if qa_filtered_dataset_id:
            values["qa_filtered_dataset_id"] = qa_filtered_dataset_id

        with get_session() as session:
            result = session.exec(
                update(GenerationTaskDB)
                .where(
                    GenerationTaskDB.task_id == task_id,
                    GenerationTaskDB.status == expected_status,
                    *(
                        (GenerationTaskDB.run_token == expected_run_token,)
                        if expected_run_token is not _RUN_TOKEN_UNSET
                        else ()
                    ),
                )
                .values(**values)
            )
            session.commit()
            if result.rowcount != 1:
                logger.info(
                    "Skipped QA output update for inactive generation task %s",
                    task_id,
                )
                return False

            logger.info(
                "Set QA output for task %s: all=%s (dataset=%s), filtered=%s (dataset=%s)",
                task_id, qa_output_path, qa_dataset_id, qa_filtered_path, qa_filtered_dataset_id,
            )
            return True


# Singleton instance
generation_task_service = GenerationTaskService()
