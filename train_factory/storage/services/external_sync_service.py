"""
External sync service for managing sync tasks, batches,
generation tracking, and training tracking.
"""

import logging
import os
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set
from datetime import datetime

from sqlmodel import Session, select, func
from sqlalchemy import text, update

from ...config.settings import get_settings
from ..database import get_engine
from ...core.time_utils import sync_now_naive
from ...enums.sync_status import (
    SyncStatus,
    BatchStatus,
    SyncGenerationStatus,
    SyncTrainingStatus,
    TrainingTargetStatus,
)
from ..entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncBatchDB,
    ExternalSyncGenerationDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)

logger = logging.getLogger(__name__)

_UNCONSUMED_BATCH_STATUSES = (
    BatchStatus.REGISTERED,
    BatchStatus.FETCHED,
    BatchStatus.GENERATION_QUEUED,
)

_SYNC_DELETING_STATUSES = {
    SyncStatus.DELETING,
    SyncStatus.DELETING_CASCADE,
}


def _utcnow_naive() -> datetime:
    """Return sync-mode timestamp as naive datetime for DB compatibility."""
    return sync_now_naive()


def sync_task_reserves_collection_name(
    task_id: str,
    current_collection_name: Optional[str],
    collection_name: str,
) -> bool:
    """Return whether one durable sync task owns a physical namespace."""
    if not isinstance(task_id, str) or not task_id:
        return False
    if not isinstance(collection_name, str) or not collection_name.strip():
        return False
    normalized_name = collection_name.strip()
    current_name = (
        current_collection_name.strip()
        if isinstance(current_collection_name, str)
        and current_collection_name.strip()
        else None
    )
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    prefixes = (
        f"tf_sync_{task_id[:8]}_",
        f"tf_sync_v2_{task_id[:8]}_",
        f"tf_sync_v3_{task_hash}_",
    )
    return normalized_name == current_name or normalized_name.startswith(prefixes)




class ExternalSyncService:
    """Service for managing external data sync."""

    _TASK_STATUS_TRANSITIONS: Dict[str, Set[str]] = {
        SyncStatus.IDLE: {
            SyncStatus.SYNCING,
            SyncStatus.GENERATING,
            SyncStatus.TRAINING,
            SyncStatus.LOADING_ADAPTER,
            SyncStatus.ERROR,
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        },
        SyncStatus.SYNCING: {
            SyncStatus.IDLE,
            SyncStatus.GENERATING,
            SyncStatus.TRAINING,
            SyncStatus.ERROR,
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        },
        SyncStatus.GENERATING: {
            SyncStatus.IDLE,
            SyncStatus.TRAINING,
            SyncStatus.ERROR,
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        },
        SyncStatus.TRAINING: {
            SyncStatus.IDLE,
            SyncStatus.LOADING_ADAPTER,
            SyncStatus.ERROR,
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        },
        SyncStatus.LOADING_ADAPTER: {
            SyncStatus.IDLE,
            SyncStatus.ERROR,
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        },
        SyncStatus.ERROR: {
            SyncStatus.IDLE,
            SyncStatus.SYNCING,
            SyncStatus.GENERATING,
            SyncStatus.TRAINING,
            SyncStatus.LOADING_ADAPTER,
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        },
        SyncStatus.DELETING: set(),
        SyncStatus.DELETING_CASCADE: set(),
    }

    _BATCH_STATUS_TRANSITIONS: Dict[str, Set[str]] = {
        BatchStatus.REGISTERED: {
            BatchStatus.FETCHED,
            BatchStatus.GENERATION_QUEUED,
            BatchStatus.GENERATION_DONE,
        },
        BatchStatus.FETCHED: {
            BatchStatus.GENERATION_QUEUED,
            BatchStatus.GENERATION_DONE,
        },
        BatchStatus.GENERATION_QUEUED: {
            BatchStatus.GENERATION_DONE,
            BatchStatus.FETCHED,
        },
        BatchStatus.GENERATION_DONE: {
            BatchStatus.FETCHED,
        },
    }

    _GENERATION_STATUS_TRANSITIONS: Dict[str, Set[str]] = {
        SyncGenerationStatus.PENDING: {
            SyncGenerationStatus.COMPLETED,
            SyncGenerationStatus.FAILED,
        },
        SyncGenerationStatus.COMPLETED: set(),
        SyncGenerationStatus.FAILED: set(),
    }

    _TRAINING_STATUS_TRANSITIONS: Dict[str, Set[str]] = {
        SyncTrainingStatus.PENDING: {
            SyncTrainingStatus.COMPLETED,
            SyncTrainingStatus.FAILED,
            SyncTrainingStatus.ADAPTER_LOAD_FAILED,
            SyncTrainingStatus.ADAPTER_FAILED,
            SyncTrainingStatus.ADAPTER_LOADED,
        },
        SyncTrainingStatus.COMPLETED: {
            SyncTrainingStatus.ADAPTER_LOADED,
            SyncTrainingStatus.ADAPTER_LOAD_FAILED,
            SyncTrainingStatus.ADAPTER_FAILED,
        },
        SyncTrainingStatus.FAILED: {
            SyncTrainingStatus.ADAPTER_LOAD_FAILED,
            SyncTrainingStatus.ADAPTER_FAILED,
            SyncTrainingStatus.ADAPTER_LOADED,
        },
        SyncTrainingStatus.ADAPTER_LOADED: {
            SyncTrainingStatus.ADAPTER_UNLOADED,
            SyncTrainingStatus.ADAPTER_LOAD_FAILED,
            SyncTrainingStatus.ADAPTER_FAILED,
        },
        SyncTrainingStatus.ADAPTER_UNLOADED: {
            SyncTrainingStatus.ADAPTER_LOADED,
            SyncTrainingStatus.ADAPTER_LOAD_FAILED,
            SyncTrainingStatus.ADAPTER_FAILED,
        },
        SyncTrainingStatus.ADAPTER_LOAD_FAILED: {
            SyncTrainingStatus.ADAPTER_LOADED,
            SyncTrainingStatus.ADAPTER_FAILED,
        },
        SyncTrainingStatus.ADAPTER_FAILED: {
            SyncTrainingStatus.ADAPTER_LOADED,
            SyncTrainingStatus.ADAPTER_LOAD_FAILED,
        },
    }

    _TARGET_STATUS_TRANSITIONS: Dict[str, Set[str]] = {
        TrainingTargetStatus.IDLE: {
            TrainingTargetStatus.READY,
            TrainingTargetStatus.TRAINING,
            TrainingTargetStatus.ERROR,
        },
        TrainingTargetStatus.READY: {
            TrainingTargetStatus.TRAINING,
            TrainingTargetStatus.IDLE,
            TrainingTargetStatus.ERROR,
        },
        TrainingTargetStatus.TRAINING: {
            TrainingTargetStatus.IDLE,
            TrainingTargetStatus.LOADING_ADAPTER,
            TrainingTargetStatus.ERROR,
        },
        TrainingTargetStatus.LOADING_ADAPTER: {
            TrainingTargetStatus.IDLE,
            TrainingTargetStatus.ERROR,
        },
        TrainingTargetStatus.ERROR: {
            TrainingTargetStatus.IDLE,
            TrainingTargetStatus.READY,
            TrainingTargetStatus.TRAINING,
        },
    }

    def __init__(self):
        self.engine = None

    def _get_engine(self):
        if self.engine is None:
            self.engine = get_engine()
        return self.engine

    @staticmethod
    def _lock_generation_scope(session: Session, generation_task_id: str):
        """Lock a generation scope in task -> generation order."""
        generation_hint = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_task_id
            )
        ).first()
        if generation_hint is None:
            return None, None

        task = session.exec(
            select(ExternalSyncTaskDB)
            .where(ExternalSyncTaskDB.task_id == generation_hint.task_id)
            .with_for_update()
        ).first()
        generation = session.exec(
            select(ExternalSyncGenerationDB)
            .where(
                ExternalSyncGenerationDB.generation_task_id == generation_task_id
            )
            .with_for_update()
        ).first()
        if generation is not None and generation.task_id != generation_hint.task_id:
            raise RuntimeError("Sync generation ownership changed while locking")
        return task, generation

    @classmethod
    def _validate_transition(
        cls,
        current_status: Optional[str],
        new_status: Optional[str],
        transition_map: Dict[str, Set[str]],
        scope: str,
    ) -> None:
        """Validate status transition against a transition map."""
        if new_status is None:
            return

        valid_statuses = set(transition_map.keys())
        if new_status not in valid_statuses:
            raise ValueError(
                f"Invalid {scope} status: {new_status}. Must be one of: {sorted(valid_statuses)}"
            )

        if not current_status or current_status == new_status:
            return

        if current_status not in transition_map:
            # Backward compatibility: allow transition from legacy/unexpected
            # old values as long as the target status is valid.
            return

        allowed = transition_map[current_status]
        if new_status not in allowed:
            raise ValueError(
                f"Invalid {scope} status transition: {current_status} -> {new_status}. "
                f"Allowed: {sorted(allowed)}"
            )

    # ─── Task CRUD ───

    def create_task(
        self,
        task_name: str,
        user_id: str,
        external_api_config_id: Optional[str] = None,
        external_api_url: str = "",
        external_auth_config: Optional[Dict[str, Any]] = None,
        sync_interval_seconds: int = 300,
        generation_threshold: int = 500,
        generation_mode: str = "doc_to_training",
        generation_config: Optional[Dict[str, Any]] = None,
        training_threshold: int = 1000,
        training_config: Optional[Dict[str, Any]] = None,
        base_deployment_id: Optional[str] = None,
        is_active: bool = True,
    ) -> Dict[str, Any]:
        config = ExternalSyncTaskDB(
            task_name=task_name,
            user_id=user_id,
            external_api_config_id=external_api_config_id,
            external_api_url=external_api_url,
            external_auth_config=external_auth_config or {},
            sync_interval_seconds=sync_interval_seconds,
            generation_threshold=generation_threshold,
            generation_mode=generation_mode,
            generation_config=generation_config,
            training_threshold=training_threshold,
            training_config=training_config,
            base_deployment_id=base_deployment_id,
            is_active=is_active,
        )
        with Session(self._get_engine()) as session:
            session.add(config)
            session.commit()
            session.refresh(config)
            logger.info(f"Created sync task: {config.task_id} for user {user_id}")
            return config.to_dict()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == task_id
            )
            config = session.exec(stmt).first()
            return config.to_dict() if config else None

    def get_task_raw(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task including sensitive auth data (for internal use).

        If external_api_config_id is set, resolves the latest URL and auth
        from the referenced external API config (supports live token rotation).
        """
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == task_id
            )
            config = session.exec(stmt).first()
            if not config:
                return None
            d = config.to_dict(mask_sensitive=False)

            # Resolve from external API config if referenced
            if config.external_api_config_id:
                from .external_api_config_service import external_api_config_service
                api_cfg = external_api_config_service.get_config_raw(config.external_api_config_id)
                if api_cfg:
                    d["external_api_url"] = api_cfg["api_url"]
                    d["external_auth_config"] = api_cfg["auth_config"]

            return d

    def list_tasks(
        self,
        user_id: Optional[str] = None,
        external_api_config_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTaskDB)
            count_stmt = select(func.count()).select_from(ExternalSyncTaskDB)

            if user_id:
                stmt = stmt.where(ExternalSyncTaskDB.user_id == user_id)
                count_stmt = count_stmt.where(ExternalSyncTaskDB.user_id == user_id)
            if external_api_config_id:
                stmt = stmt.where(ExternalSyncTaskDB.external_api_config_id == external_api_config_id)
                count_stmt = count_stmt.where(ExternalSyncTaskDB.external_api_config_id == external_api_config_id)
            if is_active is not None:
                stmt = stmt.where(ExternalSyncTaskDB.is_active == is_active)
                count_stmt = count_stmt.where(ExternalSyncTaskDB.is_active == is_active)

            total = session.exec(count_stmt).one()
            stmt = stmt.order_by(
                ExternalSyncTaskDB.created_at.desc(),
                ExternalSyncTaskDB.id.desc(),
            )
            stmt = stmt.offset(offset).limit(limit)
            configs = session.exec(stmt).all()
            return [c.to_dict() for c in configs], total

    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks (for sync manager startup).

        Resolves external API config references for each active task.
        """
        from .external_api_config_service import external_api_config_service

        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.is_active.is_(True)
            )
            configs = session.exec(stmt).all()
            results = []
            for c in configs:
                d = c.to_dict(mask_sensitive=False)
                # Resolve from external API config if referenced
                if c.external_api_config_id:
                    api_cfg = external_api_config_service.get_config_raw(c.external_api_config_id)
                    if api_cfg:
                        d["external_api_url"] = api_cfg["api_url"]
                        d["external_auth_config"] = api_cfg["auth_config"]
                results.append(d)
            return results

    def list_active_collection_consumers(
        self,
        collection_names: List[str],
    ) -> List[str]:
        """Return sync tasks whose durable Milvus namespace contains a name.

        Stopping a sync task does not detach its collection. A later restart
        assumes the persisted collection still contains the historical
        vectors, so every task row remains a consumer until the task itself is
        deleted.
        """
        names = {
            name.strip()
            for name in collection_names
            if isinstance(name, str) and name.strip()
        }
        if not names:
            return []

        with Session(self._get_engine()) as session:
            tasks = session.exec(select(ExternalSyncTaskDB)).all()

        consumers = []
        for task in tasks:
            task_id = str(task.task_id)
            if any(
                sync_task_reserves_collection_name(
                    task_id,
                    task.milvus_collection_name,
                    name,
                )
                for name in names
            ):
                consumers.append(task_id)
        return sorted(consumers)

    def update_task(self, task_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        invalidated_batches: List[Dict[str, Any]] = []
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id == task_id)
                .with_for_update()
            )
            config = session.exec(stmt).first()
            if not config:
                return None
            if config.status in {
                SyncStatus.DELETING,
                SyncStatus.DELETING_CASCADE,
            }:
                raise ValueError("Sync task deletion is in progress")

            source_identity_changed = False
            if "external_api_config_id" in kwargs:
                source_identity_changed = (
                    (kwargs.get("external_api_config_id") or "").strip()
                    != (config.external_api_config_id or "").strip()
                )
            if "external_api_url" in kwargs:
                source_identity_changed = source_identity_changed or (
                    (kwargs.get("external_api_url") or "").strip()
                    != (config.external_api_url or "").strip()
                )

            if source_identity_changed:
                has_persisted_history = bool(
                    config.last_sync_at
                    or config.last_sync_boundary_ids
                    or int(config.pending_record_count or 0)
                    or int(config.pending_training_samples or 0)
                    or int(config.total_record_count or 0)
                    or int(config.total_training_samples or 0)
                    or int(config.total_trainings or 0)
                    or config.milvus_collection_name
                )
                if not has_persisted_history:
                    history_entities = (
                        ExternalSyncBatchDB,
                        ExternalSyncGenerationDB,
                        ExternalSyncTrainingDB,
                    )
                    has_persisted_history = any(
                        session.exec(
                            select(entity.id)
                            .where(entity.task_id == task_id)
                            .limit(1)
                        ).first()
                        is not None
                        for entity in history_entities
                    )
                if has_persisted_history or config.status not in {
                    SyncStatus.IDLE,
                    SyncStatus.ERROR,
                }:
                    raise ValueError(
                        "Cannot change sync source after data ingestion has "
                        "started; create a new sync task"
                    )

            # Detect sync position reset: clearing last_sync_at means
            # "re-fetch from the beginning", so unconsumed (fetched) batches
            # and their counters must be invalidated to prevent double-counting.
            sync_position_reset = (
                "last_sync_at" in kwargs
                and kwargs["last_sync_at"] is None
                and config.last_sync_at is not None
            )

            if "status" in kwargs:
                self._validate_transition(
                    config.status,
                    kwargs.get("status"),
                    self._TASK_STATUS_TRANSITIONS,
                    "sync task",
                )

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            config.updated_at = _utcnow_naive()

            if sync_position_reset:
                invalidated_batches = self._invalidate_fetched_batches(
                    session,
                    config,
                )

            session.add(config)
            session.commit()
            session.refresh(config)
            result = config.to_dict()
            user_id = config.user_id

        if invalidated_batches:
            self._cleanup_managed_batch_files(
                task_id,
                user_id,
                invalidated_batches,
            )
        return result

    def begin_task_deletion(
        self,
        task_id: str,
        *,
        cascade: bool,
        expected_user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Persist an inactive deletion intent before external resources mutate."""
        requested_status = (
            SyncStatus.DELETING_CASCADE if cascade else SyncStatus.DELETING
        )
        deleting_statuses = {
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        }
        with Session(self._get_engine()) as session:
            config = session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not config:
                return None
            if config.user_id != expected_user_id:
                raise ValueError("Sync task owner changed during deletion")
            if config.status in deleting_statuses:
                if config.status != requested_status:
                    raise ValueError(
                        "Sync task deletion mode cannot change after cleanup starts"
                    )
                return self._task_deletion_snapshot(session, config)

            self._validate_transition(
                config.status,
                requested_status,
                self._TASK_STATUS_TRANSITIONS,
                "sync task",
            )
            config.status = requested_status
            config.is_active = False
            config.error_message = None
            config.updated_at = _utcnow_naive()
            session.add(config)
            session.commit()
            session.refresh(config)
            return self._task_deletion_snapshot(session, config)

    @staticmethod
    def _task_deletion_snapshot(
        session: Session,
        config: ExternalSyncTaskDB,
    ) -> Dict[str, Any]:
        """Attach internal row identity and exact child-config metadata."""
        result = config.to_dict()
        result["_deletion_identity"] = config.id
        result["_training_target_ids"] = tuple(
            session.exec(
                select(ExternalSyncTrainingTargetDB.target_id)
                .where(ExternalSyncTrainingTargetDB.task_id == config.task_id)
                .order_by(ExternalSyncTrainingTargetDB.id)
            ).all()
        )
        return result

    def cancel_task_deletion(
        self,
        task_id: str,
        *,
        status: str,
        is_active: bool,
        expected_user_id: str,
        expected_deleting_status: str,
    ) -> bool:
        """Restore a task only when cleanup has not mutated external resources."""
        if status in {SyncStatus.DELETING, SyncStatus.DELETING_CASCADE}:
            raise ValueError("Cannot restore a sync task to a deleting status")
        if expected_deleting_status not in {
            SyncStatus.DELETING,
            SyncStatus.DELETING_CASCADE,
        }:
            raise ValueError("Invalid expected sync task deleting status")
        with Session(self._get_engine()) as session:
            config = session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not config:
                return False
            if config.user_id != expected_user_id:
                raise ValueError("Sync task owner changed during deletion rollback")
            if config.status != expected_deleting_status:
                raise ValueError("Sync task deletion mode changed during rollback")
            if status not in self._TASK_STATUS_TRANSITIONS:
                raise ValueError(f"Invalid restored sync task status: {status}")
            config.status = status
            config.is_active = is_active
            config.updated_at = _utcnow_naive()
            session.add(config)
            session.commit()
            return True

    def finalize_task_deletion(
        self,
        task_id: str,
        *,
        cascade: bool,
        expected_user_id: str,
        expected_task_identity: int,
        batch_snapshot: Sequence[Dict[str, Any]],
        generation_snapshot: Sequence[Dict[str, Any]],
        training_snapshot: Sequence[Dict[str, Any]],
        expected_training_target_ids: Sequence[str],
    ) -> bool:
        """Delete only the exact preflighted metadata under a locked parent."""
        expected_status = (
            SyncStatus.DELETING_CASCADE if cascade else SyncStatus.DELETING
        )

        def _snapshot_map(
            snapshot: Sequence[Dict[str, Any]],
            identity_key: str,
            resource_name: str,
        ) -> Dict[Any, Dict[str, Any]]:
            mapped: Dict[Any, Dict[str, Any]] = {}
            for item in snapshot:
                identity = item.get(identity_key)
                if identity in (None, "") or identity in mapped:
                    raise ValueError(
                        f"Invalid {resource_name} deletion snapshot"
                    )
                mapped[identity] = dict(item)
            return mapped

        batch_map = _snapshot_map(batch_snapshot, "batch_id", "sync batch")
        generation_map = _snapshot_map(
            generation_snapshot,
            "id",
            "sync generation",
        )
        training_map = _snapshot_map(training_snapshot, "id", "sync training")
        target_ids = tuple(expected_training_target_ids)
        if len(set(target_ids)) != len(target_ids) or any(
            target_id in (None, "") for target_id in target_ids
        ):
            raise ValueError("Invalid sync training target deletion snapshot")

        def _require_snapshot_match(
            row: Any,
            expected: Dict[str, Any],
            fields: Sequence[str],
            resource_name: str,
        ) -> None:
            if row is None or any(
                getattr(row, field) != expected.get(field) for field in fields
            ):
                raise ValueError(f"{resource_name} metadata changed during deletion")

        with Session(self._get_engine()) as session:
            config = session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if not config:
                return False
            if (
                config.id != expected_task_identity
                or config.task_id != task_id
                or config.user_id != expected_user_id
                or config.status != expected_status
                or config.is_active
            ):
                raise ValueError("Sync task parent changed during deletion")

            batch_count = session.exec(
                select(func.count())
                .select_from(ExternalSyncBatchDB)
                .where(ExternalSyncBatchDB.task_id == task_id)
            ).one()
            if batch_count != len(batch_map):
                raise ValueError("Sync batch metadata changed during deletion")
            batch_rows = []
            for batch_id, expected in batch_map.items():
                row = session.exec(
                    select(ExternalSyncBatchDB)
                    .where(ExternalSyncBatchDB.batch_id == batch_id)
                    .with_for_update()
                ).first()
                _require_snapshot_match(
                    row,
                    expected,
                    (
                        "batch_id",
                        "task_id",
                        "user_id",
                        "storage_path",
                        "status",
                        "dataset_id",
                        "generation_task_id",
                    ),
                    "Sync batch",
                )
                batch_rows.append(row)

            generation_count = session.exec(
                select(func.count())
                .select_from(ExternalSyncGenerationDB)
                .where(ExternalSyncGenerationDB.task_id == task_id)
            ).one()
            generation_ids = tuple(generation_map)
            unknown_generation_count = generation_count
            if generation_ids:
                unknown_generation_count = session.exec(
                    select(func.count())
                    .select_from(ExternalSyncGenerationDB)
                    .where(
                        ExternalSyncGenerationDB.task_id == task_id,
                        ExternalSyncGenerationDB.id.notin_(generation_ids),
                    )
                ).one()
            if unknown_generation_count or (
                not cascade and generation_count != len(generation_map)
            ):
                raise ValueError("Sync generation metadata changed during deletion")
            generation_rows = []
            for row_id, expected in generation_map.items():
                row = session.exec(
                    select(ExternalSyncGenerationDB)
                    .where(ExternalSyncGenerationDB.id == row_id)
                    .with_for_update()
                ).first()
                if row is None and cascade:
                    continue
                _require_snapshot_match(
                    row,
                    expected,
                    (
                        "id",
                        "task_id",
                        "user_id",
                        "generation_task_id",
                        "output_dataset_id",
                        "status",
                    ),
                    "Sync generation",
                )
                generation_rows.append(row)

            training_count = session.exec(
                select(func.count())
                .select_from(ExternalSyncTrainingDB)
                .where(ExternalSyncTrainingDB.task_id == task_id)
            ).one()
            training_ids = tuple(training_map)
            unknown_training_count = training_count
            if training_ids:
                unknown_training_count = session.exec(
                    select(func.count())
                    .select_from(ExternalSyncTrainingDB)
                    .where(
                        ExternalSyncTrainingDB.task_id == task_id,
                        ExternalSyncTrainingDB.id.notin_(training_ids),
                    )
                ).one()
            if unknown_training_count or (
                not cascade and training_count != len(training_map)
            ):
                raise ValueError("Sync training metadata changed during deletion")
            training_rows = []
            for row_id, expected in training_map.items():
                row = session.exec(
                    select(ExternalSyncTrainingDB)
                    .where(ExternalSyncTrainingDB.id == row_id)
                    .with_for_update()
                ).first()
                if row is None and cascade:
                    continue
                _require_snapshot_match(
                    row,
                    expected,
                    (
                        "id",
                        "task_id",
                        "user_id",
                        "training_task_id",
                        "status",
                        "loaded_adapter_name",
                        "loaded_adapter_id",
                    ),
                    "Sync training",
                )
                training_rows.append(row)

            target_count = session.exec(
                select(func.count())
                .select_from(ExternalSyncTrainingTargetDB)
                .where(ExternalSyncTrainingTargetDB.task_id == task_id)
            ).one()
            if target_count != len(target_ids):
                raise ValueError(
                    "Sync training target metadata changed during deletion"
                )
            target_rows = []
            for target_id in target_ids:
                row = session.exec(
                    select(ExternalSyncTrainingTargetDB)
                    .where(ExternalSyncTrainingTargetDB.target_id == target_id)
                    .with_for_update()
                ).first()
                if row is None or row.task_id != task_id:
                    raise ValueError(
                        "Sync training target metadata changed during deletion"
                    )
                target_rows.append(row)

            for row in batch_rows:
                session.delete(row)
            for row in generation_rows:
                session.delete(row)
            for row in training_rows:
                session.delete(row)
            for row in target_rows:
                session.delete(row)
            session.delete(config)
            session.commit()
            return True

    @staticmethod
    def _cleanup_managed_batch_files(
        task_id: str,
        user_id: str,
        batches: List[Dict[str, Any]],
    ) -> None:
        from ...sync.sync_worker import _delete_managed_batch_files

        _deleted, failed = _delete_managed_batch_files(
            task_id,
            user_id,
            batches,
        )
        if failed:
            logger.warning(
                "Cleanup for sync task %s left %d batch files for operator "
                "cleanup",
                task_id[:8],
                failed,
            )

    def compare_and_set_task_status(
        self,
        task_id: str,
        expected_status: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update a sync-task status only while it still has the expected value."""
        self._validate_transition(
            expected_status,
            status,
            self._TASK_STATUS_TRANSITIONS,
            "sync task",
        )
        values: Dict[str, Any] = {
            "status": status,
            "updated_at": _utcnow_naive(),
        }
        if error_message is not None:
            values["error_message"] = error_message

        with Session(self._get_engine()) as session:
            result = session.exec(
                update(ExternalSyncTaskDB)
                .where(
                    ExternalSyncTaskDB.task_id == task_id,
                    ExternalSyncTaskDB.status == expected_status,
                )
                .values(**values)
            )
            session.commit()
            return result.rowcount == 1

    def _invalidate_fetched_batches(
        self, session: Session, config: ExternalSyncTaskDB
    ) -> List[Dict[str, Any]]:
        """Delete unconsumed FETCHED and REGISTERED batches and subtract their
        record counts from pending_record_count.  Called when last_sync_at is
        reset to None (re-fetch from beginning) so a re-fetch won't
        double-count."""
        task_id = config.task_id
        unconsumed_stmt = (
            select(ExternalSyncBatchDB)
            .where(ExternalSyncBatchDB.task_id == task_id)
            .where(
                ExternalSyncBatchDB.status.in_(
                    [BatchStatus.FETCHED, BatchStatus.REGISTERED]
                )
            )
        )
        unconsumed_batches = session.exec(unconsumed_stmt).all()
        if not unconsumed_batches:
            return []

        invalidated_batches = [batch.to_dict() for batch in unconsumed_batches]
        total_records = sum(b.record_count for b in unconsumed_batches)
        for b in unconsumed_batches:
            session.delete(b)

        config.pending_record_count = max(
            config.pending_record_count - total_records, 0
        )
        logger.info(
            "Sync position reset for task %s: deleted %d unconsumed batches "
            "(%d records), pending_record_count → %d",
            task_id[:8], len(unconsumed_batches), total_records,
            config.pending_record_count,
        )
        return invalidated_batches


    def delete_task(self, task_id: str) -> bool:
        batch_files: List[Dict[str, Any]] = []
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == task_id
            )
            config = session.exec(stmt).first()
            if not config:
                return False

            # Remove related sync records to avoid orphan rows.
            batch_stmt = select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.task_id == task_id
            )
            batch_rows = session.exec(batch_stmt).all()
            batch_files = [row.to_dict() for row in batch_rows]
            for row in batch_rows:
                session.delete(row)

            gen_stmt = select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.task_id == task_id
            )
            for row in session.exec(gen_stmt).all():
                session.delete(row)

            train_stmt = select(ExternalSyncTrainingDB).where(
                ExternalSyncTrainingDB.task_id == task_id
            )
            for row in session.exec(train_stmt).all():
                session.delete(row)

            from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB
            target_stmt = select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.task_id == task_id
            )
            for target in session.exec(target_stmt).all():
                session.delete(target)

            session.delete(config)
            session.commit()
            user_id = config.user_id

        if batch_files:
            self._cleanup_managed_batch_files(task_id, user_id, batch_files)
        return True

    # ─── Counter operations ───

    def increment_pending_records(self, task_id: str, count: int):
        with Session(self._get_engine()) as session:
            session.exec(
                text(
                    "UPDATE external_sync_tasks "
                    "SET pending_record_count = pending_record_count + :count, "
                    "    total_record_count = total_record_count + :count, "
                    "    updated_at = NOW() "
                    "WHERE task_id = :tid"
                ),
                params={"count": count, "tid": task_id},
            )
            session.commit()

    def restore_pending_records(self, task_id: str, count: int):
        """Restore pending_record_count WITHOUT incrementing total_record_count.

        Used when a generation fails and batches are reverted — the records
        were already counted in total when first ingested, so only the
        pending counter needs to be restored.
        """
        with Session(self._get_engine()) as session:
            session.exec(
                text(
                    "UPDATE external_sync_tasks "
                    "SET pending_record_count = pending_record_count + :count, "
                    "    updated_at = NOW() "
                    "WHERE task_id = :tid"
                ),
                params={"count": count, "tid": task_id},
            )
            session.commit()

    def reset_pending_records(self, task_id: str):
        with Session(self._get_engine()) as session:
            session.exec(
                text(
                    "UPDATE external_sync_tasks "
                    "SET pending_record_count = 0, updated_at = NOW() "
                    "WHERE task_id = :tid"
                ),
                params={"tid": task_id},
            )
            session.commit()

    def reconcile_pending_records(self, task_id: str) -> int:
        """Reconcile pending counters from unconsumed batch rows.

        If a crash happens after batch row creation but before counter update,
        retries may hit duplicate-batch path and skip increment forever.
        This method backfills missing pending/total counters from existing
        FETCHED batches only (REGISTERED batches haven't completed Stage-1
        pre-indexing and are not yet counted in pending_record_count).

        Returns:
            Number of records backfilled into counters.
        """
        with Session(self._get_engine()) as session:
            task = session.exec(
                select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
            ).first()
            if not task:
                return 0

            # 仅从已完成 Stage-1 入库的 FETCHED 批次回填，避免把尚未入库的
            # REGISTERED 批次计入 pending_record_count。
            expected_pending = session.exec(
                select(func.coalesce(func.sum(ExternalSyncBatchDB.record_count), 0)).where(
                    ExternalSyncBatchDB.task_id == task_id,
                    ExternalSyncBatchDB.status == BatchStatus.FETCHED,
                )
            ).one()
            expected_pending = int(expected_pending or 0)

            if expected_pending <= int(task.pending_record_count or 0):
                return 0

            backfill = expected_pending - int(task.pending_record_count or 0)
            task.pending_record_count = expected_pending
            task.total_record_count = int(task.total_record_count or 0) + backfill
            task.updated_at = _utcnow_naive()
            session.add(task)
            session.commit()
            logger.warning(
                "Reconciled sync counters for task %s: +%d pending records",
                task_id[:8],
                backfill,
            )
            return backfill

    def increment_pending_training_samples(self, task_id: str, count: int):
        with Session(self._get_engine()) as session:
            session.exec(
                text(
                    "UPDATE external_sync_tasks "
                    "SET pending_training_samples = pending_training_samples + :count, "
                    "    total_training_samples = total_training_samples + :count, "
                    "    updated_at = NOW() "
                    "WHERE task_id = :tid"
                ),
                params={"count": count, "tid": task_id},
            )
            session.commit()

    def reset_pending_training_samples(self, task_id: str):
        with Session(self._get_engine()) as session:
            session.exec(
                text(
                    "UPDATE external_sync_tasks "
                    "SET pending_training_samples = 0, updated_at = NOW() "
                    "WHERE task_id = :tid"
                ),
                params={"tid": task_id},
            )
            session.commit()

    def recalculate_sample_counters(self, task_id: str) -> Dict[str, Any]:
        """Recalculate total_training_samples and pending_training_samples from
        actual generation records, fixing any counter drift.

        - total_training_samples = SUM(output_sample_count) of non-disabled completed generations
        - pending_training_samples = same, but only for generations completed after the last training
        """
        with Session(self._get_engine()) as session:
            # Read current values
            task = session.exec(
                select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
            ).first()
            if not task:
                return {}

            old_total = task.total_training_samples
            old_pending = task.pending_training_samples

            # Sum all non-disabled completed generations
            rows = session.exec(
                text(
                    "SELECT COALESCE(SUM(output_sample_count), 0) as total "
                    "FROM external_sync_generations "
                    "WHERE task_id = :tid AND disabled = 0 AND status = 'completed' "
                    "  AND output_sample_count > 0"
                ).bindparams(tid=task_id)
            ).first()
            new_total = int(rows[0]) if rows else 0

            # Find the last training's created_at to determine pending window
            last_training = session.exec(
                text(
                    "SELECT MAX(created_at) FROM external_sync_trainings "
                    "WHERE task_id = :tid"
                ).bindparams(tid=task_id)
            ).first()
            last_training_at = last_training[0] if last_training and last_training[0] else None

            if last_training_at:
                # Pending = generations completed after the last training started
                pending_rows = session.exec(
                    text(
                        "SELECT COALESCE(SUM(output_sample_count), 0) as total "
                        "FROM external_sync_generations "
                        "WHERE task_id = :tid AND disabled = 0 AND status = 'completed' "
                        "  AND output_sample_count > 0 AND COALESCE(completed_at, created_at) > :since"
                    ).bindparams(tid=task_id, since=last_training_at)
                ).first()
                new_pending = int(pending_rows[0]) if pending_rows else 0
            else:
                # No training ever happened — all samples are pending
                new_pending = new_total

            # Update
            task.total_training_samples = new_total
            task.pending_training_samples = new_pending
            session.add(task)
            session.commit()

            logger.info(
                "Recalculated sample counters for %s: "
                "total %d→%d, pending %d→%d",
                task_id[:8], old_total, new_total, old_pending, new_pending,
            )

            return {
                "old_total_training_samples": old_total,
                "new_total_training_samples": new_total,
                "old_pending_training_samples": old_pending,
                "new_pending_training_samples": new_pending,
            }

    # ─── Batch operations ───

    @staticmethod
    def _file_sha256(path: str) -> Optional[str]:
        """Compute SHA256 of a local file for safe duplicate detection."""
        if not path or not os.path.isfile(path):
            return None
        try:
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return None

    @staticmethod
    def _pending_batch_totals(session: Session, task_id: str) -> tuple[int, int]:
        filters = (
            ExternalSyncBatchDB.task_id == task_id,
            ExternalSyncBatchDB.status.in_(_UNCONSUMED_BATCH_STATUSES),
        )
        batch_count = session.exec(
            select(func.count()).select_from(ExternalSyncBatchDB).where(*filters)
        ).one()
        record_count = session.exec(
            select(func.coalesce(func.sum(ExternalSyncBatchDB.record_count), 0)).where(
                *filters
            )
        ).one()
        return int(batch_count or 0), int(record_count or 0)

    @classmethod
    def _assert_pending_batch_capacity(
        cls,
        session: Session,
        task_id: str,
        *,
        additional_batches: int,
        additional_records: int,
    ) -> None:
        settings = get_settings()
        batch_count, record_count = cls._pending_batch_totals(session, task_id)
        if (
            batch_count + max(int(additional_batches or 0), 0)
            > settings.sync_pending_max_batches_per_task
        ):
            raise ValueError("External sync pending batch quota exceeded")
        if (
            record_count + max(int(additional_records or 0), 0)
            > settings.sync_pending_max_records_per_task
        ):
            raise ValueError("External sync pending record quota exceeded")

    def create_batch(
        self,
        task_id: str,
        user_id: str,
        record_count: int,
        storage_path: str,
        since_time: Optional[datetime],
        until_time: Optional[datetime] = None,
        initial_status: str = BatchStatus.FETCHED,
    ) -> tuple[Dict[str, Any], bool]:
        """Create a sync batch record.

        Returns:
            (batch_dict, is_new): is_new=False when a duplicate batch was found.

        Dedup: only treat as duplicate when BOTH time window and payload match.
        Time-window-only dedup can drop legitimate late-arriving records.
        """
        engine = self._get_engine()
        with Session(engine) as session:
            if engine.dialect.name == "sqlite":
                session.exec(text("BEGIN IMMEDIATE"))

            task = session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if task is None or task.user_id != user_id:
                raise ValueError("External sync task ownership mismatch")
            if task.status in _SYNC_DELETING_STATUSES:
                raise ValueError("Sync task deletion is in progress")

            # Candidate rows: same unconsumed window
            dedup_stmt = select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.task_id == task_id,
                ExternalSyncBatchDB.status.in_([
                    BatchStatus.FETCHED, BatchStatus.REGISTERED,
                ]),
            ).with_for_update()
            if since_time is None:
                dedup_stmt = dedup_stmt.where(ExternalSyncBatchDB.since_time.is_(None))
            else:
                dedup_stmt = dedup_stmt.where(ExternalSyncBatchDB.since_time == since_time)
            if until_time is None:
                dedup_stmt = dedup_stmt.where(ExternalSyncBatchDB.until_time.is_(None))
            else:
                dedup_stmt = dedup_stmt.where(ExternalSyncBatchDB.until_time == until_time)

            candidates = session.exec(dedup_stmt).all()
            if candidates:
                new_hash: Optional[str] = None
                for existing in candidates:
                    # Fast path: exact same file path
                    if existing.storage_path == storage_path:
                        logger.warning(
                            f"Duplicate batch for task {task_id[:8]}: "
                            f"existing {existing.batch_id[:8]} reuses storage_path."
                        )
                        return existing.to_dict(), False

                    # Different payload size => definitely not duplicate.
                    if int(existing.record_count or 0) != int(record_count or 0):
                        continue

                    existing_hash = self._file_sha256(existing.storage_path)
                    if existing_hash is None:
                        continue
                    if new_hash is None:
                        new_hash = self._file_sha256(storage_path)
                    if new_hash and existing_hash == new_hash:
                        logger.warning(
                            f"Duplicate batch for task {task_id[:8]}: "
                            f"existing {existing.batch_id[:8]} has identical payload "
                            f"(since={since_time}, until={until_time}, records={record_count})."
                        )
                        return existing.to_dict(), False

            # 限制创建时只允许 REGISTERED 或 FETCHED 作为初始状态（#12）
            _ALLOWED_INITIAL = (BatchStatus.REGISTERED, BatchStatus.FETCHED)
            if initial_status not in _ALLOWED_INITIAL:
                raise ValueError(
                    f"Invalid initial batch status: {initial_status}. "
                    f"Allowed: {_ALLOWED_INITIAL}"
                )

            self._assert_pending_batch_capacity(
                session,
                task_id,
                additional_batches=1,
                additional_records=int(record_count or 0),
            )
            batch = ExternalSyncBatchDB(
                task_id=task_id,
                user_id=user_id,
                record_count=record_count,
                storage_path=storage_path,
                since_time=since_time,
                until_time=until_time,
                status=initial_status,
            )
            session.add(batch)
            session.commit()
            session.refresh(batch)
            return batch.to_dict(), True

    def reset_completed_batches(self, task_id: str) -> int:
        """Reset generation_done batches back to fetched.

        Used when all generation outputs have been disabled and the user
        wants to regenerate from scratch with updated config.

        Returns:
            Number of batches reset and total record count restored.
        """
        engine = self._get_engine()
        with Session(engine) as session:
            if engine.dialect.name == "sqlite":
                session.exec(text("BEGIN IMMEDIATE"))

            task = session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.task_id == task_id)
                .with_for_update()
            ).first()
            if task is None:
                session.rollback()
                return 0

            stmt = (
                select(ExternalSyncBatchDB)
                .where(
                    ExternalSyncBatchDB.task_id == task_id,
                    ExternalSyncBatchDB.status == BatchStatus.GENERATION_DONE,
                )
                .with_for_update()
            )
            batches = session.exec(stmt).all()
            if not batches:
                session.rollback()
                return 0

            total_records = sum(max(int(batch.record_count or 0), 0) for batch in batches)
            self._assert_pending_batch_capacity(
                session,
                task_id,
                additional_batches=len(batches),
                additional_records=total_records,
            )
            for b in batches:
                b.status = BatchStatus.FETCHED
                b.generation_task_id = None
                session.add(b)

            session.flush()
            fetched_records = session.exec(
                select(func.coalesce(func.sum(ExternalSyncBatchDB.record_count), 0)).where(
                    ExternalSyncBatchDB.task_id == task_id,
                    ExternalSyncBatchDB.status == BatchStatus.FETCHED,
                )
            ).one()
            task.pending_record_count = int(fetched_records or 0)
            task.updated_at = _utcnow_naive()
            session.add(task)

            session.commit()
            logger.info(
                f"Reset {len(batches)} completed batches to fetched for task "
                f"{task_id[:8]} ({total_records} records restored to pending)"
            )
            return len(batches)

    def get_pending_batches(self, task_id: str) -> List[Dict[str, Any]]:
        """Get the oldest bounded prefix of fetched batches for a task."""
        settings = get_settings()
        max_batches = settings.sync_pending_max_batches_per_task
        max_records = settings.sync_pending_max_records_per_task
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalSyncBatchDB)
                .where(ExternalSyncBatchDB.task_id == task_id)
                .where(ExternalSyncBatchDB.status == BatchStatus.FETCHED)
                .order_by(
                    ExternalSyncBatchDB.fetched_at.asc(),
                    ExternalSyncBatchDB.id.asc(),
                )
                .limit(max_batches)
            )
            batches = session.exec(stmt).all()
            selected = []
            selected_records = 0
            for batch in batches:
                projected_records = selected_records + max(
                    int(batch.record_count or 0),
                    0,
                )
                if projected_records > max_records:
                    if selected:
                        break
                    raise ValueError(
                        "Oldest external sync batch exceeds pending record quota"
                    )
                selected.append(batch)
                selected_records = projected_records
            return [batch.to_dict() for batch in selected]

    def list_batches(
        self,
        task_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        oldest_first: bool = False,
    ) -> tuple[List[Dict[str, Any]], int]:
        with Session(self._get_engine()) as session:
            count_stmt = select(func.count()).select_from(ExternalSyncBatchDB).where(ExternalSyncBatchDB.task_id == task_id)
            stmt = select(ExternalSyncBatchDB).where(ExternalSyncBatchDB.task_id == task_id)
            if status:
                count_stmt = count_stmt.where(ExternalSyncBatchDB.status == status)
                stmt = stmt.where(ExternalSyncBatchDB.status == status)
            total = session.exec(count_stmt).one()

            order = (
                ExternalSyncBatchDB.fetched_at.asc()
                if oldest_first
                else ExternalSyncBatchDB.fetched_at.desc()
            )
            id_order = (
                ExternalSyncBatchDB.id.asc()
                if oldest_first
                else ExternalSyncBatchDB.id.desc()
            )
            stmt = stmt.order_by(order, id_order).offset(offset).limit(limit)
            batches = session.exec(stmt).all()
            return [b.to_dict() for b in batches], total

    def update_batch_status(
        self, batch_id: str, status: str,
        generation_task_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalSyncBatchDB)
                .where(ExternalSyncBatchDB.batch_id == batch_id)
                .with_for_update()
            )
            batch = session.exec(stmt).first()
            if batch:
                self._validate_transition(
                    batch.status,
                    status,
                    self._BATCH_STATUS_TRANSITIONS,
                    "sync batch",
                )
                batch.status = status
                if generation_task_id:
                    batch.generation_task_id = generation_task_id
                if dataset_id:
                    batch.dataset_id = dataset_id
                session.add(batch)
                session.commit()

    # ─── Generation tracking ───

    def promote_batch_to_fetched(
        self,
        task_id: str,
        batch_id: str,
        user_id: str,
    ) -> bool:
        """Atomically promote one registered batch and credit its counters once."""
        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                task = session.exec(
                    select(ExternalSyncTaskDB)
                    .where(
                        ExternalSyncTaskDB.task_id == task_id,
                        ExternalSyncTaskDB.user_id == user_id,
                    )
                    .with_for_update()
                ).first()
                batch = session.exec(
                    select(ExternalSyncBatchDB)
                    .where(
                        ExternalSyncBatchDB.batch_id == batch_id,
                        ExternalSyncBatchDB.task_id == task_id,
                        ExternalSyncBatchDB.user_id == user_id,
                    )
                    .with_for_update()
                ).first()
                if task is None or batch is None:
                    raise ValueError("Sync batch is no longer available")
                if batch.status == BatchStatus.FETCHED:
                    session.rollback()
                    return False
                if batch.status != BatchStatus.REGISTERED:
                    raise ValueError("Sync batch is no longer available")

                record_count = max(int(batch.record_count or 0), 0)
                batch.status = BatchStatus.FETCHED
                task.pending_record_count = (
                    int(task.pending_record_count or 0) + record_count
                )
                task.total_record_count = (
                    int(task.total_record_count or 0) + record_count
                )
                task.updated_at = _utcnow_naive()
                session.add(batch)
                session.add(task)
                session.commit()
                return True
            except Exception:
                session.rollback()
                raise

    def create_generation(
        self,
        task_id: str,
        generation_task_id: str,
        user_id: str,
        input_batch_ids: List[str],
        input_record_count: int = 0,
    ) -> Dict[str, Any]:
        gen = ExternalSyncGenerationDB(
            task_id=task_id,
            generation_task_id=generation_task_id,
            user_id=user_id,
            input_batch_ids=input_batch_ids,
            input_record_count=input_record_count,
        )
        with Session(self._get_engine()) as session:
            session.add(gen)
            session.commit()
            session.refresh(gen)
            return gen.to_dict()

    def create_generation_and_claim_batches(
        self,
        task_id: str,
        generation_task_id: str,
        user_id: str,
        input_batch_ids: List[str],
        input_record_count: int = 0,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically create generation tracking and claim its input batches."""
        batch_ids = list(input_batch_ids or [])
        unique_batch_ids = list(dict.fromkeys(batch_ids))
        if not batch_ids or len(unique_batch_ids) != len(batch_ids):
            raise ValueError("Sync generation batches are no longer available")

        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                task = session.exec(
                    select(ExternalSyncTaskDB)
                    .where(ExternalSyncTaskDB.task_id == task_id)
                    .with_for_update()
                ).first()
                if (
                    task is None
                    or task.user_id != user_id
                    or task.status != SyncStatus.GENERATING
                ):
                    raise ValueError("Sync generation batches are no longer available")

                pending_generation = session.exec(
                    select(ExternalSyncGenerationDB)
                    .where(
                        ExternalSyncGenerationDB.task_id == task_id,
                        ExternalSyncGenerationDB.status
                        == SyncGenerationStatus.PENDING,
                    )
                    .with_for_update()
                ).first()
                if pending_generation is not None:
                    from .background_task_admission_service import (
                        BackgroundTaskAlreadyExecuting,
                    )

                    raise BackgroundTaskAlreadyExecuting(
                        "A generation is already pending for this sync task"
                    )

                batches = list(
                    session.exec(
                        select(ExternalSyncBatchDB)
                        .where(
                            ExternalSyncBatchDB.task_id == task_id,
                            ExternalSyncBatchDB.user_id == user_id,
                            ExternalSyncBatchDB.batch_id.in_(unique_batch_ids),
                        )
                        .with_for_update()
                    ).all()
                )
                found_ids = {batch.batch_id for batch in batches}
                if found_ids != set(unique_batch_ids) or any(
                    batch.status != BatchStatus.FETCHED
                    or batch.generation_task_id is not None
                    for batch in batches
                ):
                    raise ValueError("Sync generation batches are no longer available")

                claimed_record_count = sum(
                    max(int(batch.record_count or 0), 0) for batch in batches
                )
                if int(input_record_count or 0) != claimed_record_count:
                    raise ValueError("Sync generation batches are no longer available")

                generation = ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_task_id,
                    user_id=user_id,
                    input_batch_ids=unique_batch_ids,
                    input_record_count=claimed_record_count,
                )
                session.add(generation)
                for batch in batches:
                    batch.status = BatchStatus.GENERATION_QUEUED
                    batch.generation_task_id = generation_task_id
                    batch.dataset_id = dataset_id
                    session.add(batch)

                session.flush()
                fetched_record_count = session.exec(
                    select(
                        func.coalesce(func.sum(ExternalSyncBatchDB.record_count), 0)
                    ).where(
                        ExternalSyncBatchDB.task_id == task_id,
                        ExternalSyncBatchDB.status == BatchStatus.FETCHED,
                    )
                ).one()
                task.pending_record_count = int(fetched_record_count or 0)
                task.updated_at = _utcnow_naive()
                session.add(task)

                session.commit()
                session.refresh(generation)
                return generation.to_dict()
            except Exception:
                session.rollback()
                raise

    def get_generation_by_task_id(self, generation_task_id: str) -> Optional[Dict[str, Any]]:
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_task_id
            )
            gen = session.exec(stmt).first()
            return gen.to_dict() if gen else None

    def update_generation_status(
        self,
        generation_task_id: str,
        status: str,
        output_dataset_id: Optional[str] = None,
        output_sample_count: Optional[int] = None,
    ):
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalSyncGenerationDB)
                .where(
                    ExternalSyncGenerationDB.generation_task_id
                    == generation_task_id
                )
                .with_for_update()
            )
            gen = session.exec(stmt).first()
            if gen:
                self._validate_transition(
                    gen.status,
                    status,
                    self._GENERATION_STATUS_TRANSITIONS,
                    "sync generation",
                )
                gen.status = status
                if output_dataset_id:
                    gen.output_dataset_id = output_dataset_id
                if output_sample_count is not None:
                    gen.output_sample_count = output_sample_count
                if status in (SyncGenerationStatus.COMPLETED, SyncGenerationStatus.FAILED):
                    gen.completed_at = _utcnow_naive()
                session.add(gen)
                session.commit()

    def complete_generation_and_consume_batches(
        self,
        generation_task_id: str,
        output_dataset_id: Optional[str],
        output_sample_count: int,
    ) -> Dict[str, Any]:
        """Atomically complete one generation and apply its sync-side effects."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        result: Dict[str, Any] = {
            "tracking_found": False,
            "completed": False,
            "already_completed": False,
            "task_id": None,
            "completed_batch_count": 0,
            "credited_sample_count": 0,
            "credited_target_count": 0,
            "parent_status_updated": False,
            "uses_training_targets": False,
        }
        sample_count = max(int(output_sample_count or 0), 0)
        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                task, generation = self._lock_generation_scope(
                    session,
                    generation_task_id,
                )
                if not generation:
                    session.rollback()
                    return result

                result["tracking_found"] = True
                result["task_id"] = generation.task_id
                if generation.status != SyncGenerationStatus.PENDING:
                    result["already_completed"] = (
                        generation.status == SyncGenerationStatus.COMPLETED
                    )
                    session.rollback()
                    return result

                input_batch_ids = list(generation.input_batch_ids or [])
                unique_batch_ids = list(dict.fromkeys(input_batch_ids))
                if not input_batch_ids or len(unique_batch_ids) != len(input_batch_ids):
                    session.rollback()
                    return result
                batches = list(
                    session.exec(
                        select(ExternalSyncBatchDB)
                        .where(
                            ExternalSyncBatchDB.task_id == generation.task_id,
                            ExternalSyncBatchDB.batch_id.in_(unique_batch_ids),
                        )
                        .with_for_update()
                    ).all()
                )
                found_ids = {batch.batch_id for batch in batches}
                if found_ids != set(unique_batch_ids) or any(
                    batch.user_id != generation.user_id
                    or batch.generation_task_id != generation_task_id
                    or batch.status != BatchStatus.GENERATION_QUEUED
                    for batch in batches
                ):
                    session.rollback()
                    return result

                targets = list(
                    session.exec(
                        select(ExternalSyncTrainingTargetDB)
                        .where(
                            ExternalSyncTrainingTargetDB.task_id
                            == generation.task_id
                        )
                        .with_for_update()
                    ).all()
                )
                result["uses_training_targets"] = bool(targets)

                completed_at = _utcnow_naive()
                completion = session.exec(
                    update(ExternalSyncGenerationDB)
                    .where(
                        ExternalSyncGenerationDB.id == generation.id,
                        ExternalSyncGenerationDB.status
                        == SyncGenerationStatus.PENDING,
                    )
                    .values(
                        status=SyncGenerationStatus.COMPLETED,
                        output_dataset_id=output_dataset_id,
                        output_sample_count=sample_count,
                        completed_at=completed_at,
                    )
                )
                if completion.rowcount != 1:
                    session.rollback()
                    return result

                for batch in batches:
                    batch.status = BatchStatus.GENERATION_DONE
                    session.add(batch)

                credited_targets = []
                if output_dataset_id and sample_count > 0:
                    if targets:
                        credited_targets = [
                            target
                            for target in targets
                            if target.is_active and target.data_phase == "final"
                        ]
                        for target in credited_targets:
                            target.pending_training_samples = int(
                                target.pending_training_samples or 0
                            ) + sample_count
                            target.total_training_samples = int(
                                target.total_training_samples or 0
                            ) + sample_count
                            target.updated_at = completed_at
                            session.add(target)
                    elif task:
                        task.pending_training_samples = int(
                            task.pending_training_samples or 0
                        ) + sample_count
                        task.total_training_samples = int(
                            task.total_training_samples or 0
                        ) + sample_count

                qa_sample_count = max(int(generation.qa_sample_count or 0), 0)
                if generation.qa_dataset_id and qa_sample_count > 0:
                    for target in targets:
                        if not target.is_active or target.data_phase != "qa":
                            continue
                        target.pending_training_samples = int(
                            target.pending_training_samples or 0
                        ) + qa_sample_count
                        target.total_training_samples = int(
                            target.total_training_samples or 0
                        ) + qa_sample_count
                        target.updated_at = completed_at
                        session.add(target)

                parent_status_updated = False
                if task:
                    if task.status == SyncStatus.GENERATING:
                        task.status = SyncStatus.IDLE
                        parent_status_updated = True
                    task.updated_at = completed_at
                    session.add(task)

                session.commit()
                credited = bool(output_dataset_id and sample_count > 0) and (
                    bool(credited_targets) or (not targets and task is not None)
                )
                result.update(
                    completed=True,
                    completed_batch_count=len(batches),
                    credited_sample_count=sample_count if credited else 0,
                    credited_target_count=len(credited_targets),
                    parent_status_updated=parent_status_updated,
                )
                return result
            except Exception:
                session.rollback()
                raise

    def fail_generation_and_restore_batches(
        self,
        generation_task_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Atomically fail one pending generation and restore its owned batches."""
        result: Dict[str, Any] = {
            "tracking_found": False,
            "recovered": False,
            "already_recovered": False,
            "reconciled": False,
            "task_id": None,
            "user_id": None,
            "restored_batch_count": 0,
            "restored_record_count": 0,
            "parent_status_updated": False,
        }
        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                task, generation = self._lock_generation_scope(
                    session,
                    generation_task_id,
                )
                if not generation:
                    session.rollback()
                    return result

                result["tracking_found"] = True
                result["task_id"] = generation.task_id
                result["user_id"] = generation.user_id

                raw_input_batch_ids = list(generation.input_batch_ids or [])
                input_batch_ids = list(dict.fromkeys(raw_input_batch_ids))
                if not input_batch_ids or len(input_batch_ids) != len(
                    raw_input_batch_ids
                ):
                    session.rollback()
                    return result
                batches = list(
                    session.exec(
                        select(ExternalSyncBatchDB)
                        .where(
                            ExternalSyncBatchDB.batch_id.in_(input_batch_ids),
                            ExternalSyncBatchDB.task_id == generation.task_id,
                        )
                        .with_for_update()
                    ).all()
                )
                if generation.status == SyncGenerationStatus.FAILED:
                    already_recovered = not any(
                        batch.generation_task_id == generation_task_id
                        for batch in batches
                    )
                    session.rollback()
                    if already_recovered:
                        result.update(
                            already_recovered=True,
                            reconciled=True,
                        )
                    return result
                if generation.status != SyncGenerationStatus.PENDING:
                    session.rollback()
                    return result

                if {batch.batch_id for batch in batches} != set(input_batch_ids) or any(
                    batch.user_id != generation.user_id
                    or batch.status
                    not in (BatchStatus.GENERATION_QUEUED, BatchStatus.FETCHED)
                    or (
                        batch.status == BatchStatus.GENERATION_QUEUED
                        and batch.generation_task_id != generation_task_id
                    )
                    or (
                        batch.status == BatchStatus.FETCHED
                        and batch.generation_task_id
                        not in (None, generation_task_id)
                    )
                    for batch in batches
                ):
                    session.rollback()
                    return result

                restored_records = sum(
                    max(int(batch.record_count or 0), 0)
                    for batch in batches
                    if batch.status == BatchStatus.GENERATION_QUEUED
                )
                restored_batch_count = sum(
                    batch.status == BatchStatus.GENERATION_QUEUED
                    for batch in batches
                )
                for batch in batches:
                    batch.status = BatchStatus.FETCHED
                    batch.generation_task_id = None
                    batch.dataset_id = None
                    session.add(batch)

                failure_reason = (reason or "Generation failed").strip()
                generation.status = SyncGenerationStatus.FAILED
                generation.completed_at = _utcnow_naive()
                session.add(generation)

                parent_status_updated = False
                if task:
                    session.flush()
                    fetched_record_count = session.exec(
                        select(
                            func.coalesce(
                                func.sum(ExternalSyncBatchDB.record_count),
                                0,
                            )
                        ).where(
                            ExternalSyncBatchDB.task_id == generation.task_id,
                            ExternalSyncBatchDB.status == BatchStatus.FETCHED,
                        )
                    ).one()
                    task.pending_record_count = int(fetched_record_count or 0)
                    if task.status == SyncStatus.GENERATING:
                        task.status = SyncStatus.IDLE
                        task.error_message = failure_reason
                        parent_status_updated = True
                    task.updated_at = _utcnow_naive()
                    session.add(task)

                session.commit()
                result.update(
                    recovered=True,
                    reconciled=True,
                    restored_batch_count=restored_batch_count,
                    restored_record_count=restored_records,
                    parent_status_updated=parent_status_updated,
                )
                return result
            except Exception:
                session.rollback()
                raise

    def list_pending_generations(
        self,
        *,
        limit: int = 1000,
        offset: int = 0,
        after_id: Optional[int] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """List unreconciled sync generations across tasks for startup recovery."""
        bounded_limit = max(1, min(int(limit), 1000))
        bounded_offset = max(int(offset), 0)
        with Session(self._get_engine()) as session:
            filters = [
                ExternalSyncGenerationDB.status == SyncGenerationStatus.PENDING,
            ]
            if after_id is not None:
                filters.append(ExternalSyncGenerationDB.id > max(int(after_id), 0))
            total = session.exec(
                select(func.count())
                .select_from(ExternalSyncGenerationDB)
                .where(*filters)
            ).one()
            stmt = (
                select(ExternalSyncGenerationDB)
                .where(*filters)
                .order_by(ExternalSyncGenerationDB.id.asc())
                .limit(bounded_limit)
            )
            if after_id is None:
                stmt = stmt.offset(bounded_offset)
            rows = session.exec(stmt).all()
            return [row.to_dict() for row in rows], int(total or 0)

    def list_generations(
        self, task_id: str, status: Optional[str] = None, limit: int = 50, offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        with Session(self._get_engine()) as session:
            count_stmt = select(func.count()).select_from(ExternalSyncGenerationDB).where(ExternalSyncGenerationDB.task_id == task_id)
            stmt = select(ExternalSyncGenerationDB).where(ExternalSyncGenerationDB.task_id == task_id)
            if status:
                count_stmt = count_stmt.where(ExternalSyncGenerationDB.status == status)
                stmt = stmt.where(ExternalSyncGenerationDB.status == status)
            total = session.exec(count_stmt).one()

            stmt = (
                stmt.order_by(
                    ExternalSyncGenerationDB.created_at.desc(),
                    ExternalSyncGenerationDB.id.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
            gens = session.exec(stmt).all()
            return [g.to_dict() for g in gens], total

    def has_pending_generation(self, task_id: str) -> bool:
        """Return whether a sync task still has unreconciled generation work."""
        with Session(self._get_engine()) as session:
            count = session.exec(
                select(func.count())
                .select_from(ExternalSyncGenerationDB)
                .where(
                    ExternalSyncGenerationDB.task_id == task_id,
                    ExternalSyncGenerationDB.status == SyncGenerationStatus.PENDING,
                )
            ).one()
            return int(count or 0) > 0

    def transition_task_if_no_pending_generation(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None,
        expected_status: Optional[str] = None,
    ) -> bool:
        """Atomically transition a task only when no generation claim is active."""
        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                task = session.exec(
                    select(ExternalSyncTaskDB)
                    .where(ExternalSyncTaskDB.task_id == task_id)
                    .with_for_update()
                ).first()
                if task is None or (
                    expected_status is not None and task.status != expected_status
                ):
                    session.rollback()
                    return False

                pending_generation = session.exec(
                    select(ExternalSyncGenerationDB)
                    .where(
                        ExternalSyncGenerationDB.task_id == task_id,
                        ExternalSyncGenerationDB.status
                        == SyncGenerationStatus.PENDING,
                    )
                    .with_for_update()
                ).first()
                if pending_generation is not None:
                    session.rollback()
                    return False

                self._validate_transition(
                    task.status,
                    status,
                    self._TASK_STATUS_TRANSITIONS,
                    "sync task",
                )
                task.status = status
                task.error_message = error_message
                task.updated_at = _utcnow_naive()
                session.add(task)
                session.commit()
                return True
            except Exception:
                session.rollback()
                raise

    def reset_generating_task_if_no_pending_generation(self, task_id: str) -> bool:
        """Atomically reset an orphan generating task only when no claim is active."""
        return self.transition_task_if_no_pending_generation(
            task_id,
            SyncStatus.IDLE,
            error_message=None,
            expected_status=SyncStatus.GENERATING,
        )

    def get_all_completed_generation_datasets(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all completed, non-disabled generation records that have output datasets."""
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalSyncGenerationDB)
                .where(ExternalSyncGenerationDB.task_id == task_id)
                .where(ExternalSyncGenerationDB.status == SyncGenerationStatus.COMPLETED)
                .where(ExternalSyncGenerationDB.output_dataset_id.isnot(None))
                .where(ExternalSyncGenerationDB.disabled.is_(False))
                .order_by(ExternalSyncGenerationDB.created_at.asc())
            )
            gens = session.exec(stmt).all()
            return [g.to_dict() for g in gens]

    def has_enabled_completed_generation(self, task_id: str) -> bool:
        """Return whether any enabled generation has completed for this task."""
        with Session(self._get_engine()) as session:
            count = session.exec(
                select(func.count())
                .select_from(ExternalSyncGenerationDB)
                .where(
                    ExternalSyncGenerationDB.task_id == task_id,
                    ExternalSyncGenerationDB.status
                    == SyncGenerationStatus.COMPLETED,
                    ExternalSyncGenerationDB.disabled.is_(False),
                )
            ).one()
            return int(count or 0) > 0

    def toggle_generation_disabled(
        self, generation_task_id: str, disabled: bool,
    ) -> Optional[Dict[str, Any]]:
        """Toggle the disabled flag on a generation record and adjust sample counters.

        When disabling a completed generation, also resets its associated batches
        from GENERATION_DONE back to FETCHED so they can be
        picked up by a new generation run immediately.
        """
        reset_batch_count = 0
        reset_record_count = 0
        restore_batch_count = 0
        restore_record_count = 0

        engine = self._get_engine()
        with Session(engine) as session:
            if engine.dialect.name == "sqlite":
                session.exec(text("BEGIN IMMEDIATE"))
            task, gen = self._lock_generation_scope(session, generation_task_id)
            if not gen or gen.disabled == disabled:
                session.rollback()
                return gen.to_dict() if gen else None

            gen.disabled = disabled
            session.add(gen)
            session.flush()

            # Reset associated batches when disabling a completed generation
            if disabled and gen.status == SyncGenerationStatus.COMPLETED and gen.input_batch_ids:
                batch_stmt = select(ExternalSyncBatchDB).where(
                    ExternalSyncBatchDB.batch_id.in_(gen.input_batch_ids),
                    ExternalSyncBatchDB.generation_task_id == generation_task_id,
                    ExternalSyncBatchDB.status == BatchStatus.GENERATION_DONE,
                ).with_for_update()
                batches_to_reset = session.exec(batch_stmt).all()
                self._assert_pending_batch_capacity(
                    session,
                    gen.task_id,
                    additional_batches=len(batches_to_reset),
                    additional_records=sum(
                        max(int(batch.record_count or 0), 0)
                        for batch in batches_to_reset
                    ),
                )
                for b in batches_to_reset:
                    b.status = BatchStatus.FETCHED
                    b.generation_task_id = None
                    session.add(b)
                    reset_record_count += b.record_count or 0
                reset_batch_count = len(batches_to_reset)

                if reset_record_count > 0:
                    logger.info(
                        "Generation %s disabled → reset %d batches (%d records) to fetched",
                        generation_task_id[:8], reset_batch_count, reset_record_count,
                    )

            # Restore associated batches when re-enabling a completed generation.
            # Only restore batches that are still unconsumed (fetched + no generation_task_id),
            # to avoid overriding batches already consumed by another generation run.
            if (not disabled) and gen.status == SyncGenerationStatus.COMPLETED and gen.input_batch_ids:
                batch_stmt = select(ExternalSyncBatchDB).where(
                    ExternalSyncBatchDB.batch_id.in_(gen.input_batch_ids),
                    ExternalSyncBatchDB.generation_task_id.is_(None),
                    ExternalSyncBatchDB.status == BatchStatus.FETCHED,
                ).with_for_update()
                batches_to_restore = session.exec(batch_stmt).all()
                for b in batches_to_restore:
                    b.status = BatchStatus.GENERATION_DONE
                    b.generation_task_id = generation_task_id
                    session.add(b)
                    restore_record_count += b.record_count or 0
                restore_batch_count = len(batches_to_restore)

                if restore_record_count > 0:
                    logger.info(
                        "Generation %s re-enabled → restored %d batches (%d records) to generation_done",
                        generation_task_id[:8], restore_batch_count, restore_record_count,
                    )

            if task is not None and (reset_batch_count or restore_batch_count):
                session.flush()
                fetched_records = session.exec(
                    select(
                        func.coalesce(func.sum(ExternalSyncBatchDB.record_count), 0)
                    ).where(
                        ExternalSyncBatchDB.task_id == gen.task_id,
                        ExternalSyncBatchDB.status == BatchStatus.FETCHED,
                    )
                ).one()
                task.pending_record_count = int(fetched_records or 0)
                task.updated_at = _utcnow_naive()
                session.add(task)

            # Keep the global lock order task -> generation -> batches -> targets.
            if gen.status == SyncGenerationStatus.COMPLETED:
                from ..entities.external_sync_entity import (
                    ExternalSyncTrainingTargetDB,
                )

                targets = list(
                    session.exec(
                        select(ExternalSyncTrainingTargetDB)
                        .where(
                            ExternalSyncTrainingTargetDB.task_id == gen.task_id
                        )
                        .with_for_update()
                    ).all()
                )
                direction = -1 if disabled else 1
                final_count = (
                    max(int(gen.output_sample_count or 0), 0)
                    if gen.output_dataset_id
                    else 0
                )
                qa_count = (
                    max(int(gen.qa_sample_count or 0), 0)
                    if gen.qa_dataset_id
                    else 0
                )

                if targets:
                    for target in targets:
                        if not target.is_active:
                            continue
                        count = (
                            final_count
                            if target.data_phase == "final"
                            else qa_count if target.data_phase == "qa" else 0
                        )
                        if count <= 0:
                            continue
                        delta = direction * count
                        target.pending_training_samples = max(
                            int(target.pending_training_samples or 0) + delta,
                            0,
                        )
                        target.total_training_samples = max(
                            int(target.total_training_samples or 0) + delta,
                            0,
                        )
                        target.updated_at = _utcnow_naive()
                        session.add(target)
                elif task is not None and final_count > 0:
                    delta = direction * final_count
                    task.pending_training_samples = max(
                        int(task.pending_training_samples or 0) + delta,
                        0,
                    )
                    task.total_training_samples = max(
                        int(task.total_training_samples or 0) + delta,
                        0,
                    )
                    task.updated_at = _utcnow_naive()
                    session.add(task)

            session.commit()
            session.refresh(gen)

            # Sync associated dataset status: disable → archived, enable → ready
            if gen.output_dataset_id:
                try:
                    from .dataset_service import dataset_service
                    new_status = "archived" if disabled else "ready"
                    dataset_service.update_status(gen.output_dataset_id, new_status)
                    logger.info(
                        "Generation %s %s → dataset %s status → %s",
                        generation_task_id[:8],
                        "disabled" if disabled else "enabled",
                        gen.output_dataset_id[:8],
                        new_status,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to update dataset %s status on generation toggle: %s",
                        gen.output_dataset_id[:8], e,
                    )

            result = gen.to_dict()
            if reset_batch_count > 0:
                result["reset_batch_count"] = reset_batch_count
                result["reset_record_count"] = reset_record_count
            if restore_batch_count > 0:
                result["restore_batch_count"] = restore_batch_count
                result["restore_record_count"] = restore_record_count
            return result

    def delete_generation_tracking(self, generation_task_id: str) -> bool:
        """Delete sync generation tracking record when a generation task is deleted."""
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_task_id
            )
            record = session.exec(stmt).first()
            if record:
                session.delete(record)
                session.commit()
                logger.info(f"Deleted sync generation tracking for task {generation_task_id[:8]}")
                return True
            return False

    # ─── Training tracking ───

    def create_training(
        self,
        task_id: str,
        training_task_id: str,
        user_id: str,
        input_dataset_ids: List[str],
        total_samples: int = 0,
        training_round: int = 1,
        previous_training_task_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        training = ExternalSyncTrainingDB(
            task_id=task_id,
            training_task_id=training_task_id,
            user_id=user_id,
            input_dataset_ids=input_dataset_ids,
            total_samples=total_samples,
            training_round=training_round,
            previous_training_task_id=previous_training_task_id,
            target_id=target_id,
        )
        with Session(self._get_engine()) as session:
            session.add(training)
            session.commit()
            session.refresh(training)
            return training.to_dict()

    def create_training_with_claim(
        self,
        task_id: str,
        training_task_id: str,
        user_id: str,
        input_dataset_ids: List[str],
        total_samples: int = 0,
        training_round: int = 1,
        target_id: Optional[str] = None,
        require_threshold: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Atomically claim pending samples and create sync training tracking."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                existing = session.exec(
                    select(ExternalSyncTrainingDB).where(
                        ExternalSyncTrainingDB.training_task_id == training_task_id
                    )
                ).first()
                if existing:
                    session.rollback()
                    return existing.to_dict()

                task = session.exec(
                    select(ExternalSyncTaskDB)
                    .where(ExternalSyncTaskDB.task_id == task_id)
                    .with_for_update()
                ).first()
                if not task:
                    session.rollback()
                    return None
                if task.user_id != user_id:
                    raise PermissionError("External sync task ownership mismatch")
                if task.status in {
                    SyncStatus.DELETING,
                    SyncStatus.DELETING_CASCADE,
                }:
                    raise ValueError("Sync task deletion is in progress")
                if task.status in {
                    SyncStatus.TRAINING,
                    SyncStatus.LOADING_ADAPTER,
                }:
                    session.rollback()
                    return None

                previous_parent_status = task.status
                previous_training_task_id = task.current_training_id
                previous_target_status = None
                target_config_snapshot: Dict[str, Any] = {}

                if target_id:
                    targets = list(
                        session.exec(
                            select(ExternalSyncTrainingTargetDB)
                            .where(
                                ExternalSyncTrainingTargetDB.task_id == task_id,
                                ExternalSyncTrainingTargetDB.is_active.is_(True),
                            )
                            .order_by(
                                ExternalSyncTrainingTargetDB.priority,
                                ExternalSyncTrainingTargetDB.sort_order,
                                ExternalSyncTrainingTargetDB.id,
                            )
                            .with_for_update()
                        ).all()
                    )
                    if any(
                        target.status
                        in {
                            TrainingTargetStatus.TRAINING,
                            TrainingTargetStatus.LOADING_ADAPTER,
                        }
                        for target in targets
                    ):
                        session.rollback()
                        return None

                    target = next(
                        (item for item in targets if item.target_id == target_id),
                        None,
                    )
                    eligible_statuses = {
                        TrainingTargetStatus.IDLE,
                        TrainingTargetStatus.READY,
                    }
                    if not require_threshold:
                        eligible_statuses.add(TrainingTargetStatus.ERROR)
                    if not target or target.status not in eligible_statuses:
                        session.rollback()
                        return None

                    claimed_sample_count = max(
                        int(target.pending_training_samples or 0),
                        0,
                    )
                    threshold = max(int(target.training_threshold or 0), 0)
                    if require_threshold and (
                        threshold <= 0 or claimed_sample_count < threshold
                    ):
                        session.rollback()
                        return None

                    previous_target_status = target.status
                    previous_training_task_id = target.current_training_id
                    target_config_snapshot = {
                        "target_id": target.target_id,
                        "task_id": target.task_id,
                        "target_name": target.target_name,
                        "model_type": target.model_type,
                        "data_phase": target.data_phase,
                        "training_method": target.training_method,
                        "training_config": dict(target.training_config or {}),
                        "base_model_path": target.base_model_path,
                        "base_deployment_id": target.base_deployment_id,
                    }
                    target.status = TrainingTargetStatus.TRAINING
                    target.pending_training_samples = 0
                    target.current_training_id = training_task_id
                    target.updated_at = _utcnow_naive()
                    session.add(target)
                else:
                    claimed_sample_count = max(
                        int(task.pending_training_samples or 0),
                        0,
                    )
                    threshold = max(int(task.training_threshold or 0), 0)
                    if require_threshold and (
                        threshold <= 0 or claimed_sample_count < threshold
                    ):
                        session.rollback()
                        return None
                    task.pending_training_samples = 0

                self._validate_transition(
                    task.status,
                    SyncStatus.TRAINING,
                    self._TASK_STATUS_TRANSITIONS,
                    "sync task",
                )
                task.status = SyncStatus.TRAINING
                task.current_training_id = training_task_id
                task.error_message = None
                task.updated_at = _utcnow_naive()

                training = ExternalSyncTrainingDB(
                    task_id=task_id,
                    training_task_id=training_task_id,
                    user_id=user_id,
                    input_dataset_ids=input_dataset_ids,
                    total_samples=total_samples,
                    training_round=training_round,
                    previous_training_task_id=previous_training_task_id,
                    target_id=target_id,
                    claimed_sample_count=claimed_sample_count,
                    previous_target_status=previous_target_status,
                    previous_parent_status=previous_parent_status,
                    claim_reconciled=False,
                    target_config_snapshot=target_config_snapshot,
                )
                session.add(task)
                session.add(training)
                session.commit()
                session.refresh(training)
                return training.to_dict()
            except Exception:
                session.rollback()
                raise

    def complete_training_claim(
        self,
        training_task_id: str,
    ) -> Dict[str, Any]:
        """Consume a training claim once and enter adapter-loading state."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))
                training = session.exec(
                    select(ExternalSyncTrainingDB)
                    .where(
                        ExternalSyncTrainingDB.training_task_id
                        == training_task_id
                    )
                    .with_for_update()
                ).first()
                if not training:
                    session.rollback()
                    return {"tracking_found": False, "completed": False}
                if training.claim_reconciled:
                    session.rollback()
                    return {
                        "tracking_found": True,
                        "completed": False,
                        "already_completed": training.status
                        != SyncTrainingStatus.FAILED,
                    }
                if training.status != SyncTrainingStatus.PENDING:
                    session.rollback()
                    return {"tracking_found": True, "completed": False}

                task = session.exec(
                    select(ExternalSyncTaskDB)
                    .where(ExternalSyncTaskDB.task_id == training.task_id)
                    .with_for_update()
                ).first()
                deletion_pending = bool(
                    task and task.status in _SYNC_DELETING_STATUSES
                )
                if (
                    task
                    and not deletion_pending
                    and task.current_training_id == training_task_id
                ):
                    task.status = SyncStatus.LOADING_ADAPTER
                    task.error_message = None
                    task.updated_at = _utcnow_naive()
                    session.add(task)

                if training.target_id and not deletion_pending:
                    target = session.exec(
                        select(ExternalSyncTrainingTargetDB)
                        .where(
                            ExternalSyncTrainingTargetDB.target_id
                            == training.target_id
                        )
                        .with_for_update()
                    ).first()
                    if target and target.current_training_id == training_task_id:
                        target.status = TrainingTargetStatus.LOADING_ADAPTER
                        target.updated_at = _utcnow_naive()
                        session.add(target)

                training.status = SyncTrainingStatus.COMPLETED
                training.claim_reconciled = True
                training.completed_at = _utcnow_naive()
                session.add(training)
                session.commit()
                return {
                    "tracking_found": True,
                    "completed": True,
                    "deletion_pending": deletion_pending,
                }
            except Exception:
                session.rollback()
                raise

    def fail_training_and_restore_claim(
        self,
        training_task_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Fail a pending sync training and restore its claimed samples once."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))
                training = session.exec(
                    select(ExternalSyncTrainingDB)
                    .where(
                        ExternalSyncTrainingDB.training_task_id
                        == training_task_id
                    )
                    .with_for_update()
                ).first()
                if not training:
                    session.rollback()
                    return {"tracking_found": False, "recovered": False}
                if training.claim_reconciled:
                    session.rollback()
                    return {
                        "tracking_found": True,
                        "recovered": False,
                        "already_recovered": training.status
                        == SyncTrainingStatus.FAILED,
                    }
                if training.status != SyncTrainingStatus.PENDING:
                    session.rollback()
                    return {"tracking_found": True, "recovered": False}

                restored_sample_count = max(
                    int(training.claimed_sample_count or 0),
                    0,
                )
                task = session.exec(
                    select(ExternalSyncTaskDB)
                    .where(ExternalSyncTaskDB.task_id == training.task_id)
                    .with_for_update()
                ).first()
                deletion_pending = bool(
                    task and task.status in _SYNC_DELETING_STATUSES
                )

                if training.target_id:
                    target = session.exec(
                        select(ExternalSyncTrainingTargetDB)
                        .where(
                            ExternalSyncTrainingTargetDB.target_id
                            == training.target_id
                        )
                        .with_for_update()
                    ).first()
                    if not target:
                        raise RuntimeError(
                            "Training target disappeared before claim recovery"
                        )
                    target.pending_training_samples = max(
                        int(target.pending_training_samples or 0),
                        0,
                    ) + restored_sample_count
                    if target.current_training_id == training_task_id:
                        target.status = (
                            training.previous_target_status
                            or TrainingTargetStatus.IDLE
                        )
                        target.current_training_id = (
                            training.previous_training_task_id
                        )
                    target.updated_at = _utcnow_naive()
                    session.add(target)
                elif task:
                    task.pending_training_samples = max(
                        int(task.pending_training_samples or 0),
                        0,
                    ) + restored_sample_count

                if task and task.current_training_id == training_task_id:
                    task.current_training_id = training.previous_training_task_id
                    if not deletion_pending:
                        task.status = SyncStatus.ERROR
                        task.error_message = reason
                    task.updated_at = _utcnow_naive()
                    session.add(task)

                training.status = SyncTrainingStatus.FAILED
                training.claim_reconciled = True
                training.completed_at = _utcnow_naive()
                session.add(training)
                session.commit()
                return {
                    "tracking_found": True,
                    "recovered": True,
                    "restored_sample_count": restored_sample_count,
                    "deletion_pending": deletion_pending,
                }
            except Exception:
                session.rollback()
                raise

    def get_training_by_task_id(self, training_task_id: str) -> Optional[Dict[str, Any]]:
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTrainingDB).where(
                ExternalSyncTrainingDB.training_task_id == training_task_id
            )
            training = session.exec(stmt).first()
            return training.to_dict(mask_sensitive=False) if training else None

    def update_training_status(
        self,
        training_task_id: str,
        status: str,
        output_adapter_path: Optional[str] = None,
        output_model_registry_id: Optional[str] = None,
        loaded_adapter_name: Optional[str] = None,
        loaded_adapter_id: Optional[str] = None,
    ):
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTrainingDB).where(
                ExternalSyncTrainingDB.training_task_id == training_task_id
            )
            training = session.exec(stmt).first()
            if training:
                self._validate_transition(
                    training.status,
                    status,
                    self._TRAINING_STATUS_TRANSITIONS,
                    "sync training",
                )
                training.status = status
                if output_adapter_path:
                    training.output_adapter_path = output_adapter_path
                if output_model_registry_id:
                    training.output_model_registry_id = output_model_registry_id
                if loaded_adapter_name:
                    training.loaded_adapter_name = loaded_adapter_name
                if loaded_adapter_id:
                    training.loaded_adapter_id = loaded_adapter_id
                if status in (
                    SyncTrainingStatus.COMPLETED,
                    SyncTrainingStatus.ADAPTER_LOADED,
                    SyncTrainingStatus.FAILED,
                    SyncTrainingStatus.ADAPTER_LOAD_FAILED,
                ):
                    training.completed_at = _utcnow_naive()
                session.add(training)
                session.commit()

    def mark_all_trainings_adapter_unloaded(
        self, task_id: str, exclude_training_task_id: Optional[str] = None
    ) -> int:
        """Mark ALL training records with adapter_loaded status as adapter_unloaded.

        Used when replacing adapters or unloading all adapters for a sync task.

        Args:
            task_id: Sync task id.
            exclude_training_task_id: The training this replace flow is about to
                load — it is excluded so the unload phase can't race with the
                concurrent load and mark it unloaded after it was loaded.

        Returns:
            Number of records updated.
        """
        with Session(self._get_engine()) as session:
            # 原子 UPDATE（条件 status=ADAPTER_LOADED）：并发 replace/加载流程
            # 不会因读-改-写窗口互相覆盖（LOADED -> UNLOADED 是唯一合法转移，
            # 由 WHERE 条件保证）
            stmt = (
                update(ExternalSyncTrainingDB)
                .where(
                    ExternalSyncTrainingDB.task_id == task_id,
                    ExternalSyncTrainingDB.status == SyncTrainingStatus.ADAPTER_LOADED,
                )
                .values(status=SyncTrainingStatus.ADAPTER_UNLOADED)
            )
            if exclude_training_task_id:
                stmt = stmt.where(
                    ExternalSyncTrainingDB.training_task_id
                    != exclude_training_task_id
                )
            result = session.exec(stmt)
            session.commit()
            count = result.rowcount or 0
            if count:
                logger.info(
                    f"Marked {count} training record(s) as adapter_unloaded for task {task_id[:8]}"
                )
            return count

    def get_latest_training(self, task_id: str) -> Optional[Dict[str, Any]]:
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalSyncTrainingDB)
                .where(ExternalSyncTrainingDB.task_id == task_id)
                .order_by(ExternalSyncTrainingDB.training_round.desc())
                .limit(1)
            )
            training = session.exec(stmt).first()
            return training.to_dict() if training else None

    def list_trainings(
        self, task_id: str, status: Optional[str] = None, limit: int = 50, offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        with Session(self._get_engine()) as session:
            count_stmt = select(func.count()).select_from(ExternalSyncTrainingDB).where(ExternalSyncTrainingDB.task_id == task_id)
            stmt = select(ExternalSyncTrainingDB).where(ExternalSyncTrainingDB.task_id == task_id)
            if status:
                count_stmt = count_stmt.where(ExternalSyncTrainingDB.status == status)
                stmt = stmt.where(ExternalSyncTrainingDB.status == status)
            total = session.exec(count_stmt).one()

            stmt = (
                stmt.order_by(
                    ExternalSyncTrainingDB.created_at.desc(),
                    ExternalSyncTrainingDB.id.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
            trainings = session.exec(stmt).all()
            return [t.to_dict() for t in trainings], total

    def list_pending_trainings(
        self,
        *,
        limit: int = 1000,
        offset: int = 0,
        after_id: Optional[int] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """List unreconciled sync trainings for startup recovery."""
        bounded_limit = max(1, min(int(limit), 1000))
        bounded_offset = max(int(offset), 0)
        with Session(self._get_engine()) as session:
            filters = [
                ExternalSyncTrainingDB.status == SyncTrainingStatus.PENDING,
                ExternalSyncTrainingDB.claim_reconciled.is_(False),
            ]
            if after_id is not None:
                filters.append(
                    ExternalSyncTrainingDB.id > max(int(after_id), 0)
                )
            total = session.exec(
                select(func.count())
                .select_from(ExternalSyncTrainingDB)
                .where(*filters)
            ).one()
            stmt = (
                select(ExternalSyncTrainingDB)
                .where(*filters)
                .order_by(ExternalSyncTrainingDB.id.asc())
                .limit(bounded_limit)
            )
            if after_id is None:
                stmt = stmt.offset(bounded_offset)
            rows = session.exec(stmt).all()
            return [row.to_dict() for row in rows], int(total or 0)

    def delete_training_tracking(self, training_task_id: str) -> bool:
        """Delete sync training tracking record when a training task is deleted."""
        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTrainingDB).where(
                ExternalSyncTrainingDB.training_task_id == training_task_id
            )
            record = session.exec(stmt).first()
            if record:
                session.delete(record)
                session.commit()
                logger.info(f"Deleted sync training tracking for task {training_task_id[:8]}")
                return True
            return False

    # ── Training Target CRUD ──────────────────────────────────

    @staticmethod
    def _lock_mutable_sync_parent(
        session: Session,
        task_id: str,
        expected_user_id: Optional[str] = None,
    ) -> ExternalSyncTaskDB:
        """Lock the target's parent and reject orphan or deletion-racing writes."""
        parent = session.exec(
            select(ExternalSyncTaskDB)
            .where(ExternalSyncTaskDB.task_id == task_id)
            .with_for_update()
        ).first()
        if parent is None:
            raise ValueError("Sync task not found during target mutation")
        if expected_user_id is not None and parent.user_id != expected_user_id:
            raise ValueError("Sync task ownership mismatch during target mutation")
        if parent.status in _SYNC_DELETING_STATUSES:
            raise ValueError("Sync task deletion is in progress")
        return parent

    def create_training_target(
        self,
        task_id: str,
        target_name: str,
        target_id: Optional[str] = None,
        model_type: str = "embedding",
        data_phase: str = "final",
        training_method: str = "sft",
        training_config: Optional[Dict[str, Any]] = None,
        base_model_path: str = "",
        base_deployment_id: Optional[str] = None,
        training_threshold: int = 1000,
        priority: int = 0,
        sort_order: int = 0,
        expected_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a training target for a sync task."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        engine = self._get_engine()
        with Session(engine) as session:
            if engine.dialect.name == "sqlite":
                session.exec(text("BEGIN IMMEDIATE"))
            self._lock_mutable_sync_parent(session, task_id, expected_user_id)
            target = ExternalSyncTrainingTargetDB(
                target_id=target_id or str(uuid.uuid4()),
                task_id=task_id,
                target_name=target_name,
                model_type=model_type,
                data_phase=data_phase,
                training_method=training_method,
                training_config=training_config or {},
                base_model_path=base_model_path,
                base_deployment_id=base_deployment_id,
                training_threshold=training_threshold,
                priority=priority,
                sort_order=sort_order,
            )
            session.add(target)
            session.commit()
            session.refresh(target)
            return target.to_dict()

    def list_training_targets(
        self,
        task_id: str,
        is_active: Optional[bool] = True,
        data_phase: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List training targets for a sync task."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.task_id == task_id
            )
            if is_active is not None:
                stmt = stmt.where(ExternalSyncTrainingTargetDB.is_active == is_active)
            if data_phase:
                stmt = stmt.where(ExternalSyncTrainingTargetDB.data_phase == data_phase)
            if status:
                stmt = stmt.where(ExternalSyncTrainingTargetDB.status == status)
            stmt = stmt.order_by(ExternalSyncTrainingTargetDB.sort_order)
            return [t.to_dict() for t in session.exec(stmt).all()]

    def list_training_targets_raw(
        self,
        task_id: str,
        is_active: Optional[bool] = True,
        data_phase: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List training targets with unredacted config for internal workers."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.task_id == task_id
            )
            if is_active is not None:
                stmt = stmt.where(
                    ExternalSyncTrainingTargetDB.is_active == is_active
                )
            if data_phase:
                stmt = stmt.where(
                    ExternalSyncTrainingTargetDB.data_phase == data_phase
                )
            if status:
                stmt = stmt.where(ExternalSyncTrainingTargetDB.status == status)
            stmt = stmt.order_by(ExternalSyncTrainingTargetDB.sort_order)
            return [
                target.to_dict(mask_sensitive=False)
                for target in session.exec(stmt).all()
            ]

    def get_training_target(self, target_id: str) -> Optional[Dict[str, Any]]:
        """Get a training target by target_id."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
            target = session.exec(stmt).first()
            return target.to_dict() if target else None

    def get_training_target_raw(self, target_id: str) -> Optional[Dict[str, Any]]:
        """Get a training target with unredacted config for internal workers."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        with Session(self._get_engine()) as session:
            target = session.exec(
                select(ExternalSyncTrainingTargetDB).where(
                    ExternalSyncTrainingTargetDB.target_id == target_id
                )
            ).first()
            return target.to_dict(mask_sensitive=False) if target else None

    def update_training_target(
        self,
        target_id: str,
        *,
        task_id: Optional[str] = None,
        expected_user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Update a training target."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        engine = self._get_engine()
        with Session(engine) as session:
            if engine.dialect.name == "sqlite":
                session.exec(text("BEGIN IMMEDIATE"))
            resolved_task_id = task_id
            if resolved_task_id is None:
                resolved_task_id = session.exec(
                    select(ExternalSyncTrainingTargetDB.task_id).where(
                        ExternalSyncTrainingTargetDB.target_id == target_id
                    )
                ).first()
                if resolved_task_id is None:
                    return None
            self._lock_mutable_sync_parent(
                session,
                resolved_task_id,
                expected_user_id,
            )
            stmt = (
                select(ExternalSyncTrainingTargetDB)
                .where(
                    ExternalSyncTrainingTargetDB.target_id == target_id,
                    ExternalSyncTrainingTargetDB.task_id == resolved_task_id,
                )
                .with_for_update()
            )
            target = session.exec(stmt).first()
            if not target:
                return None

            new_status = kwargs.get("status")
            if new_status and new_status != target.status:
                self._validate_transition(
                    target.status, new_status,
                    self._TARGET_STATUS_TRANSITIONS, "training_target",
                )

            for key, value in kwargs.items():
                if hasattr(target, key):
                    setattr(target, key, value)
            target.updated_at = _utcnow_naive()
            session.commit()
            session.refresh(target)
            return target.to_dict()

    def claim_training_target(
        self,
        task_id: str,
        target_id: Optional[str] = None,
        require_threshold: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Atomically reserve one training target for task creation.

        The parent sync-task row serializes claims across workers. A conditional
        target update provides an additional compare-and-set guard, while SQLite
        uses ``BEGIN IMMEDIATE`` so the concurrency contract is testable there too.
        """
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                task_stmt = select(ExternalSyncTaskDB).where(
                    ExternalSyncTaskDB.task_id == task_id
                ).with_for_update()
                if not session.exec(task_stmt).first():
                    session.rollback()
                    return None

                target_stmt = select(ExternalSyncTrainingTargetDB).where(
                    ExternalSyncTrainingTargetDB.task_id == task_id,
                    ExternalSyncTrainingTargetDB.is_active.is_(True),
                ).order_by(
                    ExternalSyncTrainingTargetDB.priority,
                    ExternalSyncTrainingTargetDB.sort_order,
                    ExternalSyncTrainingTargetDB.id,
                ).with_for_update()
                targets = list(session.exec(target_stmt).all())

                if any(
                    item.status in (
                        TrainingTargetStatus.TRAINING,
                        TrainingTargetStatus.LOADING_ADAPTER,
                    )
                    for item in targets
                ):
                    session.rollback()
                    return None

                eligible_statuses = {
                    TrainingTargetStatus.IDLE,
                    TrainingTargetStatus.READY,
                }
                if not require_threshold:
                    eligible_statuses.add(TrainingTargetStatus.ERROR)

                candidates = [
                    item
                    for item in targets
                    if item.status in eligible_statuses
                    and (target_id is None or item.target_id == target_id)
                    and (
                        not require_threshold
                        or (
                            int(item.training_threshold or 0) > 0
                            and int(item.pending_training_samples or 0)
                            >= int(item.training_threshold or 0)
                        )
                    )
                ]
                if not candidates:
                    session.rollback()
                    return None

                selected = candidates[0]
                previous_status = selected.status
                previous_training_id = selected.current_training_id
                claimed_pending_samples = int(
                    selected.pending_training_samples or 0
                )
                result = session.exec(
                    update(ExternalSyncTrainingTargetDB)
                    .where(
                        ExternalSyncTrainingTargetDB.id == selected.id,
                        ExternalSyncTrainingTargetDB.status == previous_status,
                        ExternalSyncTrainingTargetDB.is_active.is_(True),
                    )
                    .values(
                        status=TrainingTargetStatus.TRAINING,
                        pending_training_samples=0,
                        updated_at=_utcnow_naive(),
                    )
                )
                if result.rowcount != 1:
                    session.rollback()
                    return None

                claimed = selected.model_dump()
                claimed["status"] = TrainingTargetStatus.TRAINING
                claimed["pending_training_samples"] = claimed_pending_samples
                claimed["_claim_previous_status"] = previous_status
                claimed["_claim_previous_training_id"] = previous_training_id
                claimed["_claim_pending_samples"] = claimed_pending_samples
                session.commit()
                return claimed
            except Exception:
                session.rollback()
                raise

    def release_training_target_claim(
        self,
        target_id: str,
        previous_status: str,
        previous_training_id: Optional[str] = None,
        claimed_pending_samples: int = 0,
    ) -> bool:
        """Release a claim and add back exactly the samples it consumed."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        restore_count = max(0, int(claimed_pending_samples or 0))
        with Session(self._get_engine()) as session:
            result = session.exec(
                update(ExternalSyncTrainingTargetDB)
                .where(
                    ExternalSyncTrainingTargetDB.target_id == target_id,
                    ExternalSyncTrainingTargetDB.status == TrainingTargetStatus.TRAINING,
                )
                .values(
                    status=previous_status,
                    current_training_id=previous_training_id,
                    pending_training_samples=(
                        ExternalSyncTrainingTargetDB.pending_training_samples
                        + restore_count
                    ),
                    updated_at=_utcnow_naive(),
                )
            )
            session.commit()
            return result.rowcount == 1

    def delete_training_target(
        self,
        target_id: str,
        *,
        task_id: Optional[str] = None,
        expected_user_id: Optional[str] = None,
    ) -> bool:
        """Delete a training target."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        engine = self._get_engine()
        with Session(engine) as session:
            if engine.dialect.name == "sqlite":
                session.exec(text("BEGIN IMMEDIATE"))
            resolved_task_id = task_id
            if resolved_task_id is None:
                resolved_task_id = session.exec(
                    select(ExternalSyncTrainingTargetDB.task_id).where(
                        ExternalSyncTrainingTargetDB.target_id == target_id
                    )
                ).first()
                if resolved_task_id is None:
                    return False
            self._lock_mutable_sync_parent(
                session,
                resolved_task_id,
                expected_user_id,
            )
            stmt = (
                select(ExternalSyncTrainingTargetDB)
                .where(
                    ExternalSyncTrainingTargetDB.target_id == target_id,
                    ExternalSyncTrainingTargetDB.task_id == resolved_task_id,
                )
                .with_for_update()
            )
            target = session.exec(stmt).first()
            if not target:
                return False
            pending_training = session.exec(
                select(ExternalSyncTrainingDB.id).where(
                    ExternalSyncTrainingDB.target_id == target_id,
                    ExternalSyncTrainingDB.status == SyncTrainingStatus.PENDING,
                )
            ).first()
            if target.status in {
                TrainingTargetStatus.TRAINING,
                TrainingTargetStatus.LOADING_ADAPTER,
            } or pending_training is not None:
                raise ValueError(
                    "Training target has an active training and cannot be deleted"
                )
            session.delete(target)
            session.commit()
            return True

    def increment_target_pending_samples(self, target_id: str, count: int) -> None:
        """Atomically increment pending/total training samples for a target.

        Uses a single ``UPDATE ... SET col = col + :count`` so concurrent
        callbacks (parallel sync cycles, generation completions) can't lose
        updates the way a read-modify-write would. Mirrors
        increment_pending_training_samples.
        """
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        with Session(self._get_engine()) as session:
            session.exec(
                update(ExternalSyncTrainingTargetDB)
                .where(ExternalSyncTrainingTargetDB.target_id == target_id)
                .values(
                    pending_training_samples=(
                        ExternalSyncTrainingTargetDB.pending_training_samples + count
                    ),
                    total_training_samples=(
                        ExternalSyncTrainingTargetDB.total_training_samples + count
                    ),
                    updated_at=_utcnow_naive(),
                )
            )
            session.commit()

    def reset_target_pending_samples(self, target_id: str) -> None:
        """Reset pending_training_samples for a specific target to 0."""
        from ..entities.external_sync_entity import ExternalSyncTrainingTargetDB

        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
            target = session.exec(stmt).first()
            if target:
                target.pending_training_samples = 0
                target.updated_at = _utcnow_naive()
                session.commit()

    def get_all_completed_qa_datasets(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all completed QA datasets for a sync task (Phase 1 outputs)."""
        from ..entities.external_sync_entity import ExternalSyncGenerationDB

        with Session(self._get_engine()) as session:
            stmt = select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.task_id == task_id,
                ExternalSyncGenerationDB.status == SyncGenerationStatus.COMPLETED,
                ExternalSyncGenerationDB.qa_dataset_id.isnot(None),
                ExternalSyncGenerationDB.disabled.is_(False),
            )
            gens = session.exec(stmt).all()
            return [
                {
                    "dataset_id": g.qa_dataset_id,
                    "sample_count": g.qa_sample_count,
                    "generation_task_id": g.generation_task_id,
                }
                for g in gens
            ]

    def update_generation_qa_output(
        self, generation_task_id: str, qa_dataset_id: str, qa_sample_count: int
    ) -> Dict[str, Any]:
        """Atomically record QA output without crediting pending generations."""
        from ..entities.external_sync_entity import (
            ExternalSyncGenerationDB,
            ExternalSyncTrainingTargetDB,
        )

        result: Dict[str, Any] = {
            "tracking_found": False,
            "recorded": False,
            "already_recorded": False,
            "conflict": False,
            "task_id": None,
            "generation_status": None,
            "credited_sample_count": 0,
            "credited_target_count": 0,
            "schedule_training": False,
        }
        sample_count = max(int(qa_sample_count or 0), 0)
        engine = self._get_engine()
        with Session(engine) as session:
            try:
                if engine.dialect.name == "sqlite":
                    session.exec(text("BEGIN IMMEDIATE"))

                _task, generation = self._lock_generation_scope(
                    session,
                    generation_task_id,
                )
                if not generation:
                    session.rollback()
                    return result

                result.update(
                    tracking_found=True,
                    task_id=generation.task_id,
                    generation_status=generation.status,
                )

                if generation.qa_dataset_id is not None:
                    result["already_recorded"] = True
                    result["conflict"] = (
                        generation.qa_dataset_id != qa_dataset_id
                        or int(generation.qa_sample_count or 0) != sample_count
                    )
                    session.rollback()
                    return result

                if generation.status == SyncGenerationStatus.FAILED:
                    session.rollback()
                    return result

                recording = session.exec(
                    update(ExternalSyncGenerationDB)
                    .where(
                        ExternalSyncGenerationDB.id == generation.id,
                        ExternalSyncGenerationDB.qa_dataset_id.is_(None),
                        ExternalSyncGenerationDB.status.in_(
                            [
                                SyncGenerationStatus.PENDING,
                                SyncGenerationStatus.COMPLETED,
                            ]
                        ),
                    )
                    .values(
                        qa_dataset_id=qa_dataset_id,
                        qa_sample_count=sample_count,
                    )
                )
                if recording.rowcount != 1:
                    session.rollback()
                    return result

                credited_targets = []
                if (
                    generation.status == SyncGenerationStatus.COMPLETED
                    and qa_dataset_id
                    and sample_count > 0
                ):
                    credited_targets = list(
                        session.exec(
                            select(ExternalSyncTrainingTargetDB)
                            .where(
                                ExternalSyncTrainingTargetDB.task_id
                                == generation.task_id,
                                ExternalSyncTrainingTargetDB.data_phase == "qa",
                                ExternalSyncTrainingTargetDB.is_active.is_(True),
                            )
                            .with_for_update()
                        ).all()
                    )
                    updated_at = _utcnow_naive()
                    for target in credited_targets:
                        target.pending_training_samples = int(
                            target.pending_training_samples or 0
                        ) + sample_count
                        target.total_training_samples = int(
                            target.total_training_samples or 0
                        ) + sample_count
                        target.updated_at = updated_at
                        session.add(target)

                session.commit()
                result.update(
                    recorded=True,
                    credited_sample_count=(
                        sample_count if credited_targets else 0
                    ),
                    credited_target_count=len(credited_targets),
                    schedule_training=bool(credited_targets),
                )
                return result
            except Exception:
                session.rollback()
                raise


# Global singleton
external_sync_service = ExternalSyncService()
