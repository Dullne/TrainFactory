"""Atomic staging and publication for generation-attempt products."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from sqlalchemy import delete, update
from sqlmodel import select

from ...core.time_utils import now_naive
from ...enums.dataset_status import DatasetStatus
from ...utils.path_utils import hash_path
from ..database import get_session
from ..entities.dataset_asset_entity import DatasetAssetDB
from ..entities.dataset_entity import DatasetDB
from ..entities.dataset_lineage_entity import (
    DatasetLineageEdgeDB,
    build_dataset_lineage_edge_key,
)
from ..entities.generation_task_entity import GenerationStatus, GenerationTaskDB
from ..entities.milvus_collection_entity import (
    CollectionDatasetLinkDB,
    MilvusCollectionDB,
)
from .milvus_collection_service import _lock_sync_namespace_tasks
from .runtime_dependency_service import (
    lock_embedding_model_config_for_binding,
)

logger = logging.getLogger(__name__)

_DATASET_BINDING_FIELDS = {
    "output_dataset_id",
    "qa_dataset_id",
    "qa_filtered_dataset_id",
    "deep_eval_dataset_id",
}


class GenerationPublicationService:
    """Own all DB-visible generation publication mutations."""

    def __init__(self, session_factory: Optional[Callable[[], Any]] = None):
        self._session_factory = session_factory

    def _session(self):
        return (self._session_factory or get_session)()

    def stage_dataset(
        self,
        *,
        task_id: str,
        expected_run_token: str,
        dataset_id: str,
        dataset_name: str,
        storage_path: str,
        dataset_type: str,
        usage: Optional[str],
        model_type: Optional[list[str]],
        source_dataset_id: Optional[str],
        relation_type: str,
        file_format: str,
        num_rows: Optional[int],
        file_size: Optional[int],
        user_id: Optional[str],
        description: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create or repair one non-consumable product in one transaction."""
        if not expected_run_token:
            return None
        storage_hash = hash_path(storage_path)
        try:
            with self._session() as session:
                task = session.exec(
                    select(GenerationTaskDB)
                    .where(
                        GenerationTaskDB.task_id == task_id,
                        GenerationTaskDB.status == GenerationStatus.PUBLISHING,
                        GenerationTaskDB.run_token == expected_run_token,
                    )
                    .with_for_update()
                ).first()
                if task is None or task.user_id != user_id:
                    session.rollback()
                    return None

                existing = session.exec(
                    select(DatasetDB)
                    .where(
                        (
                            (DatasetDB.dataset_id == dataset_id)
                            | (DatasetDB.storage_path_hash == storage_hash)
                        )
                    )
                    .with_for_update()
                ).first()
                if existing is not None:
                    if not (
                        existing.dataset_id == dataset_id
                        and existing.status == DatasetStatus.STAGING.value
                        and existing.source_task_type == "generation"
                        and existing.source_task_id == task_id
                        and existing.generation_run_token == expected_run_token
                        and existing.user_id == user_id
                        and existing.storage_path == storage_path
                    ):
                        session.rollback()
                        return None
                    dataset = existing
                else:
                    dataset = DatasetDB(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name,
                        display_name=dataset_name,
                        description=description,
                        dataset_type=dataset_type,
                        usage=usage,
                        model_type=model_type,
                        source_type="generated",
                        source_dataset_id=source_dataset_id,
                        storage_path=storage_path,
                        storage_path_hash=storage_hash,
                        file_format=file_format,
                        source_task_type="generation",
                        source_task_id=task_id,
                        generation_run_token=expected_run_token,
                        num_rows=num_rows,
                        file_size=file_size,
                        extra_metadata=extra_metadata,
                        status=DatasetStatus.STAGING.value,
                        user_id=user_id,
                    )
                    session.add(dataset)

                asset = session.exec(
                    select(DatasetAssetDB).where(
                        DatasetAssetDB.dataset_id == dataset_id,
                        DatasetAssetDB.asset_type == "data",
                        DatasetAssetDB.storage_uri == storage_path,
                    )
                ).first()
                if asset is None:
                    session.add(
                        DatasetAssetDB(
                            dataset_id=dataset_id,
                            asset_type="data",
                            storage_uri=storage_path,
                            file_format=file_format,
                            row_count=num_rows,
                            byte_size=file_size,
                        )
                    )

                edge_key = build_dataset_lineage_edge_key(
                    source_dataset_id,
                    dataset_id,
                    relation_type,
                )
                edge = session.exec(
                    select(DatasetLineageEdgeDB).where(
                        DatasetLineageEdgeDB.edge_key == edge_key
                    )
                ).first()
                if edge is None:
                    session.add(
                        DatasetLineageEdgeDB(
                            edge_key=edge_key,
                            from_dataset_id=source_dataset_id,
                            to_dataset_id=dataset_id,
                            relation_type=relation_type,
                            op_task_type="generation",
                            op_task_id=task_id,
                            op_params=None,
                        )
                    )
                elif not (
                    edge.to_dataset_id == dataset_id
                    and edge.op_task_type == "generation"
                    and edge.op_task_id == task_id
                ):
                    session.rollback()
                    return None

                # Recheck durable ownership in the same write transaction.  A
                # concurrent recovery/restart either wins first (rowcount=0) or
                # waits for this staging-only commit and can then compensate it.
                claim = session.exec(
                    update(GenerationTaskDB)
                    .where(
                        GenerationTaskDB.task_id == task_id,
                        GenerationTaskDB.status == GenerationStatus.PUBLISHING,
                        GenerationTaskDB.run_token == expected_run_token,
                    )
                    .values(status=GenerationStatus.PUBLISHING)
                )
                if claim.rowcount != 1:
                    session.rollback()
                    return None
                session.commit()
                return dataset_id
        except Exception:
            logger.exception(
                "Failed to stage generation dataset %s for %s",
                dataset_id,
                task_id,
            )
            return None

    def complete_publication(
        self,
        task_id: str,
        *,
        expected_run_token: str,
        dataset_bindings: Dict[str, Optional[str]],
        milvus_registration: Optional[Dict[str, Any]] = None,
        artifact_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Atomically activate the complete attempt and finish its task."""
        if not expected_run_token:
            return False
        if not set(dataset_bindings).issubset(_DATASET_BINDING_FIELDS):
            raise ValueError("Invalid generation dataset binding")
        normalized_bindings = {
            field: dataset_id
            for field, dataset_id in dataset_bindings.items()
            if dataset_id
        }
        expected_dataset_ids = set(normalized_bindings.values())
        try:
            with self._session() as session:
                if milvus_registration:
                    embedding_config_id = milvus_registration.get(
                        "embedding_config_id"
                    )
                    if embedding_config_id:
                        lock_embedding_model_config_for_binding(
                            session,
                            embedding_config_id,
                            expected_user_id=milvus_registration.get("user_id"),
                        )
                task = session.exec(
                    select(GenerationTaskDB)
                    .where(
                        GenerationTaskDB.task_id == task_id,
                        GenerationTaskDB.status == GenerationStatus.PUBLISHING,
                        GenerationTaskDB.run_token == expected_run_token,
                    )
                    .with_for_update()
                ).first()
                if task is None:
                    session.rollback()
                    return False

                staged = list(
                    session.exec(
                        select(DatasetDB)
                        .where(
                            DatasetDB.source_task_type == "generation",
                            DatasetDB.source_task_id == task_id,
                            DatasetDB.generation_run_token == expected_run_token,
                            DatasetDB.status == DatasetStatus.STAGING.value,
                        )
                        .with_for_update()
                    ).all()
                )
                if {dataset.dataset_id for dataset in staged} != expected_dataset_ids:
                    session.rollback()
                    return False
                if any(dataset.user_id != task.user_id for dataset in staged):
                    session.rollback()
                    return False

                for dataset in staged:
                    asset = session.exec(
                        select(DatasetAssetDB).where(
                            DatasetAssetDB.dataset_id == dataset.dataset_id,
                            DatasetAssetDB.asset_type == "data",
                        )
                    ).first()
                    lineage = session.exec(
                        select(DatasetLineageEdgeDB).where(
                            DatasetLineageEdgeDB.to_dataset_id
                            == dataset.dataset_id,
                            DatasetLineageEdgeDB.op_task_type == "generation",
                            DatasetLineageEdgeDB.op_task_id == task_id,
                        )
                    ).first()
                    if asset is None or lineage is None:
                        session.rollback()
                        return False
                    expected_reference = dataset.storage_uri or dataset.storage_path
                    if asset.storage_uri != expected_reference:
                        session.rollback()
                        return False

                if milvus_registration:
                    if not self._prepare_collection_registration(
                        session,
                        task,
                        expected_run_token,
                        milvus_registration,
                    ):
                        session.rollback()
                        return False

                for dataset in staged:
                    dataset.status = DatasetStatus.READY.value
                    dataset.updated_at = now_naive()
                    session.add(dataset)

                values: Dict[str, Any] = {
                    "status": GenerationStatus.COMPLETED,
                    "completed_at": now_naive(),
                    "error_message": None,
                }
                values.update(normalized_bindings)
                if artifact_updates:
                    allowed_artifact_fields = {
                        "output_path",
                        "output_sample_count",
                        "qa_output_path",
                        "qa_filtered_path",
                        "deep_eval_path",
                        "filter_stats",
                        "milvus_collection",
                        "processed_docs",
                        "total_docs",
                        "progress",
                    }
                    if not set(artifact_updates).issubset(allowed_artifact_fields):
                        raise ValueError("Invalid publication artifact update")
                    values.update(artifact_updates)
                completed = session.exec(
                    update(GenerationTaskDB)
                    .where(
                        GenerationTaskDB.task_id == task_id,
                        GenerationTaskDB.status == GenerationStatus.PUBLISHING,
                        GenerationTaskDB.run_token == expected_run_token,
                    )
                    .values(**values)
                )
                if completed.rowcount != 1:
                    session.rollback()
                    return False
                session.commit()
                return True
        except Exception:
            logger.exception(
                "Generation publication transaction failed for %s",
                task_id,
            )
            return False

    def _prepare_collection_registration(
        self,
        session,
        task: GenerationTaskDB,
        run_token: str,
        registration: Dict[str, Any],
    ) -> bool:
        collection_name = registration.get("collection_name")
        dataset_id = registration.get("dataset_id")
        user_id = registration.get("user_id")
        if not collection_name or not dataset_id or user_id != task.user_id:
            return False
        _sync_tasks, namespace_claimants = _lock_sync_namespace_tasks(
            session,
            collection_name,
        )
        if namespace_claimants:
            return False
        source = session.exec(
            select(DatasetDB).where(
                DatasetDB.dataset_id == dataset_id,
                DatasetDB.status == DatasetStatus.READY.value,
                DatasetDB.user_id == user_id,
            )
        ).first()
        if source is None:
            return False

        collection = session.exec(
            select(MilvusCollectionDB)
            .where(MilvusCollectionDB.collection_name == collection_name)
            .with_for_update()
        ).first()
        if collection is None:
            collection = MilvusCollectionDB(
                collection_name=collection_name,
                embedding_config_id=registration.get("embedding_config_id"),
                embedding_model=registration.get("embedding_model"),
                embedding_endpoint=registration.get("embedding_endpoint"),
                dim=registration.get("dim") or 1024,
                metric_type=registration.get("metric_type") or "COSINE",
                hybrid_enabled=bool(registration.get("hybrid_enabled", False)),
                status="creating",
                generation_task_id=task.task_id,
                generation_run_token=run_token,
                user_id=user_id,
            )
            session.add(collection)
        elif collection.status == "active":
            if (
                collection.user_id != user_id
                or collection.deletion_owner is not None
                or collection.sync_task_id is not None
            ):
                return False
            for field in (
                "embedding_config_id",
                "embedding_model",
                "embedding_endpoint",
            ):
                expected = registration.get(field)
                actual = getattr(collection, field)
                if expected is not None and actual != expected:
                    return False
            expected_dim = registration.get("dim")
            if expected_dim is not None and collection.dim != expected_dim:
                return False
            expected_metric = registration.get("metric_type")
            if (
                expected_metric is not None
                and str(collection.metric_type).upper()
                != str(expected_metric).upper()
            ):
                return False
            expected_hybrid = registration.get("hybrid_enabled")
            if (
                expected_hybrid is not None
                and collection.hybrid_enabled is not expected_hybrid
            ):
                return False
        elif not (
            collection.status == "creating"
            and collection.generation_task_id == task.task_id
            and collection.generation_run_token == run_token
            and collection.user_id == user_id
        ):
            return False

        link = session.exec(
            select(CollectionDatasetLinkDB)
            .where(
                CollectionDatasetLinkDB.collection_name == collection_name,
                CollectionDatasetLinkDB.dataset_id == dataset_id,
            )
            .with_for_update()
        ).first()
        if link is None:
            session.add(
                CollectionDatasetLinkDB(
                    collection_name=collection_name,
                    dataset_id=dataset_id,
                    dataset_name=(
                        registration.get("dataset_name")
                        or source.dataset_name
                    ),
                    task_id=task.task_id,
                    generation_run_token=run_token,
                )
            )
        # An existing (collection, dataset) link is an already-published
        # durable relation.  Reuse it read-only; never seize or rewrite its
        # original producer provenance during another generation attempt.
        if collection.status == "creating":
            collection.status = "active"
            collection.updated_at = now_naive()
            session.add(collection)
        return True

    def compensate_attempt(
        self,
        *,
        task_id: str,
        expected_run_token: Optional[str],
        user_id: Optional[str],
        recovery_run_token: Optional[str] = None,
    ) -> bool:
        """Delete only exact-token staging metadata in one transaction."""
        try:
            with self._session() as session:
                task_query = select(GenerationTaskDB).where(
                    GenerationTaskDB.task_id == task_id
                )
                task = session.exec(task_query.with_for_update()).first()
                owns_live_attempt = bool(
                    task
                    and task.status == GenerationStatus.PUBLISHING
                    and task.run_token == expected_run_token
                )
                owns_recovery = bool(
                    task
                    and recovery_run_token
                    and task.status == GenerationStatus.RECOVERING
                    and task.run_token == recovery_run_token
                )
                owns_terminal_attempt = bool(
                    task
                    and task.status
                    in {
                        GenerationStatus.FAILED,
                        GenerationStatus.STOPPED,
                        GenerationStatus.COMPLETED,
                    }
                    and task.run_token == expected_run_token
                )
                if not (
                    owns_live_attempt
                    or owns_recovery
                    or owns_terminal_attempt
                ):
                    session.rollback()
                    return False

                staged = list(
                    session.exec(
                        select(DatasetDB)
                        .where(
                            DatasetDB.source_task_type == "generation",
                            DatasetDB.source_task_id == task_id,
                            DatasetDB.generation_run_token == expected_run_token,
                            DatasetDB.status == DatasetStatus.STAGING.value,
                            DatasetDB.user_id == user_id,
                        )
                        .with_for_update()
                    ).all()
                )
                if not staged:
                    # Nothing was made DB-visible for this attempt.  Treat the
                    # cleanup as idempotently complete so failure finalization
                    # cannot strand an empty PUBLISHING task.
                    session.rollback()
                    return True
                dataset_ids = {dataset.dataset_id for dataset in staged}
                linked = session.exec(
                    select(CollectionDatasetLinkDB.id).where(
                        CollectionDatasetLinkDB.dataset_id.in_(dataset_ids)
                    )
                ).first()
                if linked is not None:
                    session.rollback()
                    return False

                session.exec(
                    delete(DatasetAssetDB).where(
                        DatasetAssetDB.dataset_id.in_(dataset_ids)
                    )
                )
                session.exec(
                    delete(DatasetLineageEdgeDB).where(
                        DatasetLineageEdgeDB.to_dataset_id.in_(dataset_ids)
                    )
                )
                session.exec(
                    delete(DatasetDB).where(
                        DatasetDB.dataset_id.in_(dataset_ids),
                        DatasetDB.status == DatasetStatus.STAGING.value,
                        DatasetDB.generation_run_token == expected_run_token,
                    )
                )
                session.exec(
                    delete(MilvusCollectionDB).where(
                        MilvusCollectionDB.generation_task_id == task_id,
                        MilvusCollectionDB.generation_run_token
                        == expected_run_token,
                        MilvusCollectionDB.status == "creating",
                    )
                )
                session.commit()
                return True
        except Exception:
            logger.exception(
                "Failed to compensate generation staging for %s",
                task_id,
            )
            return False

    def has_staging_products(
        self,
        task_id: str,
        *,
        expected_run_token: str,
    ) -> bool:
        with self._session() as session:
            return (
                session.exec(
                    select(DatasetDB.dataset_id)
                    .where(
                        DatasetDB.source_task_type == "generation",
                        DatasetDB.source_task_id == task_id,
                        DatasetDB.generation_run_token
                        == expected_run_token,
                        DatasetDB.status == DatasetStatus.STAGING.value,
                    )
                    .limit(1)
                ).first()
                is not None
            )


generation_publication_service = GenerationPublicationService()
