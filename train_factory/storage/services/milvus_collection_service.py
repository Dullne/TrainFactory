"""Milvus collection registry service for database operations."""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Optional, Dict, Any, Iterable, List, Mapping, Tuple

from sqlalchemy.exc import IntegrityError
from sqlmodel import select, func

from train_factory.core.time_utils import now_naive

from ..database import get_session
from ..entities.external_sync_entity import ExternalSyncTaskDB
from ..entities.milvus_collection_entity import (
    MilvusCollectionDB,
    CollectionDatasetLinkDB,
)
from .dataset_service import lock_datasets_for_consumption
from .runtime_dependency_service import (
    RuntimeDependencyUnavailableError,
    lock_embedding_model_config_for_binding,
)

logger = logging.getLogger(__name__)


class MilvusCollectionUnavailableError(RuntimeError):
    """Raised when a collection cannot accept a new consumer."""


class MilvusCollectionDeletionOwnerConflictError(RuntimeError):
    """Raised when another logical operation owns a collection deletion fence."""


def _lock_embedding_config_for_collection(
    session,
    embedding_config_id: Optional[str],
    *,
    user_id: Optional[str],
):
    if not embedding_config_id:
        return None
    try:
        return lock_embedding_model_config_for_binding(
            session,
            embedding_config_id,
            expected_user_id=user_id,
        )
    except RuntimeDependencyUnavailableError as exc:
        raise MilvusCollectionUnavailableError(
            "Embedding config is unavailable"
        ) from exc


@dataclass(frozen=True)
class MilvusDeletionFenceAcquisition:
    """Describe fences created by one acquisition transaction."""

    collection_names: Tuple[str, ...]
    newly_fenced: Tuple[str, ...]
    created_placeholders: Tuple[str, ...]
    previous_creating: Tuple[str, ...] = ()


def _normalize_deletion_owner(deletion_owner: str) -> str:
    if not isinstance(deletion_owner, str) or not deletion_owner.strip():
        raise ValueError("A non-empty Milvus collection deletion owner is required")
    normalized = deletion_owner.strip()
    if len(normalized) > 128:
        raise ValueError("Milvus collection deletion owner is too long")
    return normalized


def lock_collection_for_consumption(
    session,
    collection_name: str,
    *,
    user_id: Optional[str],
) -> MilvusCollectionDB:
    """Lock one registry row and require it to remain available to consumers."""
    entry = session.exec(
        select(MilvusCollectionDB)
        .where(MilvusCollectionDB.collection_name == collection_name)
        .with_for_update()
    ).first()
    if (
        not entry
        or entry.status != "active"
        or entry.user_id != user_id
    ):
        raise MilvusCollectionUnavailableError(
            f"Milvus collection '{collection_name}' is unavailable"
        )
    return entry


def _lock_sync_namespace_tasks(session, collection_name: str):
    """Lock durable sync tasks and return claimants for one namespace."""
    from .external_sync_service import sync_task_reserves_collection_name

    tasks = list(
        session.exec(
            select(ExternalSyncTaskDB)
            .order_by(ExternalSyncTaskDB.task_id)
            .with_for_update()
        ).all()
    )
    claimants = [
        task
        for task in tasks
        if sync_task_reserves_collection_name(
            str(task.task_id),
            task.milvus_collection_name,
            collection_name,
        )
    ]
    return tasks, claimants


class MilvusCollectionService:
    """Service for managing Milvus collection registry and dataset links."""

    # ── Collection CRUD ──────────────────────────────────────────

    def register_collection(
        self,
        collection_name: str,
        embedding_config_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        dim: int = 1024,
        metric_type: str = "COSINE",
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        sync_task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a Milvus collection in the local database."""
        if sync_task_id is not None:
            if not isinstance(sync_task_id, str) or not sync_task_id.strip():
                raise ValueError("A non-empty external sync task ID is required")
            sync_task_id = sync_task_id.strip()
            if len(sync_task_id) > 36:
                raise ValueError("External sync task ID is too long")
        try:
            with get_session() as session:
                _lock_embedding_config_for_collection(
                    session,
                    embedding_config_id,
                    user_id=user_id,
                )
                sync_tasks, namespace_claimants = _lock_sync_namespace_tasks(
                    session,
                    collection_name,
                )
                if sync_task_id is not None:
                    sync_task = next(
                        (
                            task
                            for task in sync_tasks
                            if task.task_id == sync_task_id
                        ),
                        None,
                    )
                    claimant_ids = {
                        task.task_id for task in namespace_claimants
                    }
                    if (
                        sync_task is None
                        or sync_task.user_id != user_id
                        or claimant_ids != {sync_task_id}
                    ):
                        raise MilvusCollectionUnavailableError(
                            "External sync collection claimant is unavailable"
                        )
                elif namespace_claimants:
                    raise MilvusCollectionUnavailableError(
                        f"Milvus collection '{collection_name}' is reserved "
                        "by an external sync task"
                    )
                existing = session.exec(
                    select(MilvusCollectionDB)
                    .where(MilvusCollectionDB.collection_name == collection_name)
                    .with_for_update()
                ).first()
                if existing:
                    if existing.status != "active":
                        raise MilvusCollectionUnavailableError(
                            f"Milvus collection '{collection_name}' is unavailable"
                        )
                    if existing.user_id != user_id:
                        raise MilvusCollectionUnavailableError(
                            f"Milvus collection '{collection_name}' is unavailable"
                        )
                    if existing.sync_task_id != sync_task_id:
                        raise MilvusCollectionUnavailableError(
                            f"Milvus collection '{collection_name}' is unavailable"
                        )
                    logger.info("Collection '%s' already registered", collection_name)
                    return existing.to_dict()

                entry = MilvusCollectionDB(
                    collection_name=collection_name,
                    display_name=display_name or collection_name,
                    description=description,
                    embedding_config_id=embedding_config_id,
                    embedding_model=embedding_model,
                    embedding_endpoint=embedding_endpoint,
                    dim=dim,
                    metric_type=metric_type,
                    user_id=user_id,
                    sync_task_id=sync_task_id,
                )
                session.add(entry)
                session.commit()
                session.refresh(entry)
                logger.info(
                    "Registered collection '%s' (%s)",
                    collection_name,
                    entry.collection_id,
                )
                return entry.to_dict()
        except IntegrityError as exc:
            raise MilvusCollectionUnavailableError(
                f"Milvus collection '{collection_name}' changed concurrently"
            ) from exc

    def reserve_manual_collection(
        self,
        collection_name: str,
        embedding_config_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        dim: int = 1024,
        metric_type: str = "COSINE",
        hybrid_enabled: bool = False,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reserve or resume a matching manual creation intent.

        A ``creating`` row is durable recovery state after an uncertain remote
        mutation.  Only the same tenant and immutable remote configuration may
        resume it; presentation metadata is intentionally not rewritten.
        """
        if not isinstance(hybrid_enabled, bool):
            raise ValueError("Manual collection hybrid intent must be boolean")
        try:
            with get_session() as session:
                _lock_embedding_config_for_collection(
                    session,
                    embedding_config_id,
                    user_id=user_id,
                )
                _sync_tasks, namespace_claimants = _lock_sync_namespace_tasks(
                    session,
                    collection_name,
                )
                if namespace_claimants:
                    raise MilvusCollectionUnavailableError(
                        f"Milvus collection '{collection_name}' is reserved "
                        "by an external sync task"
                    )
                existing = session.exec(
                    select(MilvusCollectionDB)
                    .where(MilvusCollectionDB.collection_name == collection_name)
                    .with_for_update()
                ).first()
                if existing is not None:
                    matches_creation_intent = (
                        existing.status == "creating"
                        and existing.user_id == user_id
                        and existing.sync_task_id is None
                        and existing.deletion_owner is None
                        and existing.embedding_config_id == embedding_config_id
                        and existing.embedding_model == embedding_model
                        and existing.embedding_endpoint == embedding_endpoint
                        and existing.dim == dim
                        and str(existing.metric_type).upper()
                        == str(metric_type).upper()
                        and existing.hybrid_enabled is hybrid_enabled
                    )
                    if not matches_creation_intent:
                        raise MilvusCollectionUnavailableError(
                            f"Milvus collection '{collection_name}' is already "
                            "registered or has a conflicting creation in progress"
                        )
                    reservation = existing.to_dict()
                    reservation["_newly_created"] = False
                    return reservation
                entry = MilvusCollectionDB(
                    collection_name=collection_name,
                    display_name=display_name or collection_name,
                    description=description,
                    embedding_config_id=embedding_config_id,
                    embedding_model=embedding_model,
                    embedding_endpoint=embedding_endpoint,
                    dim=dim,
                    metric_type=str(metric_type).upper(),
                    hybrid_enabled=hybrid_enabled,
                    status="creating",
                    user_id=user_id,
                )
                session.add(entry)
                session.commit()
                session.refresh(entry)
                reservation = entry.to_dict()
                reservation["_newly_created"] = True
                return reservation
        except IntegrityError as exc:
            raise MilvusCollectionUnavailableError(
                f"Milvus collection '{collection_name}' changed concurrently"
            ) from exc

    def get_manual_creation_reservation(
        self,
        collection_name: str,
        *,
        user_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Return same-tenant durable manual creation state, if any."""
        with get_session() as session:
            entry = session.exec(
                select(MilvusCollectionDB).where(
                    MilvusCollectionDB.collection_name == collection_name
                )
            ).first()
            if entry is None:
                return None
            if (
                entry.status != "creating"
                or entry.user_id != user_id
                or entry.sync_task_id is not None
                or entry.deletion_owner is not None
            ):
                raise MilvusCollectionUnavailableError(
                    f"Milvus collection '{collection_name}' is unavailable"
                )
            reservation = entry.to_dict()
            reservation["_newly_created"] = False
            return reservation

    def activate_manual_collection(
        self,
        collection_name: str,
        *,
        collection_id: str,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Activate the exact manual reservation after remote creation."""
        with get_session() as session:
            entry = session.exec(
                select(MilvusCollectionDB)
                .where(MilvusCollectionDB.collection_name == collection_name)
                .with_for_update()
            ).first()
            if (
                entry is None
                or entry.collection_id != collection_id
                or entry.user_id != user_id
                or entry.status != "creating"
                or entry.sync_task_id is not None
            ):
                raise MilvusCollectionUnavailableError(
                    f"Milvus collection '{collection_name}' creation reservation "
                    "changed"
                )
            entry.status = "active"
            entry.updated_at = now_naive()
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry.to_dict()

    def cancel_manual_collection_creation(
        self,
        collection_name: str,
        *,
        collection_id: str,
        user_id: Optional[str],
    ) -> bool:
        """Cancel only the exact, still-pending manual creation reservation."""
        with get_session() as session:
            entry = session.exec(
                select(MilvusCollectionDB)
                .where(MilvusCollectionDB.collection_name == collection_name)
                .with_for_update()
            ).first()
            if (
                entry is None
                or entry.collection_id != collection_id
                or entry.user_id != user_id
                or entry.status != "creating"
                or entry.sync_task_id is not None
            ):
                raise MilvusCollectionUnavailableError(
                    f"Milvus collection '{collection_name}' creation reservation "
                    "changed"
                )
            session.delete(entry)
            session.commit()
            return True

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get collection by collection_id."""
        with get_session() as session:
            entry = session.exec(
                select(MilvusCollectionDB).where(
                    MilvusCollectionDB.collection_id == collection_id
                )
            ).first()
            return entry.to_dict() if entry else None

    def get_by_name(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection by Milvus collection name."""
        with get_session() as session:
            entry = session.exec(
                select(MilvusCollectionDB).where(
                    MilvusCollectionDB.collection_name == collection_name
                )
            ).first()
            return entry.to_dict() if entry else None

    def get_deletion_fence_owner(
        self,
        collection_name: str,
    ) -> Optional[str]:
        """Return an internal fence owner without exposing it in public DTOs."""
        with get_session() as session:
            return session.exec(
                select(MilvusCollectionDB.deletion_owner).where(
                    MilvusCollectionDB.collection_name == collection_name,
                    MilvusCollectionDB.status == "deleting",
                )
            ).first()

    @contextmanager
    def consumption_guard(
        self,
        collection_names: Iterable[str],
        *,
        user_id: Optional[str],
    ):
        """Hold row locks for every collection during a remote write window."""
        names = tuple(
            sorted(
                {
                    name.strip()
                    for name in collection_names
                    if isinstance(name, str) and name.strip()
                }
            )
        )
        with get_session() as session:
            for collection_name in names:
                lock_collection_for_consumption(
                    session,
                    collection_name,
                    user_id=user_id,
                )
            yield

    def list_collections(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        embedding_config_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List registered collections with optional filters."""
        with get_session() as session:
            query = select(MilvusCollectionDB)
            count_query = select(func.count()).select_from(MilvusCollectionDB)

            if user_id:
                query = query.where(MilvusCollectionDB.user_id == user_id)
                count_query = count_query.where(MilvusCollectionDB.user_id == user_id)
            if status:
                query = query.where(MilvusCollectionDB.status == status)
                count_query = count_query.where(MilvusCollectionDB.status == status)
            if embedding_config_id:
                query = query.where(
                    MilvusCollectionDB.embedding_config_id == embedding_config_id
                )
                count_query = count_query.where(
                    MilvusCollectionDB.embedding_config_id == embedding_config_id
                )

            total = session.exec(count_query).one()
            query = query.order_by(
                MilvusCollectionDB.created_at.desc(),
                MilvusCollectionDB.id.desc(),
            )
            query = query.offset(offset).limit(limit)

            entries = session.exec(query).all()
            return [e.to_dict() for e in entries], total

    def list_deletion_registry_snapshot(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return internal provenance needed by deletion preflight.

        This intentionally bypasses the public collection DTO, which must not
        expose sync claims or deletion-fence ownership.
        """
        with get_session() as session:
            total = session.exec(
                select(func.count()).select_from(MilvusCollectionDB)
            ).one()
            query = (
                select(MilvusCollectionDB)
                .order_by(
                    MilvusCollectionDB.created_at.desc(),
                    MilvusCollectionDB.id.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
            entries = session.exec(query).all()
            return [
                {
                    "collection_id": entry.collection_id,
                    "collection_name": entry.collection_name,
                    "user_id": entry.user_id,
                    "status": entry.status,
                    "sync_task_id": entry.sync_task_id,
                    "deletion_owner": entry.deletion_owner,
                }
                for entry in entries
            ], total

    def update_collection(self, collection_id: str, **kwargs) -> bool:
        """Update collection fields."""
        with get_session() as session:
            entry = session.exec(
                select(MilvusCollectionDB).where(
                    MilvusCollectionDB.collection_id == collection_id
                )
            ).first()
            if not entry:
                return False

            mutable_fields = {"display_name", "description"}
            for key, value in kwargs.items():
                if key in mutable_fields:
                    setattr(entry, key, value)
            entry.updated_at = now_naive()
            session.add(entry)
            session.commit()
            return True

    def acquire_deletion_fences(
        self,
        collection_names: Iterable[str],
        *,
        deletion_owner: str,
        user_id: Optional[str] = None,
        expected_collection_ids: Optional[
            Mapping[str, Optional[str]]
        ] = None,
        allow_manual_creating: bool = False,
    ) -> MilvusDeletionFenceAcquisition:
        """Fence collections before any remote drop or consumer preflight."""
        owner = _normalize_deletion_owner(deletion_owner)
        names = tuple(
            sorted(
                {
                    name.strip()
                    for name in collection_names
                    if isinstance(name, str) and name.strip()
                }
            )
        )
        expected_ids = (
            dict(expected_collection_ids)
            if expected_collection_ids is not None
            else None
        )
        if expected_ids is not None and set(expected_ids) != set(names):
            raise ValueError(
                "Expected Milvus collection identities must match fence names"
            )
        newly_fenced: List[str] = []
        created_placeholders: List[str] = []
        previous_creating: List[str] = []
        try:
            with get_session() as session:
                for collection_name in names:
                    entry = session.exec(
                        select(MilvusCollectionDB)
                        .where(
                            MilvusCollectionDB.collection_name == collection_name
                        )
                        .with_for_update()
                    ).first()
                    if expected_ids is not None:
                        expected_collection_id = expected_ids[collection_name]
                        identity_changed = (
                            entry is not None
                            if expected_collection_id is None
                            else entry is None
                            or entry.collection_id != expected_collection_id
                        )
                        if identity_changed:
                            raise MilvusCollectionDeletionOwnerConflictError(
                                f"Milvus collection '{collection_name}' changed "
                                "during deletion"
                            )
                    if entry is None:
                        entry = MilvusCollectionDB(
                            collection_name=collection_name,
                            status="deleting",
                            deletion_owner=owner,
                            user_id=user_id,
                        )
                        session.add(entry)
                        session.flush()
                        newly_fenced.append(collection_name)
                        created_placeholders.append(collection_name)
                        continue
                    if entry.user_id != user_id:
                        raise PermissionError(
                            "Milvus collection owner changed during deletion"
                        )
                    if entry.status == "deleting":
                        if entry.deletion_owner != owner:
                            raise MilvusCollectionDeletionOwnerConflictError(
                                f"Milvus collection '{collection_name}' is being "
                                "deleted by another operation"
                            )
                        continue
                    if entry.status == "creating":
                        exact_manual_abort = (
                            allow_manual_creating
                            and expected_ids is not None
                            and expected_ids.get(collection_name)
                            == entry.collection_id
                            and owner == f"manual:{entry.collection_id}"
                            and entry.sync_task_id is None
                            and entry.deletion_owner is None
                        )
                        if not exact_manual_abort:
                            raise MilvusCollectionUnavailableError(
                                f"Milvus collection '{collection_name}' is "
                                "unavailable"
                            )
                        entry.status = "deleting"
                        entry.deletion_owner = owner
                        entry.updated_at = now_naive()
                        session.add(entry)
                        newly_fenced.append(collection_name)
                        previous_creating.append(collection_name)
                        continue
                    if entry.status != "active":
                        raise MilvusCollectionUnavailableError(
                            f"Milvus collection '{collection_name}' is unavailable"
                        )
                    entry.status = "deleting"
                    entry.deletion_owner = owner
                    entry.updated_at = now_naive()
                    session.add(entry)
                    newly_fenced.append(collection_name)
                session.commit()
        except IntegrityError as exc:
            raise MilvusCollectionDeletionOwnerConflictError(
                "A Milvus collection deletion fence changed concurrently"
            ) from exc
        return MilvusDeletionFenceAcquisition(
            collection_names=names,
            newly_fenced=tuple(newly_fenced),
            created_placeholders=tuple(created_placeholders),
            previous_creating=tuple(previous_creating),
        )

    def restore_deletion_fences(
        self,
        collection_names: Iterable[str],
        *,
        deletion_owner: str,
        created_placeholders: Iterable[str] = (),
        previous_creating: Iterable[str] = (),
    ) -> bool:
        """Restore only fences owned and newly created by the calling operation."""
        owner = _normalize_deletion_owner(deletion_owner)
        names = tuple(
            sorted(
                {
                    name.strip()
                    for name in collection_names
                    if isinstance(name, str) and name.strip()
                }
            )
        )
        placeholders = {
            name.strip()
            for name in created_placeholders
            if isinstance(name, str) and name.strip()
        }
        creating_names = {
            name.strip()
            for name in previous_creating
            if isinstance(name, str) and name.strip()
        }
        with get_session() as session:
            for collection_name in names:
                entry = session.exec(
                    select(MilvusCollectionDB)
                    .where(MilvusCollectionDB.collection_name == collection_name)
                    .with_for_update()
                ).first()
                if entry is None:
                    continue
                if entry.status != "deleting" or entry.deletion_owner != owner:
                    raise MilvusCollectionDeletionOwnerConflictError(
                        f"Milvus collection '{collection_name}' deletion owner changed"
                    )
                if collection_name in placeholders:
                    session.delete(entry)
                    continue
                entry.status = (
                    "creating"
                    if collection_name in creating_names
                    else "active"
                )
                entry.deletion_owner = None
                entry.updated_at = now_naive()
                session.add(entry)
            session.commit()
        return True

    def list_deletion_fences(self, *, deletion_owner: str) -> List[str]:
        """List durable collection fences retained by one logical deletion."""
        owner = _normalize_deletion_owner(deletion_owner)
        with get_session() as session:
            names = session.exec(
                select(MilvusCollectionDB.collection_name)
                .where(
                    MilvusCollectionDB.status == "deleting",
                    MilvusCollectionDB.deletion_owner == owner,
                )
                .order_by(MilvusCollectionDB.collection_name)
            ).all()
            return list(names)

    def delete_collections(
        self,
        collection_names: Iterable[str],
        *,
        deletion_owner: str,
    ) -> bool:
        """Atomically finalize owned collection fences and orphaned links."""
        owner = _normalize_deletion_owner(deletion_owner)
        names = tuple(
            sorted(
                {
                    name.strip()
                    for name in collection_names
                    if isinstance(name, str) and name.strip()
                }
            )
        )
        if not names:
            return False
        with get_session() as session:
            entries = list(
                session.exec(
                    select(MilvusCollectionDB)
                    .where(MilvusCollectionDB.collection_name.in_(names))
                    .with_for_update()
                ).all()
            )
            entries_by_name = {
                entry.collection_name: entry for entry in entries
            }
            missing_names = [
                name for name in names if name not in entries_by_name
            ]
            if missing_names:
                raise MilvusCollectionDeletionOwnerConflictError(
                    "Milvus collection deletion fences are missing for: "
                    + ", ".join(missing_names)
                )
            for entry in entries:
                if (
                    entry.status != "deleting"
                    or entry.deletion_owner != owner
                ):
                    raise MilvusCollectionDeletionOwnerConflictError(
                        f"Milvus collection '{entry.collection_name}' "
                        "deletion owner changed"
                    )

            links = list(
                session.exec(
                    select(CollectionDatasetLinkDB).where(
                        CollectionDatasetLinkDB.collection_name.in_(names)
                    )
                ).all()
            )
            for link in links:
                session.delete(link)
            for entry in entries:
                session.delete(entry)
            session.commit()
            logger.info(
                "Deleted %d collection registry fences",
                len(entries),
            )
            return bool(entries or links)

    def delete_collection(
        self,
        collection_name: str,
        *,
        deletion_owner: str,
    ) -> bool:
        """Delete owned registry state and links, including orphaned link rows."""
        return self.delete_collections(
            [collection_name],
            deletion_owner=deletion_owner,
        )

    # ── Dataset Links ────────────────────────────────────────────

    def link_dataset(
        self,
        collection_name: str,
        dataset_id: str,
        dataset_name: Optional[str] = None,
        chunk_count: int = 0,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Link a dataset to a collection. Upsert if already linked."""
        with get_session() as session:
            datasets = lock_datasets_for_consumption(
                session,
                dataset_ids=(dataset_id,),
                require_all_dataset_ids=True,
            )
            lock_collection_for_consumption(
                session,
                collection_name,
                user_id=datasets[0].user_id,
            )
            existing = session.exec(
                select(CollectionDatasetLinkDB).where(
                    CollectionDatasetLinkDB.collection_name == collection_name,
                    CollectionDatasetLinkDB.dataset_id == dataset_id,
                )
            ).first()

            if existing:
                # Update chunk count and task_id
                if chunk_count:
                    existing.chunk_count = chunk_count
                if task_id:
                    existing.task_id = task_id
                if dataset_name:
                    existing.dataset_name = dataset_name
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing.to_dict()

            link = CollectionDatasetLinkDB(
                collection_name=collection_name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                chunk_count=chunk_count,
                task_id=task_id,
            )
            session.add(link)
            session.commit()
            session.refresh(link)
            logger.info("Linked dataset '%s' to collection '%s'", dataset_id, collection_name)
            return link.to_dict()

    def unlink_dataset(self, collection_name: str, dataset_id: str) -> bool:
        """Unlink a dataset from a collection."""
        with get_session() as session:
            link = session.exec(
                select(CollectionDatasetLinkDB).where(
                    CollectionDatasetLinkDB.collection_name == collection_name,
                    CollectionDatasetLinkDB.dataset_id == dataset_id,
                )
            ).first()
            if not link:
                return False
            session.delete(link)
            session.commit()
            return True

    def get_linked_datasets(
        self, collection_name: str
    ) -> List[Dict[str, Any]]:
        """Get all datasets linked to a collection."""
        with get_session() as session:
            links = session.exec(
                select(CollectionDatasetLinkDB)
                .where(CollectionDatasetLinkDB.collection_name == collection_name)
                .order_by(CollectionDatasetLinkDB.linked_at.desc())
            ).all()
            return [link.to_dict() for link in links]

    def list_dataset_links(
        self,
        dataset_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        """List link rows directly, including links to unregistered collections."""
        normalized_ids = {
            dataset_id.strip()
            for dataset_id in dataset_ids
            if isinstance(dataset_id, str) and dataset_id.strip()
        }
        if not normalized_ids:
            return []
        with get_session() as session:
            links = session.exec(
                select(CollectionDatasetLinkDB)
                .where(CollectionDatasetLinkDB.dataset_id.in_(normalized_ids))
                .order_by(CollectionDatasetLinkDB.linked_at.desc())
            ).all()
            return [link.to_dict() for link in links]

    def get_collections_for_dataset(
        self, dataset_id: str
    ) -> List[Dict[str, Any]]:
        """Get all collections that contain vectors from a dataset."""
        with get_session() as session:
            links = session.exec(
                select(CollectionDatasetLinkDB).where(
                    CollectionDatasetLinkDB.dataset_id == dataset_id
                )
            ).all()
            if not links:
                return []

            collection_names = [link.collection_name for link in links]
            collections = session.exec(
                select(MilvusCollectionDB).where(
                    MilvusCollectionDB.collection_name.in_(collection_names)
                )
            ).all()
            return [c.to_dict() for c in collections]


# Singleton instance
milvus_collection_service = MilvusCollectionService()
