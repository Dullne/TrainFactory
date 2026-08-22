"""Regression tests for durable Milvus collection deletion fences."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.external_sync_entity import ExternalSyncTaskDB
from train_factory.storage.entities.milvus_collection_entity import (
    CollectionDatasetLinkDB,
    MilvusCollectionDB,
)


milvus_module = importlib.import_module(
    "train_factory.storage.services.milvus_collection_service"
)
generation_module = importlib.import_module(
    "train_factory.storage.services.generation_task_service"
)


@pytest.fixture
def fenced_services(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[
            DatasetDB.__table__,
            MilvusCollectionDB.__table__,
            CollectionDatasetLinkDB.__table__,
            GenerationTaskDB.__table__,
            ExternalSyncTaskDB.__table__,
        ],
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(milvus_module, "get_session", test_session)
    monkeypatch.setattr(generation_module, "get_session", test_session)
    return (
        milvus_module.MilvusCollectionService(),
        generation_module.GenerationTaskService(),
        engine,
    )


def _generation_task_kwargs(**overrides):
    values = {
        "task_name": "generation",
        "input_path": "/tmp/input.jsonl",
        "llm_config": {},
        "steps_config": {},
    }
    values.update(overrides)
    return values


def test_internal_deletion_snapshot_exposes_provenance_without_public_leak(
    fenced_services,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add(
            MilvusCollectionDB(
                collection_name="sync-internal-provenance",
                user_id="user-1",
                sync_task_id="sync-task-1",
                status="deleting",
                deletion_owner="sync:sync-task-1",
            )
        )
        session.commit()

    public, public_total = collections.list_collections()
    assert public_total == 1
    assert "sync_task_id" not in public[0]
    assert "deletion_owner" not in public[0]

    internal, internal_total = collections.list_deletion_registry_snapshot()
    assert internal_total == 1
    assert internal == [
        {
            "collection_id": public[0]["collection_id"],
            "collection_name": "sync-internal-provenance",
            "user_id": "user-1",
            "status": "deleting",
            "sync_task_id": "sync-task-1",
            "deletion_owner": "sync:sync-task-1",
        }
    ]


def test_generic_collection_update_cannot_rewrite_internal_provenance(
    fenced_services,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        row = MilvusCollectionDB(
            collection_name="sync-immutable-provenance",
            user_id="user-1",
            sync_task_id="sync-task-1",
            status="deleting",
            deletion_owner="sync:sync-task-1",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        collection_id = row.collection_id

    assert collections.update_collection(
        collection_id,
        id=999,
        collection_name="replacement-name",
        user_id="user-foreign",
        sync_task_id="sync-task-foreign",
        deletion_owner="sync:sync-task-foreign",
        status="active",
        embedding_config_id="config-foreign",
        embedding_model="model-foreign",
        embedding_endpoint="https://foreign.invalid",
        dim=1536,
        metric_type="L2",
        hybrid_enabled=True,
        display_name="Updated display name",
        description="Updated description",
    )

    with Session(engine) as session:
        stored = session.exec(
            select(MilvusCollectionDB).where(
                MilvusCollectionDB.collection_id == collection_id
            )
        ).one()
        assert stored.collection_id == collection_id
        assert stored.collection_name == "sync-immutable-provenance"
        assert stored.user_id == "user-1"
        assert stored.sync_task_id == "sync-task-1"
        assert stored.deletion_owner == "sync:sync-task-1"
        assert stored.status == "deleting"
        assert stored.embedding_config_id is None
        assert stored.embedding_model is None
        assert stored.embedding_endpoint is None
        assert stored.dim == 1024
        assert stored.metric_type == "COSINE"
        assert stored.hybrid_enabled is None
        assert stored.display_name == "Updated display name"
        assert stored.description == "Updated description"


def test_collection_fence_blocks_registration_and_dataset_link(fenced_services):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add(DatasetDB(dataset_id="dataset-1", dataset_name="dataset"))
        session.add(
            MilvusCollectionDB(
                collection_name="collection-1",
                display_name="collection-1",
                status="deleting",
                deletion_owner="sync:task-1",
            )
        )
        session.commit()

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.register_collection("collection-1")
    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.link_dataset("collection-1", "dataset-1")


def test_collection_consumption_lock_revalidates_tenant(fenced_services):
    collections, generations, engine = fenced_services
    collections.register_collection("collection-1", user_id="user-2")
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id="dataset-1",
                dataset_name="dataset",
                user_id="user-1",
            )
        )
        session.commit()

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.link_dataset("collection-1", "dataset-1")
    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        generations.create_task(
            **_generation_task_kwargs(
                user_id="user-1",
                milvus_collection="collection-1",
            )
        )


def test_collection_fence_owner_controls_restore_and_final_delete(fenced_services):
    collections, _generations, engine = fenced_services
    collections.register_collection("collection-1", user_id="user-1")

    acquisition = collections.acquire_deletion_fences(
        ["collection-1"],
        deletion_owner="sync:task-1",
        user_id="user-1",
    )
    assert acquisition.newly_fenced == ("collection-1",)
    assert acquisition.created_placeholders == ()

    with pytest.raises(milvus_module.MilvusCollectionDeletionOwnerConflictError):
        collections.restore_deletion_fences(
            ["collection-1"],
            deletion_owner="sync:task-2",
        )
    with pytest.raises(milvus_module.MilvusCollectionDeletionOwnerConflictError):
        collections.delete_collection(
            "collection-1",
            deletion_owner="sync:task-2",
        )

    assert collections.restore_deletion_fences(
        acquisition.newly_fenced,
        deletion_owner="sync:task-1",
        created_placeholders=acquisition.created_placeholders,
    )
    restored = collections.get_by_name("collection-1")
    assert restored is not None
    assert restored["status"] == "active"

    collections.acquire_deletion_fences(
        ["collection-1"],
        deletion_owner="sync:task-1",
        user_id="user-1",
    )
    assert collections.delete_collection(
        "collection-1",
        deletion_owner="sync:task-1",
    )
    with Session(engine) as session:
        assert session.exec(select(MilvusCollectionDB)).all() == []


def test_missing_collection_uses_removable_deletion_placeholder(fenced_services):
    collections, _generations, engine = fenced_services

    acquisition = collections.acquire_deletion_fences(
        ["legacy-physical-collection"],
        deletion_owner="sync:task-1",
        user_id="user-1",
    )

    assert acquisition.created_placeholders == ("legacy-physical-collection",)
    assert collections.get_by_name("legacy-physical-collection")["status"] == "deleting"
    assert collections.restore_deletion_fences(
        acquisition.newly_fenced,
        deletion_owner="sync:task-1",
        created_placeholders=acquisition.created_placeholders,
    )
    with Session(engine) as session:
        assert session.exec(select(MilvusCollectionDB)).all() == []


def test_final_delete_rejects_missing_requested_fence_and_preserves_orphan_links(
    fenced_services,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add(
            CollectionDatasetLinkDB(
                collection_name="orphan-collection",
                dataset_id="dataset-1",
            )
        )
        session.commit()

    with pytest.raises(
        milvus_module.MilvusCollectionDeletionOwnerConflictError
    ):
        collections.delete_collection(
            "orphan-collection",
            deletion_owner="sync:task-1",
        )
    with Session(engine) as session:
        assert len(session.exec(select(CollectionDatasetLinkDB)).all()) == 1


def test_final_delete_removes_orphan_links_after_placeholder_is_fenced(
    fenced_services,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add(
            CollectionDatasetLinkDB(
                collection_name="orphan-collection",
                dataset_id="dataset-1",
            )
        )
        session.commit()

    acquisition = collections.acquire_deletion_fences(
        ["orphan-collection"],
        deletion_owner="sync:task-1",
        user_id="user-1",
    )

    assert acquisition.created_placeholders == ("orphan-collection",)
    assert collections.delete_collection(
        "orphan-collection",
        deletion_owner="sync:task-1",
    )
    with Session(engine) as session:
        assert session.exec(select(CollectionDatasetLinkDB)).all() == []


def test_manual_creation_reservation_blocks_ordinary_registration_until_activation(
    fenced_services,
):
    collections, _generations, _engine = fenced_services

    reservation = collections.reserve_manual_collection(
        collection_name="manual-collection",
        dim=768,
        user_id="user-1",
    )

    assert reservation["status"] == "creating"
    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.register_collection(
            "manual-collection",
            user_id="user-1",
        )

    activated = collections.activate_manual_collection(
        "manual-collection",
        collection_id=reservation["collection_id"],
        user_id="user-1",
    )
    assert activated["status"] == "active"


def test_manual_creation_retry_reuses_exact_durable_reservation(
    fenced_services,
):
    collections, _generations, _engine = fenced_services
    create_kwargs = {
        "collection_name": "manual-collection",
        "embedding_config_id": "config-1",
        "embedding_model": "embedding-1",
        "embedding_endpoint": "https://embedding.invalid/v1",
        "dim": 768,
        "metric_type": "COSINE",
        "hybrid_enabled": False,
        "display_name": "Original display name",
        "description": "Original description",
        "user_id": "user-1",
    }

    created = collections.reserve_manual_collection(**create_kwargs)
    retried = collections.reserve_manual_collection(
        **{
            **create_kwargs,
            # Mutable presentation metadata is not part of remote identity.
            "display_name": "Retried display name",
            "description": "Retried description",
        }
    )

    assert created["_newly_created"] is True
    assert retried["_newly_created"] is False
    assert retried["collection_id"] == created["collection_id"]
    assert retried["display_name"] == "Original display name"
    assert retried["description"] == "Original description"


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("user_id", "user-2"),
        ("embedding_config_id", "config-2"),
        ("embedding_model", "embedding-2"),
        ("embedding_endpoint", "https://other.invalid/v1"),
        ("dim", 1536),
        ("metric_type", "L2"),
        ("hybrid_enabled", True),
    ],
)
def test_manual_creation_retry_rejects_owner_or_immutable_config_change(
    fenced_services,
    changed_field,
    changed_value,
):
    collections, _generations, _engine = fenced_services
    create_kwargs = {
        "collection_name": "manual-collection",
        "embedding_config_id": "config-1",
        "embedding_model": "embedding-1",
        "embedding_endpoint": "https://embedding.invalid/v1",
        "dim": 768,
        "metric_type": "COSINE",
        "user_id": "user-1",
    }
    collections.reserve_manual_collection(**create_kwargs)

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.reserve_manual_collection(
            **{**create_kwargs, changed_field: changed_value}
        )


def test_manual_creation_retry_rejects_internal_sync_claim(
    fenced_services,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add(
            MilvusCollectionDB(
                collection_name="claimed-collection",
                dim=768,
                metric_type="COSINE",
                status="creating",
                user_id="user-1",
                sync_task_id="sync-task-1",
            )
        )
        session.commit()

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.reserve_manual_collection(
            collection_name="claimed-collection",
            dim=768,
            metric_type="COSINE",
            user_id="user-1",
        )


def test_manual_creating_reservation_can_be_fenced_for_abort_and_restored(
    fenced_services,
):
    collections, _generations, _engine = fenced_services
    reservation = collections.reserve_manual_collection(
        collection_name="partial-collection",
        dim=768,
        metric_type="COSINE",
        hybrid_enabled=True,
        user_id="user-1",
    )

    acquisition = collections.acquire_deletion_fences(
        ["partial-collection"],
        deletion_owner=f"manual:{reservation['collection_id']}",
        user_id="user-1",
        expected_collection_ids={
            "partial-collection": reservation["collection_id"]
        },
        allow_manual_creating=True,
    )

    assert acquisition.newly_fenced == ("partial-collection",)
    assert acquisition.previous_creating == ("partial-collection",)
    assert collections.get_by_name("partial-collection")["status"] == "deleting"

    collections.restore_deletion_fences(
        acquisition.newly_fenced,
        deletion_owner=f"manual:{reservation['collection_id']}",
        previous_creating=acquisition.previous_creating,
    )
    assert collections.get_by_name("partial-collection")["status"] == "creating"


def test_sync_claimed_creating_registry_cannot_use_manual_abort(
    fenced_services,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        row = MilvusCollectionDB(
            collection_name="sync-creating",
            status="creating",
            user_id="user-1",
            sync_task_id="sync-task-1",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        collection_id = row.collection_id

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.acquire_deletion_fences(
            ["sync-creating"],
            deletion_owner=f"manual:{collection_id}",
            user_id="user-1",
            expected_collection_ids={"sync-creating": collection_id},
            allow_manual_creating=True,
        )


def test_manual_creation_cancellation_is_identity_and_tenant_checked(
    fenced_services,
):
    collections, _generations, _engine = fenced_services
    reservation = collections.reserve_manual_collection(
        collection_name="manual-collection",
        user_id="user-1",
    )

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.cancel_manual_collection_creation(
            "manual-collection",
            collection_id="different-reservation",
            user_id="user-1",
        )
    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.cancel_manual_collection_creation(
            "manual-collection",
            collection_id=reservation["collection_id"],
            user_id="user-2",
        )

    assert collections.cancel_manual_collection_creation(
        "manual-collection",
        collection_id=reservation["collection_id"],
        user_id="user-1",
    )
    assert collections.get_by_name("manual-collection") is None


def test_sync_claim_cannot_reuse_manual_or_another_tasks_registry(
    fenced_services,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-task-1",
                    task_name="sync-1",
                    user_id="user-1",
                    milvus_collection_name="sync-collection",
                ),
                ExternalSyncTaskDB(
                    task_id="sync-task-2",
                    task_name="sync-2",
                    user_id="user-1",
                ),
            ]
        )
        session.commit()

    collections.register_collection("manual-collection", user_id="user-1")
    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.register_collection(
            "manual-collection",
            user_id="user-1",
            sync_task_id="sync-task-1",
        )

    first = collections.register_collection(
        "sync-collection",
        user_id="user-1",
        sync_task_id="sync-task-1",
    )
    retried = collections.register_collection(
        "sync-collection",
        user_id="user-1",
        sync_task_id="sync-task-1",
    )
    assert retried["collection_id"] == first["collection_id"]

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.register_collection(
            "sync-collection",
            user_id="user-1",
        )

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.register_collection(
            "sync-collection",
            user_id="user-1",
            sync_task_id="sync-task-2",
        )


def test_sync_claim_revalidates_durable_task_tenant(fenced_services):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="foreign-sync-task",
                task_name="foreign",
                user_id="user-2",
            )
        )
        session.commit()

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.register_collection(
            "sync-collection",
            user_id="user-1",
            sync_task_id="foreign-sync-task",
        )
    assert collections.get_by_name("sync-collection") is None


@pytest.mark.parametrize("preexisting_registry", [False, True])
@pytest.mark.parametrize(
    "name_kind",
    ["current", "v1", "v2", "v3"],
)
def test_non_sync_registration_rejects_every_durable_sync_namespace(
    fenced_services,
    preexisting_registry,
    name_kind,
):
    collections, _generations, engine = fenced_services
    task_id = "reserved-sync-task"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    names = {
        "current": "persisted-current",
        "v1": f"tf_sync_{task_id[:8]}_legacy",
        "v2": f"tf_sync_v2_{task_id[:8]}_legacy",
        "v3": f"tf_sync_v3_{task_hash}_base",
    }
    collection_name = names[name_kind]
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="reserved",
                user_id="user-1",
                milvus_collection_name="persisted-current",
            )
        )
        if preexisting_registry:
            session.add(
                MilvusCollectionDB(
                    collection_name=collection_name,
                    user_id="user-1",
                )
            )
        session.commit()

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.register_collection(
            collection_name,
            user_id="user-1",
        )

    stored = collections.get_by_name(collection_name)
    assert (stored is not None) is preexisting_registry


@pytest.mark.parametrize("name_kind", ["current", "v1", "v2", "v3"])
def test_manual_reservation_rejects_every_durable_sync_namespace(
    fenced_services,
    name_kind,
):
    collections, _generations, engine = fenced_services
    task_id = "reserved-sync-task"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    names = {
        "current": "persisted-current",
        "v1": f"tf_sync_{task_id[:8]}_legacy",
        "v2": f"tf_sync_v2_{task_id[:8]}_legacy",
        "v3": f"tf_sync_v3_{task_hash}_base",
    }
    collection_name = names[name_kind]
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="reserved",
                user_id="user-1",
                milvus_collection_name="persisted-current",
            )
        )
        session.commit()

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        collections.reserve_manual_collection(
            collection_name=collection_name,
            user_id="user-1",
        )
    assert collections.get_by_name(collection_name) is None


def test_collection_listing_uses_stable_unique_tiebreaker(fenced_services):
    collections, _generations, engine = fenced_services
    tied_time = datetime(2026, 1, 1, 0, 0, 0)
    with Session(engine) as session:
        session.add_all(
            [
                MilvusCollectionDB(
                    collection_name=f"collection-{index}",
                    created_at=tied_time,
                )
                for index in range(1, 4)
            ]
        )
        session.commit()

    first_page, total = collections.list_collections(limit=2, offset=0)
    second_page, _ = collections.list_collections(limit=2, offset=2)

    assert total == 3
    assert [row["collection_name"] for row in first_page] == [
        "collection-3",
        "collection-2",
    ]
    assert [row["collection_name"] for row in second_page] == [
        "collection-1"
    ]


def test_generation_creation_persists_and_locks_existing_collection(
    fenced_services,
):
    collections, generations, engine = fenced_services
    collections.register_collection("collection-1", user_id="user-1")

    task = generations.create_task(
        **_generation_task_kwargs(
            user_id="user-1",
            milvus_collection="collection-1",
        )
    )
    assert task["milvus_collection"] == "collection-1"

    collections.acquire_deletion_fences(
        ["collection-1"],
        deletion_owner="sync:task-1",
        user_id="user-1",
    )
    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        generations.create_task(
            **_generation_task_kwargs(
                user_id="user-1",
                milvus_collection="collection-1",
            )
        )

    with Session(engine) as session:
        stored = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task["task_id"]
            )
        ).one()
        stored.status = GenerationStatus.FAILED
        session.add(stored)
        session.commit()

    assert not generations.update_status(
        task["task_id"],
        GenerationStatus.PENDING,
    )


def test_active_collection_consumers_are_reported(fenced_services):
    collections, generations, _engine = fenced_services
    collections.register_collection("collection-1")
    active = generations.create_task(
        **_generation_task_kwargs(milvus_collection="collection-1")
    )
    completed = generations.create_task(
        **_generation_task_kwargs(milvus_collection="collection-1")
    )
    assert generations.update_status(
        completed["task_id"], GenerationStatus.RUNNING
    )
    assert generations.update_status(
        completed["task_id"], GenerationStatus.PUBLISHING
    )
    assert generations.update_status(
        completed["task_id"], GenerationStatus.COMPLETED
    )

    assert generations.list_active_collection_consumers(
        ["collection-1"]
    ) == [active["task_id"]]


@pytest.mark.parametrize(
    "expected_collection_id",
    ["old-collection-id", None],
    ids=["registered-snapshot", "unregistered-snapshot"],
)
def test_deletion_fence_rejects_same_name_registry_aba(
    fenced_services,
    expected_collection_id,
):
    collections, _generations, engine = fenced_services
    with Session(engine) as session:
        session.add(
            MilvusCollectionDB(
                collection_id="new-collection-id",
                collection_name="collection-1",
                status="active",
                user_id="user-1",
            )
        )
        session.commit()

    try:
        with pytest.raises(
            milvus_module.MilvusCollectionDeletionOwnerConflictError
        ):
            collections.acquire_deletion_fences(
                ["collection-1"],
                deletion_owner="manual:old-collection-id",
                user_id="user-1",
                expected_collection_ids={
                    "collection-1": expected_collection_id
                },
            )
    except TypeError as exc:
        pytest.fail(f"deletion fencing must accept registry identity snapshots: {exc}")

    with Session(engine) as session:
        current = session.exec(select(MilvusCollectionDB)).one()
        assert current.collection_id == "new-collection-id"
        assert current.status == "active"
        assert current.deletion_owner is None
