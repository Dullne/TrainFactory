"""External sync tasks are durable Milvus collection consumers."""

import hashlib

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from train_factory.storage.entities.external_sync_entity import ExternalSyncTaskDB
from train_factory.storage.services.external_sync_service import ExternalSyncService


@pytest.mark.parametrize(
    "collection_name",
    [
        "persisted-current",
        "tf_sync_sync-tas_legacy",
        "tf_sync_v2_sync-tas_legacy",
        "tf_sync_v3_{task_hash}_base",
    ],
    ids=["current", "v1", "v2", "v3"],
)
def test_each_durable_sync_namespace_is_reserved(collection_name):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[ExternalSyncTaskDB.__table__],
    )
    task_id = "sync-task-namespace"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="sync",
                user_id="user-1",
                milvus_collection_name="persisted-current",
                is_active=False,
            )
        )
        session.commit()

    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    service = ExternalSyncService()
    service.engine = engine

    assert service.list_active_collection_consumers(
        [collection_name.format(task_hash=task_hash)]
    ) == [task_id]


def test_persisted_sync_tasks_reserve_current_and_namespaced_collections():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[ExternalSyncTaskDB.__table__],
    )
    active_task_id = "active-sync-task"
    inactive_task_id = "inactive-sync-task"
    manual_task_id = "manual-sync-task"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=active_task_id,
                task_name="active",
                user_id="user-1",
                milvus_collection_name="persisted-current",
                is_active=True,
            )
        )
        session.add(
            ExternalSyncTaskDB(
                task_id=inactive_task_id,
                task_name="inactive",
                user_id="user-1",
                milvus_collection_name="inactive-current",
                is_active=False,
            )
        )
        session.add(
            ExternalSyncTaskDB(
                task_id=manual_task_id,
                task_name="manual",
                user_id="user-1",
                is_active=False,
                status="syncing",
            )
        )
        session.commit()

    task_hash = hashlib.sha256(active_task_id.encode("utf-8")).hexdigest()[:12]
    service = ExternalSyncService()
    service.engine = engine
    manual_hash = hashlib.sha256(manual_task_id.encode("utf-8")).hexdigest()[:12]

    consumers = service.list_active_collection_consumers(
        [
            "persisted-current",
            "inactive-current",
            f"tf_sync_{active_task_id[:8]}_legacy",
            f"tf_sync_v3_{task_hash}_base",
            f"tf_sync_v2_{active_task_id[:8]}_legacy",
            f"tf_sync_v3_{hashlib.sha256(inactive_task_id.encode()).hexdigest()[:12]}_base",
            f"tf_sync_v3_{manual_hash}_base",
            "unrelated",
        ]
    )

    assert consumers == sorted([active_task_id, inactive_task_id, manual_task_id])
