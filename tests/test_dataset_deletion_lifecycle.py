import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import dataset_routes
from train_factory.storage.entities.dataset_asset_entity import DatasetAssetDB
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.services.dataset_service import (
    DatasetService,
    DatasetUpdateOutcome,
)


USER = {"user_id": "user-1", "username": "alice"}
DATASET_ID = "dataset-1"


@pytest.mark.parametrize("foreign_reference_kind", ["dataset", "asset"])
def test_dataset_service_finds_cross_tenant_storage_reference_consumers(
    monkeypatch,
    foreign_reference_kind,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[DatasetDB.__table__, DatasetAssetDB.__table__],
    )
    shared_reference = (
        "/managed/shared/data.jsonl"
        if foreign_reference_kind == "dataset"
        else "s3://bucket/shared/data.jsonl"
    )
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id="owned-dataset",
                dataset_name="owned",
                user_id="user-owner",
                storage_path=(
                    shared_reference
                    if foreign_reference_kind == "dataset"
                    else None
                ),
            )
        )
        session.add(
            DatasetDB(
                dataset_id="foreign-dataset",
                dataset_name="foreign",
                user_id="user-foreign",
                storage_path=(
                    shared_reference
                    if foreign_reference_kind == "dataset"
                    else None
                ),
            )
        )
        if foreign_reference_kind == "asset":
            session.add(
                DatasetAssetDB(
                    dataset_id="owned-dataset",
                    storage_uri=shared_reference,
                )
            )
            session.add(
                DatasetAssetDB(
                    dataset_id="foreign-dataset",
                    storage_uri=shared_reference,
                )
            )
        session.commit()

    service = DatasetService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    consumers = service.list_external_storage_reference_consumers(
        [shared_reference],
        exclude_dataset_ids=["owned-dataset"],
    )

    assert {consumer["dataset_id"] for consumer in consumers} == {
        "foreign-dataset"
    }


@pytest.mark.parametrize(
    ("delete_target", "foreign_reference"),
    [
        (
            "/managed/datasets/owned",
            "/managed/datasets/owned/splits/train.jsonl",
        ),
        (
            "/managed/datasets/owned/splits/train.jsonl",
            "/managed/datasets/owned",
        ),
    ],
)
def test_dataset_service_finds_nested_local_storage_consumers(
    monkeypatch,
    delete_target,
    foreign_reference,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[DatasetDB.__table__, DatasetAssetDB.__table__],
    )
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id="owned-dataset",
                dataset_name="owned",
                user_id="user-owner",
                storage_path=delete_target,
            )
        )
        session.add(
            DatasetAssetDB(
                dataset_id="foreign-dataset",
                storage_uri=foreign_reference,
            )
        )
        session.commit()

    service = DatasetService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    consumers = service.list_external_storage_reference_consumers(
        [delete_target],
        exclude_dataset_ids=["owned-dataset"],
    )

    assert [consumer["dataset_id"] for consumer in consumers] == [
        "foreign-dataset"
    ]


def test_dataset_service_keeps_s3_object_matching_exact(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[DatasetDB.__table__, DatasetAssetDB.__table__],
    )
    with Session(engine) as session:
        session.add(
            DatasetAssetDB(
                dataset_id="foreign-dataset",
                storage_uri="s3://bucket/dataset/child.jsonl",
            )
        )
        session.commit()

    service = DatasetService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    assert service.list_external_storage_reference_consumers(
        ["s3://bucket/dataset"],
    ) == []


def _configure_delete(
    monkeypatch,
    *,
    status="ready",
    generation_consumers=(),
    training_consumers=(),
    evaluation_consumers=(),
    milvus_links=(),
    storage_delete_result=True,
    asset_cleanup=None,
    lineage_cleanup=None,
    database_delete_result=True,
    deletion_owner=None,
):
    events = []
    state = {
        "dataset_id": DATASET_ID,
        "dataset_name": "dataset",
        "user_id": USER["user_id"],
        "status": status,
        "storage_backend": "local",
        "storage_path": f"/managed/datasets/{DATASET_ID}",
        "storage_uri": None,
        "deletion_owner": (
            deletion_owner
            if deletion_owner is not None
            else (f"dataset:{DATASET_ID}" if status == "deleting" else None)
        ),
        "seen_deletion_owners": [],
    }

    def get_dataset(dataset_id):
        assert dataset_id == DATASET_ID
        return dict(state)

    def mark_deleting(dataset_id, *, deletion_owner, user_id):
        assert dataset_id == DATASET_ID
        assert user_id == USER["user_id"]
        state["seen_deletion_owners"].append(deletion_owner)
        if state["deletion_owner"] not in {None, deletion_owner}:
            raise ValueError("Dataset is owned by another deletion operation")
        events.append("mark_deleting")
        state["status"] = "deleting"
        state["deletion_owner"] = deletion_owner
        return True

    def restore_from_deleting(dataset_id, *, deletion_owner, status, user_id):
        assert dataset_id == DATASET_ID
        assert user_id == USER["user_id"]
        assert deletion_owner == state["deletion_owner"]
        state["seen_deletion_owners"].append(deletion_owner)
        events.append(("restore", status))
        state["status"] = status
        state["deletion_owner"] = None
        return True

    def delete_record(dataset_id, *, deletion_owner, user_id):
        assert dataset_id == DATASET_ID
        assert user_id == USER["user_id"]
        assert deletion_owner == state["deletion_owner"]
        state["seen_deletion_owners"].append(deletion_owner)
        events.append("delete_record")
        return database_delete_result

    monkeypatch.setattr(dataset_routes.dataset_service, "get_dataset", get_dataset)
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "mark_deleting",
        mark_deleting,
        raising=False,
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "restore_from_deleting",
        restore_from_deleting,
        raising=False,
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "delete_dataset",
        delete_record,
    )
    monkeypatch.setattr(
        dataset_routes,
        "verify_resource_ownership",
        lambda value, *_args: value,
    )
    monkeypatch.setattr(
        dataset_routes.dataset_asset_service,
        "list_assets",
        lambda **_kwargs: events.append("list_assets") or [],
    )
    monkeypatch.setattr(
        dataset_routes.dataset_asset_service,
        "delete_assets_for_dataset",
        asset_cleanup or (lambda _dataset_id: events.append("delete_assets") or 0),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_lineage_service,
        "delete_edges_for_dataset",
        lineage_cleanup or (lambda _dataset_id: events.append("delete_lineage") or 0),
    )
    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: SimpleNamespace(
            delete=lambda *_args, **_kwargs: (
                events.append("delete_storage") or storage_delete_result
            )
        ),
    )
    monkeypatch.setattr(
        dataset_routes.generation_task_service,
        "list_active_dataset_consumers",
        lambda dataset_ids: (
            events.append(("generation_consumers", tuple(dataset_ids)))
            or list(generation_consumers)
        ),
    )
    monkeypatch.setattr(
        dataset_routes.training_task_service,
        "list_active_dataset_consumers",
        lambda dataset_paths: (
            events.append(("training_consumers", tuple(dataset_paths)))
            or list(training_consumers)
        ),
    )
    monkeypatch.setattr(
        dataset_routes.evaluation_task_service,
        "list_active_dataset_consumers",
        lambda dataset_ids, dataset_paths: (
            events.append(
                (
                    "evaluation_consumers",
                    tuple(dataset_ids),
                    tuple(dataset_paths),
                )
            )
            or list(evaluation_consumers)
        ),
    )
    monkeypatch.setattr(
        dataset_routes,
        "milvus_collection_service",
        SimpleNamespace(
            list_dataset_links=lambda dataset_ids: (
                events.append(("milvus_links", tuple(dataset_ids)))
                or list(milvus_links)
            )
        ),
        raising=False,
    )
    return events, state


def test_delete_marks_before_consumer_scan_and_restores_first_conflict(monkeypatch):
    events, state = _configure_delete(
        monkeypatch,
        training_consumers=("training-1",),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert exc_info.value.status_code == 409
    assert events[:5] == [
        "mark_deleting",
        "list_assets",
        ("generation_consumers", (DATASET_ID,)),
        ("training_consumers", (f"/managed/datasets/{DATASET_ID}",)),
        (
            "evaluation_consumers",
            (DATASET_ID,),
            (f"/managed/datasets/{DATASET_ID}",),
        ),
    ]
    assert events[-1] == ("restore", "ready")
    assert state["status"] == "ready"
    assert "delete_storage" not in events
    assert "delete_record" not in events


@pytest.mark.parametrize("status", ("uploading", "downloading", "processing"))
def test_delete_rejects_dataset_while_background_write_may_be_active(
    monkeypatch,
    status,
):
    events, state = _configure_delete(monkeypatch, status=status)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert exc_info.value.status_code == 409
    assert state["status"] == status
    assert events == []


def test_delete_retry_keeps_existing_intent_on_consumer_conflict(monkeypatch):
    events, state = _configure_delete(
        monkeypatch,
        status="deleting",
        evaluation_consumers=("deep-evaluation-1",),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert exc_info.value.status_code == 409
    assert state["status"] == "deleting"
    assert not any(
        isinstance(event, tuple) and event[0] == "restore" for event in events
    )
    assert "delete_storage" not in events
    assert "delete_record" not in events


def test_delete_rejects_milvus_link_and_restores_first_attempt(monkeypatch):
    events, state = _configure_delete(
        monkeypatch,
        milvus_links=(
            {
                "collection_name": "orphaned-collection",
                "dataset_id": DATASET_ID,
            },
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert exc_info.value.status_code == 409
    assert state["status"] == "ready"
    assert ("milvus_links", (DATASET_ID,)) in events
    assert events[-1] == ("restore", "ready")
    assert "delete_storage" not in events
    assert "delete_record" not in events


def test_delete_retry_keeps_intent_when_milvus_link_still_exists(monkeypatch):
    events, state = _configure_delete(
        monkeypatch,
        status="deleting",
        milvus_links=(
            {
                "collection_name": "collection-1",
                "dataset_id": DATASET_ID,
            },
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert exc_info.value.status_code == 409
    assert state["status"] == "deleting"
    assert not any(
        isinstance(event, tuple) and event[0] == "restore" for event in events
    )
    assert "delete_storage" not in events
    assert "delete_record" not in events


@pytest.mark.parametrize(
    ("failure_kind", "kwargs"),
    (
        ("storage", {"storage_delete_result": False}),
        (
            "assets",
            {
                "asset_cleanup": lambda _dataset_id: (_ for _ in ()).throw(
                    RuntimeError("asset cleanup failed")
                )
            },
        ),
        (
            "lineage",
            {
                "lineage_cleanup": lambda _dataset_id: (_ for _ in ()).throw(
                    RuntimeError("lineage cleanup failed")
                )
            },
        ),
        ("database", {"database_delete_result": False}),
    ),
)
def test_delete_failure_keeps_durable_intent_and_database_record(
    monkeypatch,
    failure_kind,
    kwargs,
):
    events, state = _configure_delete(monkeypatch, **kwargs)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert exc_info.value.status_code == 500
    assert state["status"] == "deleting"
    if failure_kind != "database":
        assert "delete_record" not in events


def test_delete_succeeds_only_after_storage_and_metadata_cleanup(monkeypatch):
    events, state = _configure_delete(monkeypatch)

    result = asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert result == {"message": f"Dataset {DATASET_ID} deleted"}
    assert events.index("delete_record") > events.index("delete_storage")
    assert events.index("delete_record") > events.index("delete_assets")
    assert events.index("delete_record") > events.index("delete_lineage")
    assert state["seen_deletion_owners"] == [
        f"dataset:{DATASET_ID}",
        f"dataset:{DATASET_ID}",
    ]


def test_delete_retries_existing_intent_after_cleanup_recovers(monkeypatch):
    events, state = _configure_delete(monkeypatch, status="deleting")

    result = asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert result == {"message": f"Dataset {DATASET_ID} deleted"}
    assert state["status"] == "deleting"
    assert "delete_record" in events
    assert not any(
        isinstance(event, tuple) and event[0] == "restore" for event in events
    )


def test_dataset_service_marks_and_restores_deletion_state(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[DatasetDB.__table__])
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id=DATASET_ID,
                dataset_name="dataset",
                user_id=USER["user_id"],
                status="ready",
            )
        )
        session.commit()

    service = DatasetService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    deletion_owner = f"dataset:{DATASET_ID}"
    assert (
        service.mark_deleting(
            DATASET_ID,
            deletion_owner=deletion_owner,
            user_id=USER["user_id"],
        )
        is True
    )
    assert (
        service.mark_deleting(
            DATASET_ID,
            deletion_owner=deletion_owner,
            user_id=USER["user_id"],
        )
        is True
    )
    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        assert dataset.status == "deleting"
        assert dataset.deletion_owner == deletion_owner

    assert (
        service.restore_from_deleting(
            DATASET_ID,
            deletion_owner=deletion_owner,
            status="ready",
            user_id=USER["user_id"],
        )
        is True
    )
    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        assert dataset.status == "ready"
        assert dataset.deletion_owner is None


def test_dataset_service_foreign_owner_cannot_mark_restore_or_delete(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[DatasetDB.__table__])
    deletion_owner = f"dataset:{DATASET_ID}"
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id=DATASET_ID,
                dataset_name="dataset",
                user_id=USER["user_id"],
                status="deleting",
                deletion_owner=deletion_owner,
            )
        )
        session.commit()

    service = DatasetService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    operations = (
        lambda: service.mark_deleting(
            DATASET_ID,
            deletion_owner="generation:other-task",
            user_id=USER["user_id"],
        ),
        lambda: service.restore_from_deleting(
            DATASET_ID,
            deletion_owner="generation:other-task",
            status="ready",
            user_id=USER["user_id"],
        ),
        lambda: service.delete_dataset(
            DATASET_ID,
            deletion_owner="generation:other-task",
            user_id=USER["user_id"],
        ),
    )
    for operation in operations:
        with pytest.raises(
            ValueError,
            match="another deletion operation",
        ):
            operation()

    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        assert dataset.status == "deleting"
        assert dataset.deletion_owner == deletion_owner


@pytest.mark.parametrize("deletion_owner", (None, "", "   "))
def test_dataset_service_requires_nonempty_deletion_owner(
    monkeypatch,
    deletion_owner,
):
    service = DatasetService()
    monkeypatch.setattr(
        service,
        "_get_engine",
        lambda: pytest.fail("empty owner must be rejected before database access"),
    )

    operations = (
        lambda: service.mark_deleting(
            DATASET_ID,
            deletion_owner=deletion_owner,
            user_id=USER["user_id"],
        ),
        lambda: service.restore_from_deleting(
            DATASET_ID,
            deletion_owner=deletion_owner,
            status="ready",
            user_id=USER["user_id"],
        ),
        lambda: service.delete_dataset(
            DATASET_ID,
            deletion_owner=deletion_owner,
            user_id=USER["user_id"],
        ),
    )
    for operation in operations:
        with pytest.raises(ValueError, match="deletion_owner"):
            operation()


def test_delete_maps_foreign_deletion_owner_to_conflict_without_restore(monkeypatch):
    class ForeignDeletionOwnerError(ValueError):
        pass

    monkeypatch.setattr(
        dataset_routes,
        "DatasetDeletionOwnerConflictError",
        ForeignDeletionOwnerError,
        raising=False,
    )
    events, state = _configure_delete(
        monkeypatch,
        status="deleting",
        deletion_owner="generation:other-task",
    )

    def reject_foreign_owner(*_args, **_kwargs):
        raise ForeignDeletionOwnerError(
            "Dataset is owned by another deletion operation"
        )

    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "mark_deleting",
        reject_foreign_owner,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset(DATASET_ID, USER))

    assert exc_info.value.status_code == 409
    assert "another deletion operation" in exc_info.value.detail
    assert state["status"] == "deleting"
    assert state["deletion_owner"] == "generation:other-task"
    assert not any(
        isinstance(event, tuple) and event[0] == "restore" for event in events
    )


def test_delete_guard_rejects_concurrent_delete_for_same_dataset(monkeypatch):
    first_mark_started = threading.Event()
    release_first_mark = threading.Event()
    events, _state = _configure_delete(monkeypatch)
    original_mark = dataset_routes.dataset_service.mark_deleting
    mark_calls = 0
    mark_calls_lock = threading.Lock()

    def blocking_mark(*args, **kwargs):
        nonlocal mark_calls
        with mark_calls_lock:
            mark_calls += 1
            call_number = mark_calls
        if call_number == 1:
            first_mark_started.set()
            assert release_first_mark.wait(timeout=5)
        return original_mark(*args, **kwargs)

    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "mark_deleting",
        blocking_mark,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            asyncio.run,
            dataset_routes.delete_dataset(DATASET_ID, USER),
        )
        assert first_mark_started.wait(timeout=5)
        second = executor.submit(
            asyncio.run,
            dataset_routes.delete_dataset(DATASET_ID, USER),
        )
        try:
            with pytest.raises(HTTPException) as exc_info:
                second.result(timeout=2)
            assert exc_info.value.status_code == 409
        finally:
            release_first_mark.set()
        assert first.result(timeout=5) == {
            "message": f"Dataset {DATASET_ID} deleted"
        }

    assert mark_calls == 1
    assert events.count("delete_record") == 1


@pytest.fixture
def sqlite_dataset_service():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[DatasetDB.__table__])
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id=DATASET_ID,
                dataset_name="dataset",
                display_name="Original name",
                description="Original description",
                dataset_type="custom",
                usage="train",
                model_type=["llm"],
                tags=["original"],
                extra_metadata={"source": "fixture"},
                user_id=USER["user_id"],
                status="ready",
            )
        )
        session.commit()

    service = DatasetService()
    service.engine = engine
    try:
        yield service, engine
    finally:
        engine.dispose()


def test_dataset_update_cas_outcome_requires_exactly_one_result():
    with pytest.raises(ValueError, match="exactly one"):
        DatasetUpdateOutcome()
    with pytest.raises(ValueError, match="exactly one"):
        DatasetUpdateOutcome(dataset={"dataset_id": DATASET_ID}, conflict=True)

    assert DatasetUpdateOutcome(dataset={}).dataset == {}
    assert DatasetUpdateOutcome(missing=True).missing is True
    assert DatasetUpdateOutcome(ownership_mismatch=True).ownership_mismatch is True
    assert DatasetUpdateOutcome(conflict=True).conflict is True


def test_dataset_update_cas_rejects_stale_snapshot_after_delete_fence(
    sqlite_dataset_service,
):
    service, engine = sqlite_dataset_service
    snapshot = service.get_dataset(DATASET_ID)
    assert snapshot is not None

    deletion_owner = f"dataset:{DATASET_ID}"
    assert service.mark_deleting(
        DATASET_ID,
        deletion_owner=deletion_owner,
        user_id=USER["user_id"],
    )

    outcome = service.update_dataset(
        DATASET_ID,
        expected_user_id=snapshot["user_id"],
        expected_status=snapshot["status"],
        expected_deletion_owner=None,
        display_name="Stale update",
    )

    assert outcome.conflict is True
    assert outcome.dataset is None
    with Session(engine) as session:
        persisted = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        assert persisted.status == "deleting"
        assert persisted.deletion_owner == deletion_owner
        assert persisted.display_name == "Original name"


def test_dataset_update_cas_matches_none_user_id_exactly(sqlite_dataset_service):
    service, engine = sqlite_dataset_service
    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        dataset.user_id = None
        session.add(dataset)
        session.commit()

    wrong_owner = service.update_dataset(
        DATASET_ID,
        expected_user_id=USER["user_id"],
        expected_status="ready",
        expected_deletion_owner=None,
        display_name="Wrong owner update",
    )
    assert wrong_owner.ownership_mismatch is True

    legacy_owner = service.update_dataset(
        DATASET_ID,
        expected_user_id=None,
        expected_status="ready",
        expected_deletion_owner=None,
        display_name="Legacy owner update",
    )
    assert legacy_owner.dataset is not None
    assert legacy_owner.dataset["display_name"] == "Legacy owner update"


def test_dataset_update_cas_rejects_foreign_user(sqlite_dataset_service):
    service, engine = sqlite_dataset_service

    outcome = service.update_dataset(
        DATASET_ID,
        expected_user_id="user-2",
        expected_status="ready",
        expected_deletion_owner=None,
        display_name="Foreign update",
    )

    assert outcome.ownership_mismatch is True
    with Session(engine) as session:
        persisted = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        assert persisted.display_name == "Original name"
        assert persisted.user_id == USER["user_id"]


def test_dataset_metadata_update_and_ready_archived_transition_succeed(
    sqlite_dataset_service,
):
    service, engine = sqlite_dataset_service

    archived = service.update_dataset(
        DATASET_ID,
        expected_user_id=USER["user_id"],
        expected_status="ready",
        expected_deletion_owner=None,
        display_name="Archived name",
        description="Archived description",
        dataset_type="sft_instruct",
        usage="eval",
        model_type=["llm", "embedding"],
        tags=["updated", "archived"],
        extra_metadata={"reviewed": True},
        status="archived",
    )

    assert archived.dataset is not None
    assert archived.dataset["display_name"] == "Archived name"
    assert archived.dataset["description"] == "Archived description"
    assert archived.dataset["dataset_type"] == "sft_instruct"
    assert archived.dataset["usage"] == "eval"
    assert archived.dataset["model_type"] == ["llm", "embedding"]
    assert archived.dataset["tags"] == ["updated", "archived"]
    assert archived.dataset["extra_metadata"] == {"reviewed": True}
    assert archived.dataset["status"] == "archived"

    ready = service.update_dataset(
        DATASET_ID,
        expected_user_id=USER["user_id"],
        expected_status="archived",
        expected_deletion_owner=None,
        display_name="Ready again",
        status="ready",
    )

    assert ready.dataset is not None
    assert ready.dataset["display_name"] == "Ready again"
    assert ready.dataset["status"] == "ready"
    with Session(engine) as session:
        persisted = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        assert persisted.display_name == ready.dataset["display_name"]
        assert persisted.status == ready.dataset["status"]
        assert persisted.tags == ready.dataset["tags"]


def test_dataset_put_delete_thread_race_preserves_delete_fence(tmp_path):
    database_path = tmp_path / "dataset-put-delete-race.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    setup_engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    put_engine = None
    delete_engine = None
    put_thread = None
    delete_thread = None
    put_update_started = threading.Event()
    delete_committed = threading.Event()
    try:
        with setup_engine.connect() as connection:
            journal_mode = connection.exec_driver_sql(
                "PRAGMA journal_mode=WAL"
            ).scalar()
            assert journal_mode.lower() == "wal"
            connection.commit()
        SQLModel.metadata.create_all(setup_engine, tables=[DatasetDB.__table__])
        with Session(setup_engine) as session:
            session.add(
                DatasetDB(
                    dataset_id=DATASET_ID,
                    dataset_name="dataset",
                    display_name="Original name",
                    description="Original description",
                    tags=["original"],
                    user_id=USER["user_id"],
                    status="ready",
                )
            )
            session.commit()

        put_engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 5},
        )
        delete_engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 5},
        )
        thread_errors = Queue()
        results = {}
        put_dataset_sql = []
        identifier_quotes = str.maketrans("", "", '"`[]')

        def before_put_cursor_execute(
            connection,
            _cursor,
            statement,
            _parameters,
            _context,
            _executemany,
        ):
            if connection.info.get("dataset_thread_role") != "put":
                return
            normalized = "".join(
                statement.lower().translate(identifier_quotes).split()
            )
            if "datasets" not in normalized:
                return
            put_dataset_sql.append(normalized)
            if (
                normalized.startswith("updatedatasetsset")
                and not put_update_started.is_set()
            ):
                put_update_started.set()
                if not delete_committed.wait(timeout=5):
                    raise TimeoutError(
                        "DELETE did not commit before PUT was released"
                    )

        def run_put():
            try:
                with put_engine.connect() as connection:
                    connection.info["dataset_thread_role"] = "put"
                    event.listen(
                        connection,
                        "before_cursor_execute",
                        before_put_cursor_execute,
                    )
                    service = DatasetService()
                    service.engine = connection
                    results["put"] = service.update_dataset(
                        DATASET_ID,
                        expected_user_id=USER["user_id"],
                        expected_status="ready",
                        expected_deletion_owner=None,
                        display_name="Racing PUT",
                        description="Racing description",
                        tags=["racing"],
                    )
            except BaseException as exc:  # surfaced in the main test thread
                thread_errors.put(("put", exc))
                put_update_started.set()

        def run_delete():
            try:
                if not put_update_started.wait(timeout=5):
                    raise TimeoutError(
                        "PUT did not reach its first datasets UPDATE"
                    )
                with delete_engine.connect() as connection:
                    service = DatasetService()
                    service.engine = connection
                    results["delete"] = service.mark_deleting(
                        DATASET_ID,
                        deletion_owner=f"dataset:{DATASET_ID}",
                        user_id=USER["user_id"],
                    )
            except BaseException as exc:  # surfaced in the main test thread
                thread_errors.put(("delete", exc))
            finally:
                delete_committed.set()

        put_thread = threading.Thread(target=run_put, name="dataset-put")
        delete_thread = threading.Thread(target=run_delete, name="dataset-delete")
        put_thread.start()
        delete_thread.start()
        put_thread.join(timeout=10)
        delete_thread.join(timeout=10)
        if put_thread.is_alive() or delete_thread.is_alive():
            pytest.fail("PUT/DELETE race threads did not finish within the bound")

        errors = list(thread_errors.queue)
        assert errors == []
        assert results["delete"] is True
        assert results["put"].conflict is True
        assert put_dataset_sql
        first_access = put_dataset_sql[0]
        assert first_access.startswith("updatedatasetsset")
        assert "where" in first_access
        where_clause = first_access.split("where", 1)[1]
        assert "datasets.dataset_id=" in where_clause
        assert "datasets.user_id=" in where_clause
        assert "datasets.status=" in where_clause
        assert "datasets.deletion_ownerisnull" in where_clause
        assert "datasets.statusnotin" in where_clause

        with Session(setup_engine) as session:
            persisted = session.exec(
                select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
            ).one()
            assert persisted.status == "deleting"
            assert persisted.deletion_owner == f"dataset:{DATASET_ID}"
            assert persisted.display_name == "Original name"
            assert persisted.description == "Original description"
            assert persisted.tags == ["original"]
    finally:
        put_update_started.set()
        delete_committed.set()
        for thread in (put_thread, delete_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=6)
        for engine in (put_engine, delete_engine, setup_engine):
            if engine is not None:
                engine.dispose()


def test_update_dataset_maps_delete_fence_loss_to_conflict(monkeypatch):
    get_calls = []
    update_calls = []

    def get_dataset(dataset_id):
        get_calls.append(dataset_id)
        return {
            "dataset_id": dataset_id,
            "user_id": USER["user_id"],
            "status": "ready",
        }

    def update_dataset(**kwargs):
        update_calls.append(kwargs)
        return DatasetUpdateOutcome(conflict=True)

    monkeypatch.setattr(dataset_routes.dataset_service, "get_dataset", get_dataset)
    monkeypatch.setattr(dataset_routes.dataset_service, "update_dataset", update_dataset)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            dataset_routes.update_dataset(
                DATASET_ID,
                dataset_routes.UpdateDatasetRequest(display_name="Updated name"),
                USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Dataset changed while the update was in progress"
    assert get_calls == [DATASET_ID]
    assert len(update_calls) == 1
    assert update_calls[0]["expected_user_id"] == USER["user_id"]
    assert update_calls[0]["expected_status"] == "ready"
    assert update_calls[0]["expected_deletion_owner"] is None


@pytest.mark.parametrize(
    ("outcome_kind", "expected_status_code"),
    (
        ("success", 200),
        ("missing", 404),
        ("ownership_mismatch", 403),
    ),
)
def test_update_dataset_maps_nonconflict_outcomes(
    monkeypatch,
    outcome_kind,
    expected_status_code,
):
    get_calls = []
    update_calls = []
    committed_snapshot = DatasetDB(
        dataset_id=DATASET_ID,
        dataset_name="dataset",
        display_name="Committed name",
        usage="train",
        user_id=USER["user_id"],
        status="ready",
    ).to_dict()
    outcomes = {
        "success": DatasetUpdateOutcome(dataset=committed_snapshot),
        "missing": DatasetUpdateOutcome(missing=True),
        "ownership_mismatch": DatasetUpdateOutcome(ownership_mismatch=True),
    }

    def get_dataset(dataset_id):
        get_calls.append(dataset_id)
        return {
            "dataset_id": dataset_id,
            "user_id": USER["user_id"],
            "status": "ready",
        }

    def update_dataset(**kwargs):
        update_calls.append(kwargs)
        return outcomes[outcome_kind]

    monkeypatch.setattr(dataset_routes.dataset_service, "get_dataset", get_dataset)
    monkeypatch.setattr(dataset_routes.dataset_service, "update_dataset", update_dataset)
    app = FastAPI()
    app.include_router(dataset_routes.router)
    app.dependency_overrides[dataset_routes.get_current_user] = lambda: USER

    with TestClient(app) as client:
        response = client.put(
            f"/datasets/{DATASET_ID}",
            json={"display_name": "Requested name"},
        )

    assert response.status_code == expected_status_code
    assert get_calls == [DATASET_ID]
    assert len(update_calls) == 1
    if outcome_kind == "success":
        response_payload = response.json()
        assert response_payload["dataset_id"] == DATASET_ID
        assert response_payload["display_name"] == "Committed name"
        assert response_payload["status"] == "ready"


def test_dataset_put_cannot_create_unowned_deleting_fence(monkeypatch):
    service_calls = []
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "get_dataset",
        lambda *_args, **_kwargs: service_calls.append("get"),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "update_dataset",
        lambda *_args, **_kwargs: service_calls.append("update"),
    )
    app = FastAPI()
    app.include_router(dataset_routes.router)
    app.dependency_overrides[dataset_routes.get_current_user] = lambda: USER

    with TestClient(app) as client:
        for status in ("deleting", "uploading", "processing", "not-a-status"):
            response = client.put(
                f"/datasets/{DATASET_ID}",
                json={"status": status},
            )
            assert response.status_code == 422

    assert service_calls == []
    for payload, expected_status in (
        ({}, None),
        ({"status": "ready"}, "ready"),
        ({"status": "archived"}, "archived"),
    ):
        request = dataset_routes.UpdateDatasetRequest.model_validate(payload)
        assert request.status == expected_status


@pytest.mark.parametrize("current_status", ("registered", "error"))
@pytest.mark.parametrize("target_status", ("ready", "archived"))
def test_dataset_public_status_transition_rejects_nonpublic_source(
    monkeypatch,
    current_status,
    target_status,
):
    update_calls = []
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "user_id": USER["user_id"],
            "status": current_status,
        },
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "update_dataset",
        lambda *_args, **kwargs: update_calls.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            dataset_routes.update_dataset(
                DATASET_ID,
                dataset_routes.UpdateDatasetRequest(status=target_status),
                USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Dataset status cannot be changed from its current state"
    assert update_calls == []
