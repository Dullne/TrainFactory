"""Regression tests for crash-safe external sync recovery."""

import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from train_factory.api import server
from train_factory.enums.sync_status import (
    BatchStatus,
    SyncGenerationStatus,
    SyncStatus,
    SyncTrainingStatus,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncBatchDB,
    ExternalSyncGenerationDB,
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.services.external_sync_service import (
    ExternalSyncService,
    external_sync_service,
)
from train_factory.sync import level2_handler
from train_factory.sync import sync_worker


def _sync_recovery_service():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        ExternalSyncTaskDB.__table__,
        ExternalSyncBatchDB.__table__,
        ExternalSyncGenerationDB.__table__,
        ExternalSyncTrainingDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
    ):
        table.create(engine)
    service = ExternalSyncService()
    service.engine = engine
    return service, engine


def test_list_tasks_uses_unique_tie_breaker_for_offset_pagination():
    service, engine = _sync_recovery_service()
    created_at = datetime(2026, 8, 11, 10, 0, 0)
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=f"sync-task-{suffix}",
                    task_name=f"Sync task {suffix}",
                    user_id="user-pagination",
                    created_at=created_at,
                )
                for suffix in ("a", "b", "c")
            ]
        )
        session.commit()

    first, total = service.list_tasks(limit=2, offset=0)
    second, second_total = service.list_tasks(limit=2, offset=2)

    assert total == second_total == 3
    assert [task["task_id"] for task in first + second] == [
        "sync-task-c",
        "sync-task-b",
        "sync-task-a",
    ]


def test_reset_sync_position_removes_invalidated_batch_files_after_commit(
    monkeypatch,
    tmp_path,
):
    service, engine = _sync_recovery_service()
    task_id = "sync-task-reset"
    user_id = "user-reset"
    sync_root = tmp_path / "sync"
    batch_dir = sync_root / user_id / task_id
    batch_dir.mkdir(parents=True)
    fetched_path = batch_dir / "batch_fetched.jsonl"
    registered_path = batch_dir / "batch_registered.jsonl"
    fetched_path.write_text('{"content":"fetched"}\n', encoding="utf-8")
    registered_path.write_text('{"content":"registered"}\n', encoding="utf-8")
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))

    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Reset sync task",
                    user_id=user_id,
                    last_sync_at=datetime(2026, 8, 11, 10, 0, 0),
                    pending_record_count=3,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-fetched",
                    task_id=task_id,
                    user_id=user_id,
                    record_count=2,
                    storage_path=str(fetched_path),
                    status=BatchStatus.FETCHED,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-registered",
                    task_id=task_id,
                    user_id=user_id,
                    record_count=1,
                    storage_path=str(registered_path),
                    status=BatchStatus.REGISTERED,
                ),
            ]
        )
        session.commit()

    updated = service.update_task(task_id, last_sync_at=None)

    assert updated["pending_record_count"] == 0
    assert not fetched_path.exists()
    assert not registered_path.exists()
    with Session(engine) as session:
        assert session.exec(select(ExternalSyncBatchDB)).all() == []
    assert list(Path(sync_root).rglob("batch_*.jsonl")) == []


def test_delete_sync_task_removes_managed_batch_files_after_commit(
    monkeypatch,
    tmp_path,
):
    service, engine = _sync_recovery_service()
    task_id = "sync-task-delete"
    user_id = "user-delete"
    sync_root = tmp_path / "sync"
    batch_path = sync_root / user_id / task_id / "batch_delete.jsonl"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text('{"content":"delete"}\n', encoding="utf-8")
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))

    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Delete sync task",
                    user_id=user_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-delete",
                    task_id=task_id,
                    user_id=user_id,
                    record_count=1,
                    storage_path=str(batch_path),
                    status=BatchStatus.FETCHED,
                ),
            ]
        )
        session.commit()

    assert service.delete_task(task_id) is True

    assert not batch_path.exists()
    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTaskDB)).all() == []
        assert session.exec(select(ExternalSyncBatchDB)).all() == []


def _finalize_sync_task(
    service,
    intent,
    *,
    batch_snapshot=(),
    generation_snapshot=(),
    training_snapshot=(),
):
    return service.finalize_task_deletion(
        intent["task_id"],
        cascade=intent["status"] == SyncStatus.DELETING_CASCADE,
        expected_user_id=intent["user_id"],
        expected_task_identity=intent["_deletion_identity"],
        batch_snapshot=tuple(batch_snapshot),
        generation_snapshot=tuple(generation_snapshot),
        training_snapshot=tuple(training_snapshot),
        expected_training_target_ids=tuple(intent["_training_target_ids"]),
    )


def test_strict_sync_finalizer_deletes_validated_metadata_without_batch_cleanup(
    monkeypatch,
    tmp_path,
):
    service, engine = _sync_recovery_service()
    task_id = "sync-strict-finalize"
    user_id = "user-strict-finalize"
    batch_path = tmp_path / "batch.jsonl"
    batch_path.write_text("{}\n", encoding="utf-8")
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Strict finalize",
                    user_id=user_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-strict",
                    task_id=task_id,
                    user_id=user_id,
                    storage_path=str(batch_path),
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id="generation-strict",
                    user_id=user_id,
                ),
                ExternalSyncTrainingDB(
                    task_id=task_id,
                    training_task_id="training-strict",
                    user_id=user_id,
                ),
                ExternalSyncTrainingTargetDB(
                    task_id=task_id,
                    target_id="target-strict",
                    target_name="Target strict",
                ),
            ]
        )
        session.commit()

    intent = service.begin_task_deletion(
        task_id,
        cascade=False,
        expected_user_id=user_id,
    )
    batches, _ = service.list_batches(task_id=task_id, limit=500, offset=0)
    generations, _ = service.list_generations(task_id=task_id, limit=500, offset=0)
    trainings, _ = service.list_trainings(task_id=task_id, limit=500, offset=0)
    monkeypatch.setattr(
        service,
        "_cleanup_managed_batch_files",
        lambda *_args, **_kwargs: pytest.fail(
            "strict metadata finalization must not clean batch files"
        ),
    )

    assert _finalize_sync_task(
        service,
        intent,
        batch_snapshot=batches,
        generation_snapshot=generations,
        training_snapshot=trainings,
    ) is True

    assert batch_path.exists()
    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTaskDB)).all() == []
        assert session.exec(select(ExternalSyncBatchDB)).all() == []
        assert session.exec(select(ExternalSyncGenerationDB)).all() == []
        assert session.exec(select(ExternalSyncTrainingDB)).all() == []
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []


@pytest.mark.parametrize("drift_kind", ["owner", "status", "mode", "aba"])
def test_strict_sync_finalizer_rejects_parent_owner_status_or_aba_drift(
    drift_kind,
):
    service, engine = _sync_recovery_service()
    task_id = f"sync-finalizer-{drift_kind}"
    user_id = "user-finalizer-drift"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Finalizer drift",
                user_id=user_id,
            )
        )
        session.commit()

    intent = service.begin_task_deletion(
        task_id,
        cascade=False,
        expected_user_id=user_id,
    )
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        if drift_kind == "owner":
            task.user_id = "user-foreign"
            session.add(task)
        elif drift_kind == "status":
            task.status = SyncStatus.IDLE
            session.add(task)
        elif drift_kind == "mode":
            task.status = SyncStatus.DELETING_CASCADE
            session.add(task)
        else:
            session.delete(task)
            session.flush()
            session.add(
                ExternalSyncTaskDB(
                    id=int(intent["_deletion_identity"]) + 100,
                    task_id=task_id,
                    task_name="Replacement row",
                    user_id=user_id,
                    is_active=False,
                    status=SyncStatus.DELETING,
                )
            )
        session.commit()

    with pytest.raises(ValueError, match="changed during deletion"):
        _finalize_sync_task(service, intent)

    with Session(engine) as session:
        remaining = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        assert remaining is not None


def test_strict_sync_finalizer_rejects_late_batch_without_cleaning_it(
    monkeypatch,
    tmp_path,
):
    service, engine = _sync_recovery_service()
    task_id = "sync-finalizer-late-batch"
    user_id = "user-finalizer-late-batch"
    initial_path = tmp_path / "initial.jsonl"
    late_path = tmp_path / "late.jsonl"
    initial_path.write_text("{}\n", encoding="utf-8")
    late_path.write_text("{}\n", encoding="utf-8")
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Late batch",
                    user_id=user_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-initial",
                    task_id=task_id,
                    user_id=user_id,
                    storage_path=str(initial_path),
                ),
            ]
        )
        session.commit()

    intent = service.begin_task_deletion(
        task_id,
        cascade=False,
        expected_user_id=user_id,
    )
    batch_snapshot, _ = service.list_batches(
        task_id=task_id,
        limit=500,
        offset=0,
    )
    with Session(engine) as session:
        session.add(
            ExternalSyncBatchDB(
                batch_id="batch-late",
                task_id=task_id,
                user_id=user_id,
                storage_path=str(late_path),
            )
        )
        session.commit()
    monkeypatch.setattr(
        service,
        "_cleanup_managed_batch_files",
        lambda *_args, **_kwargs: pytest.fail(
            "strict metadata finalization must not clean late batch files"
        ),
    )

    with pytest.raises(ValueError, match="batch metadata changed"):
        _finalize_sync_task(
            service,
            intent,
            batch_snapshot=batch_snapshot,
        )

    assert initial_path.exists()
    assert late_path.exists()
    with Session(engine) as session:
        assert len(session.exec(select(ExternalSyncTaskDB)).all()) == 1
        assert len(session.exec(select(ExternalSyncBatchDB)).all()) == 2


def test_strict_sync_finalizer_returns_false_when_parent_is_missing():
    service, _engine = _sync_recovery_service()

    assert service.finalize_task_deletion(
        "sync-finalizer-missing",
        cascade=False,
        expected_user_id="user-finalizer-missing",
        expected_task_identity=123,
        batch_snapshot=(),
        generation_snapshot=(),
        training_snapshot=(),
        expected_training_target_ids=(),
    ) is False


def test_strict_cascade_finalizer_accepts_prevalidated_tracking_already_removed():
    service, engine = _sync_recovery_service()
    task_id = "sync-finalizer-removed-tracking"
    user_id = "user-finalizer-removed-tracking"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Removed tracking",
                    user_id=user_id,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id="generation-removed",
                    user_id=user_id,
                ),
                ExternalSyncTrainingDB(
                    task_id=task_id,
                    training_task_id="training-removed",
                    user_id=user_id,
                ),
            ]
        )
        session.commit()

    intent = service.begin_task_deletion(
        task_id,
        cascade=True,
        expected_user_id=user_id,
    )
    generations, _ = service.list_generations(task_id=task_id, limit=500, offset=0)
    trainings, _ = service.list_trainings(task_id=task_id, limit=500, offset=0)
    assert service.delete_generation_tracking("generation-removed") is True
    assert service.delete_training_tracking("training-removed") is True

    assert _finalize_sync_task(
        service,
        intent,
        generation_snapshot=generations,
        training_snapshot=trainings,
    ) is True


def test_strict_cascade_finalizer_rejects_tracking_added_after_snapshot():
    service, engine = _sync_recovery_service()
    task_id = "sync-finalizer-new-tracking"
    user_id = "user-finalizer-new-tracking"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="New tracking",
                    user_id=user_id,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id="generation-original",
                    user_id=user_id,
                ),
            ]
        )
        session.commit()

    intent = service.begin_task_deletion(
        task_id,
        cascade=True,
        expected_user_id=user_id,
    )
    generations, _ = service.list_generations(task_id=task_id, limit=500, offset=0)
    assert service.delete_generation_tracking("generation-original") is True
    with Session(engine) as session:
        session.add(
            ExternalSyncGenerationDB(
                task_id=task_id,
                generation_task_id="generation-late",
                user_id=user_id,
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="generation metadata changed"):
        _finalize_sync_task(
            service,
            intent,
            generation_snapshot=generations,
        )

    with Session(engine) as session:
        assert len(session.exec(select(ExternalSyncTaskDB)).all()) == 1
        assert len(session.exec(select(ExternalSyncGenerationDB)).all()) == 1


def test_sync_deletion_parent_fences_validate_owner_and_mode():
    service, engine = _sync_recovery_service()
    task_id = "sync-parent-fences"
    user_id = "user-parent-fences"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Parent fences",
                user_id=user_id,
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="owner changed"):
        service.begin_task_deletion(
            task_id,
            cascade=True,
            expected_user_id="user-foreign",
        )
    assert service.get_task(task_id)["status"] == SyncStatus.IDLE

    service.begin_task_deletion(
        task_id,
        cascade=True,
        expected_user_id=user_id,
    )
    with pytest.raises(ValueError, match="owner changed"):
        service.cancel_task_deletion(
            task_id,
            status=SyncStatus.IDLE,
            is_active=True,
            expected_user_id="user-foreign",
            expected_deleting_status=SyncStatus.DELETING_CASCADE,
        )
    with pytest.raises(ValueError, match="mode changed"):
        service.cancel_task_deletion(
            task_id,
            status=SyncStatus.IDLE,
            is_active=True,
            expected_user_id=user_id,
            expected_deleting_status=SyncStatus.DELETING,
        )
    assert service.get_task(task_id)["status"] == SyncStatus.DELETING_CASCADE


@pytest.mark.parametrize(
    "deleting_status",
    [SyncStatus.DELETING, SyncStatus.DELETING_CASCADE],
)
def test_create_batch_rejects_parent_with_durable_delete_intent(
    deleting_status,
    tmp_path,
):
    service, engine = _sync_recovery_service()
    task_id = f"sync-create-batch-{deleting_status}"
    user_id = "user-create-batch-deleting"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Deleting batch parent",
                user_id=user_id,
                status=deleting_status,
                is_active=False,
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="deletion is in progress"):
        service.create_batch(
            task_id=task_id,
            user_id=user_id,
            record_count=1,
            storage_path=str(tmp_path / "late.jsonl"),
            since_time=None,
        )

    with Session(engine) as session:
        assert session.exec(select(ExternalSyncBatchDB)).all() == []


@pytest.mark.parametrize("parent_state", ["missing", "foreign", "deleting"])
def test_create_training_target_requires_mutable_owned_parent(parent_state):
    service, engine = _sync_recovery_service()
    task_id = f"sync-target-parent-{parent_state}"
    expected_user_id = "user-target-parent"
    if parent_state != "missing":
        with Session(engine) as session:
            session.add(
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Target parent",
                    user_id=(
                        "user-foreign"
                        if parent_state == "foreign"
                        else expected_user_id
                    ),
                    status=(
                        SyncStatus.DELETING
                        if parent_state == "deleting"
                        else SyncStatus.IDLE
                    ),
                )
            )
            session.commit()

    with pytest.raises(ValueError, match="sync task|Sync task"):
        service.create_training_target(
            task_id=task_id,
            target_name="Blocked target",
            expected_user_id=expected_user_id,
        )

    with Session(engine) as session:
        assert session.exec(select(ExternalSyncTrainingTargetDB)).all() == []


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_training_target_mutation_rejects_deleting_parent(operation):
    service, engine = _sync_recovery_service()
    task_id = f"sync-target-{operation}"
    target_id = f"target-{operation}"
    user_id = "user-target-mutation"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Deleting target parent",
                    user_id=user_id,
                    status=SyncStatus.DELETING_CASCADE,
                    is_active=False,
                ),
                ExternalSyncTrainingTargetDB(
                    task_id=task_id,
                    target_id=target_id,
                    target_name="Original target",
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="deletion is in progress"):
        if operation == "update":
            service.update_training_target(
                target_id,
                task_id=task_id,
                expected_user_id=user_id,
                target_name="Changed target",
            )
        else:
            service.delete_training_target(
                target_id,
                task_id=task_id,
                expected_user_id=user_id,
            )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
        ).one()
        assert target.target_name == "Original target"


@pytest.mark.parametrize("operation", ["update", "delete"])
@pytest.mark.parametrize("drift", ["owner", "target_parent"])
def test_training_target_mutation_rejects_parent_identity_drift(operation, drift):
    service, engine = _sync_recovery_service()
    task_id = f"sync-target-identity-{operation}-{drift}"
    other_task_id = f"{task_id}-replacement"
    target_id = f"target-identity-{operation}-{drift}"
    expected_user_id = "user-target-owner"
    target_task_id = other_task_id if drift == "target_parent" else task_id
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Requested target parent",
                    user_id=(
                        "user-owner-replacement"
                        if drift == "owner"
                        else expected_user_id
                    ),
                    status=SyncStatus.IDLE,
                ),
                ExternalSyncTaskDB(
                    task_id=other_task_id,
                    task_name="Replacement target parent",
                    user_id="user-other",
                    status=SyncStatus.IDLE,
                ),
                ExternalSyncTrainingTargetDB(
                    task_id=target_task_id,
                    target_id=target_id,
                    target_name="Original target",
                ),
            ]
        )
        session.commit()

    if drift == "owner":
        with pytest.raises(ValueError, match="ownership mismatch"):
            if operation == "update":
                service.update_training_target(
                    target_id,
                    task_id=task_id,
                    expected_user_id=expected_user_id,
                    target_name="Changed target",
                )
            else:
                service.delete_training_target(
                    target_id,
                    task_id=task_id,
                    expected_user_id=expected_user_id,
                )
    elif operation == "update":
        assert (
            service.update_training_target(
                target_id,
                task_id=task_id,
                expected_user_id=expected_user_id,
                target_name="Changed target",
            )
            is None
        )
    else:
        assert not service.delete_training_target(
            target_id,
            task_id=task_id,
            expected_user_id=expected_user_id,
        )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
        ).one()
        assert target.task_id == target_task_id
        assert target.target_name == "Original target"


def test_generation_scope_lock_order_is_parent_then_generation():
    service = ExternalSyncService()
    events = []
    generation = SimpleNamespace(
        task_id="sync-1",
        generation_task_id="generation-1",
    )
    task = SimpleNamespace(task_id="sync-1")

    class Result:
        def __init__(self, value):
            self.value = value

        def first(self):
            return self.value

    class RecordingSession:
        def exec(self, statement):
            entity = statement.column_descriptions[0]["entity"]
            locked = statement._for_update_arg is not None
            events.append((entity.__name__, locked))
            if entity is ExternalSyncTaskDB:
                return Result(task)
            return Result(generation)

    locked_task, locked_generation = service._lock_generation_scope(
        RecordingSession(),
        "generation-1",
    )

    assert locked_task is task
    assert locked_generation is generation
    assert events == [
        ("ExternalSyncGenerationDB", False),
        ("ExternalSyncTaskDB", True),
        ("ExternalSyncGenerationDB", True),
    ]


def test_toggle_generation_disabled_reconciles_final_and_qa_target_counters():
    service, engine = _sync_recovery_service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-toggle",
                task_name="Sync toggle",
                user_id="user-1",
                status=SyncStatus.IDLE,
                pending_training_samples=100,
                total_training_samples=200,
            )
        )
        session.add(
            ExternalSyncGenerationDB(
                task_id="sync-toggle",
                generation_task_id="generation-toggle",
                user_id="user-1",
                input_batch_ids=["batch-toggle"],
                status=SyncGenerationStatus.COMPLETED,
                output_dataset_id="dataset-final",
                output_sample_count=7,
                qa_dataset_id="dataset-qa",
                qa_sample_count=3,
            )
        )
        session.add_all(
            [
                ExternalSyncBatchDB(
                    batch_id="batch-toggle",
                    task_id="sync-toggle",
                    user_id="user-1",
                    record_count=4,
                    storage_path="/data/batch-toggle.jsonl",
                    status=BatchStatus.GENERATION_DONE,
                    generation_task_id="generation-toggle",
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-final",
                    task_id="sync-toggle",
                    user_id="user-1",
                    target_name="Final",
                    data_phase="final",
                    pending_training_samples=10,
                    total_training_samples=20,
                    is_active=True,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-qa",
                    task_id="sync-toggle",
                    user_id="user-1",
                    target_name="QA",
                    data_phase="qa",
                    pending_training_samples=5,
                    total_training_samples=8,
                    is_active=True,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-inactive",
                    task_id="sync-toggle",
                    user_id="user-1",
                    target_name="Inactive",
                    data_phase="final",
                    pending_training_samples=11,
                    total_training_samples=21,
                    is_active=False,
                ),
            ]
        )
        session.commit()

    service.toggle_generation_disabled("generation-toggle", True)
    service.toggle_generation_disabled("generation-toggle", True)

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "sync-toggle"
            )
        ).one()
        targets = {
            target.target_id: target
            for target in session.exec(
                select(ExternalSyncTrainingTargetDB).where(
                    ExternalSyncTrainingTargetDB.task_id == "sync-toggle"
                )
            ).all()
        }
        batch = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-toggle"
            )
        ).one()
    assert task.pending_training_samples == 100
    assert task.total_training_samples == 200
    assert task.pending_record_count == 4
    assert batch.status == BatchStatus.FETCHED
    assert batch.generation_task_id is None
    assert targets["target-final"].pending_training_samples == 3
    assert targets["target-final"].total_training_samples == 13
    assert targets["target-qa"].pending_training_samples == 2
    assert targets["target-qa"].total_training_samples == 5
    assert targets["target-inactive"].pending_training_samples == 11
    assert targets["target-inactive"].total_training_samples == 21

    service.toggle_generation_disabled("generation-toggle", False)
    service.toggle_generation_disabled("generation-toggle", False)

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.task_id == "sync-toggle"
            )
        ).one()
        targets = {
            target.target_id: target
            for target in session.exec(
                select(ExternalSyncTrainingTargetDB).where(
                    ExternalSyncTrainingTargetDB.task_id == "sync-toggle"
                )
            ).all()
        }
        batch = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-toggle"
            )
        ).one()
    assert task.pending_training_samples == 100
    assert task.total_training_samples == 200
    assert task.pending_record_count == 0
    assert batch.status == BatchStatus.GENERATION_DONE
    assert batch.generation_task_id == "generation-toggle"
    assert targets["target-final"].pending_training_samples == 10
    assert targets["target-final"].total_training_samples == 20
    assert targets["target-qa"].pending_training_samples == 5
    assert targets["target-qa"].total_training_samples == 8
    assert targets["target-inactive"].pending_training_samples == 11
    assert targets["target-inactive"].total_training_samples == 21


def test_enabled_completed_generation_does_not_require_final_dataset():
    service, engine = _sync_recovery_service()
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncGenerationDB(
                    task_id="sync-eval",
                    generation_task_id="generation-eval",
                    user_id="user-1",
                    input_batch_ids=["batch-eval"],
                    status=SyncGenerationStatus.COMPLETED,
                    output_dataset_id=None,
                    output_sample_count=4,
                    disabled=False,
                ),
                ExternalSyncGenerationDB(
                    task_id="sync-disabled",
                    generation_task_id="generation-disabled",
                    user_id="user-1",
                    input_batch_ids=["batch-disabled"],
                    status=SyncGenerationStatus.COMPLETED,
                    output_dataset_id=None,
                    disabled=True,
                ),
            ]
        )
        session.commit()

    assert service.has_enabled_completed_generation("sync-eval")
    assert not service.has_enabled_completed_generation("sync-disabled")
    assert not service.has_enabled_completed_generation("sync-missing")


def test_generation_toggle_uses_ordered_scope_lock(monkeypatch):
    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service = ExternalSyncService()
    service.engine = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(service_module, "Session", lambda _engine: FakeSession())
    monkeypatch.setattr(
        service,
        "_lock_generation_scope",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("ordered scope lock")),
    )

    with pytest.raises(RuntimeError, match="ordered scope lock"):
        service.toggle_generation_disabled("generation-1", True)


def test_reset_completed_batches_locks_parent_before_batches(monkeypatch):
    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service = ExternalSyncService()
    service.engine = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
    statements = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def exec(self, statement):
            entity = statement.column_descriptions[0]["entity"]
            statements.append((entity, statement._for_update_arg is not None))
            if entity is ExternalSyncTaskDB:
                raise RuntimeError("parent locked first")
            pytest.fail("batches were read before the parent task lock")

    monkeypatch.setattr(service_module, "Session", lambda _engine: FakeSession())

    with pytest.raises(RuntimeError, match="parent locked first"):
        service.reset_completed_batches("sync-1")

    assert statements == [(ExternalSyncTaskDB, True)]


def test_reset_completed_batches_recomputes_pending_records():
    service, engine = _sync_recovery_service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-1",
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.GENERATING,
                pending_record_count=99,
            )
        )
        session.add_all(
            [
                ExternalSyncBatchDB(
                    batch_id="done-1",
                    task_id="sync-1",
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/done-1.jsonl",
                    status=BatchStatus.GENERATION_DONE,
                    generation_task_id="generation-old",
                ),
                ExternalSyncBatchDB(
                    batch_id="done-2",
                    task_id="sync-1",
                    user_id="user-1",
                    record_count=5,
                    storage_path="/data/done-2.jsonl",
                    status=BatchStatus.GENERATION_DONE,
                    generation_task_id="generation-old",
                ),
                ExternalSyncBatchDB(
                    batch_id="already-fetched",
                    task_id="sync-1",
                    user_id="user-1",
                    record_count=7,
                    storage_path="/data/fetched.jsonl",
                    status=BatchStatus.FETCHED,
                ),
            ]
        )
        session.commit()

    assert service.reset_completed_batches("sync-1") == 2

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == "sync-1")
        ).one()
        batches = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.task_id == "sync-1"
            )
        ).all()

    assert task.pending_record_count == 15
    assert all(batch.status == BatchStatus.FETCHED for batch in batches)
    assert all(batch.generation_task_id is None for batch in batches)


def test_generation_completion_rejects_partial_batch_ownership():
    service, engine = _sync_recovery_service()
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-1",
                    task_name="Sync task",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                    pending_record_count=0,
                ),
                ExternalSyncGenerationDB(
                    task_id="sync-1",
                    generation_task_id="generation-1",
                    user_id="user-1",
                    input_batch_ids=["batch-1", "batch-2"],
                    input_record_count=8,
                    status=SyncGenerationStatus.PENDING,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-1",
                    task_id="sync-1",
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/batch-1.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id="generation-1",
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-2",
                    task_id="sync-1",
                    user_id="user-1",
                    record_count=5,
                    storage_path="/data/batch-2.jsonl",
                    status=BatchStatus.FETCHED,
                    generation_task_id=None,
                ),
            ]
        )
        session.commit()

    result = service.complete_generation_and_consume_batches(
        "generation-1",
        output_dataset_id="dataset-1",
        output_sample_count=8,
    )

    assert result["tracking_found"] is True
    assert result["completed"] is False
    with Session(engine) as session:
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == "generation-1"
            )
        ).one()
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == "sync-1")
        ).one()

    assert generation.status == SyncGenerationStatus.PENDING
    assert task.status == SyncStatus.GENERATING


def test_create_batch_enforces_task_ownership_and_cumulative_record_quota(
    monkeypatch,
):
    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service, engine = _sync_recovery_service()
    monkeypatch.setattr(
        service_module,
        "get_settings",
        lambda: SimpleNamespace(
            sync_pending_max_batches_per_task=10,
            sync_pending_max_records_per_task=5,
        ),
        raising=False,
    )
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-1",
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.IDLE,
            )
        )
        session.commit()

    service.create_batch(
        task_id="sync-1",
        user_id="user-1",
        record_count=3,
        storage_path="/data/batch-1.jsonl",
        since_time=None,
        initial_status=BatchStatus.REGISTERED,
    )

    with pytest.raises(ValueError, match="pending record quota exceeded"):
        service.create_batch(
            task_id="sync-1",
            user_id="user-1",
            record_count=3,
            storage_path="/data/batch-2.jsonl",
            since_time=None,
            initial_status=BatchStatus.REGISTERED,
        )
    with pytest.raises(ValueError, match="ownership"):
        service.create_batch(
            task_id="sync-1",
            user_id="user-2",
            record_count=1,
            storage_path="/data/batch-3.jsonl",
            since_time=None,
            initial_status=BatchStatus.REGISTERED,
        )

    with Session(engine) as session:
        count = session.exec(select(ExternalSyncBatchDB)).all()
    assert len(count) == 1


def test_get_pending_batches_returns_fifo_prefix_for_legacy_overflow(monkeypatch):
    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service, engine = _sync_recovery_service()
    monkeypatch.setattr(
        service_module,
        "get_settings",
        lambda: SimpleNamespace(
            sync_pending_max_batches_per_task=2,
            sync_pending_max_records_per_task=100,
        ),
        raising=False,
    )
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-1",
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.IDLE,
            )
        )
        for index in range(3):
            session.add(
                ExternalSyncBatchDB(
                    batch_id=f"batch-{index}",
                    task_id="sync-1",
                    user_id="user-1",
                    record_count=1,
                    storage_path=f"/data/batch-{index}.jsonl",
                    status=BatchStatus.FETCHED,
                )
            )
        session.commit()

    pending = service.get_pending_batches("sync-1")

    assert [batch["batch_id"] for batch in pending] == ["batch-0", "batch-1"]


def test_generation_completion_is_atomic_scoped_and_idempotent():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-complete"
    generation_id = "generation-complete"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Sync task",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                    pending_training_samples=100,
                    total_training_samples=200,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    input_batch_ids=["batch-1", "batch-2"],
                    input_record_count=8,
                    status=SyncGenerationStatus.PENDING,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-1",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/batch-1.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-2",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=5,
                    storage_path="/data/batch-2.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-3",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=8,
                    storage_path="/data/batch-3.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id="other-generation",
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-4",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=13,
                    storage_path="/data/batch-4.jsonl",
                    status=BatchStatus.FETCHED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-final-active",
                    task_id=task_id,
                    target_name="Final active",
                    data_phase="final",
                    is_active=True,
                    pending_training_samples=5,
                    total_training_samples=10,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-final-inactive",
                    task_id=task_id,
                    target_name="Final inactive",
                    data_phase="final",
                    is_active=False,
                    pending_training_samples=7,
                    total_training_samples=11,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-qa-active",
                    task_id=task_id,
                    target_name="QA active",
                    data_phase="qa",
                    is_active=True,
                    pending_training_samples=17,
                    total_training_samples=19,
                ),
            ]
        )
        session.commit()

    first = service.complete_generation_and_consume_batches(
        generation_id,
        output_dataset_id="dataset-1",
        output_sample_count=13,
    )
    second = service.complete_generation_and_consume_batches(
        generation_id,
        output_dataset_id="dataset-should-not-replace",
        output_sample_count=999,
    )

    assert first == {
        "tracking_found": True,
        "completed": True,
        "already_completed": False,
        "task_id": task_id,
        "completed_batch_count": 2,
        "credited_sample_count": 13,
        "credited_target_count": 1,
        "parent_status_updated": True,
        "uses_training_targets": True,
    }
    assert second == {
        "tracking_found": True,
        "completed": False,
        "already_completed": True,
        "task_id": task_id,
        "completed_batch_count": 0,
        "credited_sample_count": 0,
        "credited_target_count": 0,
        "parent_status_updated": False,
        "uses_training_targets": False,
    }
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_id
            )
        ).one()
        batches = {
            row.batch_id: row
            for row in session.exec(select(ExternalSyncBatchDB)).all()
        }
        targets = {
            row.target_id: row
            for row in session.exec(select(ExternalSyncTrainingTargetDB)).all()
        }

    assert generation.status == SyncGenerationStatus.COMPLETED
    assert generation.output_dataset_id == "dataset-1"
    assert generation.output_sample_count == 13
    assert generation.completed_at is not None
    assert task.status == SyncStatus.IDLE
    assert task.pending_training_samples == 100
    assert task.total_training_samples == 200
    assert batches["batch-1"].status == BatchStatus.GENERATION_DONE
    assert batches["batch-2"].status == BatchStatus.GENERATION_DONE
    assert batches["batch-3"].status == BatchStatus.GENERATION_QUEUED
    assert batches["batch-4"].status == BatchStatus.FETCHED
    assert targets["target-final-active"].pending_training_samples == 18
    assert targets["target-final-active"].total_training_samples == 23
    assert targets["target-final-inactive"].pending_training_samples == 7
    assert targets["target-final-inactive"].total_training_samples == 11
    assert targets["target-qa-active"].pending_training_samples == 17
    assert targets["target-qa-active"].total_training_samples == 19


def test_generation_completion_credits_legacy_counters_once():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-legacy"
    generation_id = "generation-legacy"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Legacy sync task",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                    pending_training_samples=4,
                    total_training_samples=8,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    input_batch_ids=["batch-legacy"],
                    status=SyncGenerationStatus.PENDING,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-legacy",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=1,
                    storage_path="/data/batch-legacy.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
            ]
        )
        session.commit()

    service.complete_generation_and_consume_batches(
        generation_id,
        output_dataset_id="dataset-legacy",
        output_sample_count=7,
    )
    service.complete_generation_and_consume_batches(
        generation_id,
        output_dataset_id="dataset-legacy",
        output_sample_count=7,
    )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
    assert task.pending_training_samples == 11
    assert task.total_training_samples == 15


def test_generation_completion_rolls_back_all_state_when_commit_fails(monkeypatch):
    service, engine = _sync_recovery_service()
    task_id = "sync-task-rollback"
    generation_id = "generation-rollback"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Rollback task",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    input_batch_ids=["batch-rollback"],
                    qa_dataset_id="qa-dataset-rollback",
                    qa_sample_count=4,
                    status=SyncGenerationStatus.PENDING,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-rollback",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/batch-rollback.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-qa-rollback",
                    task_id=task_id,
                    target_name="Rollback QA target",
                    data_phase="qa",
                    pending_training_samples=2,
                    total_training_samples=3,
                ),
            ]
        )
        session.commit()

    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    real_session = service_module.Session

    class CommitFailingSession(real_session):
        def commit(self):
            raise RuntimeError("commit failed")

    monkeypatch.setattr(service_module, "Session", CommitFailingSession)

    with pytest.raises(RuntimeError, match="commit failed"):
        service.complete_generation_and_consume_batches(
            generation_id,
            output_dataset_id="dataset-rollback",
            output_sample_count=3,
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_id
            )
        ).one()
        batch = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-rollback"
            )
        ).one()
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == "target-qa-rollback"
            )
        ).one()

    assert task.status == SyncStatus.GENERATING
    assert task.pending_training_samples == 0
    assert task.total_training_samples == 0
    assert generation.status == SyncGenerationStatus.PENDING
    assert generation.output_dataset_id is None
    assert generation.output_sample_count == 0
    assert generation.completed_at is None
    assert batch.status == BatchStatus.GENERATION_QUEUED
    assert target.pending_training_samples == 2
    assert target.total_training_samples == 3


def test_qa_credit_and_schedule_wait_for_generation_completion(monkeypatch):
    service, engine = _sync_recovery_service()
    task_id = "sync-task-qa-deferred"
    generation_id = "generation-qa-deferred"
    target_id = "target-qa-deferred"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Deferred QA task",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    input_batch_ids=["batch-qa-deferred"],
                    status=SyncGenerationStatus.PENDING,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-qa-deferred",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=1,
                    storage_path="/data/batch-qa-deferred.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id=target_id,
                    task_id=task_id,
                    target_name="QA target",
                    data_phase="qa",
                    is_active=True,
                    training_threshold=5,
                ),
            ]
        )
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)
    schedules = []

    def schedule(config_id):
        schedules.append(
            (
                config_id,
                external_sync_service.get_all_completed_qa_datasets(config_id),
            )
        )

    monkeypatch.setattr(level2_handler, "_schedule_next_training", schedule)

    level2_handler.on_qa_phase_completed(
        generation_id,
        qa_dataset_id="qa-dataset-1",
        qa_sample_count=7,
    )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
        ).one()
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_id
            )
        ).one()
    assert generation.status == SyncGenerationStatus.PENDING
    assert generation.qa_dataset_id == "qa-dataset-1"
    assert generation.qa_sample_count == 7
    assert target.pending_training_samples == 0
    assert target.total_training_samples == 0
    assert schedules == []

    level2_handler.on_generation_completed(
        generation_id,
        output_dataset_id="final-dataset-1",
        output_sample_count=11,
    )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
        ).one()
    assert target.pending_training_samples == 7
    assert target.total_training_samples == 7
    assert schedules == [
        (
            task_id,
            [
                {
                    "dataset_id": "qa-dataset-1",
                    "sample_count": 7,
                    "generation_task_id": generation_id,
                }
            ],
        )
    ]


def test_failed_generation_never_credits_or_schedules_qa(monkeypatch):
    _service, engine = _sync_recovery_service()
    task_id = "sync-task-qa-failed"
    generation_id = "generation-qa-failed"
    target_id = "target-qa-failed"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Failed QA task",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    status=SyncGenerationStatus.PENDING,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id=target_id,
                    task_id=task_id,
                    target_name="QA target",
                    data_phase="qa",
                    is_active=True,
                    training_threshold=1,
                ),
            ]
        )
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)
    schedules = []
    monkeypatch.setattr(
        level2_handler,
        "_schedule_next_training",
        lambda config_id: schedules.append(config_id),
    )

    level2_handler.on_qa_phase_completed(
        generation_id,
        qa_dataset_id="qa-dataset-failed",
        qa_sample_count=9,
    )
    level2_handler.on_generation_failed(generation_id, "generation failed")
    level2_handler.on_qa_phase_completed(
        generation_id,
        qa_dataset_id="qa-dataset-failed",
        qa_sample_count=9,
    )

    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
        ).one()
    assert target.pending_training_samples == 0
    assert target.total_training_samples == 0
    assert schedules == []
    assert external_sync_service.get_all_completed_qa_datasets(task_id) == []


def test_qa_output_recording_is_idempotent_and_first_write_wins():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-qa-idempotent"
    generation_id = "generation-qa-idempotent"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Idempotent QA task",
                    user_id="user-1",
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    status=SyncGenerationStatus.PENDING,
                ),
            ]
        )
        session.commit()

    first = service.update_generation_qa_output(
        generation_id,
        qa_dataset_id="qa-dataset-original",
        qa_sample_count=6,
    )
    retry = service.update_generation_qa_output(
        generation_id,
        qa_dataset_id="qa-dataset-original",
        qa_sample_count=6,
    )
    conflict = service.update_generation_qa_output(
        generation_id,
        qa_dataset_id="qa-dataset-conflict",
        qa_sample_count=60,
    )

    assert first["recorded"] is True
    assert first["already_recorded"] is False
    assert retry["recorded"] is False
    assert retry["already_recorded"] is True
    assert retry["conflict"] is False
    assert conflict["recorded"] is False
    assert conflict["already_recorded"] is True
    assert conflict["conflict"] is True
    with Session(engine) as session:
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_id
            )
        ).one()
    assert generation.qa_dataset_id == "qa-dataset-original"
    assert generation.qa_sample_count == 6


def test_qa_callback_after_generation_completion_credits_and_schedules_once(
    monkeypatch,
):
    _service, engine = _sync_recovery_service()
    task_id = "sync-task-qa-late"
    generation_id = "generation-qa-late"
    target_id = "target-qa-late"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Late QA task",
                    user_id="user-1",
                    status=SyncStatus.IDLE,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    status=SyncGenerationStatus.COMPLETED,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id=target_id,
                    task_id=task_id,
                    target_name="Late QA target",
                    data_phase="qa",
                    is_active=True,
                    training_threshold=5,
                    pending_training_samples=2,
                    total_training_samples=3,
                ),
            ]
        )
        session.commit()

    monkeypatch.setattr(external_sync_service, "engine", engine)
    schedules = []

    def schedule(config_id):
        schedules.append(
            external_sync_service.get_all_completed_qa_datasets(config_id)
        )

    monkeypatch.setattr(level2_handler, "_schedule_next_training", schedule)

    first = level2_handler.on_qa_phase_completed(
        generation_id,
        qa_dataset_id="qa-dataset-late",
        qa_sample_count=8,
    )
    retry = level2_handler.on_qa_phase_completed(
        generation_id,
        qa_dataset_id="qa-dataset-late",
        qa_sample_count=8,
    )

    assert first["credited_sample_count"] == 8
    assert first["credited_target_count"] == 1
    assert first["schedule_training"] is True
    assert retry["recorded"] is False
    assert retry["already_recorded"] is True
    assert retry["credited_sample_count"] == 0
    assert retry["credited_target_count"] == 0
    assert retry["schedule_training"] is False
    with Session(engine) as session:
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
        ).one()
    assert target.pending_training_samples == 10
    assert target.total_training_samples == 11
    assert schedules == [
        [
            {
                "dataset_id": "qa-dataset-late",
                "sample_count": 8,
                "generation_task_id": generation_id,
            }
        ]
    ]


def test_late_qa_callback_rolls_back_metadata_and_credit_when_commit_fails(
    monkeypatch,
):
    service, engine = _sync_recovery_service()
    task_id = "sync-task-qa-late-rollback"
    generation_id = "generation-qa-late-rollback"
    target_id = "target-qa-late-rollback"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Late QA rollback task",
                    user_id="user-1",
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    status=SyncGenerationStatus.COMPLETED,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id=target_id,
                    task_id=task_id,
                    target_name="Late QA rollback target",
                    data_phase="qa",
                    is_active=True,
                    pending_training_samples=3,
                    total_training_samples=4,
                ),
            ]
        )
        session.commit()

    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    real_session = service_module.Session

    class CommitFailingSession(real_session):
        def commit(self):
            raise RuntimeError("qa commit failed")

    monkeypatch.setattr(service_module, "Session", CommitFailingSession)

    with pytest.raises(RuntimeError, match="qa commit failed"):
        service.update_generation_qa_output(
            generation_id,
            qa_dataset_id="qa-dataset-late-rollback",
            qa_sample_count=10,
        )

    with Session(engine) as session:
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_id
            )
        ).one()
        target = session.exec(
            select(ExternalSyncTrainingTargetDB).where(
                ExternalSyncTrainingTargetDB.target_id == target_id
            )
        ).one()
    assert generation.qa_dataset_id is None
    assert generation.qa_sample_count == 0
    assert target.pending_training_samples == 3
    assert target.total_training_samples == 4


def test_generation_failure_recovery_rejects_partial_batch_ownership():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-1"
    generation_id = "generation-1"
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id=task_id,
                    task_name="Sync task",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                    pending_record_count=99,
                ),
                ExternalSyncGenerationDB(
                    task_id=task_id,
                    generation_task_id=generation_id,
                    user_id="user-1",
                    input_batch_ids=[
                        "batch-1",
                        "batch-2",
                        "batch-3",
                        "batch-4",
                        "batch-5",
                    ],
                    input_record_count=999,
                    status=SyncGenerationStatus.PENDING,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-1",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/batch-1.jsonl",
                    dataset_id="dataset-raw",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-2",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=5,
                    storage_path="/data/batch-2.jsonl",
                    dataset_id="dataset-raw",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-3",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=11,
                    storage_path="/data/batch-3.jsonl",
                    dataset_id="dataset-raw",
                    status=BatchStatus.GENERATION_DONE,
                    generation_task_id=generation_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-4",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=13,
                    storage_path="/data/batch-4.jsonl",
                    dataset_id="dataset-newer",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id="newer-generation",
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-5",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=7,
                    storage_path="/data/batch-5.jsonl",
                    status=BatchStatus.FETCHED,
                    generation_task_id=generation_id,
                ),
            ]
        )
        session.commit()

    first = service.fail_generation_and_restore_batches(
        generation_id,
        "Task interrupted by server restart",
    )
    second = service.fail_generation_and_restore_batches(
        generation_id,
        "Task interrupted by server restart",
    )

    assert first == {
        "tracking_found": True,
        "recovered": False,
        "already_recovered": False,
        "reconciled": False,
        "task_id": task_id,
        "user_id": "user-1",
        "restored_batch_count": 0,
        "restored_record_count": 0,
        "parent_status_updated": False,
    }
    assert second == {
        "tracking_found": True,
        "recovered": False,
        "already_recovered": False,
        "reconciled": False,
        "task_id": task_id,
        "user_id": "user-1",
        "restored_batch_count": 0,
        "restored_record_count": 0,
        "parent_status_updated": False,
    }
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_id
            )
        ).one()
        batches = {
            row.batch_id: row
            for row in session.exec(select(ExternalSyncBatchDB)).all()
        }

    assert generation.status == SyncGenerationStatus.PENDING
    assert generation.completed_at is None
    assert task.status == SyncStatus.GENERATING
    assert task.pending_record_count == 99
    assert task.error_message is None
    assert batches["batch-1"].status == BatchStatus.GENERATION_QUEUED
    assert batches["batch-1"].generation_task_id == generation_id
    assert batches["batch-1"].dataset_id == "dataset-raw"
    assert batches["batch-2"].status == BatchStatus.GENERATION_QUEUED
    assert batches["batch-2"].generation_task_id == generation_id
    assert batches["batch-2"].dataset_id == "dataset-raw"
    assert batches["batch-3"].status == BatchStatus.GENERATION_DONE
    assert batches["batch-3"].generation_task_id == generation_id
    assert batches["batch-3"].dataset_id == "dataset-raw"
    assert batches["batch-4"].status == BatchStatus.GENERATION_QUEUED
    assert batches["batch-4"].generation_task_id == "newer-generation"
    assert batches["batch-4"].dataset_id == "dataset-newer"
    assert batches["batch-5"].status == BatchStatus.FETCHED
    assert batches["batch-5"].generation_task_id == generation_id


def test_generation_failure_recovery_restores_every_owned_input_atomically():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-recover-all"
    generation_id = "generation-recover-all"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.GENERATING,
                pending_record_count=99,
            )
        )
        session.add(
            ExternalSyncGenerationDB(
                task_id=task_id,
                generation_task_id=generation_id,
                user_id="user-1",
                input_batch_ids=["batch-1", "batch-2", "batch-3"],
                input_record_count=15,
                status=SyncGenerationStatus.PENDING,
            )
        )
        session.add_all(
            [
                ExternalSyncBatchDB(
                    batch_id="batch-1",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/batch-1.jsonl",
                    dataset_id="dataset-raw",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-2",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=5,
                    storage_path="/data/batch-2.jsonl",
                    dataset_id="dataset-raw",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id=generation_id,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-3",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=7,
                    storage_path="/data/batch-3.jsonl",
                    dataset_id="dataset-raw",
                    status=BatchStatus.FETCHED,
                    generation_task_id=generation_id,
                ),
            ]
        )
        session.commit()

    first = service.fail_generation_and_restore_batches(generation_id, "failed")
    second = service.fail_generation_and_restore_batches(generation_id, "failed")

    assert first == {
        "tracking_found": True,
        "recovered": True,
        "already_recovered": False,
        "reconciled": True,
        "task_id": task_id,
        "user_id": "user-1",
        "restored_batch_count": 2,
        "restored_record_count": 8,
        "parent_status_updated": True,
    }
    assert second == {
        "tracking_found": True,
        "recovered": False,
        "already_recovered": True,
        "reconciled": True,
        "task_id": task_id,
        "user_id": "user-1",
        "restored_batch_count": 0,
        "restored_record_count": 0,
        "parent_status_updated": False,
    }
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        generation = session.exec(
            select(ExternalSyncGenerationDB).where(
                ExternalSyncGenerationDB.generation_task_id == generation_id
            )
        ).one()
        batches = session.exec(select(ExternalSyncBatchDB)).all()
    assert task.status == SyncStatus.IDLE
    assert task.pending_record_count == 15
    assert generation.status == SyncGenerationStatus.FAILED
    assert all(batch.status == BatchStatus.FETCHED for batch in batches)
    assert all(batch.generation_task_id is None for batch in batches)
    assert all(batch.dataset_id is None for batch in batches)

    with Session(engine) as session:
        reclaimed = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-1"
            )
        ).one()
        reclaimed.status = BatchStatus.GENERATION_QUEUED
        reclaimed.generation_task_id = "new-generation"
        reclaimed.dataset_id = "new-raw-dataset"
        session.add(reclaimed)
        session.commit()

    retry_after_reclaim = service.fail_generation_and_restore_batches(
        generation_id,
        "cleanup retry",
    )
    assert retry_after_reclaim["already_recovered"] is True
    assert retry_after_reclaim["reconciled"] is True


def test_generation_creation_claims_batches_and_recomputes_pending_atomically():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-claim"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.GENERATING,
                pending_record_count=999,
            )
        )
        session.add_all(
            [
                ExternalSyncBatchDB(
                    batch_id="batch-1",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/batch-1.jsonl",
                    status=BatchStatus.FETCHED,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-2",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=5,
                    storage_path="/data/batch-2.jsonl",
                    status=BatchStatus.FETCHED,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-3",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=7,
                    storage_path="/data/batch-3.jsonl",
                    status=BatchStatus.FETCHED,
                ),
            ]
        )
        session.commit()

    created = service.create_generation_and_claim_batches(
        task_id=task_id,
        generation_task_id="generation-claim",
        user_id="user-1",
        input_batch_ids=["batch-1", "batch-2"],
        input_record_count=8,
        dataset_id="dataset-1",
    )

    assert created["generation_task_id"] == "generation-claim"
    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        batches = {
            row.batch_id: row
            for row in session.exec(select(ExternalSyncBatchDB)).all()
        }
    assert task.pending_record_count == 7
    assert batches["batch-1"].status == BatchStatus.GENERATION_QUEUED
    assert batches["batch-1"].generation_task_id == "generation-claim"
    assert batches["batch-1"].dataset_id == "dataset-1"
    assert batches["batch-2"].status == BatchStatus.GENERATION_QUEUED
    assert batches["batch-3"].status == BatchStatus.FETCHED


def test_generation_creation_conflict_rolls_back_every_change():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-conflict"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.GENERATING,
                pending_record_count=8,
            )
        )
        session.add_all(
            [
                ExternalSyncBatchDB(
                    batch_id="batch-free",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=3,
                    storage_path="/data/free.jsonl",
                    status=BatchStatus.FETCHED,
                ),
                ExternalSyncBatchDB(
                    batch_id="batch-owned",
                    task_id=task_id,
                    user_id="user-1",
                    record_count=5,
                    storage_path="/data/owned.jsonl",
                    status=BatchStatus.GENERATION_QUEUED,
                    generation_task_id="other-generation",
                ),
            ]
        )
        session.commit()

    with pytest.raises(ValueError, match="no longer available"):
        service.create_generation_and_claim_batches(
            task_id=task_id,
            generation_task_id="generation-conflict",
            user_id="user-1",
            input_batch_ids=["batch-free", "batch-owned"],
            input_record_count=8,
            dataset_id="dataset-conflict",
        )

    with Session(engine) as session:
        generations = session.exec(select(ExternalSyncGenerationDB)).all()
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        batches = {
            row.batch_id: row
            for row in session.exec(select(ExternalSyncBatchDB)).all()
        }
    assert generations == []
    assert task.pending_record_count == 8
    assert batches["batch-free"].status == BatchStatus.FETCHED
    assert batches["batch-free"].generation_task_id is None
    assert batches["batch-free"].dataset_id is None
    assert batches["batch-owned"].generation_task_id == "other-generation"


def test_generation_creation_rejects_another_pending_generation():
    from train_factory.storage.services.background_task_admission_service import (
        BackgroundTaskAlreadyExecuting,
    )

    service, engine = _sync_recovery_service()
    task_id = "sync-task-active-generation"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.GENERATING,
                pending_record_count=3,
            )
        )
        session.add(
            ExternalSyncGenerationDB(
                task_id=task_id,
                generation_task_id="generation-existing",
                user_id="user-1",
                input_batch_ids=["batch-existing"],
                input_record_count=5,
                status=SyncGenerationStatus.PENDING,
            )
        )
        session.add(
            ExternalSyncBatchDB(
                batch_id="batch-new",
                task_id=task_id,
                user_id="user-1",
                record_count=3,
                storage_path="/data/batch-new.jsonl",
                status=BatchStatus.FETCHED,
            )
        )
        session.commit()

    with pytest.raises(BackgroundTaskAlreadyExecuting, match="already pending"):
        service.create_generation_and_claim_batches(
            task_id=task_id,
            generation_task_id="generation-new",
            user_id="user-1",
            input_batch_ids=["batch-new"],
            input_record_count=3,
            dataset_id="dataset-new",
        )

    with Session(engine) as session:
        generations = session.exec(select(ExternalSyncGenerationDB)).all()
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        batch = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-new"
            )
        ).one()
    assert [row.generation_task_id for row in generations] == [
        "generation-existing"
    ]
    assert task.pending_record_count == 3
    assert batch.status == BatchStatus.FETCHED
    assert batch.generation_task_id is None
    assert batch.dataset_id is None


def test_startup_reset_is_atomic_with_pending_generation_check():
    service, engine = _sync_recovery_service()
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-safe-reset",
                    task_name="Safe reset",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                ),
                ExternalSyncTaskDB(
                    task_id="sync-pending",
                    task_name="Pending generation",
                    user_id="user-1",
                    status=SyncStatus.GENERATING,
                ),
                ExternalSyncGenerationDB(
                    task_id="sync-pending",
                    generation_task_id="generation-pending",
                    user_id="user-1",
                    input_batch_ids=["batch-pending"],
                    input_record_count=1,
                    status=SyncGenerationStatus.PENDING,
                ),
            ]
        )
        session.commit()

    assert service.reset_generating_task_if_no_pending_generation(
        "sync-safe-reset"
    )
    assert not service.reset_generating_task_if_no_pending_generation(
        "sync-pending"
    )

    with Session(engine) as session:
        tasks = {
            row.task_id: row
            for row in session.exec(select(ExternalSyncTaskDB)).all()
        }
    assert tasks["sync-safe-reset"].status == SyncStatus.IDLE
    assert tasks["sync-pending"].status == SyncStatus.GENERATING


def test_generation_creation_commit_failure_rolls_back_every_change(monkeypatch):
    service, engine = _sync_recovery_service()
    task_id = "sync-task-commit-failure"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.GENERATING,
                pending_record_count=3,
            )
        )
        session.add(
            ExternalSyncBatchDB(
                batch_id="batch-1",
                task_id=task_id,
                user_id="user-1",
                record_count=3,
                storage_path="/data/batch-1.jsonl",
                status=BatchStatus.FETCHED,
            )
        )
        session.commit()

    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    real_session = service_module.Session

    class CommitFailingSession(real_session):
        def commit(self):
            raise RuntimeError("commit failed")

    monkeypatch.setattr(service_module, "Session", CommitFailingSession)

    with pytest.raises(RuntimeError, match="commit failed"):
        service.create_generation_and_claim_batches(
            task_id=task_id,
            generation_task_id="generation-commit-failure",
            user_id="user-1",
            input_batch_ids=["batch-1"],
            input_record_count=3,
            dataset_id="dataset-1",
        )

    with Session(engine) as session:
        assert session.exec(select(ExternalSyncGenerationDB)).all() == []
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        batch = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-1"
            )
        ).one()
    assert task.pending_record_count == 3
    assert batch.status == BatchStatus.FETCHED
    assert batch.generation_task_id is None
    assert batch.dataset_id is None


def test_batch_promotion_updates_status_and_counters_once():
    service, engine = _sync_recovery_service()
    task_id = "sync-task-promote"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.SYNCING,
                pending_record_count=10,
                total_record_count=20,
            )
        )
        session.add(
            ExternalSyncBatchDB(
                batch_id="batch-promote",
                task_id=task_id,
                user_id="user-1",
                record_count=3,
                storage_path="/data/batch-promote.jsonl",
                status=BatchStatus.REGISTERED,
            )
        )
        session.commit()

    assert service.promote_batch_to_fetched(
        task_id,
        "batch-promote",
        "user-1",
    )
    assert not service.promote_batch_to_fetched(
        task_id,
        "batch-promote",
        "user-1",
    )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        batch = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-promote"
            )
        ).one()
    assert batch.status == BatchStatus.FETCHED
    assert task.pending_record_count == 13
    assert task.total_record_count == 23


def test_batch_promotion_commit_failure_rolls_back_status_and_counters(monkeypatch):
    service, engine = _sync_recovery_service()
    task_id = "sync-task-promote-failure"
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id=task_id,
                task_name="Sync task",
                user_id="user-1",
                status=SyncStatus.SYNCING,
                pending_record_count=10,
                total_record_count=20,
            )
        )
        session.add(
            ExternalSyncBatchDB(
                batch_id="batch-promote",
                task_id=task_id,
                user_id="user-1",
                record_count=3,
                storage_path="/data/batch-promote.jsonl",
                status=BatchStatus.REGISTERED,
            )
        )
        session.commit()

    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    real_session = service_module.Session

    class CommitFailingSession(real_session):
        def commit(self):
            raise RuntimeError("commit failed")

    monkeypatch.setattr(service_module, "Session", CommitFailingSession)

    with pytest.raises(RuntimeError, match="commit failed"):
        service.promote_batch_to_fetched(
            task_id,
            "batch-promote",
            "user-1",
        )

    with Session(engine) as session:
        task = session.exec(
            select(ExternalSyncTaskDB).where(ExternalSyncTaskDB.task_id == task_id)
        ).one()
        batch = session.exec(
            select(ExternalSyncBatchDB).where(
                ExternalSyncBatchDB.batch_id == "batch-promote"
            )
        ).one()
    assert batch.status == BatchStatus.REGISTERED
    assert task.pending_record_count == 10
    assert task.total_record_count == 20


def test_generation_failure_recovery_missing_tracking_is_noop():
    service, _engine = _sync_recovery_service()

    assert service.fail_generation_and_restore_batches("missing", "failed") == {
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


def test_generation_and_training_lists_use_deterministic_id_tiebreakers():
    service, engine = _sync_recovery_service()
    created_at = datetime(2026, 8, 11, 12, 0, 0)
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncGenerationDB(
                    id=1,
                    task_id="sync-order",
                    generation_task_id="generation-low",
                    user_id="user-1",
                    input_batch_ids=["batch-low"],
                    created_at=created_at,
                ),
                ExternalSyncGenerationDB(
                    id=2,
                    task_id="sync-order",
                    generation_task_id="generation-high",
                    user_id="user-1",
                    input_batch_ids=["batch-high"],
                    created_at=created_at,
                ),
                ExternalSyncTrainingDB(
                    id=1,
                    task_id="sync-order",
                    training_task_id="training-low",
                    user_id="user-1",
                    created_at=created_at,
                ),
                ExternalSyncTrainingDB(
                    id=2,
                    task_id="sync-order",
                    training_task_id="training-high",
                    user_id="user-1",
                    created_at=created_at,
                ),
            ]
        )
        session.commit()

    generations, _ = service.list_generations("sync-order")
    trainings, _ = service.list_trainings("sync-order")

    assert [row["generation_task_id"] for row in generations] == [
        "generation-high",
        "generation-low",
    ]
    assert [row["training_task_id"] for row in trainings] == [
        "training-high",
        "training-low",
    ]


def test_level2_generation_failure_delegates_to_atomic_recovery(monkeypatch):
    calls = []
    expected = {
        "tracking_found": True,
        "recovered": True,
        "restored_batch_count": 2,
        "restored_record_count": 8,
        "parent_status_updated": True,
    }
    monkeypatch.setattr(
        external_sync_service,
        "fail_generation_and_restore_batches",
        lambda generation_task_id, reason: calls.append(
            (generation_task_id, reason)
        )
        or expected,
    )

    result = level2_handler.on_generation_failed(
        "generation-1",
        error="worker failed",
    )

    assert result == expected
    assert calls == [("generation-1", "worker failed")]


def test_level2_generation_completion_commits_before_training_schedule(monkeypatch):
    events = []
    completion = {
        "tracking_found": True,
        "completed": True,
        "already_completed": False,
        "task_id": "sync-task-1",
        "completed_batch_count": 2,
        "credited_sample_count": 13,
        "credited_target_count": 1,
        "parent_status_updated": True,
        "uses_training_targets": True,
    }

    def complete(*_args, **_kwargs):
        events.append("transaction_committed")
        return completion

    def get_task(_task_id):
        events.append("task_loaded")
        return {"task_id": "sync-task-1", "status": SyncStatus.IDLE}

    monkeypatch.setattr(
        external_sync_service,
        "complete_generation_and_consume_batches",
        complete,
        raising=False,
    )
    monkeypatch.setattr(external_sync_service, "get_task_raw", get_task)
    monkeypatch.setattr(
        external_sync_service,
        "get_generation_by_task_id",
        lambda *_args, **_kwargs: pytest.fail(
            "completion callback must not use the old multi-transaction path"
        ),
    )
    monkeypatch.setattr(
        level2_handler,
        "_schedule_next_training",
        lambda task_id: events.append(f"scheduled:{task_id}"),
    )

    result = level2_handler.on_generation_completed(
        "generation-1",
        output_dataset_id="dataset-1",
        output_sample_count=13,
    )

    assert result == completion
    assert events == [
        "transaction_committed",
        "task_loaded",
        "scheduled:sync-task-1",
    ]


def test_level2_generation_completion_retry_does_not_reschedule(monkeypatch):
    completion = {
        "tracking_found": True,
        "completed": False,
        "already_completed": True,
        "task_id": "sync-task-1",
        "completed_batch_count": 0,
        "credited_sample_count": 0,
        "credited_target_count": 0,
        "parent_status_updated": False,
        "uses_training_targets": False,
    }
    monkeypatch.setattr(
        external_sync_service,
        "complete_generation_and_consume_batches",
        lambda *_args, **_kwargs: completion,
        raising=False,
    )
    monkeypatch.setattr(
        level2_handler,
        "_schedule_next_training",
        lambda *_args, **_kwargs: pytest.fail("retry rescheduled training"),
    )
    monkeypatch.setattr(
        level2_handler,
        "_trigger_training",
        lambda *_args, **_kwargs: pytest.fail("retry retriggered training"),
    )

    result = level2_handler.on_generation_completed(
        "generation-1",
        output_dataset_id="dataset-1",
        output_sample_count=13,
    )

    assert result == completion


def test_startup_generation_cleanup_recovers_sync_tracking(monkeypatch):
    recovered = []
    claims = []
    finalized = []
    run_tokens = {
        "generation-1": "generation-run-token-1",
        "generation-2": "generation-run-token-2",
    }
    task_statuses = {
        "generation-1": "pending",
        "generation-2": "running",
    }
    statuses = {
        "pending": [{"task_id": "generation-1", "status": "pending"}],
        "running": [{"task_id": "generation-2", "status": "running"}],
        "stopping": [],
        "publishing": [],
        "recovering": [],
        "restarting": [],
    }
    queried_statuses = []

    def get_task_raw(task_id):
        return {
            "task_id": task_id,
            "status": task_statuses[task_id],
            "run_token": run_tokens[task_id],
        }

    def claim_orphan_recovery(
        task_id,
        *,
        expected_status,
        expected_run_token,
    ):
        assert task_statuses[task_id] == expected_status
        assert run_tokens[task_id] == expected_run_token
        claims.append((task_id, expected_status, expected_run_token))
        task_statuses[task_id] = "recovering"
        return {
            "run_token": expected_run_token,
            "source_status": expected_status,
        }

    def finish_orphan_recovery(
        task_id,
        *,
        expected_run_token,
        terminal_status,
        **_kwargs,
    ):
        assert task_statuses[task_id] == "recovering"
        assert run_tokens[task_id] == expected_run_token
        finalized.append((task_id, terminal_status, expected_run_token))
        task_statuses[task_id] = terminal_status
        return True

    def get_all_tasks(*, status, **_kwargs):
        queried_statuses.append(status)
        return list(statuses[status]), 1

    generation_service = SimpleNamespace(
        get_all_tasks=get_all_tasks,
        get_task_raw=get_task_raw,
        claim_orphan_recovery=claim_orphan_recovery,
        finish_orphan_recovery=finish_orphan_recovery,
    )
    generation_service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    monkeypatch.setattr(
        generation_service_module,
        "generation_task_service",
        generation_service,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_generation_and_restore_batches",
        lambda task_id, reason: recovered.append((task_id, reason)) or {
            "recovered": True,
            "restored_record_count": 1,
        },
        raising=False,
    )

    server.cleanup_orphan_generation_tasks()

    assert queried_statuses == [
        "pending",
        "running",
        "stopping",
        "publishing",
        "recovering",
        "restarting",
    ]

    assert [task_id for task_id, _reason in recovered] == [
        "generation-1",
        "generation-2",
    ]
    assert all("server restart" in reason for _task_id, reason in recovered)
    assert claims == [
        (
            "generation-1",
            "pending",
            "generation-run-token-1",
        ),
        (
            "generation-2",
            "running",
            "generation-run-token-2",
        ),
    ]
    assert finalized == [
        (
            "generation-1",
            "stopped",
            "generation-run-token-1",
        ),
        (
            "generation-2",
            "stopped",
            "generation-run-token-2",
        ),
    ]


def test_startup_sync_cleanup_snapshots_every_page_before_mutation(monkeypatch):
    tasks = [
        {"task_id": f"sync-{index}", "status": SyncStatus.SYNCING}
        for index in range(1001)
    ]
    page_offsets = []
    updated = []

    def list_tasks(*, limit, offset, **_kwargs):
        page_offsets.append(offset)
        return tasks[offset : offset + limit], len(tasks)

    monkeypatch.setattr(external_sync_service, "list_tasks", list_tasks)
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda task_id, **_kwargs: updated.append(task_id) or {"task_id": task_id},
    )

    server.cleanup_orphan_sync_tasks()

    assert page_offsets == [0, 1000]
    assert updated == [task["task_id"] for task in tasks]


def test_startup_training_cleanup_pages_and_isolates_task_failures(monkeypatch):
    training_service_module = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )
    gpu_module = importlib.import_module(
        "train_factory.core.gpu_resource_manager"
    )
    pending_status = "pending"
    tasks = [
        {"task_id": f"training-{index}", "status": pending_status}
        for index in range(1001)
    ]
    page_calls = []
    status_attempts = []
    process_updates = []
    released = []

    def get_all_tasks(*, status, limit, offset):
        page_calls.append((status, offset))
        source = tasks if status == pending_status else []
        return source[offset : offset + limit], len(source)

    def update_task_status(task_id, *_args, **_kwargs):
        status_attempts.append(task_id)
        if task_id == "training-0":
            raise RuntimeError("database unavailable")
        if task_id == "training-1":
            return False
        return True

    training_service = SimpleNamespace(
        get_all_tasks=get_all_tasks,
        get_tasks_with_process_info=lambda **_kwargs: ([], 0),
        update_task_status=update_task_status,
        update_process_info=lambda task_id, **_kwargs: process_updates.append(task_id)
        or True,
    )
    gpu_manager = SimpleNamespace(
        release_gpus_for_task=lambda task_id: released.append(task_id) or True,
        cleanup_stale_allocations=lambda **_kwargs: 0,
    )
    monkeypatch.setattr(
        training_service_module,
        "training_task_service",
        training_service,
    )
    monkeypatch.setattr(gpu_module, "gpu_resource_manager", gpu_manager)

    server.cleanup_orphan_tasks()

    assert (pending_status, 0) in page_calls
    assert (pending_status, 1000) in page_calls
    assert status_attempts == [task["task_id"] for task in tasks]
    assert process_updates == [task["task_id"] for task in tasks[2:]]
    assert released == [task["task_id"] for task in tasks[2:]]


def test_startup_sync_training_recovery_uses_keyset_pages_and_isolates_failures(
    monkeypatch,
):
    training_service_module = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )
    trackings = [
        {
            "id": index,
            "training_task_id": f"training-{index}",
            "status": SyncTrainingStatus.PENDING,
        }
        for index in range(1, 1002)
    ]
    page_calls = []
    recovered = []

    def list_pending_trainings(*, limit, after_id=None):
        page_calls.append(after_id)
        rows = [
            tracking
            for tracking in trackings
            if after_id is None or tracking["id"] > after_id
        ][:limit]
        return rows, len(rows)

    def recover(task_id, reason):
        if task_id == "training-1":
            raise RuntimeError("temporary database failure")
        recovered.append((task_id, reason))
        return {"tracking_found": True, "recovered": True}

    monkeypatch.setattr(
        external_sync_service,
        "list_pending_trainings",
        list_pending_trainings,
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_training_and_restore_claim",
        recover,
    )
    monkeypatch.setattr(
        training_service_module,
        "training_task_service",
        SimpleNamespace(
            get_task=lambda task_id: {
                "task_id": task_id,
                "status": "failed",
            }
        ),
    )

    server.cleanup_orphan_sync_trainings()

    assert page_calls == [None, 1000, 1001]
    assert [task_id for task_id, _reason in recovered] == [
        f"training-{index}" for index in range(2, 1002)
    ]


def test_startup_sync_training_recovery_handles_terminal_and_lost_callbacks(
    monkeypatch,
):
    training_service_module = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )
    post_training_module = importlib.import_module(
        "train_factory.sync.post_training_handler"
    )
    trackings = [
        {"id": index, "training_task_id": task_id}
        for index, task_id in enumerate(
            (
                "missing",
                "failed",
                "stopped",
                "cancelled",
                "crashed",
                "succeeded",
                "succeeded-no-output",
            ),
            start=1,
        )
    ]
    tasks = {
        "failed": {"task_id": "failed", "status": "failed"},
        "stopped": {"task_id": "stopped", "status": "stopped"},
        "cancelled": {"task_id": "cancelled", "status": "cancelled"},
        "crashed": {"task_id": "crashed", "status": "running"},
        "succeeded": {
            "task_id": "succeeded",
            "status": "succeeded",
            "final_model_path": "/models/succeeded",
            "trained_model_registry_id": "model-1",
        },
        "succeeded-no-output": {
            "task_id": "succeeded-no-output",
            "status": "succeeded",
            "final_model_path": None,
        },
    }
    failed = []
    completed = []

    monkeypatch.setattr(
        external_sync_service,
        "list_pending_trainings",
        lambda **kwargs: (
            trackings if kwargs.get("after_id") is None else [],
            len(trackings),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_training_and_restore_claim",
        lambda task_id, reason: failed.append((task_id, reason))
        or {"recovered": True},
    )
    monkeypatch.setattr(
        training_service_module,
        "training_task_service",
        SimpleNamespace(get_task=lambda task_id: tasks.get(task_id)),
    )
    monkeypatch.setattr(
        post_training_module,
        "on_training_completed",
        lambda **kwargs: completed.append(kwargs),
    )

    server.cleanup_orphan_sync_trainings()

    assert [task_id for task_id, _reason in failed] == [
        "missing",
        "failed",
        "stopped",
        "cancelled",
        "crashed",
        "succeeded-no-output",
    ]
    assert completed == [
        {
            "training_task_id": "succeeded",
            "final_model_path": "/models/succeeded",
            "model_registry_id": "model-1",
        }
    ]
