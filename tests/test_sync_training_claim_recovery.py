"""Durable claim and idempotent recovery tests for sync-triggered training."""

from __future__ import annotations

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import mysql
from sqlmodel import Session, create_engine, select

from train_factory.enums.sync_status import (
    SyncStatus,
    SyncTrainingStatus,
    TrainingTargetStatus,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.services.external_sync_service import ExternalSyncService
from train_factory.sync import post_training_handler


def _service():
    engine = create_engine("sqlite://")
    for table in (
        ExternalSyncTaskDB.__table__,
        ExternalSyncTrainingDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
    ):
        table.create(engine)
    service = ExternalSyncService()
    service.engine = engine
    return service, engine


def test_target_training_claim_is_persisted_and_failure_restores_exact_samples():
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-target",
                task_name="target",
                user_id="user-1",
                status=SyncStatus.IDLE,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="target-1",
                task_id="sync-target",
                target_name="embedding",
                status=TrainingTargetStatus.READY,
                pending_training_samples=7,
                current_training_id="training-old",
            )
        )
        session.commit()

    claimed = service.create_training_with_claim(
        task_id="sync-target",
        training_task_id="training-new",
        user_id="user-1",
        input_dataset_ids=["dataset-1"],
        total_samples=7,
        training_round=2,
        target_id="target-1",
        require_threshold=False,
    )

    assert claimed["claimed_sample_count"] == 7
    assert claimed["previous_target_status"] == TrainingTargetStatus.READY
    with Session(engine) as session:
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        target.pending_training_samples += 3
        session.add(target)
        session.commit()

    recovered = service.fail_training_and_restore_claim(
        "training-new",
        "worker failed",
    )
    repeated = service.fail_training_and_restore_claim(
        "training-new",
        "worker failed again",
    )

    assert recovered["recovered"] is True
    assert recovered["restored_sample_count"] == 7
    assert repeated["already_recovered"] is True
    with Session(engine) as session:
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        parent = session.exec(select(ExternalSyncTaskDB)).one()
        tracking = session.exec(select(ExternalSyncTrainingDB)).one()
    assert target.pending_training_samples == 10
    assert target.status == TrainingTargetStatus.READY
    assert target.current_training_id == "training-old"
    assert parent.status == SyncStatus.ERROR
    assert tracking.status == SyncTrainingStatus.FAILED
    assert tracking.claim_reconciled is True


def test_legacy_training_failure_restores_parent_claim_once():
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-legacy",
                task_name="legacy",
                user_id="user-1",
                status=SyncStatus.IDLE,
                pending_training_samples=9,
                current_training_id="legacy-old",
            )
        )
        session.commit()

    service.create_training_with_claim(
        task_id="sync-legacy",
        training_task_id="legacy-new",
        user_id="user-1",
        input_dataset_ids=["dataset-1"],
        total_samples=9,
        training_round=2,
    )
    with Session(engine) as session:
        parent = session.exec(select(ExternalSyncTaskDB)).one()
        parent.pending_training_samples += 4
        session.add(parent)
        session.commit()

    first = service.fail_training_and_restore_claim("legacy-new", "stopped")
    second = service.fail_training_and_restore_claim("legacy-new", "stopped")

    assert first["restored_sample_count"] == 9
    assert second["already_recovered"] is True
    with Session(engine) as session:
        parent = session.exec(select(ExternalSyncTaskDB)).one()
    assert parent.pending_training_samples == 13
    assert parent.current_training_id == "legacy-old"
    assert parent.status == SyncStatus.ERROR


def test_successful_training_consumes_claim_and_cannot_restore_it_later():
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-success",
                task_name="success",
                user_id="user-1",
                status=SyncStatus.IDLE,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="target-success",
                task_id="sync-success",
                target_name="embedding",
                status=TrainingTargetStatus.READY,
                pending_training_samples=5,
            )
        )
        session.commit()

    service.create_training_with_claim(
        task_id="sync-success",
        training_task_id="training-success",
        user_id="user-1",
        input_dataset_ids=["dataset-1"],
        total_samples=5,
        training_round=1,
        target_id="target-success",
        require_threshold=False,
    )
    completed = service.complete_training_claim("training-success")
    failed = service.fail_training_and_restore_claim("training-success", "late failure")

    assert completed["completed"] is True
    assert failed["recovered"] is False
    with Session(engine) as session:
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        tracking = session.exec(select(ExternalSyncTrainingDB)).one()
    assert target.pending_training_samples == 0
    assert target.status == TrainingTargetStatus.LOADING_ADAPTER
    assert tracking.status == SyncTrainingStatus.COMPLETED


def test_failed_training_claim_does_not_cancel_cascade_deletion():
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-deleting-failure",
                task_name="deleting",
                user_id="user-1",
                status=SyncStatus.IDLE,
                pending_training_samples=4,
            )
        )
        session.commit()

    service.create_training_with_claim(
        task_id="sync-deleting-failure",
        training_task_id="training-deleting-failure",
        user_id="user-1",
        input_dataset_ids=["dataset-1"],
        total_samples=4,
        training_round=1,
    )
    with Session(engine) as session:
        parent = session.exec(select(ExternalSyncTaskDB)).one()
        parent.status = SyncStatus.DELETING_CASCADE
        session.add(parent)
        session.commit()

    recovered = service.fail_training_and_restore_claim(
        "training-deleting-failure",
        "server restarted",
    )

    assert recovered["recovered"] is True
    assert recovered["deletion_pending"] is True
    with Session(engine) as session:
        parent = session.exec(select(ExternalSyncTaskDB)).one()
    assert parent.status == SyncStatus.DELETING_CASCADE


def test_completed_training_claim_does_not_leave_deletion_state():
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-deleting-success",
                task_name="deleting",
                user_id="user-1",
                status=SyncStatus.IDLE,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="target-deleting-success",
                task_id="sync-deleting-success",
                target_name="embedding",
                status=TrainingTargetStatus.READY,
                pending_training_samples=4,
            )
        )
        session.commit()

    service.create_training_with_claim(
        task_id="sync-deleting-success",
        training_task_id="training-deleting-success",
        user_id="user-1",
        input_dataset_ids=["dataset-1"],
        total_samples=4,
        training_round=1,
        target_id="target-deleting-success",
        require_threshold=False,
    )
    with Session(engine) as session:
        parent = session.exec(select(ExternalSyncTaskDB)).one()
        parent.status = SyncStatus.DELETING_CASCADE
        session.add(parent)
        session.commit()

    completed = service.complete_training_claim("training-deleting-success")

    assert completed["completed"] is True
    assert completed["deletion_pending"] is True
    with Session(engine) as session:
        parent = session.exec(select(ExternalSyncTaskDB)).one()
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
    assert parent.status == SyncStatus.DELETING_CASCADE
    assert target.status == TrainingTargetStatus.TRAINING


def test_completion_callback_skips_adapter_load_for_deleting_parent(monkeypatch):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _task_id: {
            "task_id": "sync-deleting-callback",
            "target_id": "target-1",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "complete_training_claim",
        lambda _task_id: {
            "completed": True,
            "deletion_pending": True,
        },
    )
    monkeypatch.setattr(
        post_training_handler,
        "load_adapter_for_training",
        lambda **_kwargs: pytest.fail(
            "adapter loading must not start while the sync task is deleting"
        ),
    )

    result = post_training_handler.on_training_completed(
        training_task_id="training-deleting-callback",
        final_model_path="/models/adapter",
    )

    assert result["deletion_pending"] is True


@pytest.mark.parametrize("operation", ["complete", "fail"])
def test_claim_reconciliation_uses_mysql_task_target_training_lock_order(
    operation: str,
):
    """SQLite executes the path; MySQL compilation proves the row-lock order."""
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-lock-order",
                task_name="lock order",
                user_id="user-1",
                status=SyncStatus.IDLE,
            )
        )
        session.add_all(
            [
                ExternalSyncTrainingTargetDB(
                    target_id="target-b",
                    task_id="sync-lock-order",
                    target_name="b",
                    status=TrainingTargetStatus.READY,
                    pending_training_samples=3,
                ),
                ExternalSyncTrainingTargetDB(
                    target_id="target-a",
                    task_id="sync-lock-order",
                    target_name="a",
                    status=TrainingTargetStatus.READY,
                ),
            ]
        )
        session.commit()

    service.create_training_with_claim(
        task_id="sync-lock-order",
        training_task_id="training-lock-order",
        user_id="user-1",
        input_dataset_ids=["dataset-1"],
        total_samples=3,
        training_round=1,
        target_id="target-b",
        require_threshold=False,
    )

    locked_sql: list[str] = []

    def record_mysql_lock(execute_state):
        statement = execute_state.statement
        if getattr(statement, "_for_update_arg", None) is None:
            return
        locked_sql.append(
            " ".join(
                str(
                    statement.compile(
                        dialect=mysql.dialect(),
                        compile_kwargs={"literal_binds": True},
                    )
                ).split()
            )
        )

    event.listen(Session, "do_orm_execute", record_mysql_lock)
    try:
        if operation == "complete":
            service.complete_training_claim("training-lock-order")
        else:
            service.fail_training_and_restore_claim(
                "training-lock-order",
                "worker failed",
            )
    finally:
        event.remove(Session, "do_orm_execute", record_mysql_lock)

    lock_tables = [
        next(
            table
            for table in (
                "external_sync_tasks",
                "external_sync_training_targets",
                "external_sync_trainings",
            )
            if f"FROM {table}" in sql
        )
        for sql in locked_sql
    ]
    assert lock_tables == [
        "external_sync_tasks",
        "external_sync_training_targets",
        "external_sync_trainings",
    ]
    assert "ORDER BY external_sync_training_targets.target_id" in locked_sql[1]
    assert "ORDER BY external_sync_trainings.training_task_id" in locked_sql[2]
    assert all(sql.endswith("FOR UPDATE") for sql in locked_sql)


@pytest.mark.parametrize("operation", ["complete", "fail"])
@pytest.mark.parametrize("drift", ["task", "target"])
def test_claim_reconciliation_fails_closed_when_claim_owner_drifted(
    operation: str,
    drift: str,
):
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-owner-drift",
                task_name="owner drift",
                user_id="user-1",
                status=SyncStatus.IDLE,
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="target-owner-drift",
                task_id="sync-owner-drift",
                target_name="owner drift",
                status=TrainingTargetStatus.READY,
                pending_training_samples=5,
            )
        )
        session.commit()

    service.create_training_with_claim(
        task_id="sync-owner-drift",
        training_task_id="training-owner-drift",
        user_id="user-1",
        input_dataset_ids=["dataset-1"],
        total_samples=5,
        training_round=1,
        target_id="target-owner-drift",
        require_threshold=False,
    )
    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        if drift == "task":
            task.current_training_id = "training-new-owner"
            session.add(task)
        else:
            target.current_training_id = "training-new-owner"
            session.add(target)
        session.commit()

    if operation == "complete":
        result = service.complete_training_claim("training-owner-drift")
        assert result["completed"] is False
    else:
        result = service.fail_training_and_restore_claim(
            "training-owner-drift",
            "stale callback",
        )
        assert result["recovered"] is False
    assert result["claim_owner_mismatch"] is True

    with Session(engine) as session:
        task = session.exec(select(ExternalSyncTaskDB)).one()
        target = session.exec(select(ExternalSyncTrainingTargetDB)).one()
        training = session.exec(select(ExternalSyncTrainingDB)).one()
    assert task.status == SyncStatus.TRAINING
    assert target.status == TrainingTargetStatus.TRAINING
    assert target.pending_training_samples == 0
    assert training.status == SyncTrainingStatus.PENDING
    assert training.claim_reconciled is False


@pytest.mark.parametrize("operation", ["task", "target"])
def test_non_binding_updates_do_not_lock_active_training_rows(operation: str):
    service, engine = _service()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-non-binding",
                task_name="before",
                user_id="user-1",
                status=SyncStatus.TRAINING,
                current_training_id="training-active",
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="target-non-binding",
                task_id="sync-non-binding",
                target_name="before",
                status=TrainingTargetStatus.TRAINING,
                current_training_id="training-active",
            )
        )
        session.add(
            ExternalSyncTrainingDB(
                task_id="sync-non-binding",
                training_task_id="training-active",
                user_id="user-1",
                target_id="target-non-binding",
                status=SyncTrainingStatus.PENDING,
            )
        )
        session.commit()

    locked_sql: list[str] = []

    def record_mysql_lock(execute_state):
        statement = execute_state.statement
        if getattr(statement, "_for_update_arg", None) is None:
            return
        locked_sql.append(
            " ".join(
                str(statement.compile(dialect=mysql.dialect())).split()
            )
        )

    event.listen(Session, "do_orm_execute", record_mysql_lock)
    try:
        if operation == "task":
            service.update_task("sync-non-binding", task_name="after")
        else:
            service.update_training_target(
                "target-non-binding",
                target_name="after",
            )
    finally:
        event.remove(Session, "do_orm_execute", record_mysql_lock)

    assert not any("FROM external_sync_trainings" in sql for sql in locked_sql)
