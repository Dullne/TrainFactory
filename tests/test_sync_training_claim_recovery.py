"""Durable claim and idempotent recovery tests for sync-triggered training."""

from __future__ import annotations

import pytest
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
