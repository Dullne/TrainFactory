"""
Unit tests for ExternalSyncService.recalculate_sample_counters().

Uses an in-memory SQLite database to validate counter recalculation
without requiring a live MySQL instance.
"""

import uuid
from datetime import timedelta
from train_factory.core.time_utils import now_naive

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncGenerationDB,
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
)
from train_factory.storage.services.external_sync_service import ExternalSyncService


def _create_test_engine():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


def _make_task(session, task_id, total=0, pending=0):
    task = ExternalSyncTaskDB(
        task_id=task_id,
        task_name="test",
        user_id="u1",
        total_training_samples=total,
        pending_training_samples=pending,
    )
    session.add(task)
    session.commit()
    return task


def _make_generation(session, task_id, sample_count, disabled=False, completed_at=None):
    gen = ExternalSyncGenerationDB(
        task_id=task_id,
        generation_task_id=str(uuid.uuid4()),
        user_id="u1",
        output_sample_count=sample_count,
        status="completed",
        disabled=disabled,
        completed_at=completed_at,
    )
    session.add(gen)
    session.commit()
    return gen


def _make_training(session, task_id, created_at=None):
    training = ExternalSyncTrainingDB(
        task_id=task_id,
        training_task_id=str(uuid.uuid4()),
        user_id="u1",
        total_samples=0,
        status="completed",
        created_at=created_at or now_naive(),
    )
    session.add(training)
    session.commit()
    return training


def test_recalculate_fixes_drifted_counters():
    """counter 偏移时，recalculate 应修正为实际值。"""
    engine = _create_test_engine()
    task_id = str(uuid.uuid4())

    with Session(engine) as session:
        _make_task(session, task_id, total=999, pending=888)
        _make_generation(session, task_id, sample_count=100)
        _make_generation(session, task_id, sample_count=200)
        _make_generation(session, task_id, sample_count=50, disabled=True)

    svc = ExternalSyncService()
    svc.engine = engine

    result = svc.recalculate_sample_counters(task_id)

    assert result["old_total_training_samples"] == 999
    assert result["new_total_training_samples"] == 300  # 100 + 200, disabled excluded
    assert result["old_pending_training_samples"] == 888
    assert result["new_pending_training_samples"] == 300  # no training → all pending


def test_recalculate_with_training_boundary():
    """有训练记录时，pending 只计算训练之后完成的生成。"""
    engine = _create_test_engine()
    task_id = str(uuid.uuid4())

    now = now_naive()
    before_training = now - timedelta(hours=2)
    after_training = now - timedelta(minutes=30)
    training_time = now - timedelta(hours=1)

    with Session(engine) as session:
        _make_task(session, task_id, total=0, pending=0)
        _make_generation(session, task_id, sample_count=100, completed_at=before_training)
        _make_generation(session, task_id, sample_count=200, completed_at=after_training)
        _make_training(session, task_id, created_at=training_time)

    svc = ExternalSyncService()
    svc.engine = engine

    result = svc.recalculate_sample_counters(task_id)

    assert result["new_total_training_samples"] == 300  # 100 + 200
    assert result["new_pending_training_samples"] == 200  # only after training


def test_recalculate_no_change():
    """counter 已正确时，返回值 old == new。"""
    engine = _create_test_engine()
    task_id = str(uuid.uuid4())

    with Session(engine) as session:
        _make_task(session, task_id, total=150, pending=150)
        _make_generation(session, task_id, sample_count=150)

    svc = ExternalSyncService()
    svc.engine = engine

    result = svc.recalculate_sample_counters(task_id)

    assert result["old_total_training_samples"] == 150
    assert result["new_total_training_samples"] == 150
    assert result["old_pending_training_samples"] == 150
    assert result["new_pending_training_samples"] == 150


def test_recalculate_empty_generations():
    """没有生成记录时，counter 归零。"""
    engine = _create_test_engine()
    task_id = str(uuid.uuid4())

    with Session(engine) as session:
        _make_task(session, task_id, total=500, pending=200)

    svc = ExternalSyncService()
    svc.engine = engine

    result = svc.recalculate_sample_counters(task_id)

    assert result["new_total_training_samples"] == 0
    assert result["new_pending_training_samples"] == 0


def test_recalculate_coalesce_null_completed_at():
    """completed_at 为 NULL 时，用 created_at 代替做时间比较。"""
    engine = _create_test_engine()
    task_id = str(uuid.uuid4())

    now = now_naive()
    training_time = now - timedelta(hours=1)

    with Session(engine) as session:
        _make_task(session, task_id, total=0, pending=0)
        # completed_at=None, 但 created_at 在训练之后（由 default_factory 生成）
        gen = ExternalSyncGenerationDB(
            task_id=task_id,
            generation_task_id=str(uuid.uuid4()),
            user_id="u1",
            output_sample_count=77,
            status="completed",
            disabled=False,
            completed_at=None,
            created_at=now,  # after training
        )
        session.add(gen)
        session.commit()
        _make_training(session, task_id, created_at=training_time)

    svc = ExternalSyncService()
    svc.engine = engine

    result = svc.recalculate_sample_counters(task_id)

    assert result["new_total_training_samples"] == 77
    # COALESCE(NULL, created_at) > training_time → should be pending
    assert result["new_pending_training_samples"] == 77


def test_recalculate_nonexistent_task():
    """不存在的 task_id 返回空 dict。"""
    engine = _create_test_engine()

    svc = ExternalSyncService()
    svc.engine = engine

    result = svc.recalculate_sample_counters("nonexistent-id")

    assert result == {}
