"""Regression tests for online DeepEval collection deletion races."""

from contextlib import contextmanager
import importlib

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.external_sync_entity import ExternalSyncTaskDB
from train_factory.storage.entities.milvus_collection_entity import (
    CollectionDatasetLinkDB,
    MilvusCollectionDB,
)


deep_module = importlib.import_module(
    "train_factory.storage.services.deep_evaluation_task_service"
)
milvus_module = importlib.import_module(
    "train_factory.storage.services.milvus_collection_service"
)


def _configure_services(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[
            EvaluationTaskDB.__table__,
            MilvusCollectionDB.__table__,
            CollectionDatasetLinkDB.__table__,
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

    monkeypatch.setattr(deep_module, "get_session", test_session)
    monkeypatch.setattr(milvus_module, "get_session", test_session)
    return (
        deep_module.DeepEvaluationTaskService(),
        milvus_module.MilvusCollectionService(),
    )


def _online_worker_groups(collection_name):
    return {
        "retrieval_mode": "online",
        "milvus_collection": collection_name,
    }


def test_online_deep_evaluation_is_an_atomic_collection_consumer(monkeypatch):
    evaluations, collections = _configure_services(monkeypatch)
    collections.register_collection("collection-1", user_id="user-1")
    task = evaluations.create_task(
        task_name="online-evaluation",
        worker_groups=_online_worker_groups("collection-1"),
        user_id="user-1",
    )

    assert evaluations.list_active_collection_consumers(["collection-1"]) == [
        task["task_id"]
    ]

    assert evaluations.cancel_task(task["task_id"])
    collections.acquire_deletion_fences(
        ["collection-1"],
        deletion_owner="manual:collection-1",
        user_id="user-1",
    )

    with pytest.raises(milvus_module.MilvusCollectionUnavailableError):
        evaluations.create_task(
            task_name="blocked-evaluation",
            worker_groups=_online_worker_groups("collection-1"),
            user_id="user-1",
        )
    assert not evaluations.reset_for_resume(task["task_id"])
