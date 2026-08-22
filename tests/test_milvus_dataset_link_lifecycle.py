from contextlib import contextmanager
import importlib

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.milvus_collection_entity import (
    CollectionDatasetLinkDB,
    MilvusCollectionDB,
)
from train_factory.storage.services.dataset_service import (
    DatasetDeletionInProgressError,
    DatasetReferenceUnavailableError,
)


service_module = importlib.import_module(
    "train_factory.storage.services.milvus_collection_service"
)

DATASET_ID = "dataset-1"


def _configure_service(monkeypatch, *, dataset_status=None):
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
        ],
    )
    if dataset_status is not None:
        with Session(engine) as session:
            session.add(
                DatasetDB(
                    dataset_id=DATASET_ID,
                    dataset_name="dataset",
                    user_id="user-1",
                    status=dataset_status,
                )
            )
            session.add(
                MilvusCollectionDB(
                    collection_name="collection-1",
                    display_name="collection-1",
                    user_id="user-1",
                )
            )
            session.commit()

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(service_module, "get_session", test_session)
    return service_module.MilvusCollectionService(), engine


def test_link_dataset_rejects_deleting_dataset(monkeypatch):
    service, engine = _configure_service(monkeypatch, dataset_status="deleting")

    with pytest.raises(DatasetDeletionInProgressError):
        service.link_dataset("collection-1", DATASET_ID)

    with Session(engine) as session:
        assert session.exec(select(CollectionDatasetLinkDB)).all() == []


def test_link_dataset_rejects_missing_dataset(monkeypatch):
    service, engine = _configure_service(monkeypatch)

    with pytest.raises(DatasetReferenceUnavailableError):
        service.link_dataset("collection-1", DATASET_ID)

    with Session(engine) as session:
        assert session.exec(select(CollectionDatasetLinkDB)).all() == []


def test_link_dataset_accepts_available_dataset(monkeypatch):
    service, engine = _configure_service(monkeypatch, dataset_status="ready")

    result = service.link_dataset(
        "collection-1",
        DATASET_ID,
        dataset_name="dataset",
    )

    assert result["dataset_id"] == DATASET_ID
    with Session(engine) as session:
        links = session.exec(select(CollectionDatasetLinkDB)).all()
        assert [(link.collection_name, link.dataset_id) for link in links] == [
            ("collection-1", DATASET_ID)
        ]


def test_list_dataset_links_includes_orphaned_collection_registry_link(monkeypatch):
    service, engine = _configure_service(monkeypatch, dataset_status="ready")
    with Session(engine) as session:
        session.add(
            CollectionDatasetLinkDB(
                collection_name="unregistered-collection",
                dataset_id=DATASET_ID,
            )
        )
        session.commit()

    links = service.list_dataset_links([DATASET_ID])

    assert [(link["collection_name"], link["dataset_id"]) for link in links] == [
        ("unregistered-collection", DATASET_ID)
    ]
