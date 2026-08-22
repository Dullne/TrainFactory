import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import dataset_routes
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.services.dataset_service import DatasetService


DATASET_ID = "preview-dataset"
USER = {"user_id": "user-1", "username": "user", "role": "user"}


def _dataset_engine(
    *,
    status: str = "registered",
    sample_data=None,
    num_rows=None,
):
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
                dataset_name="preview",
                user_id=USER["user_id"],
                status=status,
                storage_backend="local",
                storage_path="/datasets/original.jsonl",
                storage_uri=None,
                file_format="jsonl",
                version=1,
                sample_data=sample_data,
                num_rows=num_rows,
            )
        )
        session.commit()
    return engine


def _install_service(monkeypatch, engine):
    service = DatasetService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)
    monkeypatch.setattr(dataset_routes, "dataset_service", service)
    return service


def _read_dataset(engine):
    with Session(engine) as session:
        return session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()


@pytest.mark.parametrize(
    ("field", "winner"),
    (
        ("status", "downloading"),
        ("status", "deleting"),
        ("storage_backend", "s3"),
        ("storage_path", "/datasets/replaced.jsonl"),
        ("storage_uri", "s3://bucket/replaced.jsonl"),
        ("file_format", "parquet"),
        ("version", 2),
    ),
)
def test_preview_does_not_cache_or_promote_stale_snapshot(
    monkeypatch,
    field,
    winner,
):
    engine = _dataset_engine()
    _install_service(monkeypatch, engine)

    def load_preview(_reference, _file_format, _limit):
        with Session(engine) as session:
            dataset = session.exec(
                select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
            ).one()
            setattr(dataset, field, winner)
            session.add(dataset)
            session.commit()
        return {
            "columns": [{"name": "text", "type": "string"}],
            "rows": [{"text": "stale"}],
            "num_rows": 1,
        }

    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: SimpleNamespace(load_preview=load_preview),
    )

    result = asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 10, USER))

    assert result["rows"] == [{"text": "stale"}]
    persisted = _read_dataset(engine)
    assert getattr(persisted, field) == winner
    assert persisted.sample_data is None
    if field != "status":
        assert persisted.status == "registered"


def test_registered_preview_promotes_and_caches_only_stable_snapshot(monkeypatch):
    engine = _dataset_engine()
    service = _install_service(monkeypatch, engine)
    calls = []

    def load_preview(reference, file_format, limit):
        calls.append((reference, file_format, limit))
        return {
            "columns": [{"name": "text", "type": "string"}],
            "rows": [{"text": "one"}, {"text": "two"}],
            "num_rows": 2,
        }

    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: SimpleNamespace(load_preview=load_preview),
    )

    first = asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 2, USER))
    second = asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 2, USER))

    assert first["rows"] == second["rows"] == [
        {"text": "one"},
        {"text": "two"},
    ]
    assert calls == [("/datasets/original.jsonl", "jsonl", 2)]
    persisted = service.get_dataset(DATASET_ID)
    assert persisted["status"] == "ready"
    assert persisted["sample_data"] == first["rows"]


def test_ready_preview_atomically_expands_cache_and_reuses_it(monkeypatch):
    engine = _dataset_engine(status="ready", sample_data=[{"text": "one"}])
    service = _install_service(monkeypatch, engine)
    calls = []

    def load_preview(_reference, _file_format, limit):
        calls.append(limit)
        return {
            "columns": [{"name": "text", "type": "string"}],
            "rows": [
                {"text": "one"},
                {"text": "two"},
                {"text": "three"},
            ],
            "num_rows": 3,
        }

    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: SimpleNamespace(load_preview=load_preview),
    )

    expanded = asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 3, USER))
    cached = asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 3, USER))

    assert expanded["rows"] == cached["rows"]
    assert calls == [3]
    persisted = service.get_dataset(DATASET_ID)
    assert persisted["status"] == "ready"
    assert persisted["sample_data"] == expanded["rows"]


def test_complete_ready_cache_serves_limit_larger_than_dataset(monkeypatch):
    cached_rows = [{"text": "one"}, {"text": "two"}]
    engine = _dataset_engine(
        status="ready",
        sample_data=cached_rows,
        num_rows=len(cached_rows),
    )
    _install_service(monkeypatch, engine)
    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: pytest.fail("complete cache must not reload storage"),
    )

    result = asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 10, USER))

    assert result["rows"] == cached_rows
    assert result["total_rows"] == len(cached_rows)


def test_smaller_ready_preview_cannot_overwrite_larger_concurrent_cache(monkeypatch):
    engine = _dataset_engine(status="ready", sample_data=[{"text": "one"}])
    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        # MySQL DATETIME (without fsp) stores only whole seconds. Two cache
        # writes in the same second therefore have the same updated_at value.
        dataset.updated_at = dataset.updated_at.replace(microsecond=0)
        session.add(dataset)
        session.commit()
    service = _install_service(monkeypatch, engine)
    calls = []
    larger_cache = [
        {"text": "one"},
        {"text": "two"},
        {"text": "three"},
    ]

    def load_preview(_reference, _file_format, limit):
        calls.append(limit)
        with Session(engine) as session:
            dataset = session.exec(
                select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
            ).one()
            dataset.sample_data = larger_cache
            session.add(dataset)
            session.commit()
        return {
            "columns": [{"name": "text", "type": "string"}],
            "rows": larger_cache[:2],
            "num_rows": 3,
        }

    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: SimpleNamespace(load_preview=load_preview),
    )

    asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 2, USER))
    cached = asyncio.run(dataset_routes.preview_dataset(DATASET_ID, 3, USER))

    assert cached["rows"] == larger_cache
    assert calls == [2]
    assert service.get_dataset(DATASET_ID)["sample_data"] == larger_cache
