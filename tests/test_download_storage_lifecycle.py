"""Regression tests for remote-download storage lifecycle guarantees."""

import asyncio
import importlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from train_factory.api.routes import dataset_routes
from train_factory.api import server
from train_factory.core import remote_download_security
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import (
    ModelRegistryDB,
    ModelVersionDB,
)

dataset_download_module = importlib.import_module(
    "train_factory.storage.services.dataset_download_service"
)
model_download_module = importlib.import_module(
    "train_factory.storage.services.model_download_service"
)
model_registry_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)


class _RecordingLease:
    def __init__(self):
        self.release_calls = 0

    def release(self):
        self.release_calls += 1


class _SingleLeaseLimiter:
    def __init__(self, lease):
        self.lease = lease

    def acquire(self, *_args, **_kwargs):
        return self.lease


class _RecordingStorageReservation:
    def __init__(self, reserved_bytes=1):
        self.reserved_bytes = reserved_bytes
        self.release_calls = 0
        self.actual_sizes = []

    def verify_actual_size(self, actual_bytes):
        self.actual_sizes.append(actual_bytes)
        self.reserved_bytes = actual_bytes

    def release(self):
        self.release_calls += 1


class _FailingThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start(self):
        raise RuntimeError("thread start failed")


class _WritingFailingThread:
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        self.thread_args = kwargs["args"]

    def start(self):
        artifact_path = Path(self.thread_args[3])
        (artifact_path / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("thread start failed")


@pytest.fixture()
def lifecycle_db(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        ModelRegistryDB.__table__,
        ModelVersionDB.__table__,
        DeploymentDB.__table__,
        ModelConfigDB.__table__,
        DatasetDB.__table__,
    ):
        table.create(engine)

    @contextmanager
    def isolated_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(model_registry_module, "get_session", isolated_session)
    monkeypatch.setattr(model_download_module, "get_session", isolated_session)
    monkeypatch.setattr(dataset_download_module, "get_session", isolated_session)
    monkeypatch.setattr(_quota_module(), "get_session", isolated_session)
    monkeypatch.setattr(model_download_module.settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(
        dataset_download_module.settings,
        "datasets_dir",
        tmp_path / "datasets",
    )
    yield isolated_session, tmp_path
    engine.dispose()


def test_deleting_downloaded_model_removes_exact_managed_directory(lifecycle_db):
    isolated_session, tmp_path = lifecycle_db
    model_id = "model-download-1"
    model_path = tmp_path / "models" / model_id
    model_path.mkdir(parents=True)
    (model_path / "weights.bin").write_bytes(b"weights")
    with isolated_session() as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name="downloaded-model",
                model_type="embedding",
                model_path=str(model_path),
                source_type="downloaded",
                download_status="completed",
            )
        )

    assert model_registry_module.model_registry_service.delete_model(model_id) is True

    assert not model_path.exists()
    with isolated_session() as session:
        assert session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).first() is None


def test_model_storage_cleanup_failure_keeps_database_retry_handle(
    lifecycle_db,
    monkeypatch,
):
    isolated_session, tmp_path = lifecycle_db
    model_id = "model-download-2"
    model_path = tmp_path / "models" / model_id
    model_path.mkdir(parents=True)
    (model_path / "weights.bin").write_bytes(b"weights")
    with isolated_session() as session:
        session.add(
            ModelRegistryDB(
                model_id=model_id,
                model_name="retryable-model",
                model_type="embedding",
                model_path=str(model_path),
                source_type="downloaded",
            )
        )

    monkeypatch.setattr(
        model_registry_module,
        "shutil",
        SimpleNamespace(
            rmtree=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("disk busy")
            )
        ),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="storage cleanup failed"):
        model_registry_module.model_registry_service.delete_model(model_id)

    with isolated_session() as session:
        assert session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).first() is not None


def test_model_dependency_rejection_precedes_storage_cleanup(lifecycle_db):
    isolated_session, tmp_path = lifecycle_db
    model_id = "model-with-config"
    model_path = tmp_path / "models" / model_id
    model_path.mkdir(parents=True)
    (model_path / "weights.bin").write_bytes(b"weights")
    with isolated_session() as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id=model_id,
                    model_name="model-with-config",
                    model_type="embedding",
                    model_path=str(model_path),
                    source_type="downloaded",
                ),
                ModelConfigDB(
                    config_id="config-1",
                    config_name="config",
                    registry_id=model_id,
                    model_type="embedding",
                    provider="local",
                    api_endpoint="http://127.0.0.1:9997",
                    model_name="model",
                ),
            ]
        )

    with pytest.raises(ValueError, match="config"):
        model_registry_module.model_registry_service.delete_model(model_id)

    assert model_path.exists()
    with isolated_session() as session:
        assert session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).first() is not None


def test_force_model_delete_stops_deployment_before_storage_cleanup(
    lifecycle_db,
    monkeypatch,
):
    isolated_session, tmp_path = lifecycle_db
    model_id = "model-with-running-deployment"
    model_path = tmp_path / "models" / model_id
    model_path.mkdir(parents=True)
    weights_path = model_path / "weights.bin"
    weights_path.write_bytes(b"weights")
    with isolated_session() as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id=model_id,
                    model_name="deployed-model",
                    model_type="embedding",
                    model_path=str(model_path),
                    source_type="downloaded",
                ),
                DeploymentDB(
                    deployment_id="deployment-1",
                    model_id=model_id,
                    deployment_name="deployed-model",
                    xinference_endpoint="http://127.0.0.1:9997",
                    deploy_mode="container",
                    container_name="trainfactory-xf-deployed-model-deadbeef",
                    status="running",
                ),
            ]
        )

    def fail_container_cleanup(*_args, **_kwargs):
        raise RuntimeError("container cleanup failed")

    monkeypatch.setattr(
        model_registry_module,
        "_remove_deployment_container",
        fail_container_cleanup,
    )

    with pytest.raises(RuntimeError, match="container cleanup failed"):
        model_registry_module.model_registry_service.delete_model(model_id, force=True)

    assert weights_path.read_bytes() == b"weights"
    with isolated_session() as session:
        assert session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).first() is not None
        assert session.exec(
            select(DeploymentDB).where(DeploymentDB.model_id == model_id)
        ).first() is not None


def test_force_model_delete_validates_storage_before_runtime_cleanup(
    lifecycle_db,
    monkeypatch,
):
    isolated_session, tmp_path = lifecycle_db
    model_id = "model-with-invalid-storage"
    outside_path = tmp_path / "outside-model"
    outside_path.mkdir()
    weights_path = outside_path / "weights.bin"
    weights_path.write_bytes(b"weights")
    with isolated_session() as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id=model_id,
                    model_name="invalid-storage-model",
                    model_type="embedding",
                    model_path=str(outside_path),
                    source_type="downloaded",
                ),
                DeploymentDB(
                    deployment_id="deployment-invalid-storage",
                    model_id=model_id,
                    deployment_name="invalid-storage-model",
                    xinference_endpoint="http://127.0.0.1:9997",
                    deploy_mode="shared",
                    model_uid="invalid-storage-model",
                    status="running",
                ),
            ]
        )

    cleanup_calls = []
    monkeypatch.setattr(
        model_registry_module,
        "_remove_deployment_container",
        lambda *_args, **_kwargs: cleanup_calls.append(True),
    )

    with pytest.raises(ValueError, match="managed directory"):
        model_registry_module.model_registry_service.delete_model(model_id, force=True)

    assert cleanup_calls == []
    assert weights_path.read_bytes() == b"weights"


def _owned_dataset():
    return {
        "dataset_id": "dataset-1",
        "dataset_name": "dataset",
        "user_id": "user-1",
        "status": "ready",
        "storage_backend": "local",
        "storage_path": "/managed/datasets/dataset-1",
        "storage_uri": None,
    }


def _allow_dataset_delete_until_cleanup(monkeypatch):
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "mark_deleting",
        lambda _dataset_id, *, user_id: user_id == "user-1",
    )
    monkeypatch.setattr(
        dataset_routes.dataset_asset_service,
        "list_assets",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        dataset_routes.generation_task_service,
        "list_active_dataset_consumers",
        lambda _dataset_ids: [],
    )
    monkeypatch.setattr(
        dataset_routes.training_task_service,
        "list_active_dataset_consumers",
        lambda _dataset_paths: [],
    )
    monkeypatch.setattr(
        dataset_routes.evaluation_task_service,
        "list_active_dataset_consumers",
        lambda _dataset_ids, _dataset_paths: [],
    )


def test_dataset_storage_cleanup_failure_keeps_database_record(monkeypatch):
    deleted_records = []
    _allow_dataset_delete_until_cleanup(monkeypatch)
    monkeypatch.setattr(dataset_routes.dataset_service, "get_dataset", lambda _id: _owned_dataset())
    monkeypatch.setattr(dataset_routes, "verify_resource_ownership", lambda value, *_args: value)
    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: SimpleNamespace(delete=lambda *_args, **_kwargs: False),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "delete_dataset",
        lambda dataset_id: deleted_records.append(dataset_id) or True,
    )
    monkeypatch.setattr(
        dataset_routes.dataset_asset_service,
        "delete_assets_for_dataset",
        lambda _id: True,
    )
    monkeypatch.setattr(
        dataset_routes.dataset_lineage_service,
        "delete_edges_for_dataset",
        lambda _id: True,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset("dataset-1", {"user_id": "user-1"}))

    assert exc_info.value.status_code == 500
    assert deleted_records == []


def test_dataset_metadata_cleanup_failure_keeps_database_record(monkeypatch):
    deleted_records = []
    _allow_dataset_delete_until_cleanup(monkeypatch)
    monkeypatch.setattr(dataset_routes.dataset_service, "get_dataset", lambda _id: _owned_dataset())
    monkeypatch.setattr(dataset_routes, "verify_resource_ownership", lambda value, *_args: value)
    monkeypatch.setattr(
        dataset_routes,
        "get_storage_backend",
        lambda _kind: SimpleNamespace(delete=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_asset_service,
        "delete_assets_for_dataset",
        lambda _id: (_ for _ in ()).throw(RuntimeError("metadata delete failed")),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "delete_dataset",
        lambda dataset_id: deleted_records.append(dataset_id) or True,
    )
    monkeypatch.setattr(
        dataset_routes.dataset_lineage_service,
        "delete_edges_for_dataset",
        lambda _id: True,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset("dataset-1", {"user_id": "user-1"}))

    assert exc_info.value.status_code == 500
    assert deleted_records == []


def test_dataset_asset_listing_failure_keeps_database_record(monkeypatch):
    _allow_dataset_delete_until_cleanup(monkeypatch)
    dataset = _owned_dataset()
    dataset.update(
        storage_backend="s3",
        storage_path=None,
        storage_uri="s3://bucket/dataset-1/data.jsonl",
    )
    deleted_records = []
    monkeypatch.setattr(dataset_routes.dataset_service, "get_dataset", lambda _id: dataset)
    monkeypatch.setattr(dataset_routes, "verify_resource_ownership", lambda value, *_args: value)
    monkeypatch.setattr(
        dataset_routes.dataset_asset_service,
        "list_assets",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("list failed")),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "delete_dataset",
        lambda dataset_id: deleted_records.append(dataset_id) or True,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dataset_routes.delete_dataset("dataset-1", {"user_id": "user-1"}))

    assert exc_info.value.status_code == 500
    assert deleted_records == []


@pytest.mark.parametrize(
    ("module", "service_factory", "start_kwargs", "entity", "id_field", "fixed_id"),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService,
            {
                "download_source": "huggingface",
                "remote_repo": "owner/model",
                "user_id": "user-1",
            },
            ModelRegistryDB,
            ModelRegistryDB.model_id,
            "model-start-failure",
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService,
            {
                "source_type": "huggingface",
                "remote_repo": "owner/dataset",
                "dataset_name": "dataset",
                "user_id": "user-1",
            },
            DatasetDB,
            DatasetDB.dataset_id,
            "dataset-start-failure",
        ),
    ),
)
def test_thread_start_failure_removes_directory_pending_row_and_memory_state(
    lifecycle_db,
    monkeypatch,
    module,
    service_factory,
    start_kwargs,
    entity,
    id_field,
    fixed_id,
):
    isolated_session, tmp_path = lifecycle_db
    lease = _RecordingLease()
    monkeypatch.setattr(module.settings, "auth_enabled", True)
    monkeypatch.setattr(module, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=_FailingThread))
    monkeypatch.setattr(module, "download_concurrency_limiter", _SingleLeaseLimiter(lease))
    monkeypatch.setattr(module, "preflight_remote_repo_size", lambda **_kwargs: 1)
    service = service_factory()

    with pytest.raises(RuntimeError, match="thread start failed"):
        service.start_download(**start_kwargs)

    root = tmp_path / ("models" if entity is ModelRegistryDB else "datasets")
    assert not (root / fixed_id).exists()
    assert fixed_id not in service._download_threads
    assert fixed_id not in service._download_progress
    assert lease.release_calls == 1
    with isolated_session() as session:
        assert session.exec(select(entity).where(id_field == fixed_id)).first() is None


@pytest.mark.parametrize(
    (
        "module",
        "service_factory",
        "start_kwargs",
        "limit_attribute",
        "fixed_id",
    ),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService,
            {
                "download_source": "modelscope",
                "remote_repo": "owner/model",
                "user_id": "user-1",
            },
            "model_download_max_bytes",
            "model-unknown-size",
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService,
            {
                "source_type": "modelscope",
                "remote_repo": "owner/dataset",
                "dataset_name": "unknown-size-dataset",
                "user_id": "user-1",
            },
            "dataset_download_max_bytes",
            "dataset-unknown-size",
        ),
    ),
)
def test_unknown_remote_size_reserves_full_download_limit(
    lifecycle_db,
    monkeypatch,
    module,
    service_factory,
    start_kwargs,
    limit_attribute,
    fixed_id,
):
    lease = _RecordingLease()
    reservation = _RecordingStorageReservation()
    reserve_calls = []

    monkeypatch.setattr(module.settings, "auth_enabled", False)
    monkeypatch.setattr(module, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=_FailingThread))
    monkeypatch.setattr(
        module,
        "download_concurrency_limiter",
        _SingleLeaseLimiter(lease),
    )
    monkeypatch.setattr(
        module,
        "preflight_remote_repo_size",
        lambda **_kwargs: None,
    )

    def reserve(_user_id, **kwargs):
        reserve_calls.append(kwargs)
        return reservation

    monkeypatch.setattr(
        module,
        "download_storage_quota_service",
        SimpleNamespace(reserve=reserve),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        service_factory().start_download(**start_kwargs)

    assert reserve_calls == [
        {
            "expected_bytes": getattr(module.settings, limit_attribute),
            "global_limit": module.settings.download_storage_max_bytes_global,
            "per_user_limit": module.settings.download_storage_max_bytes_per_user,
        }
    ]
    assert lease.release_calls == 1
    assert reservation.release_calls == 1


@pytest.mark.parametrize(
    ("module", "service_factory", "start_kwargs", "fixed_id", "root_name"),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService,
            {
                "download_source": "huggingface",
                "remote_repo": "owner/model",
                "user_id": "user-1",
            },
            "model-cleanup-failure",
            "models",
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService,
            {
                "source_type": "huggingface",
                "remote_repo": "owner/dataset",
                "dataset_name": "dataset-cleanup-failure",
                "user_id": "user-1",
            },
            "dataset-cleanup-failure",
            "datasets",
        ),
    ),
)
def test_start_rollback_releases_leases_when_pending_row_cleanup_fails(
    lifecycle_db,
    monkeypatch,
    module,
    service_factory,
    start_kwargs,
    fixed_id,
    root_name,
):
    _isolated_session, tmp_path = lifecycle_db
    lease = _RecordingLease()
    reservation = _RecordingStorageReservation()
    monkeypatch.setattr(module.settings, "auth_enabled", True)
    monkeypatch.setattr(module, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=_FailingThread))
    monkeypatch.setattr(module, "download_concurrency_limiter", _SingleLeaseLimiter(lease))
    monkeypatch.setattr(module, "preflight_remote_repo_size", lambda **_kwargs: 1)
    monkeypatch.setattr(
        module,
        "download_storage_quota_service",
        SimpleNamespace(reserve=lambda *_args, **_kwargs: reservation),
    )
    service = service_factory()
    monkeypatch.setattr(
        service,
        "_delete_pending_record",
        lambda _id: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        service.start_download(**start_kwargs)

    assert not (tmp_path / root_name / fixed_id).exists()
    assert fixed_id not in service._download_threads
    assert fixed_id not in service._download_progress
    assert lease.release_calls == 1
    assert reservation.release_calls == 1


@pytest.mark.parametrize(
    ("module", "service_factory", "start_kwargs", "entity", "id_field", "fixed_id"),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService,
            {
                "download_source": "huggingface",
                "remote_repo": "owner/model",
                "user_id": "user-1",
            },
            ModelRegistryDB,
            ModelRegistryDB.model_id,
            "model-residual-start",
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService,
            {
                "source_type": "huggingface",
                "remote_repo": "owner/dataset",
                "dataset_name": "dataset-residual-start",
                "user_id": "user-1",
            },
            DatasetDB,
            DatasetDB.dataset_id,
            "dataset-residual-start",
        ),
    ),
)
def test_thread_start_cleanup_failure_keeps_billable_failed_row(
    lifecycle_db,
    monkeypatch,
    module,
    service_factory,
    start_kwargs,
    entity,
    id_field,
    fixed_id,
):
    isolated_session, tmp_path = lifecycle_db
    lease = _RecordingLease()
    reservation = _RecordingStorageReservation()
    monkeypatch.setattr(module.settings, "auth_enabled", True)
    monkeypatch.setattr(module, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(
        module,
        "threading",
        SimpleNamespace(Thread=_WritingFailingThread),
    )
    monkeypatch.setattr(module, "download_concurrency_limiter", _SingleLeaseLimiter(lease))
    monkeypatch.setattr(module, "preflight_remote_repo_size", lambda **_kwargs: 1)
    monkeypatch.setattr(
        module,
        "download_storage_quota_service",
        SimpleNamespace(reserve=lambda *_args, **_kwargs: reservation),
    )
    monkeypatch.setattr(
        remote_download_security.shutil,
        "rmtree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("in use")),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        service_factory().start_download(**start_kwargs)

    root = tmp_path / ("models" if entity is ModelRegistryDB else "datasets")
    assert (root / fixed_id / "partial.bin").read_bytes() == b"partial"
    with isolated_session() as session:
        record = session.exec(select(entity).where(id_field == fixed_id)).first()
        assert record is not None
        assert record.file_size == len(b"partial")
        if entity is ModelRegistryDB:
            assert record.download_status == "failed"
            assert "residual storage" in record.download_error
        else:
            assert record.status == "error"
            assert "residual storage" in record.error_message
    assert lease.release_calls == 1
    assert reservation.release_calls == 1


@pytest.mark.parametrize(
    (
        "module",
        "service_factory",
        "background_method",
        "download_method",
        "record_method",
        "record_status",
        "error_field",
    ),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService,
            "_download_model_background",
            "_download_from_huggingface",
            "_update_model_record",
            {"download_status": "failed"},
            "download_error",
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService,
            "_download_dataset_background",
            "_download_from_huggingface",
            "_update_dataset_record",
            {"status": "error"},
            "error_message",
        ),
    ),
)
def test_background_cleanup_failure_records_billable_residual_bytes(
    monkeypatch,
    tmp_path,
    module,
    service_factory,
    background_method,
    download_method,
    record_method,
    record_status,
    error_field,
):
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    (artifact_path / "partial.bin").write_bytes(b"partial")
    service = service_factory()
    updates = []
    lease = _RecordingLease()
    reservation = _RecordingStorageReservation(reserved_bytes=100)
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)
    monkeypatch.setattr(
        service,
        download_method,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )
    monkeypatch.setattr(
        service,
        record_method,
        lambda _id, **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        remote_download_security.shutil,
        "rmtree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("in use")),
    )

    getattr(service, background_method)(
        "artifact-1",
        "huggingface",
        "owner/artifact",
        str(artifact_path),
        lease=lease,
        storage_reservation=reservation,
        max_bytes=100,
    )

    assert updates[-1]["file_size"] == len(b"partial")
    assert updates[-1][error_field].startswith("download failed")
    assert "residual storage" in updates[-1][error_field]
    assert all(updates[-1][key] == value for key, value in record_status.items())
    assert lease.release_calls == 1
    assert reservation.release_calls == 1


@pytest.mark.parametrize(
    ("module", "service", "method_name"),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService(),
            "_calculate_model_size",
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService(),
            "_calculate_size",
        ),
    ),
)
def test_download_size_traversal_failure_is_fail_closed(
    monkeypatch,
    tmp_path,
    module,
    service,
    method_name,
):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "data.bin").write_bytes(b"data")

    def fail_closed(_path):
        raise remote_download_security.DownloadSizeVerificationFailed(
            "Unable to verify downloaded artifact size"
        )

    monkeypatch.setattr(
        module,
        "calculate_directory_size_strict",
        fail_closed,
        raising=False,
    )

    with pytest.raises(
        remote_download_security.DownloadSizeVerificationFailed,
        match="Unable to verify",
    ):
        getattr(service, method_name)(str(artifact))


def test_strict_size_calculator_rejects_symbolic_links(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    link = artifact / "linked.bin"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable on this host")

    with pytest.raises(
        remote_download_security.DownloadSizeVerificationFailed,
        match="symbolic link",
    ):
        remote_download_security.calculate_directory_size_strict(artifact)


@pytest.mark.parametrize(
    ("source_type", "output_format", "message"),
    (
        ("modelscope", "jsonl", "ModelScope"),
        ("huggingface", "parquet", "JSONL"),
    ),
)
def test_authenticated_dataset_download_rejects_unsafe_materialization_modes_before_admission(
    monkeypatch,
    source_type,
    output_format,
    message,
):
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr(
        dataset_download_module,
        "download_concurrency_limiter",
        SimpleNamespace(acquire=lambda *_args, **_kwargs: _unexpected_admission()),
    )

    with pytest.raises(
        remote_download_security.DownloadPolicyViolation,
        match=message,
    ):
        dataset_download_module.DatasetDownloadService().start_download(
            source_type=source_type,
            remote_repo="owner/dataset",
            dataset_name="dataset",
            output_format=output_format,
            user_id="user-1",
        )


def _unexpected_admission():
    pytest.fail("unsafe dataset mode reached admission or filesystem side effects")


class _StreamingSplit:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        yield from self._rows

    def __len__(self):
        pytest.fail("authenticated streaming download attempted to materialize a split")

    def to_json(self, *_args, **_kwargs):
        pytest.fail("authenticated streaming download used the materializing writer")


def test_authenticated_huggingface_dataset_streams_rows_to_jsonl(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "train": _StreamingSplit(
                [{"text": "first"}, {"text": "second"}]
            )
        }

    storage_path = tmp_path / "dataset"
    storage_path.mkdir()
    service = dataset_download_module.DatasetDownloadService()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    num_rows, file_size = service._download_from_huggingface(
        "dataset-1",
        "owner/dataset",
        str(storage_path),
        output_format="jsonl",
        clean_cache=False,
        max_bytes=1024,
    )

    assert captured["kwargs"]["streaming"] is True
    assert captured["kwargs"]["token"] is False
    assert captured["kwargs"]["trust_remote_code"] is False
    assert num_rows == 2
    assert file_size == (storage_path / "train.jsonl").stat().st_size
    assert (storage_path / "train.jsonl").read_text(encoding="utf-8").splitlines() == [
        '{"text":"first"}',
        '{"text":"second"}',
    ]


def test_authenticated_huggingface_dataset_stops_before_oversized_row_write(
    monkeypatch,
    tmp_path,
):
    storage_path = tmp_path / "dataset"
    storage_path.mkdir()
    service = dataset_download_module.DatasetDownloadService()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_args, **_kwargs: {
            "train": _StreamingSplit([{"text": "x" * 100}])
        },
    )
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    with pytest.raises(remote_download_security.DownloadSizeExceeded):
        service._download_from_huggingface(
            "dataset-1",
            "owner/dataset",
            str(storage_path),
            max_bytes=32,
        )

    output = storage_path / "train.jsonl"
    assert not output.exists() or output.stat().st_size == 0


def test_authenticated_huggingface_dataset_counts_buffered_rows_before_write(
    monkeypatch,
    tmp_path,
):
    rows = [{"text": "a" * 10}, {"text": "b" * 10}]
    first_line = b'{"text":"aaaaaaaaaa"}\n'
    storage_path = tmp_path / "dataset"
    storage_path.mkdir()
    service = dataset_download_module.DatasetDownloadService()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_args, **_kwargs: {"train": _StreamingSplit(rows)},
    )
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    with pytest.raises(remote_download_security.DownloadSizeExceeded):
        service._download_from_huggingface(
            "dataset-1",
            "owner/dataset",
            str(storage_path),
            max_bytes=len(first_line) + 1,
        )

    assert (storage_path / "train.jsonl").read_bytes() == first_line


def test_authenticated_huggingface_dataset_expands_quota_before_disk_write(
    monkeypatch,
    tmp_path,
):
    storage_path = tmp_path / "dataset"
    storage_path.mkdir()
    service = dataset_download_module.DatasetDownloadService()
    reservation = SimpleNamespace(
        reserved_bytes=5,
        verify_actual_size=lambda size: setattr(reservation, "reserved_bytes", size),
    )
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_args, **_kwargs: {
            "train": _StreamingSplit([{"text": "expanded"}])
        },
    )
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    service._download_from_huggingface(
        "dataset-1",
        "owner/dataset",
        str(storage_path),
        max_bytes=1024,
        storage_reservation=reservation,
    )

    assert reservation.reserved_bytes >= (storage_path / "train.jsonl").stat().st_size


def test_authenticated_huggingface_dataset_falls_back_to_exact_quota_growth(
    monkeypatch,
    tmp_path,
):
    storage_path = tmp_path / "dataset"
    storage_path.mkdir()
    service = dataset_download_module.DatasetDownloadService()

    class TightReservation:
        reserved_bytes = 5

        def __init__(self):
            self.attempted_sizes = []

        def verify_actual_size(self, size):
            self.attempted_sizes.append(size)
            if size > 64:
                raise remote_download_security.DownloadStorageQuotaExceeded(
                    "quota exceeded"
                )
            self.reserved_bytes = size

    reservation = TightReservation()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_args, **_kwargs: {
            "train": _StreamingSplit([{"text": "fits exactly"}])
        },
    )
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    service._download_from_huggingface(
        "dataset-1",
        "owner/dataset",
        str(storage_path),
        max_bytes=1024,
        storage_reservation=reservation,
    )

    actual_size = (storage_path / "train.jsonl").stat().st_size
    assert reservation.attempted_sizes == [1024, actual_size]
    assert reservation.reserved_bytes == actual_size


def _quota_module():
    return importlib.import_module(
        "train_factory.storage.services.download_storage_quota_service"
    )


def test_download_storage_per_user_quota_cannot_exceed_global_quota():
    from train_factory.config.settings import Settings

    with pytest.raises(ValueError, match="per-user storage quota"):
        Settings(
            download_storage_max_bytes_global=100,
            download_storage_max_bytes_per_user=101,
            _env_file=None,
        )


def test_storage_quota_reservations_close_overlapping_download_toctou(monkeypatch):
    quota_module = _quota_module()
    service = quota_module.DownloadStorageQuotaService()
    monkeypatch.setattr(
        service,
        "_stored_usage",
        lambda user_id: (60, 30 if user_id == "user-1" else 10),
    )

    first = service.reserve(
        "user-1",
        expected_bytes=30,
        global_limit=100,
        per_user_limit=60,
    )

    with pytest.raises(
        remote_download_security.DownloadStorageQuotaExceeded,
        match="global",
    ):
        service.reserve(
            "user-2",
            expected_bytes=11,
            global_limit=100,
            per_user_limit=60,
        )

    assert service.active_reserved_total == 30
    first.release()
    assert service.active_reserved_total == 0


def test_storage_quota_rechecks_larger_final_artifact(monkeypatch):
    quota_module = _quota_module()
    service = quota_module.DownloadStorageQuotaService()
    monkeypatch.setattr(service, "_stored_usage", lambda _user_id: (60, 40))
    reservation = service.reserve(
        "user-1",
        expected_bytes=30,
        global_limit=100,
        per_user_limit=100,
    )

    with pytest.raises(
        remote_download_security.DownloadStorageQuotaExceeded,
        match="global",
    ):
        reservation.verify_actual_size(45)

    assert reservation.reserved_bytes == 30
    reservation.release()


def test_storage_quota_sums_persisted_remote_models_and_datasets(
    lifecycle_db,
    monkeypatch,
):
    isolated_session, tmp_path = lifecycle_db
    quota_module = _quota_module()
    with isolated_session() as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id="remote-model",
                    model_name="remote-model",
                    model_type="embedding",
                    model_path=str(tmp_path / "models" / "remote-model"),
                    source_type="downloaded",
                    user_id="user-1",
                    file_size=40,
                ),
                ModelRegistryDB(
                    model_id="trained-model",
                    model_name="trained-model",
                    model_type="embedding",
                    model_path=str(tmp_path / "output" / "trained-model"),
                    source_type="trained",
                    user_id="user-1",
                    file_size=1000,
                ),
                DatasetDB(
                    dataset_id="remote-dataset",
                    dataset_name="remote-dataset",
                    source_type="huggingface",
                    user_id="user-1",
                    file_size=20,
                ),
                DatasetDB(
                    dataset_id="uploaded-dataset",
                    dataset_name="uploaded-dataset",
                    source_type="uploaded",
                    user_id="user-2",
                    file_size=1000,
                ),
            ]
        )

    monkeypatch.setattr(quota_module, "get_session", isolated_session)

    assert quota_module.DownloadStorageQuotaService()._stored_usage("user-1") == (
        60,
        60,
    )


@pytest.mark.parametrize(
    ("module", "service_factory", "start_kwargs"),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService,
            {
                "download_source": "huggingface",
                "remote_repo": "owner/model",
                "user_id": "user-1",
            },
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService,
            {
                "source_type": "huggingface",
                "remote_repo": "owner/dataset",
                "dataset_name": "dataset",
                "user_id": "user-1",
            },
        ),
    ),
)
def test_storage_quota_rejection_precedes_filesystem_side_effects(
    monkeypatch,
    module,
    service_factory,
    start_kwargs,
):
    lease = _RecordingLease()
    monkeypatch.setattr(module.settings, "auth_enabled", True)
    monkeypatch.setattr(module, "download_concurrency_limiter", _SingleLeaseLimiter(lease))
    monkeypatch.setattr(module, "preflight_remote_repo_size", lambda **_kwargs: 50)
    monkeypatch.setattr(
        module,
        "download_storage_quota_service",
        SimpleNamespace(
            reserve=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                remote_download_security.DownloadStorageQuotaExceeded(
                    "Remote download global storage quota exceeded"
                )
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(module.os, "makedirs", lambda *_args, **_kwargs: _unexpected_admission())

    with pytest.raises(remote_download_security.DownloadStorageQuotaExceeded):
        service_factory().start_download(**start_kwargs)

    assert lease.release_calls == 1


@pytest.mark.parametrize(
    ("route_module", "payload", "endpoint", "service"),
    (
        (
            dataset_routes,
            dataset_routes.DownloadDatasetRequest(
                dataset_name="dataset",
                source_type="huggingface",
                remote_repo="owner/dataset",
            ),
            dataset_routes.download_dataset,
            dataset_download_module.dataset_download_service,
        ),
        (
            __import__(
                "train_factory.api.routes.registry_routes",
                fromlist=["registry_routes"],
            ),
            __import__(
                "train_factory.api.routes.registry_routes",
                fromlist=["registry_routes"],
            ).DownloadModelRequest(
                download_source="huggingface",
                remote_repo="owner/model",
            ),
            __import__(
                "train_factory.api.routes.registry_routes",
                fromlist=["registry_routes"],
            ).download_model,
            model_download_module.model_download_service,
        ),
    ),
)
def test_download_routes_map_storage_quota_to_insufficient_storage(
    monkeypatch,
    route_module,
    payload,
    endpoint,
    service,
):
    monkeypatch.setattr(route_module, "requires_tenant_provenance", lambda _user: True)
    monkeypatch.setattr(
        service,
        "start_download",
        lambda **_kwargs: (_ for _ in ()).throw(
            remote_download_security.DownloadStorageQuotaExceeded(
                "Remote download per-user storage quota exceeded"
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(payload, {"user_id": "user-1"}))

    assert exc_info.value.status_code == 507


def test_restart_cleanup_fails_download_rows_and_removes_managed_residue(
    lifecycle_db,
):
    isolated_session, tmp_path = lifecycle_db
    model_id = "orphan-model"
    dataset_id = "orphan-dataset"
    model_path = tmp_path / "models" / model_id
    dataset_path = tmp_path / "datasets" / dataset_id
    model_path.mkdir(parents=True)
    dataset_path.mkdir(parents=True)
    (model_path / "partial.bin").write_bytes(b"partial")
    (dataset_path / "partial.jsonl").write_bytes(b"partial")
    with isolated_session() as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id=model_id,
                    model_name="orphan-model",
                    model_type="embedding",
                    model_path=str(model_path),
                    source_type="downloaded",
                    download_status="pending",
                    file_size=7,
                ),
                DatasetDB(
                    dataset_id=dataset_id,
                    dataset_name="orphan-dataset",
                    storage_path=str(dataset_path),
                    source_type="huggingface",
                    status="downloading",
                    file_size=7,
                ),
            ]
        )

    assert model_download_module.ModelDownloadService().cleanup_interrupted_downloads() == 1
    assert dataset_download_module.DatasetDownloadService().cleanup_interrupted_downloads() == 1

    assert not model_path.exists()
    assert not dataset_path.exists()
    with isolated_session() as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).one()
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
        ).one()
        values = (
            model.download_status,
            model.download_error,
            model.file_size,
            dataset.status,
            dataset.error_message,
            dataset.file_size,
        )
    assert values[0] == "failed"
    assert "restart" in values[1].lower()
    assert values[2] is None
    assert values[3] == "error"
    assert "restart" in values[4].lower()
    assert values[5] is None


def test_restart_cleanup_failure_keeps_residual_storage_billable(
    lifecycle_db,
    monkeypatch,
):
    isolated_session, tmp_path = lifecycle_db
    model_id = "residual-model"
    dataset_id = "residual-dataset"
    model_path = tmp_path / "models" / model_id
    dataset_path = tmp_path / "datasets" / dataset_id
    model_path.mkdir(parents=True)
    dataset_path.mkdir(parents=True)
    (model_path / "partial.bin").write_bytes(b"partial")
    (dataset_path / "partial.jsonl").write_bytes(b"partial")
    with isolated_session() as session:
        session.add_all(
            [
                ModelRegistryDB(
                    model_id=model_id,
                    model_name="residual-model",
                    model_type="embedding",
                    model_path=str(model_path),
                    source_type="downloaded",
                    download_status="pending",
                ),
                DatasetDB(
                    dataset_id=dataset_id,
                    dataset_name="residual-dataset",
                    storage_path=str(dataset_path),
                    source_type="huggingface",
                    status="downloading",
                ),
            ]
        )

    monkeypatch.setattr(
        remote_download_security.shutil,
        "rmtree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("in use")),
    )

    assert model_download_module.ModelDownloadService().cleanup_interrupted_downloads() == 1
    assert dataset_download_module.DatasetDownloadService().cleanup_interrupted_downloads() == 1

    with isolated_session() as session:
        model = session.exec(
            select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
        ).one()
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == dataset_id)
        ).one()
        assert model.file_size == len(b"partial")
        assert dataset.file_size == len(b"partial")
        assert "residual storage" in model.download_error
        assert "residual storage" in dataset.error_message


def test_server_download_cleanup_invokes_both_download_services(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model_download_module.model_download_service,
        "cleanup_interrupted_downloads",
        lambda: calls.append("model") or 2,
        raising=False,
    )
    monkeypatch.setattr(
        dataset_download_module.dataset_download_service,
        "cleanup_interrupted_downloads",
        lambda: calls.append("dataset") or 3,
        raising=False,
    )

    server.cleanup_orphan_datasets()

    assert calls == ["model", "dataset"]
