import asyncio
import inspect
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import deep_evaluation_routes, evaluation_routes
from train_factory.auth.resource_provenance import ResourceProvenanceError
from train_factory.deep_evaluation import deep_evaluation_runner
from train_factory.deep_evaluation import task_runner as legacy_deep_task_runner
from train_factory.evaluation import evaluation_runner
from train_factory.evaluation import dataset_access
from train_factory.storage.services.evaluation_task_service import (
    evaluation_task_service,
)


CURRENT_USER = {
    "user_id": "user-1",
    "username": "alice",
    "is_admin": False,
}


def _uploaded_dataset(path, *, user_id="user-1"):
    return {
        "dataset_id": "dataset-1",
        "dataset_name": "owned-dataset",
        "storage_path": str(path),
        "storage_backend": "local",
        "source_type": "uploaded",
        "status": "ready",
        "user_id": user_id,
    }


def test_managed_evaluation_dataset_resolver_proves_owner_and_managed_path(
    monkeypatch,
    tmp_path,
):
    datasets_root = tmp_path / "datasets"
    dataset_path = datasets_root / "dataset-1"
    dataset_path.mkdir(parents=True)
    dataset = _uploaded_dataset(dataset_path)
    monkeypatch.setattr(
        dataset_access.dataset_service,
        "get_dataset",
        lambda _dataset_id: dataset,
    )
    monkeypatch.setattr(
        dataset_access,
        "get_settings",
        lambda: SimpleNamespace(
            auth_enabled=True,
            datasets_dir=datasets_root,
            minio_bucket="trainfactory",
        ),
    )

    resolved = dataset_access.resolve_managed_local_evaluation_dataset(
        "dataset-1",
        user_id="user-1",
    )

    assert resolved is dataset


def test_managed_evaluation_dataset_resolver_rejects_owner_drift(
    monkeypatch,
    tmp_path,
):
    datasets_root = tmp_path / "datasets"
    dataset_path = datasets_root / "dataset-1"
    dataset_path.mkdir(parents=True)
    monkeypatch.setattr(
        dataset_access.dataset_service,
        "get_dataset",
        lambda _dataset_id: _uploaded_dataset(dataset_path, user_id="user-2"),
    )
    monkeypatch.setattr(
        dataset_access,
        "get_settings",
        lambda: SimpleNamespace(
            auth_enabled=True,
            datasets_dir=datasets_root,
            minio_bucket="trainfactory",
        ),
    )

    with pytest.raises(ResourceProvenanceError, match="owner"):
        dataset_access.resolve_managed_local_evaluation_dataset(
            "dataset-1",
            user_id="user-1",
        )


def test_evaluation_dataset_resolver_explicitly_rejects_s3(monkeypatch):
    monkeypatch.setattr(
        dataset_access.dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": "dataset-1",
            "storage_backend": "s3",
            "storage_uri": "s3://trainfactory/datasets/dataset-1/eval.jsonl",
            "source_type": "uploaded",
            "status": "ready",
            "user_id": "user-1",
        },
    )

    with pytest.raises(
        dataset_access.UnsupportedEvaluationDatasetStorageError,
        match="S3 datasets are not supported",
    ):
        dataset_access.resolve_managed_local_evaluation_dataset(
            "dataset-1",
            user_id="user-1",
        )


def _standard_request(*, dataset_id=None, dataset_type="registered", path="/spoofed/path"):
    return evaluation_routes.CreateEvaluationRequest(
        task_name="dataset-provenance",
        model_configs=[
            evaluation_routes.ModelConfig(
                endpoint="https://reranker.example.com",
                model_name="reranker",
                name="reranker",
            )
        ],
        dataset_configs=[
            evaluation_routes.DatasetConfig(
                type=dataset_type,
                name="client-name",
                path=path,
                dataset_id=dataset_id,
            )
        ],
    )


def _enable_route_provenance(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "requires_tenant_provenance",
        lambda _current_user: True,
        raising=False,
    )


def test_standard_evaluation_dataset_schema_preserves_dataset_id():
    config = _standard_request(dataset_id="dataset-1").dataset_configs[0]

    assert config.model_dump().get("dataset_id") == "dataset-1"


def test_standard_evaluation_requires_dataset_id_when_auth_is_enabled(monkeypatch):
    _enable_route_provenance(monkeypatch, evaluation_routes)
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_evaluation_dataset_path",
        lambda path: path,
    )
    task_creations = []
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.create_evaluation_task(
                _standard_request(dataset_id=None),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert task_creations == []
    assert background_tasks.tasks == []


def test_standard_evaluation_uses_owned_server_dataset_path(monkeypatch):
    _enable_route_provenance(monkeypatch, evaluation_routes)
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda _kind, _task_id, _user_id, operation, *args, **kwargs: (
            operation(*args, **kwargs),
            type("Lease", (), {"release": lambda self: None})(),
        ),
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_managed_local_evaluation_dataset",
        lambda dataset_id, *, user_id: {
            "dataset_id": dataset_id,
            "dataset_name": "server-name",
            "storage_path": "/managed/datasets/dataset-1",
            "user_id": user_id,
        },
        raising=False,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_evaluation_dataset_path",
        lambda path, **_kwargs: path,
    )
    task_creations = []
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )

    asyncio.run(
        evaluation_routes.create_evaluation_task(
            _standard_request(dataset_id="dataset-1"),
            BackgroundTasks(),
            CURRENT_USER,
        )
    )

    assert task_creations[0]["dataset_configs"] == [
        {
            "type": "registered",
            "name": "server-name",
            "path": "/managed/datasets/dataset-1",
            "dataset_id": "dataset-1",
            "_evaluation_identity": {
                "schema_version": 2,
                "result_key": "dataset:dataset-1",
            },
        }
    ]


def test_standard_evaluation_auth_disabled_keeps_legacy_local_path(monkeypatch):
    monkeypatch.setattr(
        evaluation_routes,
        "requires_tenant_provenance",
        lambda _current_user: False,
        raising=False,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_evaluation_dataset_path",
        lambda path, **_kwargs: f"resolved:{path}",
    )

    validated = evaluation_routes._validate_dataset_configs(
        [{"type": "local", "name": "legacy", "path": "/legacy/data.jsonl"}],
        {"user_id": None, "username": "anonymous"},
    )

    assert validated == [
        {
            "type": "local",
            "name": "legacy",
            "path": "resolved:/legacy/data.jsonl",
            "_evaluation_identity": {
                "schema_version": 2,
                "result_key": (
                    "local:sha256:"
                    "9e3f5b85623f9d4e2c19ca3469038cbbb43004ee12af1c94c6502fd639f97385"
                ),
            },
        }
    ]


def test_standard_evaluation_resume_rejects_dataset_provenance_drift(monkeypatch):
    _enable_route_provenance(monkeypatch, evaluation_routes)
    task = {
        "task_id": "task-1",
        "user_id": CURRENT_USER["user_id"],
        "status": "failed",
        "model_configs": [
            {"endpoint": "https://reranker.example.com", "model_name": "reranker"}
        ],
        "dataset_configs": [
            {"type": "registered", "name": "dataset", "dataset_id": "dataset-1"}
        ],
        "results": {},
    }
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_managed_local_evaluation_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ResourceProvenanceError("Dataset owner does not match")
        ),
        raising=False,
    )
    reset_calls = []
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "reset_for_resume",
        lambda task_id: reset_calls.append(task_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.resume_evaluation_task(
                "task-1",
                BackgroundTasks(),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert reset_calls == []


def test_standard_evaluation_worker_revalidates_dataset_identity(monkeypatch):
    task = {
        "task_id": "task-1",
        "user_id": "user-1",
        "status": "running",
        "model_configs": [
            {
                "endpoint": "https://example.com/v1/rerank",
                "name": "model",
            }
        ],
        "dataset_configs": [
            {
                "type": "registered",
                "name": "dataset",
                "dataset_id": "dataset-1",
                "path": "/persisted/spoofed/path",
            }
        ],
    }
    monkeypatch.setattr(
        evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    strict_calls = []
    monkeypatch.setattr(
        evaluation_runner,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "resolve_managed_local_evaluation_dataset",
        lambda dataset_id, *, user_id: strict_calls.append((dataset_id, user_id))
        or (_ for _ in ()).throw(ResourceProvenanceError("owner drift")),
        raising=False,
    )
    running_updates = []
    completion_updates = []
    monkeypatch.setattr(
        evaluation_task_service,
        "update_status",
        lambda *args, **kwargs: running_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "complete_task",
        lambda *args, **kwargs: completion_updates.append((args, kwargs)),
    )

    evaluation_runner.run_evaluation_task(
        "task-1",
        {
            "model_configs": task["model_configs"],
            "dataset_configs": task["dataset_configs"],
        },
    )

    assert strict_calls == [("dataset-1", "user-1")]
    assert running_updates == []
    assert completion_updates[0][1]["status"] == "failed"


def test_deep_evaluation_resume_rejects_dataset_provenance_drift(monkeypatch):
    _enable_route_provenance(monkeypatch, deep_evaluation_routes)
    task = {
        "task_id": "deep-1",
        "user_id": CURRENT_USER["user_id"],
        "status": "failed",
        "model_configs": [
            {
                "group_name": "group-1",
                "embedding": {
                    "endpoint": "https://example.com/v1",
                    "model_name": "embedding",
                    "concurrency": 1,
                },
            }
        ],
        "dataset_configs": [{"dataset_id": "dataset-1"}],
        "metrics": ["mrr"],
        "worker_groups": {"retrieval_mode": "offline", "model_workers": 1},
        "results_summary": {},
        "model_progress": {},
    }
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": "dataset-1",
            "dataset_name": "dataset",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "resolve_managed_local_evaluation_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ResourceProvenanceError("Dataset path does not match its record")
        ),
        raising=False,
    )
    reset_calls = []
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "reset_for_resume",
        lambda task_id: reset_calls.append(task_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.resume_deep_evaluation_task(
                "deep-1",
                BackgroundTasks(),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert reset_calls == []


def test_deep_worker_revalidates_dataset_before_file_access(monkeypatch):
    task = {
        "task_id": "deep-1",
        "user_id": "user-1",
        "status": "pending",
        "model_configs": [{"group_name": "group-1", "rerank": {"endpoint": "https://x"}}],
        "dataset_configs": [{"dataset_id": "dataset-1"}],
        "metrics": ["mrr"],
    }
    monkeypatch.setattr(
        deep_evaluation_runner.deep_evaluation_task_service,
        "get_task",
        lambda *_args, **_kwargs: task,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "resolve_managed_local_evaluation_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ResourceProvenanceError("Dataset owner does not match")
        ),
        raising=False,
    )
    source_calls = []
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_resolve_dataset_source",
        lambda *args: source_calls.append(args) or (_ for _ in ()).throw(
            AssertionError("file access must not happen")
        ),
    )
    status_updates = []
    monkeypatch.setattr(
        deep_evaluation_runner.deep_evaluation_task_service,
        "update_status",
        lambda *args: status_updates.append(args),
    )
    monkeypatch.setattr(
        deep_evaluation_runner.deep_evaluation_task_service,
        "complete_task",
        lambda task_id, status, **kwargs: status_updates.append(
            (task_id, status, kwargs.get("error_message"))
        )
        or True,
    )

    deep_evaluation_runner.run_deep_evaluation_task("deep-1")

    assert source_calls == []
    assert status_updates
    assert status_updates[0][1] == "failed"


def test_legacy_deep_worker_dataset_resolver_requires_task_user():
    assert "user_id" in inspect.signature(
        legacy_deep_task_runner._resolve_datasets
    ).parameters
def test_local_evaluation_rejects_dataset_deletion_in_progress(monkeypatch):
    monkeypatch.setattr(
        dataset_access.dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": "dataset-deleting",
            "status": "deleting",
            "storage_backend": "local",
            "storage_path": "/managed/dataset-deleting/data.jsonl",
        },
    )

    with pytest.raises(
        dataset_access.ResourceProvenanceError,
        match="not ready",
    ):
        dataset_access.resolve_local_evaluation_dataset_record(
            "dataset-deleting"
        )
