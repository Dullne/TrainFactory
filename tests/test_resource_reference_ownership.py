import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import (
    adapter_routes,
    dataset_routes,
    deep_evaluation_routes,
    registry_routes,
)
from train_factory.storage.services.deep_evaluation_task_service import (
    deep_evaluation_task_service,
)
from train_factory.storage.services.milvus_collection_service import (
    MilvusCollectionUnavailableError,
)
from train_factory.storage.services.model_registry_service import model_registry_service
from train_factory.storage.services.training_task_service import training_task_service


CURRENT_USER = {"user_id": "user-1", "username": "alice"}


def _loaded_adapter_response(**overrides):
    response = {
        "adapter_id": "adapter-1",
        "deployment_id": "deployment-1",
        "adapter_name": "owned",
        "adapter_path": "/app/output/task-owned/final_model",
        "source_task_id": "task-owned",
        "source_model_id": None,
        "status": "loaded",
        "error_message": None,
        "user_id": "user-1",
        "loaded_at": None,
        "unloaded_at": None,
    }
    response.update(overrides)
    return response


def _mock_owned_deployment(monkeypatch):
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: {"deployment_id": deployment_id, "user_id": "user-1"},
    )


def _mock_model_registration_dependencies(monkeypatch):
    monkeypatch.setattr(
        registry_routes,
        "map_storage_path",
        lambda path: (path, None),
    )
    monkeypatch.setattr(
        registry_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )
    monkeypatch.setattr(
        registry_routes,
        "check_idempotency",
        lambda *args, **kwargs: (False, None),
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "get_model_by_path",
        lambda path: None,
    )


@pytest.mark.parametrize(
    "current_user",
    [
        CURRENT_USER,
        {"user_id": "admin-1", "username": "admin", "is_admin": True},
    ],
)
def test_authenticated_user_cannot_register_unproven_model_path(
    monkeypatch,
    current_user,
):
    registrations = []
    _mock_model_registration_dependencies(monkeypatch)
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "register_model",
        lambda **kwargs: registrations.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            registry_routes.register_model(
                registry_routes.RegisterModelRequest(
                    model_name="unproven",
                    model_path="/app/models/foreign-checkpoint",
                    model_type="embedding",
                ),
                current_user,
                None,
            )
        )

    assert exc_info.value.status_code == 403
    assert registrations == []


@pytest.mark.parametrize(
    ("task_owner", "registered_path", "expected_status"),
    [
        (None, "/app/output/task-1/final_model", 403),
        ("user-2", "/app/output/task-1/final_model", 403),
        ("user-1", "/app/output/task-1/checkpoint-10", 400),
    ],
)
def test_regular_user_model_registration_requires_owned_task_final_path(
    monkeypatch,
    task_owner,
    registered_path,
    expected_status,
):
    registrations = []
    _mock_model_registration_dependencies(monkeypatch)
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": task_owner,
            "status": "succeeded",
            "is_lora": True,
            "output_dir": "/app/output/task-1",
            "final_model_path": "/app/output/task-1/final_model",
        },
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "register_model",
        lambda **kwargs: registrations.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            registry_routes.register_model(
                registry_routes.RegisterModelRequest(
                    model_name="spoofed",
                    model_path=registered_path,
                    model_type="embedding",
                    source_task_id="task-1",
                    is_adapter=True,
                ),
                CURRENT_USER,
                None,
            )
        )

    assert exc_info.value.status_code == expected_status
    assert registrations == []


def test_regular_user_cannot_add_unproven_model_version(monkeypatch):
    versions = []
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "get_model",
        lambda model_id: {"model_id": model_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "add_version",
        lambda **kwargs: versions.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            registry_routes.add_version(
                "model-1",
                registry_routes.AddVersionRequest(
                    version="v2",
                    model_path="/app/output/foreign-task/checkpoint-10",
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert versions == []


@pytest.mark.parametrize(
    "current_user",
    [
        CURRENT_USER,
        {"user_id": "admin-1", "username": "admin", "is_admin": True},
    ],
)
def test_authenticated_user_cannot_directly_claim_dataset_storage(
    monkeypatch,
    current_user,
):
    creations = []
    monkeypatch.setattr(
        dataset_routes,
        "check_idempotency",
        lambda *args, **kwargs: (False, None),
    )
    monkeypatch.setattr(
        dataset_routes.dataset_service,
        "create_dataset",
        lambda **kwargs: creations.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            dataset_routes.create_dataset(
                dataset_routes.CreateDatasetRequest(
                    dataset_name="claimed",
                    storage_backend="local",
                    storage_path="/app/data/foreign-dataset/train.jsonl",
                ),
                current_user,
                None,
            )
        )

    assert exc_info.value.status_code == 403
    assert creations == []


def test_direct_load_adapter_requires_exactly_one_owned_source(monkeypatch):
    adapter_loads = []
    _mock_owned_deployment(monkeypatch)
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda *args, **kwargs: adapter_loads.append((args, kwargs))
        or _loaded_adapter_response(),
    )

    for source_ids in (
        {},
        {"source_task_id": "task-owned", "source_model_id": "model-owned"},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                adapter_routes.load_adapter(
                    deployment_id="deployment-1",
                    request=adapter_routes.LoadAdapterRequest(
                        adapter_name="owned",
                        adapter_path="/app/output/task-owned/final_model",
                        **source_ids,
                    ),
                    current_user=CURRENT_USER,
                )
            )

        assert exc_info.value.status_code == 400

    assert adapter_loads == []


@pytest.mark.parametrize("task_owner", [None, "user-2"])
def test_direct_load_adapter_rejects_unowned_task_source(
    monkeypatch,
    task_owner,
):
    adapter_loads = []
    _mock_owned_deployment(monkeypatch)
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": task_owner,
            "is_lora": True,
            "status": "succeeded",
            "final_model_path": "/app/output/task-foreign/final_model",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda *args, **kwargs: adapter_loads.append((args, kwargs))
        or _loaded_adapter_response(),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            adapter_routes.load_adapter(
                deployment_id="deployment-1",
                request=adapter_routes.LoadAdapterRequest(
                    adapter_name="foreign",
                    adapter_path="/app/output/task-foreign/final_model",
                    source_task_id="task-foreign",
                ),
                current_user=CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert adapter_loads == []


def test_direct_load_adapter_rejects_path_mismatch_before_side_effect(monkeypatch):
    adapter_loads = []
    _mock_owned_deployment(monkeypatch)
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "is_lora": True,
            "status": "succeeded",
            "final_model_path": "/app/output/task-owned/final_model",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda *args, **kwargs: adapter_loads.append((args, kwargs))
        or _loaded_adapter_response(),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            adapter_routes.load_adapter(
                deployment_id="deployment-1",
                request=adapter_routes.LoadAdapterRequest(
                    adapter_name="spoofed",
                    adapter_path="/app/output/task-foreign/final_model",
                    source_task_id="task-owned",
                ),
                current_user=CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 400
    assert adapter_loads == []


def test_direct_load_adapter_resolves_owned_registry_source(monkeypatch):
    adapter_loads = []
    _mock_owned_deployment(monkeypatch)
    monkeypatch.setattr(
        model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "user_id": "user-1",
            "is_adapter": True,
            "source_task_id": "task-owned",
            "model_path": "/app/models/owned-adapter",
        },
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "is_lora": True,
            "status": "succeeded",
            "final_model_path": "/app/models/owned-adapter",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda *args, **kwargs: adapter_loads.append((args, kwargs))
        or _loaded_adapter_response(
            adapter_path="/app/models/owned-adapter",
            source_task_id=None,
            source_model_id="model-owned",
        ),
    )

    response = asyncio.run(
        adapter_routes.load_adapter(
            deployment_id="deployment-1",
            request=adapter_routes.LoadAdapterRequest(
                adapter_name="owned",
                adapter_path="/app/models/owned-adapter",
                source_model_id="model-owned",
            ),
            current_user=CURRENT_USER,
        )
    )

    assert response.source_model_id == "model-owned"
    assert adapter_loads == [
        (
            (),
            {
                "deployment_id": "deployment-1",
                "adapter_name": "owned",
                "adapter_path": "/app/models/owned-adapter",
                "source_task_id": None,
                "source_model_id": "model-owned",
                "deployment_replica_id": None,
                "user_id": "user-1",
            },
        )
    ]


@pytest.mark.parametrize(
    ("model_owner", "is_adapter", "requested_path", "expected_status"),
    [
        (None, True, "/app/models/owned-adapter", 403),
        ("user-2", True, "/app/models/owned-adapter", 403),
        ("user-1", False, "/app/models/owned-adapter", 400),
        ("user-1", True, "/app/models/foreign-adapter", 400),
    ],
)
def test_direct_load_adapter_rejects_invalid_registry_source(
    monkeypatch,
    model_owner,
    is_adapter,
    requested_path,
    expected_status,
):
    adapter_loads = []
    _mock_owned_deployment(monkeypatch)
    monkeypatch.setattr(
        model_registry_service,
        "get_model",
        lambda model_id: {
            "model_id": model_id,
            "user_id": model_owner,
            "is_adapter": is_adapter,
            "model_path": "/app/models/owned-adapter",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda *args, **kwargs: adapter_loads.append((args, kwargs))
        or _loaded_adapter_response(),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            adapter_routes.load_adapter(
                deployment_id="deployment-1",
                request=adapter_routes.LoadAdapterRequest(
                    adapter_name="invalid",
                    adapter_path=requested_path,
                    source_model_id="model-owned",
                ),
                current_user=CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == expected_status
    assert adapter_loads == []


@pytest.mark.parametrize("task_owner", [None, "user-2"])
def test_load_adapter_from_task_rejects_unowned_task_before_adapter_side_effect(
    monkeypatch,
    task_owner,
):
    adapter_loads = []
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: {"deployment_id": deployment_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": task_owner,
            "is_lora": True,
            "status": "succeeded",
            "final_model_path": "/app/models/foreign-adapter",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda *args, **kwargs: adapter_loads.append((args, kwargs))
        or {
            "adapter_id": "adapter-1",
            "deployment_id": "deployment-1",
            "adapter_name": "foreign",
            "adapter_path": "/app/models/foreign-adapter",
            "status": "loaded",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            adapter_routes.load_adapter_from_task(
                deployment_id="deployment-1",
                request=adapter_routes.LoadAdapterFromTaskRequest(task_id="task-foreign"),
                current_user=CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert adapter_loads == []


@pytest.mark.parametrize("model_owner", [None, "user-2"])
def test_compare_models_rejects_unowned_model_before_compare_side_effect(
    monkeypatch,
    model_owner,
):
    model_reads = []
    comparisons = []

    def get_model(model_id):
        model_reads.append(model_id)
        owner = "user-1" if model_id == "model-owned" else model_owner
        return {"model_id": model_id, "user_id": owner}

    monkeypatch.setattr(registry_routes.model_registry_service, "get_model", get_model)
    monkeypatch.setattr(
        registry_routes.model_registry_service,
        "compare_models",
        lambda model_ids: comparisons.append(model_ids) or {},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            registry_routes.compare_models(
                registry_routes.CompareModelsRequest(
                    model_ids=["model-owned", "model-unowned"]
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert model_reads == ["model-owned", "model-unowned"]
    assert comparisons == []


def _deep_evaluation_request(
    *,
    retrieval_mode="offline",
    milvus_collection=None,
    retrieval_embedding_config=None,
):
    return deep_evaluation_routes.CreateDeepEvaluationTaskRequest(
        task_name="ownership-check",
        model_configs=[
            {
                "group_name": "embedding",
                "embedding": {
                    "endpoint": "http://127.0.0.1:8000/v1",
                    "model_name": "embed-model",
                },
            }
        ],
        dataset_configs=[{"dataset_id": "dataset-1"}],
        metrics=["mrr"],
        retrieval_mode=retrieval_mode,
        milvus_collection=milvus_collection,
        retrieval_embedding_config=retrieval_embedding_config,
    )


@pytest.mark.parametrize("collection_owner", [None, "user-2"])
def test_online_deep_evaluation_rejects_unowned_collection_before_task_creation(
    monkeypatch,
    collection_owner,
):
    task_creations = []
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *args, **kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "dataset_name": "owned dataset",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.milvus_collection_service,
        "get_by_name",
        lambda collection_name: {
            "collection_name": collection_name,
            "embedding_model": "embed-model",
            "user_id": collection_owner,
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                _deep_evaluation_request(
                    retrieval_mode="online",
                    milvus_collection="collection-foreign",
                    retrieval_embedding_config={
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "model_name": "embed-model",
                    },
                ),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert task_creations == []
    assert background_tasks.tasks == []


def test_online_deep_evaluation_rejects_deleting_collection_before_admission(
    monkeypatch,
):
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *args, **kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: [
            {"dataset_id": "dataset-1", "dataset_name": "owned dataset"}
        ],
    )
    monkeypatch.setattr(
        deep_evaluation_routes.milvus_collection_service,
        "get_by_name",
        lambda collection_name: {
            "collection_name": collection_name,
            "embedding_model": "embed-model",
            "user_id": "user-1",
            "status": "deleting",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: pytest.fail(
            "a deleting collection must be rejected before task admission"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                _deep_evaluation_request(
                    retrieval_mode="online",
                    milvus_collection="collection-deleting",
                    retrieval_embedding_config={
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "model_name": "embed-model",
                    },
                ),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert background_tasks.tasks == []


def test_online_deep_evaluation_maps_collection_fence_race_to_conflict(
    monkeypatch,
):
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *args, **kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: [
            {"dataset_id": "dataset-1", "dataset_name": "owned dataset"}
        ],
    )
    monkeypatch.setattr(
        deep_evaluation_routes.milvus_collection_service,
        "get_by_name",
        lambda collection_name: {
            "collection_name": collection_name,
            "embedding_model": "embed-model",
            "user_id": "user-1",
            "status": "active",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MilvusCollectionUnavailableError("collection is being deleted")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                _deep_evaluation_request(
                    retrieval_mode="online",
                    milvus_collection="collection-racing",
                    retrieval_embedding_config={
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "model_name": "embed-model",
                    },
                ),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert background_tasks.tasks == []


class _SecretReadGuard(dict):
    def __init__(self, *args, secret_reads, **kwargs):
        super().__init__(*args, **kwargs)
        self._secret_reads = secret_reads

    def get(self, key, default=None):
        if key == "api_key":
            self._secret_reads.append(key)
        return super().get(key, default)


@pytest.mark.parametrize("config_owner", [None, "user-2"])
def test_auto_resolved_retrieval_config_checks_owner_before_reading_secret(
    monkeypatch,
    config_owner,
):
    secret_reads = []
    task_creations = []
    config = _SecretReadGuard(
        {
            "config_id": "config-foreign",
            "config_name": "foreign embedding",
            "api_endpoint": "http://127.0.0.1:8000/v1",
            "model_name": "embed-model",
            "api_key": "must-not-be-read",
            "user_id": config_owner,
        },
        secret_reads=secret_reads,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *args, **kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "dataset_name": "owned dataset",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.milvus_collection_service,
        "get_by_name",
        lambda collection_name: {
            "collection_name": collection_name,
            "embedding_config_id": "config-foreign",
            "embedding_model": "embed-model",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.model_config_service,
        "get_config",
        lambda config_id: config,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                _deep_evaluation_request(
                    retrieval_mode="online",
                    milvus_collection="collection-owned",
                ),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert secret_reads == []
    assert task_creations == []
    assert background_tasks.tasks == []


def test_deep_evaluation_rejects_ownerless_dataset_before_task_creation(monkeypatch):
    task_creations = []
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, *args, **kwargs: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "dataset_name": "legacy dataset",
            "user_id": None,
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                _deep_evaluation_request(),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert task_creations == []
    assert background_tasks.tasks == []


def test_deep_evaluation_rejects_ownerless_model_config_before_secret_read(monkeypatch):
    secret_reads = []
    config = _SecretReadGuard(
        {
            "config_id": "config-ownerless",
            "api_key": "must-not-be-read",
            "user_id": None,
        },
        secret_reads=secret_reads,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.model_config_service,
        "get_config",
        lambda config_id: config,
    )

    with pytest.raises(HTTPException) as exc_info:
        deep_evaluation_routes._resolve_model_config("config-ownerless", CURRENT_USER)

    assert exc_info.value.status_code == 403
    assert secret_reads == []


@pytest.mark.parametrize("operation", ["get", "cancel", "resume", "delete"])
def test_deep_evaluation_task_operations_reject_ownerless_task_before_side_effect(
    monkeypatch,
    operation,
):
    side_effects = []
    status_by_operation = {
        "get": "completed",
        "cancel": "pending",
        "resume": "failed",
        "delete": "completed",
    }
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": None,
            "status": status_by_operation[operation],
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "update_status",
        lambda *args, **kwargs: side_effects.append("update_status"),
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "reset_for_resume",
        lambda *args, **kwargs: side_effects.append("reset_for_resume"),
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "delete_task",
        lambda *args, **kwargs: side_effects.append("delete_task"),
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "cancel_deep_evaluation",
        lambda *args, **kwargs: side_effects.append("cancel"),
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "clear_deep_cancellation",
        lambda *args, **kwargs: side_effects.append("clear"),
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        if operation == "get":
            asyncio.run(
                deep_evaluation_routes.get_deep_evaluation_task("task-ownerless", CURRENT_USER)
            )
        elif operation == "cancel":
            asyncio.run(
                deep_evaluation_routes.cancel_deep_evaluation_task(
                    "task-ownerless", CURRENT_USER
                )
            )
        elif operation == "resume":
            asyncio.run(
                deep_evaluation_routes.resume_deep_evaluation_task(
                    "task-ownerless", background_tasks, CURRENT_USER
                )
            )
        else:
            asyncio.run(
                deep_evaluation_routes.delete_deep_evaluation_task(
                    "task-ownerless", CURRENT_USER
                )
            )

    assert exc_info.value.status_code == 403
    assert side_effects == []
    assert background_tasks.tasks == []


def test_deep_evaluation_task_response_masks_worker_group_api_keys_without_mutation():
    task = SimpleNamespace(
        task_id="task-1",
        task_name="secrets",
        description=None,
        eval_type="embedding",
        model_configs=[{"embedding": {"api_key": "model-secret"}}],
        dataset_configs=[],
        field_mapping=None,
        metrics=["mrr"],
        worker_groups={
            "retrieval_embedding_config": {
                "api_key": "retrieval-secret",
                "model_name": "embed-model",
            }
        },
        llm_config={"api_key": "llm-secret"},
        max_samples=None,
        status="pending",
        progress=0.0,
        total_samples=0,
        processed_samples=0,
        model_progress=None,
        results=None,
        results_path=None,
        error_message=None,
        user_id="user-1",
        created_at=None,
        started_at=None,
        completed_at=None,
    )

    public_task = deep_evaluation_task_service._to_task_dict(task)
    worker_task = deep_evaluation_task_service._to_task_dict(task, mask_api_key=False)

    assert public_task["model_configs"][0]["embedding"]["api_key"] == "***"
    assert public_task["llm_config"]["api_key"] == "***"
    assert public_task["worker_groups"]["retrieval_embedding_config"]["api_key"] == "***"
    assert worker_task["worker_groups"]["retrieval_embedding_config"]["api_key"] == "retrieval-secret"
    assert task.worker_groups["retrieval_embedding_config"]["api_key"] == "retrieval-secret"
