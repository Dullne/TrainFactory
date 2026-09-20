from __future__ import annotations

import asyncio
import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID

import pytest
import sqlalchemy as sa

from train_factory.deployment.deployment_service import DeploymentService
from train_factory.api.routes import adapter_routes, model_config_routes
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.services.model_config_service import ModelConfigService
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB


config_module = importlib.import_module(
    "train_factory.storage.services.model_config_service"
)
adapter_module = importlib.import_module("train_factory.deployment.adapter_service")
evaluation_routes_module = importlib.import_module(
    "train_factory.api.routes.evaluation_routes"
)
sync_routes_module = importlib.import_module("train_factory.api.routes.sync_routes")
deployment_service_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)


def _parent(children: list[dict[str, object]]) -> dict[str, object]:
    return {
        "deployment_id": "deployment-1",
        "user_id": "user-1",
        "deploy_mode": "container",
        "inference_framework": "vllm",
        "replica_instances": children,
    }


def _child(index: int, *, status: str = "running", health: str = "HEALTHY"):
    return {
        "replica_id": f"replica-{index}",
        "deployment_id": "deployment-1",
        "replica_index": index,
        "container_name": f"container-{index}",
        "endpoint": f"http://127.0.0.1:{11000 + index}",
        "port": 11000 + index,
        "gpu_ids": [index],
        "status": status,
        "health_status": health,
    }


def test_single_child_is_selected_when_replica_id_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DeploymentService()
    monkeypatch.setattr(service, "get_deployment", lambda _deployment_id: _parent([_child(0)]))

    deployment, replica = service.resolve_replica_selection(
        "deployment-1",
        None,
        user_id="user-1",
        require_healthy=True,
    )

    assert deployment["deployment_id"] == "deployment-1"
    assert replica["replica_id"] == "replica-0"


def test_multi_child_requires_explicit_selection_without_replica_zero_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DeploymentService()
    monkeypatch.setattr(
        service,
        "get_deployment",
        lambda _deployment_id: _parent([_child(0), _child(1)]),
    )

    with pytest.raises(ValueError, match="replica_id is required"):
        service.resolve_replica_selection(
            "deployment-1",
            None,
            user_id="user-1",
            require_healthy=True,
        )


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_canonical_replica_marker_rejects_empty_child_group(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    service = DeploymentService()
    parent = {
        **_parent([]),
        "inference_framework": framework,
        "config": {"replica_schema_version": 1},
        "status": "running",
    }
    monkeypatch.setattr(service, "get_deployment", lambda _deployment_id: parent)

    with pytest.raises(ValueError, match="replica group is empty"):
        service.resolve_replica_selection(
            "deployment-1",
            None,
            user_id="user-1",
            require_healthy=True,
        )


@pytest.mark.parametrize("marker", [None, True, "1"])
def test_legacy_empty_child_group_does_not_trust_non_integer_marker(
    monkeypatch: pytest.MonkeyPatch,
    marker: object,
) -> None:
    service = DeploymentService()
    parent = {
        **_parent([]),
        "config": ({"replica_schema_version": marker} if marker is not None else {}),
        "status": "running",
    }
    monkeypatch.setattr(service, "get_deployment", lambda _deployment_id: parent)

    deployment, replica = service.resolve_replica_selection(
        "deployment-1",
        None,
        user_id="user-1",
        require_healthy=True,
    )

    assert deployment is parent
    assert replica is None


@pytest.mark.parametrize("status", ["stopped", "failed", "degraded"])
def test_legacy_empty_child_group_requires_running_parent_for_healthy_selection(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    service = DeploymentService()
    parent = {
        **_parent([]),
        "config": {},
        "status": status,
    }
    monkeypatch.setattr(service, "get_deployment", lambda _deployment_id: parent)

    with pytest.raises(ValueError, match="deployment is not running"):
        service.resolve_replica_selection(
            "deployment-1",
            None,
            user_id="user-1",
            require_healthy=True,
        )


def test_legacy_empty_child_group_allows_non_health_management_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DeploymentService()
    parent = {
        **_parent([]),
        "config": {},
        "status": "stopped",
    }
    monkeypatch.setattr(service, "get_deployment", lambda _deployment_id: parent)

    deployment, replica = service.resolve_replica_selection(
        "deployment-1",
        None,
        user_id="user-1",
        require_healthy=False,
    )

    assert deployment is parent
    assert replica is None


@pytest.mark.parametrize(
    "replica",
    [
        {**_child(1), "deployment_id": "deployment-2"},
        _child(1, status="stopped", health="UNKNOWN"),
    ],
)
def test_selected_child_must_belong_and_be_healthy(
    monkeypatch: pytest.MonkeyPatch,
    replica,
) -> None:
    service = DeploymentService()
    monkeypatch.setattr(service, "get_deployment", lambda _deployment_id: _parent([replica]))

    with pytest.raises(ValueError):
        service.resolve_replica_selection(
            "deployment-1",
            "replica-1",
            user_id="user-1",
            require_healthy=True,
        )


def test_local_config_uses_trusted_child_binding_and_persists_replica_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'selection.db'}")
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)
    ModelConfigDB.__table__.create(engine)
    with config_module.Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="deployment-1",
                model_id="model-1",
                xinference_endpoint="http://127.0.0.1:11000",
                replica=2,
                deploy_mode="container",
                container_name="container-0",
                inference_framework="vllm",
                config={"replica_schema_version": 1},
                status="degraded",
                user_id="user-1",
            )
        )
        session.add_all(
            [
                DeploymentReplicaDB(
                    replica_id=f"replica-{index}",
                    deployment_id="deployment-1",
                    replica_index=index,
                    container_name=f"container-{index}",
                    endpoint=f"http://127.0.0.1:{11000 + index}",
                    port=11000 + index,
                    gpu_ids=[index],
                    status="running",
                    health_status="HEALTHY",
                )
                for index in range(2)
            ]
        )
        session.commit()
    service = ModelConfigService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)
    monkeypatch.setattr(service, "_to_external_endpoint", lambda endpoint: endpoint)
    monkeypatch.setattr(
        config_module,
        "normalize_api_endpoint",
        lambda endpoint, _provider, _user_id: endpoint,
    )

    config = service.create_config(
        config_name="child-one",
        model_type="llm",
        provider="xinference",
        api_endpoint="https://attacker.invalid",
        model_name="served",
        source_type="local_deployed",
        deployment_id="deployment-1",
        deployment_replica_id="replica-1",
        container_name="attacker-container",
        inference_framework="sglang",
        user_id="user-1",
    )

    assert config["deployment_replica_id"] == "replica-1"
    assert config["api_endpoint"] == "http://127.0.0.1:11001"
    assert config["container_name"] == "container-1"
    assert config["inference_framework"] == "vllm"


def test_local_config_rejects_delete_claim_without_inserting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'delete-claim.db'}")
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)
    ModelConfigDB.__table__.create(engine)
    with config_module.Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="deployment-delete",
                model_id="model-1",
                xinference_endpoint="http://127.0.0.1:11000",
                replica=1,
                deploy_mode="container",
                container_name="container-0",
                inference_framework="vllm",
                config={"replica_schema_version": 1},
                status="running",
                user_id="user-1",
                replica_operation_token="delete-owner",
                replica_operation_kind="delete",
                replica_operation_generation=1,
            )
        )
        session.add(
            DeploymentReplicaDB(
                replica_id="replica-0",
                deployment_id="deployment-delete",
                replica_index=0,
                container_name="container-0",
                endpoint="http://127.0.0.1:11000",
                port=11000,
                gpu_ids=[0],
                status="running",
                health_status="HEALTHY",
            )
        )
        session.commit()

    service = ModelConfigService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    with pytest.raises(
        deployment_service_module.ReplicaOperationBusyError,
        match="being deleted",
    ):
        service.create_config(
            config_name="must-not-insert",
            model_type="llm",
            provider="xinference",
            api_endpoint="http://127.0.0.1:11000",
            model_name="served",
            source_type="local_deployed",
            deployment_id="deployment-delete",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    with config_module.Session(engine) as session:
        assert session.exec(sa.select(ModelConfigDB)).all() == []


def test_bound_local_config_route_discards_connection_identity_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {
        "config_id": "config-1",
        "config_name": "bound-config",
        "model_type": "llm",
        "provider": "xinference",
        "api_endpoint": "http://127.0.0.1:11001",
        "api_key": None,
        "model_name": "trusted-model",
        "provider_config": None,
        "default_params": None,
        "description": "before",
        "tags": None,
        "status": "active",
        "is_default": False,
        "last_check_status": None,
        "last_check_time": None,
        "last_check_error": None,
        "user_id": "user-1",
        "created_at": None,
        "updated_at": None,
        "source_type": "local_deployed",
        "deployment_id": "deployment-1",
        "deployment_replica_id": "replica-1",
        "container_name": "container-1",
        "inference_framework": "vllm",
    }
    updates: list[dict[str, object]] = []

    def get_config(_config_id: str) -> dict[str, object]:
        return dict(stored)

    def update_config(*, config_id: str, **kwargs: object) -> bool:
        assert config_id == "config-1"
        updates.append(kwargs)
        for field, value in kwargs.items():
            if value is not None:
                stored[field] = value
        return True

    monkeypatch.setattr(model_config_routes.model_config_service, "get_config", get_config)
    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "update_config",
        update_config,
    )

    response = asyncio.run(
        model_config_routes.update_config(
            "config-1",
            model_config_routes.UpdateConfigRequest(
                provider="openai",
                api_endpoint="https://attacker.invalid/v1",
                model_name="attacker-model",
                description="after",
            ),
            {"user_id": "user-1"},
        )
    )

    assert updates[0]["provider"] is None
    assert updates[0]["api_endpoint"] is None
    assert updates[0]["model_name"] is None
    assert updates[0]["description"] == "after"
    assert response.provider == "xinference"
    assert response.api_endpoint == "http://127.0.0.1:11001"
    assert response.model_name == "trusted-model"
    assert response.description == "after"


def test_bound_local_config_service_preserves_connection_identity_in_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'bound-config-update.db'}")
    DeploymentReplicaDB.__table__.create(engine)
    ModelConfigDB.__table__.create(engine)
    with config_module.Session(engine) as session:
        session.add(
            DeploymentReplicaDB(
                replica_id="replica-1",
                deployment_id="deployment-1",
                replica_index=1,
                container_name="container-1",
                endpoint="http://127.0.0.1:11001",
                port=11001,
                gpu_ids=[1],
                status="running",
                health_status="HEALTHY",
            )
        )
        session.add(
            ModelConfigDB(
                config_id="config-1",
                config_name="bound-config",
                model_type="llm",
                provider="xinference",
                api_endpoint="http://127.0.0.1:11001",
                model_name="trusted-model",
                description="before",
                source_type="local_deployed",
                deployment_id="deployment-1",
                deployment_replica_id="replica-1",
                user_id="user-1",
            )
        )
        session.commit()

    service = ModelConfigService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    assert service.update_config(
        "config-1",
        provider="openai",
        api_endpoint="https://attacker.invalid/v1",
        model_name="attacker-model",
        description="after",
    )

    with config_module.Session(engine) as session:
        config = session.execute(
            sa.select(ModelConfigDB).where(ModelConfigDB.config_id == "config-1")
        ).scalar_one()
        assert config.provider == "xinference"
        assert config.api_endpoint == "http://127.0.0.1:11001"
        assert config.model_name == "trusted-model"
        assert config.description == "after"


def test_multi_child_local_config_without_selection_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'selection-required.db'}")
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)
    ModelConfigDB.__table__.create(engine)
    with config_module.Session(engine) as session:
        session.add(
            DeploymentDB(
                deployment_id="deployment-1",
                model_id="model-1",
                xinference_endpoint="http://127.0.0.1:11000",
                replica=2,
                deploy_mode="container",
                container_name="container-0",
                inference_framework="vllm",
                config={"replica_schema_version": 1},
                status="degraded",
                user_id="user-1",
            )
        )
        session.add_all(
            [
                DeploymentReplicaDB(
                    replica_id=f"replica-{index}",
                    deployment_id="deployment-1",
                    replica_index=index,
                    container_name=f"container-{index}",
                    endpoint=f"http://127.0.0.1:{11000 + index}",
                    port=11000 + index,
                    gpu_ids=[index],
                    status="running",
                    health_status="HEALTHY",
                )
                for index in range(2)
            ]
        )
        session.commit()
    service = ModelConfigService()
    monkeypatch.setattr(service, "_get_engine", lambda: engine)

    with pytest.raises(ValueError, match="replica_id is required"):
        service.create_config(
            config_name="missing-child",
            model_type="llm",
            provider="xinference",
            api_endpoint="http://127.0.0.1:11000",
            model_name="served",
            source_type="local_deployed",
            deployment_id="deployment-1",
            user_id="user-1",
        )


@pytest.fixture
def adapter_selection_service(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'adapter-selection.db'}")
    ModelArtifactMembershipGateDB.__table__.create(engine)
    ModelRegistryDB.__table__.create(engine)
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)
    LoadedAdapterDB.__table__.create(engine)

    @contextmanager
    def get_session():
        with config_module.Session(engine) as session:
            yield session

    with config_module.Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.add(
            ModelRegistryDB(
                model_id="model-1",
                model_name="model-1",
                model_type="llm",
                model_path=str(tmp_path / "model-1"),
                source_type="trained",
                status="available",
            )
        )
        session.add(
            DeploymentDB(
                deployment_id="deployment-1",
                model_id="model-1",
                model_uid="served",
                xinference_endpoint="http://127.0.0.1:11000",
                replica=2,
                deploy_mode="container",
                container_name="container-0",
                inference_framework="vllm",
                enable_lora=True,
                config={"replica_schema_version": 1},
                status="running",
                user_id="user-1",
            )
        )
        session.add_all(
            [
                DeploymentReplicaDB(
                    replica_id=f"replica-{index}",
                    deployment_id="deployment-1",
                    replica_index=index,
                    container_name=("container-0" if index == 0 else f"container-0-r{index}"),
                    endpoint=f"http://127.0.0.1:{11000 + index}",
                    port=11000 + index,
                    gpu_ids=[index],
                    status="running",
                    health_status="HEALTHY",
                )
                for index in range(2)
            ]
        )
        session.commit()
    monkeypatch.setattr(adapter_module, "get_session", get_session)
    monkeypatch.setattr(deployment_service_module, "get_session", get_session)
    monkeypatch.setattr(
        adapter_module.model_registry_service,
        "get_model",
        lambda _model_id: {"model_type": "llm"},
    )
    return adapter_module.AdapterService(DeploymentService()), engine


def test_adapter_multi_replica_requires_selection_before_client(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, _engine = adapter_selection_service
    client_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: client_calls.append("client"),
    )

    with pytest.raises(ValueError, match="replica_id is required"):
        service.load_adapter(
            "deployment-1",
            "adapter",
            "/app/output/adapter",
            user_id="user-1",
        )

    assert client_calls == []


def test_legacy_adapter_load_and_unload_bypass_replica_claim(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(
            config_module.select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        parent.config = {"legacy": True}
        session.add(parent)
        session.exec(
            sa.delete(DeploymentReplicaDB).where(
                DeploymentReplicaDB.deployment_id == "deployment-1"
            )
        )
        session.commit()

    runtime_calls: list[tuple[str, str]] = []

    class Client:
        def load_lora_adapter(self, name, _path):
            runtime_calls.append(("load", name))

        def unload_lora_adapter(self, name):
            runtime_calls.append(("unload", name))

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    adapter = service.load_adapter(
        "deployment-1",
        "legacy-adapter",
        "/app/output/legacy-adapter",
        user_id="user-1",
    )
    assert adapter["deployment_replica_id"] is None
    assert service.unload_adapter(
        "deployment-1",
        "legacy-adapter",
        user_id="user-1",
    )
    assert runtime_calls == [
        ("load", "legacy-adapter"),
        ("unload", "legacy-adapter"),
    ]


def test_legacy_adapter_sync_fences_only_the_exact_parent_claim(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(config_module.select(DeploymentDB)).one()
        parent.config = {"legacy": True}
        parent.replica = 1
        session.add(parent)
        session.exec(sa.delete(DeploymentReplicaDB))
        session.commit()

    runtime_calls: list[str] = []

    class Client:
        @staticmethod
        def list_lora_adapters():
            runtime_calls.append("list")
            return []

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    assert service.sync_loaded_adapters(
        "deployment-1",
        user_id="user-1",
    ) == 0
    assert runtime_calls == ["list"]
    with config_module.Session(engine) as session:
        parent = session.exec(config_module.select(DeploymentDB)).one()
        assert parent.replica_operation_token is None
        assert parent.replica_operation_generation == 1


def test_sglang_embedding_adapter_load_is_supported(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(
            config_module.select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        parent.inference_framework = "sglang"
        session.add(parent)
        session.commit()
    monkeypatch.setattr(
        adapter_module.model_registry_service,
        "get_model",
        lambda _model_id: {"model_type": "embedding"},
    )
    runtime_calls: list[str] = []

    class Client:
        def load_lora_adapter(self, name, _path):
            runtime_calls.append(name)

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    result = service.load_adapter(
        "deployment-1",
        "embedding-adapter",
        "/app/output/embedding-adapter",
        deployment_replica_id="replica-0",
        user_id="user-1",
    )

    assert result["status"] == "loaded"
    assert runtime_calls == ["embedding-adapter"]


def test_degraded_group_allows_load_on_selected_healthy_child(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(
            config_module.select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        parent.status = "degraded"
        failed_child = session.exec(
            config_module.select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        failed_child.status = "failed"
        failed_child.health_status = "UNHEALTHY"
        session.add(parent)
        session.add(failed_child)
        session.commit()

    runtime_calls: list[str] = []

    class Client:
        def load_lora_adapter(self, name, _path):
            runtime_calls.append(name)

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    result = service.load_adapter(
        "deployment-1",
        "healthy-child-adapter",
        "/app/output/healthy-child-adapter",
        deployment_replica_id="replica-1",
        user_id="user-1",
    )

    assert result["status"] == "loaded"
    assert result["deployment_replica_id"] == "replica-1"
    assert runtime_calls == ["healthy-child-adapter"]


@pytest.mark.parametrize(
    "model_type",
    ["rerank", "reranker", "decoder_reranker"],
)
def test_legacy_sglang_reranker_adapter_load_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
    model_type: str,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(
            config_module.select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        parent.inference_framework = "sglang"
        parent.config = {"legacy": True}
        session.add(parent)
        session.exec(
            sa.delete(DeploymentReplicaDB).where(
                DeploymentReplicaDB.deployment_id == "deployment-1"
            )
        )
        session.commit()
    monkeypatch.setattr(
        adapter_module.model_registry_service,
        "get_model",
        lambda _model_id: {"model_type": model_type},
    )
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )

    with pytest.raises(ValueError, match="/v1/rerank"):
        service.load_adapter(
            "deployment-1",
            "reranker-adapter",
            "/app/output/reranker-adapter",
            user_id="user-1",
        )

    assert runtime_calls == []


@pytest.mark.parametrize("lookup_raises", [False, True])
def test_sglang_adapter_load_fails_closed_when_model_type_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
    lookup_raises: bool,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(
            config_module.select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        parent.inference_framework = "sglang"
        parent.config = {"legacy": True}
        session.add(parent)
        session.exec(
            sa.delete(DeploymentReplicaDB).where(
                DeploymentReplicaDB.deployment_id == "deployment-1"
            )
        )
        session.commit()

    def lookup(_model_id):
        if lookup_raises:
            raise RuntimeError("registry unavailable")
        return None

    monkeypatch.setattr(adapter_module.model_registry_service, "get_model", lookup)
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )

    with pytest.raises(ValueError, match="verify SGLang deployment model type"):
        service.load_adapter(
            "deployment-1",
            "adapter",
            "/app/output/adapter",
            user_id="user-1",
        )

    assert runtime_calls == []


def test_adapter_uses_selected_child_endpoint_and_persists_binding(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, _engine = adapter_selection_service
    endpoints: list[str] = []

    class Client:
        def load_lora_adapter(self, _name, _path):
            return None

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda _deployment, *, endpoint=None: endpoints.append(endpoint) or Client(),
    )

    adapter = service.load_adapter(
        "deployment-1",
        "adapter",
        "/app/output/adapter",
        deployment_replica_id="replica-1",
        user_id="user-1",
    )

    assert endpoints == ["http://127.0.0.1:11001"]
    assert adapter["deployment_replica_id"] == "replica-1"
    assert service.list_loaded_adapters(
        "deployment-1",
        deployment_replica_id="replica-1",
        user_id="user-1",
    )[0]["adapter_id"] == adapter["adapter_id"]


def test_adapter_unload_requires_selection_and_only_updates_selected_child(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        session.add_all(
            [
                LoadedAdapterDB(
                    deployment_id="deployment-1",
                    deployment_replica_id=f"replica-{index}",
                    adapter_name="shared-adapter",
                    adapter_path="/app/output/adapter",
                    status="loaded",
                    user_id="user-1",
                )
                for index in range(2)
            ]
        )
        session.commit()

    endpoints: list[str] = []

    class Client:
        def unload_lora_adapter(self, _name):
            return None

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda _deployment, *, endpoint=None: endpoints.append(endpoint) or Client(),
    )

    with pytest.raises(ValueError, match="replica_id is required"):
        service.unload_adapter(
            "deployment-1",
            "shared-adapter",
            user_id="user-1",
        )

    assert service.unload_adapter(
        "deployment-1",
        "shared-adapter",
        deployment_replica_id="replica-1",
        user_id="user-1",
    )
    assert endpoints == ["http://127.0.0.1:11001"]
    assert [
        item["status"]
        for item in service.list_loaded_adapters(
            "deployment-1",
            include_unloaded=True,
            deployment_replica_id="replica-0",
            user_id="user-1",
        )
    ] == ["loaded"]
    assert [
        item["status"]
        for item in service.list_loaded_adapters(
            "deployment-1",
            include_unloaded=True,
            deployment_replica_id="replica-1",
            user_id="user-1",
        )
    ] == ["unloaded"]


def test_adapter_sync_is_scoped_to_selected_child(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        session.add_all(
            [
                LoadedAdapterDB(
                    deployment_id="deployment-1",
                    deployment_replica_id=f"replica-{index}",
                    adapter_name=f"adapter-{index}",
                    adapter_path=f"/app/output/adapter-{index}",
                    status="loaded",
                    user_id="user-1",
                )
                for index in range(2)
            ]
        )
        session.commit()

    endpoints: list[str] = []

    class Client:
        def list_lora_adapters(self):
            return []

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda _deployment, *, endpoint=None: endpoints.append(endpoint) or Client(),
    )

    with pytest.raises(ValueError, match="replica_id is required"):
        service.sync_loaded_adapters("deployment-1", user_id="user-1")

    assert service.sync_loaded_adapters(
        "deployment-1",
        deployment_replica_id="replica-1",
        user_id="user-1",
    ) == 1
    assert endpoints == ["http://127.0.0.1:11001"]
    assert service.list_loaded_adapters(
        "deployment-1",
        deployment_replica_id="replica-0",
        user_id="user-1",
    )[0]["status"] == "loaded"
    assert service.list_loaded_adapters(
        "deployment-1",
        include_unloaded=True,
        deployment_replica_id="replica-1",
        user_id="user-1",
    )[0]["status"] == "unloaded"


def test_adapter_sync_allows_degraded_group_selected_healthy_child(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(
            config_module.select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        parent.status = "degraded"
        failed_child = session.exec(
            config_module.select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        failed_child.status = "failed"
        failed_child.health_status = "UNHEALTHY"
        session.add(parent)
        session.add(failed_child)
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                deployment_replica_id="replica-1",
                adapter_name="healthy-child-adapter",
                adapter_path="/app/output/healthy-child-adapter",
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()

    endpoints: list[str] = []

    class Client:
        def list_lora_adapters(self):
            return []

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda _deployment, *, endpoint=None: endpoints.append(endpoint) or Client(),
    )

    assert service.sync_loaded_adapters(
        "deployment-1",
        deployment_replica_id="replica-1",
        user_id="user-1",
    ) == 1
    assert endpoints == ["http://127.0.0.1:11001"]
    with config_module.Session(engine) as session:
        adapter = session.exec(config_module.select(LoadedAdapterDB)).one()
        assert adapter.status == "unloaded"


def test_adapter_sync_rejects_degraded_group_unhealthy_child_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(
            config_module.select(DeploymentDB).where(
                DeploymentDB.deployment_id == "deployment-1"
            )
        ).one()
        parent.status = "degraded"
        selected_child = session.exec(
            config_module.select(DeploymentReplicaDB).where(
                DeploymentReplicaDB.replica_id == "replica-0"
            )
        ).one()
        selected_child.status = "failed"
        selected_child.health_status = "UNHEALTHY"
        session.add(parent)
        session.add(selected_child)
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name="must-stay-loaded",
                adapter_path="/app/output/must-stay-loaded",
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()

    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )

    with pytest.raises(ValueError, match="not running and healthy"):
        service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    assert runtime_calls == []
    with config_module.Session(engine) as session:
        adapter = session.exec(config_module.select(LoadedAdapterDB)).one()
        assert adapter.status == "loaded"


def test_adapter_sync_keeps_legacy_degraded_parent_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(config_module.select(DeploymentDB)).one()
        parent.config = {"legacy": True}
        parent.status = "degraded"
        session.add(parent)
        session.exec(sa.delete(DeploymentReplicaDB))
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                adapter_name="legacy-must-stay-loaded",
                adapter_path="/app/output/legacy-must-stay-loaded",
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()

    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )

    assert service.sync_loaded_adapters(
        "deployment-1",
        user_id="user-1",
    ) == 0
    assert runtime_calls == []
    with config_module.Session(engine) as session:
        assert session.exec(config_module.select(LoadedAdapterDB)).one().status == "loaded"


def test_adapter_routes_forward_replica_and_user_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replica_id = UUID("11111111-1111-4111-8111-111111111111")
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: {
            "deployment_id": deployment_id,
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "unload_adapter",
        lambda *args, **kwargs: calls.append(("unload", args, kwargs)) or True,
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "list_loaded_adapters",
        lambda *args, **kwargs: calls.append(("list", args, kwargs)) or [],
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "sync_loaded_adapters",
        lambda *args, **kwargs: calls.append(("sync", args, kwargs)) or 0,
    )
    current_user = {"user_id": "user-1"}

    asyncio.run(
        adapter_routes.unload_adapter(
            "deployment-1",
            "adapter",
            replica_id=replica_id,
            current_user=current_user,
        )
    )
    asyncio.run(
        adapter_routes.list_loaded_adapters(
            "deployment-1",
            include_unloaded=False,
            replica_id=replica_id,
            current_user=current_user,
        )
    )
    asyncio.run(
        adapter_routes.sync_loaded_adapters(
            "deployment-1",
            replica_id=replica_id,
            current_user=current_user,
        )
    )

    expected_binding = {
        "deployment_replica_id": str(replica_id),
        "user_id": "user-1",
    }
    assert calls == [
        ("unload", ("deployment-1", "adapter"), expected_binding),
        (
            "list",
            ("deployment-1", False),
            expected_binding,
        ),
        ("sync", ("deployment-1",), expected_binding),
    ]


@pytest.mark.parametrize("endpoint", ["load", "from-task", "unload"])
def test_adapter_mutation_routes_map_replica_not_found_to_404(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    error = deployment_service_module.DeploymentReplicaNotFoundError(
        "deployment replica not found"
    )
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: {
            "deployment_id": deployment_id,
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "load_adapter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "unload_adapter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_direct_adapter_path",
        lambda *_args, **_kwargs: "/app/output/adapter",
    )
    monkeypatch.setattr(
        adapter_routes,
        "_resolve_owned_task_adapter_path",
        lambda *_args, **_kwargs: "/app/output/task/final_model",
    )
    monkeypatch.setattr(
        adapter_routes,
        "_begin_training_artifact_guard",
        lambda *_args, **_kwargs: SimpleNamespace(release=lambda: None),
    )

    if endpoint == "load":
        call = adapter_routes.load_adapter(
            "deployment-1",
            adapter_routes.LoadAdapterRequest(
                adapter_name="adapter",
                adapter_path="/app/output/adapter",
            ),
            {"user_id": "user-1"},
        )
    elif endpoint == "from-task":
        call = adapter_routes.load_adapter_from_task(
            "deployment-1",
            adapter_routes.LoadAdapterFromTaskRequest(
                task_id="task-1",
                adapter_name="adapter",
            ),
            {"user_id": "user-1"},
        )
    else:
        call = adapter_routes.unload_adapter(
            "deployment-1",
            "adapter",
            replica_id=None,
            current_user={"user_id": "user-1"},
        )

    with pytest.raises(adapter_routes.HTTPException) as caught:
        asyncio.run(call)

    assert caught.value.status_code == 404
    assert caught.value.detail == str(error)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ValueError("replica_id is required"), 400),
        (ValueError("deployment replica is not running and healthy"), 400),
        (
            deployment_service_module.DeploymentReplicaNotFoundError(
                "deployment replica not found"
            ),
            404,
        ),
        (
            deployment_service_module.ReplicaOperationBusyError("busy"),
            409,
        ),
        (
            deployment_service_module.ReplicaOperationLostError("lost"),
            409,
        ),
    ],
)
def test_adapter_list_route_maps_replica_selection_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: {
            "deployment_id": deployment_id,
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "list_loaded_adapters",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(adapter_routes.HTTPException) as caught:
        asyncio.run(
            adapter_routes.list_loaded_adapters(
                "deployment-1",
                include_unloaded=False,
                replica_id=None,
                current_user={"user_id": "user-1"},
            )
        )

    assert caught.value.status_code == expected_status
    assert caught.value.detail == str(error)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            deployment_service_module.DeploymentReplicaNotFoundError(
                "replica disappeared"
            ),
            "replica disappeared",
        ),
        (
            deployment_service_module.ReplicaOperationBusyError(
                "replica lifecycle is busy"
            ),
            "replica lifecycle is busy",
        ),
    ],
)
def test_adapter_sync_claim_errors_propagate_by_default(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
    error: Exception,
    message: str,
) -> None:
    service, _engine = adapter_selection_service
    monkeypatch.setattr(
        service,
        "_claim_adapter_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error), match=message):
        service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )


def test_adapter_background_sync_can_opt_into_best_effort(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, _engine = adapter_selection_service
    monkeypatch.setattr(
        service,
        "_claim_adapter_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            deployment_service_module.ReplicaOperationBusyError(
                "replica lifecycle is busy"
            )
        ),
    )

    assert (
        service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
            best_effort=True,
        )
        == 0
    )


def test_adapter_sync_missing_parent_raises_not_found_by_default(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(config_module.select(DeploymentDB)).one()
        session.delete(parent)
        session.commit()

    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )

    with pytest.raises(
        deployment_service_module.DeploymentReplicaNotFoundError,
        match="deployment not found",
    ):
        service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    assert runtime_calls == []


def test_adapter_background_sync_missing_parent_can_be_best_effort(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(config_module.select(DeploymentDB)).one()
        session.delete(parent)
        session.commit()

    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )

    assert (
        service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
            best_effort=True,
        )
        == 0
    )
    assert runtime_calls == []


@pytest.mark.parametrize(
    ("drift", "best_effort"),
    [
        ("deleted", False),
        ("deleted", True),
        ("status", False),
        ("status", True),
        ("schema", False),
        ("schema", True),
        ("replacement-owner", False),
        ("replacement-owner", True),
    ],
)
def test_adapter_sync_fails_closed_when_parent_drifts_after_claim(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
    drift: str,
    best_effort: bool,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name="must-not-change",
                adapter_path="/app/output/must-not-change",
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()

    original_claim = service._claim_adapter_operation
    claimed_generation: list[int] = []

    def claim_then_drift(*args, **kwargs):
        claim = original_claim(*args, **kwargs)
        claimed_generation.append(claim.generation)
        with config_module.Session(engine) as session:
            parent = session.exec(config_module.select(DeploymentDB)).one()
            if drift == "deleted":
                session.delete(parent)
            elif drift == "status":
                parent.status = "stopped"
                session.add(parent)
            elif drift == "schema":
                parent.config = {}
                session.add(parent)
            else:
                parent.replica_operation_token = "replacement-owner-token"
                parent.replica_operation_kind = "stop"
                parent.replica_operation_replica_id = "replica-0"
                parent.replica_operation_generation = claim.generation + 1
                session.add(parent)
            session.commit()
        return claim

    monkeypatch.setattr(service, "_claim_adapter_operation", claim_then_drift)
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )

    if best_effort:
        assert (
            service.sync_loaded_adapters(
                "deployment-1",
                deployment_replica_id="replica-0",
                user_id="user-1",
                best_effort=True,
            )
            == 0
        )
    else:
        with pytest.raises(
            deployment_service_module.ReplicaOperationLostError,
            match="ownership was lost|deleted or changed|lifecycle changed",
        ):
            service.sync_loaded_adapters(
                "deployment-1",
                deployment_replica_id="replica-0",
                user_id="user-1",
            )

    assert claimed_generation == [1]
    assert runtime_calls == []
    with config_module.Session(engine) as session:
        adapter = session.exec(config_module.select(LoadedAdapterDB)).one()
        assert adapter.status == "loaded"
        assert adapter.error_message is None
        parent = session.exec(config_module.select(DeploymentDB)).first()
        if drift == "deleted":
            assert parent is None
        elif drift == "replacement-owner":
            assert parent is not None
            assert parent.replica_operation_token == "replacement-owner-token"
            assert parent.replica_operation_kind == "stop"
            assert parent.replica_operation_replica_id == "replica-0"
            assert parent.replica_operation_generation == 2
        else:
            assert parent is not None
            assert parent.replica_operation_token is None
            assert parent.replica_operation_kind is None
            assert parent.replica_operation_replica_id is None
            assert parent.replica_operation_generation == 1


def test_adapter_sync_rejects_schema_drift_after_runtime_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        session.add(
            LoadedAdapterDB(
                deployment_id="deployment-1",
                deployment_replica_id="replica-0",
                adapter_name="must-not-change",
                adapter_path="/app/output/must-not-change",
                status="loaded",
                user_id="user-1",
            )
        )
        session.commit()

    runtime_calls: list[str] = []

    class Client:
        @staticmethod
        def list_lora_adapters():
            runtime_calls.append("list")
            with config_module.Session(engine) as session:
                parent = session.exec(config_module.select(DeploymentDB)).one()
                parent.config = {}
                session.add(parent)
                session.commit()
            return []

    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: Client(),
    )

    with pytest.raises(
        deployment_service_module.ReplicaOperationLostError,
        match="lifecycle changed",
    ):
        service.sync_loaded_adapters(
            "deployment-1",
            deployment_replica_id="replica-0",
            user_id="user-1",
        )

    assert runtime_calls == ["list"]
    with config_module.Session(engine) as session:
        adapter = session.exec(config_module.select(LoadedAdapterDB)).one()
        assert adapter.status == "loaded"
        assert adapter.error_message is None
        parent = session.exec(config_module.select(DeploymentDB)).one()
        assert parent.config == {}
        assert parent.replica_operation_token is None
        assert parent.replica_operation_generation == 1


def test_adapter_sync_route_maps_parent_deleted_after_ownership_check_to_404(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
) -> None:
    service, engine = adapter_selection_service
    with config_module.Session(engine) as session:
        parent = session.exec(config_module.select(DeploymentDB)).one()
        ownership_snapshot = {
            "deployment_id": parent.deployment_id,
            "user_id": parent.user_id,
        }
        session.delete(parent)
        session.commit()

    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )
    monkeypatch.setattr(adapter_routes, "adapter_service", service)
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda _deployment_id: ownership_snapshot,
    )

    with pytest.raises(adapter_routes.HTTPException) as caught:
        asyncio.run(
            adapter_routes.sync_loaded_adapters(
                "deployment-1",
                replica_id=UUID("00000000-0000-0000-0000-000000000001"),
                current_user={"user_id": "user-1"},
            )
        )

    assert caught.value.status_code == 404
    assert caught.value.detail == "deployment not found"
    assert runtime_calls == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            deployment_service_module.DeploymentReplicaNotFoundError(
                "replica disappeared"
            ),
            404,
        ),
        (
            deployment_service_module.ReplicaOperationBusyError(
                "replica lifecycle is busy"
            ),
            409,
        ),
        (
            deployment_service_module.ReplicaOperationLostError(
                "replica lifecycle ownership was lost"
            ),
            409,
        ),
        (ValueError("replica_id is required"), 400),
    ],
)
def test_adapter_sync_route_maps_lifecycle_claim_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: {
            "deployment_id": deployment_id,
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        adapter_routes.adapter_service,
        "sync_loaded_adapters",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(adapter_routes.HTTPException) as caught:
        asyncio.run(
            adapter_routes.sync_loaded_adapters(
                "deployment-1",
                replica_id=None,
                current_user={"user_id": "user-1"},
            )
        )

    assert caught.value.status_code == expected_status
    assert caught.value.detail == str(error)


@pytest.mark.parametrize(
    ("replica_id", "unhealthy", "message"),
    [
        (None, False, "replica_id is required"),
        ("replica-0", True, "not running and healthy"),
    ],
)
def test_adapter_sync_route_maps_invalid_replica_selection_to_bad_request(
    monkeypatch: pytest.MonkeyPatch,
    adapter_selection_service,
    replica_id,
    unhealthy: bool,
    message: str,
) -> None:
    service, engine = adapter_selection_service
    if unhealthy:
        with config_module.Session(engine) as session:
            replica = session.exec(
                config_module.select(DeploymentReplicaDB).where(
                    DeploymentReplicaDB.replica_id == "replica-0"
                )
            ).one()
            replica.status = "failed"
            replica.health_status = "UNHEALTHY"
            session.add(replica)
            session.commit()

    runtime_calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_get_inference_client",
        lambda *_args, **_kwargs: runtime_calls.append("client"),
    )
    monkeypatch.setattr(adapter_routes, "adapter_service", service)
    monkeypatch.setattr(
        adapter_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: {
            "deployment_id": deployment_id,
            "user_id": "user-1",
        },
    )

    with pytest.raises(adapter_routes.HTTPException) as caught:
        asyncio.run(
            adapter_routes.sync_loaded_adapters(
                "deployment-1",
                replica_id=replica_id,
                current_user={"user_id": "user-1"},
            )
        )

    assert caught.value.status_code == 400
    assert message in caught.value.detail
    assert runtime_calls == []
    with config_module.Session(engine) as session:
        parent = session.exec(config_module.select(DeploymentDB)).one()
        assert parent.replica_operation_token is None


def test_evaluation_deployment_source_uses_trusted_replica_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    empty_inference_catalog,
) -> None:
    selections: list[tuple[str, str | None, str | None, bool]] = []

    def resolve(deployment_id, replica_id, *, user_id, require_healthy):
        selections.append((deployment_id, replica_id, user_id, require_healthy))
        return (
            {
                "deployment_id": deployment_id,
                "inference_framework": "vllm",
                "model_uid": "served-model",
            },
            _child(1),
        )

    monkeypatch.setattr(
        evaluation_routes_module.deployment_service,
        "resolve_replica_selection",
        resolve,
    )
    monkeypatch.setattr(
        evaluation_routes_module,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )

    configs = evaluation_routes_module._validate_model_configs(
        [
            {
                "deployment_id": "deployment-1",
                "deployment_replica_id": "replica-1",
                "endpoint": "https://attacker.invalid",
                "model_name": "spoofed-model",
                "name": "selected replica",
            }
        ],
        "user-1",
    )

    assert selections == [("deployment-1", "replica-1", "user-1", True)]
    assert configs[0]["endpoint"] == "http://127.0.0.1:11001"
    assert configs[0]["model_name"] == "served-model"
    assert configs[0]["inference_framework"] == "vllm"


def test_evaluation_legacy_deployment_uses_trusted_parent_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    empty_inference_catalog,
) -> None:
    from train_factory.auth.user_service import user_service

    with config_module.Session(empty_inference_catalog) as session:
        session.add(DeploymentDB(
            deployment_id="deployment-legacy", model_id="model-1",
            xinference_endpoint="http://xinference:9997",
            inference_framework="xinference", model_uid="trusted-model-uid",
            user_id="user-1", status="running",
        ))
        session.commit()
    monkeypatch.setattr(
        user_service, "get_user",
        lambda user_id: {"user_id": user_id, "is_active": True, "is_admin": False},
    )
    selections: list[tuple[str, str | None, str | None, bool]] = []

    def resolve(deployment_id, replica_id, *, user_id, require_healthy):
        selections.append((deployment_id, replica_id, user_id, require_healthy))
        return (
            {
                "deployment_id": deployment_id,
                "xinference_endpoint": "http://xinference:9997",
                "inference_framework": "xinference",
                "model_uid": "trusted-model-uid",
            },
            None,
        )

    monkeypatch.setattr(
        evaluation_routes_module.deployment_service,
        "resolve_replica_selection",
        resolve,
    )

    configs = evaluation_routes_module._validate_model_configs(
        [
            {
                "deployment_id": "deployment-legacy",
                "endpoint": "https://attacker.invalid",
                "model_name": "spoofed-model",
                "inference_framework": "sglang",
                "name": "legacy deployment",
            }
        ],
        "user-1",
    )

    assert selections == [("deployment-legacy", None, "user-1", True)]
    assert configs[0]["endpoint"] == "http://xinference:9997"
    assert configs[0]["model_name"] == "trusted-model-uid"
    assert configs[0]["inference_framework"] == "xinference"
    assert configs[0].get("deployment_replica_id") is None


def test_evaluation_legacy_deployment_rejects_explicit_missing_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(_deployment_id, replica_id, *, user_id, require_healthy):
        assert replica_id == "missing-replica"
        assert user_id == "user-1"
        assert require_healthy is True
        raise ValueError("deployment replica not found")

    monkeypatch.setattr(
        evaluation_routes_module.deployment_service,
        "resolve_replica_selection",
        resolve,
    )

    with pytest.raises(ValueError, match="deployment replica not found"):
        evaluation_routes_module._validate_model_configs(
            [
                {
                    "deployment_id": "deployment-legacy",
                    "deployment_replica_id": "missing-replica",
                    "endpoint": "https://attacker.invalid",
                    "name": "legacy deployment",
                }
            ],
            "user-1",
        )


def test_sync_deployment_reference_requires_and_forwards_replica_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None, str | None, bool]] = []

    def resolve(deployment_id, replica_id, *, user_id, require_healthy):
        calls.append((deployment_id, replica_id, user_id, require_healthy))
        if replica_id is None:
            raise ValueError("replica_id is required for multi-replica deployment")
        return _parent([_child(0), _child(1)]), _child(1)

    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "resolve_replica_selection",
        resolve,
    )
    current_user = {"user_id": "user-1"}

    with pytest.raises(
        sync_routes_module.HTTPException,
        match="replica_id is required",
    ):
        sync_routes_module._validate_base_deployment_reference(
            "deployment-1",
            current_user,
        )

    deployment = sync_routes_module._validate_base_deployment_reference(
        "deployment-1",
        current_user,
        base_deployment_replica_id="replica-1",
    )
    assert deployment is not None
    assert deployment["deployment_id"] == "deployment-1"
    assert calls == [
        ("deployment-1", None, "user-1", True),
        ("deployment-1", "replica-1", "user-1", True),
    ]


def test_auth_disabled_sync_reference_queries_ownerless_replica_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def resolve(deployment_id, replica_id, *, user_id, require_healthy):
        calls.append((deployment_id, replica_id, user_id, require_healthy))
        return (
            {
                **_parent([_child(0)]),
                "user_id": None,
            },
            _child(0),
        )

    monkeypatch.setattr(
        sync_routes_module,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=False),
    )
    monkeypatch.setattr(
        deployment_service_module.deployment_service,
        "resolve_replica_selection",
        resolve,
    )
    monkeypatch.setattr(
        sync_routes_module,
        "verify_resource_ownership",
        lambda resource, *_args: resource,
    )

    deployment = sync_routes_module._validate_base_deployment_reference(
        "deployment-1",
        {"user_id": ""},
        base_deployment_replica_id="replica-0",
    )

    assert deployment is not None
    assert calls == [("deployment-1", "replica-0", None, True)]
