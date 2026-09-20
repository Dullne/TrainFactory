"""Registered deployment identities survive inline configs and host aliases."""

import asyncio
import importlib
from contextlib import contextmanager

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from train_factory.auth.user_service import user_service
from train_factory.config.settings import get_settings
from train_factory.deployment.deployment_service import deployment_service
from train_factory.generation.clients.llm_client import LLMClient, LLMConfig
from train_factory.generation.clients.embedding_client import EmbeddingClient, EmbeddingConfig
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.deployment_replica_entity import DeploymentReplicaDB
from train_factory.storage.services import inference_authorization_service as authorization
from train_factory.storage.services import outbound_endpoint_policy as outbound


ALICE = {"user_id": "alice", "is_admin": False, "is_active": True}


@pytest.fixture
def catalog(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    DeploymentDB.__table__.create(engine)
    DeploymentReplicaDB.__table__.create(engine)

    @contextmanager
    def session():
        with Session(engine) as value:
            yield value

    monkeypatch.setattr(authorization, "get_session", session)
    monkeypatch.setattr(outbound, "get_session", session)
    monkeypatch.setattr(get_settings(), "auth_enabled", True)
    monkeypatch.setattr(get_settings(), "xinference_endpoint", "http://10.2.3.4:9997")
    monkeypatch.setattr(deployment_service, "_get_host_ip", lambda: "10.2.3.4")
    monkeypatch.setattr(user_service, "get_user", lambda uid: dict(ALICE, user_id=uid))
    yield engine
    engine.dispose()


def add_deployment(engine, *, framework="xinference", owner="bob", endpoint="http://10.2.3.4:18123", replicas=False, mode="container"):
    with Session(engine) as session:
        session.add(DeploymentDB(
            deployment_id="deployment-1", model_id="registry-1", user_id=owner,
            model_uid="deployed-model", status="running", deploy_mode=mode,
            inference_framework=framework, xinference_endpoint=endpoint,
            port=18123 if mode == "container" else None,
            config={"replica_schema_version": 1} if replicas else {},
        ))
        if replicas:
            session.add(DeploymentReplicaDB(
                deployment_id="deployment-1", replica_id="replica-1", replica_index=0,
                container_name="vllm-replica", endpoint="http://vllm-replica:18124", port=18124,
                status="running", health_status="HEALTHY",
            ))
        session.commit()


@pytest.mark.parametrize("framework", ["xinference", "vllm", "sglang"])
@pytest.mark.parametrize("provider", ["", "custom"])
def test_foreign_container_is_protected_even_when_http_host_is_allowed(catalog, framework, provider):
    add_deployment(catalog, framework=framework)
    endpoint = outbound.validate_user_outbound_url("http://10.2.3.4:18123/v1", "alice")
    with pytest.raises(HTTPException) as exc:
        authorization.authorize_inference_config(
            {"endpoint": endpoint, "model": "deployed-model", "provider": provider},
            "alice", current_user=ALICE,
        )
    assert exc.value.status_code == 403


def test_background_client_rejects_foreign_container_before_sending(catalog):
    add_deployment(catalog)
    requests = []

    async def run():
        async def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "accepted"}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = LLMClient(LLMConfig(endpoint="http://10.2.3.4:18123/v1", model="deployed-model", user_id="alice"))
            client._client = http
            with pytest.raises(HTTPException) as exc:
                await client.chat("hello")
            assert exc.value.status_code == 403
        assert requests == []

    asyncio.run(run())


def test_default_shared_host_alias_stays_protected_without_registered_models(catalog, monkeypatch):
    monkeypatch.setattr(get_settings(), "xinference_endpoint", "http://xinference:9997")
    with pytest.raises(HTTPException) as exc:
        authorization.authorize_inference_model("http://10.2.3.4:9997/v1", "unregistered-model", "alice")
    assert exc.value.status_code == 403


@pytest.mark.parametrize("attempt_failed", [False, True])
def test_unstarted_shared_deployment_cannot_claim_an_external_api(catalog, monkeypatch, attempt_failed):
    from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
    from train_factory.storage.entities.model_artifact_membership_gate_entity import ModelArtifactMembershipGateDB

    for entity in (ModelRegistryDB, ModelArtifactMembershipGateDB):
        entity.__table__.create(catalog)
    with Session(catalog) as session:
        session.add(ModelRegistryDB(model_id="registry-1", model_name="model", model_type="embedding", model_path="/models/test", user_id="bob"))
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()
    for module_name in ("train_factory.deployment.deployment_service", "train_factory.storage.services.model_registry_service"):
        monkeypatch.setattr(importlib.import_module(module_name), "get_session", authorization.get_session)
    external = "https://api.example.test/v1"
    assert authorization.authorize_inference_model(external, "external-model", "alice") is None
    deployment_service.create_deployment(
        model_id="registry-1", xinference_endpoint=external, user_id="bob", gpu_memory_utilization=0.5,
        config={"runtime_managed": False, "read_only": True, "started_at": "2026-01-01"},
    )
    with Session(catalog) as session:
        row = session.get(DeploymentDB, 1)
        assert row.started_at is None
        assert "runtime_managed" not in row.config
        assert "read_only" not in row.config
        if attempt_failed:
            row.status = "failed"
            row.model_uid = "generated-model-12345678"
            session.add(row)
            session.commit()
    assert authorization.authorize_inference_model(external, "external-model", "alice") is None


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_previously_started_shared_service_stays_protected(catalog, status):
    from train_factory.core.time_utils import now_naive

    endpoint = "https://private-serving.example.test/v1"
    add_deployment(catalog, endpoint=endpoint, mode="shared")
    with Session(catalog) as session:
        row = session.get(DeploymentDB, 1)
        row.status = status
        row.started_at = now_naive()
        session.add(row)
        session.commit()
    for owner in ("alice", "bob"):
        with pytest.raises(HTTPException):
            authorization.authorize_inference_model(endpoint, "deployed-model", owner)


def test_user_shared_endpoint_cannot_allow_other_ports_on_the_host_alias(catalog, monkeypatch):
    from train_factory.core.ssrf import SSRFError

    monkeypatch.setattr(get_settings(), "xinference_endpoint", "http://xinference:9997")
    add_deployment(catalog, endpoint="http://xinference:18125", mode="shared", owner="alice")
    with pytest.raises(SSRFError):
        outbound.validate_user_outbound_url("http://10.2.3.4:18125", "alice")


@pytest.mark.parametrize("internal", ["http://xinference:9997", "http://xf-container:18123"])
def test_generated_xinference_alias_allows_owner_and_rejects_other_users(catalog, monkeypatch, internal):
    add_deployment(catalog, owner="alice", endpoint=internal,
                   mode="shared" if internal == "http://xinference:9997" else "container")
    monkeypatch.setattr(get_settings(), "xinference_endpoint", "http://default-xinference:9997")
    monkeypatch.setattr(deployment_service, "get_deployment", lambda _: {
        "deployment_id": "deployment-1", "user_id": "alice", "status": "running",
        "model_uid": "deployed-model", "xinference_endpoint": internal,
        "inference_framework": "xinference",
    })
    endpoint = deployment_service._normalize_endpoint_for_config(internal)
    config = {"api_endpoint": endpoint, "model_name": "deployed-model", "provider": "xinference",
              "source_type": "local_deployed", "deployment_id": "deployment-1"}
    assert authorization.authorize_inference_config(config, "alice", current_user=ALICE) == {"deployed-model"}
    assert authorization.authorize_inference_model(endpoint, "deployed-model", "alice") == {"deployed-model"}
    with pytest.raises(HTTPException) as exc:
        authorization.authorize_inference_config(
            {"model": "deployed-model", "endpoints": [{"url": endpoint}]}, "bob"
        )
    assert exc.value.status_code == 403


@pytest.mark.parametrize("alias", [False, True])
def test_replica_endpoint_authorization_uses_child_identity_and_health(catalog, alias):
    add_deployment(catalog, framework="vllm", owner="alice", replicas=True)
    endpoint = "http://10.2.3.4:18124/v1" if alias else "http://vllm-replica:18124/v1"
    assert authorization.authorize_inference_model(endpoint, "deployed-model", "alice") == {"deployed-model"}
    with pytest.raises(HTTPException):
        authorization.authorize_inference_model(endpoint, "deployed-model", "bob")
    with Session(catalog) as session:
        replica = session.get(DeploymentReplicaDB, 1)
        replica.health_status = "UNHEALTHY"
        session.add(replica)
        session.commit()
    with pytest.raises(HTTPException):
        authorization.authorize_inference_model(endpoint, "deployed-model", "alice")


def test_registered_container_preserves_owner_admin_disabled_auth_and_external_api(catalog, monkeypatch):
    add_deployment(catalog, owner="alice")
    endpoint = "http://10.2.3.4:18123/v1"
    assert authorization.authorize_inference_model(endpoint, "deployed-model", "alice") == {"deployed-model"}
    with pytest.raises(HTTPException):
        authorization.authorize_inference_model(endpoint, "another-model", "alice")
    assert authorization.authorize_inference_model(endpoint, "another-model", "admin", current_user={"is_admin": True}) is None
    assert authorization.authorize_inference_model("https://api.example.test/v1", "external-model", "alice") is None
    with Session(catalog) as session:
        deployment = session.get(DeploymentDB, 1)
        deployment.status = "stopped"
        session.add(deployment)
        session.commit()
    with pytest.raises(HTTPException):
        authorization.authorize_inference_model(endpoint, "deployed-model", "alice")
    monkeypatch.setattr(get_settings(), "auth_enabled", False)
    assert authorization.authorize_inference_model(endpoint, "another-model", "bob") is None


@pytest.mark.parametrize("stored_endpoint,mode,expected_port", [
    ("http://xf-container:18123", "container", 18123),
    ("http://10.2.3.4:8000", "container", 18123),
    ("https://public.example.test:8443/prefix", "container", 18123),
    ("http://xinference:9997", "shared", 9997),
])
def test_real_auto_created_config_reaches_http_only_at_registered_host_port(catalog, monkeypatch, stored_endpoint, mode, expected_port):
    from train_factory.core.ssrf import SSRFError
    from train_factory.storage.entities.model_config_entity import ModelConfigDB
    from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
    from train_factory.storage.entities.model_artifact_membership_gate_entity import ModelArtifactMembershipGateDB
    from train_factory.storage.services.model_config_service import model_config_service

    for entity in (ModelConfigDB, ModelRegistryDB, ModelArtifactMembershipGateDB):
        entity.__table__.create(catalog)
    with Session(catalog) as session:
        session.add(ModelRegistryDB(model_id="registry-1", model_name="model", model_type="embedding", model_path="/models/test", user_id="alice"))
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()
    add_deployment(catalog, owner="alice", endpoint=stored_endpoint, mode=mode)
    monkeypatch.setenv("HOST_IP", "10.2.3.4")
    monkeypatch.setattr(get_settings(), "xinference_endpoint", "http://xinference:9997")
    monkeypatch.setattr(model_config_service, "_get_engine", lambda: catalog)
    deployment_module = importlib.import_module("train_factory.deployment.deployment_service")
    monkeypatch.setattr(deployment_module, "get_session", authorization.get_session)
    with Session(catalog) as session:
        deployment = session.get(DeploymentDB, 1)
        deployment_service._create_config_for_deployment(deployment, {
            "model_id": "registry-1", "model_name": "model", "model_type": "embedding",
        })
    config = model_config_service.get_config_by_deployment_id("deployment-1")
    assert config is not None
    assert config["api_endpoint"] == f"http://10.2.3.4:{expected_port}"
    assert authorization.authorize_inference_config(config, "alice", current_user=ALICE) == {"deployed-model"}
    assert outbound.validate_user_outbound_url(config["api_endpoint"], "alice") == f"http://10.2.3.4:{expected_port}"
    requests = []

    async def respond(request):
        requests.append((str(request.url), request.headers["host"]))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **_: httpx.MockTransport(respond))

    async def run():
        async with EmbeddingClient(EmbeddingConfig(endpoint=config["api_endpoint"], model=config["model_name"], user_id="alice")) as client:
            assert (await client.embed("question")).tolist() == [[1.0, 0.0]]

    asyncio.run(run())
    assert requests == [(f"http://10.2.3.4:{expected_port}/v1/embeddings", f"10.2.3.4:{expected_port}")]
    with pytest.raises(HTTPException):
        authorization.authorize_inference_model(config["api_endpoint"], config["model_name"], "bob")
    tampered = dict(config, api_endpoint="http://10.2.3.4:18125")
    with pytest.raises(HTTPException):
        authorization.authorize_inference_config(tampered, "alice", current_user=ALICE)
    with pytest.raises(SSRFError):
        outbound.validate_user_outbound_url(tampered["api_endpoint"], "alice")
