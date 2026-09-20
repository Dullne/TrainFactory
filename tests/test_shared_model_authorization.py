"""Shared inference authorization must use the same endpoint as the request."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from sqlmodel import Session, create_engine
from sqlalchemy.pool import StaticPool

from train_factory.api.routes import model_config_routes as routes
from train_factory.storage.entities.deployment_entity import DeploymentDB


USER = {"user_id": "alice", "is_admin": False}
SHARED = "http://xinference:9997"


@pytest.fixture
def shared_policy(monkeypatch):
    from train_factory.storage.services import inference_authorization_service

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    DeploymentDB.__table__.create(engine)
    with Session(engine) as session:
        session.add(DeploymentDB(
            model_id="alice-registry", user_id="alice", model_uid="alice-model",
            status="running", inference_framework="xinference", deploy_mode="shared",
            xinference_endpoint=SHARED,
        ))
        session.commit()

    @contextmanager
    def session_factory():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(inference_authorization_service, "get_session", session_factory)
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(auth_enabled=True))
    monkeypatch.setattr(routes, "get_default_xinference_endpoint", lambda: SHARED)
    monkeypatch.setattr(routes, "validate_user_outbound_url", lambda endpoint, *_: endpoint)
    monkeypatch.setattr(
        routes.deployment_service,
        "list_deployments",
        lambda **_: (
            [
                {
                    "model_uid": "alice-model",
                    "xinference_endpoint": SHARED,
                    "inference_framework": "xinference",
                    "status": "running",
                    "user_id": "alice",
                }
            ],
            1,
        ),
    )
    config = dict(
        config_id="config-1",
        config_name="test",
        model_type="llm",
        provider="custom",
        api_endpoint=SHARED + "/v1",
        model_name="alice-model",
        user_id="alice",
        status="active",
        is_default=False,
    )
    monkeypatch.setattr(routes.model_config_service, "get_config", lambda *_: dict(config))
    monkeypatch.setattr(routes.model_config_service, "update_check_status", lambda *_: None)
    monkeypatch.setattr(routes, "_config_to_response", lambda value: SimpleNamespace(**value))
    yield config
    engine.dispose()


@pytest.mark.parametrize(
    "suffix", ["/v1/models", "/v1/embeddings/", "/chat/completions", "/v1/completions", "/v1/rerank"]
)
def test_shared_alias_validation_rejects_unowned_model(monkeypatch, shared_policy, suffix):
    async def validate(**_):
        return {"valid": True, "models": ["bob-model"]}

    monkeypatch.setattr(routes.model_config_service, "validate_config_async", validate)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes.validate_config(
                routes.ValidateConfigRequest(provider="custom", api_endpoint=SHARED + suffix, model_name="bob-model"),
                USER,
            )
        )
    assert exc_info.value.status_code == 403


def test_malformed_endpoint_returns_client_error(monkeypatch, shared_policy):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes.validate_config(
                routes.ValidateConfigRequest(provider="custom", api_endpoint="http://", model_name="alice-model"), USER
            )
        )
    assert exc_info.value.status_code == 400


def test_shared_alias_without_any_owned_deployments_is_blocked(monkeypatch, shared_policy):
    monkeypatch.setattr(routes.deployment_service, "list_deployments", lambda **_: ([], 0))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes.validate_config(
                routes.ValidateConfigRequest(
                    provider="custom", api_endpoint=SHARED + "/v1/models", model_name="bob-model"
                ),
                USER,
            )
        )
    assert exc_info.value.status_code == 403


def test_owned_shared_alias_validation_filters_model_list(monkeypatch, shared_policy):
    async def validate(**_):
        return {"valid": True, "models": ["alice-model", "bob-model"]}

    monkeypatch.setattr(routes.model_config_service, "validate_config_async", validate)
    result = asyncio.run(
        routes.validate_config(
            routes.ValidateConfigRequest(
                provider="custom", api_endpoint=SHARED + "/v1/models", model_name="alice-model"
            ),
            USER,
        )
    )
    assert result.models == ["alice-model"]


@pytest.mark.parametrize("validate_api", [False, True])
def test_create_cannot_save_unowned_shared_model(monkeypatch, shared_policy, validate_api):
    saved = []

    async def create(**kwargs):
        saved.append(kwargs)
        return shared_policy

    monkeypatch.setattr(routes.model_config_service, "create_config_async", create)
    request = routes.CreateConfigRequest(
        config_name="bad",
        model_type="llm",
        provider="custom",
        api_endpoint=SHARED + "/v1/models",
        model_name="bob-model",
        validate_api=validate_api,
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.create_config(request, USER))
    assert exc_info.value.status_code == 403
    assert saved == []


@pytest.mark.parametrize(
    "update", [dict(model_name="bob-model"), dict(api_endpoint=SHARED + "/v1/models", model_name="bob-model")]
)
def test_update_cannot_retarget_shared_model(monkeypatch, shared_policy, update):
    saved = []
    monkeypatch.setattr(routes.model_config_service, "update_config", lambda **kwargs: saved.append(kwargs) or True)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.update_config("config-1", routes.UpdateConfigRequest(**update), USER))
    assert exc_info.value.status_code == 403
    assert saved == []


def test_bound_local_config_metadata_edit_cannot_change_identity(monkeypatch, shared_policy):
    shared_policy.update(source_type="local_deployed", deployment_id="deployment-1")
    monkeypatch.setattr(routes.deployment_service, "list_deployments", lambda **_: ([], 0))

    def update(**kwargs):
        for field in ("model_type", "provider", "api_endpoint", "api_key", "model_name", "provider_config"):
            assert kwargs[field] is None
        shared_policy["description"] = kwargs["description"]
        return True

    monkeypatch.setattr(routes.model_config_service, "update_config", update)
    result = asyncio.run(
        routes.update_config(
            "config-1",
            routes.UpdateConfigRequest(
                api_endpoint=SHARED + "/v1/models",
                provider="custom",
                model_type="embedding",
                model_name="bob-model",
                api_key="synthetic-test-key",
                provider_config={"model": "bob-model"},
                description="updated metadata",
            ),
            USER,
        )
    )
    assert result.description == "updated metadata"
    assert result.model_name == "alice-model"


@pytest.mark.parametrize("mode", [None, "embedding_similarity", "embedding_recall"])
def test_proxy_cannot_override_owned_shared_model(monkeypatch, shared_policy, mode):
    sent = []
    if mode:
        shared_policy["model_type"] = "embedding"

    async def respond(*args, **kwargs):
        sent.append(kwargs)
        return httpx.Response(200, json={"ok": True})

    async def compute(*_):
        sent.append("embedding")
        return {}

    monkeypatch.setattr(routes, "async_request_user_outbound", respond)
    monkeypatch.setattr(routes, "_compute_embedding_similarity", compute)
    monkeypatch.setattr(routes, "_compute_embedding_recall", compute)
    request = routes.TestProxyRequest(
        path="/v1/embeddings" if mode else "/v1/chat/completions", body={"model": "bob-model", "mode": mode}
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.test_api_proxy("config-1", request, USER))
    assert exc_info.value.status_code == 403
    assert sent == []


def test_proxy_rechecks_stored_shared_model_ownership(monkeypatch, shared_policy):
    shared_policy["model_name"] = "bob-model"

    async def respond(*_, **__):
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(routes, "async_request_user_outbound", respond)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes.test_api_proxy("config-1", routes.TestProxyRequest(path="/v1/chat/completions", body={}), USER)
        )
    assert exc_info.value.status_code == 403


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_bound_non_xinference_config_can_only_call_its_own_model(monkeypatch, shared_policy, framework):
    shared_policy.update(
        source_type="local_deployed",
        deployment_id="deployment-1",
        provider="xinference",
        inference_framework=framework,
        api_endpoint="https://bound.example.test/v1",
    )
    monkeypatch.setattr(
        routes.deployment_service,
        "get_deployment",
        lambda _id: {
            "deployment_id": "deployment-1",
            "user_id": "alice",
            "inference_framework": framework,
            "model_uid": "alice-model",
            "status": "running",
            "xinference_endpoint": "https://bound.example.test/v1",
        },
    )

    async def respond(_method, _url, _user_id, **kwargs):
        return httpx.Response(200, json={"model": kwargs["json"]["model"]})

    monkeypatch.setattr(routes, "async_request_user_outbound", respond)
    result = asyncio.run(
        routes.test_api_proxy(
            "config-1",
            routes.TestProxyRequest(path="/v1/chat/completions", body={}),
            USER,
        )
    )
    assert result.success is True
    assert result.data == {"model": "alice-model"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes.test_api_proxy(
                "config-1",
                routes.TestProxyRequest(path="/v1/chat/completions", body={"model": "bob-model"}),
                USER,
            )
        )
    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    "binding", ["missing", "foreign", "stopped", "model_mismatch", "endpoint_mismatch", "forged_framework"]
)
def test_forged_or_unusable_local_binding_cannot_authorize_calls(monkeypatch, shared_policy, binding):
    shared_policy.update(
        source_type="local_deployed", deployment_id="deployment-1", provider="xinference", inference_framework="vllm"
    )
    deployment = {
        "deployment_id": "deployment-1",
        "user_id": "alice",
        "inference_framework": "vllm",
        "model_uid": "alice-model",
        "status": "running",
        "xinference_endpoint": SHARED,
    }
    if binding == "missing":
        deployment = None
    elif binding == "foreign":
        deployment["user_id"] = "bob"
    elif binding == "stopped":
        deployment["status"] = "stopped"
    elif binding == "model_mismatch":
        deployment["model_uid"] = "other-model"
    elif binding == "endpoint_mismatch":
        deployment["xinference_endpoint"] = "https://other.example.test"
    elif binding == "forged_framework":
        deployment["inference_framework"] = "xinference"
        shared_policy["model_name"] = "bob-model"
    monkeypatch.setattr(routes.deployment_service, "get_deployment", lambda _id: deployment)

    async def respond(*_args, **_kwargs):
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(routes, "async_request_user_outbound", respond)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes.test_api_proxy("config-1", routes.TestProxyRequest(path="/v1/chat/completions", body={}), USER)
        )
    assert exc_info.value.status_code == 403


@pytest.mark.parametrize("healthy", [True, False])
def test_bound_replica_authorization_uses_selected_child_endpoint(monkeypatch, shared_policy, healthy):
    endpoint = "https://replica.example.test:9002"
    shared_policy.update(
        source_type="local_deployed",
        deployment_id="deployment-1",
        deployment_replica_id="replica-2",
        provider="xinference",
        inference_framework="vllm",
        api_endpoint=endpoint + "/v1",
    )
    monkeypatch.setattr(
        routes.deployment_service,
        "get_deployment",
        lambda _id: {
            "deployment_id": "deployment-1",
            "user_id": "alice",
            "inference_framework": "vllm",
            "model_uid": "alice-model",
            "status": "running",
            "xinference_endpoint": "https://parent.example.test",
            "replica_instances": [
                {
                    "replica_id": "replica-2",
                    "deployment_id": "deployment-1",
                    "endpoint": endpoint,
                    "status": "running",
                    "health_status": "HEALTHY" if healthy else "UNHEALTHY",
                }
            ],
        },
    )

    async def respond(*_args, **_kwargs):
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(routes, "async_request_user_outbound", respond)
    request = routes.TestProxyRequest(path="/v1/chat/completions", body={})
    if healthy:
        assert asyncio.run(routes.test_api_proxy("config-1", request, USER)).success
    else:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(routes.test_api_proxy("config-1", request, USER))
        assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    "admin,endpoint,body_model",
    [
        (False, SHARED + "/v1", "alice-model"),
        (True, SHARED + "/v1", "bob-model"),
        (False, "https://external.example.test/v1", "other-external-model"),
    ],
)
def test_proxy_preserves_authorized_and_external_multi_model_calls(
    monkeypatch, shared_policy, admin, endpoint, body_model
):
    shared_policy["api_endpoint"] = endpoint

    async def respond(_method, url, _user_id, **kwargs):
        return httpx.Response(200, json={"model": kwargs["json"]["model"]})

    monkeypatch.setattr(routes, "async_request_user_outbound", respond)
    result = asyncio.run(
        routes.test_api_proxy(
            "config-1",
            routes.TestProxyRequest(path="/v1/chat/completions", body={"model": body_model}),
            {**USER, "is_admin": admin},
        )
    )
    assert result.success is True
    assert result.data == {"model": body_model}
