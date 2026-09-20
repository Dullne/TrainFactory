"""Shared model ownership applies to inline requests and background clients."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlmodel import Session, create_engine
from sqlalchemy.pool import StaticPool

from train_factory.api.routes import deep_evaluation_routes, generation_routes
from train_factory.auth.user_service import user_service
from train_factory.config.settings import get_settings
from train_factory.deployment.deployment_service import deployment_service
from train_factory.deep_evaluation.llm_judge import LLMConfig as JudgeConfig, LLMJudge
from train_factory.generation.clients.llm_client import LLMConfig, LLMClient
from train_factory.generation.clients.embedding_client import EmbeddingConfig, EmbeddingClient
from train_factory.generation.clients.rerank_client import RerankConfig, RerankClient
from train_factory.storage.services.inference_authorization_service import authorize_inference_config, authorize_inference_model
from train_factory.storage.services import inference_authorization_service
from train_factory.storage.entities.deployment_entity import DeploymentDB


SHARED = "http://xinference:9997/v1"
ALICE = {"user_id": "alice", "is_admin": False, "is_active": True}


@pytest.fixture
def shared_models(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    DeploymentDB.__table__.create(engine)
    with Session(engine) as value:
        value.add(DeploymentDB(
            model_id="alice-registry", user_id="alice", model_uid="alice-model",
            status="running", inference_framework="xinference", deploy_mode="shared",
            xinference_endpoint=SHARED,
        ))
        value.commit()

    @contextmanager
    def session():
        with Session(engine) as value:
            yield value

    monkeypatch.setattr(inference_authorization_service, "get_session", session)
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "xinference_endpoint", "http://xinference:9997")
    monkeypatch.setattr(user_service, "get_user", lambda user_id: dict(ALICE, user_id=user_id))
    monkeypatch.setattr(
        deployment_service,
        "list_deployments",
        lambda **_: ([{
            "user_id": "alice", "model_uid": "alice-model", "status": "running",
            "inference_framework": "xinference", "xinference_endpoint": SHARED,
        }], 1),
    )
    for routes in (deep_evaluation_routes, generation_routes):
        monkeypatch.setattr(routes, "validate_user_outbound_url", lambda url, *_: url)
    yield engine
    engine.dispose()


def test_inline_quick_evaluation_rejects_another_users_shared_model(monkeypatch, shared_models):
    lease = Mock()
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service, "admit_execution",
        lambda *args: (None, lease),
    )
    # Construction must never be reached for an unauthorized model.
    judge = Mock(side_effect=AssertionError("unauthorized model reached execution"))
    monkeypatch.setattr(deep_evaluation_routes, "create_llm_judge_from_dict", judge)
    request = deep_evaluation_routes.EvaluateSampleRequest(
        input="question", actual_output="answer",
        llm_config={"endpoint": SHARED, "model": "bob-model"},
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(deep_evaluation_routes.evaluate_sample(request, ALICE))
    assert exc.value.status_code == 403
    judge.assert_not_called()


@pytest.mark.parametrize("config", [
    {"endpoint": SHARED, "model": "bob-model"},
    {"endpoints": [{"url": SHARED, "model": "bob-model"}]},
    {"model": "bob-model", "endpoints": [{"url": SHARED}]},
])
def test_inline_generation_rejects_another_users_shared_model(shared_models, config):
    with pytest.raises(HTTPException) as exc:
        generation_routes._normalize_model_config_endpoints(config, "alice")
    assert exc.value.status_code == 403


@pytest.mark.parametrize("kind", ["llm", "judge", "embedding", "rerank"])
def test_background_clients_recheck_shared_model_before_network(monkeypatch, shared_models, kind):
    calls = []

    async def respond(request):
        calls.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "data": [{"index": 0, "embedding": [1.0]}],
            "results": [{"index": 0, "relevance_score": 1.0}],
        })

    async def run():
        config = dict(endpoint=SHARED, model="bob-model", user_id="alice")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            if kind == "llm":
                client = LLMClient(LLMConfig(**config))
                call = lambda: client.chat("question")
            elif kind == "judge":
                client = LLMJudge(JudgeConfig(**config))
                call = lambda: client.chat("question")
            elif kind == "embedding":
                client = EmbeddingClient(EmbeddingConfig(**config))
                call = lambda: client.embed(["question"])
            else:
                client = RerankClient(RerankConfig(**config))
                call = lambda: client.rerank("question", ["answer"])
            client._client = http
            with pytest.raises(HTTPException) as exc:
                await call()
            assert exc.value.status_code == 403
        assert calls == []

    asyncio.run(run())


@pytest.mark.parametrize("endpoint,model,user,auth_enabled,expected", [
    (SHARED, "alice-model", ALICE, True, {"alice-model"}),
    (SHARED, "bob-model", dict(ALICE, is_admin=True), True, None),
    (SHARED, "bob-model", None, False, None),
    ("https://api.example.com/v1", "external-model", ALICE, True, None),
    ("https://api.example.com/v1", "external-model", None, True, None),
])
def test_authorization_preserves_owner_admin_development_and_external_apis(
    monkeypatch, shared_models, endpoint, model, user, auth_enabled, expected,
):
    monkeypatch.setattr(get_settings(), "auth_enabled", auth_enabled)
    assert authorize_inference_model(
        endpoint, model, user.get("user_id") if user else None, current_user=user,
    ) == expected


def test_runtime_admin_is_loaded_from_current_account(monkeypatch, shared_models):
    monkeypatch.setattr(user_service, "get_user", lambda _: dict(ALICE, is_admin=True))
    assert authorize_inference_model(SHARED, "bob-model", "alice") is None


@pytest.mark.parametrize("user_id", [None, "", "anonymous"])
def test_shared_endpoint_never_treats_missing_identity_as_public(shared_models, user_id):
    with pytest.raises(HTTPException) as exc:
        authorize_inference_model(SHARED, "bob-model", user_id)
    assert exc.value.status_code == 403


def test_runtime_rejects_disabled_owner(monkeypatch, shared_models):
    monkeypatch.setattr(user_service, "get_user", lambda _: dict(ALICE, is_active=False))
    with pytest.raises(HTTPException) as exc:
        authorize_inference_model(SHARED, "alice-model", "alice")
    assert exc.value.status_code == 403


def test_registered_nondefault_shared_endpoint_rejects_other_tenant(monkeypatch, shared_models):
    endpoint = "http://other-xinference:9997/v1"
    with Session(shared_models) as session:
        session.add(DeploymentDB(
            model_id="bob-registry", user_id="bob", model_uid="bob-model", status="running",
            inference_framework="xinference", deploy_mode="shared", xinference_endpoint=endpoint,
        ))
        session.commit()
    with pytest.raises(HTTPException) as exc:
        authorize_inference_model(endpoint, "bob-model", "alice", current_user=ALICE)
    assert exc.value.status_code == 403


def test_owned_xinference_container_remains_authorized(shared_models):
    endpoint = "http://owned-container:9997/v1"
    with Session(shared_models) as session:
        session.add(DeploymentDB(
            model_id="alice-container-registry", user_id="alice", model_uid="container-model",
            status="running", inference_framework="xinference", deploy_mode="container",
            xinference_endpoint=endpoint,
        ))
        session.commit()
    assert authorize_inference_model(
        endpoint, "container-model", "alice", provider="xinference", current_user=ALICE,
    ) == {"container-model"}


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_owned_bound_non_xinference_config_can_be_used_by_sync(monkeypatch, shared_models, framework):
    monkeypatch.setattr(deployment_service, "get_deployment", lambda _: {
        "deployment_id": "owned-deployment", "user_id": "alice", "status": "running",
        "inference_framework": framework, "xinference_endpoint": "http://owned-container:8000",
        "model_uid": "owned-model",
    })
    config = {
        "source_type": "local_deployed", "deployment_id": "owned-deployment",
        "user_id": "alice", "provider": "xinference",
        "api_endpoint": "http://owned-container:8000/v1", "model_name": "owned-model",
    }
    assert authorize_inference_config(config, "alice", current_user=ALICE) == {"owned-model"}


def test_config_authorization_finds_owned_shared_model_after_first_thousand(shared_models):
    from train_factory.api.routes import model_config_routes

    with Session(shared_models) as session:
        session.add_all([
            DeploymentDB(
                model_id="registry", user_id="alice", model_uid=f"model-{index}",
                status="running", inference_framework="xinference", deploy_mode="shared",
                xinference_endpoint=SHARED,
            ) for index in range(1001)
        ])
        session.commit()
    assert "model-1000" in model_config_routes._owned_model_uids_for_endpoint(SHARED, ALICE)


def test_sync_inline_configuration_cannot_target_another_users_model(monkeypatch, shared_models):
    from train_factory.api.routes import sync_routes

    monkeypatch.setattr(sync_routes, "validate_user_outbound_url", lambda url, *_: url)
    with pytest.raises(HTTPException) as exc:
        sync_routes._validate_generation_config(
            {"llm_config": {"endpoint": SHARED, "model": "bob-model"}}, ALICE,
        )
    assert exc.value.status_code == 403


@pytest.mark.parametrize("endpoint", [
    "http://xinference.:9997/v1", "http://XINFERENCE:9997/v1/",
    "http://xinference:9997//v1", "http://xinference:9997/v1/%2e%2e/v1",
])
def test_shared_endpoint_aliases_cannot_remove_ownership_gate(shared_models, endpoint):
    with pytest.raises(HTTPException) as exc:
        authorize_inference_model(endpoint, "bob-model", "alice", current_user=ALICE)
    assert exc.value.status_code == 403


def test_inline_standard_evaluation_cannot_target_another_users_model(monkeypatch, shared_models):
    from train_factory.api.routes import evaluation_routes

    monkeypatch.setattr(evaluation_routes, "validate_user_outbound_url", lambda url, *_: url)
    with pytest.raises(HTTPException) as exc:
        evaluation_routes._validate_model_configs(
            [{"name": "test", "endpoint": SHARED, "model_name": "bob-model"}], "alice",
        )
    assert exc.value.status_code == 403


def test_deep_evaluation_task_rejects_inline_foreign_embedding(monkeypatch, shared_models):
    monkeypatch.setattr(
        deep_evaluation_routes, "_resolve_deep_evaluation_dataset_configs",
        lambda *_args: [{"name": "owned-data", "path": "/data/owned.jsonl"}],
    )
    admission = Mock(side_effect=AssertionError("unauthorized task reached persistence"))
    monkeypatch.setattr(deep_evaluation_routes.background_task_admission_service, "admit_execution", admission)
    request = deep_evaluation_routes.CreateDeepEvaluationTaskRequest(
        model_configs=[{"group_name": "foreign", "embedding": {
            "endpoint": SHARED, "model_name": "bob-model",
        }}],
        dataset_configs=[{"dataset_id": "alice-dataset", "name": "owned-data", "path": "/data/owned.jsonl"}],
        metrics=["mrr"],
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(deep_evaluation_routes.create_deep_evaluation_task(request, BackgroundTasks(), ALICE))
    assert exc.value.status_code == 403
    admission.assert_not_called()
