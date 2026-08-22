import asyncio
import importlib
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import (
    deep_evaluation_routes,
    deployment_routes,
    evaluation_routes,
    external_api_config_routes,
    model_config_routes,
)
from train_factory.core.ssrf import SSRFError
from train_factory.deep_evaluation import deep_evaluation_runner
from train_factory.evaluation import evaluation_runner

model_config_module = importlib.import_module(
    "train_factory.storage.services.model_config_service"
)
external_api_config_module = importlib.import_module(
    "train_factory.storage.services.external_api_config_service"
)
outbound_policy = importlib.import_module(
    "train_factory.storage.services.outbound_endpoint_policy"
)
sglang_client_module = importlib.import_module(
    "train_factory.deployment.sglang_client"
)
vllm_client_module = importlib.import_module(
    "train_factory.deployment.vllm_client"
)
xinference_client_module = importlib.import_module(
    "train_factory.deployment.xinference_client"
)


CURRENT_USER = {"user_id": "user-1", "username": "alice", "role": "user"}
PRIVATE_ENDPOINT = "http://10.23.45.67:8000"
PUBLIC_ENDPOINT = "http://8.8.8.8:8000"


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakePolicySession:
    def __init__(self, result_sets):
        self._result_sets = iter(result_sets)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def exec(self, statement):  # noqa: ARG002
        return _FakeResult(next(self._result_sets))


def test_generated_container_endpoint_allows_only_its_mapped_port(monkeypatch):
    deployment = SimpleNamespace(
        deployment_id="deployment-1",
        xinference_endpoint="http://xf-owned:10001",
        port=10001,
    )
    monkeypatch.setenv("HOST_IP", "10.23.45.67")
    monkeypatch.setattr(
        outbound_policy,
        "get_session",
        lambda: _FakePolicySession([[deployment], []]),
    )

    assert outbound_policy.validate_user_outbound_url(
        "http://10.23.45.67:10001/v1",
        "user-1",
    ) == "http://10.23.45.67:10001/v1"

    monkeypatch.setattr(
        outbound_policy,
        "get_session",
        lambda: _FakePolicySession([[deployment], []]),
    )
    with pytest.raises(SSRFError):
        outbound_policy.validate_user_outbound_url(
            "http://10.23.45.67:10002/v1",
            "user-1",
        )


class _FakeModelConfigSession:
    def __init__(self, config):
        self.config = config
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def exec(self, statement):  # noqa: ARG002
        return self

    def first(self):
        return self.config

    def add(self, config):
        self.config = config

    def commit(self):
        self.commits += 1


def test_model_config_service_update_rejects_private_endpoint_before_commit(
    monkeypatch,
):
    config = SimpleNamespace(
        config_id="config-1",
        config_name="safe",
        model_type="llm",
        provider="custom",
        api_endpoint=PUBLIC_ENDPOINT,
        api_key=None,
        model_name="model",
        provider_config=None,
        default_params=None,
        description=None,
        tags=None,
        status="active",
        is_default=False,
        user_id="user-1",
        updated_at=None,
    )
    session = _FakeModelConfigSession(config)
    monkeypatch.setattr(model_config_module, "Session", lambda engine: session)
    monkeypatch.setattr(
        model_config_module.model_config_service,
        "_get_engine",
        lambda: object(),
    )

    with pytest.raises(SSRFError):
        model_config_module.model_config_service.update_config(
            "config-1",
            api_endpoint=PRIVATE_ENDPOINT,
        )

    assert config.api_endpoint == PUBLIC_ENDPOINT
    assert session.commits == 0


def _model_config_payload(endpoint=PRIVATE_ENDPOINT):
    return {
        "config_id": "config-1",
        "config_name": "config",
        "model_type": "llm",
        "provider": "custom",
        "api_endpoint": endpoint,
        "api_key": None,
        "model_name": "model",
        "status": "active",
        "is_default": False,
        "user_id": "user-1",
    }


def test_model_config_test_proxy_revalidates_persisted_endpoint(monkeypatch):
    client_constructions = []

    class UnexpectedClient:
        def __init__(self, *args, **kwargs):
            client_constructions.append((args, kwargs))
            raise AssertionError("HTTP client constructed before SSRF validation")

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "get_config",
        lambda config_id: _model_config_payload(),
    )
    monkeypatch.setattr(model_config_routes.httpx, "AsyncClient", UnexpectedClient)

    with pytest.raises(SSRFError):
        asyncio.run(
            model_config_routes.test_api_proxy(
                "config-1",
                model_config_routes.TestProxyRequest(
                    path="/v1/chat/completions",
                    body={"messages": []},
                ),
                CURRENT_USER,
            )
        )

    assert client_constructions == []


def _deep_evaluation_request(endpoint=PRIVATE_ENDPOINT):
    return deep_evaluation_routes.CreateDeepEvaluationTaskRequest(
        task_name="ssrf-check",
        model_configs=[
            {
                "group_name": "embedding",
                "embedding": {
                    "endpoint": endpoint,
                    "model_name": "embed-model",
                },
            }
        ],
        dataset_configs=[{"dataset_id": "dataset-1"}],
        metrics=["mrr"],
    )


def test_deep_evaluation_rejects_private_direct_endpoint_for_regular_user(
    monkeypatch,
):
    task_creations = []
    monkeypatch.setattr(
        deep_evaluation_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "dataset_name": "dataset",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "resolve_managed_local_evaluation_dataset",
        lambda dataset_id, **_kwargs: {
            "dataset_id": dataset_id,
            "dataset_name": "dataset",
            "storage_path": "/managed/datasets/dataset-1",
            "user_id": "user-1",
        },
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(SSRFError):
        asyncio.run(
            deep_evaluation_routes.create_deep_evaluation_task(
                _deep_evaluation_request(),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert task_creations == []
    assert background_tasks.tasks == []


@pytest.mark.parametrize("uses_config_id", [False, True])
def test_deep_evaluation_runner_revalidates_endpoint_at_execution(
    monkeypatch,
    uses_config_id,
):
    if uses_config_id:
        monkeypatch.setattr(
            deep_evaluation_runner.model_config_service,
            "get_config",
            lambda config_id: _model_config_payload(),
        )
        model_config = {"config_id": "config-1"}
    else:
        model_config = {"endpoint": PRIVATE_ENDPOINT, "model_name": "model"}

    with pytest.raises(SSRFError):
        deep_evaluation_runner._resolve_model_config(model_config, "user-1")


@pytest.mark.parametrize("route_name", ["create", "quick", "bind"])
def test_deployment_write_routes_reject_private_endpoint_before_persisting(
    monkeypatch,
    route_name,
):
    service_calls = []
    monkeypatch.setattr(
        deployment_routes.model_registry_service,
        "get_model",
        lambda model_id: {"model_id": model_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        deployment_routes,
        "requires_tenant_provenance",
        lambda _current_user: False,
    )
    monkeypatch.setattr(
        deployment_routes,
        "check_idempotency",
        lambda *args, **kwargs: (False, None),
    )

    def unexpected_call(*args, **kwargs):
        service_calls.append((args, kwargs))
        raise AssertionError("deployment service called before SSRF validation")

    if route_name == "create":
        monkeypatch.setattr(
            deployment_routes.deployment_service,
            "create_deployment",
            unexpected_call,
        )
        call = deployment_routes.create_deployment(
            deployment_routes.CreateDeploymentRequest(
                model_id="model-1",
                xinference_endpoint=PRIVATE_ENDPOINT,
                auto_start=False,
            ),
            BackgroundTasks(),
            CURRENT_USER,
            "request-1",
        )
    elif route_name == "quick":
        monkeypatch.setattr(
            deployment_routes.deployment_service,
            "deploy_from_model",
            unexpected_call,
        )
        call = deployment_routes.quick_deploy(
            "model-1",
            deployment_routes.QuickDeployRequest(
                xinference_endpoint=PRIVATE_ENDPOINT,
                auto_start=False,
            ),
            CURRENT_USER,
        )
    else:
        monkeypatch.setattr(
            deployment_routes.deployment_service,
            "bind_existing_model",
            unexpected_call,
        )
        call = deployment_routes.bind_existing_model(
            deployment_routes.BindExistingModelRequest(
                endpoint=PRIVATE_ENDPOINT,
                model_uid="model-1",
            ),
            {**CURRENT_USER, "is_admin": True},
        )

    with pytest.raises(SSRFError):
        asyncio.run(call)

    assert service_calls == []


def test_deployment_runtime_map_revalidates_persisted_endpoint(monkeypatch):
    client_calls = []

    def unexpected_client(*args, **kwargs):
        client_calls.append((args, kwargs))
        raise AssertionError("runtime client constructed before SSRF validation")

    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "_get_xinference_client",
        unexpected_client,
    )

    result = deployment_routes._get_xinference_runtime_map(
        [
            {
                "deployment_id": "deployment-1",
                "deploy_mode": "shared",
                "status": "running",
                "inference_framework": "xinference",
                "xinference_endpoint": PRIVATE_ENDPOINT,
                "user_id": "user-1",
            }
        ]
    )

    assert result == {}
    assert client_calls == []


@pytest.mark.parametrize("operation", ["start", "restart", "sync", "delete"])
def test_deployment_operations_revalidate_persisted_endpoint(
    monkeypatch,
    operation,
):
    service_calls = []
    deployment = {
        "deployment_id": "deployment-1",
        "xinference_endpoint": PRIVATE_ENDPOINT,
        "deploy_mode": "shared",
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        deployment_routes.deployment_service,
        "get_deployment",
        lambda deployment_id: deployment,
    )

    def unexpected_call(*args, **kwargs):
        service_calls.append((args, kwargs))
        raise AssertionError("deployment operation ran before SSRF validation")

    if operation == "start":
        monkeypatch.setattr(
            deployment_routes.deployment_service,
            "start_deployment",
            unexpected_call,
        )
        call = deployment_routes.start_deployment("deployment-1", CURRENT_USER)
    elif operation == "restart":
        monkeypatch.setattr(
            deployment_routes.deployment_service,
            "restart_deployment",
            unexpected_call,
        )
        call = deployment_routes.restart_deployment(
            "deployment-1",
            None,
            CURRENT_USER,
        )
    elif operation == "sync":
        monkeypatch.setattr(
            deployment_routes.deployment_service,
            "sync_status",
            unexpected_call,
        )
        call = deployment_routes.sync_deployment_status(
            "deployment-1",
            CURRENT_USER,
        )
    else:
        monkeypatch.setattr(
            deployment_routes.deployment_service,
            "delete_deployment",
            unexpected_call,
        )
        call = deployment_routes.delete_deployment(
            "deployment-1",
            False,
            CURRENT_USER,
        )

    with pytest.raises(SSRFError):
        asyncio.run(call)

    assert service_calls == []


def _evaluation_request(*, endpoint=PUBLIC_ENDPOINT, dataset_type="mteb", path=None):
    return evaluation_routes.CreateEvaluationRequest(
        task_name="security-check",
        model_configs=[
            evaluation_routes.ModelConfig(
                endpoint=endpoint,
                model_name="reranker",
            )
        ],
        dataset_configs=[
            evaluation_routes.DatasetConfig(
                type=dataset_type,
                name="dataset",
                path=path,
            )
        ],
    )


def test_evaluation_route_rejects_private_model_endpoint_before_task_creation(
    monkeypatch,
):
    task_creations = []
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(SSRFError):
        asyncio.run(
            evaluation_routes.create_evaluation_task(
                _evaluation_request(endpoint=PRIVATE_ENDPOINT),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert task_creations == []
    assert background_tasks.tasks == []


def test_evaluation_route_rejects_local_dataset_outside_managed_root(
    monkeypatch,
    tmp_path,
):
    outside_file = tmp_path / "outside.jsonl"
    outside_file.write_text("{}\n", encoding="utf-8")
    task_creations = []
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs) or {"task_id": "task-1"},
    )
    monkeypatch.setattr(
        evaluation_routes,
        "requires_tenant_provenance",
        lambda _current_user: False,
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.create_evaluation_task(
                _evaluation_request(
                    dataset_type="local",
                    path=str(outside_file),
                ),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 400
    assert task_creations == []
    assert background_tasks.tasks == []


def test_evaluation_runner_path_resolver_blocks_outside_and_symlink_targets(
    tmp_path,
):
    allowed_root = tmp_path / "datasets"
    allowed_root.mkdir()
    outside_file = tmp_path / "secret.jsonl"
    outside_file.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        evaluation_runner.resolve_evaluation_dataset_path(
            str(outside_file),
            allowed_roots=[allowed_root],
        )

    link = allowed_root / "linked.jsonl"
    try:
        link.symlink_to(outside_file)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ValueError):
        evaluation_runner.resolve_evaluation_dataset_path(
            str(link),
            allowed_roots=[allowed_root],
        )


def test_evaluation_resume_revalidates_persisted_config_before_background_task(
    monkeypatch,
):
    task = {
        "task_id": "task-1",
        "user_id": "user-1",
        "status": "failed",
        "model_configs": [
            {"endpoint": PRIVATE_ENDPOINT, "model_name": "reranker"}
        ],
        "dataset_configs": [{"type": "mteb", "name": "dataset"}],
        "results": {},
    }
    reset_calls = []
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda task_id: task,
    )
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "reset_for_resume",
        lambda task_id: reset_calls.append(task_id),
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(SSRFError):
        asyncio.run(
            evaluation_routes.resume_evaluation_task(
                "task-1",
                background_tasks,
                CURRENT_USER,
            )
        )

    assert reset_calls == []
    assert background_tasks.tasks == []


@pytest.mark.parametrize("stored", [False, True])
def test_external_api_connection_tests_reject_private_endpoints_before_client(
    monkeypatch,
    stored,
):
    client_constructions = []

    class UnexpectedClient:
        def __init__(self, *args, **kwargs):
            client_constructions.append((args, kwargs))
            raise AssertionError("external client constructed before SSRF validation")

    external_client_module = importlib.import_module(
        "train_factory.sync.external_client"
    )
    monkeypatch.setattr(
        external_client_module,
        "ExternalApiClient",
        UnexpectedClient,
    )

    if stored:
        service = external_api_config_module.external_api_config_service
        monkeypatch.setattr(
            service,
            "get_config",
            lambda config_id: {"config_id": config_id, "user_id": "user-1"},
        )
        monkeypatch.setattr(
            service,
            "get_config_raw",
            lambda config_id: {
                "config_id": config_id,
                "api_url": PRIVATE_ENDPOINT,
                "auth_config": {},
            },
        )
        monkeypatch.setattr(service, "update_config", lambda *args, **kwargs: None)
        call = external_api_config_routes.test_connection_by_id(
            "config-1",
            CURRENT_USER,
        )
    else:
        call = external_api_config_routes.test_connection_inline(
            external_api_config_routes.TestConnectionRequest(
                api_url=PRIVATE_ENDPOINT,
                auth_config={},
            ),
            CURRENT_USER,
        )

    with pytest.raises(SSRFError):
        asyncio.run(call)

    assert client_constructions == []


@pytest.mark.parametrize("operation", ["create", "update"])
def test_external_api_config_writes_reject_private_endpoints_before_mutation(
    monkeypatch,
    operation,
):
    service = external_api_config_module.external_api_config_service
    mutations = []

    if operation == "create":
        monkeypatch.setattr(
            service,
            "create_config",
            lambda **kwargs: mutations.append(kwargs),
        )
        call = external_api_config_routes.create_api_config(
            external_api_config_routes.CreateExternalApiConfigRequest(
                config_name="private",
                api_url=PRIVATE_ENDPOINT,
                auth_config={},
            ),
            CURRENT_USER,
        )
    else:
        monkeypatch.setattr(
            service,
            "get_config",
            lambda config_id: {"config_id": config_id, "user_id": "user-1"},
        )
        monkeypatch.setattr(
            service,
            "update_config",
            lambda *args, **kwargs: mutations.append((args, kwargs)),
        )
        call = external_api_config_routes.update_api_config(
            "config-1",
            external_api_config_routes.UpdateExternalApiConfigRequest(
                api_url=PRIVATE_ENDPOINT,
            ),
            CURRENT_USER,
        )

    with pytest.raises(SSRFError):
        asyncio.run(call)

    assert mutations == []


def test_deep_evaluate_sample_preserves_ssrf_rejection(monkeypatch):
    releases = []
    lease = SimpleNamespace(release=lambda: releases.append(True))

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    request = deep_evaluation_routes.EvaluateSampleRequest(
        input="query",
        llm_config={
            "endpoint": PRIVATE_ENDPOINT,
            "model": "judge-model",
        },
    )

    with pytest.raises(SSRFError):
        asyncio.run(deep_evaluation_routes.evaluate_sample(request, CURRENT_USER))

    assert releases == [True]


class _SuccessfulInferenceResponse:
    status_code = 200
    text = "{}"

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": [],
            "results": [],
            "loras": [],
            "choices": [{"text": "ok"}],
            "score": 0.5,
            "version": "test",
        }


def _record_user_request(calls):
    def request(method, url, user_id=None, **kwargs):
        calls.append((method, url, kwargs))
        return _SuccessfulInferenceResponse()

    return request


def test_vllm_client_disables_redirects_for_every_request(monkeypatch):
    calls = []
    monkeypatch.setattr(
        vllm_client_module,
        "request_user_outbound",
        _record_user_request(calls),
    )
    client = vllm_client_module.VLLMClient(PUBLIC_ENDPOINT)

    client.health_check()
    client.get_model_info()
    client.get_version()
    client.load_lora_adapter("adapter", "/app/models/adapter")
    client.unload_lora_adapter("adapter")
    client.list_lora_adapters()
    client.embeddings(["text"])
    client.rerank("query", ["document"])
    client.score("query", "document")

    assert len(calls) == 9


def test_sglang_client_disables_redirects_for_every_request(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sglang_client_module,
        "request_user_outbound",
        _record_user_request(calls),
    )
    client = sglang_client_module.SGLangClient(PUBLIC_ENDPOINT)

    client.health_check()
    client.get_model_info()
    client.get_server_info()
    client.load_lora_adapter("adapter", "/app/models/adapter")
    client.unload_lora_adapter("adapter")
    client.list_lora_adapters()
    client.embeddings(["text"])
    client.rerank("query", ["document"])
    client.generate("prompt")

    assert len(calls) == 9


def test_xinference_client_disables_redirects_for_request(monkeypatch):
    calls = []

    def request(method, url, user_id=None, **kwargs):
        calls.append((method, url, kwargs))
        return _SuccessfulInferenceResponse()

    monkeypatch.setattr(xinference_client_module, "request_user_outbound", request)
    client = xinference_client_module.XinferenceClient(PUBLIC_ENDPOINT)

    client._request("GET", "/v1/models")

    assert len(calls) == 1
