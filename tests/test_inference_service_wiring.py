from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


deployment_service_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)


def _deployment(framework: str) -> SimpleNamespace:
    launch_config: dict[str, object] = {
        "framework": framework,
        "dtype": "bfloat16",
    }
    if framework == "vllm":
        launch_config["enforce_eager"] = True
    else:
        launch_config["attention_backend"] = "flashinfer"
    return SimpleNamespace(
        deployment_id=f"{framework}-deployment-id",
        container_name=f"{framework}-deployment",
        port=10001,
        gpu_id=3,
        gpu_memory_utilization=0.75,
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        xinference_endpoint="http://127.0.0.1:10001",
        user_id="user-1",
        config={"launch_config": launch_config},
    )


def _prepare_service(monkeypatch: pytest.MonkeyPatch):
    service = deployment_service_module.DeploymentService()
    monkeypatch.setattr(
        service,
        "_get_xinference_model_name",
        lambda _model: "Qwen/Qwen3-Reranker-4B",
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_settings",
        lambda: SimpleNamespace(allow_model_remote_code=False),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "wait_for_service",
        lambda *_args, **_kwargs: True,
    )
    return service


def test_vllm_service_passes_builder_argv_to_container_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _prepare_service(monkeypatch)
    deployment = _deployment("vllm")
    model = {
        "model_path": "/models/qwen3-reranker",
        "model_type": "reranker",
    }
    server_argv = ("vllm", "serve", "/models/qwen3-reranker")
    captured: dict[str, object] = {}

    def build(launch_config, **kwargs):
        captured["launch_config"] = launch_config
        captured["builder_kwargs"] = kwargs
        return server_argv

    def create(**kwargs):
        captured["creator_kwargs"] = kwargs
        return True, "created", ["docker", "run"]

    monkeypatch.setattr(deployment_service_module, "build_vllm_server_argv", build)
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "create_vllm_container",
        create,
    )

    service._start_vllm_container(deployment, model, "served-reranker")

    assert captured["launch_config"].framework == "vllm"
    assert captured["builder_kwargs"] == {
        "model_path": "/models/qwen3-reranker",
        "served_model_name": "served-reranker",
        "port": 10001,
        "gpu_memory_utilization": 0.75,
        "model_type": "reranker",
        "enable_lora": False,
        "max_loras": 4,
        "max_lora_rank": 64,
        "trust_remote_code": False,
        "model_family": "Qwen/Qwen3-Reranker-4B",
    }
    creator_kwargs = captured["creator_kwargs"]
    assert creator_kwargs["container_name"] == "vllm-deployment"
    assert creator_kwargs["port"] == 10001
    assert creator_kwargs["gpu_ids"] == (3,)
    assert creator_kwargs["server_argv"] is server_argv


def test_sglang_service_passes_builder_argv_to_container_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _prepare_service(monkeypatch)
    deployment = _deployment("sglang")
    model = {
        "model_path": "/models/qwen3-reranker",
        "model_type": "reranker",
    }
    server_argv = (
        "sglang",
        "serve",
        "--model-path",
        "/models/qwen3-reranker",
    )
    captured: dict[str, object] = {}

    def build(launch_config, **kwargs):
        captured["launch_config"] = launch_config
        captured["builder_kwargs"] = kwargs
        return server_argv

    def create(**kwargs):
        captured["creator_kwargs"] = kwargs
        return True, "created", ["docker", "run"]

    monkeypatch.setattr(deployment_service_module, "build_sglang_server_argv", build)
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "create_sglang_container",
        create,
    )

    service._start_sglang_container(deployment, model, "served-reranker")

    assert captured["launch_config"].framework == "sglang"
    assert captured["builder_kwargs"] == {
        "model_path": "/models/qwen3-reranker",
        "served_model_name": "served-reranker",
        "port": 10001,
        "gpu_memory_utilization": 0.75,
        "model_type": "reranker",
        "enable_lora": False,
        "max_loras": 4,
        "max_lora_rank": 64,
        "chat_template": (
            "/opt/trainfactory/sglang-templates/"
            "qwen3_reranker_no_think.jinja"
        ),
        "trust_remote_code": False,
        "model_family": "Qwen/Qwen3-Reranker-4B",
    }
    creator_kwargs = captured["creator_kwargs"]
    assert creator_kwargs["container_name"] == "sglang-deployment"
    assert creator_kwargs["port"] == 10001
    assert creator_kwargs["gpu_ids"] == (3,)
    assert creator_kwargs["server_argv"] is server_argv


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
@pytest.mark.parametrize("launch_config", [None, {}, []])
def test_service_rejects_explicit_invalid_launch_config(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    launch_config: object,
) -> None:
    service = _prepare_service(monkeypatch)
    deployment = _deployment(framework)
    deployment.config = {"launch_config": launch_config}
    model = {
        "model_path": "/models/qwen3-reranker",
        "model_type": "reranker",
    }
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        f"create_{framework}_container",
        lambda **_kwargs: (True, "created", "docker run"),
    )

    with pytest.raises((TypeError, ValidationError)):
        if framework == "vllm":
            service._start_vllm_container(deployment, model, "served-reranker")
        else:
            service._start_sglang_container(deployment, model, "served-reranker")


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_service_uses_legacy_fallback_only_when_launch_config_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    service = _prepare_service(monkeypatch)
    deployment = _deployment(framework)
    deployment.config = {"dtype": "float16"}
    if framework == "vllm":
        deployment.config["enforce_eager"] = True
    else:
        deployment.config["attention_backend"] = "flashinfer"
    model = {
        "model_path": "/models/qwen3-reranker",
        "model_type": "reranker",
    }
    captured: dict[str, object] = {}

    def build(launch_config, **_kwargs):
        captured["launch_config"] = launch_config
        return (framework, "serve")

    monkeypatch.setattr(
        deployment_service_module,
        f"build_{framework}_server_argv",
        build,
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        f"create_{framework}_container",
        lambda **_kwargs: (True, "created", ["docker", "run"]),
    )

    if framework == "vllm":
        service._start_vllm_container(deployment, model, "served-reranker")
    else:
        service._start_sglang_container(deployment, model, "served-reranker")

    launch_config = captured["launch_config"]
    assert launch_config.framework == framework
    assert launch_config.dtype == "float16"
    if framework == "vllm":
        assert launch_config.enforce_eager is True
    else:
        assert launch_config.attention_backend == "flashinfer"


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "parallel_field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
    ],
)
def test_single_gpu_service_rejects_parallel_topology(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    parallel_field: str,
) -> None:
    service = _prepare_service(monkeypatch)
    deployment = _deployment(framework)
    deployment.config["launch_config"][parallel_field] = 2
    model = {
        "model_path": "/models/qwen3-reranker",
        "model_type": "reranker",
    }
    monkeypatch.setattr(
        deployment_service_module,
        f"build_{framework}_server_argv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parallel topology must fail before argv rendering")
        ),
    )

    with pytest.raises(
        ValueError,
        match="single-GPU deployment does not support parallel topology",
    ):
        if framework == "vllm":
            service._start_vllm_container(deployment, model, "served-reranker")
        else:
            service._start_sglang_container(deployment, model, "served-reranker")


@pytest.mark.parametrize(
    ("model_name", "expects_qwen_template"),
    [
        ("Qwen3-Reranker-4B", True),
        ("bge-reranker-v2-m3", False),
    ],
)
def test_sglang_generated_uid_is_runtime_and_persisted_config_identity(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    expects_qwen_template: bool,
) -> None:
    class Deployment(SimpleNamespace):
        def update_status(self, status: str, error_message: str | None = None) -> None:
            self.status = status
            self.error_message = error_message

        def model_copy(self, *, deep: bool = False):
            return copy.deepcopy(self) if deep else copy.copy(self)

    deployment = Deployment(
        deployment_id="deadbeef-0000-0000-0000-000000000001",
        deployment_name="sglang-test",
        model_id="model-1",
        model_uid=None,
        status="pending",
        error_message=None,
        deploy_mode="container",
        inference_framework="sglang",
        container_name="sglang-deployment",
        port=10001,
        gpu_id=3,
        gpu_memory_utilization=0.75,
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        xinference_endpoint="http://127.0.0.1:10001",
        user_id="user-1",
        config={"launch_config": {"framework": "sglang"}},
    )

    class Result:
        def first(self):
            return deployment

    class Session:
        def __init__(self) -> None:
            self.committed_model_uids: list[str | None] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def exec(self, _statement):
            return Result()

        def add(self, _deployment) -> None:
            return None

        def commit(self) -> None:
            self.committed_model_uids.append(deployment.model_uid)

        def refresh(self, _deployment) -> None:
            return None

    session = Session()
    service = deployment_service_module.DeploymentService()
    model = {
        "model_id": "model-1",
        "model_name": model_name,
        "model_path": f"/models/{model_name}",
        "model_type": "reranker",
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(deployment_service_module, "get_session", lambda: session)
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda _model_id: model,
    )
    monkeypatch.setattr(service, "_require_managed_container", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "_claimed_legacy_deployment",
        lambda *_args, **_kwargs: deployment,
    )
    monkeypatch.setattr(service, "_legacy_container_id", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "_require_replica_operation_ownership",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        service,
        "_mark_adapters_unloaded_after_runtime_reset",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        deployment_service_module,
        "get_settings",
        lambda: SimpleNamespace(allow_model_remote_code=False),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "create_sglang_container",
        lambda **kwargs: (
            captured.setdefault("server_argv", kwargs["server_argv"]) is not None,
            "created",
            ["docker", "run"],
        ),
    )
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "wait_for_service",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        service,
        "_create_config_for_deployment",
        lambda dep, _model: captured.setdefault("config_model_uid", dep.model_uid),
    )
    monkeypatch.setattr(
        service,
        "_sync_configs_from_deployments",
        lambda _deployments: None,
    )
    monkeypatch.setattr(
        service,
        "_deployment_to_dict",
        lambda dep: {"model_uid": dep.model_uid},
    )

    claim = deployment_service_module.ReplicaOperationClaim(
        deployment_id=deployment.deployment_id,
        token="claim-token",
        generation=1,
        operation="start",
        replica_id=None,
    )
    result = service._start_legacy_deployment_claimed(claim, user_id=None)

    expected_uid = f"{model_name}-deadbeef"
    server_argv = captured["server_argv"]
    assert server_argv[server_argv.index("--served-model-name") + 1] == expected_uid
    assert ("--chat-template" in server_argv) is expects_qwen_template
    assert deployment.model_uid == expected_uid
    assert session.committed_model_uids[-1] == expected_uid
    assert captured["config_model_uid"] == expected_uid
    assert result["model_uid"] == expected_uid
