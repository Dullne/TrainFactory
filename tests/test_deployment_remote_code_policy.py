import importlib
from pathlib import Path
from types import SimpleNamespace

import yaml


docker_deployer_module = importlib.import_module(
    "train_factory.deployment.docker_deployer"
)
deployment_service_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
launch_config_module = importlib.import_module(
    "train_factory.deployment.launch_config"
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def _capture_sglang_command(monkeypatch, *, allow_remote_code: bool) -> list[str]:
    deployer = docker_deployer_module.DockerDeployer()
    captured = {}
    monkeypatch.setattr(deployer, "remove_container", lambda name: False)

    def fake_run(command, timeout=30):
        captured["command"] = command
        return True, "container-id"

    monkeypatch.setattr(deployer, "_run_command", fake_run)
    server_argv = launch_config_module.build_sglang_server_argv(
        launch_config_module.parse_launch_config({"framework": "sglang"}),
        model_path="/app/models/model-1",
        served_model_name="untrusted/model",
        port=10001,
        gpu_memory_utilization=0.9,
        model_type="reranker",
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        chat_template=None,
        trust_remote_code=allow_remote_code,
    )
    deployer.create_sglang_container(
        container_name="sglang-test",
        port=10001,
        gpu_ids=(0,),
        server_argv=server_argv,
    )
    return captured["command"]


def test_sglang_remote_code_is_disabled_by_default(monkeypatch):
    command = _capture_sglang_command(monkeypatch, allow_remote_code=False)

    assert "--trust-remote-code" not in command


def test_sglang_remote_code_requires_operator_opt_in(monkeypatch):
    command = _capture_sglang_command(monkeypatch, allow_remote_code=True)

    assert "--trust-remote-code" in command


def test_deployment_config_cannot_override_remote_code_policy():
    raw = {
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "model_kwargs": {
            "trust_remote_code": True,
            "revision": "main",
        },
        "nested": [{"trust-remote-code": True}, {"safe": "value"}],
    }

    sanitized = deployment_service_module._sanitize_deployment_config(raw)

    assert sanitized == {
        "dtype": "bfloat16",
        "model_kwargs": {"revision": "main"},
        "nested": [{}, {"safe": "value"}],
    }
    assert raw["trust_remote_code"] is True
    assert raw["model_kwargs"]["trust_remote_code"] is True


def test_trusted_launch_kwargs_force_the_operator_policy(monkeypatch):
    monkeypatch.setattr(
        deployment_service_module,
        "get_settings",
        lambda: SimpleNamespace(allow_model_remote_code=False),
    )

    kwargs = deployment_service_module._trusted_model_launch_kwargs(
        {
            "trust_remote_code": True,
            "model_kwargs": {"trust_remote_code": True},
        }
    )

    assert kwargs == {"model_kwargs": {}, "trust_remote_code": False}


def test_xinference_uses_digest_bound_guarded_compatibility_launcher():
    compose_source = (ROOT_DIR / "docker/docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert (ROOT_DIR / "docker/inference-contracts/qwen3-compatibility.json").exists()
    assert (
        "./xinference-patches:/opt/trainfactory/xinference-patches:ro"
    ) in compose_source
    assert (
        "./inference-contracts:/opt/trainfactory/inference-contracts:ro"
    ) in compose_source
    assert "apply_qwen3_compatibility.py" in compose_source


def test_shared_xinference_receives_remote_code_policy():
    compose = yaml.safe_load(
        (ROOT_DIR / "docker/docker-compose.yml").read_text(encoding="utf-8")
    )
    environment = compose["services"]["xinference"]["environment"]

    assert "ALLOW_MODEL_REMOTE_CODE=${ALLOW_MODEL_REMOTE_CODE:-false}" in environment
