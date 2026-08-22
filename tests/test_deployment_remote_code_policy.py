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

ROOT_DIR = Path(__file__).resolve().parents[1]


def _capture_sglang_command(monkeypatch, *, allow_remote_code: bool) -> str:
    deployer = docker_deployer_module.DockerDeployer()
    captured = {}
    monkeypatch.setattr(deployer, "remove_container", lambda name: False)
    monkeypatch.setattr(
        docker_deployer_module,
        "get_settings",
        lambda: SimpleNamespace(allow_model_remote_code=allow_remote_code),
        raising=False,
    )

    def fake_run(command, timeout=30):
        captured["command"] = command
        return True, "container-id"

    monkeypatch.setattr(deployer, "_run_command", fake_run)
    deployer.create_sglang_container(
        container_name="sglang-test",
        port=10001,
        gpu_id=0,
        model_path="/app/models/model-1",
        model_name="untrusted/model",
        model_type="reranker",
    )
    return captured["command"][-1]


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


def test_xinference_patches_use_operator_remote_code_policy():
    for relative_path in (
        "docker/xinference-patches/sentence_transformers_core.py",
        "docker/xinference-patches/rerank_sentence_transformers_core.py",
    ):
        source = (ROOT_DIR / relative_path).read_text(encoding="utf-8")
        assert "ALLOW_MODEL_REMOTE_CODE" in source
        assert "trust_remote_code=True" not in source


def test_shared_xinference_receives_remote_code_policy():
    compose = yaml.safe_load(
        (ROOT_DIR / "docker/docker-compose.yml").read_text(encoding="utf-8")
    )
    environment = compose["services"]["xinference"]["environment"]

    assert "ALLOW_MODEL_REMOTE_CODE=${ALLOW_MODEL_REMOTE_CODE:-false}" in environment
