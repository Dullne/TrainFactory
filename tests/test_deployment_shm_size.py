"""推理/训练容器共享内存（--shm-size）配置测试。

Docker 默认 /dev/shm 仅 64MB：vLLM/SGLang 大模型加载与长上下文、PyTorch
DataLoader 多 worker 都会因共享内存不足 OOM。断言三处 docker run 均带
--shm-size（默认 2g），DEPLOY_SHM_SIZE=0 时显式不限制。
"""
import importlib
from pathlib import Path
from types import SimpleNamespace

import yaml

docker_deployer_module = importlib.import_module(
    "train_factory.deployment.docker_deployer"
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def _capture(monkeypatch, create_call, *, deploy_shm_size=None):
    deployer = docker_deployer_module.DockerDeployer()
    captured = {}
    monkeypatch.setattr(deployer, "remove_container", lambda name: False)
    monkeypatch.setattr(
        docker_deployer_module,
        "get_settings",
        lambda: SimpleNamespace(allow_model_remote_code=False),
        raising=False,
    )
    if deploy_shm_size is not None:
        monkeypatch.setenv("DEPLOY_SHM_SIZE", deploy_shm_size)
    else:
        monkeypatch.delenv("DEPLOY_SHM_SIZE", raising=False)

    def fake_run(command, timeout=30):
        captured["command"] = command
        return True, "container-id"

    monkeypatch.setattr(deployer, "_run_command", fake_run)
    create_call(deployer)
    return captured["command"]


def _create_xinference(deployer):
    deployer.create_xinference_container(
        container_name="xinference-test",
        port=10001,
        gpu_id=0,
        model_name="model-1",
        model_uid="uid-1",
        model_path="/app/models/model-1",
        model_type="embedding",
    )


def _create_vllm(deployer):
    deployer.create_vllm_container(
        container_name="vllm-test",
        port=10001,
        gpu_id=0,
        model_path="/app/models/model-1",
    )


def _create_sglang(deployer):
    deployer.create_sglang_container(
        container_name="sglang-test",
        port=10001,
        gpu_id=0,
        model_path="/app/models/model-1",
        model_name="model-1",
        model_type="reranker",
    )


def test_inference_containers_default_to_2g_shm(monkeypatch):
    for create_call in (_create_xinference, _create_vllm, _create_sglang):
        command = _capture(monkeypatch, create_call)
        index = command.index("--shm-size")
        assert command[index + 1] == "2g"


def test_shm_size_respects_env_override(monkeypatch):
    command = _capture(monkeypatch, _create_vllm, deploy_shm_size="16g")
    assert command[command.index("--shm-size") + 1] == "16g"


def test_shm_size_zero_disables_limit(monkeypatch):
    for value in ("0", "0g", ""):
        command = _capture(monkeypatch, _create_xinference, deploy_shm_size=value)
        assert "--shm-size" not in command


def test_api_container_has_shm_size_in_compose():
    compose = yaml.safe_load(
        (ROOT_DIR / "docker/docker-compose.yml").read_text(encoding="utf-8")
    )
    api = compose["services"]["train-factory-api"]
    assert api.get("shm_size") == "${API_SHM_SIZE:-1g}"
