from __future__ import annotations

import errno
import json
import re
import socket
from types import SimpleNamespace

import pytest

from train_factory.deployment import docker_deployer as docker_deployer_module
from train_factory.deployment.docker_deployer import DockerDeployer
from train_factory.deployment.launch_config import (
    build_sglang_server_argv,
    build_vllm_server_argv,
    parse_launch_config,
)


def test_vllm_026_maps_every_supported_launch_field_to_exact_argv() -> None:
    config = parse_launch_config(
        {
            "framework": "vllm",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 3,
            "data_parallel_size": 4,
            "max_context_length": 32768,
            "max_concurrent_requests": 96,
            "dtype": "bfloat16",
            "quantization": "awq",
            "kv_cache_dtype": "fp8_e4m3",
            "enable_expert_parallel": True,
            "enforce_eager": True,
        }
    )

    argv = build_vllm_server_argv(
        config,
        model_path="/models/private model;touch /tmp/canary",
        served_model_name="served;name",
        port=10001,
        gpu_memory_utilization=0.82,
        model_type="reranker",
        enable_lora=True,
        max_loras=3,
        max_lora_rank=32,
        trust_remote_code=True,
        model_family="Qwen3-Reranker-4B",
    )

    assert argv == (
        "vllm",
        "serve",
        "/models/private model;touch /tmp/canary",
        "--host",
        "0.0.0.0",
        "--port",
        "10001",
        "--gpu-memory-utilization",
        "0.82",
        "--served-model-name",
        "served;name",
        "--allowed-origins",
        '["*"]',
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "3",
        "--data-parallel-size",
        "4",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "96",
        "--dtype",
        "bfloat16",
        "--quantization",
        "awq",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--enable-expert-parallel",
        "--enforce-eager",
        "--runner",
        "pooling",
        "--hf-overrides",
        '{"architectures":["Qwen3ForSequenceClassification"],'
        '"classifier_from_token":["no","yes"],'
        '"is_original_qwen3_reranker":true}',
        "--chat-template",
        "/vllm-workspace/examples/pooling/score/template/qwen3_reranker.jinja",
        "--trust-remote-code",
        "--enable-lora",
        "--max-loras",
        "3",
        "--max-lora-rank",
        "32",
    )


def test_vllm_026_generic_reranker_uses_pooling_without_qwen_overrides() -> None:
    argv = build_vllm_server_argv(
        parse_launch_config({"framework": "vllm"}),
        model_path="/models/bge-reranker-v2-m3",
        served_model_name="generic-reranker",
        port=10001,
        gpu_memory_utilization=0.9,
        model_type="reranker",
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        trust_remote_code=False,
        model_family="BAAI/bge-reranker-v2-m3",
    )

    assert argv[argv.index("--runner") + 1] == "pooling"
    assert "--task" not in argv
    assert "--hf-overrides" not in argv
    assert "--chat-template" not in argv


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_qwen3_decoder_reranker_uses_framework_rerank_contract(
    framework: str,
) -> None:
    common = {
        "model_path": "/models/Qwen3-Reranker-4B",
        "served_model_name": "qwen3-reranker",
        "port": 10001,
        "gpu_memory_utilization": 0.9,
        "model_type": "decoder_reranker",
        "enable_lora": False,
        "max_loras": 4,
        "max_lora_rank": 64,
        "trust_remote_code": False,
        "model_family": "Qwen/Qwen3-Reranker-4B",
    }

    if framework == "vllm":
        argv = build_vllm_server_argv(
            parse_launch_config({"framework": "vllm"}),
            **common,
        )
        assert argv[argv.index("--runner") + 1] == "pooling"
        assert "--hf-overrides" in argv
    else:
        argv = build_sglang_server_argv(
            parse_launch_config({"framework": "sglang"}),
            chat_template=(
                "/sgl-workspace/sglang/examples/chat_template/"
                "qwen3_reranker.jinja"
            ),
            **common,
        )
        assert "--disable-radix-cache" in argv
        assert "--chat-template" in argv


def test_vllm_omits_optional_false_and_null_flags() -> None:
    argv = build_vllm_server_argv(
        parse_launch_config({"framework": "vllm"}),
        model_path="/models/model",
        served_model_name="model",
        port=10001,
        gpu_memory_utilization=0.9,
        model_type="llm",
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        trust_remote_code=False,
    )

    assert "--quantization" not in argv
    assert "--max-model-len" not in argv
    assert "--max-num-seqs" not in argv
    assert "--enable-expert-parallel" not in argv
    assert "--enforce-eager" not in argv
    assert "--trust-remote-code" not in argv
    assert "--enable-lora" not in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "auto"


def test_sglang_hpc_ops_adds_required_page_size() -> None:
    argv = build_sglang_server_argv(
        parse_launch_config(
            {
                "framework": "sglang",
                "attention_backend": "hpc_ops",
            }
        ),
        model_path="/models/model",
        served_model_name="model",
        port=10001,
        gpu_memory_utilization=0.9,
        model_type="llm",
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        chat_template=None,
        trust_remote_code=False,
    )

    backend_index = argv.index("--attention-backend")
    assert argv[backend_index : backend_index + 4] == (
        "--attention-backend",
        "hpc_ops",
        "--page-size",
        "64",
    )


def test_sglang_0517_embedding_lora_maps_every_supported_launch_field_to_exact_argv() -> None:
    config = parse_launch_config(
        {
            "framework": "sglang",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 3,
            "data_parallel_size": 4,
            "expert_parallel_size": 2,
            "max_context_length": 65536,
            "max_concurrent_requests": 128,
            "dtype": "float16",
            "quantization": "gptq",
            "kv_cache_dtype": "fp8_e5m2",
            "attention_backend": "flashinfer",
        }
    )

    argv = build_sglang_server_argv(
        config,
        model_path="/models/private model;touch /tmp/canary",
        served_model_name="served;name",
        port=10002,
        gpu_memory_utilization=0.77,
        model_type="embedding",
        enable_lora=True,
        max_loras=3,
        max_lora_rank=32,
        chat_template=None,
        trust_remote_code=True,
        model_family="Qwen3-Reranker-4B",
    )

    assert argv == (
        "sglang",
        "serve",
        "--model-path",
        "/models/private model;touch /tmp/canary",
        "--served-model-name",
        "served;name",
        "--host",
        "0.0.0.0",
        "--port",
        "10002",
        "--mem-fraction-static",
        "0.77",
        "--tp-size",
        "2",
        "--pp-size",
        "3",
        "--dp-size",
        "4",
        "--ep-size",
        "2",
        "--disable-overlap-schedule",
        "--context-length",
        "65536",
        "--max-running-requests",
        "128",
        "--dtype",
        "float16",
        "--quantization",
        "gptq",
        "--kv-cache-dtype",
        "fp8_e5m2",
        "--attention-backend",
        "flashinfer",
        "--is-embedding",
        "--trust-remote-code",
        "--enable-lora",
        "--max-loaded-loras",
        "3",
        "--max-loras-per-batch",
        "3",
        "--max-lora-rank",
        "32",
        "--lora-target-modules",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


_DOCKER_PORTS_COMMAND = ["docker", "ps", "--format", "{{.Ports}}"]
_DOCKER_ENDPOINT_COMMAND = [
    "docker",
    "context",
    "inspect",
    "--format",
    "{{json .Endpoints.docker.Host}}",
]


def _port_probe_command_runner(
    ports_output: str,
    *,
    endpoint: str = "ssh://docker.example",
    endpoint_success: bool = True,
):
    def run(command: list[str]):
        if command == _DOCKER_PORTS_COMMAND:
            return True, ports_output
        if command == _DOCKER_ENDPOINT_COMMAND:
            return (
                (True, json.dumps(endpoint))
                if endpoint_success
                else (False, "endpoint unavailable")
            )
        raise AssertionError(f"unexpected Docker command: {command!r}")

    return run


def _set_probe_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform: str,
    containerized: bool = False,
) -> None:
    monkeypatch.setattr(
        docker_deployer_module,
        "_native_host_platform",
        lambda: platform,
        raising=False,
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_container_runtime_marker",
        lambda: containerized,
        raising=False,
    )


@pytest.mark.parametrize(
    "model_type",
    ["rerank", "reranker", "decoder_reranker"],
)
def test_sglang_0517_rejects_reranker_lora_without_protocol_adapter_selector(
    model_type: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"SGLang v0\.5\.17 /v1/rerank.*LoRA adapter selector",
    ):
        build_sglang_server_argv(
            parse_launch_config({"framework": "sglang"}),
            model_path="/models/reranker",
            served_model_name="reranker",
            port=10002,
            gpu_memory_utilization=0.9,
            model_type=model_type,
            enable_lora=True,
            max_loras=4,
            max_lora_rank=64,
            chat_template=None,
            trust_remote_code=False,
        )


def test_sglang_generic_reranker_does_not_receive_qwen_chat_template() -> None:
    argv = build_sglang_server_argv(
        parse_launch_config({"framework": "sglang"}),
        model_path="/models/bge-reranker-v2-m3",
        served_model_name="generic-reranker",
        port=10002,
        gpu_memory_utilization=0.9,
        model_type="reranker",
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        chat_template="/templates/qwen3_reranker_no_think.jinja",
        trust_remote_code=False,
        model_family="BAAI/bge-reranker-v2-m3",
    )

    assert "--disable-radix-cache" in argv
    assert "--chat-template" not in argv


def test_sglang_pipeline_parallelism_disables_overlap_schedule() -> None:
    argv = build_sglang_server_argv(
        parse_launch_config(
            {
                "framework": "sglang",
                "pipeline_parallel_size": 2,
            }
        ),
        model_path="/models/model",
        served_model_name="model",
        port=10002,
        gpu_memory_utilization=0.9,
        model_type="llm",
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        chat_template=None,
        trust_remote_code=False,
    )

    assert "--disable-overlap-schedule" in argv


def test_sglang_omits_optional_null_flags_and_has_no_shell_tokens() -> None:
    argv = build_sglang_server_argv(
        parse_launch_config({"framework": "sglang"}),
        model_path="/models/model",
        served_model_name="model",
        port=10001,
        gpu_memory_utilization=0.9,
        model_type="embedding",
        enable_lora=False,
        max_loras=4,
        max_lora_rank=64,
        chat_template=None,
        trust_remote_code=False,
    )

    assert argv[:2] == ("sglang", "serve")
    assert "bash" not in argv
    assert "-c" not in argv
    assert "--quantization" not in argv
    assert "--attention-backend" not in argv
    assert "--context-length" not in argv
    assert "--max-running-requests" not in argv
    assert "--is-embedding" in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "auto"


def test_sglang_rejects_served_model_name_with_colon_before_container_start() -> None:
    with pytest.raises(ValueError, match="SGLang served_model_name must not contain ':'"):
        build_sglang_server_argv(
            parse_launch_config({"framework": "sglang"}),
            model_path="/models/model",
            served_model_name="organization:model",
            port=10001,
            gpu_memory_utilization=0.9,
            model_type="llm",
            enable_lora=False,
            max_loras=4,
            max_lora_rank=64,
            chat_template=None,
            trust_remote_code=False,
        )


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_docker_command_uses_exact_gpu_list_and_never_invokes_shell(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    deployer = DockerDeployer()
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address="127.0.0.1",
        ),
    )
    server_argv = (
        ("vllm", "serve", "/models/model")
        if framework == "vllm"
        else ("sglang", "serve", "--model-path", "/models/model")
    )

    create = (
        deployer.create_vllm_container
        if framework == "vllm"
        else deployer.create_sglang_container
    )
    create(
        container_name=f"{framework}-test",
        port=10001,
        gpu_ids=(7, 3),
        server_argv=server_argv,
    )

    command = captured["command"]
    assert "--pull=missing" in command
    assert command[command.index("--gpus") + 1] == "device=7,3"
    assert command[command.index("--name") + 1] == f"{framework}-test"
    assert command[command.index("-p") + 1] == "127.0.0.1:10001:10001"
    assert "bash" not in command
    assert "sh" not in command
    assert "-c" not in command
    assert command[-len(server_argv) + 1 :] == list(server_argv[1:])


@pytest.mark.parametrize(
    ("host_bind_address", "published_port"),
    [
        ("localhost", "127.0.0.1:10001:10001"),
        ("0.0.0.0", "0.0.0.0:10001:10001"),
        ("::1", "[::1]:10001:10001"),
    ],
)
def test_new_container_publish_address_is_normalized_at_creation_time(
    monkeypatch: pytest.MonkeyPatch,
    host_bind_address: str,
    published_port: str,
) -> None:
    deployer = DockerDeployer()
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address=host_bind_address,
        ),
    )

    deployer.create_sglang_container(
        container_name="sglang-bind-test",
        port=10001,
        gpu_ids=(0,),
        server_argv=("sglang", "serve", "--model-path", "/models/model"),
    )

    command = captured["command"]
    assert command[command.index("-p") + 1] == published_port
    assert "bash" not in command
    assert "sh" not in command
    assert "-c" not in command


@pytest.mark.parametrize(
    ("builder_name", "builder_kwargs"),
    [
        (
            "create_xinference_container",
            {
                "container_name": "legacy-xinference-bind-test",
                "port": 9997,
                "gpu_id": 0,
                "model_name": "model",
                "model_uid": "served-model",
                "model_path": "/models/model",
                "model_type": "llm",
            },
        ),
        (
            "_legacy_create_vllm_container",
            {
                "container_name": "legacy-vllm-bind-test",
                "port": 9997,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
        ),
        (
            "_legacy_create_sglang_container",
            {
                "container_name": "legacy-sglang-bind-test",
                "port": 9997,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
        ),
    ],
)
def test_legacy_container_builders_publish_on_configured_address(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    builder_kwargs: dict[str, object],
) -> None:
    deployer = DockerDeployer()
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address="localhost",
        ),
    )

    getattr(deployer, builder_name)(**builder_kwargs)

    command = captured["command"]
    assert command[command.index("-p") + 1] == "127.0.0.1:9997:9997"


def test_container_creation_rejects_invalid_publish_address_before_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    docker_calls: list[list[str]] = []
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            docker_calls.append(command) is None,
            "ok",
        ),
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address="not-an-ip-address",
        ),
    )

    with pytest.raises(ValueError, match="HOST_BIND_ADDRESS has invalid address"):
        deployer.create_vllm_container(
            container_name="vllm-invalid-bind",
            port=10001,
            gpu_ids=(0,),
            server_argv=("vllm", "serve", "/models/model"),
        )

    assert docker_calls == []


@pytest.mark.parametrize(
    "ports_output",
    [
        "0.0.0.0:10001->8000/tcp",
        "127.0.0.1:10001->8000/tcp",
        "192.0.2.10:10001->8000/tcp",
        ":::10001->8000/tcp",
        "[::]:10001->8000/tcp",
        "[::1]:10001->8000/tcp",
    ],
)
def test_port_probe_recognizes_ipv4_and_ipv6_publish_mappings(
    monkeypatch: pytest.MonkeyPatch,
    ports_output: str,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner(ports_output),
    )

    assert deployer._is_port_in_use(10001) is True
    assert deployer._is_port_in_use(10002) is False


def test_port_probe_ignores_unpublished_container_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner("10001/tcp, 10002/udp"),
    )

    assert deployer._is_port_in_use(10001) is False


def test_port_probe_fails_closed_when_docker_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    local_probe_calls: list[int] = []
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda _command: (False, "daemon unavailable"),
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "_is_local_port_in_use",
        lambda port: local_probe_calls.append(port) is not None,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Docker port mapping query failed"):
        deployer._is_port_in_use(10001)

    assert local_probe_calls == []


@pytest.mark.parametrize(
    "ports_output",
    [
        "0.0.0.0:10001-10003->8000-8002/tcp",
        "127.0.0.1:10001-10003->8000-8002/tcp",
        "[::1]:10001-10003->8000-8002/tcp",
    ],
)
def test_port_probe_recognizes_every_port_in_published_host_range(
    monkeypatch: pytest.MonkeyPatch,
    ports_output: str,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner(ports_output),
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
        raising=False,
    )

    assert [deployer._is_port_in_use(port) for port in range(10001, 10005)] == [
        True,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "ports_output",
    [
        "0.0.0.0:not-a-port->80/tcp",
        "host.invalid:10001->80/tcp",
        "[::1:10001->80/tcp",
        "0.0.0.0:10001->not-a-port/tcp",
        "0.0.0.0:10001->80/unknown",
        "0.0.0.0:10001-10003->80-81/tcp",
    ],
)
def test_port_probe_rejects_malformed_or_unknown_published_segments(
    monkeypatch: pytest.MonkeyPatch,
    ports_output: str,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda _command: (True, ports_output),
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Docker port mapping output is invalid"):
        deployer._is_port_in_use(10001)


def test_port_probe_validates_every_published_segment_before_reporting_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda _command: (
            True,
            "0.0.0.0:10001->80/tcp, host.invalid:10002->81/tcp",
        ),
    )

    with pytest.raises(RuntimeError, match="Docker port mapping output is invalid"):
        deployer._is_port_in_use(10001)


def _install_fake_bind_socket(
    monkeypatch: pytest.MonkeyPatch,
    bind_error: OSError | None,
) -> list[tuple[str, tuple[str, int]]]:
    calls: list[tuple[str, tuple[str, int]]] = []

    class ProbeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def bind(self, address: tuple[str, int]) -> None:
            calls.append(("bind", address))
            if bind_error is not None:
                raise bind_error

    monkeypatch.setattr(
        docker_deployer_module,
        "socket",
        SimpleNamespace(
            AF_INET=2,
            AF_INET6=10,
            SOCK_STREAM=1,
            socket=lambda _family, _kind: ProbeSocket(),
        ),
        raising=False,
    )
    return calls


@pytest.mark.parametrize(
    ("bind_error", "expected"),
    [
        (None, False),
        (OSError(errno.EADDRINUSE, "in use"), True),
        (OSError(10048, "windows address in use"), True),
    ],
)
def test_proven_host_probe_uses_bind_to_check_non_docker_processes(
    monkeypatch: pytest.MonkeyPatch,
    bind_error: OSError | None,
    expected: bool,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner(
            "",
            endpoint="unix:///var/run/docker.sock",
        ),
    )
    _set_probe_platform(monkeypatch, platform="linux")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(host_bind_address="127.0.0.1"),
    )
    calls = _install_fake_bind_socket(monkeypatch, bind_error)

    assert deployer._is_port_in_use(10001) is expected
    assert calls == [("bind", ("127.0.0.1", 10001))]


def test_proven_host_probe_fails_closed_on_unexpected_bind_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner(
            "",
            endpoint="unix:///var/run/docker.sock",
        ),
    )
    _set_probe_platform(monkeypatch, platform="linux")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(host_bind_address="127.0.0.1"),
    )
    _install_fake_bind_socket(
        monkeypatch,
        OSError(errno.EACCES, "permission denied"),
    )

    with pytest.raises(RuntimeError, match="Local port availability probe failed"):
        deployer._is_port_in_use(10001)


def test_container_network_namespace_never_uses_local_socket_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner(
            "",
            endpoint="unix:///var/run/docker.sock",
        ),
    )
    _set_probe_platform(monkeypatch, platform="linux", containerized=True)
    monkeypatch.setenv("container", "docker")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "_is_local_port_in_use",
        lambda _port: (_ for _ in ()).throw(
            AssertionError("local socket probe must stay disabled")
        ),
        raising=False,
    )

    assert deployer._is_port_in_use(10001) is False


def test_native_windows_desktop_probe_detects_real_non_docker_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner(
            "",
            endpoint="npipe:////./pipe/dockerDesktopLinuxEngine",
        ),
    )
    _set_probe_platform(monkeypatch, platform="windows")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "get_settings",
        lambda: SimpleNamespace(host_bind_address="127.0.0.1"),
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        assert deployer._is_port_in_use(listener.getsockname()[1]) is True


def test_native_macos_desktop_uses_local_bind_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner(
            "",
            endpoint="unix:///Users/test/.docker/run/docker.sock",
        ),
    )
    _set_probe_platform(monkeypatch, platform="macos")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
    )
    calls: list[int] = []

    def local_port_in_use(port: int) -> bool:
        calls.append(port)
        return True

    monkeypatch.setattr(
        docker_deployer_module,
        "_is_local_port_in_use",
        local_port_in_use,
    )

    assert deployer._is_port_in_use(10001) is True
    assert calls == [10001]


@pytest.mark.parametrize(
    "endpoint",
    [
        "ssh://docker.example",
        "tcp://192.0.2.10:2376",
    ],
)
def test_remote_docker_endpoint_never_uses_local_bind_probe(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner("", endpoint=endpoint),
    )
    _set_probe_platform(monkeypatch, platform="windows")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
    )
    monkeypatch.setattr(
        docker_deployer_module,
        "_is_local_port_in_use",
        lambda _port: (_ for _ in ()).throw(
            AssertionError("remote Docker endpoint must not use a local bind probe")
        ),
    )

    assert deployer._is_port_in_use(10001) is False


def test_native_host_port_probe_fails_closed_when_endpoint_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner("", endpoint_success=False),
    )
    _set_probe_platform(monkeypatch, platform="windows")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="Docker endpoint query failed"):
        deployer._is_port_in_use(10001)


def test_native_host_port_probe_fails_closed_for_unknown_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        _port_probe_command_runner("", endpoint="opaque://daemon"),
    )
    _set_probe_platform(monkeypatch, platform="windows")
    monkeypatch.setattr(
        docker_deployer_module,
        "_has_proven_host_network_namespace",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="Docker endpoint is invalid"):
        deployer._is_port_in_use(10001)


@pytest.mark.parametrize(
    ("container_environment", "container_marker"),
    [
        ("podman", None),
        (None, "/run/.containerenv"),
    ],
)
def test_host_namespace_proof_rejects_oci_container_markers_before_proc_probe(
    monkeypatch: pytest.MonkeyPatch,
    container_environment: str | None,
    container_marker: str | None,
) -> None:
    if container_environment is None:
        monkeypatch.delenv("container", raising=False)
    else:
        monkeypatch.setenv("container", container_environment)
    monkeypatch.setattr(
        docker_deployer_module.os.path,
        "exists",
        lambda path: path == container_marker,
    )
    proc_calls: list[str] = []
    monkeypatch.setattr(
        docker_deployer_module.os,
        "readlink",
        lambda path: proc_calls.append(path) or "net:[1]",
    )

    assert docker_deployer_module._has_proven_host_network_namespace() is False
    assert proc_calls == []


@pytest.mark.parametrize(
    ("builder_name", "builder_kwargs"),
    [
        (
            "create_xinference_container",
            {
                "container_name": "legacy-xinference-invalid-bind",
                "port": 9997,
                "gpu_id": 0,
                "model_name": "model",
                "model_uid": "served-model",
                "model_path": "/models/model",
                "model_type": "llm",
            },
        ),
        (
            "_legacy_create_vllm_container",
            {
                "container_name": "legacy-vllm-invalid-bind",
                "port": 9997,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
        ),
        (
            "_legacy_create_sglang_container",
            {
                "container_name": "legacy-sglang-invalid-bind",
                "port": 9997,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
        ),
    ],
)
def test_legacy_builder_validates_publish_address_before_container_removal(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    builder_kwargs: dict[str, object],
) -> None:
    deployer = DockerDeployer()
    mutations: list[str] = []
    monkeypatch.setattr(deployer, "remove_container", mutations.append)
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address="not-an-ip-address",
        ),
    )

    with pytest.raises(ValueError, match="HOST_BIND_ADDRESS has invalid address"):
        getattr(deployer, builder_name)(**builder_kwargs)

    assert mutations == []


def test_docker_validates_request_before_existing_container_mutation() -> None:
    deployer = DockerDeployer()
    mutations: list[str] = []
    deployer.remove_container = mutations.append  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="GPU IDs"):
        deployer.create_vllm_container(
            container_name="vllm-test",
            port=10001,
            gpu_ids=(0, 0),
            server_argv=("vllm", "serve", "/models/model"),
        )

    assert mutations == []


def test_container_create_allows_pinned_cold_image_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    calls: list[tuple[list[str], int]] = []

    def run(command: list[str], timeout: int = 30):
        calls.append((command, timeout))
        return True, "created"

    monkeypatch.setattr(deployer, "_run_command", run)

    assert deployer._run_container_create(
        ["docker", "run", "--pull=missing", "pinned@example"],
        "cold-image",
        expected_image="pinned@example",
        expected_labels={},
    ) == (True, "created")
    assert calls == [
        (["docker", "run", "--pull=missing", "pinned@example"], 3600)
    ]


def test_container_create_timeout_accepts_exact_expected_immutable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    expected_labels = {
        "com.trainfactory.managed": "deployment-replica",
        "com.trainfactory.project": "trainfactory",
        "com.trainfactory.deployment-id": "deployment-1",
        "com.trainfactory.replica-id": "replica-1",
    }
    record = {
        "Id": "a" * 64,
        "Name": "/managed-timeout",
        "Config": {
            "Image": "pinned@example",
            "Labels": expected_labels,
        },
    }
    calls: list[tuple[list[str], int]] = []

    def run(command: list[str], timeout: int = 30):
        calls.append((command, timeout))
        if command[:2] == ["docker", "run"]:
            return False, "Command timed out"
        return True, json.dumps(record)

    monkeypatch.setattr(deployer, "_run_command", run)

    success, message = deployer._run_container_create(
        ["docker", "run", "--pull=missing", "pinned@example"],
        "managed-timeout",
        expected_image="pinned@example",
        expected_labels=expected_labels,
    )

    assert success is True
    assert "image pull exceeded 3600s timeout" in message
    assert calls == [
        (["docker", "run", "--pull=missing", "pinned@example"], 3600),
        (
            [
                "docker",
                "inspect",
                "--format",
                "{{json .}}",
                "managed-timeout",
            ],
            15,
        ),
    ]


def test_container_create_timeout_rejects_foreign_same_name_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    expected_labels = {
        "com.trainfactory.managed": "deployment-replica",
        "com.trainfactory.project": "trainfactory",
        "com.trainfactory.deployment-id": "deployment-1",
        "com.trainfactory.replica-id": "replica-1",
    }
    record = {
        "Id": "b" * 64,
        "Name": "/managed-timeout",
        "Config": {
            "Image": "pinned@example",
            "Labels": {
                **expected_labels,
                "com.trainfactory.project": "foreign",
            },
        },
    }

    def run(command: list[str], timeout: int = 30):
        if command[:2] == ["docker", "run"]:
            return False, "Command timed out"
        return True, json.dumps(record)

    monkeypatch.setattr(deployer, "_run_command", run)

    assert deployer._run_container_create(
        ["docker", "run", "--pull=missing", "pinned@example"],
        "managed-timeout",
        expected_image="pinned@example",
        expected_labels=expected_labels,
    ) == (False, "Command timed out")


@pytest.mark.parametrize(
    ("builder_name", "builder_kwargs", "image_attribute"),
    [
        (
            "_legacy_create_vllm_container",
            {
                "container_name": "legacy-vllm-timeout",
                "port": 9998,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
            "vllm_image",
        ),
        (
            "_legacy_create_sglang_container",
            {
                "container_name": "legacy-sglang-timeout",
                "port": 9999,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
            "sglang_image",
        ),
    ],
)
def test_legacy_timeout_rejects_unlabeled_same_image_foreign_container(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    builder_kwargs: dict[str, object],
    image_attribute: str,
) -> None:
    deployer = DockerDeployer()
    container_name = str(builder_kwargs["container_name"])
    record = {
        "Id": "c" * 64,
        "Name": f"/{container_name}",
        "Config": {
            "Image": getattr(deployer, image_attribute),
            "Labels": {},
        },
    }

    def run(command: list[str], timeout: int = 30):
        if command[:2] == ["docker", "run"]:
            return False, "Command timed out"
        return True, json.dumps(record)

    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(deployer, "_run_command", run)
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address="127.0.0.1",
        ),
    )

    success, message, command = getattr(deployer, builder_name)(**builder_kwargs)

    assert success is False
    assert message == "Command timed out"
    assert "--label" in command
    assert "io.trainfactory.create-attempt=" in command


@pytest.mark.parametrize(
    ("builder_name", "builder_kwargs", "image_attribute", "managed"),
    [
        (
            "create_xinference_container",
            {
                "container_name": "xinference-identity",
                "port": 9997,
                "gpu_id": 0,
                "model_name": "model",
                "model_uid": "served-model",
                "model_path": "/models/model",
                "model_type": "llm",
                "deployment_id": "deployment-1",
                "replica_id": "replica-1",
            },
            "image",
            True,
        ),
        (
            "_legacy_create_vllm_container",
            {
                "container_name": "legacy-vllm-identity",
                "port": 9998,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
            "vllm_image",
            False,
        ),
        (
            "_legacy_create_sglang_container",
            {
                "container_name": "legacy-sglang-identity",
                "port": 9999,
                "gpu_id": 0,
                "model_path": "/models/model",
                "model_name": "model",
                "model_type": "llm",
            },
            "sglang_image",
            False,
        ),
        (
            "create_vllm_container",
            {
                "container_name": "vllm-identity",
                "port": 10000,
                "gpu_ids": (0,),
                "server_argv": ("vllm", "serve", "/models/model"),
                "deployment_id": "deployment-1",
                "replica_id": "replica-1",
            },
            "vllm_image",
            True,
        ),
        (
            "create_sglang_container",
            {
                "container_name": "sglang-identity",
                "port": 10001,
                "gpu_ids": (0,),
                "server_argv": (
                    "sglang",
                    "serve",
                    "--model-path",
                    "/models/model",
                ),
                "deployment_id": "deployment-1",
                "replica_id": "replica-1",
            },
            "sglang_image",
            True,
        ),
    ],
)
def test_all_container_builders_pass_expected_timeout_identity(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    builder_kwargs: dict[str, object],
    image_attribute: str,
    managed: bool,
) -> None:
    deployer = DockerDeployer()
    captured: dict[str, object] = {}

    def capture(
        command: list[str],
        _name: str,
        *,
        expected_image: str,
        expected_labels: dict[str, str],
    ) -> tuple[bool, str]:
        captured["command"] = command
        captured["image"] = expected_image
        captured["labels"] = expected_labels
        return True, "ok"

    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(deployer, "_run_container_create", capture)
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address="127.0.0.1",
        ),
    )

    getattr(deployer, builder_name)(**builder_kwargs)

    assert captured["image"] == getattr(deployer, image_attribute)
    expected_labels = dict(captured["labels"])
    create_attempt = expected_labels.pop("io.trainfactory.create-attempt")
    assert isinstance(create_attempt, str)
    assert re.fullmatch(r"[0-9a-f]{32}", create_attempt) is not None
    assert expected_labels == (
        deployer._managed_replica_labels("deployment-1", "replica-1")
        if managed
        else {}
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert f"io.trainfactory.create-attempt={create_attempt}" in command


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_valid_container_create_never_removes_a_preexisting_name(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    deployer = DockerDeployer()
    mutations: list[str] = []
    monkeypatch.setattr(deployer, "remove_container", mutations.append)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda _command, _name, **_identity: (False, "name already exists"),
    )
    server_argv = (
        ("vllm", "serve", "/models/model")
        if framework == "vllm"
        else (
            "sglang",
            "serve",
            "--model-path",
            "/models/model",
        )
    )

    create = (
        deployer.create_vllm_container
        if framework == "vllm"
        else deployer.create_sglang_container
    )
    success, message, _command = create(
        container_name=f"{framework}-foreign",
        port=10001,
        gpu_ids=(0,),
        server_argv=server_argv,
    )

    assert success is False
    assert message == "name already exists"
    assert mutations == []


def test_managed_replica_create_adds_exact_ownership_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    captured: list[str] = []
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "trainfactory")
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.extend(command) is None,
            "ok",
        ),
    )

    success, _message, _command = deployer.create_vllm_container(
        container_name="vllm-owned",
        port=10001,
        gpu_ids=(0,),
        server_argv=("vllm", "serve", "/models/model"),
        deployment_id="deployment-1",
        replica_id="replica-1",
    )

    labels = [
        captured[index + 1]
        for index, value in enumerate(captured)
        if value == "--label"
    ]
    assert success is True
    assert labels[:4] == [
        "com.trainfactory.managed=deployment-replica",
        "com.trainfactory.project=trainfactory",
        "com.trainfactory.deployment-id=deployment-1",
        "com.trainfactory.replica-id=replica-1",
    ]
    assert re.fullmatch(
        r"io\.trainfactory\.create-attempt=[0-9a-f]{32}",
        labels[-1],
    ) is not None


def test_managed_legacy_xinference_create_adds_exact_ownership_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    captured: list[str] = []
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "trainfactory")
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.extend(command) is None,
            "ok",
        ),
    )

    success, _message, _command = deployer.create_xinference_container(
        container_name="xinference-owned",
        port=9997,
        gpu_id=0,
        model_name="model",
        model_uid="served-model",
        model_path="/models/model",
        model_type="llm",
        deployment_id="deployment-1",
        replica_id="legacy",
    )

    labels = [
        captured[index + 1]
        for index, value in enumerate(captured)
        if value == "--label"
    ]
    assert success is True
    assert labels[:4] == [
        "com.trainfactory.managed=deployment-replica",
        "com.trainfactory.project=trainfactory",
        "com.trainfactory.deployment-id=deployment-1",
        "com.trainfactory.replica-id=legacy",
    ]
    assert re.fullmatch(
        r"io\.trainfactory\.create-attempt=[0-9a-f]{32}",
        labels[-1],
    ) is not None


@pytest.mark.parametrize("invalid_port", [0, 65536, True, "9997"])
def test_xinference_create_rejects_invalid_port_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    invalid_port: object,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "remove_container",
        lambda _name: pytest.fail("invalid port must fail before mutation"),
    )

    with pytest.raises(ValueError, match="container port is invalid"):
        deployer.create_xinference_container(
            container_name="xinference-invalid-port",
            port=invalid_port,
            gpu_id=0,
            model_name="model",
            model_uid="served-model",
            model_path="/models/model",
            model_type="llm",
        )


@pytest.mark.parametrize("invalid_gpu_id", [-1, True, "0"])
def test_xinference_create_rejects_invalid_gpu_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    invalid_gpu_id: object,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "remove_container",
        lambda _name: pytest.fail("invalid GPU must fail before mutation"),
    )

    with pytest.raises(ValueError, match="GPU IDs must be non-negative integers"):
        deployer.create_xinference_container(
            container_name="xinference-invalid-gpu",
            port=9997,
            gpu_id=invalid_gpu_id,
            model_name="model",
            model_uid="served-model",
            model_path="/models/model",
            model_type="llm",
        )


def test_legacy_container_identity_accepts_only_unlabeled_or_exact_owned_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    container_id = "b" * 64
    record = {
        "Id": container_id,
        "Name": "/legacy-owned",
        "Config": {"Labels": {}},
    }
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda _command, **_kwargs: (True, json.dumps(record)),
    )

    assert deployer.get_legacy_managed_container_id(
        "legacy-owned",
        deployment_id="deployment-1",
        replica_id="legacy",
    ) == container_id

    record["Config"]["Labels"] = {
        "com.trainfactory.managed": "deployment-replica",
        "com.trainfactory.project": "foreign",
        "com.trainfactory.deployment-id": "deployment-1",
        "com.trainfactory.replica-id": "legacy",
    }
    with pytest.raises(RuntimeError, match="identity is invalid"):
        deployer.get_legacy_managed_container_id(
            "legacy-owned",
            deployment_id="deployment-1",
            replica_id="legacy",
        )


@pytest.mark.parametrize(
    ("model_name", "expects_qwen_overrides"),
    [
        ("Qwen3-Reranker-4B", True),
        ("BAAI/bge-reranker-v2-m3", False),
    ],
)
def test_legacy_vllm_helper_also_uses_026_pooling_contract(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    expects_qwen_overrides: bool,
) -> None:
    deployer = DockerDeployer()
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(allow_model_remote_code=False),
    )

    deployer._legacy_create_vllm_container(
        container_name="legacy-reranker",
        port=10001,
        gpu_id=0,
        model_path="/models/reranker",
        model_name=model_name,
        model_type="reranker",
    )

    command = captured["command"]
    assert "--task" not in command
    assert command[command.index("--runner") + 1] == "pooling"
    assert ("--hf-overrides" in command) is expects_qwen_overrides
    assert ("--chat-template" in command) is expects_qwen_overrides


def test_container_identity_rejects_foreign_project_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    container_id = "a" * 64
    record = {
        "Id": container_id,
        "Name": "/vllm-owned",
        "Config": {
            "Labels": {
                "com.trainfactory.managed": "deployment-replica",
                "com.trainfactory.project": "foreign",
                "com.trainfactory.deployment-id": "deployment-1",
                "com.trainfactory.replica-id": "replica-1",
            }
        },
    }
    mutations: list[list[str]] = []
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "trainfactory")
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda command, **_kwargs: (
            (True, json.dumps(record))
            if command[:2] == ["docker", "inspect"]
            else (mutations.append(command) is None, "")
        ),
    )

    with pytest.raises(RuntimeError, match="identity is invalid"):
        deployer.get_managed_container_id(
            "vllm-owned",
            deployment_id="deployment-1",
            replica_id="replica-1",
        )

    assert mutations == []
