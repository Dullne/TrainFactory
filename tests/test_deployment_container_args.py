from __future__ import annotations

from types import SimpleNamespace

import pytest

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
@pytest.mark.parametrize("runtime", ["xinference", "vllm", "sglang"])
def test_container_creation_rejects_invalid_publish_address_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
) -> None:
    deployer = DockerDeployer()
    docker_calls: list[list[str]] = []
    mutations: list[str] = []
    monkeypatch.setattr(deployer, "remove_container", mutations.append)
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
        if runtime == "xinference":
            deployer.create_xinference_container(
                container_name="xinference-invalid-bind",
                port=10001,
                gpu_id=0,
                model_name="model",
                model_uid="model",
                model_path="/models/model",
                model_type="embedding",
            )
        elif runtime == "vllm":
            deployer.create_vllm_container(
                container_name="vllm-invalid-bind",
                port=10001,
                gpu_ids=(0,),
                server_argv=("vllm", "serve", "/models/model"),
            )
        else:
            deployer.create_sglang_container(
                container_name="sglang-invalid-bind",
                port=10001,
                gpu_ids=(0,),
                server_argv=(
                    "sglang",
                    "serve",
                    "--model-path",
                    "/models/model",
                ),
            )

    assert docker_calls == []
    assert mutations == []


@pytest.mark.parametrize(
    ("host_ip", "host_port", "target_port"),
    [
        ("127.0.0.1", 10001, 80),
        ("::1", 10002, 443),
    ],
)
def test_port_detection_filters_loopback_bindings_by_host_port(
    monkeypatch: pytest.MonkeyPatch,
    host_ip: str,
    host_port: int,
    target_port: int,
) -> None:
    deployer = DockerDeployer()
    calls: list[list[str]] = []

    def run(command: list[str], timeout: int = 30):
        del timeout
        calls.append(command)
        requested_port = int(
            command[command.index("--filter") + 1].removeprefix("publish=")
        )
        published_binding = f"{host_ip}:{host_port}->{target_port}/tcp"
        return (
            True,
            f"container-for-{published_binding}" if requested_port == host_port else "",
        )

    monkeypatch.setattr(deployer, "_run_command", run)

    assert deployer._is_port_in_use(host_port) is True
    assert calls == [
        [
            "docker",
            "ps",
            "--filter",
            f"publish={host_port}",
            "--format",
            "{{.ID}}",
        ]
    ]


@pytest.mark.parametrize(
    ("success", "output"),
    [(True, ""), (False, "docker unavailable")],
)
def test_port_detection_treats_empty_or_failed_query_as_free(
    monkeypatch: pytest.MonkeyPatch,
    success: bool,
    output: str,
) -> None:
    deployer = DockerDeployer()
    monkeypatch.setattr(
        deployer,
        "_run_command",
        lambda _command, timeout=30: (success, output),
    )

    assert deployer._is_port_in_use(10001) is False


def test_find_available_port_skips_docker_published_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployer = DockerDeployer()
    queried_ports: list[int] = []

    def run(command: list[str], timeout: int = 30):
        del timeout
        publish_filter = command[command.index("--filter") + 1]
        port = int(publish_filter.removeprefix("publish="))
        queried_ports.append(port)
        return True, "occupied-container" if port == 10001 else ""

    monkeypatch.setattr(deployer, "_run_command", run)
    monkeypatch.setattr(deployer, "_get_db_reserved_ports", lambda: set())

    allocated = deployer.find_available_port(
        start=10001,
        end=10003,
        owner_token="request-1",
    )

    assert allocated == 10002
    assert queried_ports == [10001, 10002]
    assert deployer._reserved_ports == {10002: "request-1"}


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
    ) == (True, "created")
    assert calls == [
        (["docker", "run", "--pull=missing", "pinned@example"], 3600)
    ]
@pytest.mark.parametrize("runtime", ["xinference", "vllm", "sglang"])
def test_valid_container_create_removes_preexisting_name_after_validation(
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
) -> None:
    deployer = DockerDeployer()
    events: list[tuple[str, str]] = []

    def publish(port: int) -> str:
        events.append(("publish", str(port)))
        return f"127.0.0.1:{port}:{port}"

    def remove(container_name: str) -> bool:
        events.append(("remove", container_name))
        return True

    def create(_command: list[str], container_name: str):
        events.append(("create", container_name))
        return True, "created"

    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer._published_container_port",
        publish,
    )
    monkeypatch.setattr(deployer, "remove_container", remove)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        create,
    )
    monkeypatch.setattr(
        "train_factory.deployment.docker_deployer.get_settings",
        lambda: SimpleNamespace(
            allow_model_remote_code=False,
            host_bind_address="127.0.0.1",
        ),
    )
    container_name = f"{runtime}-stale"

    if runtime == "xinference":
        success, _message, _command = deployer.create_xinference_container(
            container_name=container_name,
            port=10001,
            gpu_id=0,
            model_name="model",
            model_uid="model",
            model_path="/models/model",
            model_type="embedding",
        )
    elif runtime == "vllm":
        success, _message, _command = deployer.create_vllm_container(
            container_name=container_name,
            port=10001,
            gpu_ids=(0,),
            server_argv=("vllm", "serve", "/models/model"),
        )
    else:
        success, _message, _command = deployer.create_sglang_container(
            container_name=container_name,
            port=10001,
            gpu_ids=(0,),
            server_argv=(
                "sglang",
                "serve",
                "--model-path",
                "/models/model",
            ),
        )

    assert success is True
    assert events == [
        ("publish", "10001"),
        ("remove", container_name),
        ("create", container_name),
    ]
