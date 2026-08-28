from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from train_factory.deployment.launch_config import (
    SglangLaunchConfig,
    VllmLaunchConfig,
    normalize_launch_config,
    parse_launch_config,
)
from train_factory.api.routes.deployment_routes import (
    CreateContainerDeploymentRequest,
    CreateDeploymentRequest,
    _attach_launch_config,
    _deployment_to_response,
)


def _vllm_payload() -> dict[str, object]:
    return {
        "framework": "vllm",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 2,
        "max_context_length": 32768,
        "max_concurrent_requests": 256,
        "dtype": "bfloat16",
        "quantization": "awq",
        "kv_cache_dtype": "fp8_e4m3",
        "gpu_pool": [0, 1, 2, 3, 4, 5, 6, 7],
        "replica_gpu_overrides": [
            {"replica_index": 1, "gpu_ids": [4, 5, 6, 7]},
        ],
        "allow_gpu_reuse": False,
        "enable_expert_parallel": True,
        "enforce_eager": True,
    }


def _sglang_payload() -> dict[str, object]:
    return {
        "framework": "sglang",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "max_context_length": 16384,
        "max_concurrent_requests": 128,
        "dtype": "auto",
        "quantization": "gptq_marlin",
        "kv_cache_dtype": "bf16",
        "gpu_pool": [2, 3],
        "replica_gpu_overrides": [],
        "allow_gpu_reuse": False,
        "expert_parallel_size": 2,
        "attention_backend": "flashinfer",
    }


def test_deployment_response_keeps_independent_replica_instances() -> None:
    response = _deployment_to_response(
        {
            "deployment_id": "deployment-1",
            "model_id": "model-1",
            "model_uid": None,
            "deployment_name": "group",
            "xinference_endpoint": "http://127.0.0.1:11000",
            "replica": 2,
            "gpu_memory_utilization": 0.8,
            "deploy_mode": "container",
            "container_name": "group",
            "gpu_id": 0,
            "port": 11000,
            "inference_framework": "vllm",
            "enable_lora": False,
            "max_loras": 4,
            "max_lora_rank": 64,
            "config": {"launch_config": {"framework": "vllm"}},
            "status": "stopped",
            "error_message": None,
            "user_id": "user-1",
            "created_at": None,
            "updated_at": None,
            "started_at": None,
            "stopped_at": None,
            "replica_instances": [
                {
                    "replica_id": "replica-1",
                    "deployment_id": "deployment-1",
                    "replica_index": 0,
                    "container_name": "group",
                    "endpoint": "http://127.0.0.1:11000",
                    "port": 11000,
                    "gpu_ids": [0],
                    "status": "stopped",
                    "health_status": "UNKNOWN",
                    "error_message": None,
                    "created_at": None,
                    "updated_at": None,
                    "started_at": None,
                    "stopped_at": None,
                }
            ],
        }
    )

    assert len(response.replica_instances) == 1
    assert response.replica_instances[0].replica_id == "replica-1"
    assert response.replica_instances[0].gpu_ids == [0]


def test_framework_discriminator_returns_concrete_models() -> None:
    vllm = parse_launch_config(_vllm_payload())
    sglang = parse_launch_config(_sglang_payload())

    assert type(vllm) is VllmLaunchConfig
    assert type(sglang) is SglangLaunchConfig
    assert vllm.framework == "vllm"
    assert sglang.framework == "sglang"


@pytest.mark.parametrize(
    ("payload_factory", "field", "value"),
    [
        (_vllm_payload, "unknown", "--hostile"),
        (_sglang_payload, "enable_expert_parallel", True),
        (_vllm_payload, "attention_backend", "flashinfer"),
        (_sglang_payload, "enforce_eager", True),
    ],
)
def test_unknown_and_cross_framework_fields_are_rejected(
    payload_factory,
    field: str,
    value: object,
) -> None:
    payload = payload_factory()
    payload[field] = value

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "max_context_length",
        "max_concurrent_requests",
    ],
)
def test_boolean_is_not_accepted_as_integer(field: str) -> None:
    payload = _vllm_payload()
    payload[field] = True

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("tensor_parallel_size", 0),
        ("tensor_parallel_size", 65),
        ("pipeline_parallel_size", 0),
        ("pipeline_parallel_size", 65),
        ("data_parallel_size", 0),
        ("data_parallel_size", 65),
        ("max_context_length", 0),
        ("max_context_length", 4_194_305),
        ("max_concurrent_requests", 0),
        ("max_concurrent_requests", 4097),
    ],
)
def test_numeric_bounds_are_strict(field: str, invalid: int) -> None:
    payload = _vllm_payload()
    payload[field] = invalid

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


def test_optional_capacity_fields_accept_none() -> None:
    payload = _vllm_payload()
    payload["max_context_length"] = None
    payload["max_concurrent_requests"] = None

    config = parse_launch_config(payload)

    assert config.max_context_length is None
    assert config.max_concurrent_requests is None


@pytest.mark.parametrize(
    ("payload_factory", "field", "invalid"),
    [
        (_vllm_payload, "quantization", "nvfp4_online"),
        (_sglang_payload, "quantization", "deepspeedfp"),
        (_vllm_payload, "kv_cache_dtype", "bf16"),
        (_sglang_payload, "kv_cache_dtype", "fp8_inc"),
        (_sglang_payload, "attention_backend", "not-a-backend"),
        (_vllm_payload, "dtype", "torch.float16"),
    ],
)
def test_versioned_enum_values_reject_cross_framework_or_unknown_values(
    payload_factory,
    field: str,
    invalid: str,
) -> None:
    payload = payload_factory()
    payload[field] = invalid

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantization", "auto_awq"),
        ("quantization", "deepseek_v4_fp8"),
        ("quantization", "fp8_per_channel"),
        ("kv_cache_dtype", "turboquant_k8v4"),
        ("kv_cache_dtype", "nvfp4"),
    ],
)
def test_vllm_026_accepts_current_tagged_enum_values(field: str, value: str) -> None:
    payload = _vllm_payload()
    payload[field] = value

    assert getattr(parse_launch_config(payload), field) == value


@pytest.mark.parametrize("removed", ["deepspeedfp", "bitblas", "auto-round"])
def test_vllm_026_rejects_removed_quantization_values(removed: str) -> None:
    payload = _vllm_payload()
    payload["quantization"] = removed

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


def test_sglang_0517_accepts_hpc_ops_attention_backend() -> None:
    payload = _sglang_payload()
    payload["attention_backend"] = "hpc_ops"

    assert parse_launch_config(payload).attention_backend == "hpc_ops"


@pytest.mark.parametrize("invalid_ep", [0, 3, 65])
def test_sglang_expert_parallel_size_is_one_or_tensor_parallel_size(
    invalid_ep: int,
) -> None:
    payload = _sglang_payload()
    payload["expert_parallel_size"] = invalid_ep

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


def test_sglang_expert_parallel_size_one_is_valid() -> None:
    payload = _sglang_payload()
    payload["expert_parallel_size"] = 1

    assert parse_launch_config(payload).expert_parallel_size == 1


@pytest.mark.parametrize(
    "gpu_pool",
    [[0, 1, 0], [-1, 0], [False, 1]],
)
def test_gpu_pool_rejects_duplicates_negative_ids_and_booleans(
    gpu_pool: list[object],
) -> None:
    payload = _vllm_payload()
    payload["gpu_pool"] = gpu_pool

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


def test_gpu_pool_order_is_preserved() -> None:
    payload = _vllm_payload()
    payload["gpu_pool"] = [7, 3, 5, 1]
    payload["replica_gpu_overrides"] = []

    assert parse_launch_config(payload).gpu_pool == [7, 3, 5, 1]


@pytest.mark.parametrize(
    "overrides",
    [
        [
            {"replica_index": 0, "gpu_ids": [0, 1]},
            {"replica_index": 0, "gpu_ids": [2, 3]},
        ],
        [{"replica_index": -1, "gpu_ids": [0]}],
        [{"replica_index": False, "gpu_ids": [0]}],
        [{"replica_index": 0, "gpu_ids": []}],
        [{"replica_index": 0, "gpu_ids": [0, 0]}],
        [{"replica_index": 0, "gpu_ids": [-1]}],
        [{"replica_index": 0, "gpu_ids": [False]}],
    ],
)
def test_replica_gpu_overrides_have_strict_local_shape(
    overrides: list[dict[str, object]],
) -> None:
    payload = _vllm_payload()
    payload["replica_gpu_overrides"] = overrides

    with pytest.raises(ValidationError):
        parse_launch_config(payload)


def test_override_index_must_fit_request_replica_count() -> None:
    payload = _vllm_payload()
    payload["replica_gpu_overrides"] = [
        {"replica_index": 2, "gpu_ids": [0, 1, 2, 3]},
    ]

    with pytest.raises(ValueError, match="replica index"):
        normalize_launch_config(
            framework="vllm",
            replica_count=2,
            gpu_id=None,
            legacy_config=None,
            launch_config=payload,
        )


def test_legacy_single_replica_is_adapted_to_strict_vllm_config() -> None:
    config = normalize_launch_config(
        framework="vllm",
        replica_count=1,
        gpu_id=3,
        legacy_config={"dtype": "bfloat16", "enforce_eager": True},
        launch_config=None,
    )

    assert type(config) is VllmLaunchConfig
    assert config.gpu_pool == [3]
    assert config.dtype == "bfloat16"
    assert config.enforce_eager is True


def test_legacy_single_replica_is_adapted_to_strict_sglang_config() -> None:
    config = normalize_launch_config(
        framework="sglang",
        replica_count=1,
        gpu_id=5,
        legacy_config={"dtype": "float16", "attention_backend": "triton"},
        launch_config=None,
    )

    assert type(config) is SglangLaunchConfig
    assert config.gpu_pool == [5]
    assert config.dtype == "float16"
    assert config.attention_backend == "triton"


def test_legacy_multi_replica_without_launch_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="launch_config"):
        normalize_launch_config(
            framework="vllm",
            replica_count=2,
            gpu_id=0,
            legacy_config=None,
            launch_config=None,
        )


@pytest.mark.parametrize(
    ("gpu_id", "legacy_config"),
    [
        (7, None),
        (None, {"dtype": "float16"}),
        (None, {"enforce_eager": False}),
    ],
)
def test_conflicting_legacy_and_typed_values_are_rejected(
    gpu_id: int | None,
    legacy_config: dict[str, object] | None,
) -> None:
    payload = _vllm_payload()
    if gpu_id is not None:
        payload["gpu_pool"] = [0, 1, 2, 3]

    with pytest.raises(ValueError, match="conflict"):
        normalize_launch_config(
            framework="vllm",
            replica_count=2,
            gpu_id=gpu_id,
            legacy_config=legacy_config,
            launch_config=payload,
        )


@pytest.mark.parametrize(
    "legacy_config",
    [
        {"dtype": 1},
        {"enforce_eager": 1},
        {"enforce_eager": 0},
    ],
)
def test_legacy_conflict_comparison_is_type_strict(
    legacy_config: dict[str, object],
) -> None:
    payload = _vllm_payload()
    payload["gpu_pool"] = [0]
    payload["replica_gpu_overrides"] = []

    with pytest.raises(ValueError, match="conflict"):
        normalize_launch_config(
            framework="vllm",
            replica_count=1,
            gpu_id=0,
            legacy_config=legacy_config,
            launch_config=payload,
        )


def test_direct_normalization_rejects_boolean_legacy_gpu_id() -> None:
    payload = _vllm_payload()
    payload["gpu_pool"] = [0]
    payload["replica_gpu_overrides"] = []

    with pytest.raises(ValueError, match="gpu_id"):
        normalize_launch_config(
            framework="vllm",
            replica_count=1,
            gpu_id=False,
            legacy_config=None,
            launch_config=payload,
        )


def test_normalization_does_not_mutate_request_payloads() -> None:
    payload = _vllm_payload()
    legacy = {"dtype": "float16"}
    payload_before = deepcopy(payload)
    legacy_before = deepcopy(legacy)

    with pytest.raises(ValueError):
        normalize_launch_config(
            framework="vllm",
            replica_count=2,
            gpu_id=None,
            legacy_config=legacy,
            launch_config=payload,
        )

    assert payload == payload_before
    assert legacy == legacy_before


def test_xinference_does_not_accept_container_launch_config() -> None:
    with pytest.raises(ValueError, match="Xinference"):
        normalize_launch_config(
            framework="xinference",
            replica_count=1,
            gpu_id=None,
            legacy_config=None,
            launch_config=_vllm_payload(),
        )


@pytest.mark.parametrize(
    "request_model",
    [CreateDeploymentRequest, CreateContainerDeploymentRequest],
)
def test_api_request_accepts_matching_typed_launch_config(request_model) -> None:
    request = request_model(
        model_id="model-1",
        replica=2,
        inference_framework="vllm",
        launch_config=_vllm_payload(),
    )

    normalized = request.normalized_launch_config()

    assert type(normalized) is VllmLaunchConfig
    assert normalized.tensor_parallel_size == 2
    assert request.model_dump(mode="json")["launch_config"]["framework"] == "vllm"


def test_normalized_launch_config_is_attached_without_mutating_existing_config() -> (
    None
):
    original = {"external_api_config_id": "api-config-1"}
    normalized = parse_launch_config(_vllm_payload())

    result = _attach_launch_config(original, normalized)

    assert result is not original
    assert original == {"external_api_config_id": "api-config-1"}
    assert result == {
        "external_api_config_id": "api-config-1",
        "launch_config": normalized.model_dump(mode="json"),
    }


@pytest.mark.parametrize(
    "request_model",
    [CreateDeploymentRequest, CreateContainerDeploymentRequest],
)
def test_api_request_rejects_multi_replica_without_launch_config(
    request_model,
) -> None:
    with pytest.raises(ValidationError, match="launch_config"):
        request_model(
            model_id="model-1",
            replica=2,
            inference_framework="vllm",
        )


def test_api_request_keeps_legacy_single_replica_compatible() -> None:
    request = CreateDeploymentRequest(
        model_id="model-1",
        replica=1,
        gpu_id=0,
        inference_framework="vllm",
        config={"dtype": "bfloat16", "enforce_eager": True},
    )

    normalized = request.normalized_launch_config()

    assert type(normalized) is VllmLaunchConfig
    assert normalized.gpu_pool == [0]
    assert normalized.dtype == "bfloat16"


def test_api_request_rejects_xinference_launch_config() -> None:
    with pytest.raises(ValidationError, match="Xinference"):
        CreateDeploymentRequest(
            model_id="model-1",
            inference_framework="xinference",
            launch_config=_vllm_payload(),
        )


def test_api_request_rejects_framework_mismatch() -> None:
    with pytest.raises(ValidationError, match="framework"):
        CreateDeploymentRequest(
            model_id="model-1",
            inference_framework="sglang",
            launch_config=_vllm_payload(),
        )


def test_api_request_rejects_launch_config_hidden_in_freeform_config() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        CreateDeploymentRequest(
            model_id="model-1",
            inference_framework="vllm",
            config={"launch_config": _vllm_payload()},
        )


@pytest.mark.parametrize(("field", "value"), [("replica", True), ("gpu_id", False)])
def test_api_request_rejects_boolean_integer_fields(field: str, value: bool) -> None:
    kwargs = {
        "model_id": "model-1",
        "inference_framework": "vllm",
        "replica": 1,
        "gpu_id": 0,
    }
    kwargs[field] = value

    with pytest.raises(ValidationError):
        CreateDeploymentRequest(**kwargs)
