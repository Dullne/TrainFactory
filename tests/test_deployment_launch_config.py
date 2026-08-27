from __future__ import annotations

import pytest
from pydantic import ValidationError

from train_factory.deployment.launch_config import (
    SglangLaunchConfig,
    VllmLaunchConfig,
    parse_launch_config,
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
        "expert_parallel_size": 2,
        "attention_backend": "flashinfer",
    }


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
        (_vllm_payload, "gpu_id", 0),
        (_vllm_payload, "gpu_ids", [0]),
        (_vllm_payload, "gpu_pool", [0, 1]),
        (_vllm_payload, "replica_gpu_overrides", []),
        (_vllm_payload, "allow_gpu_reuse", False),
    ],
)
def test_unknown_cross_framework_and_gpu_placement_fields_are_rejected(
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
