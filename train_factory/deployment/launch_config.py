"""Strict, version-pinned launch configuration for container inference."""

import json
from typing import Annotated, Any, Literal, Mapping, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
    model_validator,
)


ParallelSize = Annotated[StrictInt, Field(ge=1, le=64)]
ContextLength = Annotated[StrictInt, Field(ge=1, le=4_194_304)]
ConcurrentRequests = Annotated[StrictInt, Field(ge=1, le=4096)]

ModelDtype: TypeAlias = Literal[
    "auto",
    "half",
    "float16",
    "bfloat16",
    "float",
    "float32",
]

# vLLM v0.26.0: vllm/model_executor/layers/quantization/__init__.py
VllmQuantization: TypeAlias = Literal[
    "awq",
    "auto_awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "humming",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "online",
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "mxfp8",
]
VllmKvCacheDtype: TypeAlias = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
]

# SGLang v0.5.17: python/sglang/srt/server_args.py
SglangQuantization: TypeAlias = Literal[
    "awq",
    "fp8",
    "mxfp8",
    "gptq",
    "marlin",
    "gptq_marlin",
    "awq_marlin",
    "bitsandbytes",
    "gguf",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "nvfp4_online",
    "modelopt_mixed",
    "petit_nvfp4",
    "w8a8_int8",
    "w8a8_fp8",
    "moe_wna16",
    "w4afp8",
    "mxfp4",
    "auto-round",
    "auto-round-int8",
    "compressed-tensors",
    "modelslim",
    "mxfp_w4a8",
    "quark",
    "quark_int4fp8_moe",
    "quark_mxfp4",
    "mlx_q4",
    "mlx_q8",
    "unquant",
    "humming",
]
SglangKvCacheDtype: TypeAlias = Literal[
    "auto",
    "fp8_e5m2",
    "fp8_e4m3",
    "mxfp8",
    "bf16",
    "bfloat16",
    "nvfp4",
    "fp4_mx_block16",
    "fp4_e2m1",
]
SglangAttentionBackend: TypeAlias = Literal[
    "triton",
    "torch_native",
    "flex_attention",
    "dsa",
    "nsa",
    "dsv4",
    "compressed",
    "cutlass_mla",
    "fa3",
    "fa4",
    "flashinfer",
    "flashmla",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "trtllm_mha",
    "dual_chunk_flash_attn",
    "hpc_ops",
    "aiter",
    "wave",
    "intel_amx",
    "ascend",
    "intel_xpu",
]


class CommonLaunchConfig(BaseModel):
    """Fields shared by the pinned vLLM and SGLang adapters."""

    model_config = ConfigDict(extra="forbid")

    tensor_parallel_size: ParallelSize = 1
    pipeline_parallel_size: ParallelSize = 1
    data_parallel_size: ParallelSize = 1
    max_context_length: ContextLength | None = None
    max_concurrent_requests: ConcurrentRequests | None = None
    dtype: ModelDtype = "auto"


class VllmLaunchConfig(CommonLaunchConfig):
    """Supported vLLM v0.26.0 launch surface."""

    framework: Literal["vllm"]
    quantization: VllmQuantization | None = None
    kv_cache_dtype: VllmKvCacheDtype = "auto"
    enable_expert_parallel: StrictBool = False
    enforce_eager: StrictBool = False


class SglangLaunchConfig(CommonLaunchConfig):
    """Supported SGLang v0.5.17 launch surface."""

    framework: Literal["sglang"]
    quantization: SglangQuantization | None = None
    kv_cache_dtype: SglangKvCacheDtype = "auto"
    expert_parallel_size: ParallelSize = 1
    attention_backend: SglangAttentionBackend | None = None

    @model_validator(mode="after")
    def _validate_expert_parallel_size(self) -> "SglangLaunchConfig":
        if self.expert_parallel_size not in (1, self.tensor_parallel_size):
            raise ValueError(
                "SGLang expert_parallel_size must be 1 or tensor_parallel_size"
            )
        return self


LaunchConfig: TypeAlias = Annotated[
    VllmLaunchConfig | SglangLaunchConfig,
    Field(discriminator="framework"),
]
_LAUNCH_CONFIG_ADAPTER = TypeAdapter(LaunchConfig)


def parse_launch_config(payload: Mapping[str, Any]) -> LaunchConfig:
    """Parse a mapping into the pinned framework-specific configuration."""

    return _LAUNCH_CONFIG_ADAPTER.validate_python(dict(payload))


def _validate_builder_inputs(
    *,
    model_path: str,
    served_model_name: str,
    port: int,
    gpu_memory_utilization: float,
    model_type: str,
    enable_lora: bool,
    max_loras: int,
    max_lora_rank: int,
    trust_remote_code: bool,
) -> None:
    for name, value in (
        ("model_path", model_path),
        ("served_model_name", served_model_name),
    ):
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError(f"{name} is invalid")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port is invalid")
    if (
        type(gpu_memory_utilization) not in (int, float)
        or not 0 < gpu_memory_utilization <= 1
    ):
        raise ValueError("gpu_memory_utilization is invalid")
    if model_type not in ("embedding", "rerank", "reranker", "llm"):
        raise ValueError("model_type is invalid")
    if type(enable_lora) is not bool or type(trust_remote_code) is not bool:
        raise ValueError("boolean launch policy is invalid")
    if type(max_loras) is not int or not 1 <= max_loras <= 64:
        raise ValueError("max_loras is invalid")
    if type(max_lora_rank) is not int or not 1 <= max_lora_rank <= 1024:
        raise ValueError("max_lora_rank is invalid")


def canonical_inference_model_type(model_type: str) -> str:
    """Map registry model types to the framework-facing inference type."""

    if model_type == "decoder_reranker":
        return "reranker"
    return model_type


def xinference_model_type(model_type: str) -> str:
    """Map registry model types to Xinference's public launch vocabulary."""

    model_type = canonical_inference_model_type(model_type)
    if model_type == "reranker":
        return "rerank"
    return model_type


def _append_common_optional_args(
    argv: list[str],
    *,
    trust_remote_code: bool,
) -> None:
    if trust_remote_code:
        argv.append("--trust-remote-code")


def is_qwen3_reranker(model_type: str, model_family: str | None) -> bool:
    model_type = canonical_inference_model_type(model_type)
    if model_type not in ("rerank", "reranker") or model_family is None:
        return False
    normalized = model_family.casefold()
    return "qwen3" in normalized and "rerank" in normalized


def xinference_model_launch_overrides(
    *,
    model_type: str,
    model_family: str | None,
) -> dict[str, str]:
    """Return closed Xinference 3.1 overrides for known model families."""

    if is_qwen3_reranker(model_type, model_family):
        return {"torch_dtype": "bfloat16"}
    return {}


def build_vllm_server_argv(
    config: VllmLaunchConfig,
    *,
    model_path: str,
    served_model_name: str,
    port: int,
    gpu_memory_utilization: float,
    model_type: str,
    enable_lora: bool,
    max_loras: int,
    max_lora_rank: int,
    trust_remote_code: bool,
    model_family: str | None = None,
) -> tuple[str, ...]:
    """Render the closed vLLM v0.26.0 application argv."""

    if not isinstance(config, VllmLaunchConfig):
        raise TypeError("vLLM argv requires VllmLaunchConfig")
    model_type = canonical_inference_model_type(model_type)
    _validate_builder_inputs(
        model_path=model_path,
        served_model_name=served_model_name,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        model_type=model_type,
        enable_lora=enable_lora,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        trust_remote_code=trust_remote_code,
    )
    if model_family is not None and (
        not isinstance(model_family, str) or not model_family or "\0" in model_family
    ):
        raise ValueError("model_family is invalid")
    argv = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--served-model-name",
        served_model_name,
        "--allowed-origins",
        '["*"]',
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(config.pipeline_parallel_size),
        "--data-parallel-size",
        str(config.data_parallel_size),
    ]
    if config.max_context_length is not None:
        argv.extend(["--max-model-len", str(config.max_context_length)])
    if config.max_concurrent_requests is not None:
        argv.extend(["--max-num-seqs", str(config.max_concurrent_requests)])
    argv.extend(["--dtype", config.dtype])
    if config.quantization is not None:
        argv.extend(["--quantization", config.quantization])
    argv.extend(["--kv-cache-dtype", config.kv_cache_dtype])
    if config.enable_expert_parallel:
        argv.append("--enable-expert-parallel")
    if config.enforce_eager:
        argv.append("--enforce-eager")
    if model_type == "embedding":
        argv.extend(["--runner", "pooling"])
    elif model_type in ("rerank", "reranker"):
        argv.extend(["--runner", "pooling"])
        if is_qwen3_reranker(model_type, model_family):
            hf_overrides = json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True,
                },
                separators=(",", ":"),
            )
            argv.extend(
                [
                    "--hf-overrides",
                    hf_overrides,
                    "--chat-template",
                    "/vllm-workspace/examples/pooling/score/template/"
                    "qwen3_reranker.jinja",
                ]
            )
    _append_common_optional_args(argv, trust_remote_code=trust_remote_code)
    if enable_lora:
        argv.extend(
            [
                "--enable-lora",
                "--max-loras",
                str(max_loras),
                "--max-lora-rank",
                str(max_lora_rank),
            ]
        )
    return tuple(argv)


def build_sglang_server_argv(
    config: SglangLaunchConfig,
    *,
    model_path: str,
    served_model_name: str,
    port: int,
    gpu_memory_utilization: float,
    model_type: str,
    enable_lora: bool,
    max_loras: int,
    max_lora_rank: int,
    chat_template: str | None,
    trust_remote_code: bool,
    model_family: str | None = None,
) -> tuple[str, ...]:
    """Render the closed SGLang v0.5.17 application argv."""

    if not isinstance(config, SglangLaunchConfig):
        raise TypeError("SGLang argv requires SglangLaunchConfig")
    model_type = canonical_inference_model_type(model_type)
    _validate_builder_inputs(
        model_path=model_path,
        served_model_name=served_model_name,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        model_type=model_type,
        enable_lora=enable_lora,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        trust_remote_code=trust_remote_code,
    )
    if enable_lora and model_type in ("rerank", "reranker"):
        raise ValueError(
            "SGLang v0.5.17 /v1/rerank has no LoRA adapter selector; "
            "disable LoRA for SGLang reranker deployments"
        )
    if ":" in served_model_name:
        raise ValueError("SGLang served_model_name must not contain ':'")
    if chat_template is not None and (
        not isinstance(chat_template, str) or not chat_template or "\0" in chat_template
    ):
        raise ValueError("chat_template is invalid")
    if model_family is not None and (
        not isinstance(model_family, str) or not model_family or "\0" in model_family
    ):
        raise ValueError("model_family is invalid")
    argv = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--mem-fraction-static",
        str(gpu_memory_utilization),
        "--tp-size",
        str(config.tensor_parallel_size),
        "--pp-size",
        str(config.pipeline_parallel_size),
        "--dp-size",
        str(config.data_parallel_size),
        "--ep-size",
        str(config.expert_parallel_size),
    ]
    if config.pipeline_parallel_size > 1:
        argv.append("--disable-overlap-schedule")
    if config.max_context_length is not None:
        argv.extend(["--context-length", str(config.max_context_length)])
    if config.max_concurrent_requests is not None:
        argv.extend(["--max-running-requests", str(config.max_concurrent_requests)])
    argv.extend(["--dtype", config.dtype])
    if config.quantization is not None:
        argv.extend(["--quantization", config.quantization])
    argv.extend(["--kv-cache-dtype", config.kv_cache_dtype])
    if config.attention_backend is not None:
        argv.extend(["--attention-backend", config.attention_backend])
        if config.attention_backend == "hpc_ops":
            argv.extend(["--page-size", "64"])
    if model_type == "embedding":
        argv.append("--is-embedding")
    elif model_type in ("rerank", "reranker"):
        argv.append("--disable-radix-cache")
        if chat_template is not None and is_qwen3_reranker(
            model_type,
            model_family,
        ):
            argv.extend(["--chat-template", chat_template])
    _append_common_optional_args(argv, trust_remote_code=trust_remote_code)
    if enable_lora:
        argv.extend(
            [
                "--enable-lora",
                "--max-loaded-loras",
                str(max_loras),
                "--max-loras-per-batch",
                str(max_loras),
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-target-modules",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
    return tuple(argv)
