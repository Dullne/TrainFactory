"""
External Model Configuration API routes.

Provides endpoints for managing external model API configurations.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...auth.dependencies import get_current_user
from ...config.settings import get_settings
from ...storage.services.inference_authorization_service import (
    authorize_inference_config,
    authorize_inference_model,
    canonical_model_endpoint,
    owned_model_uids_for_endpoint,
)
from ...storage.services.outbound_endpoint_policy import (
    async_request_user_outbound,
    validate_user_outbound_url,
)
from ...storage.services.milvus_collection_service import milvus_collection_service
from ...storage.services.model_config_service import (
    model_config_service,
    normalize_api_endpoint_url,
)
from ...storage.services.runtime_dependency_service import (
    RuntimeDependencyUnavailableError,
)
from ...deployment.deployment_service import (
    deployment_service,
    get_default_xinference_endpoint,
)
from ...generation.clients.embedding_client import EmbeddingClient, EmbeddingConfig

logger = logging.getLogger(__name__)

# Cache for deployment inference_framework lookup
_deployment_framework_cache: Dict[str, str] = {}

router = APIRouter()

MAX_TEST_TEXT_BYTES = 8_192
MAX_SIMILARITY_TEXTS_PER_SIDE = 128
MAX_SIMILARITY_CELLS = 10_000
MAX_RECALL_QUERIES = 20
MAX_RECALL_TOP_K = 100
MAX_TEST_EMBEDDING_BATCH_SIZE = 256
TEST_PROXY_TIMEOUT_SECONDS = 60
TEST_PROXY_MAX_RETRIES = 3


# === Request/Response Models ===

class CreateConfigRequest(BaseModel):
    """Create configuration request."""
    config_name: str = Field(..., description="Configuration name")
    model_type: str = Field(..., description="Model type: llm, embedding, rerank")
    provider: str = Field(..., description="Provider: openai, azure, xinference, etc.")
    api_endpoint: str = Field(..., description="API endpoint URL")
    model_name: str = Field(..., description="Model name/ID for API calls")
    api_key: Optional[str] = Field(default=None, description="API key (if required)")
    provider_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific configuration"
    )
    default_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default parameters for API calls"
    )
    description: Optional[str] = Field(default=None, description="Description")
    tags: Optional[List[str]] = Field(default=None, description="Tags")
    is_default: bool = Field(default=False, description="Set as default for this model type")
    # Validation options
    validate_api: bool = Field(default=False, description="Validate API connectivity before saving")
    fail_on_invalid: bool = Field(default=False, description="Reject if validation fails")


class UpdateConfigRequest(BaseModel):
    """Update configuration request."""
    config_name: Optional[str] = Field(default=None, description="Configuration name")
    model_type: Optional[str] = Field(default=None, description="Model type: llm, embedding, rerank")
    provider: Optional[str] = Field(default=None, description="Provider: openai, azure, xinference, etc.")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint URL")
    api_key: Optional[str] = Field(default=None, description="API key")
    model_name: Optional[str] = Field(default=None, description="Model name/ID")
    provider_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific configuration"
    )
    default_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default parameters"
    )
    description: Optional[str] = Field(default=None, description="Description")
    tags: Optional[List[str]] = Field(default=None, description="Tags")
    status: Optional[str] = Field(default=None, description="Status: active, inactive")
    is_default: Optional[bool] = Field(default=None, description="Set as default")


class ConfigResponse(BaseModel):
    """Configuration response."""
    config_id: str
    config_name: str
    model_type: str
    provider: str
    api_endpoint: str
    api_key: Optional[str]  # May be masked in response
    model_name: str
    provider_config: Optional[Dict[str, Any]]
    default_params: Optional[Dict[str, Any]]
    description: Optional[str]
    tags: Optional[List[str]]
    status: str
    is_default: bool
    last_check_status: Optional[str]
    last_check_time: Optional[str]
    last_check_error: Optional[str]
    user_id: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    # Source and deployment info
    source_type: Optional[str] = None  # external_api or local_deployed
    deployment_id: Optional[str] = None
    deployment_replica_id: Optional[str] = None
    container_name: Optional[str] = None  # Container name for local deployed
    inference_framework: Optional[str] = None  # xinference, vllm, sglang
    # Optional API validation result, populated when create/update requested validate_api
    validation_result: Optional[Dict[str, Any]] = None


class ConfigListResponse(BaseModel):
    """Configuration list response."""
    configs: List[ConfigResponse]
    total: int


class StatsResponse(BaseModel):
    """Statistics response."""
    total: int
    by_type: Dict[str, int]
    by_provider: Dict[str, int]
    by_status: Dict[str, int]


class CheckResponse(BaseModel):
    """Connectivity check response."""
    success: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    warning: Optional[str] = None
    models: Optional[List[str]] = None
    model_details: Optional[List[Dict[str, Any]]] = None


# === Helper Functions ===

def _mask_api_key(api_key: Optional[str]) -> Optional[str]:
    """Mask API key for security."""
    if not api_key:
        return None
    if len(api_key) <= 8:
        return "****"
    return api_key[:4] + "****" + api_key[-4:]


def _verify_model_config_access(
    config: Optional[Dict[str, Any]],
    current_user: Dict[str, Any],
    *,
    allow_public: bool = False,
) -> Dict[str, Any]:
    """Require ownership for secret-bearing operations.

    Ownerless configs are readable for backward-compatible shared selections,
    but authenticated users cannot mutate them or use their stored credentials
    for connectivity/proxy requests.
    """
    if not config:
        raise HTTPException(status_code=404, detail="Model config not found")

    user_id = current_user.get("user_id")
    if user_id in (None, "anonymous"):
        return config

    owner_id = config.get("user_id")
    if owner_id == user_id or (allow_public and not owner_id):
        return config

    raise HTTPException(
        status_code=403,
        detail="Not authorized to access this model config",
    )


def _canonical_model_endpoint(endpoint: str, provider: str = "xinference") -> str:
    """Canonicalize an API base URL for deployment ownership comparisons."""
    return canonical_model_endpoint(endpoint, provider)


def _owned_model_uids_for_endpoint(
    endpoint: str,
    current_user: Dict[str, Any],
    provider: str = "xinference",
) -> set[str]:
    return owned_model_uids_for_endpoint(endpoint, current_user.get("user_id"), provider)


def _authorize_shared_model_listing(
    *,
    provider: str,
    endpoint: str,
    model_name: Optional[str],
    current_user: Dict[str, Any],
) -> Optional[set[str]]:
    """Scope shared Xinference model listing to a regular user's own model."""
    if not get_settings().auth_enabled or current_user.get("is_admin"):
        return None

    owned_model_uids = _owned_model_uids_for_endpoint(endpoint, current_user, provider)
    return authorize_inference_model(
        endpoint, model_name, current_user.get("user_id"), provider=provider,
        current_user=current_user, owned_model_uids=owned_model_uids,
        default_endpoint=get_default_xinference_endpoint(),
    )


def _filter_shared_model_listing(
    result: Dict[str, Any],
    allowed_model_uids: Optional[set[str]],
) -> Dict[str, Any]:
    """Remove cross-tenant model identifiers and paths from validation output."""
    if allowed_model_uids is None:
        return result

    filtered = dict(result)
    models = filtered.get("models")
    if isinstance(models, list):
        filtered["models"] = [
            model for model in models if model in allowed_model_uids
        ]

    model_details = filtered.get("model_details")
    if isinstance(model_details, list):
        safe_details = []
        for detail in model_details:
            if not isinstance(detail, dict):
                continue
            model_id = detail.get("id") or detail.get("model_uid")
            if model_id not in allowed_model_uids:
                continue
            safe_details.append({
                key: value
                for key, value in detail.items()
                if key not in {"path", "model_path", "model_uri"}
            })
        filtered["model_details"] = safe_details

    filtered.pop("warning", None)
    if filtered.get("message"):
        filtered["message"] = "Connection successful"
    if filtered.get("error"):
        filtered["error"] = "Authorized model is unavailable"
    return filtered


def _authorize_config_model(
    config: Dict[str, Any], current_user: Dict[str, Any]
) -> Optional[set[str]]:
    """Authorize the config owner and its trusted deployment/model binding."""
    if not get_settings().auth_enabled or current_user.get("is_admin"):
        return None
    _verify_model_config_access(config, current_user)
    return authorize_inference_config(
        config, current_user.get("user_id"), current_user=current_user,
    )


def _config_to_response(config: Dict[str, Any], mask_key: bool = True) -> ConfigResponse:
    """Convert config dict to response."""
    deployment_id = config.get("deployment_id")

    # Use container_name and inference_framework from config if available
    container_name = config.get("container_name")
    inference_framework = config.get("inference_framework")

    # Fall back to deployment lookup if not in config
    if deployment_id and not inference_framework:
        # Check cache first
        if deployment_id in _deployment_framework_cache:
            inference_framework = _deployment_framework_cache[deployment_id]
        else:
            try:
                deployment = deployment_service.get_deployment(deployment_id)
                if deployment:
                    inference_framework = deployment.get("inference_framework")
                    _deployment_framework_cache[deployment_id] = inference_framework
            except Exception:
                pass

    return ConfigResponse(
        config_id=config["config_id"],
        config_name=config["config_name"],
        model_type=config["model_type"],
        provider=config["provider"],
        api_endpoint=config["api_endpoint"],
        api_key=_mask_api_key(config.get("api_key")) if mask_key else config.get("api_key"),
        model_name=config["model_name"],
        provider_config=config.get("provider_config"),
        default_params=config.get("default_params"),
        description=config.get("description"),
        tags=config.get("tags"),
        status=config["status"],
        is_default=config["is_default"],
        last_check_status=config.get("last_check_status"),
        last_check_time=config["last_check_time"].isoformat() if config.get("last_check_time") else None,
        last_check_error=config.get("last_check_error"),
        user_id=config.get("user_id"),
        created_at=config["created_at"].isoformat() if config.get("created_at") else None,
        updated_at=config["updated_at"].isoformat() if config.get("updated_at") else None,
        source_type=config.get("source_type", "external_api"),
        deployment_id=deployment_id,
        deployment_replica_id=config.get("deployment_replica_id"),
        container_name=container_name,
        inference_framework=inference_framework,
    )


def _configs_to_response(configs: List[Dict[str, Any]]) -> List[ConfigResponse]:
    """Convert configs, including deployment lookups, off the API event loop."""
    return [_config_to_response(config) for config in configs]


# === API Endpoints ===

class ValidateConfigRequest(BaseModel):
    """Validate configuration request (without saving)."""
    provider: str = Field(..., description="Provider: openai, azure, xinference, etc.")
    api_endpoint: str = Field(..., description="API endpoint URL")
    api_key: Optional[str] = Field(default=None, description="API key (if required)")
    model_name: Optional[str] = Field(default=None, description="Model name/ID")
    provider_config: Optional[Dict[str, Any]] = Field(default=None, description="Provider-specific config")


class ValidationResponse(BaseModel):
    """Validation response."""
    valid: bool
    latency_ms: Optional[float] = None
    models: Optional[List[str]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


@router.post("/configs/validate", response_model=ValidationResponse)
async def validate_config(
    request: ValidateConfigRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Validate API configuration without saving.

    Use this to test connectivity before creating a config.
    Supports:
    - OpenAI-compatible APIs (openai, azure, deepseek, zhipu, qwen, moonshot, custom)
    - Self-hosted services (xinference, ollama)
    - Embedding providers (jina, cohere, voyage)

    Features:
    - URL normalization (auto-adds protocol, normalizes paths)
    - Model existence verification (for OpenAI-compatible APIs)
    - Retry on connection errors
    - Structured error types
    """
    allowed_model_uids = await run_in_threadpool(
        _authorize_shared_model_listing,
        provider=request.provider,
        endpoint=request.api_endpoint,
        model_name=request.model_name,
        current_user=current_user,
    )
    result = await model_config_service.validate_config_async(
        provider=request.provider,
        api_endpoint=request.api_endpoint,
        api_key=request.api_key,
        model_name=request.model_name,
        provider_config=request.provider_config,
        user_id=current_user.get("user_id"),
    )
    result = _filter_shared_model_listing(result, allowed_model_uids)
    return ValidationResponse(**result)


@router.post("/configs", response_model=ConfigResponse)
async def create_config(
    request: CreateConfigRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Create a new model configuration.

    Set `validate=true` to check API connectivity before saving.
    Set `fail_on_invalid=true` to reject if validation fails.

    When validation is enabled:
    - For OpenAI-compatible APIs: verifies model_name exists in available models
    - URL is automatically normalized (protocol added, paths standardized)
    - Connection errors trigger automatic retry (up to 2 retries)
    """
    # "reranker" 为 legacy 命名（_TEST_PROXY_PATHS 与 vLLM 分支均兼容），前端仍使用
    valid_types = ["llm", "embedding", "rerank", "reranker", "multimodal", "speech", "decoder_reranker"]
    if request.model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {request.model_type}. Must be one of {valid_types}"
        )

    allowed_model_uids = await run_in_threadpool(
        _authorize_shared_model_listing,
        provider=request.provider,
        endpoint=request.api_endpoint,
        model_name=request.model_name,
        current_user=current_user,
    )

    try:
        config = await model_config_service.create_config_async(
            config_name=request.config_name,
            model_type=request.model_type,
            provider=request.provider,
            api_endpoint=request.api_endpoint,
            model_name=request.model_name,
            api_key=request.api_key,
            provider_config=request.provider_config,
            default_params=request.default_params,
            description=request.description,
            tags=request.tags,
            is_default=request.is_default,
            user_id=current_user.get("user_id"),
            validate=request.validate_api,
            save_on_validation_failure=not request.fail_on_invalid,
        )

        response = await run_in_threadpool(_config_to_response, config)

        # Include validation result if validation was performed
        if "validation" in config:
            response.validation_result = _filter_shared_model_listing(
                config["validation"],
                allowed_model_uids,
            )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/configs", response_model=ConfigListResponse)
async def list_configs(
    model_type: Optional[str] = None,
    provider: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all configurations with optional filters."""
    configs, total = await run_in_threadpool(
        model_config_service.list_configs,
        model_type=model_type,
        provider=provider,
        status=status,
        user_id=current_user.get("user_id"),
        limit=limit,
        offset=offset,
    )
    return ConfigListResponse(
        configs=await run_in_threadpool(_configs_to_response, configs),
        total=total,
    )


@router.get("/configs/search", response_model=ConfigListResponse)
async def search_configs(
    query: str = Query(..., description="Search query"),
    model_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Search configurations by name, description, or model name."""
    configs = await run_in_threadpool(
        model_config_service.search_configs,
        query=query,
        model_type=model_type,
        user_id=current_user.get("user_id"),
        limit=limit,
    )
    return ConfigListResponse(
        configs=await run_in_threadpool(_configs_to_response, configs),
        total=len(configs),
    )


@router.get("/configs/stats", response_model=StatsResponse)
async def get_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get configuration statistics."""
    return await run_in_threadpool(
        model_config_service.get_stats,
        user_id=current_user.get("user_id"),
    )


@router.get("/configs/default/{model_type}", response_model=ConfigResponse)
async def get_default_config(
    model_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get default configuration for a model type."""
    config = await run_in_threadpool(
        model_config_service.get_default_config,
        model_type,
        current_user.get("user_id"),
    )
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No default config found for model_type: {model_type}"
        )
    return await run_in_threadpool(_config_to_response, config)


# === Provider Templates (must be before /{config_id} routes) ===

@router.get("/configs/templates")
async def get_provider_templates(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get configuration templates for common providers."""
    return {
        "openai": {
            "provider": "openai",
            "api_endpoint": "https://api.openai.com/v1",
            "models": {
                "llm": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "embedding": ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
            },
            "default_params": {
                "llm": {"temperature": 0.7, "max_tokens": 2048},
                "embedding": {"dimensions": 1536},
            }
        },
        "azure": {
            "provider": "azure",
            "api_endpoint": "https://{resource}.openai.azure.com",
            "provider_config": {
                "api_version": "2024-02-01",
                "deployment_name": "{deployment}"
            },
            "models": {
                "llm": ["gpt-4", "gpt-35-turbo"],
                "embedding": ["text-embedding-ada-002"],
            }
        },
        "xinference": {
            "provider": "xinference",
            "api_endpoint": "http://localhost:9997",
            "models": {
                "llm": ["qwen2.5-instruct", "llama-3.1-instruct"],
                "embedding": ["bge-m3", "bge-large-zh"],
                "rerank": ["bge-reranker-v2-m3", "bge-reranker-large"],
            }
        },
        "ollama": {
            "provider": "ollama",
            "api_endpoint": "http://localhost:11434",
            "models": {
                "llm": ["llama3.1", "qwen2.5", "mistral"],
                "embedding": ["nomic-embed-text", "mxbai-embed-large"],
            }
        },
        "zhipu": {
            "provider": "zhipu",
            "api_endpoint": "https://open.bigmodel.cn/api/paas/v4",
            "models": {
                "llm": ["glm-4", "glm-4-flash", "glm-3-turbo"],
                "embedding": ["embedding-2"],
            }
        },
        "qwen": {
            "provider": "qwen",
            "api_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": {
                "llm": ["qwen-max", "qwen-plus", "qwen-turbo"],
                "embedding": ["text-embedding-v3"],
            }
        },
        "deepseek": {
            "provider": "deepseek",
            "api_endpoint": "https://api.deepseek.com",
            "models": {
                "llm": ["deepseek-chat", "deepseek-coder"],
            }
        },
        "jina": {
            "provider": "jina",
            "api_endpoint": "https://api.jina.ai/v1",
            "models": {
                "embedding": ["jina-embeddings-v3", "jina-clip-v2"],
                "rerank": ["jina-reranker-v2-base-multilingual"],
            }
        },
        "cohere": {
            "provider": "cohere",
            "api_endpoint": "https://api.cohere.ai/v1",
            "models": {
                "llm": ["command-r-plus", "command-r"],
                "embedding": ["embed-multilingual-v3.0", "embed-english-v3.0"],
                "rerank": ["rerank-multilingual-v3.0", "rerank-english-v3.0"],
            }
        },
        "voyage": {
            "provider": "voyage",
            "api_endpoint": "https://api.voyageai.com/v1",
            "models": {
                "embedding": ["voyage-3", "voyage-3-lite", "voyage-multilingual-2"],
                "rerank": ["rerank-2", "rerank-2-lite"],
            }
        },
    }


@router.get("/configs/{config_id}", response_model=ConfigResponse)
async def get_config(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get configuration by ID."""
    config = _verify_model_config_access(
        await run_in_threadpool(model_config_service.get_config, config_id),
        current_user,
        allow_public=True,
    )
    return await run_in_threadpool(_config_to_response, config)


@router.put("/configs/{config_id}", response_model=ConfigResponse)
async def update_config(
    config_id: str,
    request: UpdateConfigRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update configuration."""
    existing_config = _verify_model_config_access(
        await run_in_threadpool(model_config_service.get_config, config_id),
        current_user,
    )
    has_local_binding = (
        existing_config.get("source_type") == "local_deployed"
        and bool(
            existing_config.get("deployment_id")
            or existing_config.get("deployment_replica_id")
        )
    )
    if not has_local_binding:
        effective = dict(existing_config)
        for field in ("provider", "api_endpoint", "model_name"):
            value = getattr(request, field)
            if value is not None:
                effective[field] = value
        await run_in_threadpool(
            _authorize_shared_model_listing,
            provider=effective.get("provider", ""),
            endpoint=effective.get("api_endpoint", ""),
            model_name=effective.get("model_name"),
            current_user=current_user,
        )
    # Bound local configs keep their server-controlled connection identity.
    # Metadata edits must still work while the deployment is stopped.
    success = await run_in_threadpool(
        model_config_service.update_config,
        config_id=config_id,
        config_name=request.config_name,
        model_type=None if has_local_binding else request.model_type,
        provider=None if has_local_binding else request.provider,
        api_endpoint=None if has_local_binding else request.api_endpoint,
        api_key=None if has_local_binding else request.api_key,
        model_name=None if has_local_binding else request.model_name,
        provider_config=None if has_local_binding else request.provider_config,
        default_params=request.default_params,
        description=request.description,
        tags=request.tags,
        status=request.status,
        is_default=request.is_default,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Config not found: {config_id}")

    config = await run_in_threadpool(model_config_service.get_config, config_id)
    return await run_in_threadpool(_config_to_response, config)


@router.delete("/configs/{config_id}")
async def delete_config(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete configuration."""
    _verify_model_config_access(
        await run_in_threadpool(model_config_service.get_config, config_id),
        current_user,
    )
    try:
        success = await run_in_threadpool(model_config_service.delete_config, config_id)
    except RuntimeDependencyUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not success:
        raise HTTPException(status_code=404, detail=f"Config not found: {config_id}")
    return {"message": f"Config {config_id} deleted"}


@router.post("/configs/{config_id}/set-default")
async def set_default(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Set configuration as default for its model type."""
    _verify_model_config_access(
        await run_in_threadpool(model_config_service.get_config, config_id),
        current_user,
    )
    success = await run_in_threadpool(model_config_service.set_default, config_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Config not found: {config_id}")
    return {"message": f"Config {config_id} set as default"}


@router.post("/configs/{config_id}/check", response_model=CheckResponse)
async def check_connectivity(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Check connectivity to the configured API endpoint."""
    config = await run_in_threadpool(model_config_service.get_config, config_id)
    _verify_model_config_access(config, current_user)
    config = dict(config)
    config["api_endpoint"] = validate_user_outbound_url(
        config.get("api_endpoint", ""),
        current_user.get("user_id"),
    )
    allowed_model_uids = await run_in_threadpool(_authorize_config_model, config, current_user)

    result = await model_config_service.check_connectivity(config_id)
    result = _filter_shared_model_listing(result, allowed_model_uids)
    return CheckResponse(**result)


# === API Test Proxy ===

def _normalize_text_list(
    value: Any,
    field_name: str,
    *,
    max_items: int,
) -> List[str]:
    if value is None:
        raise HTTPException(status_code=400, detail=f"缺少字段: {field_name}")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HTTPException(status_code=400, detail=f"{field_name} 需为字符串或字符串数组")
    texts = [item.strip() for item in value if item.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail=f"{field_name} 不能为空")
    if len(texts) > max_items:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} exceeds the maximum of {max_items} items",
        )
    if any(len(text.encode("utf-8")) > MAX_TEST_TEXT_BYTES for text in texts):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} contains text larger than {MAX_TEST_TEXT_BYTES} bytes",
        )
    return texts


def _bounded_int(
    value: Any,
    field_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be an integer",
        ) from None
    if parsed < minimum or parsed > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {minimum} and {maximum}",
        )
    return parsed


async def _compute_embedding_similarity(
    config: Dict[str, Any],
    body: Dict[str, Any],
) -> Dict[str, Any]:
    sentence1 = _normalize_text_list(
        body.get("sentence1"),
        "sentence1",
        max_items=MAX_SIMILARITY_TEXTS_PER_SIDE,
    )
    sentence2 = _normalize_text_list(
        body.get("sentence2"),
        "sentence2",
        max_items=MAX_SIMILARITY_TEXTS_PER_SIDE,
    )
    if len(sentence1) * len(sentence2) > MAX_SIMILARITY_CELLS:
        raise HTTPException(
            status_code=400,
            detail=f"Similarity matrix exceeds {MAX_SIMILARITY_CELLS} cells",
        )

    endpoint = config.get("api_endpoint")
    model_name = body.get("model") or config.get("model_name")
    if not endpoint or not model_name:
        raise HTTPException(status_code=400, detail="Embedding 配置缺少 api_endpoint 或 model_name")

    batch_size = _bounded_int(
        body.get("batch_size"),
        "batch_size",
        default=32,
        minimum=1,
        maximum=MAX_TEST_EMBEDDING_BATCH_SIZE,
    )

    embedding_config = EmbeddingConfig(
        endpoint=endpoint,
        model=model_name,
        api_key=config.get("api_key"),
        batch_size=batch_size,
        timeout=TEST_PROXY_TIMEOUT_SECONDS,
        max_retries=TEST_PROXY_MAX_RETRIES,
        user_id=config.get("user_id"),
    )

    async with EmbeddingClient(embedding_config) as client:
        embeddings = await client.embed(sentence1 + sentence2)

    left = embeddings[:len(sentence1)]
    right = embeddings[len(sentence1):]

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    left = left / left_norm
    right = right / right_norm

    similarity = np.dot(left, right.T).tolist()

    return {
        "sentence1": [{"index": idx, "text": text} for idx, text in enumerate(sentence1)],
        "sentence2": [{"index": idx, "text": text} for idx, text in enumerate(sentence2)],
        "similarity": similarity,
    }


def _get_milvus_client_from_env():
    """创建 MilvusClient（从环境变量读取连接信息）。"""
    from ...generation.clients.milvus_client import MilvusClient, MilvusConfig
    client = MilvusClient(MilvusConfig())
    client.connect()
    return client


async def _compute_embedding_recall(
    config: Dict[str, Any],
    body: Dict[str, Any],
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    """召回测试：embed query → 搜索 Milvus collection → 返回 top-k 结果。"""
    queries = _normalize_text_list(
        body.get("queries"),
        "queries",
        max_items=MAX_RECALL_QUERIES,
    )
    collection_name = body.get("collection_name")
    if not collection_name:
        raise HTTPException(status_code=400, detail="缺少 collection_name")
    import re
    if not isinstance(collection_name, str) or not re.match(r'^[a-zA-Z0-9_]{1,255}$', collection_name):
        raise HTTPException(status_code=400, detail="collection_name 格式无效（仅允许字母数字下划线，最长 255 字符）")
    top_k = _bounded_int(
        body.get("top_k"),
        "top_k",
        default=10,
        minimum=1,
        maximum=MAX_RECALL_TOP_K,
    )
    batch_size = _bounded_int(
        body.get("batch_size"),
        "batch_size",
        default=32,
        minimum=1,
        maximum=MAX_TEST_EMBEDDING_BATCH_SIZE,
    )

    endpoint = config.get("api_endpoint")
    model_name = body.get("model") or config.get("model_name")
    if not endpoint or not model_name:
        raise HTTPException(status_code=400, detail="缺少 api_endpoint 或 model_name")

    registered = milvus_collection_service.get_by_name(collection_name)
    user_id = current_user.get("user_id")
    if user_id not in (None, "anonymous"):
        if not registered:
            raise HTTPException(status_code=404, detail="Collection not found")
        if registered.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this collection",
            )

    milvus_client = _get_milvus_client_from_env()
    try:
        if not milvus_client.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"向量库 '{collection_name}' 不存在")

        emb_config = EmbeddingConfig(
            endpoint=endpoint,
            model=model_name,
            api_key=config.get("api_key"),
            batch_size=batch_size,
            timeout=TEST_PROXY_TIMEOUT_SECONDS,
            max_retries=TEST_PROXY_MAX_RETRIES,
            user_id=config.get("user_id"),
        )
        async with EmbeddingClient(emb_config) as client:
            query_vectors = await client.embed(queries)

        results = []
        for i, query_text in enumerate(queries):
            query_vec = query_vectors[i:i+1]
            hits = milvus_client.search_similar(
                collection_name=collection_name,
                query_vectors=query_vec,
                top_k=top_k,
            )
            results.append({
                "query": query_text,
                "hits": hits[0] if hits else [],
            })

        return {
            "collection_name": collection_name,
            "model": model_name,
            "top_k": top_k,
            "results": results,
        }
    finally:
        milvus_client.close()


class TestProxyRequest(BaseModel):
    """API test proxy request."""
    path: str = Field(..., description="API path, e.g., /v1/embeddings, /v1/rerank")
    body: Dict[str, Any] = Field(..., description="Request body")


class TestProxyResponse(BaseModel):
    """API test proxy response."""
    success: bool
    status_code: int
    latency_ms: float
    data: Optional[Any] = None  # 可以是 Dict 或 List（SGLang 返回数组）
    error: Optional[str] = None


_TEST_PROXY_PATHS = {
    "embedding": frozenset({"/v1/embeddings"}),
    "rerank": frozenset({"/v1/rerank"}),
    "reranker": frozenset({"/v1/rerank"}),
    "llm": frozenset({"/v1/chat/completions", "/v1/completions"}),
}


def _validate_test_proxy_path(path: str, model_type: str) -> str:
    """Allow the test proxy to call inference endpoints, never management APIs."""
    normalized_model_type = str(model_type or "").strip().lower()
    allowed_paths = _TEST_PROXY_PATHS.get(normalized_model_type, frozenset())
    if not isinstance(path, str) or path not in allowed_paths:
        raise HTTPException(
            status_code=400,
            detail="Test proxy path is not allowed for this model type",
        )
    return path


@router.post("/configs/{config_id}/test", response_model=TestProxyResponse)
async def test_api_proxy(
    config_id: str,
    request: TestProxyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    API 测试代理接口。

    通过后端转发 API 请求，自动处理：
    - vLLM reranker 的 Cohere 兼容请求转发
    - 自动添加 model 字段
    - 统一返回格式

    请求示例:
    ```json
    {
        "path": "/v1/rerank",
        "body": {
            "query": "What is machine learning?",
            "documents": ["ML is AI.", "Python is a language."]
        }
    }
    ```
    """
    config = await run_in_threadpool(model_config_service.get_config, config_id)
    _verify_model_config_access(config, current_user)
    config = dict(config)
    config["api_endpoint"] = validate_user_outbound_url(
        config.get("api_endpoint", ""),
        current_user.get("user_id"),
    )

    # 获取框架类型（从 deployment 表中查询）
    # Persisted config/deployment metadata is trusted; the request body is not.
    inference_framework = config.get('inference_framework') or ''
    deployment_id = config.get('deployment_id')
    if deployment_id and not inference_framework:
        # 使用缓存
        if deployment_id in _deployment_framework_cache:
            inference_framework = _deployment_framework_cache[deployment_id] or ''
        else:
            try:
                deployment = await run_in_threadpool(
                    deployment_service.get_deployment,
                    deployment_id,
                )
                if deployment:
                    inference_framework = (
                        deployment.get('inference_framework')
                        or inference_framework
                    )
                    _deployment_framework_cache[deployment_id] = inference_framework
            except Exception:
                pass
    inference_framework = inference_framework.lower()
    config["inference_framework"] = inference_framework
    model_type = config.get('model_type', '')
    model_name = config.get('model_name', '')
    proxy_path = _validate_test_proxy_path(request.path, model_type)

    # 构建请求体
    body = request.body.copy()
    allowed_model_uids = await run_in_threadpool(_authorize_config_model, config, current_user)
    if allowed_model_uids is not None:
        if "model" in body and body["model"] != model_name:
            raise HTTPException(
                status_code=403,
                detail="Shared model requests must use the authorized config model",
            )
        body["model"] = model_name
    body.pop("framework", None)
    body.pop("inference_framework", None)
    mode = body.pop("mode", None)
    if not mode and ("sentence1" in body or "sentence2" in body):
        mode = "embedding_similarity"

    async def _mark_config_healthy() -> None:
        """API 测试成功仅标记为 healthy，不在此路径将失败回写为 error。"""
        try:
            await run_in_threadpool(
                model_config_service.update_check_status,
                config_id,
                "healthy",
            )
        except Exception:
            # 测试请求本身不应因状态写回失败而中断
            pass

    if mode == "embedding_similarity":
        if model_type != "embedding":
            raise HTTPException(status_code=400, detail="相似度测试仅支持 Embedding 模型")
        start_time = time.time()
        try:
            data = await _compute_embedding_similarity(config, body)
            latency_ms = (time.time() - start_time) * 1000
            await _mark_config_healthy()
            return TestProxyResponse(
                success=True,
                status_code=200,
                latency_ms=latency_ms,
                data=data,
            )
        except HTTPException:
            raise
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return TestProxyResponse(
                success=False,
                status_code=500,
                latency_ms=latency_ms,
                error=str(e),
            )

    if mode == "embedding_recall":
        if model_type != "embedding":
            raise HTTPException(status_code=400, detail="召回测试仅支持 Embedding 模型")
        start_time = time.time()
        try:
            data = await _compute_embedding_recall(config, body, current_user)
            latency_ms = (time.time() - start_time) * 1000
            await _mark_config_healthy()
            return TestProxyResponse(
                success=True,
                status_code=200,
                latency_ms=latency_ms,
                data=data,
            )
        except HTTPException:
            raise
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return TestProxyResponse(
                success=False,
                status_code=500,
                latency_ms=latency_ms,
                error=str(e),
            )

    # 自动添加 model 字段
    # - LLM 类型始终需要 model 字段 (OpenAI 兼容 API)
    # - vLLM 和 Xinference 的 embedding/reranker 也需要
    if 'model' not in body:
        if model_type == 'llm' or inference_framework == 'vllm' or config.get('provider') == 'xinference':
            body['model'] = model_name

    if model_type in ('rerank', 'reranker'):
        instruction = body.pop('instruction', None)
        body.pop('instruct', None)
        if isinstance(instruction, str) and instruction.strip():
            instruction_field = {
                'vllm': 'instruction',
                'sglang': 'instruct',
            }.get(inference_framework)
            if instruction_field:
                body[instruction_field] = instruction

    # 构建 URL（避免 /v1 重复）
    endpoint = config['api_endpoint'].rstrip('/')
    path = proxy_path
    # 如果 endpoint 以 /v1 结尾，而 path 以 /v1 开头，去掉 path 的 /v1 前缀
    if endpoint.endswith('/v1') and path.startswith('/v1'):
        path = path[3:]  # 去掉 /v1
    url = f"{endpoint}{path}"

    # 构建 headers
    headers = {'Content-Type': 'application/json'}
    if config.get('api_key'):
        headers['Authorization'] = f"Bearer {config['api_key']}"

    # 发送请求
    start_time = time.time()
    try:
        response = await async_request_user_outbound(
            "POST",
            url,
            current_user.get("user_id"),
            timeout=60.0,
            json=body,
            headers=headers,
        )
        latency_ms = (time.time() - start_time) * 1000

        try:
            data = response.json()
        except Exception:
            data = {'raw': response.text}

        if response.status_code >= 400:
            if isinstance(data, dict):
                err = data.get('error')
                error_msg = (
                    (err.get('message') if isinstance(err, dict) else err)
                    or data.get('detail')
                    or str(data)
                )
            else:
                error_msg = str(data)
            return TestProxyResponse(
                success=False,
                status_code=response.status_code,
                latency_ms=latency_ms,
                data=data,
                error=error_msg,
            )

        await _mark_config_healthy()
        return TestProxyResponse(
            success=True,
            status_code=response.status_code,
            latency_ms=latency_ms,
            data=data,
        )

    except httpx.TimeoutException:
        latency_ms = (time.time() - start_time) * 1000
        return TestProxyResponse(
            success=False,
            status_code=504,
            latency_ms=latency_ms,
            error='Request timeout',
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return TestProxyResponse(
            success=False,
            status_code=500,
            latency_ms=latency_ms,
            error=str(e),
        )


# === Collection Matching ===

@router.get("/configs/{config_id}/collections")
async def list_matching_collections(
    config_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """列出所有向量库，标注与当前模型配置的匹配关系。"""
    config = await run_in_threadpool(model_config_service.get_config, config_id)
    config = _verify_model_config_access(config, current_user, allow_public=True)

    from ...storage.services.milvus_collection_service import milvus_collection_service

    user_id = current_user.get("user_id")
    if user_id == "anonymous":
        user_id = None
    if not user_id:
        # auth disabled fallback: keep response scoped to config owner when available
        user_id = config.get("user_id")

    registered, _total = await run_in_threadpool(
        milvus_collection_service.list_collections,
        user_id=user_id,
        limit=1000,
    )

    result_collections = []
    for coll in registered:
        coll_config_id = coll.get("embedding_config_id")
        coll_model = coll.get("embedding_model")

        if coll_config_id and coll_config_id == config_id:
            match_type = "exact"
        elif coll_model and coll_model == config.get("model_name"):
            match_type = "model_match"
        else:
            match_type = "mismatch"

        result_collections.append({
            "name": coll.get("collection_name"),
            "match_type": match_type,
            "embedding_model": coll_model,
            "embedding_config_id": coll_config_id,
            "dim": coll.get("dim"),
            "status": coll.get("status"),
        })

    # 按匹配度排序：exact > model_match > mismatch
    order = {"exact": 0, "model_match": 1, "mismatch": 2}
    result_collections.sort(key=lambda c: order.get(c["match_type"], 3))

    return {"collections": result_collections}


# === Batch Operations ===

@router.post("/configs/check-all")
async def check_all_connectivity(
    model_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Check connectivity for all active configurations."""
    user_id = current_user.get("user_id")
    configs, _total = await run_in_threadpool(
        model_config_service.list_configs,
        model_type=model_type,
        status="active",
        user_id=user_id,
    )
    if user_id not in (None, "anonymous"):
        configs = [config for config in configs if config.get("user_id") == user_id]

    results = {}
    for config in configs:
        try:
            allowed_model_uids = await run_in_threadpool(_authorize_config_model, config, current_user)
        except HTTPException as exc:
            results[config["config_id"]] = {
                "config_name": config["config_name"],
                "model_type": config["model_type"],
                "provider": config["provider"],
                "success": False,
                "error": str(exc.detail),
            }
            continue
        result = await model_config_service.check_connectivity(config["config_id"])
        result = _filter_shared_model_listing(result, allowed_model_uids)
        results[config["config_id"]] = {
            "config_name": config["config_name"],
            "model_type": config["model_type"],
            "provider": config["provider"],
            **result
        }

    return {
        "total": len(results),
        "healthy": sum(1 for r in results.values() if r.get("success")),
        "results": results
    }
