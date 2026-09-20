"""
External Model Configuration Service.

Provides CRUD operations for external model API configurations.
"""

import asyncio
import logging
import time
from train_factory.core.time_utils import now_naive
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from sqlalchemy import update

from sqlmodel import Session, select, func, or_

from ..database import get_engine
from ..entities.model_config_entity import ModelConfigDB
from ..entities.deployment_entity import DeploymentDB
from ..entities.deployment_replica_entity import DeploymentReplicaDB
from ..entities.milvus_collection_entity import MilvusCollectionDB
from .model_registry_service import model_registry_service
from .outbound_endpoint_policy import (
    async_request_user_outbound,
    create_pinned_async_client,
    validate_user_outbound_url,
)
from .runtime_dependency_service import (
    RuntimeDependencyReferences,
    RuntimeDependencyUnavailableError,
    lock_active_runtime_dependency_consumers,
    lock_runtime_dependencies,
    snapshot_runtime_executions,
)
from ...enums import ValidationErrorType

logger = logging.getLogger(__name__)


# === URL Normalization ===

def normalize_api_endpoint(
    endpoint: str,
    provider: str,
    user_id: Optional[str] = None,
) -> str:
    """Normalize the request URL, then apply its user-scoped outbound policy."""
    return validate_user_outbound_url(
        normalize_api_endpoint_url(endpoint, provider), user_id
    )


def normalize_api_endpoint_url(endpoint: str, provider: str) -> str:
    """
    规范化 API 端点 URL

    处理逻辑:
    1. 自动添加 https:// (如果缺失)
    2. 验证 URL 格式
    3. 根据 provider 规范化路径
       - OpenAI 兼容: 确保以 /v1 结尾
       - Ollama: 确保以 /api 结尾
       - 其他: 保持原样
    """
    if not endpoint:
        raise ValueError("API endpoint is required")

    trimmed = endpoint.strip()
    if not trimmed:
        raise ValueError("API endpoint cannot be empty")

    # 自动添加协议
    if "://" not in trimmed:
        # 本地服务默认用 http
        if trimmed.startswith("localhost") or trimmed.startswith("127.0.0.1"):
            trimmed = f"http://{trimmed}"
        else:
            trimmed = f"https://{trimmed}"

    parsed = urlparse(trimmed)
    if not parsed.netloc:
        raise ValueError("Invalid endpoint: missing host")

    # 规范化路径
    path = parsed.path.rstrip("/")

    # 容错：自动移除用户可能误填的 API 路径后缀
    suffix_to_remove = [
        "/embeddings",
        "/chat/completions",
        "/completions",
        "/models",
        "/rerank",
        "/v1/embeddings",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/models",
        "/v1/rerank",
    ]
    for suffix in suffix_to_remove:
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            logger.info(f"Auto-corrected endpoint: removed '{suffix}' suffix")
            break

    # OpenAI 兼容 Provider：确保以 /v1 结尾
    openai_compatible = ["openai", "deepseek", "zhipu", "qwen", "moonshot", "custom"]
    if provider in openai_compatible:
        if not path.endswith("/v1"):
            path = f"{path}/v1" if path else "/v1"
    elif provider == "ollama":
        # Ollama 不需要特殊路径，保持原样
        pass

    final_url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return final_url


# === Model Type Normalization ===

# Mapping from legacy/alternative names to standard names
MODEL_TYPE_ALIASES = {
    "reranker": "rerank",  # model_registry uses 'reranker', model_config uses 'rerank'
}


def normalize_model_type(model_type: str) -> str:
    """
    Normalize model_type to standard values.

    model_registry uses 'reranker' but model_config uses 'rerank'.
    This ensures consistency when creating or querying configs.

    Args:
        model_type: The model type to normalize

    Returns:
        Normalized model type string
    """
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


class ModelConfigService:
    """Service for managing external model configurations."""

    def __init__(self):
        self.engine = None

    def _get_engine(self):
        if self.engine is None:
            self.engine = get_engine()
        return self.engine

    # === Validation Methods ===

    async def _validate_openai_compatible(
        self,
        provider: str,
        api_endpoint: str,
        api_key: Optional[str],
        model_name: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        使用 OpenAI SDK 校验 OpenAI 兼容 API

        优点:
        - 精确的异常类型 (AuthenticationError, APIConnectionError 等)
        - 验证 model_name 是否在可用列表中
        """
        http_client = None
        try:
            from openai import AsyncOpenAI, AuthenticationError, APIConnectionError, APIStatusError

            http_client = create_pinned_async_client(
                api_endpoint,
                user_id,
                timeout=timeout,
            )

            # Azure 特殊处理
            if provider == "azure":
                from openai import AsyncAzureOpenAI
                api_version = (provider_config or {}).get("api_version", "2024-02-01")
                client = AsyncAzureOpenAI(
                    api_key=api_key or "",
                    azure_endpoint=api_endpoint.rsplit("/v1", 1)[0],  # 移除 /v1 后缀
                    api_version=api_version,
                    timeout=timeout,
                    http_client=http_client,
                )
            else:
                client = AsyncOpenAI(
                    api_key=api_key or "dummy-key",  # 某些服务不需要 key
                    base_url=api_endpoint,
                    timeout=timeout,
                    http_client=http_client,
                )

            start_time = time.time()
            models_response = await client.models.list()
            latency_ms = (time.time() - start_time) * 1000

            # 提取模型列表
            available_models = [m.id for m in models_response.data if m.id]

            # 模型名称验证改为"软验证"（警告但不阻止）
            # 原因：很多 API 提供商的 /models 端点不返回所有可用模型
            # 如：阿里云 dashscope、Azure 部署模型、Fine-tuned 模型等
            warning = None
            if model_name and available_models and model_name not in available_models:
                warning = f"模型 '{model_name}' 未在 /models 列表中找到，但这可能是正常的（部分 API 不返回完整列表）"
                logger.warning(warning)

            result = {
                "valid": True,
                "latency_ms": round(latency_ms, 2),
                "models": available_models[:20],
                "message": f"连接成功，共 {len(available_models)} 个可用模型"
            }
            if warning:
                result["warning"] = warning
            return result

        except AuthenticationError:
            return {
                "valid": False,
                "error": "认证失败：API Key 无效或缺失",
                "error_type": ValidationErrorType.AUTH_ERROR.value
            }
        except APIConnectionError as e:
            return {
                "valid": False,
                "error": f"连接失败：{str(e)}",
                "error_type": ValidationErrorType.CONNECTION_ERROR.value
            }
        except APIStatusError as e:
            return {
                "valid": False,
                "error": f"API 错误 ({e.status_code}): {str(e)}",
                "error_type": ValidationErrorType.API_ERROR.value
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"校验异常: {str(e)}",
                "error_type": ValidationErrorType.UNKNOWN_ERROR.value
            }
        finally:
            if http_client is not None:
                await http_client.aclose()

    async def _validate_http(
        self,
        provider: str,
        api_endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        使用 httpx 校验非 OpenAI 兼容的 Provider
        """
        import httpx

        # Provider 配置映射
        PROVIDER_CONFIG = {
            "xinference": {"path": "/v1/models", "auth": None, "models_key": "data", "id_key": "id"},
            "ollama": {"path": "/api/tags", "auth": None, "models_key": "models", "id_key": "name"},
            "jina": {"path": "/", "auth": "bearer"},
            "cohere": {"path": "/", "auth": "bearer"},
            "voyage": {"path": "/", "auth": "bearer"},
        }

        config = PROVIDER_CONFIG.get(provider, {"path": "/", "auth": "bearer" if api_key else None})

        headers = {}
        if config.get("auth") == "bearer" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            start_time = time.time()
            url = f"{api_endpoint}{config['path']}"
            response = await async_request_user_outbound(
                "GET",
                url,
                user_id,
                timeout=timeout,
                headers=headers,
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # 尝试解析模型列表
                models = []
                try:
                    data = response.json()
                    models_key = config.get("models_key")
                    id_key = config.get("id_key")
                    if models_key and id_key:
                        models = [m.get(id_key) for m in data.get(models_key, []) if m.get(id_key)]
                except Exception:
                    pass

                return {
                    "valid": True,
                    "latency_ms": round(latency_ms, 2),
                    "models": models[:20] if models else None,
                    "message": f"{provider} 连接成功"
                }
            elif response.status_code == 401:
                return {
                    "valid": False,
                    "error": "认证失败：API Key 无效",
                    "error_type": ValidationErrorType.AUTH_ERROR.value
                }
            elif response.status_code == 403:
                return {
                    "valid": False,
                    "error": "访问被拒绝：检查 API Key 权限",
                    "error_type": ValidationErrorType.AUTH_ERROR.value
                }
            else:
                return {
                    "valid": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "error_type": ValidationErrorType.API_ERROR.value
                }

        except httpx.ConnectError:
            return {
                "valid": False,
                "error": f"无法连接到 {api_endpoint}",
                "error_type": ValidationErrorType.CONNECTION_ERROR.value
            }
        except httpx.TimeoutException:
            return {
                "valid": False,
                "error": f"连接超时 ({timeout}s)",
                "error_type": ValidationErrorType.TIMEOUT_ERROR.value
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "error_type": ValidationErrorType.UNKNOWN_ERROR.value
            }

    def validate_config(
        self,
        provider: str,
        api_endpoint: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
        retry_count: int = 2,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        校验外部模型 API 配置 (同步入口)

        Args:
            provider: 提供商名称
            api_endpoint: API 端点 URL
            api_key: API 密钥 (可选)
            model_name: 模型名称 (可选，用于验证模型是否存在)
            provider_config: 提供商特定配置 (如 Azure 的 api_version)
            timeout: 超时时间 (秒)
            retry_count: 重试次数

        Returns:
            {
                "valid": bool,
                "latency_ms": float,
                "models": [str],  # 可选
                "message": str,   # 成功时
                "error": str,     # 失败时
                "error_type": str # 错误分类
            }
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.validate_config_async(
                provider=provider,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model_name=model_name,
                provider_config=provider_config,
                timeout=timeout,
                retry_count=retry_count,
                user_id=user_id,
            )
        )

    async def validate_config_async(
        self,
        provider: str,
        api_endpoint: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
        retry_count: int = 2,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        校验外部模型 API 配置 (异步入口)
        """
        logger.info(f"Validating config: {provider} @ {api_endpoint}")

        # 1. 输入验证
        if not provider:
            return {
                "valid": False,
                "error": "Provider 不能为空",
                "error_type": ValidationErrorType.INVALID_INPUT.value
            }
        if not api_endpoint:
            return {
                "valid": False,
                "error": "API endpoint 不能为空",
                "error_type": ValidationErrorType.INVALID_INPUT.value
            }

        # 2. URL 规范化
        try:
            normalized_endpoint = normalize_api_endpoint(
                api_endpoint,
                provider,
                user_id,
            )
            logger.debug(f"Normalized endpoint: {api_endpoint} -> {normalized_endpoint}")
        except ValueError as e:
            return {
                "valid": False,
                "error": str(e),
                "error_type": ValidationErrorType.INVALID_URL.value
            }

        # 3. 根据 Provider 选择校验方法
        openai_compatible = ["openai", "azure", "deepseek", "zhipu", "qwen", "moonshot", "custom"]

        # 4. 带重试的校验
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                if provider in openai_compatible:
                    result = await self._validate_openai_compatible(
                        provider,
                        normalized_endpoint,
                        api_key,
                        model_name,
                        provider_config,
                        timeout,
                        user_id,
                    )
                else:
                    result = await self._validate_http(
                        provider,
                        normalized_endpoint,
                        api_key,
                        timeout,
                        user_id,
                    )

                # 成功或非连接错误，直接返回
                if result.get("valid") or result.get("error_type") not in [
                    ValidationErrorType.CONNECTION_ERROR.value,
                    ValidationErrorType.TIMEOUT_ERROR.value
                ]:
                    return result

                last_error = result

            except Exception as e:
                logger.warning(f"Validation attempt {attempt + 1} failed: {e}")
                last_error = {
                    "valid": False,
                    "error": str(e),
                    "error_type": ValidationErrorType.UNKNOWN_ERROR.value
                }

            # 重试前等待 (递增退避)
            if attempt < retry_count:
                await asyncio.sleep(1 * (attempt + 1))

        return last_error or {
            "valid": False,
            "error": "校验失败",
            "error_type": ValidationErrorType.UNKNOWN_ERROR.value
        }

    def _resolve_local_deployment_binding(
        self,
        *,
        deployment_id: str | None,
        deployment_replica_id: str | None,
        user_id: str | None,
        require_healthy: bool,
        session: Session | None = None,
        lock: bool = False,
    ) -> tuple[str, str, str, str | None, bool]:
        """Resolve trusted local endpoint and container fields from one child row."""
        if not deployment_id:
            raise ValueError("local deployment config requires deployment_id")
        if session is None:
            with Session(self._get_engine()) as owned_session:
                return self._resolve_local_deployment_binding(
                    deployment_id=deployment_id,
                    deployment_replica_id=deployment_replica_id,
                    user_id=user_id,
                    require_healthy=require_healthy,
                    session=owned_session,
                    lock=lock,
                )

        deployment_statement = select(DeploymentDB).where(
            DeploymentDB.deployment_id == deployment_id
        )
        if lock:
            deployment_statement = deployment_statement.with_for_update()
        deployment = session.exec(deployment_statement).first()
        if deployment is None or (
            user_id is not None and deployment.user_id != user_id
        ):
            raise ValueError("deployment not found")
        if deployment.replica_operation_kind == "delete":
            from ...deployment.deployment_service import ReplicaOperationBusyError

            raise ReplicaOperationBusyError("deployment is being deleted")

        replica_statement = (
            select(DeploymentReplicaDB)
            .where(DeploymentReplicaDB.deployment_id == deployment_id)
            .order_by(DeploymentReplicaDB.replica_index)
        )
        if lock:
            replica_statement = replica_statement.with_for_update()
        replicas = list(session.exec(replica_statement).all())
        if not replicas:
            return (
                deployment.xinference_endpoint,
                deployment.container_name or "",
                deployment.inference_framework or "xinference",
                None,
                False,
            )
        if deployment_replica_id is None:
            if len(replicas) != 1:
                raise ValueError(
                    "replica_id is required for multi-replica deployment"
                )
            replica = replicas[0]
        else:
            replica = next(
                (
                    item
                    for item in replicas
                    if item.replica_id == deployment_replica_id
                ),
                None,
            )
            if replica is None:
                raise ValueError("deployment replica not found")
        if require_healthy and not (
            replica.status == "running" and replica.health_status == "HEALTHY"
        ):
            raise ValueError("deployment replica is not running and healthy")
        return (
            replica.endpoint,
            replica.container_name,
            deployment.inference_framework or "xinference",
            replica.replica_id,
            True,
        )

    # === CRUD Operations ===

    def create_config(
        self,
        config_name: str,
        model_type: str,
        provider: str,
        api_endpoint: str,
        model_name: str,
        api_key: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_default: bool = False,
        user_id: Optional[str] = None,
        validate: bool = False,
        save_on_validation_failure: bool = True,
        # New parameters for local deployment
        source_type: str = "external_api",
        registry_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        deployment_replica_id: Optional[str] = None,
        container_name: Optional[str] = None,
        inference_framework: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new model configuration.

        Args:
            validate: If True, validate API connectivity before saving
            save_on_validation_failure: If True, save config even if validation fails (with pending status)

        Returns:
            Dict with config data and optional 'validation' field

        Raises:
            ValueError: If validation fails and save_on_validation_failure is False
        """
        logger.info(f"Creating model config: {config_name} ({model_type}/{provider})")

        # Normalize model_type (e.g., 'reranker' -> 'rerank')
        model_type = normalize_model_type(model_type)

        trusted_child_binding = False
        if source_type == "local_deployed":
            (
                trusted_endpoint,
                trusted_container,
                trusted_framework,
                deployment_replica_id,
                trusted_child_binding,
            ) = self._resolve_local_deployment_binding(
                deployment_id=deployment_id,
                deployment_replica_id=deployment_replica_id,
                user_id=user_id,
                require_healthy=validate,
            )
            if trusted_child_binding:
                api_endpoint = trusted_endpoint
                container_name = trusted_container
                inference_framework = trusted_framework

        # Normalize untrusted or legacy endpoints (auto-fix common mistakes).
        if not trusted_child_binding:
            try:
                api_endpoint = normalize_api_endpoint(api_endpoint, provider, user_id)
                logger.info(f"Normalized endpoint: {api_endpoint}")
            except ValueError as e:
                raise ValueError(f"Invalid API endpoint: {e}")

        validation_result = None
        initial_status = "active"

        # Validate if requested
        if validate:
            validation_result = self.validate_config(
                provider=provider,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model_name=model_name,
                provider_config=provider_config,
                user_id=user_id,
            )

            if not validation_result.get("valid"):
                if not save_on_validation_failure:
                    raise ValueError(f"Validation failed: {validation_result.get('error')}")
                # Save with pending status
                initial_status = "pending"
                logger.warning(f"Validation failed, saving with pending status: {validation_result.get('error')}")

        with Session(self._get_engine()) as session:
            if registry_id:
                model_registry_service.lock_model_reference(
                    session,
                    registry_id,
                )
            if source_type == "local_deployed":
                (
                    trusted_endpoint,
                    trusted_container,
                    trusted_framework,
                    deployment_replica_id,
                    trusted_child_binding,
                ) = self._resolve_local_deployment_binding(
                    deployment_id=deployment_id,
                    deployment_replica_id=deployment_replica_id,
                    user_id=user_id,
                    require_healthy=validate,
                    session=session,
                    lock=True,
                )
                if trusted_child_binding:
                    api_endpoint = trusted_endpoint
                    container_name = trusted_container
                    inference_framework = trusted_framework

            # If setting as default, unset other defaults for same type
            if is_default and initial_status == "active":
                self._unset_defaults(session, model_type, user_id)

            config = ModelConfigDB(
                config_name=config_name,
                model_type=model_type,
                provider=provider,
                api_endpoint=api_endpoint,
                model_name=model_name,
                api_key=api_key,
                provider_config=provider_config,
                default_params=default_params,
                description=description,
                tags=tags,
                is_default=is_default if initial_status == "active" else False,
                user_id=user_id,
                status=initial_status,
                last_check_status="healthy" if (validation_result and validation_result.get("valid")) else (
                    "error" if validation_result else None
                ),
                last_check_time=now_naive() if validation_result else None,
                last_check_error=validation_result.get("error") if validation_result else None,
                # Local deployment fields
                source_type=source_type,
                registry_id=registry_id,
                deployment_id=deployment_id,
                deployment_replica_id=deployment_replica_id,
                container_name=container_name,
                inference_framework=inference_framework,
            )

            session.add(config)
            session.commit()
            session.refresh(config)

            logger.info(f"Model config created: {config.config_id} (status: {initial_status})")

            result = self._to_dict(config)
            if validation_result:
                result["validation"] = validation_result

            return result

    async def create_config_async(
        self,
        config_name: str,
        model_type: str,
        provider: str,
        api_endpoint: str,
        model_name: str,
        api_key: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_default: bool = False,
        user_id: Optional[str] = None,
        validate: bool = False,
        save_on_validation_failure: bool = True,
        # New parameters for local deployment
        source_type: str = "external_api",
        registry_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        deployment_replica_id: Optional[str] = None,
        container_name: Optional[str] = None,
        inference_framework: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new model configuration (async version).

        Args:
            validate: If True, validate API connectivity before saving
            save_on_validation_failure: If True, save config even if validation fails (with pending status)

        Returns:
            Dict with config data and optional 'validation' field

        Raises:
            ValueError: If validation fails and save_on_validation_failure is False
        """
        logger.info(f"Creating model config: {config_name} ({model_type}/{provider})")

        # Normalize model_type (e.g., 'reranker' -> 'rerank')
        model_type = normalize_model_type(model_type)

        trusted_child_binding = False
        if source_type == "local_deployed":
            (
                trusted_endpoint,
                trusted_container,
                trusted_framework,
                deployment_replica_id,
                trusted_child_binding,
            ) = self._resolve_local_deployment_binding(
                deployment_id=deployment_id,
                deployment_replica_id=deployment_replica_id,
                user_id=user_id,
                require_healthy=validate,
            )
            if trusted_child_binding:
                api_endpoint = trusted_endpoint
                container_name = trusted_container
                inference_framework = trusted_framework

        if not trusted_child_binding:
            try:
                api_endpoint = normalize_api_endpoint(api_endpoint, provider, user_id)
                logger.info(f"Normalized endpoint: {api_endpoint}")
            except ValueError as e:
                raise ValueError(f"Invalid API endpoint: {e}")

        validation_result = None
        initial_status = "active"

        # Validate if requested (async)
        if validate:
            validation_result = await self.validate_config_async(
                provider=provider,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model_name=model_name,
                provider_config=provider_config,
                user_id=user_id,
            )

            if not validation_result.get("valid"):
                if not save_on_validation_failure:
                    raise ValueError(f"Validation failed: {validation_result.get('error')}")
                # Save with pending status
                initial_status = "pending"
                logger.warning(f"Validation failed, saving with pending status: {validation_result.get('error')}")

        with Session(self._get_engine()) as session:
            if registry_id:
                model_registry_service.lock_model_reference(
                    session,
                    registry_id,
                )
            if source_type == "local_deployed":
                (
                    trusted_endpoint,
                    trusted_container,
                    trusted_framework,
                    deployment_replica_id,
                    trusted_child_binding,
                ) = self._resolve_local_deployment_binding(
                    deployment_id=deployment_id,
                    deployment_replica_id=deployment_replica_id,
                    user_id=user_id,
                    require_healthy=validate,
                    session=session,
                    lock=True,
                )
                if trusted_child_binding:
                    api_endpoint = trusted_endpoint
                    container_name = trusted_container
                    inference_framework = trusted_framework

            # If setting as default, unset other defaults for same type
            if is_default and initial_status == "active":
                self._unset_defaults(session, model_type, user_id)

            config = ModelConfigDB(
                config_name=config_name,
                model_type=model_type,
                provider=provider,
                api_endpoint=api_endpoint,
                model_name=model_name,
                api_key=api_key,
                provider_config=provider_config,
                default_params=default_params,
                description=description,
                tags=tags,
                is_default=is_default if initial_status == "active" else False,
                user_id=user_id,
                status=initial_status,
                last_check_status="healthy" if (validation_result and validation_result.get("valid")) else (
                    "error" if validation_result else None
                ),
                last_check_time=now_naive() if validation_result else None,
                last_check_error=validation_result.get("error") if validation_result else None,
                # Local deployment fields
                source_type=source_type,
                registry_id=registry_id,
                deployment_id=deployment_id,
                deployment_replica_id=deployment_replica_id,
                container_name=container_name,
                inference_framework=inference_framework,
            )

            session.add(config)
            session.commit()
            session.refresh(config)

            logger.info(f"Model config created: {config.config_id} (status: {initial_status})")

            result = self._to_dict(config)
            if validation_result:
                result["validation"] = validation_result

            return result

    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration by ID."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(ModelConfigDB.config_id == config_id)
            config = session.exec(stmt).first()
            return self._to_dict(config) if config else None

    def get_config_by_name(
        self,
        config_name: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get configuration by name."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(ModelConfigDB.config_name == config_name)
            if user_id:
                stmt = stmt.where(ModelConfigDB.user_id == user_id)
            config = session.exec(stmt).first()
            return self._to_dict(config) if config else None

    def get_config_by_deployment_id(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration by deployment ID."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(ModelConfigDB.deployment_id == deployment_id)
            config = session.exec(stmt).first()
            return self._to_dict(config) if config else None

    def list_configs_by_deployment_id(
        self,
        deployment_id: str,
    ) -> List[Dict[str, Any]]:
        """Return every child-bound configuration for a deployment group."""
        with Session(self._get_engine()) as session:
            configs = session.exec(
                select(ModelConfigDB)
                .where(ModelConfigDB.deployment_id == deployment_id)
                .order_by(ModelConfigDB.created_at)
            ).all()
            return [self._to_dict(config) for config in configs]

    def get_config_by_deployment_replica_id(
        self,
        deployment_id: str,
        deployment_replica_id: str,
    ) -> Optional[Dict[str, Any]]:
        with Session(self._get_engine()) as session:
            config = session.exec(
                select(ModelConfigDB)
                .where(ModelConfigDB.deployment_id == deployment_id)
                .where(
                    ModelConfigDB.deployment_replica_id
                    == deployment_replica_id
                )
            ).first()
            return self._to_dict(config) if config else None

    def list_configs(
        self,
        model_type: Optional[str] = None,
        provider: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List configurations with optional filters.

        Returns:
            Tuple of (configs, total_count)
        """
        with Session(self._get_engine()) as session:
            # Build filter conditions
            conditions = []
            if model_type:
                conditions.append(ModelConfigDB.model_type == model_type)
            if provider:
                conditions.append(ModelConfigDB.provider == provider)
            if status:
                conditions.append(ModelConfigDB.status == status)
            if user_id:
                # Show user's own configs AND public configs (user_id is NULL)
                conditions.append(
                    or_(ModelConfigDB.user_id == user_id, ModelConfigDB.user_id.is_(None))
                )

            # Get total count
            count_stmt = select(func.count()).select_from(ModelConfigDB)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Get paginated data
            stmt = select(ModelConfigDB)
            for cond in conditions:
                stmt = stmt.where(cond)
            stmt = stmt.order_by(ModelConfigDB.created_at.desc())
            stmt = stmt.offset(offset).limit(limit)

            configs = session.exec(stmt).all()
            return [self._to_dict(c) for c in configs], total

    def update_config(
        self,
        config_id: str,
        config_name: Optional[str] = None,
        model_type: Optional[str] = None,
        provider: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        is_default: Optional[bool] = None,
    ) -> bool:
        """Update configuration."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(ModelConfigDB.config_id == config_id)
            config = session.exec(stmt).first()

            if not config:
                return False

            has_local_binding = (
                getattr(config, "source_type", None) == "local_deployed"
                and bool(
                    getattr(config, "deployment_id", None)
                    or getattr(config, "deployment_replica_id", None)
                )
            )
            if has_local_binding:
                model_type = None
                provider = None
                api_endpoint = None
                api_key = None
                model_name = None
                provider_config = None

            normalized_endpoint = None
            if api_endpoint is not None or provider is not None:
                normalized_endpoint = normalize_api_endpoint(
                    api_endpoint if api_endpoint is not None else config.api_endpoint,
                    provider if provider is not None else config.provider,
                    config.user_id,
                )

            if config_name is not None:
                config.config_name = config_name
            if model_type is not None:
                config.model_type = model_type
            if provider is not None:
                config.provider = provider
            if normalized_endpoint is not None:
                config.api_endpoint = normalized_endpoint
            if api_key is not None:
                config.api_key = api_key
            if model_name is not None:
                config.model_name = model_name
            if provider_config is not None:
                config.provider_config = provider_config
            if default_params is not None:
                config.default_params = default_params
            if description is not None:
                config.description = description
            if tags is not None:
                config.tags = tags
            if status is not None:
                config.status = status
            if is_default is not None:
                if is_default:
                    self._unset_defaults(session, config.model_type, config.user_id)
                config.is_default = is_default

            config.updated_at = now_naive()
            session.add(config)
            session.commit()

            logger.info(f"Model config updated: {config_id}")
            return True

    def delete_config(self, config_id: str) -> bool:
        """Delete configuration."""
        execution_snapshot = snapshot_runtime_executions()
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(ModelConfigDB.config_id == config_id)
            config = session.exec(stmt).first()

            if not config:
                return False

            references = RuntimeDependencyReferences(config_ids=(config_id,))
            locked_dependencies = lock_runtime_dependencies(
                session,
                references,
            )
            locked_config = next(
                (
                    item
                    for item in locked_dependencies.configs
                    if item.config_id == config_id
                ),
                None,
            )
            if locked_config is None:
                raise RuntimeDependencyUnavailableError(
                    "Model config changed while acquiring deletion locks; retry"
                )

            consumers = lock_active_runtime_dependency_consumers(
                session,
                references,
                execution_snapshot=execution_snapshot,
            )
            if consumers:
                raise RuntimeDependencyUnavailableError(
                    "Model config is referenced by an active runtime task"
                )

            milvus_references = list(
                session.exec(
                    select(MilvusCollectionDB)
                    .where(
                        MilvusCollectionDB.embedding_config_id == config_id
                    )
                    .order_by(MilvusCollectionDB.collection_name)
                    .execution_options(populate_existing=True)
                    .with_for_update()
                ).all()
            )
            if milvus_references:
                raise RuntimeDependencyUnavailableError(
                    "Model config is referenced by a Milvus collection"
                )

            session.delete(locked_config)
            session.commit()

            logger.info(f"Model config deleted: {config_id}")
            return True

    # === Default Management ===

    def get_default_config(
        self,
        model_type: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get default configuration for a model type."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(
                ModelConfigDB.model_type == model_type,
                ModelConfigDB.is_default.is_(True),
                ModelConfigDB.status == "active"
            )
            if user_id:
                stmt = stmt.where(ModelConfigDB.user_id == user_id)

            config = session.exec(stmt).first()
            return self._to_dict(config) if config else None

    def set_default(self, config_id: str) -> bool:
        """Set configuration as default for its model type."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(ModelConfigDB.config_id == config_id)
            config = session.exec(stmt).first()

            if not config:
                return False

            # Unset other defaults
            self._unset_defaults(session, config.model_type, config.user_id)

            # Set this as default
            config.is_default = True
            config.updated_at = now_naive()
            session.add(config)
            session.commit()

            logger.info(f"Set default config for {config.model_type}: {config_id}")
            return True

    def _unset_defaults(
        self,
        session: Session,
        model_type: str,
        user_id: Optional[str]
    ):
        """Unset all default flags for a model type（原子 UPDATE，防并发双默认）。"""
        stmt = update(ModelConfigDB).where(
            ModelConfigDB.model_type == model_type,
            ModelConfigDB.is_default.is_(True),
        )
        if user_id:
            stmt = stmt.where(ModelConfigDB.user_id == user_id)

        session.exec(stmt.values(is_default=False))

    # === Search ===

    def search_configs(
        self,
        query: str,
        model_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search configurations by name or description."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(
                (ModelConfigDB.config_name.contains(query)) |
                (ModelConfigDB.description.contains(query)) |
                (ModelConfigDB.model_name.contains(query))
            )

            if model_type:
                stmt = stmt.where(ModelConfigDB.model_type == model_type)
            if user_id:
                stmt = stmt.where(ModelConfigDB.user_id == user_id)

            stmt = stmt.limit(limit)
            configs = session.exec(stmt).all()
            return [self._to_dict(c) for c in configs]

    # === Health Check ===

    def update_check_status(
        self,
        config_id: str,
        status: str,
        error: Optional[str] = None
    ) -> bool:
        """Update health check status."""
        with Session(self._get_engine()) as session:
            stmt = select(ModelConfigDB).where(ModelConfigDB.config_id == config_id)
            config = session.exec(stmt).first()

            if not config:
                return False

            config.last_check_status = status
            config.last_check_time = now_naive()
            config.last_check_error = error
            config.updated_at = now_naive()

            # Update main status based on check result
            if status == "error" and config.status == "active":
                config.status = "error"
            elif status == "healthy" and config.status == "error":
                config.status = "active"

            session.add(config)
            session.commit()

            return True

    async def check_connectivity(self, config_id: str) -> Dict[str, Any]:
        """
        Check connectivity to the configured API endpoint.

        Returns:
            Dict with 'success', 'latency_ms', and optional 'error'
        """
        import time

        config = self.get_config(config_id)
        if not config:
            return {"success": False, "error": "Config not found"}

        provider = config["provider"]
        endpoint = config["api_endpoint"]
        api_key = config.get("api_key")
        model_name = config["model_name"]

        # Normalize endpoint for check
        try:
            endpoint = normalize_api_endpoint(
                endpoint,
                provider,
                config.get("user_id"),
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        try:
            start_time = time.time()
            headers: Dict[str, str] = {}
            params: Optional[Dict[str, str]] = None
            if provider in ["openai", "custom"]:
                request_url = f"{endpoint}/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            elif provider == "azure":
                request_url = f"{endpoint}/openai/models"
                headers = {"api-key": api_key} if api_key else {}
                params = {
                    "api-version": (config.get("provider_config") or {}).get(
                        "api_version",
                        "2024-02-01",
                    )
                }
            elif provider == "xinference":
                request_url = f"{endpoint}/v1/models"
            elif provider == "ollama":
                request_url = f"{endpoint}/api/tags"
            else:
                request_url = endpoint

            response = await async_request_user_outbound(
                "GET",
                request_url,
                config.get("user_id"),
                timeout=10.0,
                headers=headers,
                params=params,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                    # Parse models from response
                    models = []
                    model_details = []
                    try:
                        data = response.json()
                        # Handle different response formats
                        if isinstance(data, dict):
                            # OpenAI/vLLM/SGLang: {"data": [{"id": "model1"}, ...]}
                            # Ollama: {"models": [{"name": "model1"}, ...]}
                            model_list = data.get("data") or data.get("models") or []
                        elif isinstance(data, list):
                            model_list = data
                        else:
                            model_list = []

                        def _coerce_int(value: Any) -> Optional[int]:
                            if value is None:
                                return None
                            try:
                                return int(value)
                            except Exception:
                                return None

                        def _get_value(item: Dict[str, Any], keys: List[str], meta: Dict[str, Any]) -> Any:
                            for key in keys:
                                if key in item and item[key] is not None:
                                    return item[key]
                                if key in meta and meta[key] is not None:
                                    return meta[key]
                            return None

                        for m in model_list:
                            if isinstance(m, str):
                                models.append(m)
                                model_details.append({"id": m})
                                continue
                            if not isinstance(m, dict):
                                continue

                            meta = {}
                            for meta_key in ("metadata", "config", "model_config", "info"):
                                if isinstance(m.get(meta_key), dict):
                                    meta = m.get(meta_key) or {}
                                    break

                            model_id = _get_value(m, ["id", "name", "model", "model_name"], meta)
                            if model_id:
                                models.append(str(model_id))

                            context_length = _get_value(
                                m,
                                ["context_length", "max_context_length", "max_model_len", "max_sequence_length", "max_seq_len", "context_window", "max_position_embeddings"],
                                meta,
                            )
                            max_tokens = _get_value(m, ["max_tokens", "max_completion_tokens", "max_output_tokens"], meta)
                            max_completion_tokens = _get_value(m, ["max_completion_tokens", "max_output_tokens"], meta)
                            embedding_dim = _get_value(m, ["embedding_dim", "embedding_dimension", "dimension", "dim"], meta)
                            model_type = _get_value(m, ["model_type", "type", "object", "task", "modality"], meta)
                            owned_by = _get_value(m, ["owned_by", "provider", "owner", "organization"], meta)
                            created = _get_value(m, ["created", "created_at", "created_time"], meta)
                            capabilities = _get_value(m, ["capabilities", "features", "supported_features"], meta)

                            parent = _get_value(m, ["parent"], meta)

                            model_details.append({
                                "id": str(model_id) if model_id is not None else None,
                                "model_type": model_type,
                                "owned_by": owned_by,
                                "created": created,
                                "context_length": _coerce_int(context_length),
                                "max_tokens": _coerce_int(max_tokens),
                                "max_completion_tokens": _coerce_int(max_completion_tokens),
                                "embedding_dim": _coerce_int(embedding_dim),
                                "capabilities": capabilities,
                                "parent": str(parent) if parent else None,
                            })
                    except Exception:
                        pass  # Ignore parse errors

                    requires_exact_model = (
                        provider == "xinference"
                        and bool(model_name)
                        and (
                            config.get("source_type") == "local_deployed"
                            or bool(config.get("deployment_id"))
                        )
                    )

                    if requires_exact_model and model_name not in models:
                        available_models = ", ".join(models[:5]) if models else "none"
                        error = (
                            f"Model '{model_name}' not found in Xinference endpoint "
                            f"(available: {available_models})"
                        )
                        self.update_check_status(config_id, "error", error)
                        self._sync_deployment_status(config_id, success=False, error=error)
                        return {
                            "success": False,
                            "latency_ms": round(latency_ms, 2),
                            "models": models,
                            "model_details": model_details,
                            "error": error,
                        }

                    # Endpoint is reachable - mark as healthy
                    # For local deployed Xinference configs, the target model must
                    # still exist on the shared endpoint; otherwise the config is stale.
                    self.update_check_status(config_id, "healthy")
                    self._sync_deployment_status(config_id, success=True)

                    # Add warning if model_name not found (but still mark as healthy)
                    warning = None
                    if model_name and models and model_name not in models:
                        warning = f"Model '{model_name}' not found in endpoint (available: {', '.join(models[:5])})"

                    result = {
                        "success": True,
                        "latency_ms": round(latency_ms, 2),
                        "models": models,
                        "model_details": model_details,
                    }
                    if warning:
                        result["warning"] = warning
                    return result
            else:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                self.update_check_status(config_id, "error", error)
                self._sync_deployment_status(config_id, success=False, error=error)
                return {"success": False, "latency_ms": round(latency_ms, 2), "error": error}

        except Exception as e:
            error = str(e)
            self.update_check_status(config_id, "error", error)
            self._sync_deployment_status(config_id, success=False, error=error)
            return {"success": False, "error": error}

    def _sync_deployment_status(self, config_id: str, success: bool, error: Optional[str] = None) -> None:
        """Sync linked deployment status when config connectivity changes."""
        try:
            config = self.get_config(config_id)
            if not config or not config.get("deployment_id"):
                return
            deployment_id = config["deployment_id"]
            with Session(self._get_engine()) as session:
                stmt = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
                dep = session.exec(stmt).first()
                if not dep:
                    return
                changed = False

                if success:
                    # Respect DeploymentDB state transitions:
                    # Only promote to running for in-progress startup states.
                    # A connectivity check must not revive stopped/failed deployments.
                    if dep.status in ("starting", "restarting"):
                        dep.update_status("running")
                        changed = True
                    elif dep.status == "running":
                        # Refresh running state to clear stale errors.
                        dep.update_status("running")
                        changed = True
                else:
                    failure_reason = error or "Connection check failed"
                    # A config connectivity check is observational only: it must not
                    # stop a running deployment that has not actually been stopped.
                    if dep.status in ("pending", "starting", "restarting"):
                        dep.update_status("failed", failure_reason)
                        changed = True

                if changed:
                    session.add(dep)
                    session.commit()
        except Exception as e:
            logger.warning(f"Failed to sync deployment status for config {config_id}: {e}")

    # === Statistics ===

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration statistics via SQL aggregation (no full-row load)."""
        with Session(self._get_engine()) as session:
            def _grouped(column):
                stmt = select(column, func.count()).group_by(column)
                if user_id:
                    stmt = stmt.where(ModelConfigDB.user_id == user_id)
                # row[0] = group value, row[1] = count; skip null/empty keys
                return {
                    row[0]: row[1]
                    for row in session.exec(stmt).all()
                    if row[0]
                }

            count_stmt = select(func.count()).select_from(ModelConfigDB)
            if user_id:
                count_stmt = count_stmt.where(ModelConfigDB.user_id == user_id)
            total = session.exec(count_stmt).one()

            return {
                "total": total,
                "by_type": _grouped(ModelConfigDB.model_type),
                "by_provider": _grouped(ModelConfigDB.provider),
                "by_status": _grouped(ModelConfigDB.status),
            }

    # === Helper Methods ===

    def _get_host_ip(self) -> str:
        """Get host IP for external endpoint conversion."""
        import os
        import socket

        # First try environment variable
        host_ip = os.environ.get("HOST_IP", "").strip()
        if host_ip and host_ip != "host.docker.internal":
            return host_ip

        # Try to resolve host.docker.internal
        try:
            resolved_ip = socket.gethostbyname("host.docker.internal")
            if resolved_ip and not resolved_ip.startswith("127."):
                return resolved_ip
        except socket.gaierror:
            pass

        # Fallback to reading from /proc/net/route
        try:
            with open("/proc/net/route") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[1] == "00000000":
                        gateway_hex = parts[2]
                        gateway_bytes = bytes.fromhex(gateway_hex)
                        gateway_ip = ".".join(str(b) for b in reversed(gateway_bytes))
                        if gateway_ip and not gateway_ip.startswith("127."):
                            return gateway_ip
        except Exception:
            pass

        return "172.17.0.1"

    def _to_external_endpoint(self, endpoint: str) -> str:
        """Convert internal Docker endpoint to external accessible address."""
        import re

        if not endpoint:
            return endpoint

        host_ip = self._get_host_ip()

        # Replace internal addresses with host IP
        if "xinference:" in endpoint:
            endpoint = endpoint.replace("xinference", host_ip)
        elif "localhost" in endpoint:
            endpoint = endpoint.replace("localhost", host_ip)
        elif "127.0.0.1" in endpoint:
            endpoint = endpoint.replace("127.0.0.1", host_ip)
        elif "172.17.0.1" in endpoint:
            endpoint = endpoint.replace("172.17.0.1", host_ip)
        elif endpoint.startswith("http://xf-"):
            # Container name pattern: http://xf-xxx:port -> http://host_ip:port
            match = re.search(r':(\d+)$', endpoint)
            if match:
                port = match.group(1)
                endpoint = f"http://{host_ip}:{port}"

        return endpoint

    def _to_dict(self, config: ModelConfigDB) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        # For local deployed configs, dynamically convert endpoint to external address
        api_endpoint = config.api_endpoint
        if config.source_type == "local_deployed" and api_endpoint:
            api_endpoint = self._to_external_endpoint(api_endpoint)

        return {
            "config_id": config.config_id,
            "config_name": config.config_name,
            "model_type": config.model_type,
            "provider": config.provider,
            "api_endpoint": api_endpoint,
            "api_key": config.api_key,
            "model_name": config.model_name,
            "provider_config": config.provider_config,
            "default_params": config.default_params,
            "description": config.description,
            "tags": config.tags,
            "status": config.status,
            "is_default": config.is_default,
            "last_check_status": config.last_check_status,
            "last_check_time": config.last_check_time,
            "last_check_error": config.last_check_error,
            "user_id": config.user_id,
            "created_at": config.created_at,
            "updated_at": config.updated_at,
            # Local deployment fields
            "source_type": config.source_type,
            "registry_id": config.registry_id,
            "deployment_id": config.deployment_id,
            "deployment_replica_id": config.deployment_replica_id,
            "container_name": config.container_name,
            "inference_framework": config.inference_framework,
        }


# Singleton instance
model_config_service = ModelConfigService()
