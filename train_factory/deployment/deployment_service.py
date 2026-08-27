"""
Deployment service for managing model deployments.

Supports two deployment modes:
- shared: Connect to a shared Xinference service
- container: Auto-create Docker container running Xinference
"""

import logging
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import timedelta
from uuid import uuid4
from train_factory.core.time_utils import now_naive

from sqlmodel import select, func

from ..storage.database import get_session
from ..storage.entities.deployment_entity import DeploymentDB
from ..storage.entities.model_config_entity import ModelConfigDB
from ..storage.services.model_registry_service import model_registry_service
from ..config.settings import get_settings
from .xinference_client import XinferenceClient
from .docker_deployer import docker_deployer
from .launch_config import (
    SglangLaunchConfig,
    VllmLaunchConfig,
    build_sglang_server_argv,
    build_vllm_server_argv,
    is_qwen3_reranker,
    parse_launch_config,
    xinference_model_launch_overrides,
    xinference_model_type,
)

logger = logging.getLogger(__name__)

# Timeout constants for intermediate states
STARTING_TIMEOUT_MINUTES = 10  # Timeout for "starting" state
STOPPING_TIMEOUT_MINUTES = 5   # Timeout for "stopping" state

RUNTIME_MANAGED_CONFIG_KEY = "runtime_managed"
READ_ONLY_CONFIG_KEY = "read_only"
SGLANG_QWEN3_RERANK_TEMPLATE = (
    "/opt/trainfactory/sglang-templates/"
    "qwen3_reranker_no_think.jinja"
)


def _sglang_qwen3_chat_template(
    model_type: str,
    model_family: str,
) -> str | None:
    if is_qwen3_reranker(model_type, model_family):
        return SGLANG_QWEN3_RERANK_TEMPLATE
    return None


def _validate_single_gpu_launch_config(
    launch_config: VllmLaunchConfig | SglangLaunchConfig,
) -> None:
    topology_size = (
        launch_config.tensor_parallel_size
        * launch_config.pipeline_parallel_size
        * launch_config.data_parallel_size
    )
    if topology_size > 1:
        raise ValueError(
            "single-GPU deployment does not support parallel topology"
        )


def get_default_xinference_endpoint() -> str:
    """Get the default Xinference endpoint from settings."""
    settings = get_settings()
    return settings.xinference_endpoint or "http://xinference:9997"


def get_shared_xinference_container_name() -> str:
    """Get the shared Xinference container name for local deployments."""
    return os.environ.get("XINFERENCE_CONTAINER_NAME", "xinference")


def _sanitize_remote_code_value(value: Any) -> Any:
    """Copy deployment config while removing user-controlled remote-code flags."""
    if isinstance(value, dict):
        return {
            key: _sanitize_remote_code_value(item)
            for key, item in value.items()
            if not (
                isinstance(key, str)
                and key.strip().lower().replace("-", "_") == "trust_remote_code"
            )
        }
    if isinstance(value, list):
        return [_sanitize_remote_code_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_remote_code_value(item) for item in value)
    return value


def _sanitize_deployment_config(
    config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Remove operator-owned policy fields from caller-controlled config."""
    if config is None:
        return None
    sanitized = _sanitize_remote_code_value(config)
    if isinstance(sanitized, dict):
        sanitized.pop(RUNTIME_MANAGED_CONFIG_KEY, None)
        sanitized.pop(READ_ONLY_CONFIG_KEY, None)
    return sanitized


def _trusted_model_launch_kwargs(
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build launch kwargs with the operator-owned remote-code policy."""
    sanitized = _sanitize_deployment_config(config) or {}
    sanitized["trust_remote_code"] = get_settings().allow_model_remote_code
    return sanitized


class DeploymentService:
    """Service for model deployment management."""

    @staticmethod
    def _normalize_deploy_mode(mode: Optional[str]) -> str:
        """Normalize legacy deploy_mode values."""
        if not mode or mode == "external":
            return "shared"
        return mode

    @staticmethod
    def _mark_unmanaged_binding(
        config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist the immutable lifecycle policy for an observed external model."""
        marked = dict(config or {})
        marked[RUNTIME_MANAGED_CONFIG_KEY] = False
        marked[READ_ONLY_CONFIG_KEY] = True
        return marked

    @staticmethod
    def _has_unmanaged_binding_marker(deployment: DeploymentDB) -> bool:
        config = deployment.config or {}
        return (
            config.get(RUNTIME_MANAGED_CONFIG_KEY) is False
            and config.get(READ_ONLY_CONFIG_KEY) is True
        )

    def _is_unmanaged_binding(self, deployment: DeploymentDB) -> bool:
        """Recognize current and legacy records created by bind-existing."""
        if self._has_unmanaged_binding_marker(deployment):
            return True

        try:
            model = model_registry_service.get_model(deployment.model_id)
        except Exception as exc:
            logger.warning(
                "Unable to resolve deployment %s provenance: %s",
                deployment.deployment_id,
                exc,
            )
            return True
        return not model or model.get("source_type") == "external_bind"

    @staticmethod
    def _build_managed_container_name(
        deployment_id: str,
        model_name: str,
        inference_framework: Optional[str],
    ) -> str:
        project = os.environ.get("COMPOSE_PROJECT_NAME", "trainfactory")
        model_name_clean = model_name.replace("/", "-").replace("_", "-").lower()
        prefix = {
            "vllm": "vllm",
            "sglang": "sglang",
            "xinference": "xf",
        }.get(inference_framework or "xinference", "xf")
        return (
            f"{project}-{prefix}-{model_name_clean[:20]}-"
            f"{deployment_id[:8]}"
        )

    def _require_managed_container(
        self,
        deployment: DeploymentDB,
        model: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Allow Docker lifecycle calls only for service-created containers."""
        if self._normalize_deploy_mode(deployment.deploy_mode) != "container":
            raise ValueError("Container operation requires container deployment mode")

        if model is None:
            model = model_registry_service.get_model(deployment.model_id)
        if not model:
            raise ValueError(
                "Cannot verify managed container name because the model is missing"
            )

        expected = self._build_managed_container_name(
            deployment.deployment_id,
            model["model_name"],
            deployment.inference_framework,
        )
        if deployment.container_name != expected:
            raise ValueError(
                "Deployment does not reference its service-managed container name"
            )

    def _get_xinference_client(
        self,
        endpoint: str,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> XinferenceClient:
        """Create Xinference client with configured timeout."""
        if timeout is None:
            timeout = get_settings().xinference_request_timeout
        return XinferenceClient(endpoint, timeout=timeout, user_id=user_id)

    def _estimate_model_memory_requirements(self, file_size_bytes: int) -> dict:
        """
        根据模型文件大小估算显存需求。

        显存组成：
        1. 模型权重 = 参数量 × bytes_per_param (FP16=2, FP32=4, INT8=1)
        2. KV Cache = 2 × n_layers × hidden_size × seq_len × batch_size × dtype_size
        3. 激活内存 ≈ 模型权重的 10-30%
        4. 框架开销 ≈ 1-2GB

        参考: https://docs.vllm.ai, https://github.com/vllm-project/vllm

        Args:
            file_size_bytes: 模型文件大小（字节）

        Returns:
            dict: {model_weights_gb, kv_cache_gb, activation_gb, overhead_gb, total_gb}
        """
        file_size_gb = file_size_bytes / (1024 ** 3)

        # 估算参数量：文件大小 / bytes_per_param
        # safetensors/bin 文件通常是 FP16 (2 bytes) 或 BF16 (2 bytes)
        estimated_params_b = file_size_gb / 2  # 单位: Billions

        # 1. 模型权重显存 (推理时通常使用 FP16)
        model_weights_gb = file_size_gb

        # 2. KV Cache 估算
        # 经验值: 小模型 (7B-14B) ~0.15 MB/token, 大模型 (70B+) ~0.35 MB/token
        # 假设 max_seq_len=4096, batch_size=1
        if estimated_params_b < 10:
            kv_bytes_per_token = 0.15 * 1024 * 1024  # 0.15 MB
        elif estimated_params_b < 40:
            kv_bytes_per_token = 0.25 * 1024 * 1024  # 0.25 MB
        else:
            kv_bytes_per_token = 0.35 * 1024 * 1024  # 0.35 MB

        default_seq_len = 4096
        kv_cache_gb = (kv_bytes_per_token * default_seq_len) / (1024 ** 3)

        # 3. 激活内存 (约模型权重的 20%)
        activation_gb = model_weights_gb * 0.2

        # 4. 框架开销 (vLLM/SGLang 约 1-2GB)
        overhead_gb = 1.5

        total_gb = model_weights_gb + kv_cache_gb + activation_gb + overhead_gb

        return {
            "estimated_params_b": estimated_params_b,
            "model_weights_gb": model_weights_gb,
            "kv_cache_gb": kv_cache_gb,
            "activation_gb": activation_gb,
            "overhead_gb": overhead_gb,
            "total_gb": total_gb,
        }

    def _check_deployment_feasibility(self, file_size_bytes: int, gpu_id: Optional[int] = None) -> tuple:
        """
        检查是否可以在指定 GPU 上部署模型。

        Args:
            file_size_bytes: 模型文件大小（字节）
            gpu_id: GPU 设备 ID

        Returns:
            tuple: (can_deploy: bool, reason: str, gpu_free_gb: float, required_gb: float)
        """
        requirements = self._estimate_model_memory_requirements(file_size_bytes)
        required_gb = requirements["total_gb"]

        # 获取 GPU 剩余显存
        gpu_free_gb = 0.0
        try:
            gpu_info = docker_deployer.get_gpu_memory_usage()
            if gpu_info:
                if gpu_id is not None and gpu_id in gpu_info:
                    gpu_free_gb = gpu_info[gpu_id]["free_mb"] / 1024
                else:
                    # 使用剩余显存最多的 GPU
                    best_gpu = max(gpu_info.values(), key=lambda x: x["free_mb"])
                    gpu_free_gb = best_gpu["free_mb"] / 1024
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return False, f"无法获取 GPU 信息: {e}", 0.0, required_gb

        # 预留 10% 安全余量
        safe_free_gb = gpu_free_gb * 0.9

        if required_gb > safe_free_gb:
            return False, (
                f"GPU 剩余显存不足: 需要 {required_gb:.1f}GB, "
                f"可用 {gpu_free_gb:.1f}GB (安全阈值 {safe_free_gb:.1f}GB)"
            ), gpu_free_gb, required_gb

        return True, "可以部署", gpu_free_gb, required_gb

    def _calculate_gpu_memory_utilization(self, file_size_bytes: Optional[int], gpu_id: Optional[int] = None) -> float:
        """
        根据模型文件大小和 GPU 显存自动计算 gpu_memory_utilization。

        计算公式 (基于 vLLM):
        gpu_memory_utilization = (model_weights + kv_cache + activations + overhead) / gpu_total

        Args:
            file_size_bytes: 模型文件大小（字节）
            gpu_id: GPU 设备 ID

        Returns:
            推荐的 gpu_memory_utilization 值
        """
        if not file_size_bytes or file_size_bytes <= 0:
            return 0.1  # 默认保守值

        # 获取 GPU 总显存
        gpu_total_gb = 80.0  # 默认值
        try:
            gpu_info = docker_deployer.get_gpu_memory_usage()
            if gpu_info:
                if gpu_id is not None and gpu_id in gpu_info:
                    gpu_total_gb = gpu_info[gpu_id]["total_mb"] / 1024
                else:
                    first_gpu = next(iter(gpu_info.values()))
                    gpu_total_gb = first_gpu["total_mb"] / 1024
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}, using default {gpu_total_gb}GB")

        # 计算显存需求
        requirements = self._estimate_model_memory_requirements(file_size_bytes)
        required_gb = requirements["total_gb"]

        # 计算利用率 (增加 10% 余量)
        utilization = (required_gb * 1.1) / gpu_total_gb
        utilization = max(0.05, min(0.95, utilization))  # 限制在 0.05-0.95 之间

        # 向上取整到 0.05 的倍数
        utilization = round(utilization * 20 + 0.49) / 20

        logger.info(
            f"Auto-calculated gpu_memory_utilization: "
            f"model={requirements['model_weights_gb']:.1f}GB, "
            f"kv_cache={requirements['kv_cache_gb']:.1f}GB, "
            f"activation={requirements['activation_gb']:.1f}GB, "
            f"total_required={required_gb:.1f}GB, "
            f"gpu_total={gpu_total_gb:.1f}GB, "
            f"utilization={utilization}"
        )
        return utilization

    def _deployment_to_dict(self, deployment: DeploymentDB) -> Dict[str, Any]:
        """Convert deployment ORM object to dictionary."""
        # Dynamically convert internal endpoint to external accessible address
        external_endpoint = self._to_external_endpoint(deployment.xinference_endpoint)

        unmanaged_binding = self._has_unmanaged_binding_marker(deployment)
        container_name = None if unmanaged_binding else deployment.container_name
        if (
            not unmanaged_binding
            and not container_name
            and self._normalize_deploy_mode(deployment.deploy_mode) == "shared"
        ):
            if (deployment.inference_framework or "xinference") == "xinference":
                container_name = get_shared_xinference_container_name()

        return {
            "id": deployment.id,
            "deployment_id": deployment.deployment_id,
            "model_id": deployment.model_id,
            "model_uid": deployment.model_uid,
            "deployment_name": deployment.deployment_name,
            "xinference_endpoint": external_endpoint,
            "replica": deployment.replica,
            "gpu_memory_utilization": deployment.gpu_memory_utilization,
            "deploy_mode": self._normalize_deploy_mode(deployment.deploy_mode),
            "container_name": container_name,
            "gpu_id": deployment.gpu_id,
            "port": deployment.port,
            "inference_framework": deployment.inference_framework,
            "enable_lora": deployment.enable_lora,
            "max_loras": deployment.max_loras,
            "max_lora_rank": deployment.max_lora_rank,
            "external_api_config_id": deployment.external_api_config_id,
            "config": deployment.config,
            "status": deployment.status,
            "error_message": deployment.error_message,
            "user_id": deployment.user_id,
            "created_at": deployment.created_at,
            "updated_at": deployment.updated_at,
            "started_at": deployment.started_at,
            "stopped_at": deployment.stopped_at,
        }

    @staticmethod
    def _extract_external_api_config_id(config: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract normalized external_api_config_id from deployment config."""
        if not config:
            return None
        value = config.get("external_api_config_id")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # ==================== Deployment Management ====================

    def create_deployment(
        self,
        model_id: str,
        xinference_endpoint: str,
        deployment_name: Optional[str] = None,
        replica: int = 1,
        gpu_memory_utilization: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        deploy_mode: str = "shared",
        gpu_id: Optional[int] = None,
        port: Optional[int] = None,
        inference_framework: str = "xinference",
        enable_lora: bool = False,
        max_loras: int = 4,
        max_lora_rank: int = 64,
        external_api_config_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new deployment record.

        Args:
            model_id: Model ID to deploy
            xinference_endpoint: Xinference endpoint
            deployment_name: Optional deployment name
            replica: Number of replicas
            gpu_memory_utilization: GPU memory utilization (auto-calculated if None)
            config: Additional configuration
            user_id: User ID for isolation
            deploy_mode: "shared" or "container"
            gpu_id: GPU device ID (for container mode)
            port: Port to expose (for container mode)
            inference_framework: Framework to use (vllm, sglang, xinference)
            enable_lora: Enable LoRA hot-loading
            max_loras: Maximum number of LoRA adapters
            max_lora_rank: Maximum LoRA rank
            external_api_config_id: Optional external API config ID (tenant binding)
        """
        config = _sanitize_deployment_config(config)

        # Verify model exists
        model = model_registry_service.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        deploy_mode = self._normalize_deploy_mode(deploy_mode)

        # Auto-calculate gpu_memory_utilization if not specified
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self._calculate_gpu_memory_utilization(model.get('file_size'))

        normalized_external_api_config_id = (
            (external_api_config_id or "").strip()
            or self._extract_external_api_config_id(config)
        )

        with get_session() as session:
            deployment = DeploymentDB(
                model_id=model_id,
                xinference_endpoint=xinference_endpoint,
                deployment_name=deployment_name or f"{model['model_name']}-deployment",
                replica=replica,
                gpu_memory_utilization=gpu_memory_utilization,
                deploy_mode=deploy_mode,
                container_name=None,  # Will be set after we have deployment_id
                gpu_id=gpu_id,
                port=port,
                inference_framework=inference_framework,
                enable_lora=enable_lora,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
                external_api_config_id=normalized_external_api_config_id,
                config=config,
                user_id=user_id,
                status="pending",
            )
            session.add(deployment)
            session.commit()
            session.refresh(deployment)

            # Generate container name using deployment_id for uniqueness
            container_name = None
            if deploy_mode == "container":
                container_name = self._build_managed_container_name(
                    deployment.deployment_id,
                    model["model_name"],
                    inference_framework,
                )
            elif (inference_framework or "xinference") == "xinference":
                container_name = get_shared_xinference_container_name()

            if container_name:
                deployment.container_name = container_name
                session.commit()
                session.refresh(deployment)

            logger.info(f"Created deployment: {deployment.deployment_id} (mode={deploy_mode}, container={container_name})")
            return self._deployment_to_dict(deployment)

    def bind_existing_model(
        self,
        endpoint: str,
        model_uid: str,
        model_name: Optional[str] = None,
        model_type: str = "embedding",
        deployment_name: Optional[str] = None,
        inference_framework: str = "xinference",
        container_name: Optional[str] = None,
        gpu_id: Optional[int] = None,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Bind an existing model running on an inference endpoint.

        Creates a deployment record for a model already running on the server,
        without launching a new model. The deployment will start in 'running' status.

        Args:
            endpoint: Inference server endpoint
            model_uid: Model UID on the server
            model_name: Display name for the model
            model_type: Model type (embedding/rerank/llm)
            deployment_name: Optional deployment name
            inference_framework: Framework type (xinference/vllm/sglang)
            container_name: Docker container name (for local deployments)
            gpu_id: GPU device ID (for local deployments)
            user_id: User ID for isolation
            config: Additional deployment config metadata
        """
        config = self._mark_unmanaged_binding(
            _sanitize_deployment_config(config)
        )
        endpoint = endpoint.rstrip("/")
        model_uid = model_uid.strip()
        if not model_uid:
            raise ValueError("model_uid is required")

        # Generate deployment name if not provided
        if not deployment_name:
            deployment_name = f"bound-{model_uid[:20]}"

        with get_session() as session:
            duplicate_stmt = (
                select(DeploymentDB)
                .where(DeploymentDB.xinference_endpoint == endpoint)
                .where(DeploymentDB.model_uid == model_uid)
            )
            if session.exec(duplicate_stmt).first():
                raise ValueError(
                    f"Model {model_uid} at {endpoint} is already bound"
                )

            # Bound models use tenant-owned placeholder registry entries but never
            # inherit Docker/runtime lifecycle authority from the binding request.
            existing_models = model_registry_service.list_models(
                limit=1000,
                user_id=user_id,
            )[0]
            matching_model = None
            for existing_model in existing_models:
                if (
                    existing_model.get("model_path", "").rstrip("/") == endpoint
                    and existing_model.get("model_name") == model_uid
                ):
                    matching_model = existing_model
                    break

            if matching_model:
                model_id = matching_model["model_id"]
            else:
                registry_entry = model_registry_service.register_model(
                    model_name=model_name or model_uid,
                    model_path=endpoint,
                    model_type=model_type,
                    source_type="external_bind",
                    description=f"External model bound from {endpoint}",
                    extra_metadata={
                        "bound_model_uid": model_uid,
                        "bind_endpoint": endpoint,
                    },
                    path_unique_key=model_uid,
                    user_id=user_id,
                )
                model_id = registry_entry["model_id"]

            normalized_external_api_config_id = self._extract_external_api_config_id(config)
            deployment = DeploymentDB(
                model_id=model_id,
                model_uid=model_uid,
                deployment_name=deployment_name,
                xinference_endpoint=endpoint,
                replica=1,
                gpu_memory_utilization=0.0,  # Unknown for bound models
                deploy_mode="shared",
                inference_framework=inference_framework,
                container_name=None,
                gpu_id=None,
                external_api_config_id=normalized_external_api_config_id,
                config=config,
                status="running",  # Already running
                user_id=user_id,
            )
            # Set started_at since it's already running
            deployment.started_at = now_naive()
            session.add(deployment)
            session.commit()
            session.refresh(deployment)

            logger.info(f"Bound existing model: {model_uid} at {endpoint} -> deployment {deployment.deployment_id}")

            # Auto-create model config
            self._create_config_for_deployment(deployment, {
                "model_name": model_name or model_uid,
                "model_type": model_type,
                "model_path": endpoint,
            })

            return self._deployment_to_dict(deployment)

    def create_container_deployment(
        self,
        model_id: str,
        deployment_name: Optional[str] = None,
        gpu_id: Optional[int] = None,
        port: Optional[int] = None,
        replica: int = 1,
        gpu_memory_utilization: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        auto_start: bool = True,
        inference_framework: str = "xinference",
        enable_lora: bool = False,
        max_loras: int = 4,
        max_lora_rank: int = 64,
        external_api_config_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a container-based deployment (auto-creates Docker container).

        Args:
            model_id: Model ID to deploy
            deployment_name: Optional deployment name
            gpu_id: GPU device ID (auto-select if None)
            port: Port to expose (auto-assign if None)
            replica: Number of replicas
            gpu_memory_utilization: GPU memory utilization (auto-calculated if None)
            config: Additional configuration
            user_id: User ID for isolation
            auto_start: Start deployment immediately after creation
            inference_framework: Framework to use (vllm, sglang, xinference)
            enable_lora: Enable LoRA hot-loading
            max_loras: Maximum number of LoRA adapters
            max_lora_rank: Maximum LoRA rank
            external_api_config_id: Optional external API config ID (tenant binding)
        """
        # Get model info to generate container name
        model = model_registry_service.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Auto-select GPU if not specified
        if gpu_id is None:
            gpu_id = docker_deployer.select_gpu(min_free_mb=4000)
            logger.info(f"Auto-selected GPU {gpu_id}")

        # Check deployment feasibility based on model size and GPU memory
        file_size = model.get('file_size')
        if file_size:
            can_deploy, reason, gpu_free_gb, required_gb = self._check_deployment_feasibility(file_size, gpu_id)
            if not can_deploy:
                raise ValueError(f"部署检查失败: {reason}")
            logger.info(f"Deployment feasibility check passed: required={required_gb:.1f}GB, available={gpu_free_gb:.1f}GB")

        # Auto-calculate gpu_memory_utilization if not specified
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self._calculate_gpu_memory_utilization(file_size, gpu_id)

        # Auto-assign port if not specified
        port_was_auto_assigned = port is None
        port_reservation_owner = None
        if port_was_auto_assigned:
            port_reservation_owner = uuid4().hex
            port = docker_deployer.find_available_port(
                owner_token=port_reservation_owner
            )
            logger.info(f"Auto-assigned port {port}")

        try:
            # Create deployment record first (container_name generated inside using deployment_id)
            # Use placeholder endpoint, will be updated after container_name is known
            deployment = self.create_deployment(
                model_id=model_id,
                xinference_endpoint="",  # Will be updated below
                deployment_name=deployment_name,
                replica=replica,
                gpu_memory_utilization=gpu_memory_utilization,
                config=config,
                user_id=user_id,
                deploy_mode="container",
                gpu_id=gpu_id,
                port=port,
                inference_framework=inference_framework,
                enable_lora=enable_lora,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
                external_api_config_id=external_api_config_id,
            )

            # Now update endpoint using the generated container_name
            container_name = deployment.get("container_name")
            endpoint = f"http://{container_name}:{port}"
            logger.info(f"Container endpoint: {endpoint}")

            # Update endpoint in database
            with get_session() as session:
                statement = select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment["deployment_id"]
                )
                db_deployment = session.exec(statement).first()
                if db_deployment:
                    db_deployment.xinference_endpoint = endpoint
                    session.commit()
                    deployment["xinference_endpoint"] = endpoint

            if auto_start:
                result = self.start_deployment(deployment["deployment_id"])
                if port_reservation_owner is not None:
                    docker_deployer.confirm_port(
                        port,
                        owner_token=port_reservation_owner,
                    )
                return result

            if port_reservation_owner is not None:
                docker_deployer.confirm_port(
                    port,
                    owner_token=port_reservation_owner,
                )
            return deployment

        except Exception:
            # Release the reserved port on failure
            if port_was_auto_assigned and port_reservation_owner is not None:
                docker_deployer.release_port(
                    port,
                    owner_token=port_reservation_owner,
                )
                logger.info(f"Released port {port} due to deployment failure")
            raise

    def start_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Start a deployment on Xinference."""
        with get_session() as session:
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            if self._is_unmanaged_binding(deployment):
                raise ValueError(
                    "Externally bound deployments are read-only and cannot be started"
                )

            if deployment.status == "running":
                return self._deployment_to_dict(deployment)

            # Get model info
            model = model_registry_service.get_model(deployment.model_id)
            if not model:
                deployment.update_status("failed", "Model not found")
                session.add(deployment)
                session.commit()
                raise ValueError(f"Model not found: {deployment.model_id}")

            if self._normalize_deploy_mode(deployment.deploy_mode) == "container":
                self._require_managed_container(deployment, model)

            # Ensure shared Xinference deployments record container name
            if deployment.deploy_mode != "container" and not deployment.container_name:
                if (deployment.inference_framework or "xinference") == "xinference":
                    deployment.container_name = get_shared_xinference_container_name()

            # Update status to starting
            deployment.update_status("starting")
            session.add(deployment)
            session.commit()

            # Generate model_uid if not set
            model_uid = deployment.model_uid or f"{model['model_name']}-{deployment.deployment_id[:8]}"

            try:
                if deployment.deploy_mode == "container":
                    # Container mode: create Docker container with Xinference
                    self._start_container_deployment(deployment, model, model_uid)
                else:
                    # External mode: launch model on existing Xinference
                    self._start_external_deployment(deployment, model, model_uid)

                # Update deployment with success
                deployment.model_uid = model_uid
                deployment.update_status("running")
                session.add(deployment)
                session.commit()
                session.refresh(deployment)

                logger.info(f"Started deployment {deployment_id}: {model_uid}")

                # Auto-create model config for the deployment
                self._create_config_for_deployment(deployment, model)
                self._sync_configs_from_deployments([deployment])

                return self._deployment_to_dict(deployment)

            except Exception as e:
                error_msg = str(e)
                cleanup_error = None

                # A launch request can fail after the runtime has already been
                # created. Reclaim it before finalizing the database state.
                if deployment.deploy_mode == "container" and deployment.container_name:
                    try:
                        removed = docker_deployer.remove_container(
                            deployment.container_name
                        )
                        if not removed:
                            raise RuntimeError(
                                f"Failed to remove container {deployment.container_name}"
                            )
                        logger.info(
                            f"Cleaned up failed container: {deployment.container_name}"
                        )
                    except Exception as exc:
                        cleanup_error = exc
                elif model_uid:
                    try:
                        client = self._get_xinference_client(
                            deployment.xinference_endpoint,
                            user_id=deployment.user_id,
                        )
                        client.terminate_model(model_uid)
                        logger.info(f"Cleaned up failed shared model: {model_uid}")
                    except Exception as exc:
                        cleanup_error = exc

                deployment.model_uid = model_uid
                if cleanup_error is None:
                    deployment.update_status("failed", error_msg)
                else:
                    deployment.update_status(
                        "stopping",
                        f"{error_msg}; runtime cleanup failed: {cleanup_error}",
                    )
                session.add(deployment)
                session.commit()
                self._sync_configs_from_deployments([deployment])
                logger.error(f"Failed to start deployment {deployment_id}: {e}")
                if cleanup_error is not None:
                    logger.warning(
                        "Failed to cleanup runtime for deployment %s: %s",
                        deployment_id,
                        cleanup_error,
                    )

                raise

    def _convert_model_path_for_xinference(self, model_path: str) -> str:
        """
        Convert model path from API container format to Xinference container format.

        API container: /app/models/xxx -> Xinference: /models/xxx
        API container: /app/output/xxx -> Xinference: /app/output/xxx (same mount)

        Note: Xinference mounts models at /models, not /app/models.
        """
        if model_path.startswith("/app/models/"):
            return model_path.replace("/app/models/", "/models/", 1)
        return model_path

    def _get_xinference_model_name(self, model: Dict[str, Any]) -> str:
        """
        Get the model name that Xinference recognizes.

        For trained models, we need to use the base model's name since Xinference
        only recognizes built-in model names. The model_path will point to our
        trained checkpoint.
        """
        # If this is a trained model with a base_model_path pointing to a registered model
        base_model_path = model.get('base_model_path', '')
        if base_model_path and base_model_path.startswith('/app/models/'):
            # Extract model_id from path like /app/models/cade0d26-6cc1-482a-b00f-2661883f9d5f
            base_model_id = base_model_path.split('/')[-1]
            # Look up the base model to get its name
            base_model = model_registry_service.get_model(base_model_id)
            if base_model:
                logger.info(f"Using base model name '{base_model['model_name']}' for trained model '{model['model_name']}'")
                return base_model['model_name']

        # Fallback to the model's own name
        return model['model_name']

    def _start_external_deployment(
        self,
        deployment: DeploymentDB,
        model: Dict[str, Any],
        model_uid: str,
    ) -> None:
        """Start deployment on existing Xinference service."""
        client = self._get_xinference_client(
            deployment.xinference_endpoint,
            user_id=deployment.user_id,
        )
        model_path = self._convert_model_path_for_xinference(model["model_path"])
        xinference_model_name = self._get_xinference_model_name(model)

        extra_kwargs = _trusted_model_launch_kwargs(deployment.config)
        extra_kwargs.update(
            xinference_model_launch_overrides(
                model_type=model["model_type"],
                model_family=xinference_model_name,
            )
        )
        if deployment.gpu_id is not None:
            extra_kwargs["gpu_idx"] = deployment.gpu_id

        client.launch_model(
            model_uid=model_uid,
            model_name=xinference_model_name,
            model_path=model_path,
            model_type=xinference_model_type(model["model_type"]),
            replica=deployment.replica,
            gpu_memory_utilization=deployment.gpu_memory_utilization,
            **extra_kwargs,
        )

        config = deployment.config or {}
        deployment.config = {
            **config,
            "dtype": config.get("dtype") or "bfloat16",
        }

    def _start_container_deployment(
        self,
        deployment: DeploymentDB,
        model: Dict[str, Any],
        model_uid: str,
    ) -> None:
        """Start deployment by creating Docker container with appropriate framework."""
        if not deployment.container_name or deployment.port is None or deployment.gpu_id is None:
            raise ValueError("Container deployment requires container_name, port, and gpu_id")

        framework = deployment.inference_framework or "xinference"

        if framework == "vllm":
            self._start_vllm_container(deployment, model, model_uid)
        elif framework == "sglang":
            self._start_sglang_container(deployment, model, model_uid)
        else:
            self._start_xinference_container(deployment, model, model_uid)

        logger.info(f"Container deployment started: {deployment.container_name} on port {deployment.port}")

    def _start_xinference_container(
        self,
        deployment: DeploymentDB,
        model: Dict[str, Any],
        model_uid: str,
    ) -> None:
        """Start Xinference container deployment."""
        xinference_model_name = self._get_xinference_model_name(model)
        success, msg, docker_cmd = docker_deployer.create_xinference_container(
            container_name=deployment.container_name,
            port=deployment.port,
            gpu_id=deployment.gpu_id,
            model_name=xinference_model_name,
            model_uid=model_uid,
            model_path=model["model_path"],
            model_type=xinference_model_type(model["model_type"]),
        )
        config = deployment.config or {}
        deployment.config = {
            **config,
            "docker_cmd": docker_cmd,
            "dtype": config.get("dtype") or "bfloat16",
        }

        if not success:
            raise RuntimeError(f"Failed to create Xinference container: {msg}")

        if not docker_deployer.wait_for_xinference(
            deployment.xinference_endpoint,
            timeout=180,
            user_id=deployment.user_id,
        ):
            logs = docker_deployer.get_container_logs(deployment.container_name, tail=50)
            raise RuntimeError(f"Xinference service did not start. Logs:\n{logs}")

        if not docker_deployer.wait_for_model(
            deployment.xinference_endpoint,
            model_uid,
            timeout=300,
            user_id=deployment.user_id,
        ):
            logs = docker_deployer.get_container_logs(deployment.container_name, tail=50)
            raise RuntimeError(f"Model failed to load. Logs:\n{logs}")

    def _start_vllm_container(
        self,
        deployment: DeploymentDB,
        model: Dict[str, Any],
        model_uid: str,
    ) -> None:
        """Start vLLM container deployment."""
        config = deployment.config or {}
        if "launch_config" in config:
            launch_payload = config["launch_config"]
        else:
            launch_payload = {
                "framework": "vllm",
                "dtype": config.get("dtype") or "auto",
                "enforce_eager": config.get("enforce_eager", False),
            }
        launch_config = parse_launch_config(launch_payload)
        if not isinstance(launch_config, VllmLaunchConfig):
            raise ValueError("deployment launch_config is not vLLM")
        _validate_single_gpu_launch_config(launch_config)
        server_argv = build_vllm_server_argv(
            launch_config,
            model_path=model["model_path"],
            served_model_name=model_uid,
            port=deployment.port,
            gpu_memory_utilization=deployment.gpu_memory_utilization,
            model_type=model["model_type"],
            enable_lora=deployment.enable_lora,
            max_loras=deployment.max_loras,
            max_lora_rank=deployment.max_lora_rank,
            trust_remote_code=get_settings().allow_model_remote_code,
            model_family=self._get_xinference_model_name(model),
        )
        success, msg, docker_cmd = docker_deployer.create_vllm_container(
            container_name=deployment.container_name,
            port=deployment.port,
            gpu_ids=(deployment.gpu_id,),
            server_argv=server_argv,
        )
        deployment.config = {
            **config,
            "docker_cmd": docker_cmd,
            "dtype": launch_config.dtype,
            "enforce_eager": launch_config.enforce_eager,
        }

        if not success:
            raise RuntimeError(f"Failed to create vLLM container: {msg}")

        if not docker_deployer.wait_for_service(
            deployment.xinference_endpoint,
            timeout=300,
            user_id=deployment.user_id,
            framework="vllm",
        ):
            logs = docker_deployer.get_container_logs(deployment.container_name, tail=50)
            raise RuntimeError(f"vLLM service did not start. Logs:\n{logs}")

    def _start_sglang_container(
        self,
        deployment: DeploymentDB,
        model: Dict[str, Any],
        model_uid: str,
    ) -> None:
        """Start SGLang container deployment."""
        sglang_model_name = self._get_xinference_model_name(model)
        chat_template = _sglang_qwen3_chat_template(
            model["model_type"],
            sglang_model_name,
        )

        config = deployment.config or {}
        if "launch_config" in config:
            launch_payload = config["launch_config"]
        else:
            launch_payload = {
                "framework": "sglang",
                "dtype": config.get("dtype") or "auto",
                "attention_backend": config.get("attention_backend"),
            }
        launch_config = parse_launch_config(launch_payload)
        if not isinstance(launch_config, SglangLaunchConfig):
            raise ValueError("deployment launch_config is not SGLang")
        _validate_single_gpu_launch_config(launch_config)
        server_argv = build_sglang_server_argv(
            launch_config,
            model_path=model["model_path"],
            served_model_name=model_uid,
            port=deployment.port,
            gpu_memory_utilization=deployment.gpu_memory_utilization,
            model_type=model["model_type"],
            enable_lora=deployment.enable_lora,
            max_loras=deployment.max_loras,
            max_lora_rank=deployment.max_lora_rank,
            chat_template=chat_template,
            trust_remote_code=get_settings().allow_model_remote_code,
            model_family=sglang_model_name,
        )
        success, msg, docker_cmd = docker_deployer.create_sglang_container(
            container_name=deployment.container_name,
            port=deployment.port,
            gpu_ids=(deployment.gpu_id,),
            server_argv=server_argv,
        )
        deployment.config = {
            **config,
            "docker_cmd": docker_cmd,
            "dtype": launch_config.dtype,
            **(
                {"attention_backend": launch_config.attention_backend}
                if launch_config.attention_backend
                else {}
            ),
        }

        if not success:
            raise RuntimeError(f"Failed to create SGLang container: {msg}")

        if not docker_deployer.wait_for_service(
            deployment.xinference_endpoint,
            timeout=300,
            user_id=deployment.user_id,
            framework="sglang",
        ):
            logs = docker_deployer.get_container_logs(deployment.container_name, tail=50)
            raise RuntimeError(f"SGLang service did not start. Logs:\n{logs}")

    def stop_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Stop a running deployment."""
        with get_session() as session:
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            if self._is_unmanaged_binding(deployment):
                raise ValueError(
                    "Externally bound deployments are read-only and cannot be stopped"
                )

            if self._normalize_deploy_mode(deployment.deploy_mode) == "container":
                self._require_managed_container(deployment)

            if deployment.status == "stopped":
                self._sync_configs_from_deployments([deployment])
                return self._deployment_to_dict(deployment)

            # For states with no running resources, go directly to stopped
            if deployment.status in ("pending", "failed"):
                if deployment.status == "failed":
                    deployment.update_status("pending")
                deployment.update_status("stopped")
                session.add(deployment)
                session.commit()
                session.refresh(deployment)
                self._sync_configs_from_deployments([deployment])
                logger.info(f"Stopped deployment {deployment_id} (was {deployment.status})")
                return self._deployment_to_dict(deployment)

            # Update status to stopping (for running/starting states)
            deployment.update_status("stopping")
            session.add(deployment)
            session.commit()

            try:
                if deployment.deploy_mode == "container" and deployment.container_name:
                    # Container mode: stop and remove container
                    stopped = docker_deployer.stop_container(deployment.container_name)
                    if not stopped:
                        raise RuntimeError(
                            f"Failed to stop container {deployment.container_name}"
                        )
                    logger.info(f"Stopped container {deployment.container_name}")
                elif deployment.model_uid:
                    # External mode: terminate model on Xinference
                    try:
                        client = self._get_xinference_client(
                            deployment.xinference_endpoint,
                            user_id=deployment.user_id,
                        )
                        client.terminate_model(deployment.model_uid)
                    except Exception as terminate_error:
                        raise RuntimeError(
                            f"Failed to terminate model {deployment.model_uid}"
                        ) from terminate_error

                deployment.update_status("stopped")
                session.add(deployment)
                session.commit()
                session.refresh(deployment)
                self._sync_configs_from_deployments([deployment])

                logger.info(f"Stopped deployment {deployment_id}")
                return self._deployment_to_dict(deployment)

            except Exception as e:
                # Keep a retryable state until the runtime confirms cleanup.
                deployment.update_status("stopping", str(e))
                session.add(deployment)
                session.commit()
                self._sync_configs_from_deployments([deployment])
                logger.error(f"Failed to stop deployment {deployment_id}: {e}")
                raise RuntimeError(
                    f"Failed to stop deployment {deployment_id}: {e}"
                ) from e

    def restart_deployment(
        self,
        deployment_id: str,
        mode: str = "auto",
        reset_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Restart a deployment to clear GPU memory cache.

        Supports different restart modes based on the inference framework:
        - vLLM/SGLang (single-model container): Always restart container
        - Xinference (multi-model shared): Can reload model only or restart container

        Args:
            deployment_id: Deployment ID to restart
            mode: Restart mode
                - "auto": Auto-select based on framework (recommended)
                    - vLLM/SGLang → restart container
                    - Xinference → reload model (unload + launch)
                - "model": Reload model only (Xinference only, preserves other models)
                - "container": Force restart container (affects all models in container)
            reset_gpu: Reset GPU memory after stopping (container mode only)

        Returns:
            Updated deployment dict

        Raises:
            ValueError: If deployment not found or invalid mode
            RuntimeError: If restart operation fails
        """
        with get_session() as session:
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            if self._is_unmanaged_binding(deployment):
                raise ValueError(
                    "Externally bound deployments are read-only and cannot be restarted"
                )

            framework = deployment.inference_framework or "xinference"
            deploy_mode = self._normalize_deploy_mode(deployment.deploy_mode)
            if deploy_mode == "container":
                self._require_managed_container(deployment)

            # Validate mode
            if mode not in ("auto", "model", "container"):
                raise ValueError(f"Invalid restart mode: {mode}. Use 'auto', 'model', or 'container'")

            # Model-only reload is only supported for Xinference
            if mode == "model" and framework != "xinference":
                raise ValueError(f"Mode 'model' is only supported for Xinference, not {framework}")

            # Container restart requires container mode
            if mode == "container" and deploy_mode != "container":
                raise ValueError("Mode 'container' requires container deployment mode")

            # Determine actual restart strategy
            fallback_start = False
            if mode == "auto":
                if framework in ("vllm", "sglang"):
                    # vLLM/SGLang: always restart container (single model = container)
                    actual_mode = "container"
                else:
                    # Xinference: prefer model reload to avoid affecting other models
                    if deploy_mode == "container":
                        # Container mode Xinference: reload model
                        actual_mode = "model"
                    else:
                        # External Xinference: reload model
                        actual_mode = "model"
            else:
                actual_mode = mode

            if actual_mode == "model" and not deployment.model_uid:
                if deployment.status in ("failed", "stopped", "pending"):
                    logger.info(
                        f"Deployment {deployment_id} missing model_uid; "
                        "fallback to start_deployment."
                    )
                    fallback_start = True
                else:
                    raise ValueError("Model reload requires model_uid")

            if not fallback_start:
                logger.info(f"Restarting deployment {deployment_id}: mode={mode} -> actual={actual_mode}, framework={framework}")

                # Update status to restarting
                deployment.update_status("restarting")
                session.add(deployment)
                session.commit()

                try:
                    if actual_mode == "container":
                        # Restart container (vLLM/SGLang or forced container restart)
                        self._restart_container(deployment, reset_gpu)
                        # Persist container restart status update
                        session.add(deployment)
                        session.commit()
                    else:
                        # Reload model (Xinference)
                        self._reload_xinference_model(deployment, session)

                    # Refresh and return
                    session.refresh(deployment)
                    self._sync_configs_from_deployments([deployment])
                    return self._deployment_to_dict(deployment)

                except Exception as e:
                    # Mark as failed on error
                    deployment.update_status("failed", str(e))
                    session.add(deployment)
                    session.commit()
                    self._sync_configs_from_deployments([deployment])
                    raise

        if fallback_start:
            return self.start_deployment(deployment_id)

    def _restart_container(
        self,
        deployment: DeploymentDB,
        reset_gpu: bool = False,
    ) -> bool:
        """
        Restart deployment by restarting Docker container.

        This completely restarts the container process, clearing all GPU memory.
        Used for vLLM/SGLang (single-model) or forced Xinference container restart.

        Args:
            deployment: Deployment record
            reset_gpu: Reset GPU after stopping (experimental)

        Returns:
            True if successful
        """
        self._require_managed_container(deployment)

        gpu_id = deployment.gpu_id

        # Optionally reset GPU memory
        if reset_gpu and gpu_id is not None:
            # Stop container first
            docker_deployer.stop_container(deployment.container_name)
            success, message = docker_deployer.reset_gpu(gpu_id)
            if not success:
                logger.warning(f"GPU reset failed: {message}")
            # Start container again
            docker_deployer.restart_container(deployment.container_name)
        else:
            # Direct restart
            success, message = docker_deployer.restart_container(deployment.container_name)
            if not success:
                raise RuntimeError(f"Container restart failed: {message}")

        # Wait for service to be ready
        framework = deployment.inference_framework or "xinference"
        timeout = 300  # 5 minutes

        if framework == "xinference":
            # Wait for Xinference service + model
            if not docker_deployer.wait_for_xinference(
                deployment.xinference_endpoint,
                timeout=timeout,
                user_id=deployment.user_id,
            ):
                raise RuntimeError("Xinference service did not restart in time")
            if deployment.model_uid:
                if not docker_deployer.wait_for_model(
                    deployment.xinference_endpoint,
                    deployment.model_uid,
                    timeout=timeout,
                    user_id=deployment.user_id,
                ):
                    raise RuntimeError("Model did not reload after container restart")
        else:
            # vLLM/SGLang: wait for /v1/models endpoint
            if not docker_deployer.wait_for_service(
                deployment.xinference_endpoint,
                timeout=timeout,
                user_id=deployment.user_id,
                framework=framework,
            ):
                raise RuntimeError(f"{framework} service did not restart in time")

        # Update status to running
        deployment.update_status("running")
        logger.info(f"Container restart successful: {deployment.container_name}")
        return True

    def _reload_xinference_model(
        self,
        deployment: DeploymentDB,
        session,
    ) -> bool:
        """
        Reload model on Xinference by unloading and relaunching.

        This clears the model's GPU memory without affecting other models
        running on the same Xinference instance.

        Args:
            deployment: Deployment record
            session: Database session

        Returns:
            True if successful
        """
        if not deployment.model_uid:
            raise ValueError("Model reload requires model_uid")

        # Get model info for relaunch
        model = model_registry_service.get_model(deployment.model_id)
        if not model:
            raise ValueError(f"Model not found: {deployment.model_id}")

        client = self._get_xinference_client(
            deployment.xinference_endpoint,
            user_id=deployment.user_id,
        )
        model_uid = deployment.model_uid

        # Step 1: Unload (terminate) the model
        logger.info(f"Unloading model {model_uid} from Xinference")
        try:
            client.terminate_model(model_uid)
        except Exception as e:
            logger.warning(f"Failed to terminate model (may already be stopped): {e}")

        # Wait a moment for GPU memory to be released
        time.sleep(2)

        # Step 2: Relaunch the model
        logger.info(f"Relaunching model {model_uid} on Xinference")

        # Convert model path for Xinference container
        model_path = self._convert_model_path_for_xinference(model['model_path'])
        xinference_model_name = self._get_xinference_model_name(model)

        extra_kwargs = _trusted_model_launch_kwargs(deployment.config)
        extra_kwargs.update(
            xinference_model_launch_overrides(
                model_type=model["model_type"],
                model_family=xinference_model_name,
            )
        )

        client.launch_model(
            model_uid=model_uid,
            model_name=xinference_model_name,
            model_path=model_path,
            model_type=xinference_model_type(model["model_type"]),
            replica=deployment.replica,
            gpu_memory_utilization=deployment.gpu_memory_utilization,
            **extra_kwargs,
        )

        # Wait for model to be ready
        if not docker_deployer.wait_for_model(
            deployment.xinference_endpoint,
            model_uid,
            timeout=300,
            user_id=deployment.user_id,
        ):
            raise RuntimeError("Model did not reload in time")

        # Update status to running
        deployment.update_status("running")
        session.add(deployment)
        session.commit()

        logger.info(f"Model reload successful: {model_uid}")
        return True

    def delete_deployment(self, deployment_id: str, force: bool = False) -> bool:
        """Delete a deployment record.

        Args:
            deployment_id: The deployment ID to delete
            force: If True, also delete related model configs

        Returns:
            True if deleted, False if deployment not found

        Raises:
            ValueError: If deployment has related configs and force=False
        """
        with get_session() as session:
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                return False

            unmanaged_binding = self._is_unmanaged_binding(deployment)
            if (
                not unmanaged_binding
                and self._normalize_deploy_mode(deployment.deploy_mode) == "container"
            ):
                self._require_managed_container(deployment)

            # Check for related model configs
            config_stmt = select(ModelConfigDB).where(ModelConfigDB.deployment_id == deployment_id)
            configs = session.exec(config_stmt).all()

            if configs:
                if not force:
                    raise ValueError(
                        f"Cannot delete deployment {deployment_id}: has {len(configs)} config(s). "
                        "Use force=True to delete anyway."
                    )
                # Force delete: remove related configs
                for config in configs:
                    session.delete(config)
                    logger.warning(f"Force deleted config {config.config_id} for deployment {deployment_id}")

            # Stop and clean up resources
            if (
                not unmanaged_binding
                and deployment.deploy_mode == "container"
                and deployment.container_name
            ):
                # Attempt container cleanup if no other running deployment shares the same container name
                other_stmt = (
                    select(DeploymentDB)
                    .where(DeploymentDB.container_name == deployment.container_name)
                    .where(DeploymentDB.deployment_id != deployment_id)
                    .where(DeploymentDB.status == "running")
                )
                other_running = session.exec(other_stmt).first()
                if other_running:
                    logger.info(
                        f"Skipping container removal: another running deployment "
                        f"{other_running.deployment_id} shares container {deployment.container_name}"
                    )
                else:
                    try:
                        removed = docker_deployer.remove_container(
                            deployment.container_name
                        )
                        if not removed:
                            raise RuntimeError(
                                f"Failed to remove container {deployment.container_name}"
                            )
                        logger.info(f"Removed container {deployment.container_name}")
                    except Exception as cleanup_error:
                        raise RuntimeError(
                            "Failed to remove container during deployment deletion: "
                            f"{deployment.container_name}"
                        ) from cleanup_error
            elif (
                not unmanaged_binding
                and deployment.status
                in {"running", "starting", "restarting", "stopping"}
                and deployment.model_uid
            ):
                # External mode: terminate model
                try:
                    client = self._get_xinference_client(
                        deployment.xinference_endpoint,
                        user_id=deployment.user_id,
                    )
                    client.terminate_model(deployment.model_uid)
                except Exception as terminate_error:
                    raise RuntimeError(
                        "Failed to terminate model during deployment deletion: "
                        f"{deployment.model_uid}"
                    ) from terminate_error

            session.delete(deployment)
            session.commit()
            logger.info(f"Deleted deployment {deployment_id}")
            return True

    def update_deployment_config(
        self,
        deployment_id: str,
        config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update deployment config.

        Args:
            deployment_id: Deployment ID
            config: New config to persist (replaces existing config)

        Returns:
            Updated deployment dict
        """
        with get_session() as session:
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            unmanaged_binding = self._is_unmanaged_binding(deployment)
            config = _sanitize_deployment_config(config)
            if unmanaged_binding:
                config = self._mark_unmanaged_binding(config)
            deployment.config = config
            deployment.external_api_config_id = self._extract_external_api_config_id(config)
            deployment.updated_at = now_naive()
            session.add(deployment)
            session.commit()
            session.refresh(deployment)
            return self._deployment_to_dict(deployment)

    # ==================== Query Operations ====================

    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment by ID."""
        with get_session() as session:
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if deployment:
                return self._deployment_to_dict(deployment)
            return None

    def list_deployments(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sync: bool = False,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List deployments with optional filters.

        Args:
            sync: If True, sync status with Xinference before returning.

        Returns:
            Tuple of (deployments, total_count)
        """
        # Sync running deployments with Xinference if requested
        if sync:
            self.sync_all_running()

        with get_session() as session:
            # Build filter conditions
            conditions = []
            if model_id:
                conditions.append(DeploymentDB.model_id == model_id)
            if status:
                conditions.append(DeploymentDB.status == status)
            if user_id:
                conditions.append(DeploymentDB.user_id == user_id)

            # Get total count
            count_stmt = select(func.count()).select_from(DeploymentDB)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Get paginated data
            statement = select(DeploymentDB)
            for cond in conditions:
                statement = statement.where(cond)
            statement = statement.order_by(DeploymentDB.created_at.desc())
            statement = statement.offset(offset).limit(limit)
            deployments = session.exec(statement).all()
            return [self._deployment_to_dict(d) for d in deployments], total

    # ==================== Status Sync ====================

    def sync_status(self, deployment_id: str) -> Dict[str, Any]:
        """Sync deployment status with Xinference."""
        with get_session() as session:
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            if not deployment.model_uid:
                return self._deployment_to_dict(deployment)

            try:
                client = self._get_xinference_client(
                    deployment.xinference_endpoint,
                    user_id=deployment.user_id,
                )
                status = client.get_model_status(deployment.model_uid)

                # Map Xinference status to our status
                if status == "running":
                    if deployment.status != "running":
                        deployment.update_status("running")
                elif status == "stopped":
                    if deployment.status not in ["stopped", "pending"]:
                        deployment.update_status("stopped")

                deployment.updated_at = now_naive()
                session.add(deployment)
                session.commit()
                session.refresh(deployment)

                return self._deployment_to_dict(deployment)

            except Exception as e:
                logger.warning(f"Failed to sync status for {deployment_id}: {e}")
                return self._deployment_to_dict(deployment)

    def sync_all_running(self) -> int:
        """
        Sync status for all deployments marked as 'running', 'starting', 'restarting', or 'stopping'.
        Checks if models are actually running and updates DB accordingly.
        - For container mode: checks Docker container status
        - For external mode: checks Xinference model status
        - Handles timeout for stuck 'starting'/'stopping' states
        Returns number of deployments that were updated.
        """
        updated_count = 0
        now = now_naive()
        starting_timeout = timedelta(minutes=STARTING_TIMEOUT_MINUTES)
        stopping_timeout = timedelta(minutes=STOPPING_TIMEOUT_MINUTES)

        with get_session() as session:
            # Get all running, starting, restarting, or stopping deployments
            statement = select(DeploymentDB).where(
                DeploymentDB.status.in_(["running", "starting", "restarting", "stopping"])
            )
            deployments = session.exec(statement).all()

            # Separate container and external deployments
            container_deployments = []
            external_deployments = []
            for dep in deployments:
                if dep.deploy_mode == "container" and dep.container_name:
                    container_deployments.append(dep)
                elif dep.model_uid:
                    external_deployments.append(dep)

            # Check container deployments via Docker
            for dep in container_deployments:
                container_running = docker_deployer.container_running(dep.container_name)
                container_exists = docker_deployer.container_exists(dep.container_name)

                if dep.status == "stopping":
                    # Check timeout for stopping state
                    if dep.updated_at and (now - dep.updated_at) > stopping_timeout:
                        dep.update_status("failed", f"Stopping timed out after {STOPPING_TIMEOUT_MINUTES} minutes (auto-sync)")
                        session.add(dep)
                        updated_count += 1
                        logger.warning(f"Auto-sync: deployment {dep.deployment_id} stopping timed out")
                    elif not container_running:
                        # Container stopped successfully
                        dep.update_status("stopped")
                        session.add(dep)
                        updated_count += 1
                        logger.info(f"Auto-sync: deployment {dep.deployment_id} stopped successfully")
                elif dep.status in ("starting", "restarting"):
                    # Check timeout for starting state
                    if dep.updated_at and (now - dep.updated_at) > starting_timeout:
                        timeout_label = "Restarting" if dep.status == "restarting" else "Starting"
                        dep.update_status("failed", f"{timeout_label} timed out after {STARTING_TIMEOUT_MINUTES} minutes (auto-sync)")
                        session.add(dep)
                        updated_count += 1
                        logger.warning(f"Auto-sync: deployment {dep.deployment_id} {dep.status} timed out")
                    elif not container_running:
                        if not container_exists:
                            dep.update_status("failed", "Container not found (auto-sync)")
                        else:
                            dep.update_status("failed", "Container stopped during restart (auto-sync)" if dep.status == "restarting" else "Container stopped during startup (auto-sync)")
                        session.add(dep)
                        updated_count += 1
                        logger.info(f"Auto-sync: container deployment {dep.deployment_id} marked as {dep.status}")
                    elif container_running:
                        # Container is running, update status to running
                        dep.update_status("running")
                        session.add(dep)
                        updated_count += 1
                        logger.info(f"Auto-sync: container deployment {dep.deployment_id} marked as running")
                elif not container_running:
                    # Running deployment with no running container = stopped
                    dep.update_status("stopped", "Container not running (auto-sync)")
                    session.add(dep)
                    updated_count += 1
                    logger.info(f"Auto-sync: container deployment {dep.deployment_id} marked as {dep.status}")
                elif dep.error_message and "auto-sync" in dep.error_message:
                    # Clear stale error message if container is running
                    dep.error_message = None
                    dep.updated_at = now_naive()
                    session.add(dep)
                    logger.info(f"Auto-sync: cleared error for running container {dep.deployment_id}")

            # Group external deployments by xinference_endpoint to minimize API calls
            endpoint_models: Dict[Tuple[str, Optional[str]], List[str]] = {}
            for dep in external_deployments:
                endpoint_key = (dep.xinference_endpoint, dep.user_id)
                if endpoint_key not in endpoint_models:
                    endpoint_models[endpoint_key] = []
                endpoint_models[endpoint_key].append(dep.model_uid)

            # Keep discovery results scoped to the endpoint and user that were
            # queried. An endpoint failure means its state is unknown, not empty.
            running_models_by_endpoint: Dict[
                Tuple[str, Optional[str]], set[str]
            ] = {}
            for endpoint_key in endpoint_models:
                endpoint, user_id = endpoint_key
                try:
                    client = self._get_xinference_client(
                        endpoint,
                        timeout=3,
                        user_id=user_id,
                    )
                    models = client.list_models()
                    running_models_by_endpoint[endpoint_key] = {
                        model.get("id")
                        for model in models
                        if model.get("id")
                    }
                except Exception as e:
                    logger.warning(f"Failed to get models from {endpoint}: {e}")

            # Update external deployments based on actual status
            for dep in external_deployments:
                endpoint_key = (dep.xinference_endpoint, dep.user_id)
                running_models = running_models_by_endpoint.get(endpoint_key)
                if running_models is None:
                    continue
                if dep.model_uid not in running_models:
                    if dep.status == "restarting":
                        if dep.updated_at and (now - dep.updated_at) > starting_timeout:
                            dep.update_status("failed", f"Restarting timed out after {STARTING_TIMEOUT_MINUTES} minutes (auto-sync)")
                            session.add(dep)
                            updated_count += 1
                            logger.warning(f"Auto-sync: external deployment {dep.deployment_id} restarting timed out")
                        continue
                    dep.update_status("stopped", "Model not running on Xinference (auto-sync)")
                    session.add(dep)
                    updated_count += 1
                    logger.info(f"Auto-sync: external deployment {dep.deployment_id} marked as stopped")
                elif dep.error_message and "auto-sync" in dep.error_message:
                    # Clear stale error message if model is running
                    dep.error_message = None
                    dep.updated_at = now_naive()
                    session.add(dep)
                    logger.info(f"Auto-sync: cleared error for running model {dep.deployment_id}")

            session.commit()

            # Sync config status for all deployments that changed
            if updated_count > 0:
                self._sync_configs_from_deployments(deployments)

            return updated_count

    def _sync_configs_from_deployments(self, deployments: list) -> None:
        """Sync linked model config status based on deployment status."""
        try:
            from ..storage.services.model_config_service import model_config_service
            for dep in deployments:
                config = model_config_service.get_config_by_deployment_id(dep.deployment_id)
                if not config:
                    continue
                config_id = config["config_id"]
                if dep.status == "running":
                    if config.get("last_check_status") != "healthy" or config.get("status") != "active":
                        model_config_service.update_check_status(config_id, "healthy")
                elif dep.status in ("stopped", "failed"):
                    if config.get("last_check_status") != "error" or config.get("status") != "error":
                        error_msg = dep.error_message or f"Deployment {dep.status}"
                        model_config_service.update_check_status(config_id, "error", error_msg)
        except Exception as e:
            logger.warning(f"Failed to sync config status from deployments: {e}")

    # ==================== Quick Deploy ====================

    def deploy_from_model(
        self,
        model_id: str,
        xinference_endpoint: str,
        auto_start: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Quick deploy a registered model."""
        deployment = self.create_deployment(
            model_id=model_id,
            xinference_endpoint=xinference_endpoint,
            **kwargs,
        )

        if auto_start:
            return self.start_deployment(deployment["deployment_id"])

        return deployment

    # ==================== Auto Config Creation ====================

    def _get_host_ip(self) -> str:
        """
        Get host IP address for external access.

        Returns the host's actual IP address that can be accessed from outside
        the Docker network.
        """
        import os
        import socket

        # First try environment variable (set in docker-compose.yml or .env)
        host_ip = os.environ.get("HOST_IP", "").strip()
        if host_ip and host_ip != "host.docker.internal":
            logger.info(f"Using HOST_IP from environment: {host_ip}")
            return host_ip

        # Try to resolve host.docker.internal (works on Docker Desktop for Mac/Windows)
        try:
            resolved_ip = socket.gethostbyname("host.docker.internal")
            if resolved_ip and not resolved_ip.startswith("127."):
                logger.info(f"Resolved host.docker.internal to {resolved_ip}")
                return resolved_ip
        except socket.gaierror:
            pass

        # On Linux: get default gateway from /proc/net/route
        try:
            with open("/proc/net/route") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[1] == "00000000":
                        # Default route found, gateway is in hex (little-endian)
                        gateway_hex = parts[2]
                        gateway_bytes = bytes.fromhex(gateway_hex)
                        gateway_ip = ".".join(str(b) for b in reversed(gateway_bytes))
                        if gateway_ip and not gateway_ip.startswith("127."):
                            # For Docker bridge, gateway is usually host
                            # But we need the actual host IP, not Docker gateway
                            # Try to get the host IP from the gateway's network
                            # Gateway like 172.19.0.1 -> Host is likely on same network
                            logger.info(f"Detected gateway: {gateway_ip}")
                            # Check if XINFERENCE_ENDPOINT has a resolvable host
                            xinf_endpoint = os.environ.get("XINFERENCE_ENDPOINT", "")
                            if "://" in xinf_endpoint:
                                host_part = xinf_endpoint.split("://")[1].split(":")[0]
                                if host_part not in ["localhost", "127.0.0.1", "xinference"]:
                                    try:
                                        resolved = socket.gethostbyname(host_part)
                                        if not resolved.startswith("127."):
                                            logger.info(f"Using IP from XINFERENCE_ENDPOINT: {resolved}")
                                            return resolved
                                    except BaseException:
                                        pass
                            # Return gateway as approximation (works for many setups)
                            return gateway_ip
        except Exception as e:
            logger.debug(f"Failed to read /proc/net/route: {e}")

        # Final fallback
        logger.warning("Could not detect host IP, using 172.17.0.1")
        return "172.17.0.1"

    def _to_external_endpoint(self, endpoint: str) -> str:
        """
        Convert internal Docker endpoint to external accessible address.

        Replaces Docker internal addresses (container names, localhost, xinference)
        with the host IP so users can access from outside Docker network.
        """
        if not endpoint:
            return endpoint

        host_ip = self._get_host_ip()

        # Replace common internal addresses with host IP
        # Order matters: check more specific patterns first
        if "xinference:" in endpoint:
            endpoint = endpoint.replace("xinference", host_ip)
        elif "localhost" in endpoint:
            endpoint = endpoint.replace("localhost", host_ip)
        elif "127.0.0.1" in endpoint:
            endpoint = endpoint.replace("127.0.0.1", host_ip)
        elif "172.17.0.1" in endpoint:
            # Docker bridge IP - replace with actual host IP
            endpoint = endpoint.replace("172.17.0.1", host_ip)
        else:
            # Container name patterns: http://xf-xxx:port, http://vllm-xxx:port, http://sglang-xxx:port
            # Convert to http://host_ip:port
            import re
            container_prefixes = ("http://xf-", "http://vllm-", "http://sglang-")
            if any(endpoint.startswith(prefix) for prefix in container_prefixes):
                match = re.search(r':(\d+)$', endpoint)
                if match:
                    port = match.group(1)
                    endpoint = f"http://{host_ip}:{port}"

        return endpoint

    def _normalize_endpoint_for_config(self, endpoint: str) -> str:
        """
        Normalize endpoint URL for config creation.

        Replace localhost with actual host IP for container accessibility.
        """
        # Use the unified conversion method
        return self._to_external_endpoint(endpoint)

    def _create_config_for_deployment(
        self,
        deployment: DeploymentDB,
        model: Dict[str, Any],
    ) -> None:
        """
        Auto-create a model config after successful deployment.

        Args:
            deployment: The deployment record
            model: The model info dict
        """
        try:
            from ..storage.services.model_config_service import model_config_service

            # Check if config already exists for this deployment
            existing = model_config_service.get_config_by_deployment_id(deployment.deployment_id)
            if existing:
                logger.info(f"Config already exists for deployment {deployment.deployment_id}, skipping")
                return

            # Generate config name
            config_name = f"{model['model_name']}-deployed-{deployment.deployment_id[:8]}"

            # Build API endpoint for the deployed model
            # For container mode: use host IP + port (accessible from anywhere)
            # For external mode: normalize localhost to host IP
            if deployment.deploy_mode == "container" and deployment.port:
                host_ip = self._get_host_ip()
                api_endpoint = f"http://{host_ip}:{deployment.port}"
                logger.info(f"Container mode: using host endpoint {api_endpoint}")
            else:
                api_endpoint = self._normalize_endpoint_for_config(deployment.xinference_endpoint)

            logger.info(f"Auto-creating config for deployment: {config_name}")

            model_config_service.create_config(
                config_name=config_name,
                model_type=model['model_type'],
                provider="xinference",
                api_endpoint=api_endpoint,
                model_name=deployment.model_uid,
                description=f"Auto-created from deployment: {deployment.deployment_name or deployment.deployment_id}",
                user_id=deployment.user_id,
                validate=False,  # Skip validation (avoid async conflict); validate via UI later
                save_on_validation_failure=True,
                # Mark as local deployment
                source_type="local_deployed",
                registry_id=model['model_id'],
                deployment_id=deployment.deployment_id,
                container_name=deployment.container_name,
                inference_framework=deployment.inference_framework,
            )

            logger.info(f"Auto-created config for deployment {deployment.deployment_id}: {config_name}")

        except Exception as e:
            # Don't fail the deployment if config creation fails
            logger.warning(f"Failed to auto-create config for deployment {deployment.deployment_id}: {e}")

    def create_config_for_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Create a model config for an existing deployment.

        Use this when automatic config creation failed during deployment
        but the deployment is actually running.

        Args:
            deployment_id: Deployment ID

        Returns:
            Created config dict

        Raises:
            ValueError: If deployment not found or not in running state
        """
        from ..storage.services.model_config_service import model_config_service

        # Get deployment
        deployment_dict = self.get_deployment(deployment_id)
        if not deployment_dict:
            raise ValueError(f"Deployment not found: {deployment_id}")

        if deployment_dict["status"] != "running":
            raise ValueError(f"Deployment is not running (status: {deployment_dict['status']})")

        # Get model info
        model = model_registry_service.get_model(deployment_dict["model_id"])
        if not model:
            raise ValueError(f"Model not found: {deployment_dict['model_id']}")

        # Check if config already exists for this deployment
        existing_configs, _ = model_config_service.list_configs()
        for cfg in existing_configs:
            if cfg.get("deployment_id") == deployment_id:
                raise ValueError(f"Config already exists for deployment: {cfg['config_id']}")

        # Generate config name
        config_name = f"{model['model_name']}-deployed-{deployment_id[:8]}"

        # Build API endpoint
        # For container mode: use host IP + port (accessible from anywhere)
        # For external mode: normalize localhost to host IP
        if deployment_dict.get("deploy_mode") == "container" and deployment_dict.get("port"):
            host_ip = self._get_host_ip()
            api_endpoint = f"http://{host_ip}:{deployment_dict['port']}"
            logger.info(f"Container mode: using host endpoint {api_endpoint}")
        else:
            api_endpoint = self._normalize_endpoint_for_config(deployment_dict["xinference_endpoint"])

        logger.info(f"Creating config for deployment: {config_name}")

        config = model_config_service.create_config(
            config_name=config_name,
            model_type=model['model_type'],
            provider="xinference",
            api_endpoint=api_endpoint,
            model_name=deployment_dict["model_uid"],
            description=f"Auto-created from deployment: {deployment_dict['deployment_name'] or deployment_id}",
            user_id=deployment_dict.get("user_id"),
            validate=False,  # Skip validation (avoid async conflict); user can validate via UI
            save_on_validation_failure=True,
            source_type="local_deployed",
            registry_id=model['model_id'],
            deployment_id=deployment_id,
            container_name=deployment_dict.get("container_name"),
            inference_framework=deployment_dict.get("inference_framework"),
        )

        logger.info(f"Created config for deployment {deployment_id}: {config['config_id']}")
        return config


# Global service instance
deployment_service = DeploymentService()
