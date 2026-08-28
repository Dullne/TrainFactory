"""
Deployment service for managing model deployments.

Supports two deployment modes:
- shared: Connect to a shared Xinference service
- container: Auto-create Docker container running Xinference
"""

import hashlib
import json
import logging
import os
import posixpath
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
from train_factory.core.time_utils import now_naive

from sqlalchemy import delete, update
from sqlmodel import select, func

from ..storage.database import get_session
from ..storage.entities.deployment_entity import DeploymentDB
from ..storage.entities.deployment_replica_entity import DeploymentReplicaDB
from ..storage.entities.loaded_adapter_entity import LoadedAdapterDB
from ..storage.entities.model_config_entity import ModelConfigDB
from ..storage.entities.external_api_config_entity import ExternalApiConfigDB
from ..storage.services.model_registry_service import model_registry_service
from ..storage.services.model_artifact_membership_service import (
    lock_model_artifact_membership,
)
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
    xinference_model_type,
    xinference_model_launch_overrides,
)
from .replica_planner import (
    DeploymentPlan,
    _plan_gpu_assignments,
    plan_deployment_replicas,
)

logger = logging.getLogger(__name__)

# Timeout constants for intermediate states
STARTING_TIMEOUT_MINUTES = 10  # Timeout for "starting" state
STOPPING_TIMEOUT_MINUTES = 5   # Timeout for "stopping" state

RUNTIME_MANAGED_CONFIG_KEY = "runtime_managed"
READ_ONLY_CONFIG_KEY = "read_only"
REPLICA_SCHEMA_VERSION_KEY = "replica_schema_version"
DEFERRED_AUTO_START_CONFIG_KEY = "_deferred_auto_start"
REGISTRY_ARTIFACT_SIGNATURE_CONFIG_KEY = "_registry_artifact_signature"
LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY = (
    "_operator_054_legacy_replica_schema_marker_backup"
)
REPLICA_OPERATION_LEASE_SECONDS = 120
REPLICA_OPERATION_HEARTBEAT_SECONDS = 15
LEGACY_RUNTIME_REPLICA_ID = "legacy"
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


class ReplicaOperationBusyError(RuntimeError):
    """Raised when another process already owns a replica lifecycle fence."""


class ReplicaOperationLostError(RuntimeError):
    """Raised when a stale worker no longer owns its lifecycle generation."""


class DeploymentReplicaNotFoundError(ValueError):
    """Raised when a replica resource must remain indistinguishable from denied."""


class DeploymentReplicaStateConflictError(ValueError):
    """Raised when the requested replica lifecycle action conflicts with state."""


class _ReplicaContainerIdentityUncertainError(RuntimeError):
    """Raised when a created container cannot be proven safe to remove."""

    def __init__(self, replica_id: str) -> None:
        super().__init__(
            "replica container identity could not be verified; "
            "deployment plan retained"
        )
        self.replica_id = replica_id


@dataclass(frozen=True)
class ReplicaOperationClaim:
    deployment_id: str
    token: str
    generation: int
    operation: str
    replica_id: str | None


@dataclass
class _DeploymentDeleteProgress:
    """Track whether a failed delete may already have changed the runtime."""

    runtime_cleanup_started: bool = False


@dataclass(frozen=True)
class _LegacyStatusSyncSnapshot:
    """Detached legacy deployment state used to fence remote status probes."""

    deployment_id: str
    generation: int
    status: str
    updated_at: datetime | None
    model_id: str
    model_uid: str | None
    xinference_endpoint: str
    user_id: str | None
    deploy_mode: str
    container_name: str | None
    inference_framework: str
    error_message: str | None
    started_at: datetime | None
    stopped_at: datetime | None


_LEGACY_SYNC_UNCHANGED = object()


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
        sanitized.pop(REPLICA_SCHEMA_VERSION_KEY, None)
        sanitized.pop(DEFERRED_AUTO_START_CONFIG_KEY, None)
        sanitized.pop(REGISTRY_ARTIFACT_SIGNATURE_CONFIG_KEY, None)
        sanitized.pop(LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY, None)
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

    def __init__(self) -> None:
        self._claim_heartbeat_lock = threading.Lock()
        self._claim_heartbeats: Dict[str, tuple[threading.Event, threading.Thread]] = {}

    @staticmethod
    def _registry_artifact_signature(model: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the exact registry artifact chosen under the model lock."""
        model_path = str(model.get("model_path") or "")
        return {
            "model_id": model.get("model_id"),
            "model_path_sha256": hashlib.sha256(
                model_path.encode("utf-8")
            ).hexdigest(),
            "version": model.get("version"),
            "model_name": model.get("model_name"),
            "model_type": model.get("model_type"),
            "base_model_path": model.get("base_model_path"),
            "source_type": model.get("source_type"),
            "is_adapter": model.get("is_adapter"),
            "file_size": model.get("file_size"),
        }

    @classmethod
    def _require_persisted_registry_artifact(
        cls,
        deployment: Dict[str, Any],
        model: Dict[str, Any],
    ) -> None:
        expected = (deployment.get("config") or {}).get(
            REGISTRY_ARTIFACT_SIGNATURE_CONFIG_KEY
        )
        if expected is None:
            return
        if expected != cls._registry_artifact_signature(model):
            raise DeploymentReplicaStateConflictError(
                "deployment registry model artifact changed; recreate deployment"
            )

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
        model_name_clean = re.sub(
            r"[^a-z0-9_.-]+",
            "-",
            model_name.lower(),
        ).strip("-._")
        if not model_name_clean:
            model_name_clean = "model"
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

    def _deployment_to_dict_with_replicas(
        self,
        session: Any,
        deployment: DeploymentDB,
    ) -> Dict[str, Any]:
        """Return one parent deployment with its ordered child endpoints."""
        result = self._deployment_to_dict(deployment)
        replicas = session.exec(
            select(DeploymentReplicaDB)
            .where(DeploymentReplicaDB.deployment_id == deployment.deployment_id)
            .order_by(DeploymentReplicaDB.replica_index)
        ).all()
        result["replica_instances"] = [replica.to_dict() for replica in replicas]
        return result

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

    @classmethod
    def _normalize_external_api_config_id(
        cls,
        config: Optional[Dict[str, Any]],
        explicit_config_id: Optional[str],
    ) -> Optional[str]:
        explicit = (explicit_config_id or "").strip() or None
        embedded = cls._extract_external_api_config_id(config)
        if explicit and embedded and explicit != embedded:
            raise ValueError(
                "external_api_config_id mismatch between field and config payload"
            )
        return explicit or embedded

    @staticmethod
    def _lock_external_api_configs(
        session: Any,
        config_ids: Tuple[Optional[str], ...],
        *,
        expected_user_id: Optional[str],
    ) -> Tuple[ExternalApiConfigDB, ...]:
        """Lock exact API references before any model/deployment write lock."""
        normalized_ids = tuple(
            sorted(
                {
                    config_id.strip()
                    for config_id in config_ids
                    if isinstance(config_id, str) and config_id.strip()
                }
            )
        )
        if not normalized_ids:
            return ()
        configs = tuple(
            session.exec(
                select(ExternalApiConfigDB)
                .where(ExternalApiConfigDB.config_id.in_(normalized_ids))
                .order_by(ExternalApiConfigDB.config_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        actual_ids = {config.config_id for config in configs}
        missing = next(
            (
                config_id
                for config_id in normalized_ids
                if config_id not in actual_ids
            ),
            None,
        )
        if missing is not None:
            raise ValueError(f"External API config not found: {missing}")
        if expected_user_id is None:
            raise ValueError("External API config ownership context is required")
        if any(config.user_id != expected_user_id for config in configs):
            raise ValueError("External API config ownership mismatch")
        return configs

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

        normalized_external_api_config_id = (
            self._normalize_external_api_config_id(
                config,
                external_api_config_id,
            )
        )

        with get_session() as session:
            self._lock_external_api_configs(
                session,
                (normalized_external_api_config_id,),
                expected_user_id=user_id,
            )
            locked_model = model_registry_service.lock_model_reference(
                session,
                model_id,
            )
            model = locked_model.to_dict()
            if gpu_memory_utilization is None:
                gpu_memory_utilization = self._calculate_gpu_memory_utilization(
                    model.get("file_size")
                )
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

        normalized_external_api_config_id = self._extract_external_api_config_id(
            config
        )

        with get_session() as session:
            self._lock_external_api_configs(
                session,
                (normalized_external_api_config_id,),
                expected_user_id=user_id,
            )
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

            model_registry_service.lock_model_reference(session, model_id)
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

    def _create_single_container_deployment(
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

    @staticmethod
    def _replica_group_to_dict(
        deployment: DeploymentDB,
        replicas: list[DeploymentReplicaDB],
    ) -> Dict[str, Any]:
        result = deployment.to_dict()
        result["replica_instances"] = [
            replica.to_dict()
            for replica in sorted(replicas, key=lambda item: item.replica_index)
        ]
        return result

    @staticmethod
    def _load_replica_group(deployment_id: str) -> Dict[str, Any]:
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise RuntimeError("replica deployment record is missing")
            replicas = list(
                session.exec(
                    select(DeploymentReplicaDB)
                    .where(DeploymentReplicaDB.deployment_id == deployment_id)
                    .order_by(DeploymentReplicaDB.replica_index)
                ).all()
            )
            return DeploymentService._replica_group_to_dict(
                deployment,
                replicas,
            )

    @staticmethod
    def _delete_replica_group(claim: ReplicaOperationClaim) -> None:
        """Delete a newly-created plan only while its exact create fence is owned."""
        with get_session() as session:
            parent_owned = (
                select(DeploymentDB.id)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                )
                .exists()
            )
            session.exec(
                delete(DeploymentReplicaDB).where(
                    DeploymentReplicaDB.deployment_id == claim.deployment_id,
                    parent_owned,
                )
            )
            result = session.exec(
                delete(DeploymentDB).where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                )
            )
            if result.rowcount != 1:
                session.rollback()
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            session.commit()

    @staticmethod
    def _mark_replica_plan_recovery_required(
        claim: ReplicaOperationClaim,
    ) -> None:
        """Retain a failed plan under an expired fence for safe recovery."""
        error_message = "replica cleanup incomplete; recovery required"
        stale_at = now_naive() - timedelta(
            seconds=REPLICA_OPERATION_LEASE_SECONDS + 1
        )
        with get_session() as session:
            parent_owned = (
                select(DeploymentDB.id)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                )
                .exists()
            )
            session.exec(
                update(DeploymentReplicaDB)
                .where(
                    DeploymentReplicaDB.deployment_id == claim.deployment_id,
                    parent_owned,
                )
                .values(
                    status="failed",
                    health_status="UNHEALTHY",
                    error_message=error_message,
                    stopped_at=now_naive(),
                    updated_at=now_naive(),
                )
            )
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                )
                .values(
                    status="failed",
                    health_status="UNHEALTHY",
                    error_message=error_message,
                    replica_operation_heartbeat_at=stale_at,
                    updated_at=now_naive(),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            session.commit()

    def _persist_replica_plan(
        self,
        *,
        deployment_id: str,
        model: Dict[str, Any],
        deployment_name: str | None,
        replica_count: int,
        gpu_memory_utilization: float,
        config: Dict[str, Any],
        user_id: str | None,
        inference_framework: str,
        enable_lora: bool,
        max_loras: int,
        max_lora_rank: int,
        external_api_config_id: str | None,
        plan: DeploymentPlan,
        auto_start: bool,
        defer_start: bool,
        gpu_memory_was_auto: bool = False,
    ) -> tuple[ReplicaOperationClaim, Dict[str, Any]]:
        first = plan.replicas[0]
        initial_status = "starting" if auto_start else "stopped"
        claim = ReplicaOperationClaim(
            deployment_id=deployment_id,
            token=str(uuid4()),
            generation=1,
            operation="create",
            replica_id=None,
        )
        claimed_at = now_naive()
        with get_session() as session:
            normalized_external_api_config_id = (
                self._normalize_external_api_config_id(
                    config,
                    external_api_config_id,
                )
            )
            self._lock_external_api_configs(
                session,
                (normalized_external_api_config_id,),
                expected_user_id=user_id,
            )
            locked_model = model_registry_service.lock_model_reference(
                session,
                model["model_id"],
            )
            trusted_model = locked_model.to_dict()
            if (
                trusted_model.get("model_name") != model.get("model_name")
                or (
                    gpu_memory_was_auto
                    and trusted_model.get("file_size") != model.get("file_size")
                )
            ):
                raise ReplicaOperationBusyError(
                    "registry model changed while planning deployment; retry"
                )
            persisted_config = dict(config)
            persisted_config[REPLICA_SCHEMA_VERSION_KEY] = 1
            persisted_config[REGISTRY_ARTIFACT_SIGNATURE_CONFIG_KEY] = (
                self._registry_artifact_signature(trusted_model)
            )
            if defer_start:
                persisted_config[DEFERRED_AUTO_START_CONFIG_KEY] = True
            else:
                persisted_config.pop(DEFERRED_AUTO_START_CONFIG_KEY, None)
            deployment = DeploymentDB(
                deployment_id=deployment_id,
                model_id=trusted_model["model_id"],
                model_uid=f"{trusted_model['model_name']}-{deployment_id[:8]}",
                deployment_name=deployment_name
                or f"{trusted_model['model_name']}-deployment",
                xinference_endpoint=first.endpoint,
                replica=replica_count,
                gpu_memory_utilization=gpu_memory_utilization,
                deploy_mode="container",
                container_name=first.container_name,
                gpu_id=first.gpu_ids[0],
                port=first.port,
                inference_framework=inference_framework,
                enable_lora=enable_lora,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
                external_api_config_id=normalized_external_api_config_id,
                config=persisted_config,
                user_id=user_id,
                status=initial_status,
                replica_operation_token=claim.token,
                replica_operation_kind=claim.operation,
                replica_operation_generation=claim.generation,
                replica_operation_started_at=claimed_at,
                replica_operation_heartbeat_at=claimed_at,
            )
            session.add(deployment)
            for planned in plan.replicas:
                session.add(
                    DeploymentReplicaDB(
                        deployment_id=deployment_id,
                        replica_index=planned.replica_index,
                        container_name=planned.container_name,
                        endpoint=planned.endpoint,
                        port=planned.port,
                        gpu_ids=list(planned.gpu_ids),
                        status="pending" if auto_start else "stopped",
                    )
                )
            session.commit()
        self._register_claim_heartbeat(claim)
        return claim, trusted_model

    def _start_planned_replica(
        self,
        *,
        claim: ReplicaOperationClaim,
        deployment: Dict[str, Any],
        replica: Dict[str, Any],
        model: Dict[str, Any],
        launch_config: VllmLaunchConfig | SglangLaunchConfig,
        owned_containers: list[str],
    ) -> None:
        framework = deployment["inference_framework"]
        model_uid = f"{model['model_name']}-{deployment['deployment_id'][:8]}"
        model_family = self._get_xinference_model_name(model)
        common = {
            "model_path": model["model_path"],
            "served_model_name": model_uid,
            "port": replica["port"],
            "gpu_memory_utilization": deployment["gpu_memory_utilization"],
            "model_type": model["model_type"],
            "model_family": model_family,
            "enable_lora": deployment["enable_lora"],
            "max_loras": deployment["max_loras"],
            "max_lora_rank": deployment["max_lora_rank"],
            "trust_remote_code": get_settings().allow_model_remote_code,
        }
        if framework == "vllm" and isinstance(launch_config, VllmLaunchConfig):
            server_argv = build_vllm_server_argv(launch_config, **common)
            self._require_replica_operation_ownership(claim)
            success, _message, _command = docker_deployer.create_vllm_container(
                container_name=replica["container_name"],
                port=replica["port"],
                gpu_ids=tuple(replica["gpu_ids"]),
                server_argv=server_argv,
                deployment_id=deployment["deployment_id"],
                replica_id=replica["replica_id"],
            )
        elif framework == "sglang" and isinstance(
            launch_config,
            SglangLaunchConfig,
        ):
            chat_template = _sglang_qwen3_chat_template(
                model["model_type"],
                model_family,
            )
            server_argv = build_sglang_server_argv(
                launch_config,
                **common,
                chat_template=chat_template,
            )
            self._require_replica_operation_ownership(claim)
            success, _message, _command = docker_deployer.create_sglang_container(
                container_name=replica["container_name"],
                port=replica["port"],
                gpu_ids=tuple(replica["gpu_ids"]),
                server_argv=server_argv,
                deployment_id=deployment["deployment_id"],
                replica_id=replica["replica_id"],
            )
        else:
            raise ValueError("deployment launch_config framework is invalid")
        if not success:
            raise RuntimeError("replica container creation failed")
        try:
            container_id = docker_deployer.get_managed_container_id(
                replica["container_name"],
                deployment_id=deployment["deployment_id"],
                replica_id=replica["replica_id"],
            )
        except Exception:
            # A successful `docker run` can race a transient inspect failure.
            # Re-resolve the exact name + deployment + replica identity so the
            # outer rollback can remove only a label-verified container.
            try:
                container_id = docker_deployer.get_managed_container_id(
                    replica["container_name"],
                    deployment_id=deployment["deployment_id"],
                    replica_id=replica["replica_id"],
                )
            except Exception as retry_error:
                raise _ReplicaContainerIdentityUncertainError(
                    replica["replica_id"]
                ) from retry_error
            owned_containers.append(container_id)
            raise
        owned_containers.append(container_id)
        if not docker_deployer.wait_for_service(
            replica["endpoint"],
            timeout=300,
            user_id=deployment["user_id"],
            framework=framework,
        ):
            raise RuntimeError("replica readiness check failed")

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
        defer_start: bool = False,
    ) -> Dict[str, Any]:
        """Create an independently addressable vLLM/SGLang replica group."""
        if inference_framework not in ("vllm", "sglang"):
            return self._create_single_container_deployment(
                model_id=model_id,
                deployment_name=deployment_name,
                gpu_id=gpu_id,
                port=port,
                replica=replica,
                gpu_memory_utilization=gpu_memory_utilization,
                config=config,
                user_id=user_id,
                auto_start=auto_start and not defer_start,
                inference_framework=inference_framework,
                enable_lora=enable_lora,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
                external_api_config_id=external_api_config_id,
            )
        if port is not None:
            raise ValueError("explicit replica ports are not supported")
        model = model_registry_service.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        sanitized_config = _sanitize_deployment_config(config) or {}
        launch_payload = sanitized_config.get("launch_config")
        if launch_payload is None:
            if replica != 1:
                raise ValueError("multi-replica container deployment requires launch_config")
            selected_gpu = gpu_id
            if selected_gpu is None:
                selected_gpu = docker_deployer.select_gpu(min_free_mb=4000)
            launch_payload = {
                "framework": inference_framework,
                "gpu_pool": [selected_gpu],
                "dtype": sanitized_config.get("dtype") or "auto",
            }
        launch_config = parse_launch_config(launch_payload)
        if launch_config.framework != inference_framework:
            raise ValueError("launch_config framework does not match deployment")
        sanitized_config["launch_config"] = launch_config.model_dump(mode="json")
        gpu_memory_was_auto = gpu_memory_utilization is None
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self._calculate_gpu_memory_utilization(
                model.get("file_size")
            )

        gpu_info = docker_deployer.get_gpu_memory_usage()
        gpu_inventory = list(gpu_info)
        if not gpu_inventory:
            raise ValueError("trusted GPU inventory is unavailable")
        deployment_id = str(uuid4())
        base_container_name = self._build_managed_container_name(
            deployment_id,
            model["model_name"],
            inference_framework,
        )
        reservation_owner = uuid4().hex
        plan = plan_deployment_replicas(
            base_container_name=base_container_name,
            replica_count=replica,
            launch_config=launch_config,
            gpu_inventory=gpu_inventory,
            port_allocator=docker_deployer,
            owner_token=reservation_owner,
        )
        plan_persisted = False
        creation_claim: ReplicaOperationClaim | None = None
        created_containers: list[str] = []
        try:
            if any(
                docker_deployer.container_exists_authoritative(
                    item.container_name
                )
                for item in plan.replicas
            ):
                raise RuntimeError("foreign container collision")
            creation_claim, model = self._persist_replica_plan(
                deployment_id=deployment_id,
                model=model,
                deployment_name=deployment_name,
                replica_count=replica,
                gpu_memory_utilization=gpu_memory_utilization,
                config=sanitized_config,
                user_id=user_id,
                inference_framework=inference_framework,
                enable_lora=enable_lora,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
                external_api_config_id=external_api_config_id,
                plan=plan,
                auto_start=auto_start,
                defer_start=defer_start,
                gpu_memory_was_auto=gpu_memory_was_auto,
            )
            plan_persisted = True
            assert creation_claim is not None
            if not auto_start or defer_start:
                for item in plan.replicas:
                    if not docker_deployer.confirm_port(
                        item.port,
                        owner_token=reservation_owner,
                    ):
                        raise RuntimeError("replica port confirmation failed")
                result = self._load_replica_group(deployment_id)
                if not self._release_replica_operation(creation_claim):
                    raise ReplicaOperationLostError(
                        "deployment replica operation ownership was lost"
                    )
                creation_claim = None
                return result

            group = self._load_replica_group(deployment_id)
            for child in group["replica_instances"]:
                if docker_deployer.container_exists_authoritative(
                    child["container_name"]
                ):
                    raise RuntimeError("foreign container collision")
                self._write_replica_state(
                    creation_claim,
                    child["replica_id"],
                    status="starting",
                    health_status="UNKNOWN",
                )
                self._start_planned_replica(
                    claim=creation_claim,
                    deployment=group,
                    replica=child,
                    model=model,
                    launch_config=launch_config,
                    owned_containers=created_containers,
                )
                self._write_replica_state(
                    creation_claim,
                    child["replica_id"],
                    status="running",
                    health_status="HEALTHY",
                )
            self._refresh_group_status(deployment_id, creation_claim)
            for item in plan.replicas:
                if not docker_deployer.confirm_port(
                    item.port,
                    owner_token=reservation_owner,
                ):
                    raise RuntimeError("replica port confirmation failed")
            result = self._load_replica_group(deployment_id)
            if not self._release_replica_operation(creation_claim):
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            creation_claim = None
            if replica == 1:
                try:
                    self.create_config_for_deployment(
                        deployment_id,
                        deployment_replica_id=result["replica_instances"][0][
                            "replica_id"
                        ],
                        user_id=user_id,
                    )
                except Exception as config_error:
                    logger.warning(
                        "Failed to auto-create replica config for deployment %s: %s",
                        deployment_id,
                        config_error,
                    )
            return result
        except Exception as exc:
            if isinstance(exc, _ReplicaContainerIdentityUncertainError):
                if creation_claim is None:
                    raise ReplicaOperationLostError(
                        "deployment replica operation ownership was lost"
                    ) from None
                self._write_replica_state(
                    creation_claim,
                    exc.replica_id,
                    status="failed",
                    health_status="UNHEALTHY",
                    error_message=str(exc),
                )
                self._refresh_group_status(deployment_id, creation_claim)
                for item in plan.replicas:
                    docker_deployer.confirm_port(
                        item.port,
                        owner_token=reservation_owner,
                    )
                if not self._release_replica_operation(creation_claim):
                    raise ReplicaOperationLostError(
                        "deployment replica operation ownership was lost"
                    ) from None
                creation_claim = None
                raise RuntimeError(str(exc)) from None
            cleanup_failed = False
            for container_id in reversed(created_containers):
                try:
                    if creation_claim is None:
                        raise ReplicaOperationLostError(
                            "deployment replica operation ownership was lost"
                        )
                    self._require_replica_operation_ownership(creation_claim)
                    if not docker_deployer.remove_container_identity(container_id):
                        cleanup_failed = True
                except ReplicaOperationLostError:
                    for item in plan.replicas:
                        docker_deployer.confirm_port(
                            item.port,
                            owner_token=reservation_owner,
                        )
                    raise
                except Exception:
                    cleanup_failed = True
            if cleanup_failed:
                for item in plan.replicas:
                    docker_deployer.confirm_port(
                        item.port,
                        owner_token=reservation_owner,
                    )
                if creation_claim is None:
                    raise ReplicaOperationLostError(
                        "deployment replica operation ownership was lost"
                    ) from None
                self._stop_claim_heartbeat(creation_claim)
                self._mark_replica_plan_recovery_required(creation_claim)
                raise RuntimeError(
                    "replica deployment failed; owned cleanup incomplete"
                ) from None
            if plan_persisted:
                try:
                    self._delete_replica_group(creation_claim)
                except ReplicaOperationLostError:
                    for item in plan.replicas:
                        docker_deployer.confirm_port(
                            item.port,
                            owner_token=reservation_owner,
                        )
                    raise
            for item in reversed(plan.replicas):
                docker_deployer.release_port(
                    item.port,
                    owner_token=reservation_owner,
                )
            if isinstance(
                exc,
                (ValueError, ReplicaOperationBusyError, ReplicaOperationLostError),
            ):
                raise
            raise RuntimeError("replica deployment failed") from None
        finally:
            if creation_claim is not None:
                self._stop_claim_heartbeat(creation_claim)

    @staticmethod
    def derive_group_status(replicas: List[Dict[str, Any]]) -> str:
        """Derive the parent compatibility status from canonical child state."""
        if not replicas:
            raise ValueError("deployment replica group is empty")
        if all(
            item.get("status") == "running"
            and item.get("health_status") == "HEALTHY"
            for item in replicas
        ):
            return "running"
        if all(item.get("status") == "stopped" for item in replicas):
            return "stopped"
        if all(item.get("status") == "failed" for item in replicas):
            return "failed"
        return "degraded"

    @staticmethod
    def _require_replica_access(deployment: DeploymentDB, user_id: str | None) -> None:
        if user_id is not None and deployment.user_id != user_id:
            raise DeploymentReplicaNotFoundError("deployment access denied")

    @staticmethod
    def _expected_replica_container_name(
        deployment: DeploymentDB,
        replica: DeploymentReplicaDB,
    ) -> str:
        if not deployment.container_name:
            raise ValueError("deployment replica container name is invalid")
        expected = (
            deployment.container_name
            if replica.replica_index == 0
            else f"{deployment.container_name}-r{replica.replica_index}"
        )
        if replica.container_name != expected:
            raise ValueError("deployment replica container name is invalid")
        return expected

    def list_replicas(
        self,
        deployment_id: str,
        user_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """List ordered replicas only after binding parent ownership."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise DeploymentReplicaNotFoundError("deployment not found")
            self._require_replica_access(deployment, user_id)
            replicas = session.exec(
                select(DeploymentReplicaDB)
                .where(DeploymentReplicaDB.deployment_id == deployment_id)
                .order_by(DeploymentReplicaDB.replica_index)
            ).all()
            return [replica.to_dict() for replica in replicas]

    def get_replica(
        self,
        deployment_id: str,
        replica_id: str,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """Get a child only through its owned parent and exact composite key."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise DeploymentReplicaNotFoundError("deployment not found")
            self._require_replica_access(deployment, user_id)
            replica = session.exec(
                select(DeploymentReplicaDB)
                .where(DeploymentReplicaDB.deployment_id == deployment_id)
                .where(DeploymentReplicaDB.replica_id == replica_id)
            ).first()
            if replica is None:
                raise DeploymentReplicaNotFoundError(
                    "deployment replica not found"
                )
            self._expected_replica_container_name(deployment, replica)
            return replica.to_dict()

    def resolve_replica_selection(
        self,
        deployment_id: str,
        deployment_replica_id: str | None,
        *,
        user_id: str | None,
        require_healthy: bool,
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
        """Resolve a child from trusted parent state without replica-0 fallback."""
        deployment = self.get_deployment(deployment_id)
        if deployment is None:
            raise ValueError("deployment not found")
        if user_id is not None and deployment.get("user_id") != user_id:
            raise ValueError("deployment access denied")
        replicas = list(deployment.get("replica_instances") or [])
        config = deployment.get("config") or {}
        uses_replica_lifecycle = (
            deployment.get("inference_framework") in ("vllm", "sglang")
            and isinstance(config, dict)
            and type(config.get(REPLICA_SCHEMA_VERSION_KEY)) is int
            and config.get(REPLICA_SCHEMA_VERSION_KEY) == 1
        )
        if uses_replica_lifecycle and not replicas:
            raise ValueError("deployment replica group is empty")
        if require_healthy and uses_replica_lifecycle:
            deployment = self._sync_replica_group_status(deployment_id)
            replicas = list(deployment.get("replica_instances") or [])
            if not replicas:
                raise ValueError("deployment replica group is empty")
        if not replicas:
            if deployment_replica_id is not None:
                raise ValueError("deployment replica not found")
            if require_healthy and deployment.get("status") != "running":
                raise ValueError("deployment is not running")
            return deployment, None
        if deployment_replica_id is None:
            if len(replicas) != 1:
                raise ValueError("replica_id is required for multi-replica deployment")
            replica = replicas[0]
        else:
            replica = next(
                (
                    item
                    for item in replicas
                    if item.get("replica_id") == deployment_replica_id
                ),
                None,
            )
            if replica is None:
                raise ValueError("deployment replica not found")
        if replica.get("deployment_id") != deployment_id:
            raise ValueError("deployment replica not found")
        if require_healthy and not (
            replica.get("status") == "running"
            and replica.get("health_status") == "HEALTHY"
        ):
            raise ValueError("deployment replica is not running and healthy")
        return deployment, replica

    @staticmethod
    def _uses_replica_lifecycle(deployment: DeploymentDB) -> bool:
        config = deployment.config or {}
        return (
            deployment.inference_framework in ("vllm", "sglang")
            and isinstance(config, dict)
            and type(config.get(REPLICA_SCHEMA_VERSION_KEY)) is int
            and config.get(REPLICA_SCHEMA_VERSION_KEY) == 1
        )

    def _claimed_legacy_deployment(
        self,
        session,
        claim: ReplicaOperationClaim,
        *,
        user_id: str | None,
        lock: bool = True,
    ) -> DeploymentDB:
        if claim.replica_id is not None:
            raise DeploymentReplicaStateConflictError(
                "legacy deployment operations require replica_id=None"
            )
        statement = select(DeploymentDB).where(
            DeploymentDB.deployment_id == claim.deployment_id,
            DeploymentDB.replica_operation_token == claim.token,
            DeploymentDB.replica_operation_generation == claim.generation,
            DeploymentDB.replica_operation_kind == claim.operation,
            DeploymentDB.replica_operation_replica_id.is_(None),
        )
        if lock:
            statement = statement.with_for_update()
        deployment = session.exec(statement).first()
        if deployment is None:
            session.rollback()
            raise ReplicaOperationLostError(
                "deployment replica operation ownership was lost"
            )
        self._require_replica_access(deployment, user_id)
        if self._uses_replica_lifecycle(deployment):
            session.rollback()
            raise DeploymentReplicaStateConflictError(
                "deployment changed to replica lifecycle during legacy operation"
            )
        return deployment

    def _legacy_container_id(self, deployment: DeploymentDB) -> str | None:
        self._require_managed_container(deployment)
        if not deployment.container_name:
            raise ValueError("legacy container name is missing")
        if not docker_deployer.container_exists_authoritative(
            deployment.container_name
        ):
            return None
        return docker_deployer.get_legacy_managed_container_id(
            deployment.container_name,
            deployment_id=deployment.deployment_id,
            replica_id=LEGACY_RUNTIME_REPLICA_ID,
        )

    def _legacy_container_id_for_identity(
        self,
        *,
        deployment_id: str,
        model_id: str,
        deploy_mode: str,
        container_name: str | None,
        inference_framework: str | None,
    ) -> str | None:
        """Resolve a legacy container through a detached immutable identity."""
        if self._normalize_deploy_mode(deploy_mode) != "container":
            raise ValueError("Container operation requires container deployment mode")
        model = model_registry_service.get_model(model_id)
        if not model:
            raise ValueError(
                "Cannot verify managed container name because the model is missing"
            )
        expected = self._build_managed_container_name(
            deployment_id,
            model["model_name"],
            inference_framework,
        )
        if container_name != expected:
            raise ValueError(
                "Deployment does not reference its service-managed container name"
            )
        if not container_name:
            raise ValueError("legacy container name is missing")
        if not docker_deployer.container_exists_authoritative(
            container_name
        ):
            return None
        return docker_deployer.get_legacy_managed_container_id(
            container_name,
            deployment_id=deployment_id,
            replica_id=LEGACY_RUNTIME_REPLICA_ID,
        )

    def _observe_legacy_runtime_status(
        self,
        deployment: DeploymentDB,
    ) -> str | None:
        """Observe one legacy runtime without mutating it or trusting DB status."""
        if self._is_unmanaged_binding(deployment):
            return None
        if self._normalize_deploy_mode(deployment.deploy_mode) == "container":
            container_id = self._legacy_container_id(deployment)
            if container_id is None:
                return "stopped"
            return (
                "running"
                if docker_deployer.container_running_identity(container_id)
                else "stopped"
            )
        if not deployment.model_uid:
            return "stopped"
        client = self._get_xinference_client(
            deployment.xinference_endpoint,
            user_id=deployment.user_id,
        )
        return "running" if client.get_model(deployment.model_uid) else "stopped"

    @staticmethod
    def _legacy_adapter_recovery_signature(
        adapter: LoadedAdapterDB,
    ) -> tuple[str, str, str, str, str | None]:
        return (
            adapter.adapter_id,
            adapter.adapter_name,
            adapter.adapter_path,
            adapter.status,
            adapter.source_model_id,
        )

    @staticmethod
    def _runtime_reset_adapter_recovery_signature(
        adapter: LoadedAdapterDB,
    ) -> tuple[object, ...]:
        return (
            adapter.adapter_id,
            adapter.deployment_id,
            adapter.deployment_replica_id,
            adapter.adapter_name,
            adapter.adapter_path,
            adapter.source_task_id,
            adapter.source_model_id,
            adapter.status,
            adapter.error_message,
            adapter.user_id,
            adapter.loaded_at,
            adapter.unloaded_at,
        )

    @staticmethod
    def _runtime_adapter_identity(item: object) -> tuple[str, str | None]:
        if not isinstance(item, dict):
            return "", None
        name = str(item.get("name") or item.get("id") or "")
        raw_path = (
            item.get("path")
            or item.get("adapter_path")
            or item.get("lora_path")
            or item.get("root")
        )
        path = None
        if isinstance(raw_path, str) and raw_path.strip():
            path = posixpath.normpath(raw_path.strip().replace("\\", "/"))
        return name, path

    def _list_runtime_adapters_for_endpoint(
        self,
        *,
        framework: str,
        endpoint: str,
        user_id: str | None,
    ) -> list[dict[str, Any]]:
        if framework == "vllm":
            from .vllm_client import VLLMClient

            client = VLLMClient(
                endpoint,
                user_id=user_id,
            )
        elif framework == "sglang":
            from .sglang_client import SGLangClient

            client = SGLangClient(
                endpoint,
                user_id=user_id,
            )
        else:
            raise RuntimeError(
                "adapter recovery requires a vLLM or SGLang runtime"
            )
        adapters = client.list_lora_adapters()
        if not isinstance(adapters, list):
            raise RuntimeError("adapter runtime list returned an invalid response")
        return adapters

    def _list_legacy_runtime_adapters(
        self,
        deployment: DeploymentDB,
    ) -> list[dict[str, Any]]:
        return self._list_runtime_adapters_for_endpoint(
            framework=deployment.inference_framework or "xinference",
            endpoint=deployment.xinference_endpoint,
            user_id=deployment.user_id,
        )

    def _runtime_reset_adapter_ids_absent_from_observation(
        self,
        adapters: list[LoadedAdapterDB],
        *,
        runtime_adapters: list[dict[str, Any]],
    ) -> set[str]:
        runtime_identities = [
            self._runtime_adapter_identity(item) for item in runtime_adapters
        ]
        absent: set[str] = set()
        for adapter in adapters:
            matching_paths = [
                path
                for name, path in runtime_identities
                if name == adapter.adapter_name
            ]
            if not matching_paths:
                absent.add(adapter.adapter_id)
                continue
            expected_path = posixpath.normpath(
                adapter.adapter_path.strip().replace("\\", "/")
            )
            if expected_path not in matching_paths:
                raise RuntimeError(
                    "adapter runtime identity could not be verified during recovery"
                )
        return absent

    def _reconcile_stale_legacy_adapters(
        self,
        adapters: list[LoadedAdapterDB],
        *,
        operation: str | None,
        runtime_adapters: list[dict[str, Any]],
    ) -> dict[str, str]:
        runtime_identities = [
            self._runtime_adapter_identity(item) for item in runtime_adapters
        ]
        updates: dict[str, str] = {}
        for adapter in adapters:
            matching_paths = [
                path
                for name, path in runtime_identities
                if name == adapter.adapter_name
            ]
            if not matching_paths:
                updates[adapter.adapter_id] = (
                    "failed" if operation == "load_adapter" else "unloaded"
                )
                continue
            expected_path = posixpath.normpath(
                adapter.adapter_path.strip().replace("\\", "/")
            )
            if expected_path not in matching_paths:
                raise RuntimeError(
                    "legacy adapter runtime identity could not be verified"
                )
            updates[adapter.adapter_id] = "loaded"
        return updates

    def _terminate_legacy_shared_runtime(
        self,
        claim: ReplicaOperationClaim,
        deployment: DeploymentDB,
        *,
        client=None,
    ) -> None:
        """Terminate a shared model, reconciling an uncertain response exactly."""
        if not deployment.model_uid:
            return
        if client is None:
            client = self._get_xinference_client(
                deployment.xinference_endpoint,
                user_id=deployment.user_id,
            )
        self._require_replica_operation_ownership(claim)
        try:
            client.terminate_model(deployment.model_uid)
            return
        except ReplicaOperationLostError:
            raise
        except Exception as terminate_error:
            self._require_replica_operation_ownership(claim)
            try:
                current = client.get_model(deployment.model_uid)
            except Exception as observe_error:
                raise RuntimeError(
                    "shared runtime state could not be verified after terminate "
                    "failure"
                ) from observe_error
            if current is not None:
                raise RuntimeError(
                    "shared runtime remains present after terminate failure"
                ) from terminate_error
            logger.info(
                "Shared runtime %s is already absent after terminate failure",
                deployment.model_uid,
            )

    def _claim_replica_operation(
        self,
        deployment_id: str,
        *,
        operation: str,
        replica_id: str | None,
        user_id: str | None,
        _model_delete_token: str | None = None,
    ) -> ReplicaOperationClaim:
        """Atomically claim one parent generation across processes and sessions."""
        if operation not in {
            "start",
            "stop",
            "restart",
            "recreate",
            "delete",
            "load_adapter",
            "unload_adapter",
        }:
            raise ValueError("invalid deployment replica operation")
        claimed_at = now_naive()
        token = str(uuid4())

        def try_claim() -> int | None:
            with get_session() as session:
                fence_model_runtime = operation in {
                    "start",
                    "restart",
                    "recreate",
                    "load_adapter",
                }
                if fence_model_runtime or _model_delete_token is not None:
                    lock_model_artifact_membership(session)
                    candidate = session.exec(
                        select(DeploymentDB).where(
                            DeploymentDB.deployment_id == deployment_id
                        )
                    ).first()
                    if candidate is None:
                        raise DeploymentReplicaNotFoundError(
                            "deployment not found"
                        )
                    model_registry_service.lock_model_reference(
                        session,
                        candidate.model_id,
                        membership_gate_locked=True,
                        delete_owner_token=_model_delete_token,
                    )
                deployment = session.exec(
                    select(DeploymentDB).where(
                        DeploymentDB.deployment_id == deployment_id
                    )
                ).first()
                if deployment is None:
                    raise DeploymentReplicaNotFoundError("deployment not found")
                self._require_replica_access(deployment, user_id)
                uses_replica_lifecycle = self._uses_replica_lifecycle(deployment)
                if not uses_replica_lifecycle and replica_id is not None:
                    raise DeploymentReplicaStateConflictError(
                        "legacy deployment operations require replica_id=None"
                    )
                if uses_replica_lifecycle and replica_id is not None:
                    replica = session.exec(
                        select(DeploymentReplicaDB).where(
                            DeploymentReplicaDB.deployment_id == deployment_id,
                            DeploymentReplicaDB.replica_id == replica_id,
                        )
                    ).first()
                    if replica is None:
                        raise DeploymentReplicaNotFoundError(
                            "deployment replica not found"
                        )
                    self._expected_replica_container_name(deployment, replica)

                statement = update(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id,
                    DeploymentDB.replica_operation_token.is_(None),
                )
                if user_id is not None:
                    statement = statement.where(DeploymentDB.user_id == user_id)
                result = session.exec(
                    statement.values(
                        replica_operation_token=token,
                        replica_operation_kind=operation,
                        replica_operation_replica_id=replica_id,
                        replica_operation_generation=(
                            DeploymentDB.replica_operation_generation + 1
                        ),
                        replica_operation_started_at=claimed_at,
                        replica_operation_heartbeat_at=claimed_at,
                    )
                )
                if result.rowcount != 1:
                    session.rollback()
                    return None
                generation = session.exec(
                    select(DeploymentDB.replica_operation_generation).where(
                        DeploymentDB.deployment_id == deployment_id,
                        DeploymentDB.replica_operation_token == token,
                    )
                ).one()
                session.commit()
                return generation

        generation = try_claim()
        if generation is None and self._recover_stale_replica_operation(
            deployment_id,
            user_id=user_id,
            observed_at=claimed_at,
        ):
            generation = try_claim()
        if generation is None:
            raise ReplicaOperationBusyError(
                "deployment replica operation already in progress"
            )
        claim = ReplicaOperationClaim(
            deployment_id=deployment_id,
            token=token,
            generation=generation,
            operation=operation,
            replica_id=replica_id,
        )
        self._register_claim_heartbeat(claim)
        return claim

    @staticmethod
    def _claim_is_stale(
        deployment: DeploymentDB,
        *,
        observed_at,
    ) -> bool:
        heartbeat_at = (
            deployment.replica_operation_heartbeat_at
            or deployment.replica_operation_started_at
        )
        return bool(
            deployment.replica_operation_token
            and heartbeat_at is not None
            and heartbeat_at
            < observed_at - timedelta(seconds=REPLICA_OPERATION_LEASE_SECONDS)
        )

    def _observe_replica_runtime_state(
        self,
        deployment: DeploymentDB,
        replica: DeploymentReplicaDB,
    ) -> tuple[str, str, str | None]:
        container_name = self._expected_replica_container_name(deployment, replica)
        if not docker_deployer.container_exists_authoritative(container_name):
            return "stopped", "UNKNOWN", None
        container_id = docker_deployer.get_managed_container_id(
            container_name,
            deployment_id=deployment.deployment_id,
            replica_id=replica.replica_id,
        )
        if not docker_deployer.container_running_identity(container_id):
            return "stopped", "UNKNOWN", None
        if docker_deployer.probe_service_ready(
            replica.endpoint,
            user_id=deployment.user_id,
            framework=deployment.inference_framework,
            timeout=2.0,
        ):
            return "running", "HEALTHY", None
        return "running", "UNHEALTHY", "replica readiness probe failed"

    @staticmethod
    def _stale_lifecycle_operation_may_have_reset_runtime(
        operation: str | None,
        *,
        persisted_status: str,
        observed_status: str,
    ) -> bool:
        if operation == "stop":
            return observed_status == "stopped"
        if operation in {"start", "recreate"}:
            return persisted_status == "starting"
        if operation == "restart":
            return persisted_status == "restarting"
        return False

    def _recover_stale_replica_operation(
        self,
        deployment_id: str,
        *,
        user_id: str | None,
        observed_at=None,
    ) -> bool:
        """Fence and reconcile an expired owner only after runtime identity checks."""
        observed_at = observed_at or now_naive()
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise DeploymentReplicaNotFoundError("deployment not found")
            self._require_replica_access(deployment, user_id)
            if not self._claim_is_stale(deployment, observed_at=observed_at):
                return False
            uses_replica_lifecycle = self._uses_replica_lifecycle(deployment)
            old_token = deployment.replica_operation_token
            old_generation = deployment.replica_operation_generation
            old_heartbeat = (
                deployment.replica_operation_heartbeat_at
                or deployment.replica_operation_started_at
            )
            old_operation = deployment.replica_operation_kind
            old_replica_id = deployment.replica_operation_replica_id
            runtime_deployment = deployment.model_copy(deep=True)
            stale_adapters = (
                [
                    adapter.model_copy(deep=True)
                    for adapter in session.exec(
                        select(LoadedAdapterDB)
                        .where(
                            LoadedAdapterDB.deployment_id == deployment_id,
                            LoadedAdapterDB.deployment_replica_id.is_(None),
                            LoadedAdapterDB.status
                            == (
                                "loading"
                                if old_operation == "load_adapter"
                                else "unloading"
                            ),
                        )
                        .order_by(LoadedAdapterDB.adapter_id)
                    ).all()
                ]
                if not uses_replica_lifecycle
                and old_operation in {"load_adapter", "unload_adapter"}
                else []
            )
            runtime_reset_binding_condition = (
                LoadedAdapterDB.deployment_replica_id.is_(None)
            )
            if uses_replica_lifecycle:
                runtime_reset_binding_condition = (
                    LoadedAdapterDB.deployment_replica_id.isnot(None)
                    if old_replica_id is None
                    else LoadedAdapterDB.deployment_replica_id == old_replica_id
                )
            runtime_reset_adapters = (
                [
                    adapter.model_copy(deep=True)
                    for adapter in session.exec(
                        select(LoadedAdapterDB)
                        .where(
                            LoadedAdapterDB.deployment_id == deployment_id,
                            runtime_reset_binding_condition,
                            LoadedAdapterDB.status.in_(
                                ("loading", "loaded", "unloading")
                            ),
                        )
                        .order_by(LoadedAdapterDB.adapter_id)
                    ).all()
                ]
                if old_operation in {"start", "stop", "restart", "recreate"}
                and deployment.inference_framework in ("vllm", "sglang")
                else []
            )
            replicas = (
                [
                    replica.model_copy(deep=True)
                    for replica in session.exec(
                        select(DeploymentReplicaDB)
                        .where(DeploymentReplicaDB.deployment_id == deployment_id)
                        .order_by(DeploymentReplicaDB.replica_index)
                    ).all()
                ]
                if uses_replica_lifecycle
                else []
            )
        if uses_replica_lifecycle and not replicas and old_operation != "delete":
            return False
        try:
            if uses_replica_lifecycle:
                observed_states = [
                    (
                        replica.replica_id,
                        *self._observe_replica_runtime_state(
                            runtime_deployment,
                            replica,
                        ),
                    )
                    for replica in replicas
                ]
                observed_legacy_status = None
            else:
                observed_states = []
                observed_legacy_status = self._observe_legacy_runtime_status(
                    runtime_deployment
                )
            if stale_adapters:
                runtime_adapters = self._list_legacy_runtime_adapters(
                    runtime_deployment
                )
                stale_adapter_updates = self._reconcile_stale_legacy_adapters(
                    stale_adapters,
                    operation=old_operation,
                    runtime_adapters=runtime_adapters,
                )
            else:
                stale_adapter_updates = {}
            runtime_reset_adapter_ids: set[str] = set()
            if runtime_reset_adapters:
                if uses_replica_lifecycle:
                    observed_status_by_replica = {
                        replica_id: status
                        for replica_id, status, _health, _error in observed_states
                    }
                    for replica in replicas:
                        if (
                            old_replica_id is not None
                            and replica.replica_id != old_replica_id
                        ):
                            continue
                        observed_status = observed_status_by_replica[
                            replica.replica_id
                        ]
                        if not self._stale_lifecycle_operation_may_have_reset_runtime(
                            old_operation,
                            persisted_status=replica.status,
                            observed_status=observed_status,
                        ):
                            continue
                        bound_adapters = [
                            adapter
                            for adapter in runtime_reset_adapters
                            if adapter.deployment_replica_id == replica.replica_id
                        ]
                        if observed_status == "stopped":
                            runtime_reset_adapter_ids.update(
                                adapter.adapter_id for adapter in bound_adapters
                            )
                        elif bound_adapters:
                            runtime_reset_adapter_ids.update(
                                self._runtime_reset_adapter_ids_absent_from_observation(
                                    bound_adapters,
                                    runtime_adapters=(
                                        self._list_runtime_adapters_for_endpoint(
                                            framework=(
                                                runtime_deployment.inference_framework
                                            ),
                                            endpoint=replica.endpoint,
                                            user_id=runtime_deployment.user_id,
                                        )
                                    ),
                                )
                            )
                else:
                    observed_status = observed_legacy_status or "stopped"
                    if self._stale_lifecycle_operation_may_have_reset_runtime(
                        old_operation,
                        persisted_status=runtime_deployment.status,
                        observed_status=observed_status,
                    ):
                        if observed_status == "stopped":
                            runtime_reset_adapter_ids.update(
                                adapter.adapter_id
                                for adapter in runtime_reset_adapters
                            )
                        else:
                            runtime_reset_adapter_ids.update(
                                self._runtime_reset_adapter_ids_absent_from_observation(
                                    runtime_reset_adapters,
                                    runtime_adapters=(
                                        self._list_runtime_adapters_for_endpoint(
                                            framework=(
                                                runtime_deployment.inference_framework
                                            ),
                                            endpoint=(
                                                runtime_deployment.xinference_endpoint
                                            ),
                                            user_id=runtime_deployment.user_id,
                                        )
                                    ),
                                )
                            )
        except Exception as exc:
            logger.warning(
                "Refusing stale replica claim recovery for %s: %s",
                deployment_id,
                exc,
            )
            return False

        adapters_to_reset = [
            adapter
            for adapter in runtime_reset_adapters
            if adapter.adapter_id in runtime_reset_adapter_ids
        ]

        recovery_token = str(uuid4())
        cutoff = observed_at - timedelta(seconds=REPLICA_OPERATION_LEASE_SECONDS)
        with get_session() as session:
            stale_clock = DeploymentDB.replica_operation_heartbeat_at < cutoff
            if old_heartbeat is None:
                stale_clock = (
                    DeploymentDB.replica_operation_heartbeat_at.is_(None)
                    & (DeploymentDB.replica_operation_started_at < cutoff)
                )
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == deployment_id,
                    DeploymentDB.replica_operation_token == old_token,
                    DeploymentDB.replica_operation_generation == old_generation,
                    stale_clock,
                )
                .values(
                    replica_operation_token=recovery_token,
                    replica_operation_kind="recover",
                    replica_operation_replica_id=None,
                    replica_operation_generation=(
                        DeploymentDB.replica_operation_generation + 1
                    ),
                    replica_operation_started_at=observed_at,
                    replica_operation_heartbeat_at=observed_at,
                )
            )
            if result.rowcount != 1:
                session.rollback()
                return False
            if adapters_to_reset:
                locked_active_adapters = list(
                    session.exec(
                        select(LoadedAdapterDB)
                        .where(
                            LoadedAdapterDB.deployment_id == deployment_id,
                            LoadedAdapterDB.status.in_(
                                ("loading", "loaded", "unloading")
                            ),
                        )
                        .order_by(LoadedAdapterDB.adapter_id)
                        .with_for_update()
                    ).all()
                )
                locked_adapters_to_reset = [
                    adapter
                    for adapter in locked_active_adapters
                    if adapter.adapter_id in runtime_reset_adapter_ids
                ]
                if [
                    self._runtime_reset_adapter_recovery_signature(adapter)
                    for adapter in locked_adapters_to_reset
                ] != [
                    self._runtime_reset_adapter_recovery_signature(adapter)
                    for adapter in adapters_to_reset
                ]:
                    session.rollback()
                    return False
                session.exec(
                    update(LoadedAdapterDB)
                    .where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        LoadedAdapterDB.adapter_id.in_(
                            [adapter.adapter_id for adapter in adapters_to_reset]
                        ),
                        LoadedAdapterDB.status.in_(
                            ("loading", "loaded", "unloading")
                        ),
                    )
                    .values(
                        status="unloaded",
                        error_message=None,
                        unloaded_at=observed_at,
                    )
                )
            if not uses_replica_lifecycle:
                current = session.exec(
                    select(DeploymentDB)
                    .where(
                        DeploymentDB.deployment_id == deployment_id,
                        DeploymentDB.replica_operation_token == recovery_token,
                        DeploymentDB.replica_operation_generation
                        == old_generation + 1,
                        DeploymentDB.replica_operation_kind == "recover",
                    )
                    .with_for_update()
                ).first()
                if current is None:
                    session.rollback()
                    return False
                if stale_adapters:
                    locked_adapters = list(
                        session.exec(
                            select(LoadedAdapterDB)
                            .where(
                                LoadedAdapterDB.adapter_id.in_(
                                    [
                                        adapter.adapter_id
                                        for adapter in stale_adapters
                                    ]
                                )
                            )
                            .order_by(LoadedAdapterDB.adapter_id)
                            .with_for_update()
                        ).all()
                    )
                    if [
                        self._legacy_adapter_recovery_signature(adapter)
                        for adapter in locked_adapters
                    ] != [
                        self._legacy_adapter_recovery_signature(adapter)
                        for adapter in stale_adapters
                    ]:
                        session.rollback()
                        return False
                    for adapter in locked_adapters:
                        target_status = stale_adapter_updates[adapter.adapter_id]
                        error_message = (
                            "Adapter was absent during stale load recovery"
                            if target_status == "failed"
                            else None
                        )
                        adapter.force_status(target_status, error_message)
                        session.add(adapter)
                if observed_legacy_status is not None:
                    current.update_status(observed_legacy_status)
                    session.add(current)
            session.commit()
        recovery_claim = ReplicaOperationClaim(
            deployment_id=deployment_id,
            token=recovery_token,
            generation=old_generation + 1,
            operation="recover",
            replica_id=None,
        )
        try:
            if uses_replica_lifecycle:
                for replica_id, status, health_status, error_message in observed_states:
                    self._write_replica_state(
                        recovery_claim,
                        replica_id,
                        status=status,
                        health_status=health_status,
                        error_message=error_message,
                    )
                if observed_states:
                    self._refresh_group_status(deployment_id, recovery_claim)
        finally:
            self._release_replica_operation(recovery_claim)
        return True

    def recover_stale_replica_operations(self, *, observed_at=None) -> int:
        """Recover expired deployment claims during startup without a request."""
        observed_at = observed_at or now_naive()
        with get_session() as session:
            candidates = [
                (
                    deployment.deployment_id,
                    deployment.user_id,
                )
                for deployment in session.exec(
                    select(DeploymentDB)
                    .where(DeploymentDB.replica_operation_token.isnot(None))
                    .order_by(DeploymentDB.deployment_id)
                ).all()
                if self._claim_is_stale(
                    deployment,
                    observed_at=observed_at,
                )
            ]
        recovered = 0
        for deployment_id, owner_user_id in candidates:
            try:
                if self._recover_stale_replica_operation(
                    deployment_id,
                    user_id=owner_user_id,
                    observed_at=observed_at,
                ):
                    recovered += 1
            except Exception as exc:
                logger.warning(
                    "Failed to recover stale deployment claim %s: %s",
                    deployment_id,
                    exc,
                )
        return recovered

    def _heartbeat_replica_operation(self, claim: ReplicaOperationClaim) -> bool:
        with get_session() as session:
            replica_condition = DeploymentDB.replica_operation_replica_id.is_(None)
            if claim.replica_id is not None:
                replica_condition = (
                    DeploymentDB.replica_operation_replica_id == claim.replica_id
                )
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                    DeploymentDB.replica_operation_kind == claim.operation,
                    replica_condition,
                )
                .values(replica_operation_heartbeat_at=now_naive())
            )
            session.commit()
            return result.rowcount == 1

    def _register_claim_heartbeat(self, claim: ReplicaOperationClaim) -> None:
        stop = threading.Event()

        def maintain() -> None:
            while not stop.wait(REPLICA_OPERATION_HEARTBEAT_SECONDS):
                try:
                    if not self._heartbeat_replica_operation(claim):
                        return
                except Exception as exc:
                    logger.warning(
                        "Replica operation heartbeat failed for %s: %s",
                        claim.deployment_id,
                        exc,
                    )

        thread = threading.Thread(
            target=maintain,
            name=f"replica-heartbeat-{claim.token[:8]}",
            daemon=True,
        )
        with self._claim_heartbeat_lock:
            self._claim_heartbeats[claim.token] = (stop, thread)
        thread.start()

    def _stop_claim_heartbeat(self, claim: ReplicaOperationClaim) -> None:
        with self._claim_heartbeat_lock:
            heartbeat = self._claim_heartbeats.pop(claim.token, None)
        if heartbeat is None:
            return
        stop, thread = heartbeat
        stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _release_replica_operation(self, claim: ReplicaOperationClaim) -> bool:
        """Release only the exact token and generation originally claimed."""
        self._stop_claim_heartbeat(claim)
        with get_session() as session:
            replica_condition = DeploymentDB.replica_operation_replica_id.is_(None)
            if claim.replica_id is not None:
                replica_condition = (
                    DeploymentDB.replica_operation_replica_id == claim.replica_id
                )
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                    DeploymentDB.replica_operation_kind == claim.operation,
                    replica_condition,
                )
                .values(
                    replica_operation_token=None,
                    replica_operation_kind=None,
                    replica_operation_replica_id=None,
                    replica_operation_started_at=None,
                    replica_operation_heartbeat_at=None,
                )
            )
            session.commit()
            return result.rowcount == 1

    @staticmethod
    def _require_replica_operation_ownership(
        claim: ReplicaOperationClaim,
    ) -> None:
        """Fence and renew the exact claim immediately before a runtime mutation."""
        with get_session() as session:
            replica_condition = DeploymentDB.replica_operation_replica_id.is_(None)
            if claim.replica_id is not None:
                replica_condition = (
                    DeploymentDB.replica_operation_replica_id == claim.replica_id
                )
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                    DeploymentDB.replica_operation_kind == claim.operation,
                    replica_condition,
                )
                .values(replica_operation_heartbeat_at=now_naive())
            )
            if result.rowcount != 1:
                session.rollback()
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            session.commit()

    @staticmethod
    def _mark_adapters_unloaded_after_runtime_reset(
        claim: ReplicaOperationClaim,
        replica_id: str | None,
    ) -> None:
        """Fence the claim and reconcile adapters after a confirmed runtime reset."""

        if claim.replica_id not in (None, replica_id):
            raise ReplicaOperationLostError(
                "deployment replica operation ownership was lost"
            )
        with get_session() as session:
            replica_condition = DeploymentDB.replica_operation_replica_id.is_(None)
            if claim.replica_id is not None:
                replica_condition = (
                    DeploymentDB.replica_operation_replica_id == replica_id
                )
            reset_at = now_naive()
            parent_result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                    DeploymentDB.replica_operation_kind == claim.operation,
                    replica_condition,
                )
                .values(replica_operation_heartbeat_at=reset_at)
            )
            if parent_result.rowcount != 1:
                session.rollback()
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            adapter_binding_condition = (
                LoadedAdapterDB.deployment_replica_id.is_(None)
                if replica_id is None
                else LoadedAdapterDB.deployment_replica_id == replica_id
            )
            session.exec(
                update(LoadedAdapterDB)
                .where(
                    LoadedAdapterDB.deployment_id == claim.deployment_id,
                    adapter_binding_condition,
                    LoadedAdapterDB.status.in_(("loading", "loaded", "unloading")),
                )
                .values(
                    status="unloaded",
                    error_message=None,
                    unloaded_at=reset_at,
                )
            )
            session.commit()

    @staticmethod
    def _write_replica_state(
        claim: ReplicaOperationClaim,
        replica_id: str,
        *,
        status: str,
        health_status: str,
        error_message: str | None = None,
    ) -> None:
        with get_session() as session:
            now = now_naive()
            parent_owned = (
                select(DeploymentDB.id)
                .where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                )
                .exists()
            )
            values: Dict[str, Any] = {
                "status": status,
                "health_status": health_status,
                "error_message": error_message,
                "updated_at": now,
            }
            if status == "running":
                values["started_at"] = func.coalesce(
                    DeploymentReplicaDB.started_at,
                    now,
                )
            if status in {"stopped", "failed"}:
                values["stopped_at"] = now
            result = session.exec(
                update(DeploymentReplicaDB)
                .where(
                    DeploymentReplicaDB.replica_id == replica_id,
                    DeploymentReplicaDB.deployment_id == claim.deployment_id,
                    parent_owned,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                session.rollback()
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            session.commit()

    def _refresh_group_status(
        self,
        deployment_id: str,
        claim: ReplicaOperationClaim,
    ) -> Dict[str, Any]:
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise RuntimeError("deployment record is missing")
            replicas = list(
                session.exec(
                    select(DeploymentReplicaDB)
                    .where(DeploymentReplicaDB.deployment_id == deployment_id)
                    .order_by(DeploymentReplicaDB.replica_index)
                ).all()
            )
            status = self.derive_group_status(
                [replica.to_dict() for replica in replicas]
            )
            health_status = (
                "HEALTHY"
                if status == "running"
                else "UNHEALTHY"
                if status == "failed"
                else "UNKNOWN"
            )
            error_message = (
                None if status in {"running", "stopped"} else "replica group degraded"
            )
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                )
                .values(
                    status=status,
                    health_status=health_status,
                    error_message=error_message,
                    updated_at=now_naive(),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            session.commit()
            session.expire_all()
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).one()
            return self._deployment_to_dict_with_replicas(session, deployment)

    def _replica_runtime_context(
        self,
        deployment_id: str,
        replica_id: str,
        user_id: str | None,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Any]:
        replica = self.get_replica(deployment_id, replica_id, user_id=user_id)
        deployment = self.get_deployment(deployment_id)
        if deployment is None:
            raise DeploymentReplicaNotFoundError("deployment not found")
        model = model_registry_service.get_model(deployment["model_id"])
        if not model:
            raise DeploymentReplicaStateConflictError(
                "deployment model not found"
            )
        self._require_persisted_registry_artifact(deployment, model)
        launch_config = parse_launch_config(
            (deployment.get("config") or {}).get("launch_config")
        )
        return deployment, replica, model, launch_config

    def _cleanup_owned_replica_containers(
        self,
        claim: ReplicaOperationClaim,
        container_ids: List[str],
    ) -> bool:
        cleanup_ok = True
        for container_id in reversed(container_ids):
            try:
                self._require_replica_operation_ownership(claim)
                if not docker_deployer.remove_container_identity(container_id):
                    cleanup_ok = False
            except ReplicaOperationLostError:
                raise
            except Exception:
                cleanup_ok = False
        return cleanup_ok

    @staticmethod
    def _replica_has_exact_running_runtime(
        deployment_id: str,
        replica: Dict[str, Any],
    ) -> bool:
        """Trust a healthy DB child only after exact immutable runtime checks."""
        if not (
            replica["status"] == "running"
            and replica["health_status"] == "HEALTHY"
        ):
            return False
        if not docker_deployer.container_exists_authoritative(
            replica["container_name"]
        ):
            return False
        try:
            container_id = docker_deployer.get_managed_container_id(
                replica["container_name"],
                deployment_id=deployment_id,
                replica_id=replica["replica_id"],
            )
            return docker_deployer.container_running_identity(container_id)
        except Exception:
            return False

    def start_replica(
        self,
        deployment_id: str,
        replica_id: str,
        user_id: str | None = None,
        *,
        _claim: ReplicaOperationClaim | None = None,
    ) -> Dict[str, Any]:
        owns_claim = _claim is None
        claim = _claim or self._claim_replica_operation(
            deployment_id,
            operation="start",
            replica_id=replica_id,
            user_id=user_id,
        )
        owned_containers: List[str] = []
        try:
            if owns_claim:
                self._set_deferred_start_intent(claim, enabled=False)
            deployment, replica, model, launch_config = self._replica_runtime_context(
                deployment_id,
                replica_id,
                user_id,
            )
            if self._replica_has_exact_running_runtime(deployment_id, replica):
                if owns_claim:
                    self._refresh_group_status(deployment_id, claim)
                return self.get_replica(
                    deployment_id,
                    replica_id,
                    user_id=user_id,
                )
            self._write_replica_state(
                claim,
                replica_id,
                status="starting",
                health_status="UNKNOWN",
            )
            if docker_deployer.container_exists_authoritative(
                replica["container_name"]
            ):
                container_id = docker_deployer.get_managed_container_id(
                    replica["container_name"],
                    deployment_id=deployment_id,
                    replica_id=replica_id,
                )
                self._require_replica_operation_ownership(claim)
                success, _message = docker_deployer.restart_container_identity(
                    container_id
                )
                if not success:
                    raise RuntimeError("replica container start failed")
                self._mark_adapters_unloaded_after_runtime_reset(
                    claim,
                    replica_id,
                )
                if not docker_deployer.wait_for_service(
                    replica["endpoint"],
                    timeout=300,
                    user_id=deployment["user_id"],
                    framework=deployment["inference_framework"],
                ):
                    raise RuntimeError("replica readiness check failed")
            else:
                self._mark_adapters_unloaded_after_runtime_reset(
                    claim,
                    replica_id,
                )
                self._start_planned_replica(
                    claim=claim,
                    deployment=deployment,
                    replica=replica,
                    model=model,
                    launch_config=launch_config,
                    owned_containers=owned_containers,
                )
            self._write_replica_state(
                claim,
                replica_id,
                status="running",
                health_status="HEALTHY",
            )
            if owns_claim:
                self._refresh_group_status(deployment_id, claim)
            return self.get_replica(deployment_id, replica_id, user_id=user_id)
        except (
            ReplicaOperationLostError,
            DeploymentReplicaNotFoundError,
            DeploymentReplicaStateConflictError,
        ):
            raise
        except Exception:
            self._cleanup_owned_replica_containers(
                claim,
                owned_containers,
            )
            try:
                self._write_replica_state(
                    claim,
                    replica_id,
                    status="failed",
                    health_status="UNHEALTHY",
                    error_message="replica lifecycle operation failed",
                )
                if owns_claim:
                    self._refresh_group_status(deployment_id, claim)
            except ReplicaOperationLostError:
                raise
            raise RuntimeError("replica lifecycle operation failed") from None
        finally:
            if owns_claim and not self._release_replica_operation(claim):
                logger.error(
                    "Replica operation fence was lost before release: %s",
                    deployment_id,
                )

    def stop_replica(
        self,
        deployment_id: str,
        replica_id: str,
        user_id: str | None = None,
        *,
        _claim: ReplicaOperationClaim | None = None,
    ) -> Dict[str, Any]:
        owns_claim = _claim is None
        claim = _claim or self._claim_replica_operation(
            deployment_id,
            operation="stop",
            replica_id=replica_id,
            user_id=user_id,
        )
        try:
            if owns_claim:
                self._set_deferred_start_intent(claim, enabled=False)
            replica = self.get_replica(deployment_id, replica_id, user_id=user_id)
            if docker_deployer.container_exists_authoritative(
                replica["container_name"]
            ):
                container_id = docker_deployer.get_managed_container_id(
                    replica["container_name"],
                    deployment_id=deployment_id,
                    replica_id=replica_id,
                )
                self._require_replica_operation_ownership(claim)
                if not docker_deployer.stop_container_identity(container_id):
                    raise RuntimeError("replica container stop failed")
            self._mark_adapters_unloaded_after_runtime_reset(
                claim,
                replica_id,
            )
            self._write_replica_state(
                claim,
                replica_id,
                status="stopped",
                health_status="UNKNOWN",
            )
            if owns_claim:
                self._refresh_group_status(deployment_id, claim)
            return self.get_replica(deployment_id, replica_id, user_id=user_id)
        except (
            ReplicaOperationLostError,
            DeploymentReplicaNotFoundError,
            DeploymentReplicaStateConflictError,
        ):
            raise
        except Exception:
            try:
                self._write_replica_state(
                    claim,
                    replica_id,
                    status="failed",
                    health_status="UNHEALTHY",
                    error_message="replica lifecycle operation failed",
                )
                if owns_claim:
                    self._refresh_group_status(deployment_id, claim)
            except ReplicaOperationLostError:
                raise
            raise RuntimeError("replica lifecycle operation failed") from None
        finally:
            if owns_claim and not self._release_replica_operation(claim):
                logger.error(
                    "Replica operation fence was lost before release: %s",
                    deployment_id,
                )

    def restart_replica(
        self,
        deployment_id: str,
        replica_id: str,
        user_id: str | None = None,
        *,
        _claim: ReplicaOperationClaim | None = None,
    ) -> Dict[str, Any]:
        owns_claim = _claim is None
        claim = _claim or self._claim_replica_operation(
            deployment_id,
            operation="restart",
            replica_id=replica_id,
            user_id=user_id,
        )
        try:
            if owns_claim:
                self._set_deferred_start_intent(claim, enabled=False)
            replica = self.get_replica(deployment_id, replica_id, user_id=user_id)
            self._write_replica_state(
                claim,
                replica_id,
                status="restarting",
                health_status="UNKNOWN",
            )
            if not docker_deployer.container_exists_authoritative(
                replica["container_name"]
            ):
                raise RuntimeError("replica container is missing")
            container_id = docker_deployer.get_managed_container_id(
                replica["container_name"],
                deployment_id=deployment_id,
                replica_id=replica_id,
            )
            self._require_replica_operation_ownership(claim)
            success, _message = docker_deployer.restart_container_identity(container_id)
            if not success:
                raise RuntimeError("replica container restart failed")
            self._mark_adapters_unloaded_after_runtime_reset(
                claim,
                replica_id,
            )
            deployment = self.get_deployment(deployment_id)
            if deployment is None or not docker_deployer.wait_for_service(
                replica["endpoint"],
                timeout=300,
                user_id=deployment["user_id"],
                framework=deployment["inference_framework"],
            ):
                raise RuntimeError("replica readiness check failed")
            self._write_replica_state(
                claim,
                replica_id,
                status="running",
                health_status="HEALTHY",
            )
            if owns_claim:
                self._refresh_group_status(deployment_id, claim)
            return self.get_replica(deployment_id, replica_id, user_id=user_id)
        except (
            ReplicaOperationLostError,
            DeploymentReplicaNotFoundError,
            DeploymentReplicaStateConflictError,
        ):
            raise
        except Exception:
            try:
                self._write_replica_state(
                    claim,
                    replica_id,
                    status="failed",
                    health_status="UNHEALTHY",
                    error_message="replica lifecycle operation failed",
                )
                if owns_claim:
                    self._refresh_group_status(deployment_id, claim)
            except ReplicaOperationLostError:
                raise
            raise RuntimeError("replica lifecycle operation failed") from None
        finally:
            if owns_claim and not self._release_replica_operation(claim):
                logger.error(
                    "Replica operation fence was lost before release: %s",
                    deployment_id,
                )

    def recreate_replica(
        self,
        deployment_id: str,
        replica_id: str,
        user_id: str | None = None,
        *,
        _claim: ReplicaOperationClaim | None = None,
    ) -> Dict[str, Any]:
        owns_claim = _claim is None
        claim = _claim or self._claim_replica_operation(
            deployment_id,
            operation="recreate",
            replica_id=replica_id,
            user_id=user_id,
        )
        owned_containers: List[str] = []
        try:
            if owns_claim:
                self._set_deferred_start_intent(claim, enabled=False)
            deployment, replica, model, launch_config = self._replica_runtime_context(
                deployment_id,
                replica_id,
                user_id,
            )
            self._write_replica_state(
                claim,
                replica_id,
                status="starting",
                health_status="UNKNOWN",
            )
            if docker_deployer.container_exists_authoritative(
                replica["container_name"]
            ):
                container_id = docker_deployer.get_managed_container_id(
                    replica["container_name"],
                    deployment_id=deployment_id,
                    replica_id=replica_id,
                )
                self._require_replica_operation_ownership(claim)
                if not docker_deployer.remove_container_identity(container_id):
                    raise RuntimeError("replica container removal failed")
            self._mark_adapters_unloaded_after_runtime_reset(
                claim,
                replica_id,
            )
            self._start_planned_replica(
                claim=claim,
                deployment=deployment,
                replica=replica,
                model=model,
                launch_config=launch_config,
                owned_containers=owned_containers,
            )
            self._write_replica_state(
                claim,
                replica_id,
                status="running",
                health_status="HEALTHY",
            )
            if owns_claim:
                self._refresh_group_status(deployment_id, claim)
            return self.get_replica(deployment_id, replica_id, user_id=user_id)
        except (
            ReplicaOperationLostError,
            DeploymentReplicaNotFoundError,
            DeploymentReplicaStateConflictError,
        ):
            raise
        except Exception:
            self._cleanup_owned_replica_containers(
                claim,
                owned_containers,
            )
            try:
                self._write_replica_state(
                    claim,
                    replica_id,
                    status="failed",
                    health_status="UNHEALTHY",
                    error_message="replica lifecycle operation failed",
                )
                if owns_claim:
                    self._refresh_group_status(deployment_id, claim)
            except ReplicaOperationLostError:
                raise
            raise RuntimeError("replica lifecycle operation failed") from None
        finally:
            if owns_claim and not self._release_replica_operation(claim):
                logger.error(
                    "Replica operation fence was lost before release: %s",
                    deployment_id,
                )

    def _operate_replica_group(
        self,
        deployment_id: str,
        operation: str,
        user_id: str | None,
        *,
        _claim: ReplicaOperationClaim | None = None,
        _replica_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        owns_claim = _claim is None
        claim = _claim or self._claim_replica_operation(
            deployment_id,
            operation=operation,
            replica_id=None,
            user_id=user_id,
        )
        try:
            if owns_claim:
                self._set_deferred_start_intent(claim, enabled=False)
            replicas = self.list_replicas(deployment_id, user_id=user_id)
            if _replica_ids is not None:
                replicas = [
                    replica
                    for replica in replicas
                    if replica["replica_id"] in _replica_ids
                ]
            method = {
                "start": self.start_replica,
                "stop": self.stop_replica,
                "restart": self.restart_replica,
            }[operation]
            failed = False

            def operate(replica: Dict[str, Any]) -> None:
                method(
                    deployment_id,
                    replica["replica_id"],
                    user_id=user_id,
                    _claim=claim,
                )

            if operation == "start" and len(replicas) > 1:
                with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
                    futures = [executor.submit(operate, replica) for replica in replicas]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except (
                            ReplicaOperationLostError,
                            DeploymentReplicaStateConflictError,
                        ):
                            for pending in futures:
                                pending.cancel()
                            raise
                        except Exception:
                            failed = True
            else:
                for replica in replicas:
                    try:
                        operate(replica)
                    except (
                        ReplicaOperationLostError,
                        DeploymentReplicaStateConflictError,
                    ):
                        raise
                    except Exception:
                        failed = True
            result = self._refresh_group_status(deployment_id, claim)
            if failed:
                raise RuntimeError("replica group operation failed") from None
            return result
        finally:
            if owns_claim and not self._release_replica_operation(claim):
                logger.error(
                    "Replica group operation fence was lost before release: %s",
                    deployment_id,
                )

    def start_deferred_deployment(
        self,
        deployment_id: str,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """Start only an untouched auto-start plan, consuming it at most once."""
        claim = self._claim_replica_operation(
            deployment_id,
            operation="start",
            replica_id=None,
            user_id=user_id,
        )
        claim_released = False
        try:
            group = self._load_replica_group(deployment_id)
            config = group.get("config") or {}
            if config.get(DEFERRED_AUTO_START_CONFIG_KEY) is not True:
                return group
            target_replica_ids = {
                replica["replica_id"]
                for replica in group["replica_instances"]
                if not self._replica_has_exact_running_runtime(
                    deployment_id,
                    replica,
                )
            }
            try:
                result = self._operate_replica_group(
                    deployment_id,
                    "start",
                    user_id,
                    _claim=claim,
                    _replica_ids=target_replica_ids,
                )
            except Exception:
                self._set_deferred_start_intent(claim, enabled=False)
                raise
            replica_instances = result["replica_instances"]
            self._set_deferred_start_intent(claim, enabled=False)
            if not self._release_replica_operation(claim):
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            claim_released = True
            if len(replica_instances) == 1:
                try:
                    self.create_config_for_deployment(
                        deployment_id,
                        deployment_replica_id=replica_instances[0]["replica_id"],
                        user_id=user_id,
                    )
                except Exception as config_error:
                    logger.warning(
                        "Failed to auto-create replica config for deployment %s: %s",
                        deployment_id,
                        config_error,
                    )
            return result
        finally:
            if not claim_released and not self._release_replica_operation(claim):
                logger.error(
                    "Deferred deployment start fence was lost before release: %s",
                    deployment_id,
                )

    @staticmethod
    def _set_deferred_start_intent(
        claim: ReplicaOperationClaim,
        *,
        enabled: bool,
    ) -> None:
        """Update durable auto-start intent only while the exact claim is owned."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == claim.deployment_id,
                    DeploymentDB.replica_operation_token == claim.token,
                    DeploymentDB.replica_operation_generation == claim.generation,
                )
            ).first()
            if deployment is None:
                raise ReplicaOperationLostError(
                    "deployment replica operation ownership was lost"
                )
            config = dict(deployment.config or {})
            if enabled:
                config[DEFERRED_AUTO_START_CONFIG_KEY] = True
            else:
                config.pop(DEFERRED_AUTO_START_CONFIG_KEY, None)
            deployment.config = config
            deployment.updated_at = now_naive()
            session.add(deployment)
            session.commit()

    def list_deferred_start_deployment_ids(self) -> List[str]:
        """List durable auto-start plans that a new process must resume."""
        with get_session() as session:
            deployments = list(session.exec(select(DeploymentDB)).all())
        return sorted(
            deployment.deployment_id
            for deployment in deployments
            if self._uses_replica_lifecycle(deployment)
            and isinstance(deployment.config, dict)
            and deployment.config.get(DEFERRED_AUTO_START_CONFIG_KEY) is True
        )

    def start_deployment(
        self,
        deployment_id: str,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """Start a deployment on Xinference."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            uses_replica_lifecycle = self._uses_replica_lifecycle(deployment)
            self._require_replica_access(deployment, user_id)
        if uses_replica_lifecycle:
            return self._operate_replica_group(deployment_id, "start", user_id)

        claim = self._claim_replica_operation(
            deployment_id,
            operation="start",
            replica_id=None,
            user_id=user_id,
        )
        try:
            return self._start_legacy_deployment_claimed(
                claim,
                user_id=user_id,
            )
        finally:
            if not self._release_replica_operation(claim):
                logger.error(
                    "Legacy deployment start fence was lost before release: %s",
                    deployment_id,
                )

    def _start_legacy_deployment_claimed(
        self,
        claim: ReplicaOperationClaim,
        *,
        user_id: str | None,
    ) -> Dict[str, Any]:
        with get_session() as session:
            deployment = self._claimed_legacy_deployment(
                session,
                claim,
                user_id=user_id,
            )
            model_id = deployment.model_id
        model = model_registry_service.get_model(model_id)
        if not model:
            with get_session() as session:
                deployment = self._claimed_legacy_deployment(
                    session,
                    claim,
                    user_id=user_id,
                )
                deployment.update_status("failed", "Model not found")
                session.add(deployment)
                session.commit()
            raise ValueError(f"Model not found: {model_id}")

        with get_session() as session:
            deployment = self._claimed_legacy_deployment(
                session,
                claim,
                user_id=user_id,
            )
            if self._has_unmanaged_binding_marker(deployment) or (
                model.get("source_type") == "external_bind"
            ):
                raise ValueError(
                    "Externally bound deployments are read-only and cannot be started"
                )
            if deployment.status == "running":
                return self._deployment_to_dict(deployment)
            if self._normalize_deploy_mode(deployment.deploy_mode) == "container":
                self._require_managed_container(deployment, model)
            if deployment.deploy_mode != "container" and not deployment.container_name:
                if (deployment.inference_framework or "xinference") == "xinference":
                    deployment.container_name = get_shared_xinference_container_name()
            deployment.update_status("starting")
            model_uid = deployment.model_uid or (
                f"{model['model_name']}-{deployment.deployment_id[:8]}"
            )
            runtime_deployment = deployment.model_copy(deep=True)
            session.add(deployment)
            session.commit()

        try:
            if runtime_deployment.deploy_mode == "container":
                container_id = self._legacy_container_id(runtime_deployment)
                if container_id is not None:
                    self._require_replica_operation_ownership(claim)
                    if not docker_deployer.remove_container_identity(container_id):
                        raise RuntimeError("failed to remove prior legacy container")
                if (runtime_deployment.inference_framework or "xinference") in (
                    "vllm",
                    "sglang",
                ):
                    self._mark_adapters_unloaded_after_runtime_reset(
                        claim,
                        None,
                    )
                self._require_replica_operation_ownership(claim)
                self._start_container_deployment(
                    runtime_deployment,
                    model,
                    model_uid,
                )
            else:
                self._require_replica_operation_ownership(claim)
                self._start_external_deployment(
                    runtime_deployment,
                    model,
                    model_uid,
                )

            with get_session() as session:
                deployment = self._claimed_legacy_deployment(
                    session,
                    claim,
                    user_id=user_id,
                )
                deployment.model_uid = model_uid
                deployment.config = runtime_deployment.config
                deployment.update_status("running")
                session.add(deployment)
                session.commit()
                session.refresh(deployment)
                persisted_deployment = deployment.model_copy(deep=True)
                result = self._deployment_to_dict(deployment)
            self._create_config_for_deployment(persisted_deployment, model)
            self._sync_configs_from_deployments([persisted_deployment])
            return result
        except ReplicaOperationLostError:
            raise
        except Exception as exc:
            cleanup_error = None
            try:
                if (
                    runtime_deployment.deploy_mode == "container"
                    and runtime_deployment.container_name
                ):
                    container_id = self._legacy_container_id(runtime_deployment)
                    if container_id is not None:
                        self._require_replica_operation_ownership(claim)
                        if not docker_deployer.remove_container_identity(container_id):
                            raise RuntimeError("legacy runtime cleanup failed")
                elif model_uid:
                    cleanup_deployment = runtime_deployment.model_copy(
                        deep=True
                    )
                    cleanup_deployment.model_uid = model_uid
                    self._terminate_legacy_shared_runtime(
                        claim,
                        cleanup_deployment,
                    )
            except ReplicaOperationLostError:
                raise
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
            with get_session() as session:
                deployment = self._claimed_legacy_deployment(
                    session,
                    claim,
                    user_id=user_id,
                )
                deployment.model_uid = model_uid
                if cleanup_error is None:
                    deployment.update_status("failed", str(exc))
                else:
                    deployment.update_status(
                        "stopping",
                        f"{exc}; runtime cleanup failed: {cleanup_error}",
                    )
                session.add(deployment)
                session.commit()
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

        # Convert model path for Xinference container
        model_path = self._convert_model_path_for_xinference(model['model_path'])

        # Get the model name that Xinference recognizes
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
            model_type=xinference_model_type(model['model_type']),
            replica=deployment.replica,
            gpu_memory_utilization=deployment.gpu_memory_utilization,
            **extra_kwargs,
        )

        # Record actual config for display
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
        # Get the model name that Xinference recognizes (must be built-in name)
        xinference_model_name = self._get_xinference_model_name(model)

        # Create Docker container
        success, msg, docker_cmd = docker_deployer.create_xinference_container(
            container_name=deployment.container_name,
            port=deployment.port,
            gpu_id=deployment.gpu_id,
            model_name=xinference_model_name,
            model_uid=model_uid,
            model_path=model['model_path'],
            model_type=xinference_model_type(model['model_type']),
            deployment_id=deployment.deployment_id,
            replica_id=LEGACY_RUNTIME_REPLICA_ID,
        )
        config = deployment.config or {}
        deployment.config = {
            **config,
            "docker_cmd": docker_cmd,
            "dtype": config.get("dtype") or "bfloat16",
        }

        if not success:
            raise RuntimeError(f"Failed to create Xinference container: {msg}")

        # Wait for Xinference to be ready
        if not docker_deployer.wait_for_xinference(
            deployment.xinference_endpoint,
            timeout=180,
            user_id=deployment.user_id,
        ):
            logs = docker_deployer.get_container_logs(deployment.container_name, tail=50)
            raise RuntimeError(f"Xinference service did not start. Logs:\n{logs}")

        # Wait for model to be loaded
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
                "gpu_pool": [deployment.gpu_id],
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
            deployment_id=deployment.deployment_id,
            replica_id=LEGACY_RUNTIME_REPLICA_ID,
        )
        deployment.config = {
            **config,
            "docker_cmd": docker_cmd,
            "dtype": launch_config.dtype,
            "enforce_eager": launch_config.enforce_eager,
        }

        if not success:
            raise RuntimeError(f"Failed to create vLLM container: {msg}")

        # Wait for vLLM to be ready（要求 /v1/models 含已加载模型，而非仅 /health 200）
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
        # Use base model name for trained models so SGLang can identify model type from path
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
                "gpu_pool": [deployment.gpu_id],
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
            deployment_id=deployment.deployment_id,
            replica_id=LEGACY_RUNTIME_REPLICA_ID,
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

        # Wait for SGLang to be ready（要求 /v1/models 含已加载模型）
        if not docker_deployer.wait_for_service(
            deployment.xinference_endpoint,
            timeout=300,
            user_id=deployment.user_id,
            framework="sglang",
        ):
            logs = docker_deployer.get_container_logs(deployment.container_name, tail=50)
            raise RuntimeError(f"SGLang service did not start. Logs:\n{logs}")

    def stop_deployment(
        self,
        deployment_id: str,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """Stop a running deployment."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            self._require_replica_access(deployment, user_id)
            uses_replica_lifecycle = self._uses_replica_lifecycle(deployment)
        if uses_replica_lifecycle:
            return self._operate_replica_group(deployment_id, "stop", user_id)

        claim = self._claim_replica_operation(
            deployment_id,
            operation="stop",
            replica_id=None,
            user_id=user_id,
        )
        try:
            return self._stop_legacy_deployment_claimed(
                claim,
                user_id=user_id,
            )
        finally:
            if not self._release_replica_operation(claim):
                logger.error(
                    "Legacy deployment stop fence was lost before release: %s",
                    deployment_id,
                )

    def _stop_legacy_deployment_claimed(
        self,
        claim: ReplicaOperationClaim,
        *,
        user_id: str | None,
    ) -> Dict[str, Any]:
        with get_session() as session:
            deployment = self._claimed_legacy_deployment(
                session,
                claim,
                user_id=user_id,
            )
            if self._is_unmanaged_binding(deployment):
                raise ValueError(
                    "Externally bound deployments are read-only and cannot be stopped"
                )
            if self._normalize_deploy_mode(deployment.deploy_mode) == "container":
                self._require_managed_container(deployment)
            if deployment.status == "stopped":
                return self._deployment_to_dict(deployment)
            if deployment.status in ("pending", "failed"):
                if deployment.status == "failed":
                    deployment.update_status("pending")
                deployment.update_status("stopped")
                session.add(deployment)
                result = self._deployment_to_dict(deployment)
                session.commit()
                return result
            deployment.update_status("stopping")
            runtime_deployment = deployment.model_copy(deep=True)
            session.add(deployment)
            session.commit()

        try:
            if (
                runtime_deployment.deploy_mode == "container"
                and runtime_deployment.container_name
            ):
                container_id = self._legacy_container_id(runtime_deployment)
                if container_id is not None:
                    self._require_replica_operation_ownership(claim)
                    if not docker_deployer.stop_container_identity(container_id):
                        raise RuntimeError(
                            f"Failed to stop container {runtime_deployment.container_name}"
                        )
                if (runtime_deployment.inference_framework or "xinference") in (
                    "vllm",
                    "sglang",
                ):
                    self._mark_adapters_unloaded_after_runtime_reset(
                        claim,
                        None,
                    )
            elif runtime_deployment.model_uid:
                self._terminate_legacy_shared_runtime(
                    claim,
                    runtime_deployment,
                )

            with get_session() as session:
                deployment = self._claimed_legacy_deployment(
                    session,
                    claim,
                    user_id=user_id,
                )
                deployment.update_status("stopped")
                session.add(deployment)
                result = self._deployment_to_dict(deployment)
                session.commit()
            return result
        except ReplicaOperationLostError:
            raise
        except Exception as exc:
            with get_session() as session:
                deployment = self._claimed_legacy_deployment(
                    session,
                    claim,
                    user_id=user_id,
                )
                deployment.update_status("stopping", str(exc))
                session.add(deployment)
                session.commit()
            raise RuntimeError(
                f"Failed to stop deployment {claim.deployment_id}: {exc}"
            ) from exc

    def restart_deployment(
        self,
        deployment_id: str,
        mode: str = "auto",
        reset_gpu: bool = False,
        user_id: str | None = None,
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
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            self._require_replica_access(deployment, user_id)
            uses_replica_lifecycle = self._uses_replica_lifecycle(deployment)
        if uses_replica_lifecycle:
            if mode not in ("auto", "container"):
                raise ValueError(
                    "replica groups only support container restart mode"
                )
            return self._operate_replica_group(
                deployment_id,
                "restart",
                user_id,
            )

        claim = self._claim_replica_operation(
            deployment_id,
            operation="restart",
            replica_id=None,
            user_id=user_id,
        )
        try:
            return self._restart_legacy_deployment_claimed(
                claim,
                mode=mode,
                reset_gpu=reset_gpu,
                user_id=user_id,
            )
        finally:
            if not self._release_replica_operation(claim):
                logger.error(
                    "Legacy deployment restart fence was lost before release: %s",
                    deployment_id,
                )

    def _restart_legacy_deployment_claimed(
        self,
        claim: ReplicaOperationClaim,
        *,
        mode: str,
        reset_gpu: bool,
        user_id: str | None,
    ) -> Dict[str, Any]:
        with get_session() as session:
            deployment = self._claimed_legacy_deployment(
                session,
                claim,
                user_id=user_id,
            )
            if self._is_unmanaged_binding(deployment):
                raise ValueError(
                    "Externally bound deployments are read-only and cannot be restarted"
                )
            framework = deployment.inference_framework or "xinference"
            deploy_mode = self._normalize_deploy_mode(deployment.deploy_mode)
            if deploy_mode == "container":
                self._require_managed_container(deployment)
            if mode not in ("auto", "model", "container"):
                raise ValueError(
                    f"Invalid restart mode: {mode}. Use 'auto', 'model', or 'container'"
                )
            if mode == "model" and framework != "xinference":
                raise ValueError(
                    f"Mode 'model' is only supported for Xinference, not {framework}"
                )
            if mode == "container" and deploy_mode != "container":
                raise ValueError("Mode 'container' requires container deployment mode")
            actual_mode = (
                "container"
                if mode == "auto" and framework in ("vllm", "sglang")
                else ("model" if mode == "auto" else mode)
            )
            fallback_start = (
                actual_mode == "model"
                and not deployment.model_uid
                and deployment.status in ("failed", "stopped", "pending")
            )
            if actual_mode == "model" and not deployment.model_uid and not fallback_start:
                raise ValueError("Model reload requires model_uid")
            if fallback_start:
                session.commit()
            else:
                deployment.update_status("restarting")
                runtime_deployment = deployment.model_copy(deep=True)
                session.add(deployment)
                session.commit()

        if fallback_start:
            return self._start_legacy_deployment_claimed(
                claim,
                user_id=user_id,
            )

        try:
            if actual_mode == "container":
                container_id = self._legacy_container_id(runtime_deployment)
                if container_id is None:
                    raise RuntimeError(
                        f"Container not found: {runtime_deployment.container_name}"
                    )
                if reset_gpu and runtime_deployment.gpu_id is not None:
                    self._require_replica_operation_ownership(claim)
                    if not docker_deployer.stop_container_identity(container_id):
                        raise RuntimeError("Container stop before GPU reset failed")
                    if framework in ("vllm", "sglang"):
                        self._mark_adapters_unloaded_after_runtime_reset(
                            claim,
                            None,
                        )
                    self._require_replica_operation_ownership(claim)
                    reset_ok, reset_message = docker_deployer.reset_gpu(
                        runtime_deployment.gpu_id
                    )
                    if not reset_ok:
                        logger.warning("GPU reset failed: %s", reset_message)
                self._require_replica_operation_ownership(claim)
                success, message = docker_deployer.restart_container_identity(
                    container_id
                )
                if not success:
                    raise RuntimeError(f"Container restart failed: {message}")
                if framework in ("vllm", "sglang"):
                    self._mark_adapters_unloaded_after_runtime_reset(
                        claim,
                        None,
                    )
                if not docker_deployer.wait_for_service(
                    runtime_deployment.xinference_endpoint,
                    timeout=300,
                    user_id=runtime_deployment.user_id,
                    framework=framework,
                ):
                    raise RuntimeError(f"{framework} service did not restart in time")
            else:
                model = model_registry_service.get_model(runtime_deployment.model_id)
                if not model:
                    raise ValueError(
                        f"Model not found: {runtime_deployment.model_id}"
                    )
                client = self._get_xinference_client(
                    runtime_deployment.xinference_endpoint,
                    user_id=runtime_deployment.user_id,
                )
                self._terminate_legacy_shared_runtime(
                    claim,
                    runtime_deployment,
                    client=client,
                )
                self._require_replica_operation_ownership(claim)
                model_path = self._convert_model_path_for_xinference(
                    model["model_path"]
                )
                model_name = self._get_xinference_model_name(model)
                extra_kwargs = _trusted_model_launch_kwargs(
                    runtime_deployment.config
                )
                extra_kwargs.update(
                    xinference_model_launch_overrides(
                        model_type=model["model_type"],
                        model_family=model_name,
                    )
                )
                client.launch_model(
                    model_uid=runtime_deployment.model_uid,
                    model_name=model_name,
                    model_path=model_path,
                    model_type=xinference_model_type(model["model_type"]),
                    replica=runtime_deployment.replica,
                    gpu_memory_utilization=(
                        runtime_deployment.gpu_memory_utilization
                    ),
                    **extra_kwargs,
                )
                if not docker_deployer.wait_for_model(
                    runtime_deployment.xinference_endpoint,
                    runtime_deployment.model_uid,
                    timeout=300,
                    user_id=runtime_deployment.user_id,
                ):
                    raise RuntimeError("Model did not reload in time")

            with get_session() as session:
                deployment = self._claimed_legacy_deployment(
                    session,
                    claim,
                    user_id=user_id,
                )
                deployment.update_status("running")
                session.add(deployment)
                result = self._deployment_to_dict(deployment)
                session.commit()
            return result
        except ReplicaOperationLostError:
            raise
        except Exception as exc:
            with get_session() as session:
                deployment = self._claimed_legacy_deployment(
                    session,
                    claim,
                    user_id=user_id,
                )
                deployment.update_status("failed", str(exc))
                session.add(deployment)
                session.commit()
            raise

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
            model_type=xinference_model_type(model['model_type']),
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

    @staticmethod
    def _lock_deployment_delete_runtime_dependency_scope(
        session,
        claim: ReplicaOperationClaim,
        *,
        execution_snapshot,
    ) -> tuple[DeploymentDB, tuple[ModelConfigDB, ...]]:
        """Lock one deployment-delete dependency graph for the exact owner."""
        from ..storage.entities.milvus_collection_entity import (
            MilvusCollectionDB,
        )
        from ..storage.entities.model_registry_entity import ModelRegistryDB
        from ..storage.services.runtime_dependency_service import (
            RuntimeDependencyChangedError,
            RuntimeDependencyReferences,
            RuntimeDependencyUnavailableError,
            lock_active_runtime_dependency_consumers,
        )

        if claim.operation != "delete" or claim.replica_id is not None:
            raise ReplicaOperationLostError(
                "deployment deletion ownership was lost"
            )

        preliminary_deployment = session.exec(
            select(DeploymentDB).where(
                DeploymentDB.deployment_id == claim.deployment_id
            )
        ).first()
        if preliminary_deployment is None:
            raise ReplicaOperationLostError(
                "deployment deletion ownership was lost"
            )
        preliminary_configs = list(
            session.exec(
                select(ModelConfigDB)
                .where(ModelConfigDB.deployment_id == claim.deployment_id)
                .order_by(ModelConfigDB.config_id)
            ).all()
        )
        preliminary_config_signatures = {
            config.config_id: (config.registry_id, config.deployment_id)
            for config in preliminary_configs
        }
        model_ids = tuple(
            sorted(
                {
                    model_id
                    for model_id in (
                        preliminary_deployment.model_id,
                        *(
                            config.registry_id
                            for config in preliminary_configs
                        ),
                    )
                    if isinstance(model_id, str) and model_id
                }
            )
        )
        if model_ids:
            locked_models = list(
                session.exec(
                    select(ModelRegistryDB)
                    .where(ModelRegistryDB.model_id.in_(model_ids))
                    .order_by(ModelRegistryDB.model_id)
                    .execution_options(populate_existing=True)
                    .with_for_update()
                ).all()
            )
            if {model.model_id for model in locked_models} != set(model_ids):
                raise RuntimeDependencyUnavailableError(
                    "Deployment model dependency is unavailable"
                )

        deployment = session.exec(
            select(DeploymentDB)
            .where(
                DeploymentDB.deployment_id == claim.deployment_id,
                DeploymentDB.replica_operation_token == claim.token,
                DeploymentDB.replica_operation_generation == claim.generation,
                DeploymentDB.replica_operation_kind == "delete",
                DeploymentDB.replica_operation_replica_id.is_(None),
            )
            .execution_options(populate_existing=True)
            .with_for_update()
        ).first()
        if deployment is None:
            raise ReplicaOperationLostError(
                "deployment deletion ownership was lost"
            )
        if deployment.model_id != preliminary_deployment.model_id:
            raise RuntimeDependencyChangedError(
                "Deployment model changed during deletion; retry"
            )

        configs = list(
            session.exec(
                select(ModelConfigDB)
                .where(ModelConfigDB.deployment_id == claim.deployment_id)
                .order_by(ModelConfigDB.config_id)
                .execution_options(populate_existing=True)
                .with_for_update()
            ).all()
        )
        config_signatures = {
            config.config_id: (config.registry_id, config.deployment_id)
            for config in configs
        }
        if config_signatures != preliminary_config_signatures:
            raise RuntimeDependencyChangedError(
                "Deployment config membership changed during deletion; retry"
            )

        references = RuntimeDependencyReferences(
            config_ids=tuple(sorted(config_signatures)),
            deployment_ids=(claim.deployment_id,),
        )
        consumers = lock_active_runtime_dependency_consumers(
            session,
            references,
            execution_snapshot=execution_snapshot,
        )
        if consumers:
            raise RuntimeDependencyUnavailableError(
                "Deployment is referenced by an active runtime task"
            )

        if references.config_ids:
            milvus_references = list(
                session.exec(
                    select(MilvusCollectionDB)
                    .where(
                        MilvusCollectionDB.embedding_config_id.in_(
                            references.config_ids
                        )
                    )
                    .order_by(MilvusCollectionDB.collection_name)
                    .execution_options(populate_existing=True)
                    .with_for_update()
                ).all()
            )
            if milvus_references:
                raise RuntimeDependencyUnavailableError(
                    "Deployment config is referenced by a Milvus collection"
                )

        return deployment, tuple(configs)

    def _delete_replica_deployment_group(
        self,
        deployment_id: str,
        *,
        force: bool,
        user_id: str | None = None,
    ) -> bool:
        """Delete owned child runtimes before deleting their bindings and parent."""
        claim = self._claim_replica_operation(
            deployment_id,
            operation="delete",
            replica_id=None,
            user_id=user_id,
        )
        deleted = False
        progress = _DeploymentDeleteProgress()
        try:
            from ..storage.services.runtime_dependency_service import (
                snapshot_runtime_executions,
            )

            execution_snapshot = snapshot_runtime_executions()
            with get_session() as session:
                deployment, configs = (
                    self._lock_deployment_delete_runtime_dependency_scope(
                        session,
                        claim,
                        execution_snapshot=execution_snapshot,
                    )
                )
                self._require_replica_access(deployment, user_id)
                from ..storage.services.external_sync_service import (
                    external_sync_service,
                )

                external_sync_service.require_deployment_unreferenced(
                    session,
                    deployment_id,
                )
                replicas = list(
                    session.exec(
                        select(DeploymentReplicaDB)
                        .where(DeploymentReplicaDB.deployment_id == deployment_id)
                        .order_by(DeploymentReplicaDB.replica_index)
                    ).all()
                )
                adapters = list(
                    session.exec(
                        select(LoadedAdapterDB).where(
                            LoadedAdapterDB.deployment_id == deployment_id
                        )
                    ).all()
                )
                if (configs or adapters) and not force:
                    raise ValueError("deployment has associated replica bindings")
                for replica in replicas:
                    self._expected_replica_container_name(deployment, replica)

            cleanup_failed = False
            for replica in replicas:
                try:
                    present = docker_deployer.container_exists_authoritative(
                        replica.container_name
                    )
                except Exception:
                    cleanup_failed = True
                    break
                if not present:
                    continue
                try:
                    container_id = docker_deployer.get_managed_container_id(
                        replica.container_name,
                        deployment_id=deployment_id,
                        replica_id=replica.replica_id,
                    )
                    self._require_replica_operation_ownership(claim)
                    progress.runtime_cleanup_started = True
                    if not docker_deployer.remove_container_identity(container_id):
                        cleanup_failed = True
                except ReplicaOperationLostError:
                    raise
                except Exception:
                    cleanup_failed = True
            if cleanup_failed:
                raise RuntimeError("replica group cleanup failed") from None

            final_execution_snapshot = snapshot_runtime_executions()
            with get_session() as session:
                deployment, configs = (
                    self._lock_deployment_delete_runtime_dependency_scope(
                        session,
                        claim,
                        execution_snapshot=final_execution_snapshot,
                    )
                )
                self._require_replica_access(deployment, user_id)
                external_sync_service.require_deployment_unreferenced(
                    session,
                    deployment_id,
                )
                adapters = list(
                    session.exec(
                        select(LoadedAdapterDB).where(
                            LoadedAdapterDB.deployment_id == deployment_id
                        )
                    ).all()
                )
                if (configs or adapters) and not force:
                    raise ValueError("deployment has associated replica bindings")
                for config in configs:
                    session.delete(config)
                for adapter in adapters:
                    session.delete(adapter)
                for replica in session.exec(
                    select(DeploymentReplicaDB).where(
                        DeploymentReplicaDB.deployment_id == deployment_id
                    )
                ).all():
                    session.delete(replica)
                session.delete(deployment)
                session.commit()
                deleted = True
            return True
        finally:
            if deleted:
                self._stop_claim_heartbeat(claim)
            elif progress.runtime_cleanup_started:
                self._stop_claim_heartbeat(claim)
            elif not self._release_replica_operation(claim):
                logger.error(
                    "Replica deletion fence was lost before release: %s",
                    deployment_id,
                )

    def delete_deployment(
        self,
        deployment_id: str,
        force: bool = False,
        user_id: str | None = None,
    ) -> bool:
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
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                return False
            self._require_replica_access(deployment, user_id)
            uses_replica_lifecycle = self._uses_replica_lifecycle(deployment)
        if uses_replica_lifecycle:
            return self._delete_replica_deployment_group(
                deployment_id,
                force=force,
                user_id=user_id,
            )

        claim = self._claim_replica_operation(
            deployment_id,
            operation="delete",
            replica_id=None,
            user_id=user_id,
        )
        deleted = False
        progress = _DeploymentDeleteProgress()
        try:
            deleted = self._delete_legacy_deployment_claimed(
                claim,
                force=force,
                user_id=user_id,
                progress=progress,
            )
            return deleted
        finally:
            if deleted:
                self._stop_claim_heartbeat(claim)
            elif progress.runtime_cleanup_started:
                self._stop_claim_heartbeat(claim)
            elif not self._release_replica_operation(claim):
                logger.error(
                    "Legacy deployment deletion fence was lost before release: %s",
                    deployment_id,
                )

    def _delete_legacy_deployment_claimed(
        self,
        claim: ReplicaOperationClaim,
        *,
        force: bool,
        user_id: str | None,
        progress: _DeploymentDeleteProgress,
    ) -> bool:
        from ..storage.services.external_sync_service import (
            external_sync_service,
        )

        from ..storage.services.runtime_dependency_service import (
            snapshot_runtime_executions,
        )

        execution_snapshot = snapshot_runtime_executions()
        with get_session() as session:
            deployment, configs = (
                self._lock_deployment_delete_runtime_dependency_scope(
                    session,
                    claim,
                    execution_snapshot=execution_snapshot,
                )
            )
            self._require_replica_access(deployment, user_id)
            if self._uses_replica_lifecycle(deployment):
                raise DeploymentReplicaStateConflictError(
                    "deployment changed to replica lifecycle during legacy operation"
                )
            external_sync_service.require_deployment_unreferenced(
                session,
                claim.deployment_id,
            )
            if configs and not force:
                raise ValueError(
                    f"Cannot delete deployment {claim.deployment_id}: "
                    f"has {len(configs)} config(s). Use force=True to delete anyway."
                )
            unmanaged_binding = self._is_unmanaged_binding(deployment)
            if (
                not unmanaged_binding
                and self._normalize_deploy_mode(deployment.deploy_mode)
                == "container"
            ):
                self._require_managed_container(deployment)
            shared_container = False
            if deployment.container_name:
                shared_container = (
                    session.exec(
                        select(DeploymentDB).where(
                            DeploymentDB.container_name == deployment.container_name,
                            DeploymentDB.deployment_id != claim.deployment_id,
                            DeploymentDB.status == "running",
                        )
                    ).first()
                    is not None
                )
            runtime_deployment = deployment.model_copy(deep=True)
            session.commit()

        if (
            not unmanaged_binding
            and runtime_deployment.deploy_mode == "container"
            and runtime_deployment.container_name
            and not shared_container
        ):
            container_id = self._legacy_container_id(runtime_deployment)
            if container_id is not None:
                self._require_replica_operation_ownership(claim)
                progress.runtime_cleanup_started = True
                if not docker_deployer.remove_container_identity(container_id):
                    raise RuntimeError(
                        "Failed to remove container during deployment deletion: "
                        f"{runtime_deployment.container_name}"
                    )
        elif (
            not unmanaged_binding
            and runtime_deployment.status
            in {"running", "starting", "restarting", "stopping"}
            and runtime_deployment.model_uid
        ):
            progress.runtime_cleanup_started = True
            self._terminate_legacy_shared_runtime(
                claim,
                runtime_deployment,
            )

        final_execution_snapshot = snapshot_runtime_executions()
        with get_session() as session:
            deployment, configs = (
                self._lock_deployment_delete_runtime_dependency_scope(
                    session,
                    claim,
                    execution_snapshot=final_execution_snapshot,
                )
            )
            self._require_replica_access(deployment, user_id)
            if self._uses_replica_lifecycle(deployment):
                raise DeploymentReplicaStateConflictError(
                    "deployment changed to replica lifecycle during legacy operation"
                )
            external_sync_service.require_deployment_unreferenced(
                session,
                claim.deployment_id,
            )
            if configs and not force:
                raise ValueError(
                    f"Cannot delete deployment {claim.deployment_id}: "
                    f"has {len(configs)} config(s). Use force=True to delete anyway."
                )
            for config in configs:
                session.delete(config)
            for adapter in session.exec(
                select(LoadedAdapterDB).where(
                    LoadedAdapterDB.deployment_id == claim.deployment_id
                )
            ).all():
                session.delete(adapter)
            session.delete(deployment)
            session.commit()
        logger.info("Deleted legacy deployment %s", claim.deployment_id)
        return True

    def update_deployment_config(
        self,
        deployment_id: str,
        config: Optional[Dict[str, Any]],
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """Update deployment config.

        Args:
            deployment_id: Deployment ID
            config: New config to persist (replaces existing config)

        Returns:
            Updated deployment dict
        """
        sanitized_config = _sanitize_deployment_config(config)
        new_external_api_config_id = self._extract_external_api_config_id(
            sanitized_config
        )
        with get_session() as session:
            observed = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if observed is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            self._require_replica_access(observed, user_id)
            observed_model_id = observed.model_id
            observed_user_id = observed.user_id
            observed_external_api_config_id = (
                observed.external_api_config_id
                or self._extract_external_api_config_id(observed.config)
            )

        with get_session() as session:
            self._lock_external_api_configs(
                session,
                (
                    observed_external_api_config_id,
                    new_external_api_config_id,
                ),
                expected_user_id=user_id or observed_user_id,
            )
            model_registry_service.lock_model_reference(
                session,
                observed_model_id,
            )
            deployment = session.exec(
                select(DeploymentDB)
                .where(DeploymentDB.deployment_id == deployment_id)
                .with_for_update()
            ).first()
            if deployment is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            self._require_replica_access(deployment, user_id)
            current_external_api_config_id = (
                deployment.external_api_config_id
                or self._extract_external_api_config_id(deployment.config)
            )
            if (
                deployment.model_id != observed_model_id
                or deployment.user_id != observed_user_id
                or current_external_api_config_id
                != observed_external_api_config_id
            ):
                raise ReplicaOperationBusyError(
                    "deployment changed while acquiring external API locks"
                )

            existing_config = (
                deployment.config if isinstance(deployment.config, dict) else {}
            )
            replicas = session.exec(
                select(DeploymentReplicaDB)
                .where(DeploymentReplicaDB.deployment_id == deployment_id)
                .order_by(DeploymentReplicaDB.replica_index)
                .with_for_update()
            ).all()
            uses_replica_lifecycle = (
                deployment.inference_framework in ("vllm", "sglang")
                and (
                    bool(replicas)
                    or (
                        type(existing_config.get(REPLICA_SCHEMA_VERSION_KEY))
                        is int
                        and existing_config.get(REPLICA_SCHEMA_VERSION_KEY) == 1
                    )
                )
            )
            persisted_config = sanitized_config
            if LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY in existing_config:
                persisted_config = dict(persisted_config or {})
                persisted_config[LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY] = (
                    existing_config[LEGACY_REPLICA_SCHEMA_MARKER_BACKUP_KEY]
                )
            if uses_replica_lifecycle:
                persisted_config = dict(persisted_config or {})
                launch_config_was_provided = "launch_config" in persisted_config
                requested_launch_payload = persisted_config.get("launch_config")
                existing_launch_payload = existing_config.get("launch_config")
                existing_launch_config = None
                if existing_launch_payload is not None:
                    try:
                        existing_launch_config = parse_launch_config(
                            existing_launch_payload
                        )
                    except (TypeError, ValueError):
                        existing_launch_config = None

                if not launch_config_was_provided:
                    if existing_launch_config is None:
                        raise ValueError(
                            "existing launch_config is missing or invalid; "
                            "provide a valid launch_config to repair"
                        )
                    launch_config = existing_launch_config
                else:
                    try:
                        launch_config = parse_launch_config(requested_launch_payload)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("launch_config is invalid") from exc

                if launch_config.framework != deployment.inference_framework:
                    raise ValueError(
                        "launch_config framework does not match deployment"
                    )

                if not replicas:
                    raise DeploymentReplicaStateConflictError(
                        "replica lifecycle deployment has no persisted child plan"
                    )

                current_assignments = tuple(
                    tuple(replica.gpu_ids or []) for replica in replicas
                )
                if existing_launch_config is None:
                    default_launch_config = parse_launch_config(
                        {"framework": deployment.inference_framework}
                    )
                    derivable_repair_fields = {
                        "framework",
                        "tensor_parallel_size",
                        "pipeline_parallel_size",
                        "data_parallel_size",
                        "gpu_pool",
                        "replica_gpu_overrides",
                        "allow_gpu_reuse",
                    }
                    proposed_dump = launch_config.model_dump(mode="json")
                    default_dump = default_launch_config.model_dump(mode="json")
                    has_nonderivable_override = any(
                        proposed_dump[field] != default_dump[field]
                        for field in proposed_dump.keys() - derivable_repair_fields
                    )
                    if (
                        any(
                            len(assignment) != 1
                            for assignment in current_assignments
                        )
                        or launch_config.tensor_parallel_size != 1
                        or launch_config.pipeline_parallel_size != 1
                        or launch_config.data_parallel_size != 1
                        or has_nonderivable_override
                    ):
                        raise ValueError(
                            "launch_config topology cannot be repaired unambiguously; "
                            "recreate deployment"
                        )
                gpu_inventory = tuple(
                    dict.fromkeys(
                        gpu_id
                        for assignment in current_assignments
                        for gpu_id in assignment
                    )
                )
                try:
                    proposed_assignments = _plan_gpu_assignments(
                        launch_config=launch_config,
                        replica_count=len(replicas),
                        gpu_inventory=gpu_inventory,
                    )
                except ValueError as exc:
                    raise ValueError(
                        "launch_config topology does not match persisted child plan"
                    ) from exc
                if proposed_assignments != current_assignments:
                    raise ValueError(
                        "launch_config topology does not match persisted child plan"
                    )

                if existing_launch_config is not None:
                    topology_fields = (
                        "framework",
                        "tensor_parallel_size",
                        "pipeline_parallel_size",
                        "data_parallel_size",
                        "gpu_pool",
                        "replica_gpu_overrides",
                        "allow_gpu_reuse",
                        "expert_parallel_size",
                        "enable_expert_parallel",
                    )

                    def topology(config_value):
                        dumped = config_value.model_dump(mode="json")
                        return tuple(
                            (
                                key,
                                json.dumps(
                                    dumped.get(key),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                            for key in topology_fields
                        )

                    if topology(existing_launch_config) != topology(launch_config):
                        raise ValueError(
                            "launch_config topology is immutable; recreate deployment"
                        )

                persisted_config["launch_config"] = launch_config.model_dump(
                    mode="json"
                )
                persisted_config[REPLICA_SCHEMA_VERSION_KEY] = 1
                for operator_key in (
                    DEFERRED_AUTO_START_CONFIG_KEY,
                    REGISTRY_ARTIFACT_SIGNATURE_CONFIG_KEY,
                ):
                    if operator_key in existing_config:
                        persisted_config[operator_key] = existing_config[operator_key]
            if self._is_unmanaged_binding(deployment):
                persisted_config = self._mark_unmanaged_binding(
                    persisted_config
                )
            deployment.config = persisted_config
            deployment.external_api_config_id = (
                new_external_api_config_id
            )
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
                return self._deployment_to_dict_with_replicas(session, deployment)
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
            return [
                self._deployment_to_dict_with_replicas(session, deployment)
                for deployment in deployments
            ], total

    # ==================== Status Sync ====================

    @staticmethod
    def _write_replica_sync_state(
        deployment_id: str,
        replica_id: str,
        *,
        generation: int,
        status: str,
        health_status: str,
        error_message: str | None,
    ) -> bool:
        """Persist an observation only if no lifecycle ABA occurred."""
        with get_session() as session:
            parent_stable = (
                select(DeploymentDB.id)
                .where(
                    DeploymentDB.deployment_id == deployment_id,
                    DeploymentDB.replica_operation_token.is_(None),
                    DeploymentDB.replica_operation_generation == generation,
                )
                .exists()
            )
            now = now_naive()
            values: Dict[str, Any] = {
                "status": status,
                "health_status": health_status,
                "error_message": error_message,
                "updated_at": now,
            }
            if status == "running":
                values["started_at"] = func.coalesce(
                    DeploymentReplicaDB.started_at,
                    now,
                )
            if status in {"stopped", "failed"}:
                values["stopped_at"] = now
            result = session.exec(
                update(DeploymentReplicaDB)
                .where(
                    DeploymentReplicaDB.deployment_id == deployment_id,
                    DeploymentReplicaDB.replica_id == replica_id,
                    parent_stable,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                session.rollback()
                return False
            session.commit()
            return True

    def _sync_replica_group_status(
        self,
        deployment_id: str,
        *,
        allow_busy: bool = False,
    ) -> Dict[str, Any]:
        """Reconcile each canonical child through its exact managed identity."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            if deployment.replica_operation_token is not None:
                if allow_busy:
                    return self._deployment_to_dict_with_replicas(session, deployment)
                raise ReplicaOperationBusyError(
                    "deployment replica operation already in progress"
                )
            generation = deployment.replica_operation_generation
            probe_user_id = deployment.user_id
            probe_framework = deployment.inference_framework or ""
            replicas = [
                replica.to_dict()
                for replica in session.exec(
                    select(DeploymentReplicaDB)
                    .where(DeploymentReplicaDB.deployment_id == deployment_id)
                    .order_by(DeploymentReplicaDB.replica_index)
                ).all()
            ]

        for replica in replicas:
            status = "stopped"
            health_status = "UNKNOWN"
            error_message = None
            try:
                present = docker_deployer.container_exists_authoritative(
                    replica["container_name"]
                )
            except Exception as exc:
                logger.warning(
                    "Refusing replica status sync for %s/%s: %s",
                    deployment_id,
                    replica["replica_id"],
                    exc,
                )
                continue
            try:
                if present:
                    container_id = docker_deployer.get_managed_container_id(
                        replica["container_name"],
                        deployment_id=deployment_id,
                        replica_id=replica["replica_id"],
                    )
                    if docker_deployer.container_running_identity(container_id):
                        status = "running"
                        if docker_deployer.probe_service_ready(
                            replica["endpoint"],
                            user_id=probe_user_id,
                            framework=probe_framework,
                            timeout=2.0,
                        ):
                            health_status = "HEALTHY"
                        else:
                            health_status = "UNHEALTHY"
                            error_message = "replica readiness probe failed"
            except Exception:
                status = "failed"
                health_status = "UNHEALTHY"
                error_message = "managed replica identity check failed"
            if not self._write_replica_sync_state(
                deployment_id,
                replica["replica_id"],
                generation=generation,
                status=status,
                health_status=health_status,
                error_message=error_message,
            ):
                if not allow_busy:
                    raise ReplicaOperationBusyError(
                        "deployment replica operation changed during status sync"
                    )
                current = self.get_deployment(deployment_id)
                if current is None:
                    raise ValueError(f"Deployment not found: {deployment_id}")
                return current

        with get_session() as session:
            children = list(
                session.exec(
                    select(DeploymentReplicaDB)
                    .where(DeploymentReplicaDB.deployment_id == deployment_id)
                    .order_by(DeploymentReplicaDB.replica_index)
                ).all()
            )
            status = self.derive_group_status(
                [replica.to_dict() for replica in children]
            )
            health_status = (
                "HEALTHY"
                if status == "running"
                else "UNHEALTHY"
                if status == "failed"
                else "UNKNOWN"
            )
            error_message = (
                None if status in {"running", "stopped"} else "replica group degraded"
            )
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == deployment_id,
                    DeploymentDB.replica_operation_token.is_(None),
                    DeploymentDB.replica_operation_generation == generation,
                )
                .values(
                    status=status,
                    health_status=health_status,
                    error_message=error_message,
                    updated_at=now_naive(),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                if not allow_busy:
                    raise ReplicaOperationBusyError(
                        "deployment replica operation changed during status sync"
                    )
            else:
                session.commit()
        current = self.get_deployment(deployment_id)
        if current is None:
            raise DeploymentReplicaNotFoundError("deployment not found")
        return current

    @staticmethod
    def _legacy_status_sync_snapshot(
        deployment: DeploymentDB,
    ) -> _LegacyStatusSyncSnapshot:
        return _LegacyStatusSyncSnapshot(
            deployment_id=deployment.deployment_id,
            generation=deployment.replica_operation_generation,
            status=deployment.status,
            updated_at=deployment.updated_at,
            model_id=deployment.model_id,
            model_uid=deployment.model_uid,
            xinference_endpoint=deployment.xinference_endpoint,
            user_id=deployment.user_id,
            deploy_mode=deployment.deploy_mode,
            container_name=deployment.container_name,
            inference_framework=deployment.inference_framework,
            error_message=deployment.error_message,
            started_at=deployment.started_at,
            stopped_at=deployment.stopped_at,
        )

    @staticmethod
    def _legacy_status_sync_values(
        snapshot: _LegacyStatusSyncSnapshot,
        *,
        status: str | None = None,
        error_message: object = _LEGACY_SYNC_UNCHANGED,
    ) -> Dict[str, Any]:
        """Build the trusted state transition produced by one runtime probe."""
        next_status = status or snapshot.status
        if next_status != snapshot.status:
            allowed = DeploymentDB.VALID_TRANSITIONS.get(snapshot.status)
            if allowed is not None and next_status not in allowed:
                raise ValueError(
                    f"Invalid status transition: {snapshot.status} -> {next_status}"
                )

        now = now_naive()
        values: Dict[str, Any] = {
            "status": next_status,
            "updated_at": now,
        }
        if error_message is not _LEGACY_SYNC_UNCHANGED:
            values["error_message"] = error_message
        elif next_status == "running" and next_status != snapshot.status:
            values["error_message"] = None
        if next_status == "running" and snapshot.started_at is None:
            values["started_at"] = now
        if next_status in {"stopped", "failed"} and next_status != snapshot.status:
            values["stopped_at"] = now
        return values

    def _write_legacy_status_sync_state(
        self,
        snapshot: _LegacyStatusSyncSnapshot,
        *,
        status: str | None = None,
        error_message: object = _LEGACY_SYNC_UNCHANGED,
    ) -> bool:
        """Conditionally persist one probe only while its exact snapshot is current."""
        values = self._legacy_status_sync_values(
            snapshot,
            status=status,
            error_message=error_message,
        )
        with get_session() as session:
            result = session.exec(
                update(DeploymentDB)
                .where(
                    DeploymentDB.deployment_id == snapshot.deployment_id,
                    DeploymentDB.replica_operation_token.is_(None),
                    DeploymentDB.replica_operation_generation == snapshot.generation,
                    DeploymentDB.status == snapshot.status,
                    DeploymentDB.updated_at == snapshot.updated_at,
                    DeploymentDB.model_id == snapshot.model_id,
                    DeploymentDB.model_uid == snapshot.model_uid,
                    DeploymentDB.xinference_endpoint == snapshot.xinference_endpoint,
                    DeploymentDB.user_id == snapshot.user_id,
                    DeploymentDB.deploy_mode == snapshot.deploy_mode,
                    DeploymentDB.container_name == snapshot.container_name,
                    DeploymentDB.inference_framework
                    == snapshot.inference_framework,
                    DeploymentDB.error_message == snapshot.error_message,
                    DeploymentDB.started_at == snapshot.started_at,
                    DeploymentDB.stopped_at == snapshot.stopped_at,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                session.rollback()
                return False
            session.commit()
            return True

    def _current_deployment_after_status_sync(
        self,
        deployment_id: str,
    ) -> Dict[str, Any]:
        current = self.get_deployment(deployment_id)
        if current is None:
            raise DeploymentReplicaNotFoundError("deployment not found")
        return current

    def sync_status(self, deployment_id: str) -> Dict[str, Any]:
        """Sync deployment status with Xinference."""
        with get_session() as session:
            statement = (
                select(DeploymentDB)
                .where(DeploymentDB.deployment_id == deployment_id)
                .with_for_update()
            )
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            if self._uses_replica_lifecycle(deployment):
                replica_lifecycle = True
                snapshot = None
            else:
                replica_lifecycle = False
                if deployment.replica_operation_token is not None:
                    raise ReplicaOperationBusyError(
                        "deployment replica operation already in progress"
                    )
                snapshot = self._legacy_status_sync_snapshot(deployment)

            current = self._deployment_to_dict(deployment)

        if replica_lifecycle:
            return self._sync_replica_group_status(deployment_id)

        assert snapshot is not None
        if not snapshot.model_uid:
            return current

        try:
            client = self._get_xinference_client(
                snapshot.xinference_endpoint,
                user_id=snapshot.user_id,
            )
            runtime_status = client.get_model_status(snapshot.model_uid)
            next_status = None
            if runtime_status == "running" and snapshot.status != "running":
                next_status = "running"
            elif runtime_status == "stopped" and snapshot.status not in {
                "stopped",
                "pending",
            }:
                next_status = "stopped"
            persisted = self._write_legacy_status_sync_state(
                snapshot,
                status=next_status,
            )
        except Exception as exc:
            logger.warning("Failed to sync status for %s: %s", deployment_id, exc)
            return self._current_deployment_after_status_sync(deployment_id)
        if not persisted:
            self._current_deployment_after_status_sync(deployment_id)
            raise ReplicaOperationBusyError(
                "deployment changed during status sync"
            )
        return self._current_deployment_after_status_sync(deployment_id)

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
        active_statuses = [
            "running",
            "degraded",
            "starting",
            "restarting",
            "stopping",
        ]

        # Take only detached snapshots. No ORM row or transaction survives into
        # a Docker/Xinference probe.
        with get_session() as session:
            candidates = list(
                session.exec(
                    select(DeploymentDB)
                    .where(DeploymentDB.status.in_(active_statuses))
                    .order_by(DeploymentDB.deployment_id)
                    .with_for_update()
                ).all()
            )
            replica_deployment_ids: List[str] = []
            legacy_snapshots: List[_LegacyStatusSyncSnapshot] = []
            for deployment in candidates:
                if self._uses_replica_lifecycle(deployment):
                    replica_deployment_ids.append(deployment.deployment_id)
                elif deployment.replica_operation_token is None:
                    legacy_snapshots.append(
                        self._legacy_status_sync_snapshot(deployment)
                    )

        for replica_deployment_id in replica_deployment_ids:
            before = self.get_deployment(replica_deployment_id)
            after = self._sync_replica_group_status(
                replica_deployment_id,
                allow_busy=True,
            )
            before_state = (
                before.get("status") if before else None,
                [
                    (child.get("status"), child.get("health_status"))
                    for child in (before or {}).get("replica_instances", [])
                ],
            )
            after_state = (
                after.get("status"),
                [
                    (child.get("status"), child.get("health_status"))
                    for child in after.get("replica_instances", [])
                ],
            )
            if before_state != after_state:
                updated_count += 1

        container_snapshots = [
            snapshot
            for snapshot in legacy_snapshots
            if snapshot.deploy_mode == "container" and snapshot.container_name
        ]
        external_snapshots = [
            snapshot
            for snapshot in legacy_snapshots
            if snapshot not in container_snapshots and snapshot.model_uid
        ]
        config_sync_ids: set[str] = set()

        for snapshot in container_snapshots:
            try:
                container_id = self._legacy_container_id_for_identity(
                    deployment_id=snapshot.deployment_id,
                    model_id=snapshot.model_id,
                    deploy_mode=snapshot.deploy_mode,
                    container_name=snapshot.container_name,
                    inference_framework=snapshot.inference_framework,
                )
                container_exists = container_id is not None
                container_running = bool(
                    container_id
                    and docker_deployer.container_running_identity(container_id)
                )
            except Exception as exc:
                logger.warning(
                    "Refusing auto-sync for deployment %s: %s",
                    snapshot.deployment_id,
                    exc,
                )
                continue

            next_status: str | None = None
            error_message: object = _LEGACY_SYNC_UNCHANGED
            count_update = False
            if snapshot.status == "stopping":
                if snapshot.updated_at and (now - snapshot.updated_at) > stopping_timeout:
                    next_status = "failed"
                    error_message = (
                        f"Stopping timed out after {STOPPING_TIMEOUT_MINUTES} "
                        "minutes (auto-sync)"
                    )
                    count_update = True
                elif not container_running:
                    next_status = "stopped"
                    count_update = True
            elif snapshot.status in {"starting", "restarting"}:
                if snapshot.updated_at and (now - snapshot.updated_at) > starting_timeout:
                    timeout_label = (
                        "Restarting" if snapshot.status == "restarting" else "Starting"
                    )
                    next_status = "failed"
                    error_message = (
                        f"{timeout_label} timed out after "
                        f"{STARTING_TIMEOUT_MINUTES} minutes (auto-sync)"
                    )
                    count_update = True
                elif not container_running:
                    next_status = "failed"
                    if not container_exists:
                        error_message = "Container not found (auto-sync)"
                    elif snapshot.status == "restarting":
                        error_message = "Container stopped during restart (auto-sync)"
                    else:
                        error_message = "Container stopped during startup (auto-sync)"
                    count_update = True
                else:
                    next_status = "running"
                    count_update = True
            elif not container_running:
                next_status = "stopped"
                error_message = "Container not running (auto-sync)"
                count_update = True
            elif snapshot.error_message and "auto-sync" in snapshot.error_message:
                error_message = None

            if next_status is None and error_message is _LEGACY_SYNC_UNCHANGED:
                continue
            try:
                persisted = self._write_legacy_status_sync_state(
                    snapshot,
                    status=next_status,
                    error_message=error_message,
                )
            except ValueError as exc:
                logger.warning(
                    "Refusing auto-sync transition for deployment %s: %s",
                    snapshot.deployment_id,
                    exc,
                )
                continue
            if persisted and count_update:
                updated_count += 1
                config_sync_ids.add(snapshot.deployment_id)

        endpoint_models: Dict[Tuple[str, Optional[str]], List[str]] = {}
        for snapshot in external_snapshots:
            endpoint_key = (snapshot.xinference_endpoint, snapshot.user_id)
            endpoint_models.setdefault(endpoint_key, []).append(snapshot.model_uid or "")

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
                    model.get("id") for model in models if model.get("id")
                }
            except Exception as exc:
                logger.warning("Failed to get models from %s: %s", endpoint, exc)

        for snapshot in external_snapshots:
            endpoint_key = (snapshot.xinference_endpoint, snapshot.user_id)
            running_models = running_models_by_endpoint.get(endpoint_key)
            if running_models is None:
                continue
            next_status = None
            error_message = _LEGACY_SYNC_UNCHANGED
            count_update = False
            if snapshot.model_uid not in running_models:
                if snapshot.status == "restarting":
                    if (
                        snapshot.updated_at
                        and (now - snapshot.updated_at) > starting_timeout
                    ):
                        next_status = "failed"
                        error_message = (
                            f"Restarting timed out after {STARTING_TIMEOUT_MINUTES} "
                            "minutes (auto-sync)"
                        )
                        count_update = True
                    else:
                        continue
                else:
                    next_status = "stopped"
                    error_message = "Model not running on Xinference (auto-sync)"
                    count_update = True
            elif snapshot.error_message and "auto-sync" in snapshot.error_message:
                error_message = None

            if next_status is None and error_message is _LEGACY_SYNC_UNCHANGED:
                continue
            try:
                persisted = self._write_legacy_status_sync_state(
                    snapshot,
                    status=next_status,
                    error_message=error_message,
                )
            except ValueError as exc:
                logger.warning(
                    "Refusing auto-sync transition for deployment %s: %s",
                    snapshot.deployment_id,
                    exc,
                )
                continue
            if persisted and count_update:
                updated_count += 1
                config_sync_ids.add(snapshot.deployment_id)

        if config_sync_ids:
            with get_session() as session:
                changed = [
                    deployment.model_copy(deep=True)
                    for deployment in session.exec(
                        select(DeploymentDB)
                        .where(DeploymentDB.deployment_id.in_(config_sync_ids))
                        .order_by(DeploymentDB.deployment_id)
                    ).all()
                ]
            self._sync_configs_from_deployments(changed)

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

            if self._uses_replica_lifecycle(deployment):
                replicas = self.list_replicas(
                    deployment.deployment_id,
                    user_id=deployment.user_id,
                )
                if len(replicas) != 1:
                    logger.info(
                        "Skipping automatic config for multi-replica deployment %s",
                        deployment.deployment_id,
                    )
                    return
                self.create_config_for_deployment(
                    deployment.deployment_id,
                    deployment_replica_id=replicas[0]["replica_id"],
                    user_id=deployment.user_id,
                )
                return

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

    def create_config_for_deployment(
        self,
        deployment_id: str,
        deployment_replica_id: str | None = None,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
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
        deployment_dict, selected_replica = self.resolve_replica_selection(
            deployment_id,
            deployment_replica_id,
            user_id=user_id,
            require_healthy=True,
        )

        if selected_replica is None and deployment_dict["status"] != "running":
            raise ValueError(
                "Deployment is not running "
                f"(status: {deployment_dict['status']})"
            )

        # Get model info
        model = model_registry_service.get_model(deployment_dict["model_id"])
        if not model:
            raise ValueError(f"Model not found: {deployment_dict['model_id']}")

        # Check if config already exists for this deployment
        if selected_replica is not None:
            existing = model_config_service.get_config_by_deployment_replica_id(
                deployment_id,
                selected_replica["replica_id"],
            )
        else:
            existing = model_config_service.get_config_by_deployment_id(deployment_id)
        if existing:
            raise ValueError(
                f"Config already exists for deployment: {existing['config_id']}"
            )

        # Generate config name
        config_name = f"{model['model_name']}-deployed-{deployment_id[:8]}"
        if selected_replica is not None:
            config_name = f"{config_name}-r{selected_replica['replica_index']}"

        # Build API endpoint
        # For container mode: use host IP + port (accessible from anywhere)
        # For external mode: normalize localhost to host IP
        if selected_replica is not None:
            api_endpoint = self._normalize_endpoint_for_config(
                selected_replica["endpoint"]
            )
            container_name = selected_replica["container_name"]
        elif deployment_dict.get("deploy_mode") == "container" and deployment_dict.get("port"):
            host_ip = self._get_host_ip()
            api_endpoint = f"http://{host_ip}:{deployment_dict['port']}"
            container_name = deployment_dict.get("container_name")
            logger.info(f"Container mode: using host endpoint {api_endpoint}")
        else:
            api_endpoint = self._normalize_endpoint_for_config(deployment_dict["xinference_endpoint"])
            container_name = deployment_dict.get("container_name")

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
            deployment_replica_id=(
                selected_replica["replica_id"]
                if selected_replica is not None
                else None
            ),
            container_name=container_name,
            inference_framework=deployment_dict.get("inference_framework"),
        )

        logger.info(f"Created config for deployment {deployment_id}: {config['config_id']}")
        return config


# Global service instance
deployment_service = DeploymentService()
