"""
Docker-based inference deployment manager.

Automatically creates and manages Docker containers running inference frameworks
(Xinference, vLLM, SGLang) for model deployment.
"""

import logging
import os
import re
import shlex
import subprocess
import threading
import time
from ipaddress import ip_address
from typing import Optional, Dict, Any, Tuple, Set, List, Sequence
import requests

from ..config.public_origin import is_loopback_bind_address
from ..config.settings import get_settings
from ..storage.services.outbound_endpoint_policy import request_user_outbound
from .launch_config import (
    xinference_model_launch_overrides,
    xinference_model_type,
)

logger = logging.getLogger(__name__)

# Default Docker images for different frameworks
# 镜像版本在根目录 .env 中统一配置，通过环境变量读取
# Xinference v3.1.0 with digest-bound compatibility payloads.
DEFAULT_XINFERENCE_IMAGE = os.environ.get(
    "XINFERENCE_IMAGE",
    "xprobe/xinference:v3.1.0@sha256:ec41459d15cc1c18842370c267e9c9a12a0001245dea9fe3b939e4075dc18178",
)
# vLLM v0.26.0 uses the pooling runner for embedding and rerank models.
DEFAULT_VLLM_IMAGE = os.environ.get(
    "VLLM_IMAGE",
    "vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52",
)
# SGLang v0.5.17 official release.
DEFAULT_SGLANG_IMAGE = os.environ.get(
    "SGLANG_IMAGE",
    "lmsysorg/sglang:v0.5.17@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155",
)
INFERENCE_CONTAINER_CREATE_TIMEOUT_SECONDS = 3600

# Port range for auto-assignment
PORT_RANGE_START = int(os.environ.get("PORT_RANGE_START", "9997"))
PORT_RANGE_END = int(os.environ.get("PORT_RANGE_END", "10100"))

# Volume mounts for containers
DEFAULT_DATA_VOLUME = os.environ.get("DATA_VOLUME", "/data/train_res:/data")
DEFAULT_MODELS_VOLUME = os.environ.get("MODELS_VOLUME", "/workspace/train-factory/models:/app/models")
DEFAULT_OUTPUT_VOLUME = os.environ.get("OUTPUT_VOLUME", "/workspace/train-factory/output:/app/output")
# 额外的 volume 挂载：外部模型目录（如 ModelScope/HuggingFace 下载的原始模型）
DEFAULT_EXTRA_MODEL_VOLUME = os.environ.get("EXTRA_MODEL_VOLUME", "/data/models:/data/models:ro")
DEFAULT_XINFERENCE_PATCH_VOLUME = os.environ.get(
    "XINFERENCE_PATCH_VOLUME",
    "/workspace/train-factory/docker/xinference-patches:"
    "/opt/trainfactory/xinference-patches:ro",
)
DEFAULT_XINFERENCE_CONTRACT_VOLUME = os.environ.get(
    "XINFERENCE_CONTRACT_VOLUME",
    "/workspace/train-factory/docker/inference-contracts:"
    "/opt/trainfactory/inference-contracts:ro",
)
DEFAULT_SGLANG_TEMPLATE_VOLUME = os.environ.get(
    "SGLANG_TEMPLATE_VOLUME",
    "/workspace/train-factory/docker/sglang-templates:"
    "/opt/trainfactory/sglang-templates:ro",
)

# HuggingFace mirror for China
DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Docker network name — must match compose stack so API container can reach deployment containers
DEFAULT_DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK_NAME", "trainfactory_network")

# 推理容器共享内存：Docker 默认 /dev/shm 仅 64MB，vLLM/SGLang 加载大模型、
# 长上下文 tokenizer 时共享内存不足会 OOM。默认 2g，设 0/0g = 不限制（不加参数）。
DEFAULT_SHM_SIZE = os.environ.get("DEPLOY_SHM_SIZE", "2g")


def _shm_size_args() -> List[str]:
    """返回 docker run 的 --shm-size 参数；DEPLOY_SHM_SIZE 为 0/0g/空 = 不限制。"""
    raw = os.environ.get("DEPLOY_SHM_SIZE")
    if raw is None:
        raw = DEFAULT_SHM_SIZE
    raw = raw.strip()
    if raw.lower() in ("0", "0g", ""):
        return []
    return ["--shm-size", raw]


def _validated_gpu_ids(gpu_ids: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(gpu_ids)
    if not normalized:
        raise ValueError("GPU IDs must not be empty")
    if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in normalized):
        raise ValueError("GPU IDs must be non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("GPU IDs must not contain duplicates")
    return normalized


def _validated_server_argv(
    server_argv: Sequence[str],
    *,
    executable: str,
) -> tuple[str, ...]:
    normalized = tuple(server_argv)
    if not normalized or normalized[0] != executable:
        raise ValueError("server argv executable is invalid")
    if any(
        not isinstance(value, str) or not value or "\0" in value
        for value in normalized
    ):
        raise ValueError("server argv contains an invalid value")
    return normalized


def _normalized_host_bind_address() -> tuple[str, bool]:
    settings = get_settings()
    raw_address = getattr(settings, "host_bind_address", "127.0.0.1")
    is_loopback_bind_address(raw_address, "HOST_BIND_ADDRESS")
    if raw_address.lower() == "localhost":
        return "127.0.0.1", False
    parsed_address = ip_address(raw_address)
    return str(parsed_address), parsed_address.version == 6


def _published_container_port(port: int) -> str:
    host_address, is_ipv6 = _normalized_host_bind_address()
    if is_ipv6:
        host_address = f"[{host_address}]"
    return f"{host_address}:{port}:{port}"


class DockerDeployer:
    """Manages Docker containers for inference model deployment."""

    def __init__(
        self,
        xinference_image: str = DEFAULT_XINFERENCE_IMAGE,
        vllm_image: str = DEFAULT_VLLM_IMAGE,
        sglang_image: str = DEFAULT_SGLANG_IMAGE,
        data_volume: str = DEFAULT_DATA_VOLUME,
        models_volume: str = DEFAULT_MODELS_VOLUME,
        output_volume: str = DEFAULT_OUTPUT_VOLUME,
        xinference_patch_volume: str = DEFAULT_XINFERENCE_PATCH_VOLUME,
        xinference_contract_volume: str = DEFAULT_XINFERENCE_CONTRACT_VOLUME,
        sglang_template_volume: str = DEFAULT_SGLANG_TEMPLATE_VOLUME,
    ):
        """
        Initialize Docker deployer.

        Args:
            xinference_image: Docker image for Xinference
            vllm_image: Docker image for vLLM
            sglang_image: Docker image for SGLang
            data_volume: Volume mount for data (host:container)
            models_volume: Volume mount for models (host:container)
            output_volume: Volume mount for training output (host:container)
        """
        self.xinference_image = xinference_image
        self.vllm_image = vllm_image
        self.sglang_image = sglang_image
        # Backwards compatibility
        self.image = xinference_image
        self.data_volume = data_volume
        self.models_volume = models_volume
        self.output_volume = output_volume
        self.extra_model_volume = DEFAULT_EXTRA_MODEL_VOLUME
        self.xinference_patch_volume = xinference_patch_volume
        self.xinference_contract_volume = xinference_contract_volume
        self.sglang_template_volume = sglang_template_volume
        self.hf_endpoint = DEFAULT_HF_ENDPOINT
        # Lock for thread-safe port allocation
        self._port_lock = threading.Lock()
        # Track transient reservations by opaque owner so unrelated lifecycle
        # operations cannot release another request's pre-DB allocation.
        self._reserved_ports: Dict[int, str] = {}

    @staticmethod
    def _cmd_to_string(cmd: list) -> str:
        """Convert a command list to a human-readable string."""
        return " ".join(shlex.quote(arg) if " " in arg or "\n" in arg or "\t" in arg else arg for arg in cmd)

    def _run_command(self, cmd: list, timeout: int = 30) -> Tuple[bool, str]:
        """Run shell command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def _run_container_create(
        self,
        cmd: list,
        container_name: str,
        timeout: int = INFERENCE_CONTAINER_CREATE_TIMEOUT_SECONDS,
    ) -> Tuple[bool, str]:
        """
        Run `docker run -d` for container creation.

        首次拉取大镜像（vLLM/SGLang 镜像可达 10GB+）时 180s 可能不够：
        超时后容器可能已在 daemon 侧创建成功（拉取恰好完成）。此时用
        `docker inspect` 复核，避免把"慢但成功"误报为失败。
        """
        success, output = self._run_command(cmd, timeout=timeout)
        if success or "timed out" not in output:
            return success, output

        # Timeout: verify whether the daemon created the container anyway.
        ok, inspect_output = self._run_command(
            ["docker", "inspect", container_name], timeout=15
        )
        if ok:
            logger.warning(
                f"Container {container_name} created despite pull exceeding {timeout}s timeout"
            )
            return True, f"Container {container_name} created (image pull exceeded {timeout}s timeout)"
        return success, output

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use on the host."""
        success, output = self._run_command(
            [
                "docker",
                "ps",
                "--filter",
                f"publish={port}",
                "--format",
                "{{.ID}}",
            ]
        )
        return success and bool(output.strip())

    def _get_db_reserved_ports(self) -> Set[int]:
        """Get ports reserved by deployments and configs in the database.

        This catches ports that are allocated in DB but container not yet created,
        preventing race conditions across process restarts.
        Also checks model_configs to avoid conflicts with external API endpoints.
        """
        reserved_ports: Set[int] = set()
        try:
            from sqlmodel import select
            from ..storage.database import get_session
            from ..storage.entities.deployment_entity import DeploymentDB
            from ..storage.entities.model_config_entity import ModelConfigDB

            with get_session() as session:
                # Get all container deployments that are pending/starting/running
                # These may have reserved ports even if container isn't visible yet
                statement = select(DeploymentDB).where(
                    DeploymentDB.deploy_mode == "container",
                    DeploymentDB.port.isnot(None),
                    DeploymentDB.status.in_(["pending", "starting", "running", "restarting"])
                )
                deployments = session.exec(statement).all()
                reserved_ports.update({d.port for d in deployments if d.port is not None})

                # Also check model_configs for ports used by external APIs
                # to avoid deploying on ports that configs are pointing to
                config_statement = select(ModelConfigDB).where(
                    ModelConfigDB.status == "active",
                    ModelConfigDB.api_endpoint.isnot(None)
                )
                configs = session.exec(config_statement).all()
                for config in configs:
                    if config.api_endpoint:
                        # Extract port from endpoint URL
                        import re
                        match = re.search(r':(\d+)', config.api_endpoint)
                        if match:
                            port = int(match.group(1))
                            # Only reserve ports in our deployment range
                            if PORT_RANGE_START <= port <= PORT_RANGE_END:
                                reserved_ports.add(port)

                return reserved_ports
        except Exception as e:
            logger.warning(f"Failed to query DB for reserved ports: {e}")
            return reserved_ports

    def find_available_port(
        self,
        start: int = PORT_RANGE_START,
        end: int = PORT_RANGE_END,
        *,
        owner_token: str,
    ) -> int:
        """Find an available port in the given range.

        Uses a lock to prevent concurrent allocation of the same port.
        Also checks database for ports reserved by pending deployments.
        Returns a reserved port. The same owner token must be supplied to
        release_port() or confirm_port().
        """
        if not owner_token:
            raise ValueError("Port reservation owner token must be non-empty")
        with self._port_lock:
            # Get ports reserved in database (handles multi-process/restart scenarios)
            db_reserved_ports = self._get_db_reserved_ports()

            for port in range(start, end):
                # Skip if port is reserved by another concurrent request (in-memory)
                if port in self._reserved_ports:
                    continue
                # Skip if port is reserved in database (cross-process safety)
                if port in db_reserved_ports:
                    continue
                # Check if port is in use by existing containers
                if not self._is_port_in_use(port):
                    # Reserve the port to prevent concurrent allocation
                    self._reserved_ports[port] = owner_token
                    logger.debug(f"Reserved port {port}")
                    return port
            raise RuntimeError(f"No available ports in range {start}-{end}")

    def release_port(self, port: int, *, owner_token: str) -> bool:
        """Release a transient reservation only when the caller owns it."""
        with self._port_lock:
            if self._reserved_ports.get(port) != owner_token:
                return False
            del self._reserved_ports[port]
            logger.debug(f"Released port {port}")
            return True

    def confirm_port(self, port: int, *, owner_token: str) -> bool:
        """Confirm a durable allocation and clear its transient reservation."""
        with self._port_lock:
            if self._reserved_ports.get(port) != owner_token:
                return False
            del self._reserved_ports[port]
            logger.debug(f"Confirmed port {port} in use")
            return True

    def container_exists(self, container_name: str) -> bool:
        """Check if a container exists (running or stopped)."""
        success, output = self._run_command([
            "docker", "ps", "-a", "--format", "{{.Names}}",
            "--filter", f"name=^{container_name}$"
        ])
        return success and container_name in output

    def container_running(self, container_name: str) -> bool:
        """Check if a container is running."""
        success, output = self._run_command([
            "docker", "ps", "--format", "{{.Names}}",
            "--filter", f"name=^{container_name}$"
        ])
        return success and container_name in output

    def stop_container(self, container_name: str) -> bool:
        """Stop a running container."""
        if not self.container_running(container_name):
            return True
        success, _ = self._run_command(["docker", "stop", container_name], timeout=30)
        return success

    def remove_container(self, container_name: str) -> bool:
        """Remove a container (stops first if running)."""
        if self.container_running(container_name):
            self.stop_container(container_name)
        if self.container_exists(container_name):
            success, _ = self._run_command(["docker", "rm", container_name])
            return success
        return True

    def restart_container(self, container_name: str, timeout: int = 30) -> Tuple[bool, str]:
        """
        Restart a Docker container.

        This completely restarts the container process, clearing all GPU memory cache.
        Equivalent to `docker restart <container_name>`.

        Args:
            container_name: Name of the container to restart
            timeout: Timeout in seconds for the restart operation

        Returns:
            Tuple of (success, message)
        """
        if not self.container_exists(container_name):
            return False, f"Container not found: {container_name}"

        logger.info(f"Restarting container: {container_name}")
        success, output = self._run_command(
            ["docker", "restart", container_name],
            timeout=timeout
        )

        if success:
            logger.info(f"Container restarted: {container_name}")
            return True, "Container restarted successfully"
        else:
            logger.error(f"Failed to restart container {container_name}: {output}")
            return False, f"Failed to restart container: {output}"

    def create_xinference_container(
        self,
        container_name: str,
        port: int,
        gpu_id: int = 0,
        model_name: str = None,
        model_uid: str = None,
        model_path: str = None,
        model_type: str = "embedding",
    ) -> Tuple[bool, str, str]:
        """
        Create and start a Xinference container.

        Args:
            container_name: Name for the container
            port: Port to expose
            gpu_id: GPU device ID to use
            model_name: Model name for Xinference
            model_uid: Model UID for Xinference
            model_path: Path to model files (container path)
            model_type: Model type (embedding, rerank, llm)

        Returns:
            (success, message)
        """
        model_type = xinference_model_type(model_type)
        if model_type not in ("embedding", "rerank", "llm"):
            raise ValueError("model_type is invalid")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("container port is invalid")
        _validated_gpu_ids((gpu_id,))
        published_port = _published_container_port(port)

        model_name = model_name or model_uid
        launch_overrides = xinference_model_launch_overrides(
            model_type=model_type,
            model_family=model_name,
        )
        dtype_arg = (
            " \\\n                --torch_dtype bfloat16"
            if launch_overrides
            else ""
        )
        inner_cmd = f"""
            xinference-local -H 0.0.0.0 -p {port} &
            until curl -s http://127.0.0.1:{port}/v1/cluster/auth > /dev/null 2>&1; do
                sleep 5
            done &&
            xinference launch \
                --endpoint http://127.0.0.1:{port} \
                --model-name {shlex.quote(model_name)} \
                --model-uid {shlex.quote(model_uid)} \
                --model-path {shlex.quote(model_path)} \
                --model-format pytorch \
                --model-type {shlex.quote(model_type)}{dtype_arg} &&
            tail -f /dev/null
        """

        # Build docker run command
        # Join compose network so API container can access this container by name
        cmd = [
            "docker", "run", "--pull=missing", "-d",
            "--name", container_name,
            "--network", DEFAULT_DOCKER_NETWORK,
            "--gpus", f"device={gpu_id}",
            "-p", published_port,
            "-e", f"HF_ENDPOINT={self.hf_endpoint}",
            "-e", (
                "ALLOW_MODEL_REMOTE_CODE=true"
                if get_settings().allow_model_remote_code
                else "ALLOW_MODEL_REMOTE_CODE=false"
            ),
            "-v", self.data_volume,
            "-v", self.models_volume,
            "-v", self.output_volume,
            "-v", self.extra_model_volume,
            "-v", self.xinference_patch_volume,
            "-v", self.xinference_contract_volume,
            *_shm_size_args(),
            "--log-opt", "max-size=10m",
            "--log-opt", "max-file=3",
            "--entrypoint", "python",
            self.image,
            "/opt/trainfactory/xinference-patches/apply_qwen3_compatibility.py",
            "--image-ref",
            self.image,
            "--contract",
            "/opt/trainfactory/inference-contracts/qwen3-compatibility.json",
            "--",
            "bash", "-c", inner_cmd,
        ]

        docker_cmd = self._cmd_to_string(cmd)
        logger.info(f"Creating container {container_name} on port {port} with GPU {gpu_id}")
        self.remove_container(container_name)
        success, output = self._run_container_create(cmd, container_name)

        if success:
            logger.info(f"Container {container_name} created successfully")
            return True, f"Container {container_name} created on port {port}", docker_cmd
        else:
            logger.error(f"Failed to create container {container_name}: {output}")
            return False, output, docker_cmd

    def wait_for_xinference(
        self,
        endpoint: str,
        timeout: int = 120,
        poll_interval: int = 5,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Wait for Xinference service to be ready.

        Args:
            endpoint: Xinference endpoint (e.g., http://localhost:9997)
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between health checks

        Returns:
            True if service is ready, False on timeout
        """
        endpoint = endpoint.rstrip("/")
        health_url = f"{endpoint}/v1/models"

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = request_user_outbound(
                    "GET",
                    health_url,
                    user_id,
                    timeout=5,
                )
                if response.status_code == 200:
                    logger.info(f"Xinference at {endpoint} is ready")
                    return True
            except requests.exceptions.RequestException:
                pass

            logger.debug(f"Waiting for Xinference at {endpoint}...")
            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for Xinference at {endpoint}")
        return False

    def wait_for_model(
        self,
        endpoint: str,
        model_uid: str,
        timeout: int = 180,
        poll_interval: int = 10,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Wait for a specific model to be loaded.

        Args:
            endpoint: Xinference endpoint
            model_uid: Model UID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between checks

        Returns:
            True if model is loaded, False on timeout
        """
        endpoint = endpoint.rstrip("/")
        model_url = f"{endpoint}/v1/models/{model_uid}"

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = request_user_outbound(
                    "GET",
                    model_url,
                    user_id,
                    timeout=5,
                )
                if response.status_code == 200:
                    logger.info(f"Model {model_uid} is loaded")
                    return True
            except requests.exceptions.RequestException:
                pass

            logger.debug(f"Waiting for model {model_uid} to load...")
            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for model {model_uid}")
        return False

    def get_container_logs(self, container_name: str, tail: int = 50) -> str:
        """Get recent logs from a container."""
        success, output = self._run_command([
            "docker", "logs", "--tail", str(tail), container_name
        ])
        return output if success else ""

    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, Any]]:
        """Get GPU memory usage for all GPUs."""
        success, output = self._run_command([
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits"
        ])

        if not success:
            return {}

        result = {}
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_id = int(parts[0])
                result[gpu_id] = {
                    "used_mb": int(parts[1]),
                    "total_mb": int(parts[2]),
                    "free_mb": int(parts[3]),
                    "utilization": round(int(parts[1]) / int(parts[2]), 2) if int(parts[2]) > 0 else 0,
                }
        return result

    def reset_gpu(self, gpu_id: int) -> Tuple[bool, str]:
        """Reset a GPU to clear driver-level memory (requires no active processes)."""
        success, output = self._run_command(
            ["nvidia-smi", "-i", str(gpu_id), "--gpu-reset"],
            timeout=30,
        )
        message = output.strip() or ("GPU reset succeeded" if success else "GPU reset failed")
        if success:
            logger.info(f"GPU {gpu_id} reset successfully")
        else:
            logger.warning(f"Failed to reset GPU {gpu_id}: {message}")
        return success, message

    def select_gpu(self, min_free_mb: int = 4000) -> Optional[int]:
        """Select a GPU with enough free memory."""
        gpu_info = self.get_gpu_memory_usage()
        if not gpu_info:
            return 0  # Default to GPU 0 if cannot query

        # Sort by free memory (descending)
        sorted_gpus = sorted(
            gpu_info.items(),
            key=lambda x: x[1]["free_mb"],
            reverse=True
        )

        for gpu_id, info in sorted_gpus:
            if info["free_mb"] >= min_free_mb:
                return gpu_id

        # Return GPU with most free memory even if below threshold
        if sorted_gpus:
            return sorted_gpus[0][0]
        return 0

    def list_xinference_containers(self) -> list:
        """List service-managed inference containers for the current project."""
        success, output = self._run_command([
            "docker", "ps", "-a", "--format", "{{.Names}}",
        ])
        if not success or not output.strip():
            return []

        project = os.environ.get("COMPOSE_PROJECT_NAME", "trainfactory")
        managed_name = re.compile(
            rf"^{re.escape(project)}-(?:xf|vllm|sglang)-.+-[0-9a-f]{{8}}$",
            re.IGNORECASE,
        )
        return [
            name
            for raw_name in output.splitlines()
            if (name := raw_name.strip()) and managed_name.fullmatch(name)
        ]

    def cleanup_orphan_containers(self, valid_container_names: set) -> int:
        """
        Cleanup orphan containers that are not in the valid set.

        Args:
            valid_container_names: Set of container names from active deployments

        Returns:
            Number of containers removed
        """
        managed_containers = self.list_xinference_containers()
        removed_count = 0

        for container_name in managed_containers:
            if container_name not in valid_container_names:
                logger.warning(f"Found orphan container: {container_name}")
                if self.remove_container(container_name):
                    removed_count += 1
                    logger.info(f"Removed orphan container: {container_name}")
                else:
                    logger.error(f"Failed to remove orphan container: {container_name}")

        return removed_count

    # ==================== vLLM Container ====================

    def create_vllm_container(
        self,
        container_name: str,
        port: int,
        gpu_ids: Sequence[int],
        server_argv: Sequence[str],
    ) -> Tuple[bool, str, str]:
        """Create vLLM from a validated immutable application argv."""
        normalized_gpu_ids = _validated_gpu_ids(gpu_ids)
        normalized_server_argv = _validated_server_argv(
            server_argv,
            executable="vllm",
        )
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("container port is invalid")
        if not isinstance(container_name, str) or not container_name:
            raise ValueError("container name is invalid")

        env_vars = ["-e", f"HF_ENDPOINT={self.hf_endpoint}"]
        if "--enable-lora" in normalized_server_argv:
            env_vars.extend(["-e", "VLLM_ALLOW_RUNTIME_LORA_UPDATING=1"])
        command = [
            "docker",
            "run",
            "--pull=missing",
            "-d",
            "--name",
            container_name,
            "--network",
            DEFAULT_DOCKER_NETWORK,
            "--gpus",
            f"device={','.join(map(str, normalized_gpu_ids))}",
            "-p",
            _published_container_port(port),
            "-v",
            self.data_volume,
            "-v",
            self.models_volume,
            "-v",
            self.output_volume,
            "-v",
            self.extra_model_volume,
            *_shm_size_args(),
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=3",
            "--entrypoint",
            normalized_server_argv[0],
            *env_vars,
            self.vllm_image,
            *normalized_server_argv[1:],
        ]
        docker_cmd = self._cmd_to_string(command)
        logger.info(
            "Creating vLLM container %s on port %s with GPUs %s",
            container_name,
            port,
            normalized_gpu_ids,
        )
        self.remove_container(container_name)
        success, output = self._run_container_create(command, container_name)
        if success:
            return True, f"Container {container_name} created on port {port}", docker_cmd
        return False, output, docker_cmd

    # ==================== SGLang Container ====================

    def create_sglang_container(
        self,
        container_name: str,
        port: int,
        gpu_ids: Sequence[int],
        server_argv: Sequence[str],
    ) -> Tuple[bool, str, str]:
        """Create SGLang from a validated immutable application argv."""
        normalized_gpu_ids = _validated_gpu_ids(gpu_ids)
        normalized_server_argv = _validated_server_argv(
            server_argv,
            executable="sglang",
        )
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("container port is invalid")
        if not isinstance(container_name, str) or not container_name:
            raise ValueError("container name is invalid")

        command = [
            "docker",
            "run",
            "--pull=missing",
            "-d",
            "--name",
            container_name,
            "--network",
            DEFAULT_DOCKER_NETWORK,
            "--gpus",
            f"device={','.join(map(str, normalized_gpu_ids))}",
            "-p",
            _published_container_port(port),
            "-e",
            f"HF_ENDPOINT={self.hf_endpoint}",
            "-v",
            self.data_volume,
            "-v",
            self.models_volume,
            "-v",
            self.output_volume,
            "-v",
            self.extra_model_volume,
            "-v",
            self.sglang_template_volume,
            *_shm_size_args(),
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=3",
            "--entrypoint",
            normalized_server_argv[0],
            self.sglang_image,
            *normalized_server_argv[1:],
        ]
        docker_cmd = self._cmd_to_string(command)
        logger.info(
            "Creating SGLang container %s on port %s with GPUs %s",
            container_name,
            port,
            normalized_gpu_ids,
        )
        self.remove_container(container_name)
        success, output = self._run_container_create(command, container_name)
        if success:
            return True, f"Container {container_name} created on port {port}", docker_cmd
        return False, output, docker_cmd

    # ==================== Generic Wait Methods ====================

    def wait_for_service(
        self,
        endpoint: str,
        timeout: int = 180,
        poll_interval: int = 5,
        user_id: Optional[str] = None,
        framework: str = "",
    ) -> bool:
        """
        Wait for an inference service to be ready.

        Works with vLLM, SGLang, and Xinference.

        Args:
            endpoint: Service endpoint (e.g., http://localhost:8000)
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between health checks
            framework: inference framework ('vllm'/'sglang' force a model-level
                readiness check; others accept any 200 health probe)

        Returns:
            True if service is ready, False on timeout
        """
        endpoint = endpoint.rstrip("/")
        framework = (framework or "").lower()

        # vLLM/SGLang：HTTP 服务器在模型权重加载完成前就响应 /health 200，
        # 必须等 /v1/models 返回 200 且含已加载模型才标记就绪（否则部署显示
        # running 但推理持续 503）。
        if framework in ("vllm", "sglang"):
            models_url = f"{endpoint}/v1/models"
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = request_user_outbound(
                        "GET",
                        models_url,
                        user_id,
                        timeout=5,
                    )
                    if response.status_code == 200:
                        try:
                            if response.json().get("data"):
                                logger.info(
                                    f"Model ready at {endpoint} (framework={framework})"
                                )
                                return True
                        except ValueError:
                            pass
                except requests.exceptions.RequestException:
                    pass
                logger.debug(f"Waiting for model at {endpoint}...")
                time.sleep(poll_interval)
            logger.warning(
                f"Timeout waiting for model at {endpoint} (framework={framework})"
            )
            return False

        # Try multiple health check endpoints (different frameworks use different ones)
        health_endpoints = [
            f"{endpoint}/health",
            f"{endpoint}/v1/models",
        ]

        start_time = time.time()
        while time.time() - start_time < timeout:
            for health_url in health_endpoints:
                try:
                    response = request_user_outbound(
                        "GET",
                        health_url,
                        user_id,
                        timeout=5,
                    )
                    if response.status_code == 200:
                        logger.info(f"Service at {endpoint} is ready")
                        return True
                except requests.exceptions.RequestException:
                    pass

            logger.debug(f"Waiting for service at {endpoint}...")
            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for service at {endpoint}")
        return False

    def list_inference_containers(self) -> list:
        """List all inference containers (xf-, vllm-, sglang- prefixes, plus shared Xinference)."""
        success, output = self._run_command([
            "docker", "ps", "-a", "--format", "{{.Names}}"
        ])
        if not success or not output.strip():
            return []

        containers = []
        shared_name = os.environ.get("XINFERENCE_CONTAINER_NAME", "xinference")
        for name in output.strip().split('\n'):
            name = name.strip()
            if name == shared_name or name.startswith(('xf-', 'vllm-', 'sglang-')):
                containers.append(name)
        return containers

    def list_running_containers(self) -> list:
        """List all running Docker containers."""
        success, output = self._run_command([
            "docker", "ps", "--format", "{{.Names}}"
        ])
        if not success or not output.strip():
            return []
        return [name.strip() for name in output.strip().split('\n') if name.strip()]

    def _get_container_pids(self, container_name: str) -> Set[int]:
        """Get host PIDs for a container using docker top."""
        success, output = self._run_command([
            "docker", "top", container_name, "-eo", "pid"
        ])
        if not success or not output.strip():
            return set()

        pids: Set[int] = set()
        lines = output.strip().split('\n')
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line.split()[0])
                pids.add(pid)
            except ValueError:
                continue
        return pids

    def get_container_gpu_usage(self, container_name: str, gpu_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get real-time GPU memory usage for a container.

        Args:
            container_name: Docker container name
            gpu_id: Optional GPU ID to get specific GPU's total memory for percentage calculation

        Returns:
            Dict with gpu_memory_used_mb and gpu_memory_used_percent, or None if not found.
        """
        try:
            # Get container's main PID
            success, output = self._run_command([
                "docker", "inspect", "--format", "{{.State.Pid}}", container_name
            ])
            if not success or not output.strip():
                return None

            container_pid = output.strip()
            if container_pid == "0":
                return None

            detail = self._get_container_gpu_usage_detail(container_name, container_pid)
            if not detail:
                return None

            used_by_gpu, gpu_total_memory = detail
            total_used_mb = sum(used_by_gpu.values())
            if total_used_mb <= 0:
                return None

            # Use specific gpu_id's total memory if provided, otherwise use container's actual GPU
            if gpu_id is not None and gpu_id in gpu_total_memory:
                total_gpu_memory = gpu_total_memory[gpu_id]
            elif used_by_gpu:
                total_gpu_memory = sum(gpu_total_memory.get(idx, 0) for idx in used_by_gpu.keys())
            else:
                total_gpu_memory = gpu_total_memory.get(0, 0)

            percent = (total_used_mb / total_gpu_memory * 100) if total_gpu_memory > 0 else 0
            return {
                "gpu_memory_used_mb": total_used_mb,
                "gpu_memory_used_percent": round(percent, 1),
            }
        except Exception as e:
            logger.debug(f"Failed to get GPU usage for {container_name}: {e}")
            return None

    def get_container_gpu_usage_by_gpu(self, container_name: str) -> Dict[int, Dict[str, Any]]:
        """
        Get per-GPU memory usage for a container.

        Returns:
            Dict mapping gpu_id -> usage info.
        """
        try:
            success, output = self._run_command([
                "docker", "inspect", "--format", "{{.State.Pid}}", container_name
            ])
            if not success or not output.strip():
                return {}

            container_pid = output.strip()
            if container_pid == "0":
                return {}

            detail = self._get_container_gpu_usage_detail(container_name, container_pid)
            if not detail:
                return {}

            used_by_gpu, gpu_total_memory = detail
            result: Dict[int, Dict[str, Any]] = {}
            for gpu_idx, used_mb in used_by_gpu.items():
                total_gpu_memory = gpu_total_memory.get(gpu_idx, 0)
                percent = (used_mb / total_gpu_memory * 100) if total_gpu_memory > 0 else 0
                result[gpu_idx] = {
                    "gpu_memory_used_mb": used_mb,
                    "gpu_memory_used_percent": round(percent, 1),
                }
            return result
        except Exception as e:
            logger.debug(f"Failed to get per-GPU usage for {container_name}: {e}")
            return {}

    def _get_container_gpu_usage_detail(
        self,
        container_name: str,
        container_pid: Optional[str] = None,
    ) -> Optional[Tuple[Dict[int, int], Dict[int, int]]]:
        """Return per-GPU used memory and total memory for container processes."""
        # Get container PID if not provided
        if not container_pid:
            success, output = self._run_command([
                "docker", "inspect", "--format", "{{.State.Pid}}", container_name
            ])
            if not success or not output.strip():
                return None
            container_pid = output.strip()
            if container_pid == "0":
                return None

        # Get container's cgroup ID (cgroup v1/v2) or fallback to docker top
        cgroup_id = None
        container_pids: Set[int] = set()
        try:
            with open(f"/proc/{container_pid}/cgroup", "r") as f:
                cgroup_content = f.read()
            import re
            # Common patterns:
            # - /docker/<id>
            # - /system.slice/docker-<id>.scope
            # - /kubepods/.../<id>
            match = re.search(r'/docker/([0-9a-f]{12,64})', cgroup_content)
            if not match:
                match = re.search(r'docker-([0-9a-f]{12,64})\\.scope', cgroup_content)
            if not match:
                match = re.search(r'/kubepods[^/]+/pod[^/]+/([0-9a-f]{12,64})', cgroup_content)
            if match:
                cgroup_id = match.group(1)[:12]
        except (FileNotFoundError, PermissionError):
            cgroup_id = None

        # Fallback: use docker top to list container PIDs when cgroup is unavailable
        if not cgroup_id:
            container_pids = self._get_container_pids(container_name)

        # Get GPU processes and their memory usage (include GPU index)
        success, output = self._run_command([
            "nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory",
            "--format=csv,noheader,nounits"
        ])
        if not success or not output.strip():
            return None

        # Get per-GPU total memory for percentage calculation
        success_mem, output_mem = self._run_command([
            "nvidia-smi", "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits"
        ])
        gpu_total_memory: Dict[int, int] = {}
        if success_mem and output_mem.strip():
            for line in output_mem.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 2:
                    idx = int(parts[0].strip())
                    mem = int(parts[1].strip())
                    gpu_total_memory[idx] = mem

        # Get GPU index to bus_id mapping
        success_bus, output_bus = self._run_command([
            "nvidia-smi", "--query-gpu=index,gpu_bus_id",
            "--format=csv,noheader"
        ])
        bus_to_index: Dict[str, int] = {}
        if success_bus and output_bus.strip():
            for line in output_bus.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 2:
                    idx = int(parts[0].strip())
                    bus_id = parts[1].strip()
                    bus_to_index[bus_id] = idx

        # Find processes belonging to this container
        used_by_gpu: Dict[int, int] = {}
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                bus_id = parts[0].strip()
                pid = parts[1].strip()
                mem_mb = int(parts[2].strip())

                # Check if this PID belongs to our container
                try:
                    if cgroup_id:
                        with open(f"/proc/{pid}/cgroup", "r") as f:
                            pid_cgroup = f.read()
                        if cgroup_id not in pid_cgroup:
                            continue
                    elif container_pids:
                        if int(pid) not in container_pids:
                            continue
                    else:
                        continue

                    gpu_idx = bus_to_index.get(bus_id)
                    if gpu_idx is None:
                        continue
                    used_by_gpu[gpu_idx] = used_by_gpu.get(gpu_idx, 0) + mem_mb
                except (FileNotFoundError, PermissionError, ValueError):
                    continue

        if not used_by_gpu:
            return None

        return used_by_gpu, gpu_total_memory

    def get_container_model_gpu_usage(self, container_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get per-model GPU usage for a container (model processes only).

        Returns:
            Dict mapping model_uid -> usage info.
        """
        try:
            container_pids = self._get_container_pids(container_name)
            if not container_pids:
                return {}

            success, output = self._run_command([
                "nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
                "--format=csv,noheader,nounits"
            ])
            if not success or not output.strip():
                return {}

            success_mem, output_mem = self._run_command([
                "nvidia-smi", "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits"
            ])
            gpu_total_memory: Dict[int, int] = {}
            if success_mem and output_mem.strip():
                for line in output_mem.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        idx = int(parts[0].strip())
                        mem = int(parts[1].strip())
                        gpu_total_memory[idx] = mem

            success_bus, output_bus = self._run_command([
                "nvidia-smi", "--query-gpu=index,gpu_bus_id",
                "--format=csv,noheader"
            ])
            bus_to_index: Dict[str, int] = {}
            if success_bus and output_bus.strip():
                for line in output_bus.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        idx = int(parts[0].strip())
                        bus_id = parts[1].strip()
                        bus_to_index[bus_id] = idx

            def _extract_model_uid_from_name(name: str) -> Optional[str]:
                if not name:
                    return None
                import re
                cleaned = name
                if "Model:" in name:
                    cleaned = name.split("Model:", 1)[1].strip()
                else:
                    cleaned = name.strip()
                    # Only accept names with a UID-like suffix to avoid noise
                    if not re.search(r"-[0-9a-f]{8}(?:-\d+)?$", cleaned, re.IGNORECASE):
                        return None
                cleaned = cleaned.strip()
                if not cleaned:
                    return None
                cleaned = re.sub(r"-\d+$", "", cleaned)
                if len(cleaned) < 8:
                    return None
                return cleaned

            def _extract_model_uid_from_cmdline(pid: int) -> Optional[str]:
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        raw = f.read()
                except (FileNotFoundError, PermissionError):
                    return None
                if not raw:
                    return None
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    return None
                parts = [p for p in text.split("\0") if p]
                for i, part in enumerate(parts):
                    if part in ("--model-uid", "--model_uid"):
                        if i + 1 < len(parts):
                            return parts[i + 1]
                    if part.startswith("--model-uid=") or part.startswith("--model_uid="):
                        return part.split("=", 1)[1]
                if parts:
                    return _extract_model_uid_from_name(parts[0])
                return None

            def _read_ppid(pid: int) -> Optional[int]:
                try:
                    with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.startswith("PPid:"):
                                _, value = line.split(":", 1)
                                return int(value.strip())
                except (FileNotFoundError, PermissionError, ValueError):
                    return None
                return None

            def _extract_model_uid_from_comm(pid: int) -> Optional[str]:
                try:
                    with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="ignore") as f:
                        name = f.read().strip()
                        return _extract_model_uid_from_name(name)
                except (FileNotFoundError, PermissionError):
                    return None

            def _extract_model_uid_from_pid(pid: int) -> Optional[str]:
                uid = _extract_model_uid_from_cmdline(pid)
                if uid:
                    return uid
                return _extract_model_uid_from_comm(pid)

            def _find_model_uid_from_parents(pid: int, max_depth: int = 6) -> Optional[str]:
                current = pid
                visited = set()
                for _ in range(max_depth):
                    if current in visited:
                        break
                    visited.add(current)
                    ppid = _read_ppid(current)
                    if not ppid or ppid == 1:
                        break
                    if ppid not in container_pids:
                        break
                    uid = _extract_model_uid_from_pid(ppid)
                    if uid:
                        return uid
                    current = ppid
                return None

            usage_by_model: Dict[str, Dict[str, Any]] = {}
            for line in output.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 4:
                    continue
                bus_id, pid_raw, process_name, mem_raw = parts
                try:
                    pid = int(pid_raw)
                except ValueError:
                    continue
                if pid not in container_pids:
                    continue
                model_uid = _extract_model_uid_from_name(process_name)
                if not model_uid:
                    model_uid = _extract_model_uid_from_cmdline(pid)
                if not model_uid:
                    model_uid = _find_model_uid_from_parents(pid)
                if not model_uid:
                    continue
                try:
                    mem_mb = int(float(mem_raw))
                except ValueError:
                    continue
                gpu_idx = bus_to_index.get(bus_id)
                if gpu_idx is None:
                    continue

                entry = usage_by_model.setdefault(model_uid, {"gpu_ids": set(), "used_mb": 0})
                entry["gpu_ids"].add(gpu_idx)
                entry["used_mb"] += mem_mb

            result: Dict[str, Dict[str, Any]] = {}
            for model_uid, entry in usage_by_model.items():
                gpu_ids = sorted(entry["gpu_ids"])
                used_mb = entry["used_mb"]
                if not gpu_ids:
                    continue
                if len(gpu_ids) == 1:
                    gpu_id = gpu_ids[0]
                    total_mem = gpu_total_memory.get(gpu_id, 0)
                else:
                    gpu_id = None
                    total_mem = sum(gpu_total_memory.get(g, 0) for g in gpu_ids)
                percent = (used_mb / total_mem * 100) if total_mem > 0 else 0
                result[model_uid] = {
                    "gpu_id": gpu_id,
                    "gpu_ids": gpu_ids,
                    "gpu_memory_used_mb": used_mb,
                    "gpu_memory_used_percent": round(percent, 1),
                }

            return result
        except Exception as e:
            logger.debug(f"Failed to get model GPU usage for {container_name}: {e}")
            return {}

    def get_all_container_gpu_usage(self) -> Dict[str, Dict[str, Any]]:
        """
        Get GPU usage for all running containers.

        Returns:
            Dict mapping container_name to usage info.
        """
        result = {}
        containers = self.list_running_containers()
        for container_name in containers:
            usage = self.get_container_gpu_usage(container_name)
            if usage:
                result[container_name] = usage
        return result

    def get_model_gpu_usage(self, container_name: str, model_uid: str) -> Optional[Dict[str, Any]]:
        """
        Get GPU memory usage for a specific model within a shared container.

        For shared containers (like Xinference), multiple models run in the same container
        but may use different GPUs. This method matches the model_uid in nvidia-smi process
        names to get per-model GPU usage.

        Process name format: "Model: {model_name}-{model_uid[:8]}-{replica}"

        Args:
            container_name: Docker container name
            model_uid: Model UID to match (e.g., "Qwen3-Reranker-0.6B-19969540")

        Returns:
            Dict with gpu_id, gpu_memory_used_mb, gpu_memory_used_percent, or None if not found.
        """
        import re

        try:
            # Get container's main PID to verify container exists
            success, output = self._run_command([
                "docker", "inspect", "--format", "{{.State.Pid}}", container_name
            ])
            if not success or not output.strip() or output.strip() == "0":
                return None

            # Get GPU processes with process names
            success, output = self._run_command([
                "nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
                "--format=csv,noheader,nounits"
            ])
            if not success or not output.strip():
                return None

            # Get per-GPU total memory
            success_mem, output_mem = self._run_command([
                "nvidia-smi", "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits"
            ])
            gpu_total_memory = {}
            if success_mem and output_mem.strip():
                for line in output_mem.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        idx = int(parts[0].strip())
                        mem = int(parts[1].strip())
                        gpu_total_memory[idx] = mem

            # Get GPU index to bus_id mapping
            success_bus, output_bus = self._run_command([
                "nvidia-smi", "--query-gpu=index,gpu_bus_id",
                "--format=csv,noheader"
            ])
            bus_to_index = {}
            if success_bus and output_bus.strip():
                for line in output_bus.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        idx = int(parts[0].strip())
                        bus_id = parts[1].strip()
                        bus_to_index[bus_id] = idx

            # Extract short model_uid for matching (first 8 chars after last hyphen section)
            # model_uid format: "ModelName-xxxxxxxx" -> match "xxxxxxxx"
            uid_match = re.search(r'-([0-9a-f]{8})(?:-\d+)?$', model_uid, re.IGNORECASE)
            if uid_match:
                short_uid = uid_match.group(1)
            else:
                # Fallback: use last 8 chars
                short_uid = model_uid[-8:] if len(model_uid) >= 8 else model_uid

            # Find process matching this model_uid
            for line in output.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    bus_id = parts[0].strip()
                    process_name = parts[2].strip()
                    mem_mb = int(parts[3].strip())

                    # Match model_uid in process name
                    # Process name format: "Model: Qwen3-Reranker-0.6B-19969540-0"
                    if short_uid.lower() in process_name.lower():
                        gpu_id = bus_to_index.get(bus_id)
                        total_mem = gpu_total_memory.get(gpu_id, 0) if gpu_id is not None else 0
                        percent = (mem_mb / total_mem * 100) if total_mem > 0 else 0

                        return {
                            "gpu_id": gpu_id,
                            "gpu_memory_used_mb": mem_mb,
                            "gpu_memory_used_percent": round(percent, 1),
                        }

            return None
        except Exception as e:
            logger.debug(f"Failed to get model GPU usage for {model_uid} in {container_name}: {e}")
            return None


# Global instance
docker_deployer = DockerDeployer()
