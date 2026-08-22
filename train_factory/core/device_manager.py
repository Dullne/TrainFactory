"""
Device management for training.

Handles GPU detection, configuration, and device allocation.
"""

import os
import logging

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device configuration and GPU setup for training."""

    @staticmethod
    def cleanup_gpu_environment():
        """
        Clean up GPU environment, removing potential residual state.
        Call before training starts to ensure a clean GPU environment.
        """
        try:
            if torch.cuda.is_available():
                logger.info("Cleaning up GPU environment...")

                # Clear all GPU caches
                torch.cuda.empty_cache()

                # Collect IPC memory
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

                # Reset CUDA context if possible
                try:
                    torch.cuda.reset_accumulated_memory_stats()
                except Exception:
                    pass

                logger.info(f"GPU environment cleanup complete, visible GPUs: {torch.cuda.device_count()}")
            else:
                logger.info("CPU mode, skipping GPU environment cleanup")

        except Exception as e:
            logger.error(f"GPU environment cleanup failed: {e}")

    @staticmethod
    def get_training_device(user_device=None) -> str:
        """
        Get training device configuration.

        Args:
            user_device: User-specified device configuration

        Returns:
            str: Device string, 'cuda' or 'cpu'
        """
        requested_device = str(user_device or "").strip().lower()
        if requested_device == "cpu":
            logger.info("Explicit CPU device requested for training")
            return "cpu"

        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")

        if cuda_visible:
            if torch.cuda.is_available():
                visible_gpus = cuda_visible.split(",")
                gpu_count = torch.cuda.device_count()
                logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
                logger.info(f"Using {len(visible_gpus)} specified GPUs for training")

                # Display detailed info for each visible GPU
                for i in range(gpu_count):
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        original_id = visible_gpus[i] if i < len(visible_gpus) else "unknown"
                        logger.info(f"   GPU {i} (original GPU{original_id}): {gpu_name} ({gpu_memory:.1f}GB)")
                    except Exception as e:
                        logger.info(f"   GPU {i}: Failed to get info - {e}")
                return "cuda"
            else:
                logger.warning("CUDA_VISIBLE_DEVICES set but system doesn't support CUDA, falling back to CPU")
                return "cpu"
        else:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"No CUDA_VISIBLE_DEVICES set, detected {gpu_count} available GPUs")
                for i in range(gpu_count):
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                    except Exception:
                        logger.info(f"   GPU {i}: Failed to get info")
                return "cuda"
            else:
                logger.info("System doesn't support CUDA, using CPU for training")
                return "cpu"

    @staticmethod
    def prepare_model_for_training(model, device: str):
        """
        Prepare model for training.

        Args:
            model: Model to prepare
            device: Device configuration string, 'cuda' or 'cpu'

        Returns:
            Prepared model
        """
        try:
            # Check and remove any DataParallel wrapper
            if hasattr(model, '_modules'):
                for module_name, module in model._modules.items():
                    if isinstance(module, torch.nn.DataParallel):
                        original_module = module.module
                        model._modules[module_name] = original_module
                        logger.info(f"Removed DataParallel wrapper from module {module_name}")

            logger.info("Model training preparation complete")
            return model

        except Exception as e:
            logger.error(f"Model preparation failed: {str(e)}")
            logger.warning("Continuing with original model configuration")
            return model

    @staticmethod
    def get_gpu_count(device_config=None) -> int:
        """
        Get the number of available GPUs.

        Args:
            device_config: Device configuration string, e.g., 'cuda:0,1'

        Returns:
            int: Number of GPUs
        """
        if not torch.cuda.is_available():
            return 0

        if device_config and ',' in str(device_config):
            return len(str(device_config).split(','))
        else:
            return torch.cuda.device_count()

    @staticmethod
    def check_gpu_bf16_support() -> bool:
        """
        Check if GPU supports bf16.

        Returns:
            bool: True if bf16 is supported
        """
        if not torch.cuda.is_available():
            return False

        try:
            device = torch.cuda.current_device()
            gpu_capability = torch.cuda.get_device_capability(device)
            # Ampere architecture (compute capability 8.0+) fully supports bf16
            return gpu_capability[0] >= 8
        except Exception:
            return False

    @staticmethod
    def get_gpu_memory_info() -> list:
        """
        Get memory information for all available GPUs.
        Uses pynvml for detailed info, falls back to torch.cuda.

        Returns:
            list: List of dicts with GPU memory info
        """
        gpu_info = []

        # Try pynvml first for more detailed info
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    # Get GPU name
                    try:
                        name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                    except Exception:
                        name = f"GPU {i}"

                    # Get utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                        mem_util = util.memory
                    except Exception:
                        gpu_util = None
                        mem_util = None

                    # Get temperature
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        temperature = None

                    # Get power
                    try:
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    except Exception:
                        power_usage = None
                        power_limit = None

                    gpu_info.append({
                        "index": i,
                        "name": name,
                        "total_memory_gb": mem_info.total / 1024**3,
                        "used_memory_gb": mem_info.used / 1024**3,
                        "free_memory_gb": mem_info.free / 1024**3,
                        "memory_usage_percent": round((mem_info.used / mem_info.total) * 100, 1) if mem_info.total > 0 else 0,
                        "gpu_utilization_percent": gpu_util,
                        "memory_utilization_percent": mem_util,
                        "temperature_c": temperature,
                        "power_usage_w": power_usage,
                        "power_limit_w": power_limit,
                    })
                except Exception as e:
                    gpu_info.append({
                        "index": i,
                        "error": str(e),
                    })
            return gpu_info

        except Exception:
            pass

        # Fallback to torch.cuda
        if not torch.cuda.is_available():
            return []

        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                reserved = torch.cuda.memory_reserved(i)
                gpu_info.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1024**3,
                    "used_memory_gb": reserved / 1024**3,
                    "free_memory_gb": (props.total_memory - reserved) / 1024**3,
                    "memory_usage_percent": round((reserved / props.total_memory) * 100, 1) if props.total_memory > 0 else 0,
                })
            except Exception as e:
                gpu_info.append({
                    "index": i,
                    "error": str(e),
                })
        return gpu_info
