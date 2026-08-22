"""Common utility functions for TrainFactory."""

import sys
import logging
from typing import Dict, Any, Optional

from loguru import logger as loguru_logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure loguru
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    if log_file:
        loguru_logger.add(
            log_file,
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            retention="7 days"
        )


def init_swanlab(training_config: Dict[str, Any]) -> bool:
    """
    Initialize SwanLab for experiment tracking.

    Args:
        training_config: Training configuration dictionary

    Returns:
        True if initialization successful
    """
    try:
        import swanlab

        api_key = training_config.get('swanlab_api_key')
        if not api_key:
            loguru_logger.warning("SwanLab API key not provided, skipping initialization")
            return False

        project = training_config.get('swanlab_project', 'train-factory')
        experiment_name = training_config.get('task_name') or training_config.get('task_id', 'unnamed')

        swanlab.init(
            project=project,
            experiment_name=experiment_name,
            config=training_config
        )

        loguru_logger.info(f"SwanLab initialized: project={project}, experiment={experiment_name}")
        return True

    except ImportError:
        loguru_logger.warning("SwanLab not installed, skipping initialization")
        return False
    except Exception as e:
        loguru_logger.error(f"SwanLab initialization failed: {e}")
        return False


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.

    Returns:
        Dictionary with GPU information
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "devices": []}

        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}"
            })

        return {
            "available": True,
            "count": len(devices),
            "devices": devices
        }

    except Exception as e:
        return {"available": False, "error": str(e)}
