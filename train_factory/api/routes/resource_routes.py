"""资源监控 API 路由"""
import copy
import logging

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from ...auth.dependencies import get_current_user
from ...config.settings import get_settings
from ...core.gpu_resource_manager import gpu_resource_manager
from ...monitoring.resource_monitor import get_resource_monitor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/resources",
    tags=["resources"],
    dependencies=[Depends(get_current_user)],
)


def _can_view_host_process_details(current_user: Dict[str, Any]) -> bool:
    return not get_settings().auth_enabled or bool(current_user.get("is_admin"))


def _redact_resource_status(
    resource_status: Dict[str, Any],
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    if _can_view_host_process_details(current_user):
        return resource_status
    redacted = copy.deepcopy(resource_status)
    for detail in (redacted.get("gpu_details") or {}).values():
        if isinstance(detail, dict):
            detail["task_id"] = None
    return redacted


def _unavailable_gpu_status() -> Dict[str, Any]:
    return {
        "available": False,
        "partial": False,
        "error_code": "gpu_monitor_unavailable",
        "device_state": "unknown",
        "total_gpus": None,
        "allocated_gpus": None,
        "free_gpus": None,
        "gpu_details": {},
    }


def _get_gpu_status(current_user: Dict[str, Any]) -> Dict[str, Any]:
    try:
        status = _redact_resource_status(
            gpu_resource_manager.get_resource_status(),
            current_user,
        )
    except Exception:
        logger.exception("GPU resource monitoring failed")
        return _unavailable_gpu_status()

    total_gpus = status.get("total_gpus")
    detail_errors = []
    for detail in (status.get("gpu_details") or {}).values():
        if isinstance(detail, dict) and detail.get("error"):
            detail_errors.append(detail)
            detail.pop("error", None)
            detail["error_code"] = "gpu_metrics_unavailable"
    status["available"] = True
    status["partial"] = bool(detail_errors)
    status["error_code"] = "partial_gpu_data" if detail_errors else None
    status["device_state"] = "no_devices" if total_gpus == 0 else "detected"
    return status


def _serialize_system_usage(
    system_usage: Any,
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    errors = list(getattr(system_usage, "errors", []) or [])
    available = bool(getattr(system_usage, "available", True))
    partial = bool(getattr(system_usage, "partial", False))
    error_code = getattr(system_usage, "error_code", None)
    can_view_storage_paths = _can_view_host_process_details(current_user)
    disk_volumes = []
    for raw_volume in list(getattr(system_usage, "disk_volumes", []) or []):
        if not isinstance(raw_volume, dict):
            continue
        volume = dict(raw_volume)
        if not can_view_storage_paths:
            volume.pop("paths", None)
            volume["mountpoint"] = None
        disk_volumes.append(volume)
    return {
        "available": available,
        "partial": partial,
        "error_code": error_code,
        "errors": errors,
        "cpu_percent": system_usage.cpu_percent,
        "memory_percent": system_usage.memory_percent,
        "memory_used_mb": system_usage.memory_used_mb,
        "memory_total_mb": system_usage.memory_total_mb,
        "memory_limit_set": getattr(system_usage, "memory_limit_set", False),
        "disk_usage_percent": system_usage.disk_usage_percent,
        "disk_used_gb": system_usage.disk_used_gb,
        "disk_total_gb": system_usage.disk_total_gb,
        "disk_mountpoint": (
            getattr(system_usage, "disk_mountpoint", None)
            if can_view_storage_paths
            else None
        ),
        "disk_volumes": disk_volumes,
        "open_files": system_usage.open_files,
        "thread_count": system_usage.thread_count,
    }


def _unavailable_system_status() -> Dict[str, Any]:
    return {
        "available": False,
        "partial": False,
        "error_code": "resource_monitor_unavailable",
        "errors": [
            {"scope": "system", "error_code": "resource_monitor_unavailable"}
        ],
        "cpu_percent": None,
        "memory_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": None,
        "memory_limit_set": False,
        "disk_usage_percent": None,
        "disk_used_gb": None,
        "disk_total_gb": None,
        "disk_mountpoint": None,
        "disk_volumes": [],
        "open_files": None,
        "thread_count": None,
    }


@router.get("/status")
async def get_resource_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    获取系统资源状态

    返回 GPU、CPU、内存等资源的详细状态信息
    """
    gpu_status = _get_gpu_status(current_user)

    # 获取系统资源使用情况
    resource_monitor = get_resource_monitor()
    try:
        system_usage = resource_monitor.get_current_usage()
        system_status = _serialize_system_usage(system_usage, current_user)
    except Exception:
        logger.exception("System resource monitoring failed")
        system_status = _unavailable_system_status()

    errors = list(system_status["errors"])
    if not gpu_status["available"]:
        errors.append({"scope": "gpu", "error_code": gpu_status["error_code"]})
    elif gpu_status["partial"]:
        errors.append({"scope": "gpu", "error_code": "partial_gpu_data"})
    available = bool(gpu_status["available"] or system_status["available"])
    partial = bool(
        gpu_status["partial"]
        or system_status["partial"]
        or not gpu_status["available"]
        or not system_status["available"]
    )

    return {
        "available": available,
        "partial": partial,
        "error_code": (
            "partial_resource_data"
            if partial and available
            else "resource_monitor_unavailable"
            if not available
            else None
        ),
        "errors": errors,
        "gpu": gpu_status,
        "system": system_status,
    }


@router.get("/gpus")
async def get_gpu_list(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    获取 GPU 列表信息

    返回简化的 GPU 信息列表
    """
    gpus = gpu_resource_manager.get_gpu_info_simple()
    if not _can_view_host_process_details(current_user):
        gpus = [{**gpu, "allocated_task": None} for gpu in gpus]
    return {
        "total": len(gpus),
        "gpus": gpus
    }


@router.get("/gpus/processes")
async def get_gpu_processes(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    获取 GPU 进程占用信息

    返回按 GPU 分组的进程列表
    """
    processes = gpu_resource_manager.get_gpu_processes()
    if not _can_view_host_process_details(current_user):
        processes = [
            {
                **{key: value for key, value in gpu.items() if key != "processes"},
                "uuid": None,
                "processes": [
                    {"gpu_memory_mb": process.get("gpu_memory_mb")}
                    for process in gpu.get("processes", [])
                ],
            }
            for gpu in processes
        ]
    return {
        "total": len(processes),
        "gpus": processes
    }


@router.post("/cleanup")
async def cleanup_resources(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    清理系统资源

    执行垃圾回收和资源清理
    """
    if get_settings().auth_enabled and not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator access required")

    resource_monitor = get_resource_monitor()
    result = resource_monitor.cleanup_resources()

    # 清理过期的 GPU 分配
    stale_count = gpu_resource_manager.cleanup_stale_allocations()
    result["stale_gpu_allocations_cleaned"] = stale_count

    return result
