import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from train_factory.api.routes import resource_routes
from train_factory.monitoring.resource_monitor import ResourceUsage


PROCESS_DETAILS = [
    {
        "gpu_index": 0,
        "name": "GPU",
        "uuid": "GPU-secret-uuid",
        "total_memory_mb": 24576,
        "processes": [
            {
                "pid": 4242,
                "name": "python",
                "cmdline": "python train.py --secret value",
                "user": "host-user",
                "gpu_memory_mb": 1024,
                "container_id": "abcdef1234567890",
                "container_name": "other-tenant-model",
                "started_at": "2026-08-09T20:00:00",
            }
        ],
    }
]


def _set_auth(monkeypatch, enabled=True):
    monkeypatch.setattr(
        resource_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=enabled),
        raising=False,
    )


def test_regular_user_receives_redacted_gpu_process_details(monkeypatch):
    _set_auth(monkeypatch)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_gpu_processes",
        lambda: PROCESS_DETAILS,
    )

    response = asyncio.run(
        resource_routes.get_gpu_processes(
            current_user={"user_id": "user-1", "is_admin": False}
        )
    )

    process = response["gpus"][0]["processes"][0]
    assert process == {"gpu_memory_mb": 1024}
    assert response["gpus"][0]["uuid"] is None


def test_administrator_receives_full_gpu_process_details(monkeypatch):
    _set_auth(monkeypatch)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_gpu_processes",
        lambda: PROCESS_DETAILS,
    )

    response = asyncio.run(
        resource_routes.get_gpu_processes(
            current_user={"user_id": "admin-1", "is_admin": True}
        )
    )

    assert response["gpus"][0]["processes"][0]["pid"] == 4242
    assert response["gpus"][0]["processes"][0]["cmdline"].startswith("python")


def test_regular_user_does_not_receive_allocated_task_ids(monkeypatch):
    _set_auth(monkeypatch)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_gpu_info_simple",
        lambda: [{"id": 0, "is_allocated": True, "allocated_task": "task-secret"}],
    )
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_resource_status",
        lambda: {
            "gpu_details": {0: {"status": "allocated", "task_id": "task-secret"}}
        },
    )
    monkeypatch.setattr(
        resource_routes,
        "get_resource_monitor",
        lambda: SimpleNamespace(
            get_current_usage=lambda: SimpleNamespace(
                cpu_percent=1,
                memory_percent=2,
                memory_used_mb=3,
                memory_total_mb=4,
                memory_limit_set=False,
                disk_usage_percent=5,
                disk_used_gb=6,
                disk_total_gb=7,
                disk_mountpoint="/",
                open_files=8,
                thread_count=9,
            )
        ),
    )
    user = {"user_id": "user-1", "is_admin": False}

    gpu_list = asyncio.run(resource_routes.get_gpu_list(current_user=user))
    status = asyncio.run(resource_routes.get_resource_status(current_user=user))

    assert gpu_list["gpus"][0]["allocated_task"] is None
    assert status["gpu"]["gpu_details"][0]["task_id"] is None


def test_regular_user_cannot_trigger_global_resource_cleanup(monkeypatch):
    _set_auth(monkeypatch)
    monkeypatch.setattr(
        resource_routes,
        "get_resource_monitor",
        lambda: (_ for _ in ()).throw(AssertionError("cleanup side effect ran")),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            resource_routes.cleanup_resources(
                current_user={"user_id": "user-1", "is_admin": False}
            )
        )

    assert exc_info.value.status_code == 403


def test_regular_user_does_not_receive_configured_storage_paths(monkeypatch):
    _set_auth(monkeypatch)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_resource_status",
        lambda: {
            "total_gpus": 0,
            "allocated_gpus": 0,
            "free_gpus": 0,
            "gpu_details": {},
        },
    )
    monkeypatch.setattr(
        resource_routes,
        "get_resource_monitor",
        lambda: SimpleNamespace(
            get_current_usage=lambda: ResourceUsage(
                cpu_percent=1.0,
                memory_percent=2.0,
                memory_used_mb=3.0,
                memory_total_mb=4.0,
                disk_usage_percent=75.0,
                disk_used_gb=75.0,
                disk_total_gb=100.0,
                disk_mountpoint="/host/private/storage",
                disk_volumes=[
                    {
                        "scope": "datasets",
                        "scopes": ["datasets"],
                        "paths": ["/host/private/storage/datasets"],
                        "mountpoint": "/host/private/storage",
                        "usage_percent": 75.0,
                        "used_gb": 75.0,
                        "total_gb": 100.0,
                        "free_gb": 25.0,
                    }
                ],
                open_files=8,
                thread_count=9,
            )
        ),
    )

    response = asyncio.run(
        resource_routes.get_resource_status(
            current_user={"user_id": "user-1", "is_admin": False}
        )
    )

    assert response["system"]["disk_mountpoint"] is None
    assert response["system"]["disk_volumes"][0]["mountpoint"] is None
    assert "paths" not in response["system"]["disk_volumes"][0]
    assert "/host/private" not in repr(response)


def _healthy_system_usage():
    return ResourceUsage(
        cpu_percent=1.0,
        memory_percent=2.0,
        memory_used_mb=3.0,
        memory_total_mb=4.0,
        disk_usage_percent=5.0,
        disk_used_gb=6.0,
        disk_total_gb=7.0,
        open_files=8,
        thread_count=9,
    )


def test_resource_status_distinguishes_gpu_monitor_failure_from_no_devices(monkeypatch):
    _set_auth(monkeypatch, enabled=False)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_resource_status",
        lambda: (_ for _ in ()).throw(RuntimeError("NVML failed")),
    )
    monkeypatch.setattr(
        resource_routes,
        "get_resource_monitor",
        lambda: SimpleNamespace(get_current_usage=_healthy_system_usage),
    )

    response = asyncio.run(
        resource_routes.get_resource_status(current_user={"user_id": "user-1"})
    )

    assert response["available"] is True
    assert response["partial"] is True
    assert response["gpu"]["available"] is False
    assert response["gpu"]["device_state"] == "unknown"
    assert response["gpu"]["total_gpus"] is None
    assert {error["scope"] for error in response["errors"]} == {"gpu"}


def test_resource_status_marks_zero_gpu_as_available_no_devices(monkeypatch):
    _set_auth(monkeypatch, enabled=False)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_resource_status",
        lambda: {
            "total_gpus": 0,
            "allocated_gpus": 0,
            "free_gpus": 0,
            "gpu_details": {},
        },
    )
    monkeypatch.setattr(
        resource_routes,
        "get_resource_monitor",
        lambda: SimpleNamespace(get_current_usage=_healthy_system_usage),
    )

    response = asyncio.run(
        resource_routes.get_resource_status(current_user={"user_id": "user-1"})
    )

    assert response["partial"] is False
    assert response["gpu"]["available"] is True
    assert response["gpu"]["device_state"] == "no_devices"
    assert response["gpu"]["error_code"] is None


def test_resource_status_does_not_expose_raw_gpu_monitor_errors(monkeypatch):
    _set_auth(monkeypatch, enabled=False)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_resource_status",
        lambda: {
            "total_gpus": 1,
            "allocated_gpus": 0,
            "free_gpus": 1,
            "gpu_details": {
                0: {
                    "status": "free",
                    "task_id": None,
                    "error": "NVML path /host/secret failed with token=abc",
                }
            },
        },
    )
    monkeypatch.setattr(
        resource_routes,
        "get_resource_monitor",
        lambda: SimpleNamespace(get_current_usage=_healthy_system_usage),
    )

    response = asyncio.run(
        resource_routes.get_resource_status(current_user={"user_id": "user-1"})
    )

    detail = response["gpu"]["gpu_details"][0]
    assert "error" not in detail
    assert detail["error_code"] == "gpu_metrics_unavailable"
    assert response["gpu"]["partial"] is True


def test_resource_status_is_partial_when_system_monitor_raises(monkeypatch):
    _set_auth(monkeypatch, enabled=False)
    monkeypatch.setattr(
        resource_routes.gpu_resource_manager,
        "get_resource_status",
        lambda: {
            "total_gpus": 0,
            "allocated_gpus": 0,
            "free_gpus": 0,
            "gpu_details": {},
        },
    )
    monkeypatch.setattr(
        resource_routes,
        "get_resource_monitor",
        lambda: SimpleNamespace(
            get_current_usage=lambda: (_ for _ in ()).throw(
                RuntimeError("psutil platform failure")
            )
        ),
    )

    response = asyncio.run(
        resource_routes.get_resource_status(current_user={"user_id": "user-1"})
    )

    assert response["available"] is True
    assert response["partial"] is True
    assert response["system"]["available"] is False
    assert response["system"]["cpu_percent"] is None
    assert response["system"]["error_code"] == "resource_monitor_unavailable"
    assert {error["scope"] for error in response["errors"]} == {"system"}
