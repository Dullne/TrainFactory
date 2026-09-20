"""Long generation workers must leave deployment control capacity available."""

import asyncio
import threading
import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest

from train_factory.api.routes import deployment_routes
from train_factory.storage.services.background_task_admission_service import BackgroundTaskAdmissionService


def test_generation_cannot_starve_deployment_stop_on_small_default_pool(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=1, per_user_limit=1)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})
    _, lease = service.admit_execution(
        "generation", "long-generation", "alice", lambda: True,
    )
    entered, release = threading.Event(), threading.Event()
    deployment = {
        "deployment_id": "owned", "user_id": "alice", "model_id": "model",
        "xinference_endpoint": "https://example.com", "replica": 1,
        "gpu_memory_utilization": 0.8, "inference_framework": "vllm",
        "status": "stopped", "deploy_mode": "container",
    }
    monkeypatch.setattr(deployment_routes.deployment_service, "get_deployment", lambda _id: deployment)
    monkeypatch.setattr(deployment_routes, "_validate_deployment_endpoint", lambda _deployment: None)

    def stop(deployment_id, *, user_id):
        assert (deployment_id, user_id) == ("owned", "alice")
        return deployment

    monkeypatch.setattr(deployment_routes.deployment_service, "stop_deployment", stop)

    async def generate():
        entered.set()
        release.wait(3)

    async def exercise():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        worker = asyncio.create_task(service.run_async(lease, generate))
        try:
            async with asyncio.timeout(2):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            result = await asyncio.wait_for(deployment_routes.stop_deployment(
                "owned", {"user_id": "alice", "role": "user"},
            ), 0.5)
            assert result.status == "stopped"
            assert service.is_executing("generation", "long-generation")
        finally:
            release.set()
            await worker

    try:
        asyncio.run(exercise())
    finally:
        service.shutdown_async_workers()


def test_shutdown_cancels_queued_work_but_retains_running_lease_until_exit(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=2, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})
    leases = [service.admit_execution("generation", key, "alice", lambda: True)[1]
              for key in ("running", "queued")]
    service._async_executor = ThreadPoolExecutor(max_workers=1)
    entered, release, shutdown_started = threading.Event(), threading.Event(), threading.Event()
    executed = []

    async def running():
        entered.set()
        release.wait(3)

    async def queued():
        executed.append("queued")

    def shutdown():
        shutdown_started.set()
        service.shutdown_async_workers()

    async def exercise():
        worker = asyncio.create_task(service.run_async(leases[0], running))
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        waiter = asyncio.create_task(service.run_async(leases[1], queued))
        await asyncio.sleep(0)
        closing = asyncio.create_task(asyncio.to_thread(shutdown))
        try:
            async with asyncio.timeout(2):
                while not shutdown_started.is_set():
                    await asyncio.sleep(0.001)
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            assert not service.is_executing("generation", "queued")
            assert service.is_executing("generation", "running")
            assert not closing.done()
            assert not executed
        finally:
            release.set()
            await worker
            await closing
        assert not service.is_executing("generation", "running")
        _, rejected_lease = service.admit_execution("generation", "rejected", "alice", lambda: True)
        with pytest.raises(RuntimeError, match="shutting down"):
            await service.run_async(rejected_lease, queued)
        assert not service.is_executing("generation", "rejected")
        service.start_async_workers()
        _, restarted_lease = service.admit_execution("generation", "restarted", "alice", lambda: True)
        await service.run_async(restarted_lease, queued)
        assert executed == ["queued"]

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        service.shutdown_async_workers()


def test_generation_worker_keeps_request_context_without_leaking_between_jobs(monkeypatch):
    identity = contextvars.ContextVar("executor_test_identity")
    service = BackgroundTaskAdmissionService(global_limit=1, per_user_limit=1)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})

    async def operation():
        previous = identity.get()
        identity.set("worker-only")
        return previous

    async def exercise():
        for owner in ("alice", "bob"):
            identity.set(owner)
            _, lease = service.admit_execution("generation", owner, owner, lambda: True)
            assert await service.run_async(lease, operation) == owner
            assert identity.get() == owner

    try:
        asyncio.run(exercise())
    finally:
        service.shutdown_async_workers()
