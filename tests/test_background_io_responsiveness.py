"""Slow task dependencies must not occupy the API event loop or free live leases."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from train_factory.api import server
from train_factory.generation.clients.milvus_client import MilvusClient
from train_factory.generation.pipeline import DatasetGenerationPipeline, PipelineConfig
from train_factory.storage.services.audit_log_service import audit_log_service
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskAdmissionService,
)


@pytest.mark.parametrize("stage", ("commit", "identity"))
def test_slow_audit_database_work_does_not_block_another_request(monkeypatch, stage):
    from train_factory.auth import dependencies

    entered, release = threading.Event(), threading.Event()
    database_returned = threading.Event()
    recorded = []

    def block():
        entered.set()
        release.wait(2)
        database_returned.set()

    def log(**fields):
        if stage == "commit":
            block()
        recorded.append(fields)

    def get_user(_user_id):
        block()
        return {
            "user_id": "user-1", "username": "alice", "token_version": 1,
            "is_active": True, "is_admin": False,
        }

    monkeypatch.setattr(audit_log_service, "log", log)
    monkeypatch.setattr(server.settings, "rate_limit_enabled", False)
    monkeypatch.setattr(server.settings, "auth_enabled", True)
    monkeypatch.setattr(dependencies, "decode_token", lambda _token: {
        "sub": "user-1", "username": "alice", "ver": 1,
    })
    monkeypatch.setattr(dependencies.user_service, "get_user", get_user)
    app = server.create_app()

    @app.post("/api/audit-probe")
    async def write():
        return {"saved": True}

    @app.get("/probe-ping")
    async def ping():
        return {"available": True}

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            if stage == "identity":
                client.cookies.set("access_token", "test-token")
            request = asyncio.create_task(client.post("/api/audit-probe"))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                assert not database_returned.is_set(), "audit database work occupied the event loop"
                response = await asyncio.wait_for(client.get("/probe-ping"), 1)
                assert response.json() == {"available": True}
            finally:
                release.set()
                response = await request
            assert response.status_code == 200
            assert recorded[0]["endpoint"] == "/api/audit-probe"
            assert recorded[0]["user_id"] == ("user-1" if stage == "identity" else "anonymous")

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_waiter", (False, True))
def test_generation_milvus_work_keeps_api_loop_and_live_lease(monkeypatch, tmp_path, cancel_waiter):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    search_returned = threading.Event()
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})
    _, lease = service.admit_execution(
        "generation", None, "user-1", lambda: {"task_id": "task-1"},
    )
    pipeline = DatasetGenerationPipeline(PipelineConfig(
        input_path=str(tmp_path / "input.jsonl"),
        output_path=str(tmp_path / "output.jsonl"),
        generation_mode="qa_to_eval",
    ))

    async def embed(_texts):
        return np.array([[0.1, 0.2]])

    def search(*_args, **_kwargs):
        entered.set()
        release.wait(2)
        search_returned.set()
        return [[{"chunk_content": "context"}]]

    async def operation():
        try:
            return await pipeline._generate_eval_data_batch(
                [{"query": "question", "answer": "answer", "chunk_id": "source"}],
                SimpleNamespace(embed=embed), SimpleNamespace(search_similar=search), "owned",
            )
        finally:
            exited.set()

    async def exercise():
        task = asyncio.create_task(service.run_async(lease, operation))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert not search_returned.is_set(), "generation occupied the API loop until search returned"
            assert service.is_executing("generation", "task-1")
            if cancel_waiter:
                pipeline.stop()
                task.cancel()
                await asyncio.sleep(0)
                assert service.is_executing("generation", "task-1"), "live worker lost its lease"
        finally:
            release.set()
            try:
                result = await task
            except asyncio.CancelledError:
                result = None
            assert await asyncio.to_thread(exited.wait, 3)
        if not cancel_waiter:
            assert result[0]["retrieval_context"] == ["context"]
        for _ in range(100):
            if not service.is_executing("generation", "task-1"):
                break
            await asyncio.sleep(0.01)
        assert not service.is_executing("generation", "task-1")

    asyncio.run(exercise())


def test_failed_generation_worker_releases_its_execution_lease(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})
    _, lease = service.admit_execution(
        "generation", None, "user-1", lambda: {"task_id": "failed-task"},
    )

    async def fail():
        raise ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        asyncio.run(service.run_async(lease, fail))
    assert not service.is_executing("generation", "failed-task")


def test_cancelled_queued_generation_does_not_run_or_leak_capacity(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=2)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})
    _, lease = service.admit_execution(
        "generation", None, "user-1", lambda: {"task_id": "queued-task"},
    )
    entered, release = threading.Event(), threading.Event()
    executed = []

    def occupy_worker():
        entered.set()
        release.wait(3)

    async def operation():
        executed.append(True)

    async def exercise():
        loop = asyncio.get_running_loop()
        service._async_executor = ThreadPoolExecutor(max_workers=1)
        blocker = loop.run_in_executor(service._async_executor, occupy_worker)
        try:
            async with asyncio.timeout(2):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            task = asyncio.create_task(service.run_async(lease, operation))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not service.is_executing("generation", "queued-task")
        finally:
            release.set()
            await blocker
            service.shutdown_async_workers()
        assert executed == []

    asyncio.run(exercise())


def test_milvus_connection_and_retrieval_use_bounded_rpc_deadlines(monkeypatch):
    import pymilvus

    def bounded(*_args, **kwargs):
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, (float, int)) and 0 < timeout <= 600, "RPC has no finite deadline"

    def rpc_search(*args, **kwargs):
        bounded(*args, **kwargs)
        return [[]]

    class Collection:
        def __init__(self, *_args, **kwargs):
            bounded(**kwargs)

        load = staticmethod(bounded)
        search = staticmethod(rpc_search)

    monkeypatch.setattr(pymilvus.connections, "connect", bounded)
    monkeypatch.setattr(pymilvus.connections, "disconnect", lambda *_args: None)
    monkeypatch.setattr(pymilvus, "Collection", Collection)
    client = MilvusClient()
    client.connect()
    try:
        assert client.search_similar("owned", np.array([[0.1, 0.2]])) == [[]]
    finally:
        client.close()


@pytest.mark.parametrize("operation", ("info", "browse"))
@pytest.mark.parametrize("close_fails", (False, True))
def test_milvus_metadata_uses_deadlines_and_closes_its_connection(monkeypatch, operation, close_fails):
    import pymilvus

    closed = []

    def bounded(*_args, **kwargs):
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, (float, int)) and 0 < timeout <= 600

    class Metadata:
        def __init__(self, **kwargs):
            bounded(**kwargs)
            assert kwargs["uri"] == "http://localhost:19530"
            assert kwargs["token"] == "metadata-test-token"

        def get_collection_stats(self, _name, **kwargs):
            bounded(**kwargs)
            return {"row_count": 7}

        def list_indexes(self, _name, **kwargs):
            bounded(**kwargs)
            return ["vector-index"]

        def close(self):
            closed.append("metadata")
            if close_fails:
                raise RuntimeError("metadata disconnect failed")

    class Collection:
        schema = SimpleNamespace(description="owned", fields=[])

        def __init__(self, *_args, **kwargs):
            bounded(**kwargs)

        load = staticmethod(bounded)

        @property
        def indexes(self):
            pytest.fail("ORM indexes property performs an unbounded RPC")

        @property
        def num_entities(self):
            pytest.fail("ORM num_entities property performs an unbounded RPC")

        def index(self, **kwargs):
            bounded(**kwargs)
            assert kwargs["index_name"] == "vector-index"
            return SimpleNamespace(field_name="vector", params={
                "index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128},
            })

        def query(self, **kwargs):
            bounded(**kwargs)
            return [{"chunk_id": "first"}]

    monkeypatch.setattr(pymilvus, "MilvusClient", Metadata)
    monkeypatch.setattr(pymilvus, "Collection", Collection)
    monkeypatch.setattr(pymilvus.connections, "connect", bounded)
    monkeypatch.setattr(pymilvus.connections, "disconnect", lambda _alias: closed.append("orm"))
    monkeypatch.setattr(pymilvus.utility, "has_collection", lambda *_args, **kwargs: bounded(**kwargs) or True)
    monkeypatch.setattr(pymilvus.utility, "load_state", lambda *_args, **kwargs: bounded(**kwargs) or "Loaded")
    from train_factory.generation.clients.milvus_client import MilvusConfig

    with MilvusClient(MilvusConfig(host="localhost", token="metadata-test-token")) as client:
        if operation == "info":
            result = client.get_collection_info("owned")
            assert result["num_entities"] == 7
            assert result["indexes"] == [{
                "field_name": "vector", "index_type": "IVF_FLAT", "metric_type": "COSINE",
                "params": {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
            }]
        else:
            assert client.query_entities("owned") == ([{"chunk_id": "first"}], 7)
    assert sorted(closed) == ["metadata", "orm"]
