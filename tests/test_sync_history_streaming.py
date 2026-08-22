"""Regression tests for bounded, complete sync history backfill."""

from __future__ import annotations

import asyncio
import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_factory.enums.sync_status import BatchStatus
from train_factory.sync import sync_worker


class _FakeSyncService:
    def __init__(self, registered_batches, historical_batches):
        self.registered_batches = registered_batches
        self.historical_batches = historical_batches
        self.history_scan_offsets = []
        self.promoted_batch_ids = []

    def list_batches(
        self,
        *,
        status=None,
        limit=200,
        offset=0,
        oldest_first=False,
        **_kwargs,
    ):
        if status == BatchStatus.REGISTERED:
            batches = [
                batch
                for batch in self.registered_batches
                if batch["status"] == BatchStatus.REGISTERED
            ]
        else:
            self.history_scan_offsets.append(offset)
            batches = self.historical_batches
        return batches[offset : offset + limit], len(batches)

    def promote_batch_to_fetched(self, task_id, batch_id, user_id):
        for batch in self.registered_batches:
            if (
                batch["task_id"] == task_id
                and batch["batch_id"] == batch_id
                and batch["user_id"] == user_id
            ):
                batch["status"] = BatchStatus.FETCHED
                self.promoted_batch_ids.append(batch_id)
                return True
        return False

    def update_task(self, *_args, **_kwargs):
        return True


def _write_documents(path: Path, prefix: str, count: int) -> dict[str, int]:
    encoded_lines = []
    sizes = {}
    for index in range(count):
        doc_id = f"{prefix}-{index}"
        line = (
            json.dumps(
                {
                    "content": f"content-{index}",
                    "metadata": {"chunk_id": doc_id},
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        encoded_lines.append(line)
        sizes[doc_id] = len(line)
    path.write_bytes(b"".join(encoded_lines))
    return sizes


def _registered_batch(path: Path, index: int) -> dict:
    return {
        "batch_id": f"registered-{index}",
        "task_id": "task-history-streaming",
        "user_id": "user-history-streaming",
        "record_count": 1,
        "storage_path": str(path),
        "status": BatchStatus.REGISTERED,
    }


def _install_pre_index_harness(
    monkeypatch,
    *,
    service: _FakeSyncService,
    max_docs: int,
    max_bytes: int,
    fail_adapter_call: int | None = None,
    guard_entry_error: BaseException | None = None,
    fatal_write_error: BaseException | None = None,
):
    sync_service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    embedding_client_module = importlib.import_module(
        "train_factory.generation.clients.embedding_client"
    )
    embedding_filter_module = importlib.import_module(
        "train_factory.generation.steps.embedding_filter_step"
    )
    milvus_client_module = importlib.import_module(
        "train_factory.generation.clients.milvus_client"
    )
    milvus_service_module = importlib.import_module(
        "train_factory.storage.services.milvus_collection_service"
    )

    adapter_windows = []
    service.registry_events = []
    service.registry_lock_depth = 0

    class FakeMilvusRegistry:
        def register_collection(self, *, collection_name, **_kwargs):
            service.registry_events.append(
                (
                    "register",
                    collection_name,
                    _kwargs.get("sync_task_id"),
                )
            )
            return {"collection_name": collection_name, "status": "active"}

        @contextmanager
        def consumption_guard(self, collection_names, *, user_id):
            service.registry_events.append(
                ("lock-enter", tuple(sorted(collection_names)), user_id)
            )
            if guard_entry_error is not None:
                raise guard_entry_error
            service.registry_lock_depth += 1
            try:
                yield
            finally:
                service.registry_lock_depth -= 1
                service.registry_events.append("lock-exit")

    class FakeEmbeddingClient:
        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class FakeMilvusClient:
        def __init__(self, _config):
            pass

        def connect(self):
            return None

        def close(self):
            service.registry_events.append("client-close")
            return None

    class RecordingEmbeddingFilterStep:
        def __init__(self, *args, collection_name, **kwargs):
            self.collection_name = collection_name

        async def pre_index_all_chunks(self, documents):
            assert service.registry_lock_depth == 1
            service.registry_events.append(("write", self.collection_name))
            if fatal_write_error is not None:
                raise fatal_write_error
            if self.collection_name == "adapter-collection":
                adapter_windows.append(tuple(document.doc_id for document in documents))
                if fail_adapter_call == len(adapter_windows):
                    raise RuntimeError("adapter window failed")
            return len(documents)

    monkeypatch.setattr(
        sync_service_module,
        "external_sync_service",
        service,
    )
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: SimpleNamespace(
            sync_historical_max_docs=max_docs,
            sync_historical_max_bytes=max_bytes,
        ),
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: {
            "endpoint": "http://embedding.invalid",
            "model": "embedding-model",
        },
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_sync_milvus_connection_config",
        lambda _config: {},
    )
    monkeypatch.setattr(
        sync_worker,
        "_get_sync_targets",
        lambda *_args, **_kwargs: [
            {
                "collection_name": "base-collection",
                "model_name": "embedding-model",
                "label": "base",
                "type": "base",
            },
            {
                "collection_name": "adapter-collection",
                "model_name": "adapter-model",
                "label": "adapter",
                "type": "adapter",
            },
        ],
    )
    monkeypatch.setattr(
        embedding_client_module,
        "EmbeddingConfig",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        embedding_client_module,
        "EmbeddingClient",
        FakeEmbeddingClient,
    )
    monkeypatch.setattr(
        milvus_client_module,
        "MilvusConfig",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        milvus_client_module,
        "MilvusClient",
        FakeMilvusClient,
    )
    monkeypatch.setattr(
        embedding_filter_module,
        "EmbeddingFilterStep",
        RecordingEmbeddingFilterStep,
    )
    monkeypatch.setattr(
        milvus_service_module,
        "milvus_collection_service",
        FakeMilvusRegistry(),
    )

    return adapter_windows


def test_pre_index_registers_and_locks_every_collection_during_remote_writes(
    monkeypatch,
    tmp_path,
):
    batch_path = tmp_path / "batch.jsonl"
    _write_documents(batch_path, "current", 1)
    service = _FakeSyncService([], [])
    _install_pre_index_harness(
        monkeypatch,
        service=service,
        max_docs=10,
        max_bytes=10_000,
    )

    result = asyncio.run(
        sync_worker._pre_index_to_milvus(_config(), str(batch_path))
    )

    assert result["ready_for_generation"] is True
    assert service.registry_events[:3] == [
        ("register", "base-collection", "task-history-streaming"),
        ("register", "adapter-collection", "task-history-streaming"),
        (
            "lock-enter",
            ("adapter-collection", "base-collection"),
            "user-history-streaming",
        ),
    ]
    assert service.registry_events[-2:] == ["client-close", "lock-exit"]
    assert ("write", "base-collection") in service.registry_events
    assert ("write", "adapter-collection") in service.registry_events


def test_pre_index_closes_client_when_guard_entry_is_interrupted(
    monkeypatch,
    tmp_path,
):
    class GuardEntryInterrupted(BaseException):
        pass

    batch_path = tmp_path / "batch.jsonl"
    _write_documents(batch_path, "current", 1)
    service = _FakeSyncService([], [])
    _install_pre_index_harness(
        monkeypatch,
        service=service,
        max_docs=10,
        max_bytes=10_000,
        guard_entry_error=GuardEntryInterrupted("guard entry interrupted"),
    )

    with pytest.raises(GuardEntryInterrupted, match="guard entry interrupted"):
        asyncio.run(sync_worker._pre_index_to_milvus(_config(), str(batch_path)))

    assert service.registry_events == [
        ("register", "base-collection", "task-history-streaming"),
        ("register", "adapter-collection", "task-history-streaming"),
        (
            "lock-enter",
            ("adapter-collection", "base-collection"),
            "user-history-streaming",
        ),
        "client-close",
    ]


def test_pre_index_closes_client_before_releasing_guard_on_abrupt_write_failure(
    monkeypatch,
    tmp_path,
):
    class RemoteWriteInterrupted(BaseException):
        pass

    batch_path = tmp_path / "batch.jsonl"
    _write_documents(batch_path, "current", 1)
    service = _FakeSyncService([], [])
    _install_pre_index_harness(
        monkeypatch,
        service=service,
        max_docs=10,
        max_bytes=10_000,
        fatal_write_error=RemoteWriteInterrupted("remote write interrupted"),
    )

    with pytest.raises(RemoteWriteInterrupted, match="remote write interrupted"):
        asyncio.run(sync_worker._pre_index_to_milvus(_config(), str(batch_path)))

    assert service.registry_events[-3:] == [
        ("write", "base-collection"),
        "client-close",
        "lock-exit",
    ]
    assert service.registry_lock_depth == 0


def _config() -> dict:
    return {
        "task_id": "task-history-streaming",
        "user_id": "user-history-streaming",
        "generation_config": {},
        "milvus_collection_name": "base-collection",
    }


@pytest.mark.parametrize("limiter", ["docs", "bytes"])
def test_registered_recovery_streams_one_history_batch_once_in_bounded_windows(
    monkeypatch,
    tmp_path,
    limiter,
):
    history_path = tmp_path / "history.jsonl"
    line_sizes = _write_documents(history_path, "history", 5)
    current_paths = []
    for index in range(2):
        current_path = tmp_path / f"registered-{index}.jsonl"
        _write_documents(current_path, f"current-{index}", 1)
        current_paths.append(current_path)

    registered = [
        _registered_batch(path, index) for index, path in enumerate(current_paths)
    ]
    service = _FakeSyncService(
        registered,
        [{"batch_id": "history", "storage_path": str(history_path)}],
    )
    one_line_bytes = next(iter(line_sizes.values()))
    max_docs = 2 if limiter == "docs" else 10
    max_bytes = 10_000 if limiter == "docs" else one_line_bytes * 2
    adapter_windows = _install_pre_index_harness(
        monkeypatch,
        service=service,
        max_docs=max_docs,
        max_bytes=max_bytes,
    )

    promoted = asyncio.run(sync_worker._promote_registered_batches(_config()))

    assert promoted == 2
    assert service.promoted_batch_ids == ["registered-0", "registered-1"]
    assert service.history_scan_offsets == [0]
    assert adapter_windows == [
        ("history-0", "history-1"),
        ("history-2", "history-3"),
        ("history-4",),
    ]
    assert [doc_id for window in adapter_windows for doc_id in window] == [
        f"history-{index}" for index in range(5)
    ]
    for window in adapter_windows:
        assert len(window) <= max_docs
        assert sum(line_sizes[doc_id] for doc_id in window) <= max_bytes


def test_registered_recovery_restarts_from_first_window_after_second_window_fails(
    monkeypatch,
    tmp_path,
):
    history_path = tmp_path / "history-retry.jsonl"
    _write_documents(history_path, "retry", 5)
    current_path = tmp_path / "registered-retry.jsonl"
    _write_documents(current_path, "current", 1)
    service = _FakeSyncService(
        [_registered_batch(current_path, 0)],
        [{"batch_id": "history", "storage_path": str(history_path)}],
    )
    adapter_windows = _install_pre_index_harness(
        monkeypatch,
        service=service,
        max_docs=2,
        max_bytes=10_000,
        fail_adapter_call=2,
    )

    first_promoted = asyncio.run(sync_worker._promote_registered_batches(_config()))

    assert first_promoted == 0
    assert service.promoted_batch_ids == []
    assert adapter_windows == [
        ("retry-0", "retry-1"),
        ("retry-2", "retry-3"),
    ]

    second_promoted = asyncio.run(sync_worker._promote_registered_batches(_config()))

    assert second_promoted == 1
    assert service.promoted_batch_ids == ["registered-0"]
    assert service.history_scan_offsets == [0, 0]
    assert adapter_windows == [
        ("retry-0", "retry-1"),
        ("retry-2", "retry-3"),
        ("retry-0", "retry-1"),
        ("retry-2", "retry-3"),
        ("retry-4",),
    ]


@pytest.mark.parametrize("history_kind", ["missing", "invalid-json", "empty-content"])
def test_registered_recovery_fails_closed_for_invalid_tracked_history(
    monkeypatch,
    tmp_path,
    history_kind,
):
    history_path = tmp_path / "invalid-history.jsonl"
    if history_kind == "invalid-json":
        history_path.write_text("not-json\n", encoding="utf-8")
    elif history_kind == "empty-content":
        history_path.write_text(
            json.dumps({"content": "", "metadata": {"chunk_id": "empty"}}) + "\n",
            encoding="utf-8",
        )

    current_path = tmp_path / "registered-invalid-history.jsonl"
    _write_documents(current_path, "current", 1)
    service = _FakeSyncService(
        [_registered_batch(current_path, 0)],
        [{"batch_id": "history", "storage_path": str(history_path)}],
    )
    _install_pre_index_harness(
        monkeypatch,
        service=service,
        max_docs=2,
        max_bytes=10_000,
    )

    promoted = asyncio.run(sync_worker._promote_registered_batches(_config()))

    assert promoted == 0
    assert service.promoted_batch_ids == []
