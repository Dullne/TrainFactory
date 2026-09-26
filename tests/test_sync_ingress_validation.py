"""Fail-closed ingress and monotonic cursor regressions for external sync."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from train_factory.core import time_utils
from train_factory.sync import sync_worker


class _SyncService:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.saved_batches = []

    def update_task(self, task_id, **updates):
        assert task_id == self.config["task_id"]
        self.config.update(updates)
        return dict(self.config)

    def get_task_raw(self, task_id):
        assert task_id == self.config["task_id"]
        return dict(self.config)

    def create_batch(self, **values):
        batch = {
            **values,
            "batch_id": f"batch-{len(self.saved_batches)}",
            "status": values["initial_status"],
        }
        self.saved_batches.append(batch)
        return batch, True


def _config() -> dict:
    return {
        "task_id": "task-ingress",
        "user_id": "user-ingress",
        "external_api_url": "https://source.invalid/items",
        "external_auth_config": {},
        "generation_threshold": 0,
        "training_threshold": 0,
        "last_sync_at": None,
        "last_sync_boundary_ids": [],
        "boundary_rollback_seconds": 5,
    }


def _item(item_id: str, created_at: str, text: str = "content") -> dict:
    return {
        "id": item_id,
        "session_id": "session",
        "doc_id": f"doc-{item_id}",
        "text": text,
        "created_at": created_at,
    }


def _install_run_once_harness(monkeypatch, tmp_path: Path, pages: list[list[dict]]):
    monkeypatch.setattr(time_utils, "get_app_timezone", lambda: ZoneInfo("Asia/Shanghai"))
    service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service = _SyncService(_config())
    saved_items = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def fetch_all_since(self, _since):
            return pages.pop(0)

    def save_batch(_config, items):
        saved_items.append(list(items))
        path = tmp_path / f"batch-{len(saved_items)}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        return str(path)

    async def noop(*_args, **_kwargs):
        return 0

    async def promote(_config, _batch, historical_docs_cache=None):
        return True

    monkeypatch.setattr(service_module, "external_sync_service", service)
    monkeypatch.setattr(sync_worker, "ExternalApiClient", Client)
    monkeypatch.setattr(sync_worker, "validate_user_outbound_url", lambda url, _uid: url)
    monkeypatch.setattr(sync_worker, "_update_api_config_status", lambda *_args: None)
    monkeypatch.setattr(sync_worker, "_promote_registered_batches", noop)
    monkeypatch.setattr(sync_worker, "_promote_batch_to_fetched", promote)
    monkeypatch.setattr(sync_worker, "_check_thresholds", noop)
    monkeypatch.setattr(sync_worker, "_save_batch", save_batch)
    monkeypatch.setattr(sync_worker, "_utcnow_naive", lambda: datetime(2026, 8, 11, 12, 0, 0))
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: SimpleNamespace(
            sync_max_record_bytes=512,
            sync_max_future_skew_seconds=300,
            sync_pending_max_records_per_task=100,
            sync_generation_max_input_bytes=10_000,
            sync_boundary_max_ids=100,
            sync_boundary_max_bytes=10_000,
        ),
    )
    return service, saved_items


@pytest.mark.parametrize(
    "bad_item",
    [
        {"id": "missing-text", "created_at": "2026-08-11T10:00:00+08:00"},
        _item("blank-text", "2026-08-11T10:00:00+08:00", "   "),
        {"id": "missing-time", "text": "content"},
        _item("invalid-time", "not-a-time"),
        _item("future-time", "2026-08-11T12:05:01+08:00"),
        _item("oversized", "2026-08-11T10:00:00+08:00", "x" * 1024),
    ],
)
def test_invalid_ingress_page_does_not_write_or_advance_cursor(
    monkeypatch,
    tmp_path,
    bad_item,
):
    service, saved_items = _install_run_once_harness(
        monkeypatch,
        tmp_path,
        [[bad_item]],
    )
    service.config["last_sync_at"] = datetime(2026, 8, 11, 10, 0, 0)
    original_cursor = service.config["last_sync_at"]

    asyncio.run(sync_worker.run_once(dict(service.config)))

    assert saved_items == []
    assert service.saved_batches == []
    assert service.config["last_sync_at"] == original_cursor
    assert service.config["status"] == "error"


def test_run_once_deduplicates_repeated_records_within_one_cycle(
    monkeypatch,
    tmp_path,
):
    duplicate = _item("same", "2026-08-11T10:00:00+08:00")
    service, saved_items = _install_run_once_harness(
        monkeypatch,
        tmp_path,
        [[duplicate, dict(duplicate), dict(duplicate)]],
    )

    asyncio.run(sync_worker.run_once(dict(service.config)))

    assert [[item["id"] for item in batch] for batch in saved_items] == [["same"]]
    assert service.saved_batches[0]["record_count"] == 1


def test_boundary_is_retained_and_cursor_never_moves_backwards(
    monkeypatch,
    tmp_path,
):
    at_t = _item("at-t", "2026-08-11T10:00:00+08:00")
    at_t_plus_two = _item("at-t-plus-two", "2026-08-11T10:00:02+08:00")
    late = _item("late", "2026-08-11T09:59:58+08:00")
    service, saved_items = _install_run_once_harness(
        monkeypatch,
        tmp_path,
        [[at_t, at_t_plus_two], [at_t, late]],
    )
    service.config["last_sync_at"] = datetime(2026, 8, 11, 10, 0, 0)
    service.config["last_sync_boundary_ids"] = [sync_worker._item_dedup_key(at_t)]

    asyncio.run(sync_worker.run_once(dict(service.config)))

    assert [item["id"] for item in saved_items[0]] == ["at-t-plus-two"]
    assert service.config["last_sync_at"] == datetime(2026, 8, 11, 10, 0, 2)
    assert sync_worker._item_dedup_key(at_t) in service.config["last_sync_boundary_ids"]

    asyncio.run(sync_worker.run_once(dict(service.config)))

    assert [item["id"] for item in saved_items[1]] == ["late"]
    assert service.config["last_sync_at"] == datetime(2026, 8, 11, 10, 0, 2)


def test_stable_chunk_ids_are_versioned_session_aware_and_unambiguous():
    single = sync_worker._build_stable_chunk_id(
        {"external_id": "a:b", "session_id": "", "doc_id": ""}
    )
    composite = sync_worker._build_stable_chunk_id(
        {"external_id": "a", "session_id": "", "doc_id": "b"}
    )
    first_session = sync_worker._build_stable_chunk_id(
        {"external_id": "", "session_id": "session-1", "doc_id": "doc"}
    )
    second_session = sync_worker._build_stable_chunk_id(
        {"external_id": "", "session_id": "session-2", "doc_id": "doc"}
    )

    assert single.startswith("v2:")
    assert composite.startswith("v2:")
    assert single != composite
    assert first_session != second_session


def test_successful_empty_sync_clears_previous_source_error(monkeypatch, tmp_path):
    service, _saved = _install_run_once_harness(monkeypatch, tmp_path, [[]])
    service.config.update(status="error", error_message="previous source failure")

    asyncio.run(sync_worker.run_once(dict(service.config)))

    assert service.config["status"] == "idle"
    assert service.config["error_message"] is None
