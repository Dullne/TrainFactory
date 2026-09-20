"""Regressions for sync generation staging rollback."""

import asyncio
import builtins
import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_factory.sync import sync_worker


def _sync_limit_settings(**overrides):
    values = {
        "sync_storage_max_bytes_global": 20 * 1024**3,
        "sync_storage_max_bytes_per_user": 5 * 1024**3,
        "sync_pending_max_batches_per_task": 1_000,
        "sync_pending_max_records_per_task": 1_000_000,
        "sync_generation_max_input_bytes": 1024**3,
        "sync_max_record_bytes": 8 * 1024**2,
        "sync_max_future_skew_seconds": 300,
        "sync_boundary_max_ids": 10_000,
        "sync_boundary_max_bytes": 4 * 1024**2,
        "sync_historical_max_docs": 100_000,
        "sync_historical_max_bytes": 64 * 1024**2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_staging_harness(monkeypatch, tmp_path):
    asset_module = importlib.import_module(
        "train_factory.storage.services.dataset_asset_service"
    )
    admission_module = importlib.import_module(
        "train_factory.storage.services.background_task_admission_service"
    )
    dataset_module = importlib.import_module(
        "train_factory.storage.services.dataset_service"
    )
    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    lineage_module = importlib.import_module(
        "train_factory.storage.services.dataset_lineage_service"
    )
    milvus_module = importlib.import_module(
        "train_factory.storage.services.milvus_collection_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )

    state = SimpleNamespace(
        assets=set(),
        dataset_names=[],
        datasets=set(),
        deletion_owners={},
        collection_registrations=[],
        generation_failures=[],
        generation_outputs=[],
        generation_tasks={},
        generation_task_kwargs=[],
        generation_tracking_calls=[],
        leases=[],
        lineages=set(),
        merged_paths=[],
        recovery_calls=[],
        events=[],
    )

    monkeypatch.setenv("GENERATION_OUTPUT_DIR", str(tmp_path / "generated"))
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(tmp_path / "sync"))
    monkeypatch.setattr(sync_worker, "_restore_stopped_queued_batches", lambda _id: 0)
    monkeypatch.setattr(
        sync_worker,
        "_select_generation_batches",
        lambda _config, batches: batches,
    )

    def merge_batches(config, _batches):
        state.events.append("merge")
        path = Path(config["_generation_merged_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"content":"document"}\n', encoding="utf-8")
        state.merged_paths.append(path)
        return str(path)

    monkeypatch.setattr(sync_worker, "_merge_batches", merge_batches)

    def create_dataset(**_kwargs):
        state.events.append("dataset")
        state.dataset_names.append(_kwargs["dataset_name"])
        dataset_id = _kwargs["dataset_id"]
        state.datasets.add(dataset_id)
        return {"dataset_id": dataset_id}

    monkeypatch.setattr(
        dataset_module.dataset_service, "create_dataset", create_dataset
    )

    def get_dataset(dataset_id):
        if dataset_id not in state.datasets:
            return None
        return {
            "dataset_id": dataset_id,
            "user_id": "user-1",
            "source_task_type": "sync",
            "source_task_id": "sync-1",
        }

    def mark_deleting(dataset_id, *, deletion_owner, user_id):
        assert user_id == "user-1"
        if dataset_id not in state.datasets:
            return False
        current_owner = state.deletion_owners.get(dataset_id)
        assert current_owner in {None, deletion_owner}
        state.deletion_owners[dataset_id] = deletion_owner
        return True

    def delete_dataset(dataset_id, *, deletion_owner, user_id):
        assert user_id == "user-1"
        assert state.deletion_owners.get(dataset_id) == deletion_owner
        state.deletion_owners.pop(dataset_id, None)
        state.datasets.discard(dataset_id)
        return True

    monkeypatch.setattr(dataset_module.dataset_service, "get_dataset", get_dataset)
    monkeypatch.setattr(
        dataset_module.dataset_service,
        "mark_deleting",
        mark_deleting,
    )
    monkeypatch.setattr(
        dataset_module.dataset_service,
        "delete_dataset",
        delete_dataset,
    )

    def delete_lineage(dataset_id):
        deleted = int(dataset_id in state.lineages)
        state.lineages.discard(dataset_id)
        return deleted

    monkeypatch.setattr(
        lineage_module.dataset_lineage_service,
        "create_edge",
        lambda **kwargs: state.lineages.add(kwargs["to_dataset_id"]),
    )
    monkeypatch.setattr(
        lineage_module.dataset_lineage_service,
        "delete_edges_for_dataset",
        delete_lineage,
    )

    def delete_assets(dataset_id):
        deleted = int(dataset_id in state.assets)
        state.assets.discard(dataset_id)
        return deleted

    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "create_asset",
        lambda **kwargs: state.assets.add(kwargs["dataset_id"]),
    )
    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "delete_assets_for_dataset",
        delete_assets,
    )

    fake_sync_service = SimpleNamespace(
        update_task=lambda *_args, **_kwargs: None,
        has_pending_generation=lambda _task_id: False,
        get_pending_batches=lambda _task_id: [
            {"batch_id": "batch-1", "record_count": 1}
        ],
        has_enabled_completed_generation=lambda _task_id: False,
        get_all_completed_generation_datasets=lambda _task_id: [],
        reset_completed_batches=lambda _task_id: 0,
        reset_generating_task_if_no_pending_generation=lambda _task_id: True,
        create_generation_and_claim_batches=lambda **kwargs: (
            state.events.append("tracking")
            or state.generation_tracking_calls.append(kwargs)
            or {"generation_task_id": kwargs["generation_task_id"]}
        ),
        fail_generation_and_restore_batches=lambda *args: (
            state.recovery_calls.append(args)
            or {
                "tracking_found": bool(state.generation_tracking_calls),
                "recovered": bool(state.generation_tracking_calls),
            }
        ),
    )
    monkeypatch.setattr(sync_module, "external_sync_service", fake_sync_service)

    generation_count = 0

    def create_generation_task(**_kwargs):
        nonlocal generation_count
        generation_count += 1
        state.events.append("generation_task")
        state.generation_task_kwargs.append(_kwargs)
        task_id = _kwargs["task_id"]
        state.generation_tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "run_token": f"generation-run-token-{generation_count}",
        }
        return {"task_id": _kwargs["task_id"]}

    def get_generation_task_raw(task_id):
        task = state.generation_tasks.get(task_id)
        return dict(task) if task else None

    def set_generation_output(
        task_id,
        output_path,
        output_count,
        *,
        expected_status=None,
        expected_run_token=None,
    ):
        task = state.generation_tasks.get(task_id)
        if not task:
            return False
        if expected_status is not None and task["status"] != expected_status:
            return False
        if expected_run_token is not None and task["run_token"] != expected_run_token:
            return False
        state.generation_outputs.append((task_id, output_path, output_count))
        return True

    def update_generation_status(
        task_id,
        status,
        error_message=None,
        *,
        expected_status=None,
        expected_run_token=None,
    ):
        task = state.generation_tasks.get(task_id)
        if not task:
            return False
        if expected_status is not None and task["status"] != expected_status:
            return False
        if expected_run_token is not None and task["run_token"] != expected_run_token:
            return False
        if (
            status == "failed"
            and task["status"]
            not in {"pending", "running", "stopping", "publishing", "recovering"}
        ):
            return False
        task["status"] = status
        state.generation_failures.append(
            (
                (task_id, status, error_message),
                {
                    "expected_status": expected_status,
                    "expected_run_token": expected_run_token,
                },
            )
        )
        return True

    fake_generation_service = SimpleNamespace(
        create_task=create_generation_task,
        get_task_raw=get_generation_task_raw,
        set_output=set_generation_output,
        update_status=update_generation_status,
    )
    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        fake_generation_service,
    )
    monkeypatch.setattr(
        milvus_module,
        "milvus_collection_service",
        SimpleNamespace(
            register_collection=lambda **kwargs: (
                state.collection_registrations.append(kwargs)
                or {"collection_name": kwargs["collection_name"]}
            )
        ),
    )

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        lease = SimpleNamespace(release_calls=0, released=False)

        def release():
            if lease.released:
                return
            lease.released = True
            lease.release_calls += 1

        lease.release = release
        state.leases.append(lease)
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        admission_module.background_task_admission_service,
        "assert_capacity_available",
        lambda _user_id: None,
    )
    monkeypatch.setattr(
        admission_module.background_task_admission_service,
        "admit_execution",
        admit,
    )
    return state


def test_generation_trigger_does_not_reset_enabled_eval_or_qa_completion(
    monkeypatch,
    tmp_path,
):
    _install_staging_harness(monkeypatch, tmp_path)
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    service = sync_module.external_sync_service
    reset_calls = []
    service.get_pending_batches = lambda _task_id: []
    service.has_enabled_completed_generation = lambda _task_id: True
    service.reset_completed_batches = lambda task_id: reset_calls.append(task_id) or 0

    asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert reset_calls == []


def _trigger_config():
    return {
        "task_id": "sync-1",
        "user_id": "user-1",
        "generation_mode": "qa_extraction",
        "generation_config": {},
    }


def _assert_no_staging_artifacts(state):
    assert state.datasets == set()
    assert state.lineages == set()
    assert state.assets == set()
    assert all(not path.exists() for path in state.merged_paths)


def test_sync_tracking_is_durable_before_generation_staging(monkeypatch, tmp_path):
    state = _install_staging_harness(monkeypatch, tmp_path)
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )

    asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert state.events[:4] == [
        "tracking",
        "generation_task",
        "merge",
        "dataset",
    ]


def test_sync_generation_persists_existing_collection_before_admission(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    config = _trigger_config()
    config["milvus_collection_name"] = "sync-collection-1"

    asyncio.run(sync_worker._trigger_generation(config))

    assert state.collection_registrations == [
        {
            "collection_name": "sync-collection-1",
            "embedding_config_id": None,
            "embedding_model": None,
            "embedding_endpoint": None,
            "dim": 1024,
            "user_id": "user-1",
            "sync_task_id": "sync-1",
        }
    ]
    assert state.generation_task_kwargs[0]["milvus_collection"] == (
        "sync-collection-1"
    )


def _write_managed_batch(root, user_id, task_id, name, content):
    path = root / user_id / task_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "batch_id": name.removesuffix(".jsonl"),
        "task_id": task_id,
        "user_id": user_id,
        "storage_path": str(path),
    }


def test_merge_attempt_paths_are_unique_and_fully_tenant_scoped(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(tmp_path / "sync"))
    monkeypatch.setattr(
        sync_worker,
        "_utcnow_naive",
        lambda: datetime(2026, 8, 11, 10, 11, 12),
    )
    user_id = "1234567890abcdef-user-one"
    task_id = "deadbeef-task-one"
    sync_root = tmp_path / "sync"
    first_batch = _write_managed_batch(
        sync_root,
        user_id,
        task_id,
        "batch_first.jsonl",
        '{"content":"first"}\n',
    )
    second_batch = _write_managed_batch(
        sync_root,
        user_id,
        task_id,
        "batch_second.jsonl",
        '{"content":"second"}\n',
    )

    first_path = Path(
        sync_worker._merge_batches(
            {
                "task_id": task_id,
                "user_id": user_id,
                "_generation_attempt_id": "attempt-one",
            },
            [first_batch],
        )
    )
    second_path = Path(
        sync_worker._merge_batches(
            {
                "task_id": task_id,
                "user_id": user_id,
                "_generation_attempt_id": "attempt-two",
            },
            [second_batch],
        )
    )
    other_tenant_path = Path(
        sync_worker._merge_batches(
            {
                "task_id": "deadbeef-task-two",
                "user_id": "1234567890abcdef-user-two",
                "_generation_attempt_id": "attempt-one",
            },
            [
                _write_managed_batch(
                    sync_root,
                    "1234567890abcdef-user-two",
                    "deadbeef-task-two",
                    "batch_second.jsonl",
                    '{"content":"second"}\n',
                )
            ],
        )
    )

    assert len({first_path, second_path, other_tenant_path}) == 3
    assert first_path.relative_to(tmp_path / "sync").parts[:2] == (
        user_id,
        task_id,
    )
    assert other_tenant_path.relative_to(tmp_path / "sync").parts[:2] == (
        "1234567890abcdef-user-two",
        "deadbeef-task-two",
    )
    assert first_path.read_text(encoding="utf-8") == '{"content":"first"}\n'


def test_merge_accepts_legacy_truncated_batch_scope(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    user_id = "1234567890abcdef-user-one"
    task_id = "deadbeef-task-one"
    legacy_path = (
        sync_root / user_id[:16] / task_id[:8] / "batch_legacy.jsonl"
    )
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text('{"content":"legacy"}\n', encoding="utf-8")

    merged_path = Path(
        sync_worker._merge_batches(
            {
                "task_id": task_id,
                "user_id": user_id,
                "_generation_attempt_id": "legacy-attempt",
            },
            [
                {
                    "batch_id": "batch-legacy",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": str(legacy_path),
                }
            ],
        )
    )

    assert merged_path.read_text(encoding="utf-8") == '{"content":"legacy"}\n'


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user_id", "../other-user"),
        ("task_id", "other/task"),
        ("_generation_attempt_id", "..\\other-attempt"),
        ("task_id", "C:task"),
        ("user_id", "user."),
        ("user_id", "CON"),
    ],
)
def test_merge_rejects_unsafe_path_components(monkeypatch, tmp_path, field, value):
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(tmp_path / "sync"))
    batch = _write_managed_batch(
        tmp_path / "sync",
        "user-one",
        "task-one",
        "batch_document.jsonl",
        '{"content":"document"}\n',
    )
    config = {
        "task_id": "task-one",
        "user_id": "user-one",
        "_generation_attempt_id": "attempt-one",
    }
    config[field] = value

    with pytest.raises(ValueError, match="path component"):
        sync_worker._merge_batches(config, [batch])

    assert not (tmp_path / "other-user").exists()
    assert not (tmp_path / "other-attempt.jsonl").exists()


def test_save_batch_uses_full_tenant_scope_uuid_and_exclusive_create(
    monkeypatch,
    tmp_path,
):
    sync_root = tmp_path / "sync"
    user_id = "1234567890abcdef-user-one"
    task_id = "deadbeef-task-one"
    fixed_uuid = "a" * 32
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(
        sync_worker.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=fixed_uuid),
    )

    expected = sync_root / user_id / task_id / f"batch_{fixed_uuid}.jsonl"
    expected.parent.mkdir(parents=True)
    expected.write_text("preserve existing data\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sync_worker._save_batch(
            {"task_id": task_id, "user_id": user_id},
            [{"id": "1", "text": "new data"}],
        )

    assert expected.read_text(encoding="utf-8") == "preserve existing data\n"


def test_save_batch_rejects_symlinked_tenant_directory(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    user_dir = sync_root / "user-one"
    user_dir.mkdir(parents=True)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))

    def report_user_directory_as_symlink(path):
        return path == user_dir or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_user_directory_as_symlink)

    with pytest.raises(ValueError, match="symbolic link"):
        sync_worker._save_batch(
            {"task_id": "task-one", "user_id": "user-one"},
            [{"id": "1", "text": "document"}],
        )


def test_save_batch_enforces_global_storage_quota_before_file_creation(
    monkeypatch,
    tmp_path,
):
    sync_root = tmp_path / "sync"
    existing = sync_root / "other-user" / "other-task" / "batch_existing.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"123456789")
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(
            sync_storage_max_bytes_global=10,
            sync_storage_max_bytes_per_user=10,
        ),
        raising=False,
    )

    with pytest.raises(ValueError) as exc_info:
        sync_worker._save_batch(
            {"task_id": "task-one", "user_id": "user-one"},
            [{"id": "secret-id", "text": "secret-content"}],
        )

    message = str(exc_info.value)
    assert "global sync storage quota exceeded" in message.lower()
    assert str(sync_root) not in message
    assert "secret-content" not in message
    assert list(sync_root.rglob("batch_*.jsonl")) == [existing]


@pytest.mark.parametrize(
    ("max_ids", "max_bytes", "item_ids"),
    [
        (2, 10_000, ["one", "two", "three"]),
        (10, 12, ["identifier-too-large"]),
    ],
)
def test_boundary_ids_fail_closed_at_count_and_encoded_byte_limits(
    monkeypatch,
    max_ids,
    max_bytes,
    item_ids,
):
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(
            sync_boundary_max_ids=max_ids,
            sync_boundary_max_bytes=max_bytes,
        ),
        raising=False,
    )
    max_time = datetime(2026, 8, 11, 10, 0, 0)
    items = [
        {
            "id": item_id,
            "created_at": "2026-08-11T10:00:00",
        }
        for item_id in item_ids
    ]

    with pytest.raises(
        sync_worker.SyncResourceLimitExceeded,
        match="boundary deduplication limit exceeded",
    ):
        sync_worker._get_boundary_ids(items, max_time)


def test_run_once_does_not_save_or_advance_cursor_when_boundary_limit_is_exceeded(
    monkeypatch,
):
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    updates = []
    save_calls = []
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(
            sync_boundary_max_ids=2,
            sync_boundary_max_bytes=10_000,
        ),
    )
    monkeypatch.setattr(
        sync_worker,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(
        sync_worker,
        "_update_api_config_status",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        sync_worker,
        "_utcnow_naive",
        lambda: datetime(2026, 8, 15, 12, 0, 0),
    )

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def fetch_all_since(self, _since):
            return [
                {
                    "id": f"record-{index}",
                    "text": f"content-{index}",
                    "created_at": "2026-08-15T12:00:00",
                }
                for index in range(3)
            ]

    async def promote_registered(_config):
        return 0

    def unexpected_save(*_args):
        save_calls.append(True)
        raise AssertionError("batch must not be saved with an incomplete boundary")

    fake_service = SimpleNamespace(
        update_task=lambda _task_id, **kwargs: updates.append(kwargs) or True,
    )
    monkeypatch.setattr(sync_worker, "ExternalApiClient", Client)
    monkeypatch.setattr(
        sync_worker,
        "_promote_registered_batches",
        promote_registered,
    )
    monkeypatch.setattr(sync_worker, "_save_batch", unexpected_save)
    monkeypatch.setattr(sync_module, "external_sync_service", fake_service)

    with pytest.raises(
        sync_worker.SyncResourceLimitExceeded,
        match="boundary deduplication limit exceeded",
    ):
        asyncio.run(
            sync_worker.run_once(
                {
                    "task_id": "boundary-limit-task",
                    "user_id": "user-one",
                    "external_api_url": "https://api.example.test/items",
                    "external_auth_config": {},
                    "last_sync_boundary_ids": [],
                }
            )
        )

    assert save_calls == []
    assert not any(
        "last_sync_at" in update or "last_sync_boundary_ids" in update
        for update in updates
    )


def test_boundary_ids_are_unambiguous_private_hashes_with_missing_ids():
    max_time = datetime(2026, 8, 11, 10, 0, 0)
    items = [
        {
            "id": "private:a",
            "session_id": "b",
            "doc_id": "c",
            "created_at": "2026-08-11T10:00:00",
            "content": "first",
        },
        {
            "id": "private",
            "session_id": "a:b",
            "doc_id": "c",
            "created_at": "2026-08-11T10:00:00",
            "content": "second",
        },
        {
            "created_at": "2026-08-11T10:00:00",
            "content": "missing identity one",
        },
        {
            "created_at": "2026-08-11T10:00:00",
            "content": "missing identity two",
        },
        {
            "session_id": "shared-session",
            "created_at": "2026-08-11T10:00:00",
            "text": "session record one",
        },
        {
            "session_id": "shared-session",
            "created_at": "2026-08-11T10:00:01",
            "text": "session record two",
        },
    ]

    keys = sync_worker._get_boundary_ids(items, max_time)

    assert len(keys) == len(set(keys)) == 6
    assert all(len(key) == 64 for key in keys)
    assert all("private" not in key for key in keys)
    missing_identity_chunk_ids = [
        json.loads(sync_worker._serialize_batch_item(item))["metadata"]["chunk_id"]
        for item in items[2:]
    ]
    assert len(set(missing_identity_chunk_ids)) == 4


def test_boundary_read_accepts_legacy_key_and_writes_versioned_chunk_id():
    item = {
        "id": "record-1",
        "session_id": "session-1",
        "doc_id": "doc-1",
        "created_at": "2026-08-11T10:00:00",
        "text": "document",
    }
    legacy_key = "record-1:session-1:doc-1"

    assert sync_worker._item_matches_boundary(item, {legacy_key})
    metadata = json.loads(sync_worker._serialize_batch_item(item))["metadata"]
    assert metadata["chunk_id"].startswith("v2:")
    assert metadata["chunk_id"] != legacy_key

    opaque_id_item = {
        "id": "urn:uuid:record-1",
        "created_at": "2026-08-11T10:00:00",
        "text": "opaque identifier",
    }
    opaque_metadata = json.loads(
        sync_worker._serialize_batch_item(opaque_id_item)
    )["metadata"]
    assert opaque_metadata["chunk_id"].startswith("v2:")


def test_sync_storage_quota_counts_legacy_truncated_user_directory(
    monkeypatch,
    tmp_path,
):
    sync_root = tmp_path / "sync"
    user_id = "1234567890abcdef-user-one"
    legacy_file = (
        sync_root / user_id[:16] / "deadbeef" / "batch_legacy.jsonl"
    )
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_bytes(b"123456789")
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(
            sync_storage_max_bytes_global=100,
            sync_storage_max_bytes_per_user=10,
        ),
        raising=False,
    )
    task_dir = sync_worker._managed_directory(user_id, "task-one", create=True)

    with pytest.raises(ValueError, match="Per-user sync storage quota exceeded"):
        with sync_worker._reserve_sync_storage(
            user_id,
            task_dir / "batch_new.jsonl",
            2,
        ):
            pytest.fail("legacy user bytes were omitted from quota accounting")


def test_sync_storage_reservations_close_concurrent_quota_race(
    monkeypatch,
    tmp_path,
):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(
            sync_storage_max_bytes_global=15,
            sync_storage_max_bytes_per_user=15,
        ),
        raising=False,
    )
    task_dir = sync_worker._managed_directory("user-one", "task-one", create=True)

    with sync_worker._reserve_sync_storage(
        "user-one", task_dir / "batch_first.jsonl", 10
    ):
        with pytest.raises(ValueError, match="storage quota exceeded"):
            with sync_worker._reserve_sync_storage(
                "user-one", task_dir / "batch_second.jsonl", 10
            ):
                pytest.fail("overlapping reservation exceeded the quota")


def test_merge_rejects_input_bytes_before_creating_output(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(sync_generation_max_input_bytes=5),
        raising=False,
    )
    source = sync_root / "user-one" / "task-one" / "batch_source.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"content":"too large"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="generation input byte limit"):
        sync_worker._merge_batches(
            {"task_id": "task-one", "user_id": "user-one"},
            [
                {
                    "batch_id": "batch-1",
                    "task_id": "task-one",
                    "user_id": "user-one",
                    "record_count": 1,
                    "storage_path": str(source),
                }
            ],
        )

    assert not (source.parent / "merged").exists()


def test_generation_selects_fifo_prefix_within_input_byte_limit(
    monkeypatch,
    tmp_path,
):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(sync_generation_max_input_bytes=8),
        raising=False,
    )
    batches = []
    for index in range(3):
        path = sync_root / "user-one" / "task-one" / f"batch_{index}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("abc\n", encoding="utf-8", newline="")
        batches.append(
            {
                "batch_id": f"batch-{index}",
                "task_id": "task-one",
                "user_id": "user-one",
                "record_count": 1,
                "storage_path": str(path),
            }
        )

    selected = sync_worker._select_generation_batches(
        {"task_id": "task-one", "user_id": "user-one"},
        batches,
    )

    assert [batch["batch_id"] for batch in selected] == ["batch-0", "batch-1"]


def test_registered_recovery_retries_stale_batches_in_bounded_round_robin(
    monkeypatch,
):
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    batches = [
        {
            "batch_id": f"batch-{index}",
            "task_id": "task-one",
            "user_id": "user-one",
            "record_count": 1,
            "storage_path": f"/data/batch-{index}.jsonl",
            "status": "registered",
            "fetched_at": "2020-01-01T00:00:00",
        }
        for index in range(60)
    ]
    page_calls = []
    attempts = []

    def list_batches(*, limit, offset, oldest_first=False, **_kwargs):
        page_calls.append((limit, offset, oldest_first))
        return batches[offset : offset + limit], len(batches)

    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        SimpleNamespace(list_batches=list_batches),
    )
    monkeypatch.setattr(
        sync_worker,
        "_load_all_historical_documents",
        lambda _config: [],
    )

    async def fail_promotion(_config, batch, _historical_docs_cache=None):
        attempts.append(batch["batch_id"])
        return False

    monkeypatch.setattr(sync_worker, "_promote_batch_to_fetched", fail_promotion)

    asyncio.run(
        sync_worker._promote_registered_batches(
            {"task_id": "task-one", "user_id": "user-one"}
        )
    )
    asyncio.run(
        sync_worker._promote_registered_batches(
            {"task_id": "task-one", "user_id": "user-one"}
        )
    )

    assert attempts == [f"batch-{index}" for index in range(60)]
    assert page_calls == [(50, 0, True), (50, 50, True)]


def test_historical_backfill_stops_reading_inside_batch_at_byte_budget(
    monkeypatch,
    tmp_path,
):
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    batch_path = tmp_path / "batch_history.jsonl"
    lines = [
        json.dumps(
            {
                "content": f"document-{index}",
                "metadata": {"chunk_id": f"chunk-{index}"},
            }
        )
        + "\n"
        for index in range(3)
    ]
    batch_path.write_text("".join(lines), encoding="utf-8", newline="")
    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        SimpleNamespace(
            list_batches=lambda **_kwargs: (
                [{"storage_path": str(batch_path)}],
                1,
            )
        ),
    )
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(
            sync_historical_max_docs=100,
            sync_historical_max_bytes=len(lines[0].encode("utf-8")),
        ),
        raising=False,
    )

    documents = sync_worker._load_all_historical_documents(
        {"task_id": "task-history"}
    )

    assert [document.doc_id for document in documents] == ["chunk-0"]


def test_batch_reader_bounds_single_line_before_materializing_it(monkeypatch):
    read_sizes = []

    class BoundedStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            pytest.fail("reader iterated an unbounded binary line")

        def readline(self, size=-1):
            read_sizes.append(size)
            return b"x" * size

    monkeypatch.setattr(builtins, "open", lambda *_args, **_kwargs: BoundedStream())

    documents, bytes_read, capped = sync_worker._read_documents_from_batch(
        "/data/oversized.jsonl",
        max_docs=10,
        max_bytes=8,
    )

    assert documents == []
    assert bytes_read == 0
    assert capped is True
    assert read_sizes == [9]


def test_duplicate_batch_temp_file_is_removed_before_cursor_update(
    monkeypatch,
    tmp_path,
):
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    temp_path = tmp_path / "batch_duplicate_temp.jsonl"
    existing_path = tmp_path / "batch_existing.jsonl"
    temp_path.write_text('{"content":"duplicate"}\n', encoding="utf-8")
    existing_path.write_text('{"content":"existing"}\n', encoding="utf-8")
    update_calls = []

    def update_task(_task_id, **kwargs):
        update_calls.append(kwargs)
        if "last_sync_at" in kwargs:
            assert not temp_path.exists()
            raise RuntimeError("cursor update unavailable")
        return True

    fake_service = SimpleNamespace(
        update_task=update_task,
        create_batch=lambda **_kwargs: (
            {
                "batch_id": "batch-existing",
                "task_id": "task-one",
                "user_id": "user-one",
                "record_count": 1,
                "storage_path": str(existing_path),
                "status": "fetched",
            },
            False,
        ),
    )
    monkeypatch.setattr(sync_module, "external_sync_service", fake_service)
    monkeypatch.setattr(
        sync_worker,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(sync_worker, "_save_batch", lambda *_args: str(temp_path))
    monkeypatch.setattr(sync_worker, "_update_api_config_status", lambda *_args: None)

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def fetch_all_since(self, _since):
            return [
                {
                    "id": "record-1",
                    "text": "content",
                    "created_at": "2026-08-11T10:00:00",
                }
            ]

    async def promote_registered(_config):
        return 0

    monkeypatch.setattr(sync_worker, "ExternalApiClient", Client)
    monkeypatch.setattr(
        sync_worker,
        "_utcnow_naive",
        lambda: datetime(2026, 8, 11, 12, 0, 0),
    )
    monkeypatch.setattr(
        sync_worker,
        "_promote_registered_batches",
        promote_registered,
    )

    with pytest.raises(RuntimeError, match="cursor update unavailable"):
        asyncio.run(
            sync_worker.run_once(
                {
                    "task_id": "task-one",
                    "user_id": "user-one",
                    "external_api_url": "https://example.test/api",
                    "external_auth_config": {},
                    "last_sync_boundary_ids": [],
                }
            )
        )

    assert not temp_path.exists()
    assert existing_path.exists()
    assert len(update_calls) == 2


def test_merge_rejects_symlinked_batch_parent(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    batch = _write_managed_batch(
        sync_root,
        "user-one",
        "task-one",
        "batch_parent_link.jsonl",
        '{"content":"document"}\n',
    )
    user_dir = sync_root / "user-one"
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))

    def report_user_directory_as_symlink(path):
        return path == user_dir or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_user_directory_as_symlink)

    with pytest.raises(ValueError, match="symbolic link"):
        sync_worker._merge_batches(
            {
                "task_id": "task-one",
                "user_id": "user-one",
                "_generation_attempt_id": "attempt-parent-link",
            },
            [batch],
        )


def test_jsonl_stats_rejects_record_larger_than_configured_limit(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "oversized.jsonl"
    source.write_bytes(b"x" * 65 + b"\n")
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(sync_max_record_bytes=64),
    )

    with pytest.raises(sync_worker.SyncResourceLimitExceeded, match="record byte"):
        sync_worker._normalized_jsonl_stats(source)


def test_merge_removes_partial_output_when_record_grows_after_stats(
    monkeypatch,
    tmp_path,
):
    sync_root = tmp_path / "sync"
    batch = _write_managed_batch(
        sync_root,
        "user-one",
        "task-one",
        "batch_growing.jsonl",
        '{"content":"small"}\n',
    )
    source_path = Path(batch["storage_path"])
    real_stats = sync_worker._normalized_jsonl_stats
    stats_calls = 0

    def stats_then_grow(path):
        nonlocal stats_calls
        stats_calls += 1
        result = real_stats(path)
        if stats_calls == 1:
            source_path.write_bytes(b"x" * 65 + b"\n")
        return result

    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(
        sync_worker,
        "get_settings",
        lambda: _sync_limit_settings(sync_max_record_bytes=64),
    )
    monkeypatch.setattr(sync_worker, "_normalized_jsonl_stats", stats_then_grow)

    with pytest.raises(sync_worker.SyncResourceLimitExceeded, match="record byte"):
        sync_worker._merge_batches(
            {
                "task_id": "task-one",
                "user_id": "user-one",
                "_generation_attempt_id": "attempt-growing",
            },
            [batch],
        )

    assert not (
        sync_root
        / "user-one"
        / "task-one"
        / "merged"
        / "merged_attempt-growing.jsonl"
    ).exists()


@pytest.mark.parametrize("failure_kind", ["missing", "directory", "outside"])
def test_merge_rejects_unmanaged_or_nonregular_batch_sources(
    monkeypatch,
    tmp_path,
    failure_kind,
):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    managed_dir = sync_root / "user-one" / "task-one"
    managed_dir.mkdir(parents=True)
    if failure_kind == "missing":
        batch_path = managed_dir / "batch_missing.jsonl"
        expected_error = FileNotFoundError
    elif failure_kind == "directory":
        batch_path = managed_dir / "batch_directory.jsonl"
        batch_path.mkdir()
        expected_error = ValueError
    else:
        batch_path = tmp_path / "outside" / "batch_outside.jsonl"
        batch_path.parent.mkdir()
        batch_path.write_text('{"content":"outside"}\n', encoding="utf-8")
        expected_error = ValueError

    with pytest.raises(expected_error):
        sync_worker._merge_batches(
            {
                "task_id": "task-one",
                "user_id": "user-one",
                "_generation_attempt_id": f"attempt-{failure_kind}",
            },
            [
                {
                    "batch_id": f"batch-{failure_kind}",
                    "task_id": "task-one",
                    "user_id": "user-one",
                    "storage_path": str(batch_path),
                }
            ],
        )

    assert not (
        managed_dir / "merged" / f"merged_attempt-{failure_kind}.jsonl"
    ).exists()


def test_merge_rejects_batch_owned_by_another_task(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    batch = _write_managed_batch(
        sync_root,
        "user-one",
        "task-one",
        "batch_owned.jsonl",
        '{"content":"owned"}\n',
    )
    batch["task_id"] = "task-two"

    with pytest.raises(ValueError, match="does not belong"):
        sync_worker._merge_batches(
            {
                "task_id": "task-one",
                "user_id": "user-one",
                "_generation_attempt_id": "attempt-owner",
            },
            [batch],
        )


def test_merge_rejects_unreadable_batch(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    batch = _write_managed_batch(
        sync_root,
        "user-one",
        "task-one",
        "batch_unreadable.jsonl",
        '{"content":"unreadable"}\n',
    )
    original_open = builtins.open

    def deny_batch_read(file, mode="r", *args, **kwargs):
        if Path(file) == Path(batch["storage_path"]) and "r" in mode:
            raise PermissionError("batch is unreadable")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_batch_read)

    with pytest.raises(PermissionError, match="unreadable"):
        sync_worker._merge_batches(
            {
                "task_id": "task-one",
                "user_id": "user-one",
                "_generation_attempt_id": "attempt-unreadable",
            },
            [batch],
        )


def test_merge_rejects_symlink_batch(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    real_batch = _write_managed_batch(
        sync_root,
        "user-one",
        "task-one",
        "batch_real.jsonl",
        '{"content":"real"}\n',
    )
    link_path = Path(real_batch["storage_path"]).with_name("batch_link.jsonl")
    try:
        os.symlink(real_batch["storage_path"], link_path)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        sync_worker._merge_batches(
            {
                "task_id": "task-one",
                "user_id": "user-one",
                "_generation_attempt_id": "attempt-link",
            },
            [
                {
                    "batch_id": "batch-link",
                    "task_id": "task-one",
                    "user_id": "user-one",
                    "storage_path": str(link_path),
                }
            ],
        )


def test_sync_generation_handoff_binds_output_to_run_token(monkeypatch, tmp_path):
    state = _install_staging_harness(monkeypatch, tmp_path)
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )

    result = asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    task_id, output_path, _count = state.generation_outputs[0]
    run_token = result["generation_run_token"]
    expected_path = pipeline_module.resolve_generation_attempt_output_path(
        str(tmp_path / "generated" / f"generated_{task_id}.jsonl"),
        task_id,
        run_token,
    )
    assert result["pipeline_config"].run_token == run_token
    assert Path(result["pipeline_config"].output_path) == expected_path
    assert Path(output_path) == expected_path
    assert Path(output_path).name == f"generated_{task_id}.jsonl"
    assert run_token not in output_path


def test_stop_during_sync_staging_cleans_up_without_returning_handoff(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    asset_module = importlib.import_module(
        "train_factory.storage.services.dataset_asset_service"
    )
    generation_entity_module = importlib.import_module(
        "train_factory.storage.entities.generation_task_entity"
    )
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )

    def create_asset_then_stop(**kwargs):
        state.assets.add(kwargs["dataset_id"])
        task = next(iter(state.generation_tasks.values()))
        assert task["status"] == generation_entity_module.GenerationStatus.PENDING
        task["status"] = generation_entity_module.GenerationStatus.STOPPED

    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "create_asset",
        create_asset_then_stop,
    )

    with pytest.raises(RuntimeError, match="lost ownership during sync staging"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    task = next(iter(state.generation_tasks.values()))
    assert task["status"] == generation_entity_module.GenerationStatus.STOPPED
    assert [lease.release_calls for lease in state.leases] == [1]
    assert len(state.recovery_calls) == 1
    _assert_no_staging_artifacts(state)


def _create_staged_sync_generation_handoff(monkeypatch, tmp_path):
    state = _install_staging_harness(monkeypatch, tmp_path)
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    return state, asyncio.run(sync_worker._trigger_generation(_trigger_config()))


def test_stop_after_handoff_return_is_cleaned_before_manager_launch(
    monkeypatch,
    tmp_path,
):
    from train_factory.storage.entities.generation_task_entity import GenerationStatus
    from train_factory.sync import sync_manager as manager_module

    state, handoff = _create_staged_sync_generation_handoff(monkeypatch, tmp_path)
    state.generation_tasks[handoff["gen_task_id"]]["status"] = (
        GenerationStatus.STOPPED
    )
    scheduled = []

    def capture_task(coroutine, **_kwargs):
        coroutine.close()
        scheduled.append(True)
        return SimpleNamespace()

    monkeypatch.setattr(manager_module.asyncio, "create_task", capture_task)

    manager_module.SyncManager()._launch_generation(handoff)

    assert scheduled == []
    assert [lease.release_calls for lease in state.leases] == [1]
    assert len(state.recovery_calls) == 1
    assert state.generation_tasks[handoff["gen_task_id"]]["status"] == (
        GenerationStatus.STOPPED
    )
    _assert_no_staging_artifacts(state)


def test_stop_after_manager_preflight_is_cleaned_before_worker_claim(
    monkeypatch,
    tmp_path,
    request,
):
    from train_factory.api.routes import generation_routes
    from train_factory.storage.entities.generation_task_entity import GenerationStatus
    from train_factory.sync import sync_manager as manager_module

    admission_module = importlib.import_module(
        "train_factory.storage.services.background_task_admission_service"
    )
    admission = admission_module.BackgroundTaskAdmissionService(
        global_limit=1, per_user_limit=1,
    )
    monkeypatch.setattr(admission_module, "background_task_admission_service", admission)
    request.addfinalizer(admission.shutdown_async_workers)
    state, handoff = _create_staged_sync_generation_handoff(monkeypatch, tmp_path)
    generation_service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        generation_service_module.generation_task_service,
    )
    scheduled = []

    def capture_task(coroutine, **_kwargs):
        scheduled.append(coroutine)
        return SimpleNamespace()

    monkeypatch.setattr(manager_module.asyncio, "create_task", capture_task)
    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        lambda *_args, **_kwargs: pytest.fail(
            "stopped handoff must not construct a generation pipeline"
        ),
    )

    manager_module.SyncManager()._launch_generation(handoff)
    assert len(scheduled) == 1
    state.generation_tasks[handoff["gen_task_id"]]["status"] = (
        GenerationStatus.STOPPED
    )
    asyncio.run(scheduled.pop())

    assert [lease.release_calls for lease in state.leases] == [1]
    assert len(state.recovery_calls) == 1
    assert state.generation_tasks[handoff["gen_task_id"]]["status"] == (
        GenerationStatus.STOPPED
    )
    _assert_no_staging_artifacts(state)


def test_generation_attempt_dataset_names_are_unique_with_frozen_clock(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_utcnow_naive",
        lambda: datetime(2026, 8, 11, 10, 11, 12),
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )

    for _ in range(2):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert len(state.dataset_names) == 2
    assert len(set(state.dataset_names)) == 2


def test_merge_failure_removes_only_the_partial_attempt_file(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(tmp_path / "sync"))
    valid_batch = _write_managed_batch(
        tmp_path / "sync",
        "user-one",
        "task-one",
        "batch_valid.jsonl",
        '{"content":"written"}\n',
    )
    invalid_path = tmp_path / "sync" / "user-one" / "task-one" / "batch_invalid.jsonl"
    invalid_path.mkdir(parents=True)
    invalid_batch = {
        "batch_id": "batch-invalid",
        "task_id": "task-one",
        "user_id": "user-one",
        "storage_path": str(invalid_path),
    }
    config = {
        "task_id": "task-one",
        "user_id": "user-one",
        "_generation_attempt_id": "partial-attempt",
    }
    previous_attempt = (
        tmp_path
        / "sync"
        / "user-one"
        / "task-one"
        / "merged"
        / "merged_previous-attempt.jsonl"
    )
    previous_attempt.parent.mkdir(parents=True)
    previous_attempt.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="regular file"):
        sync_worker._merge_batches(
            config,
            [
                valid_batch,
                invalid_batch,
            ],
        )

    partial_path = previous_attempt.with_name("merged_partial-attempt.jsonl")
    assert not partial_path.exists()
    assert previous_attempt.read_text(encoding="utf-8") == "preserve me\n"


def test_tracking_recovery_failure_keeps_pending_task_and_staging(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    asset_module = importlib.import_module(
        "train_factory.storage.services.dataset_asset_service"
    )
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "create_asset",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("asset registration failed")
        ),
    )
    monkeypatch.setattr(
        sync_module.external_sync_service,
        "fail_generation_and_restore_batches",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("recovery unavailable")),
    )

    with pytest.raises(RuntimeError, match="asset registration failed"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert state.generation_failures == []
    assert [lease.release_calls for lease in state.leases] == [1]
    assert len(state.datasets) == 1
    assert state.lineages == state.datasets
    assert state.assets == set()
    assert all(path.exists() for path in state.merged_paths)


def test_post_tracking_failure_cleans_staging_after_successful_recovery(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )

    def commit_then_fail(**kwargs):
        state.generation_tracking_calls.append(kwargs)
        raise RuntimeError("tracking result unavailable")

    monkeypatch.setattr(
        sync_module.external_sync_service,
        "create_generation_and_claim_batches",
        commit_then_fail,
    )

    with pytest.raises(RuntimeError, match="tracking result unavailable"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert len(state.generation_tracking_calls) == 1
    assert len(state.recovery_calls) == 1
    _assert_no_staging_artifacts(state)
    assert state.generation_failures == []


def test_cleanup_failure_preserves_merged_file_for_recovery(monkeypatch, tmp_path):
    state = _install_staging_harness(monkeypatch, tmp_path)
    asset_module = importlib.import_module(
        "train_factory.storage.services.dataset_asset_service"
    )
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    def create_asset_then_fail(**kwargs):
        state.assets.add(kwargs["dataset_id"])
        raise RuntimeError("staging failed")

    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "create_asset",
        create_asset_then_fail,
    )
    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "delete_assets_for_dataset",
        lambda _dataset_id: (_ for _ in ()).throw(
            RuntimeError("metadata cleanup failed")
        ),
    )

    with pytest.raises(RuntimeError, match="staging failed"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert len(state.datasets) == 1
    assert state.assets == state.datasets
    assert all(path.exists() for path in state.merged_paths)
    assert state.generation_failures == []


def test_manager_reconcile_uses_terminal_reset_then_authoritative_read(monkeypatch):
    from train_factory.sync import sync_manager as manager_module

    events = []
    refreshed = {
        "task_id": "sync-reconcile",
        "status": "idle",
    }
    service = SimpleNamespace(
        reset_generating_task_if_no_pending_generation=lambda task_id: (
            events.append(("reset", task_id)) or True
        ),
        get_task_raw=lambda task_id: events.append(("read", task_id)) or refreshed,
    )
    monkeypatch.setattr(
        manager_module,
        "_reconcile_terminal_sync_generation",
        lambda task_id: events.append(("terminal", task_id)),
    )

    result = manager_module._reconcile_sync_generation_state(
        service,
        "sync-reconcile",
    )

    assert result is refreshed
    assert events == [
        ("terminal", "sync-reconcile"),
        ("reset", "sync-reconcile"),
        ("read", "sync-reconcile"),
    ]


@pytest.mark.parametrize(
    ("failure_step", "error_type", "expected_events"),
    (
        ("terminal", "RuntimeError", ["terminal"]),
        ("reset", "LookupError", ["terminal", "reset"]),
        ("read", "OSError", ["terminal", "reset", "read"]),
    ),
)
def test_manager_reconcile_failure_is_classified_and_redacted(
    monkeypatch,
    caplog,
    failure_step,
    error_type,
    expected_events,
):
    import logging

    from train_factory.sync import sync_manager as manager_module

    events = []
    secret = "credential-bearing-database-detail"

    def terminal(_task_id):
        events.append("terminal")
        if failure_step == "terminal":
            raise RuntimeError(secret)

    def reset(_task_id):
        events.append("reset")
        if failure_step == "reset":
            raise LookupError(secret)
        return False

    def read(_task_id):
        events.append("read")
        if failure_step == "read":
            raise OSError(secret)
        return {"task_id": "sync-sensitive-task", "status": "generating"}

    monkeypatch.setattr(manager_module, "_reconcile_terminal_sync_generation", terminal)
    service = SimpleNamespace(
        reset_generating_task_if_no_pending_generation=reset,
        get_task_raw=read,
    )
    caplog.set_level(logging.WARNING, logger=manager_module.__name__)

    with pytest.raises(
        RuntimeError,
        match="^Sync generation state could not be reconciled$",
    ) as exc_info:
        manager_module._reconcile_sync_generation_state(
            service,
            "sync-sensitive-task",
        )

    assert type(exc_info.value).__name__ == "SyncGenerationReconciliationError"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    assert events == expected_events
    assert secret not in caplog.text
    assert error_type in caplog.text
    assert "sync-sen" in caplog.text


def test_startup_sync_cleanup_preserves_unrecovered_generation(monkeypatch):
    from train_factory.api import server
    from train_factory.storage.entities.generation_task_entity import GenerationStatus

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    state = SimpleNamespace(
        generation_status=GenerationStatus.PENDING,
        generation_run_token="generation-run-token-1",
        parent_status="generating",
        atomic_reset_calls=[],
    )

    def get_all_tasks(*, status, **_kwargs):
        tasks = (
            [{"task_id": "generation-1", "status": state.generation_status}]
            if status == state.generation_status
            else []
        )
        return tasks, len(tasks)

    def get_task_raw(_task_id):
        return {
            "task_id": "generation-1",
            "status": state.generation_status,
            "run_token": state.generation_run_token,
        }

    def update_status(_task_id, status, *, expected_run_token=None, **_kwargs):
        if expected_run_token != state.generation_run_token:
            return False
        state.generation_status = status
        return True

    generation_service = SimpleNamespace(
        get_all_tasks=get_all_tasks,
        get_task_raw=get_task_raw,
        update_status=update_status,
    )
    sync_service = SimpleNamespace(
        fail_generation_and_restore_batches=lambda *_args: (
            _ for _ in ()
        ).throw(RuntimeError("recovery unavailable")),
        list_tasks=lambda **_kwargs: (
            [{"task_id": "sync-1", "status": state.parent_status}],
            1,
        ),
        has_pending_generation=lambda _task_id: pytest.fail(
            "startup cleanup used a non-atomic pending check"
        ),
        reset_generating_task_if_no_pending_generation=lambda task_id: (
            state.atomic_reset_calls.append(task_id) or False
        ),
        update_task=lambda _task_id, **kwargs: setattr(
            state, "parent_status", kwargs["status"]
        )
        or True,
    )
    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        generation_service,
    )
    monkeypatch.setattr(sync_module, "external_sync_service", sync_service)

    server.cleanup_orphan_generation_tasks()
    server.cleanup_orphan_sync_tasks()

    assert state.generation_status == GenerationStatus.PENDING
    assert state.parent_status == "generating"
    assert state.atomic_reset_calls == ["sync-1"]


def test_stopped_generation_restore_uses_atomic_tracking_recovery(monkeypatch):
    from train_factory.storage.entities.generation_task_entity import GenerationStatus

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    recoveries = []
    sync_service = SimpleNamespace(
        list_generations=lambda **_kwargs: (
            [
                {"generation_task_id": "generation-1"},
            ],
            1,
        ),
        list_batches=lambda **_kwargs: pytest.fail(
            "stopped recovery searched batch status instead of tracking"
        ),
        fail_generation_and_restore_batches=lambda task_id, reason: (
            recoveries.append((task_id, reason))
            or {
                "tracking_found": True,
                "recovered": True,
                "restored_batch_count": 2,
            }
        ),
        update_batch_status=lambda *_args, **_kwargs: pytest.fail(
            "stopped recovery used non-atomic per-batch updates"
        ),
    )
    monkeypatch.setattr(sync_module, "external_sync_service", sync_service)
    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        SimpleNamespace(
            get_task=lambda _task_id: {"status": GenerationStatus.STOPPED}
        ),
    )

    restored = sync_worker._restore_stopped_queued_batches("sync-1")

    assert restored == 2
    assert recoveries == [
        ("generation-1", "Stopped generation batches restored for retry")
    ]


def test_generation_trigger_rejects_existing_pending_generation(monkeypatch):
    from train_factory.storage.services.background_task_admission_service import (
        BackgroundTaskAlreadyExecuting,
    )

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    fake_sync_service = SimpleNamespace(
        has_pending_generation=lambda _task_id: True,
        list_generations=lambda **_kwargs: ([], 0),
        update_task=lambda *_args, **_kwargs: pytest.fail(
            "active generation status was overwritten"
        ),
    )
    monkeypatch.setattr(sync_module, "external_sync_service", fake_sync_service)

    with pytest.raises(BackgroundTaskAlreadyExecuting, match="already pending"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))


def test_generation_no_batch_idle_transition_is_atomic(monkeypatch):
    from train_factory.storage.services.background_task_admission_service import (
        BackgroundTaskAlreadyExecuting,
    )

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    updates = []

    def update_task(_task_id, **kwargs):
        if kwargs["status"] == "idle":
            pytest.fail("worker used a non-atomic idle transition")
        updates.append(kwargs["status"])

    fake_sync_service = SimpleNamespace(
        list_generations=lambda **_kwargs: ([], 0),
        has_pending_generation=lambda _task_id: False,
        update_task=update_task,
        get_pending_batches=lambda _task_id: [],
        get_all_completed_generation_datasets=lambda _task_id: [
            {"generation_task_id": "completed"}
        ],
        has_enabled_completed_generation=lambda _task_id: True,
        reset_generating_task_if_no_pending_generation=lambda _task_id: False,
    )
    monkeypatch.setattr(sync_module, "external_sync_service", fake_sync_service)

    with pytest.raises(BackgroundTaskAlreadyExecuting, match="already pending"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert updates == ["generating"]


@pytest.mark.parametrize("entrypoint", ["run_once", "trigger_generation"])
def test_manager_error_handler_preserves_unrecovered_generation(
    monkeypatch,
    entrypoint,
):
    from train_factory.sync import sync_manager as manager_module

    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    updates = []
    observed_statuses = iter(["idle", "generating"])
    service = SimpleNamespace(
        get_task_raw=lambda _task_id: {
            "task_id": "sync-1",
            "status": next(observed_statuses),
        },
        has_pending_generation=lambda _task_id: pytest.fail(
            "manager used a non-atomic pending check"
        ),
        transition_task_if_no_pending_generation=lambda *_args, **_kwargs: False,
        update_task=lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(sync_module, "external_sync_service", service)
    manager = manager_module.SyncManager()

    async def fail_cycle(*_args, **_kwargs):
        raise RuntimeError("recovery unavailable")

    monkeypatch.setattr(manager, "_run_cycle_with_handoff", fail_cycle)

    with pytest.raises(RuntimeError, match="recovery unavailable"):
        asyncio.run(getattr(manager, entrypoint)("sync-1"))

    assert updates == []


@pytest.mark.parametrize("transition_result", [False, RuntimeError("database unavailable")])
def test_manager_error_recovery_is_fail_closed(monkeypatch, transition_result):
    from train_factory.sync import sync_manager as manager_module

    updates = []

    transitions = []

    def transition(*args, **kwargs):
        transitions.append((args, kwargs))
        if isinstance(transition_result, Exception):
            raise transition_result
        return transition_result

    service = SimpleNamespace(
        has_pending_generation=lambda _task_id: pytest.fail(
            "manager used a non-atomic pending check"
        ),
        transition_task_if_no_pending_generation=transition,
        update_task=lambda *args, **kwargs: updates.append((args, kwargs)),
    )

    assert not manager_module._update_task_sync(service, "sync-1")
    assert not manager_module._set_task_error(service, "sync-1", "failed")
    assert updates == []
    assert len(transitions) == 2


def test_public_task_operation_lock_serializes_external_lifecycle_work():
    from train_factory.sync.sync_manager import SyncManager

    async def exercise_lock():
        manager = SyncManager()
        entered = []

        async def contender():
            async with manager.task_operation_lock("sync-1"):
                entered.append("contender")

        async with manager.task_operation_lock("sync-1"):
            pending = asyncio.create_task(contender())
            await asyncio.sleep(0)
            assert entered == []
        await pending
        assert entered == ["contender"]

    asyncio.run(exercise_lock())


def test_unlaunched_handoff_recovery_failure_keeps_generation_active(
    monkeypatch,
):
    from train_factory.sync.sync_manager import SyncManager

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    events = []
    release_calls = []
    recovery_attempts = 0

    def recover(*_args):
        nonlocal recovery_attempts
        recovery_attempts += 1
        events.append("recover")
        if recovery_attempts == 1:
            raise RuntimeError("recovery unavailable")
        return {"tracking_found": True, "recovered": True}

    def finalize(*_args, expected_run_token=None, **_kwargs):
        assert expected_run_token == "generation-run-token-1"
        events.append("finalize")
        return True

    monkeypatch.setattr(
        sync_module.external_sync_service,
        "fail_generation_and_restore_batches",
        recover,
    )
    monkeypatch.setattr(
        generation_module.generation_task_service,
        "update_status",
        finalize,
    )
    monkeypatch.setattr(
        SyncManager,
        "_cleanup_unlaunched_generation_staging",
        lambda _result: events.append("cleanup"),
        raising=False,
    )
    result = {
        "gen_task_id": "generation-1",
        "generation_run_token": "generation-run-token-1",
        "sync_task_id": "sync-1",
        "user_id": "user-1",
        "raw_dataset_id": "raw-dataset-1",
        "merged_path": "/data/merged.jsonl",
        "execution_lease": SimpleNamespace(
            release=lambda: release_calls.append(True)
        ),
    }

    SyncManager._finalize_unlaunched_generation(result, "launch failed")
    assert events == ["recover"]

    SyncManager._finalize_unlaunched_generation(result, "launch failed")
    assert events == ["recover", "recover", "cleanup", "finalize"]
    assert release_calls == [True, True]


def test_unlaunched_handoff_recovered_false_keeps_generation_active(
    monkeypatch,
):
    from train_factory.sync.sync_manager import SyncManager

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    events = []

    def finalize(*_args, expected_run_token=None, **_kwargs):
        assert expected_run_token == "generation-run-token-1"
        events.append("finalize")
        return True

    monkeypatch.setattr(
        sync_module.external_sync_service,
        "fail_generation_and_restore_batches",
        lambda *_args: events.append("recover")
        or {"tracking_found": True, "recovered": False},
    )
    monkeypatch.setattr(
        generation_module.generation_task_service,
        "update_status",
        finalize,
    )
    monkeypatch.setattr(
        SyncManager,
        "_cleanup_unlaunched_generation_staging",
        lambda _result: events.append("cleanup"),
        raising=False,
    )

    SyncManager._finalize_unlaunched_generation(
        {
            "gen_task_id": "generation-1",
            "generation_run_token": "generation-run-token-1",
            "execution_lease": SimpleNamespace(
                release=lambda: events.append("release")
            ),
        },
        "launch failed",
    )

    assert events == ["release", "recover"]


def test_unlaunched_handoff_cleanup_failure_keeps_durable_pending_marker(
    monkeypatch,
):
    from train_factory.sync.sync_manager import SyncManager

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    events = []
    recovery_attempts = 0
    cleanup_attempts = 0

    def recover(*_args):
        nonlocal recovery_attempts
        recovery_attempts += 1
        events.append("recover")
        if recovery_attempts == 1:
            return {"tracking_found": True, "recovered": True}
        return {
            "tracking_found": True,
            "recovered": False,
            "already_recovered": True,
            "reconciled": True,
        }

    def cleanup(_result):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        events.append("cleanup")
        if cleanup_attempts == 1:
            raise RuntimeError("cleanup unavailable")

    def finalize(*_args, expected_run_token=None, **_kwargs):
        assert expected_run_token == "generation-run-token-1"
        events.append("finalize")
        return True

    monkeypatch.setattr(
        sync_module.external_sync_service,
        "fail_generation_and_restore_batches",
        recover,
    )
    monkeypatch.setattr(
        generation_module.generation_task_service,
        "update_status",
        finalize,
    )
    monkeypatch.setattr(
        SyncManager,
        "_cleanup_unlaunched_generation_staging",
        cleanup,
    )
    result = {
        "gen_task_id": "generation-1",
        "generation_run_token": "generation-run-token-1",
        "sync_task_id": "sync-1",
        "user_id": "user-1",
        "raw_dataset_id": "raw-dataset-1",
        "merged_path": "/data/merged.jsonl",
    }

    SyncManager._finalize_unlaunched_generation(result, "launch failed")
    assert events == ["recover", "cleanup"]

    SyncManager._finalize_unlaunched_generation(result, "launch failed")
    assert events == ["recover", "cleanup", "recover", "cleanup", "finalize"]


def test_missing_managed_merged_directory_is_already_clean(monkeypatch, tmp_path):
    sync_root = tmp_path / "sync"
    monkeypatch.setattr(sync_worker, "SYNC_DATA_DIR", str(sync_root))
    merged_path = sync_root / "user-1" / "sync-1" / "merged" / "merged_a.jsonl"

    assert not sync_worker._delete_managed_merged_file(
        "sync-1",
        "user-1",
        str(merged_path),
    )


def test_staging_cleanup_retries_orphan_metadata_without_dataset_row(monkeypatch):
    from train_factory.sync.sync_manager import SyncManager

    asset_module = importlib.import_module(
        "train_factory.storage.services.dataset_asset_service"
    )
    dataset_module = importlib.import_module(
        "train_factory.storage.services.dataset_service"
    )
    lineage_module = importlib.import_module(
        "train_factory.storage.services.dataset_lineage_service"
    )
    events = []
    monkeypatch.setattr(
        dataset_module.dataset_service,
        "get_dataset",
        lambda _dataset_id: None,
    )
    monkeypatch.setattr(
        dataset_module.dataset_service,
        "delete_dataset",
        lambda _dataset_id: pytest.fail("missing dataset was deleted again"),
    )
    monkeypatch.setattr(
        lineage_module.dataset_lineage_service,
        "delete_edges_for_dataset",
        lambda dataset_id: events.append(("lineage", dataset_id)),
    )
    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "delete_assets_for_dataset",
        lambda dataset_id: events.append(("assets", dataset_id)),
    )
    monkeypatch.setattr(
        sync_worker,
        "_delete_managed_merged_file",
        lambda task_id, user_id, path: events.append(
            ("file", task_id, user_id, path)
        ),
    )

    SyncManager._cleanup_unlaunched_generation_staging(
        {
            "sync_task_id": "sync-1",
            "user_id": "user-1",
            "raw_dataset_id": "raw-dataset-1",
            "merged_path": "/data/merged.jsonl",
        }
    )

    assert events == [
        ("lineage", "raw-dataset-1"),
        ("assets", "raw-dataset-1"),
        ("file", "sync-1", "user-1", "/data/merged.jsonl"),
    ]


def test_startup_cleanup_retries_tracking_before_stopping_generation(
    monkeypatch,
):
    from train_factory.api import server
    from train_factory.storage.entities.generation_task_entity import GenerationStatus

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    publication_module = importlib.import_module(
        "train_factory.storage.services.generation_publication_service"
    )
    state = SimpleNamespace(
        status=GenerationStatus.PENDING,
        run_token="generation-run-token-1",
        recovery_attempts=0,
    )
    events = []
    claims = []
    compensations = []

    def get_all_tasks(*, status, **_kwargs):
        tasks = (
            [{"task_id": "generation-1", "status": state.status}]
            if status == state.status
            else []
        )
        return tasks, len(tasks)

    def get_task_raw(_task_id):
        return {
            "task_id": "generation-1",
            "status": state.status,
            "run_token": state.run_token,
        }

    def claim_orphan_recovery(
        task_id,
        *,
        expected_status,
        expected_run_token,
    ):
        assert expected_status == state.status
        assert expected_run_token == state.run_token
        claims.append((task_id, expected_status, expected_run_token))
        state.status = GenerationStatus.RECOVERING
        return {
            "run_token": state.run_token,
            "source_status": expected_status,
        }

    def finish_orphan_recovery(
        _task_id,
        *,
        expected_run_token,
        terminal_status,
        **_kwargs,
    ):
        assert state.status == GenerationStatus.RECOVERING
        assert expected_run_token == state.run_token
        events.append("finalize")
        state.status = terminal_status
        return True

    def recover(*_args):
        state.recovery_attempts += 1
        events.append("recover")
        if state.recovery_attempts == 1:
            raise RuntimeError("recovery unavailable")
        return {"tracking_found": True, "recovered": True}

    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        SimpleNamespace(
            get_all_tasks=get_all_tasks,
            get_task_raw=get_task_raw,
            claim_orphan_recovery=claim_orphan_recovery,
            finish_orphan_recovery=finish_orphan_recovery,
        ),
    )
    monkeypatch.setattr(
        sync_module.external_sync_service,
        "fail_generation_and_restore_batches",
        recover,
    )
    monkeypatch.setattr(
        sync_module.external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        publication_module.generation_publication_service,
        "compensate_attempt",
        lambda **kwargs: compensations.append(kwargs) or True,
    )

    server.cleanup_orphan_generation_tasks()
    assert state.status == GenerationStatus.RECOVERING
    assert events == ["recover"]
    assert claims == [
        (
            "generation-1",
            GenerationStatus.PENDING,
            "generation-run-token-1",
        )
    ]

    server.cleanup_orphan_generation_tasks()
    assert state.status == GenerationStatus.FAILED
    assert events == ["recover", "recover", "finalize"]
    assert len(claims) == 1
    assert compensations == [
        {
            "task_id": "generation-1",
            "expected_run_token": "generation-run-token-1",
            "recovery_run_token": "generation-run-token-1",
            "user_id": None,
        }
    ]


def test_startup_retries_staging_cleanup_after_tracking_was_recovered(
    monkeypatch,
):
    from train_factory.api import server
    from train_factory.storage.entities.generation_task_entity import GenerationStatus
    from train_factory.sync.sync_manager import SyncManager

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    publication_module = importlib.import_module(
        "train_factory.storage.services.generation_publication_service"
    )
    state = SimpleNamespace(
        status=GenerationStatus.PENDING,
        run_token="generation-run-token-1",
        cleanup_attempts=0,
    )
    events = []
    claims = []
    compensations = []
    task = {
        "task_id": "generation-1",
        "status": GenerationStatus.PENDING,
        "user_id": "user-1",
        "source_dataset_id": "raw-dataset-1",
        "input_path": "/data/merged.jsonl",
    }

    def get_all_tasks(*, status, **_kwargs):
        tasks = [task] if status == state.status else []
        return tasks, len(tasks)

    def get_task_raw(_task_id):
        return {
            **task,
            "status": state.status,
            "run_token": state.run_token,
        }

    def claim_orphan_recovery(
        task_id,
        *,
        expected_status,
        expected_run_token,
    ):
        assert expected_status == state.status
        assert expected_run_token == state.run_token
        claims.append((task_id, expected_status, expected_run_token))
        state.status = GenerationStatus.RECOVERING
        return {
            "run_token": state.run_token,
            "source_status": expected_status,
        }

    def finish_orphan_recovery(
        _task_id,
        *,
        expected_run_token,
        terminal_status,
        **_kwargs,
    ):
        assert state.status == GenerationStatus.RECOVERING
        assert expected_run_token == state.run_token
        events.append("finalize")
        state.status = terminal_status
        return True

    def cleanup(payload):
        state.cleanup_attempts += 1
        events.append(("cleanup", payload))
        if state.cleanup_attempts == 1:
            raise RuntimeError("cleanup unavailable")

    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        SimpleNamespace(
            get_all_tasks=get_all_tasks,
            get_task_raw=get_task_raw,
            claim_orphan_recovery=claim_orphan_recovery,
            finish_orphan_recovery=finish_orphan_recovery,
        ),
    )
    monkeypatch.setattr(
        sync_module.external_sync_service,
        "fail_generation_and_restore_batches",
        lambda *_args: events.append("recover")
        or {
            "tracking_found": True,
            "recovered": False,
            "already_recovered": True,
            "reconciled": True,
            "task_id": "sync-1",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        sync_module.external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        SyncManager,
        "_cleanup_unlaunched_generation_staging",
        cleanup,
    )
    monkeypatch.setattr(
        publication_module.generation_publication_service,
        "compensate_attempt",
        lambda **kwargs: compensations.append(kwargs) or True,
    )

    server.cleanup_orphan_generation_tasks()
    assert state.status == GenerationStatus.RECOVERING
    assert claims == [
        (
            "generation-1",
            GenerationStatus.PENDING,
            "generation-run-token-1",
        )
    ]
    assert events[:2] == ["recover", ("cleanup", {
        "gen_task_id": "generation-1",
        "sync_task_id": "sync-1",
        "user_id": "user-1",
        "raw_dataset_id": "raw-dataset-1",
        "merged_path": "/data/merged.jsonl",
    })]

    server.cleanup_orphan_generation_tasks()
    assert state.status == GenerationStatus.FAILED
    assert len(claims) == 1
    assert compensations == [
        {
            "task_id": "generation-1",
            "expected_run_token": "generation-run-token-1",
            "recovery_run_token": "generation-run-token-1",
            "user_id": "user-1",
        }
    ]
    assert events[-3:] == [
        "recover",
        ("cleanup", {
            "gen_task_id": "generation-1",
            "sync_task_id": "sync-1",
            "user_id": "user-1",
            "raw_dataset_id": "raw-dataset-1",
            "merged_path": "/data/merged.jsonl",
        }),
        "finalize",
    ]


@pytest.mark.parametrize(
    (
        "generation_mode",
        "output_dataset_id",
        "deep_eval_dataset_id",
        "output_sample_count",
        "expected_callback",
    ),
    [
        ("doc_to_training", "dataset-1", None, 17, "complete"),
        ("doc_to_eval", None, "deep-eval-dataset-1", 17, "complete"),
        ("doc_to_eval", None, None, 17, "fail"),
        ("doc_to_eval", None, None, 0, "complete"),
        ("doc_to_training", None, None, 17, "fail"),
        ("unknown", "dataset-1", None, 17, "fail"),
    ],
)
def test_startup_reconciles_completed_generation_tracking(
    monkeypatch,
    generation_mode,
    output_dataset_id,
    deep_eval_dataset_id,
    output_sample_count,
    expected_callback,
):
    from train_factory.api import server
    from train_factory.enums.sync_status import SyncGenerationStatus
    from train_factory.storage.entities.generation_task_entity import GenerationStatus

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    generation_routes_module = importlib.import_module(
        "train_factory.api.routes.generation_routes"
    )
    level2_module = importlib.import_module("train_factory.sync.level2_handler")
    callbacks = []

    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        SimpleNamespace(
            get_task=lambda _task_id: {
                "task_id": "generation-1",
                "status": GenerationStatus.COMPLETED,
                "generation_mode": generation_mode,
                "output_dataset_id": output_dataset_id,
                "deep_eval_dataset_id": deep_eval_dataset_id,
                "output_sample_count": output_sample_count,
                "user_id": "user-1",
            },
            get_all_tasks=lambda **_kwargs: ([], 0),
            update_status=lambda *_args, **_kwargs: pytest.fail(
                "completed generation was rewritten"
            ),
        ),
    )
    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        SimpleNamespace(
            list_pending_generations=lambda **_kwargs: (
                [
                    {
                        "id": 1,
                        "generation_task_id": "generation-1",
                        "status": SyncGenerationStatus.PENDING,
                        "user_id": "user-1",
                    }
                ],
                1,
            ),
            get_generation_by_task_id=lambda _task_id: {
                "task_id": "sync-1",
                "user_id": "user-1",
            },
            fail_generation_and_restore_batches=lambda task_id, reason: (
                callbacks.append(("fail", task_id, reason))
                or {"tracking_found": True, "recovered": True}
            ),
        ),
    )
    monkeypatch.setattr(
        generation_routes_module.dataset_service,
        "get_dataset",
        lambda dataset_id: (
            {
                "dataset_id": dataset_id,
                "user_id": "user-1",
                "source_task_type": "generation",
                "source_task_id": "generation-1",
                "num_rows": output_sample_count,
            }
            if dataset_id in {output_dataset_id, deep_eval_dataset_id} - {None}
            else None
        ),
    )
    monkeypatch.setattr(
        level2_module,
        "on_generation_completed",
        lambda generation_task_id, output_dataset_id, output_sample_count: callbacks.append(
            ("complete", generation_task_id, output_dataset_id, output_sample_count)
        )
        or {"tracking_found": True, "completed": True},
    )

    server.cleanup_orphan_generation_tasks()

    assert callbacks[0][0] == expected_callback
    assert callbacks[0][1] == "generation-1"


def test_startup_pending_tracking_scan_keyset_pages_beyond_previous_cap(monkeypatch):
    from train_factory.api import server
    from train_factory.storage.entities.generation_task_entity import GenerationStatus

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    generation_routes_module = importlib.import_module(
        "train_factory.api.routes.generation_routes"
    )
    level2_module = importlib.import_module("train_factory.sync.level2_handler")
    pending = [
        {"id": index + 1, "generation_task_id": f"generation-{index}"}
        for index in range(5)
    ]
    page_calls = []
    completed = []

    def list_pending_generations(*, limit, after_id=None, **_kwargs):
        page_calls.append((limit, after_id))
        remaining = [
            row
            for row in pending
            if row["id"] > (after_id or 0)
            and row["generation_task_id"] not in completed
        ]
        return remaining[:limit], len(remaining)

    monkeypatch.setattr(server, "_SYNC_RECONCILIATION_PAGE_SIZE", 2, raising=False)
    monkeypatch.setattr(
        server, "_MAX_SYNC_STARTUP_RECONCILIATIONS", 3, raising=False
    )
    monkeypatch.setattr(
        generation_module,
        "generation_task_service",
        SimpleNamespace(
            get_task=lambda task_id: {
                "task_id": task_id,
                "status": GenerationStatus.COMPLETED,
                "generation_mode": "doc_to_eval",
                "output_dataset_id": None,
                "deep_eval_dataset_id": f"deep-{task_id}",
                "output_sample_count": 1,
                "user_id": "user-1",
            },
            get_all_tasks=lambda **_kwargs: ([], 0),
        ),
    )
    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        SimpleNamespace(
            list_pending_generations=list_pending_generations,
            get_generation_by_task_id=lambda _task_id: {
                "task_id": "sync-1",
                "user_id": "user-1",
            },
            fail_generation_and_restore_batches=lambda *_args: pytest.fail(
                "completed eval tracking was failed"
            ),
        ),
    )
    monkeypatch.setattr(
        generation_routes_module.dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "user_id": "user-1",
            "source_task_type": "generation",
            "source_task_id": dataset_id.removeprefix("deep-"),
            "num_rows": 1,
        },
    )
    monkeypatch.setattr(
        level2_module,
        "on_generation_completed",
        lambda generation_task_id, **_kwargs: completed.append(generation_task_id)
        or {"tracking_found": True, "completed": True},
    )

    server.cleanup_orphan_generation_tasks()

    assert page_calls == [(2, None), (2, 2), (2, 4), (2, 5)]
    assert completed == [
        "generation-0",
        "generation-1",
        "generation-2",
        "generation-3",
        "generation-4",
    ]


def test_generation_callback_failure_preserves_pending_tracking(monkeypatch):
    generation_routes = importlib.import_module(
        "train_factory.api.routes.generation_routes"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    level2_module = importlib.import_module("train_factory.sync.level2_handler")

    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        SimpleNamespace(
            get_generation_by_task_id=lambda _task_id: {"task_id": "sync-1"},
            update_task=lambda *_args, **_kwargs: pytest.fail(
                "callback fallback bypassed pending tracking"
            ),
        ),
    )
    monkeypatch.setattr(
        level2_module,
        "on_generation_completed",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    assert generation_routes._finalize_sync_generation_tracking(
        "generation-1",
        generation_mode="doc_to_training",
        output_dataset_id="dataset-1",
        output_sample_count=3,
    ) is False


def test_generation_delete_rejects_pending_sync_tracking(monkeypatch):
    from fastapi import HTTPException

    generation_routes = importlib.import_module(
        "train_factory.api.routes.generation_routes"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: {
            "task_id": "generation-1",
            "status": "completed",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        SimpleNamespace(
            get_generation_by_task_id=lambda _task_id: {"status": "pending"}
        ),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "delete_task",
        lambda _task_id: pytest.fail("pending sync generation was deleted"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                "generation-1",
                current_user={"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 409


@pytest.mark.parametrize("tracking_status", ["pending", "failed"])
def test_generation_restart_rejects_nonrestartable_sync_tracking(
    monkeypatch,
    tracking_status,
):
    from fastapi import BackgroundTasks, HTTPException

    generation_routes = importlib.import_module(
        "train_factory.api.routes.generation_routes"
    )
    sync_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: {
            "task_id": "generation-1",
            "status": "completed",
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        sync_module,
        "external_sync_service",
        SimpleNamespace(
            get_generation_by_task_id=lambda _task_id: {"status": tracking_status}
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda _task: pytest.fail("non-restartable sync generation was restarted"),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "get_task_raw",
        lambda _task_id: {
            "task_id": "generation-1",
            "status": "completed",
            "run_token": "generation-run-token-1",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.restart_task(
                "generation-1",
                BackgroundTasks(),
                current_user={"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 409


def test_dataset_post_commit_failure_uses_stable_id_for_cleanup(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    dataset_module = importlib.import_module(
        "train_factory.storage.services.dataset_service"
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    created_ids = []

    def commit_then_fail(**kwargs):
        dataset_id = kwargs.get("dataset_id")
        created_ids.append(dataset_id)
        state.datasets.add(dataset_id)
        raise RuntimeError("dataset result unavailable")

    monkeypatch.setattr(
        dataset_module.dataset_service,
        "create_dataset",
        commit_then_fail,
    )

    with pytest.raises(RuntimeError, match="dataset result unavailable"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert len(created_ids) == 1
    assert isinstance(created_ids[0], str) and created_ids[0]
    _assert_no_staging_artifacts(state)
    assert [lease.release_calls for lease in state.leases] == [1]


def test_generation_task_post_commit_failure_uses_stable_id_for_finalization(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    persisted_statuses = {}

    def commit_then_fail(**kwargs):
        task_id = kwargs.get("task_id")
        persisted_statuses[task_id] = "pending"
        state.generation_tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "run_token": "generation-run-token-post-commit",
        }
        raise RuntimeError("generation result unavailable")

    def update_status(
        task_id,
        status,
        *_args,
        expected_run_token=None,
        **_kwargs,
    ):
        if expected_run_token != state.generation_tasks[task_id]["run_token"]:
            return False
        state.generation_tasks[task_id]["status"] = status
        persisted_statuses[task_id] = status
        return True

    monkeypatch.setattr(
        generation_module.generation_task_service,
        "create_task",
        commit_then_fail,
    )
    monkeypatch.setattr(
        generation_module.generation_task_service,
        "update_status",
        update_status,
    )

    with pytest.raises(RuntimeError, match="generation result unavailable"):
        asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    assert len(persisted_statuses) == 1
    task_id, status = next(iter(persisted_statuses.items()))
    assert isinstance(task_id, str) and task_id
    assert status == "failed"
    _assert_no_staging_artifacts(state)


def test_generation_task_service_accepts_caller_supplied_stable_id(
    monkeypatch,
    tmp_path,
):
    from contextlib import contextmanager

    from sqlmodel import Session, SQLModel, create_engine

    from train_factory.storage.entities.generation_task_entity import GenerationTaskDB

    generation_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'stable-generation-id.db'}")
    SQLModel.metadata.create_all(engine, tables=[GenerationTaskDB.__table__])

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(generation_module, "get_session", test_session)
    service = generation_module.GenerationTaskService()

    created = service.create_task(
        task_id="stable-generation-id",
        task_name="stable",
        input_path="/managed/input.jsonl",
        llm_config={},
        steps_config={},
    )

    assert created["task_id"] == "stable-generation-id"
    assert service.get_task("stable-generation-id")["task_id"] == (
        "stable-generation-id"
    )


def test_repeated_model_resolution_failures_do_not_leak_staging(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("model resolution failed")
        ),
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="model resolution failed"):
            asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    _assert_no_staging_artifacts(state)
    assert state.generation_tracking_calls == []
    assert state.recovery_calls == []
    assert state.leases == []


def test_repeated_asset_registration_failures_do_not_leak_staging(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    asset_module = importlib.import_module(
        "train_factory.storage.services.dataset_asset_service"
    )
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )

    def fail_after_asset_insert(**kwargs):
        state.assets.add(kwargs["dataset_id"])
        raise RuntimeError("asset registration failed")

    monkeypatch.setattr(
        asset_module.dataset_asset_service,
        "create_asset",
        fail_after_asset_insert,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="asset registration failed"):
            asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    _assert_no_staging_artifacts(state)
    assert len(state.generation_tracking_calls) == 2
    assert len(state.recovery_calls) == 2
    assert [lease.release_calls for lease in state.leases] == [1, 1]


def test_repeated_validation_failures_rollback_staging_and_generation_task(
    monkeypatch,
    tmp_path,
):
    state = _install_staging_harness(monkeypatch, tmp_path)
    pipeline_module = importlib.import_module("train_factory.generation.pipeline")
    monkeypatch.setattr(
        sync_worker,
        "_resolve_config_from_gen_cfg",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_generation_resource_config",
        lambda _config: (_ for _ in ()).throw(RuntimeError("validation failed")),
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="validation failed"):
            asyncio.run(sync_worker._trigger_generation(_trigger_config()))

    _assert_no_staging_artifacts(state)
    assert state.generation_tracking_calls == []
    assert state.leases == []
    assert state.generation_failures == []
