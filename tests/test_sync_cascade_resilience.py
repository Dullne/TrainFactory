"""Failure-boundary regressions for external-sync cascade deletion."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from train_factory.api.routes import sync_routes
from train_factory.storage.services.dataset_asset_service import (
    dataset_asset_service,
)
from train_factory.storage.services.dataset_lineage_service import (
    dataset_lineage_service,
)
from train_factory.storage.services.dataset_service import dataset_service
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    background_task_admission_service,
)
from train_factory.storage.services.deep_evaluation_task_service import (
    deep_evaluation_task_service,
)
from train_factory.storage.services.evaluation_task_service import (
    evaluation_task_service,
)
from train_factory.storage.services.external_sync_service import (
    external_sync_service,
)
from train_factory.storage.services.generation_task_service import (
    generation_task_service,
)
from train_factory.storage.services.milvus_collection_service import (
    milvus_collection_service,
)
from train_factory.storage.services.training_task_service import (
    training_task_service,
)


@pytest.fixture(autouse=True)
def _isolated_cascade_services(monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "list_external_storage_reference_consumers",
        lambda *_args, **_kwargs: [],
        raising=False,
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_fences",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_collections",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        lambda **_kwargs: ([], 0),
        raising=False,
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_dataset_links",
        lambda _dataset_ids: [],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "get_linked_datasets",
        lambda _collection_name: [],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **_kwargs: SimpleNamespace(
            newly_fenced=tuple(sorted(names)),
            created_placeholders=(),
        ),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "restore_deletion_fences",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_collection_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_artifact_reference_consumers",
        lambda *_args, **_kwargs: [],
        raising=False,
    )
    monkeypatch.setattr(
        deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_active_collection_consumers",
        lambda *_args, **_kwargs: [],
    )


def _install_fake_milvus_client(
    monkeypatch,
    *,
    remote_names=(),
    dropped=None,
):
    from train_factory.generation.clients import milvus_client as milvus_module

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeClient:
        def __init__(self, _config):
            pass

        def connect(self):
            return None

        def list_collections(self):
            return list(remote_names)

        def drop_collection(self, collection_name):
            if dropped is not None:
                dropped.append(collection_name)

        def close(self):
            return None

    monkeypatch.setattr(milvus_module, "MilvusConfig", FakeConfig)
    monkeypatch.setattr(milvus_module, "MilvusClient", FakeClient)


@pytest.mark.parametrize(
    ("consumer_ids", "should_reject"),
    [
        (["sync-consumer-owner"], False),
        (["sync-consumer-owner", "sync-consumer-other"], True),
    ],
)
def test_milvus_preflight_excludes_current_sync_and_rejects_other_sync_consumers(
    monkeypatch,
    consumer_ids,
    should_reject,
):
    task_id = "sync-consumer-owner"
    collection_name = f"tf_sync_v2_{task_id[:8]}_current"
    events = []
    dropped = []
    _install_fake_milvus_client(monkeypatch, dropped=dropped)

    def acquire(names, *, deletion_owner, **_kwargs):
        normalized = tuple(sorted(names))
        events.append(("fence", normalized, deletion_owner))
        return SimpleNamespace(
            newly_fenced=normalized,
            created_placeholders=(collection_name,),
        )

    def list_sync_consumers(names):
        events.append(("sync-consumers", tuple(sorted(names))))
        return list(consumer_ids)

    def restore(names, *, deletion_owner, created_placeholders):
        events.append(
            (
                "restore",
                tuple(names),
                deletion_owner,
                tuple(created_placeholders),
            )
        )
        return True

    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        acquire,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_active_collection_consumers",
        list_sync_consumers,
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "restore_deletion_fences",
        restore,
    )

    call = lambda: sync_routes._delete_sync_milvus_collections(  # noqa: E731
        {
            "task_id": task_id,
            "user_id": "user-1",
            "milvus_collection_name": collection_name,
            "generation_config": {},
        },
        [],
    )
    if should_reject:
        with pytest.raises(sync_routes.SyncMilvusCollectionConsumerError):
            call()
        assert dropped == []
        assert events[-1] == (
            "restore",
            (collection_name,),
            f"sync:{task_id}",
            (collection_name,),
        )
    else:
        assert call() == [collection_name]
        assert dropped == [collection_name]
        assert not any(event[0] == "restore" for event in events)

    assert events[0][0] == "fence"
    assert events[1] == ("sync-consumers", (collection_name,))


def test_milvus_restore_failure_keeps_parent_delete_intent(monkeypatch):
    task_id = "sync-milvus-restore-failure"
    collection_name = f"tf_sync_v2_{task_id[:8]}_current"
    parent_rollbacks = []
    dropped = []
    _install_fake_milvus_client(monkeypatch, dropped=dropped)
    monkeypatch.setattr(
        external_sync_service,
        "list_active_collection_consumers",
        lambda _names: (_ for _ in ()).throw(RuntimeError("consumer lookup failed")),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "restore_deletion_fences",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="restore sync Milvus deletion fences"):
        sync_routes._delete_sync_milvus_collections(
            {
                "task_id": task_id,
                "user_id": "user-1",
                "milvus_collection_name": collection_name,
                "generation_config": {},
            },
            [],
            on_preflight_failure=lambda: parent_rollbacks.append(True),
        )

    assert parent_rollbacks == []
    assert dropped == []


def test_milvus_fence_validates_every_registry_and_remote_snapshot_incarnation(
    monkeypatch,
):
    task_id = "sync-incarnation"
    user_id = "user-incarnation"
    task_hash = sync_routes.hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    registered_name = f"tf_sync_v3_{task_hash}_registered"
    remote_only_name = f"tf_sync_v3_{task_hash}_remote-only"
    captured = {}
    _install_fake_milvus_client(
        monkeypatch,
        remote_names=(remote_only_name,),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        lambda **_kwargs: (
            [
                {
                    "collection_id": "registry-incarnation-1",
                    "collection_name": registered_name,
                    "user_id": user_id,
                    "status": "active",
                    "sync_task_id": task_id,
                    "deletion_owner": None,
                }
            ],
            1,
        ),
        raising=False,
    )

    def acquire(names, **kwargs):
        captured["names"] = tuple(sorted(names))
        captured["expected_collection_ids"] = kwargs.get(
            "expected_collection_ids"
        )
        return SimpleNamespace(
            newly_fenced=tuple(sorted(names)),
            created_placeholders=(remote_only_name,),
        )

    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        acquire,
    )

    deleted = sync_routes._delete_sync_milvus_collections(
        {
            "task_id": task_id,
            "user_id": user_id,
            "generation_config": {"embedding_config": {"model": "embedding"}},
        },
        [],
    )

    assert deleted == sorted((registered_name, remote_only_name))
    assert captured == {
        "names": tuple(sorted((registered_name, remote_only_name))),
        "expected_collection_ids": {
            registered_name: "registry-incarnation-1",
            remote_only_name: None,
        },
    }


@pytest.mark.parametrize(
    ("registry_sync_task_id", "registry_user_id"),
    [
        (None, "user-provenance"),
        ("sync-foreign-task", "user-provenance"),
        ("sync-provenance", "user-cross-tenant"),
    ],
)
def test_active_sync_namespace_registry_requires_exact_internal_provenance(
    monkeypatch,
    registry_sync_task_id,
    registry_user_id,
):
    task_id = "sync-provenance"
    user_id = "user-provenance"
    collection_name = f"tf_sync_v2_{task_id[:8]}_current"
    dropped = []
    _install_fake_milvus_client(monkeypatch, dropped=dropped)
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        lambda **_kwargs: (
            [
                {
                    "collection_id": "registry-provenance-id",
                    "collection_name": collection_name,
                    "user_id": registry_user_id,
                    "status": "active",
                    "sync_task_id": registry_sync_task_id,
                    "deletion_owner": None,
                }
            ],
            1,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid registry provenance must fail before fencing"
        ),
    )

    with pytest.raises(sync_routes.SyncMilvusCollectionConsumerError):
        sync_routes._delete_sync_milvus_collections(
            {
                "task_id": task_id,
                "user_id": user_id,
                "milvus_collection_name": collection_name,
                "generation_config": {},
            },
            [],
        )

    assert dropped == []


def test_nonprefixed_exact_current_allows_matching_registry_claim(monkeypatch):
    task_id = "sync-nonprefix-current"
    user_id = "user-nonprefix-current"
    collection_name = "persisted_sync_current_collection"
    dropped = []
    _install_fake_milvus_client(monkeypatch, dropped=dropped)
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        lambda **_kwargs: (
            [
                {
                    "collection_id": "registry-nonprefix-id",
                    "collection_name": collection_name,
                    "user_id": user_id,
                    "status": "active",
                    "sync_task_id": task_id,
                    "deletion_owner": None,
                }
            ],
            1,
        ),
        raising=False,
    )

    deleted = sync_routes._delete_sync_milvus_collections(
        {
            "task_id": task_id,
            "user_id": user_id,
            "milvus_collection_name": collection_name,
            "generation_config": {},
        },
        [],
    )

    assert deleted == [collection_name]
    assert dropped == [collection_name]


def test_current_task_durable_fence_allows_null_legacy_registry_claim(
    monkeypatch,
):
    task_id = "sync-durable-claim"
    user_id = "user-durable-claim"
    collection_name = f"tf_sync_v2_{task_id[:8]}_current"
    snapshot_calls = []
    dropped = []
    _install_fake_milvus_client(monkeypatch, dropped=dropped)
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_fences",
        lambda **_kwargs: [collection_name],
    )

    def internal_snapshot(**_kwargs):
        snapshot_calls.append(True)
        return (
            [
                {
                    "collection_id": "registry-durable-id",
                    "collection_name": collection_name,
                    "user_id": user_id,
                    "status": "deleting",
                    "sync_task_id": None,
                    "deletion_owner": f"sync:{task_id}",
                }
            ],
            1,
        )

    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        internal_snapshot,
        raising=False,
    )

    deleted = sync_routes._delete_sync_milvus_collections(
        {
            "task_id": task_id,
            "user_id": user_id,
            "milvus_collection_name": collection_name,
            "generation_config": {},
        },
        [],
    )

    assert snapshot_calls == [True]
    assert deleted == [collection_name]
    assert dropped == [collection_name]


def test_milvus_consumer_query_permission_error_restores_new_fences(
    monkeypatch,
):
    task_id = "sync-query-permission"
    collection_name = f"tf_sync_v2_{task_id[:8]}_current"
    restored = []
    dropped = []
    _install_fake_milvus_client(monkeypatch, dropped=dropped)
    monkeypatch.setattr(
        external_sync_service,
        "list_active_collection_consumers",
        lambda _names: (_ for _ in ()).throw(PermissionError("consumer owner drift")),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **_kwargs: SimpleNamespace(
            newly_fenced=tuple(sorted(names)),
            created_placeholders=(collection_name,),
        ),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "restore_deletion_fences",
        lambda names, **kwargs: restored.append(
            (tuple(names), tuple(kwargs["created_placeholders"]))
        )
        or True,
    )

    with pytest.raises(PermissionError, match="consumer owner drift"):
        sync_routes._delete_sync_milvus_collections(
            {
                "task_id": task_id,
                "user_id": "user-1",
                "milvus_collection_name": collection_name,
                "generation_config": {},
            },
            [],
        )

    assert restored == [((collection_name,), (collection_name,))]
    assert dropped == []


def _patch_single_owned_dataset_preflight(
    monkeypatch,
    events,
    *,
    task_id,
    user_id,
    dataset_status="ready",
):
    dataset_id = "dataset-preflight"
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": False,
        "status": "deleting_cascade",
        "milvus_collection_name": f"tf_sync_v2_{task_id[:8]}_current",
        "generation_config": {},
    }
    dataset = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "status": dataset_status,
        "storage_path": None,
        "storage_uri": None,
        "source_task_type": "sync",
        "source_task_id": task_id,
    }
    monkeypatch.setattr(
        dataset_service,
        "list_datasets",
        lambda **_kwargs: ([{"dataset_id": dataset_id}], 1),
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: dict(dataset),
    )
    monkeypatch.setattr(
        dataset_service,
        "mark_deleting",
        lambda requested_id, **_kwargs: events.append(("mark", requested_id))
        or True,
    )
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda requested_id, **kwargs: events.append(
            ("restore", requested_id, kwargs["status"])
        )
        or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda _task_id, **kwargs: events.append(
            ("cancel", kwargs["status"], kwargs["is_active"])
        )
        or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: pytest.fail("parent tracking must survive preflight failure"),
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        dataset_asset_service,
        "list_assets",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    return config


@pytest.mark.parametrize(
    "failure_site",
    ["dataset_consumer", "get_task_raw", "registry_snapshot", "milvus_consumer"],
)
def test_new_cascade_preflight_exceptions_restore_dataset_parent_and_worker(
    monkeypatch,
    failure_site,
):
    task_id = f"sync-preflight-{failure_site}"
    user_id = "user-preflight"
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )
    _install_fake_milvus_client(monkeypatch)

    if failure_site == "dataset_consumer":
        monkeypatch.setattr(
            generation_task_service,
            "list_active_dataset_consumers",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("dataset consumer preflight failed")
            ),
        )
        expected_error = PermissionError
        expected_message = "dataset consumer preflight failed"
    elif failure_site == "get_task_raw":
        monkeypatch.setattr(
            external_sync_service,
            "get_task_raw",
            lambda _task_id: (_ for _ in ()).throw(
                RuntimeError("raw config preflight failed")
            ),
        )
        expected_error = RuntimeError
        expected_message = "raw config preflight failed"
    elif failure_site == "registry_snapshot":
        monkeypatch.setattr(
            milvus_collection_service,
            "list_deletion_registry_snapshot",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("registry snapshot preflight failed")
            ),
            raising=False,
        )
        expected_error = RuntimeError
        expected_message = "registry snapshot preflight failed"
    else:
        monkeypatch.setattr(
            external_sync_service,
            "list_active_collection_consumers",
            lambda _names: (_ for _ in ()).throw(
                RuntimeError("Milvus consumer preflight failed")
            ),
        )
        expected_error = RuntimeError
        expected_message = "Milvus consumer preflight failed"

    with pytest.raises(expected_error, match=expected_message):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert events == [
        ("mark", "dataset-preflight"),
        ("restore", "dataset-preflight", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]


def test_dataset_restore_failure_keeps_parent_delete_intent(monkeypatch):
    task_id = "sync-dataset-restore-failure"
    user_id = "user-dataset-restore-failure"
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda requested_id, **_kwargs: events.append(("restore", requested_id))
        or False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "parent intent must remain when a child fence cannot be restored"
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="restore sync pre-destructive deletion state",
    ):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={**config, "status": "idle", "is_active": True},
            resume_worker=lambda: pytest.fail(
                "worker must remain stopped when a child fence remains"
            ),
        )

    assert events == [
        ("mark", "dataset-preflight"),
        ("restore", "dataset-preflight"),
    ]


def test_existing_durable_intent_survives_generic_preflight_exception(
    monkeypatch,
):
    task_id = "sync-durable-preflight"
    user_id = "user-durable"
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
        dataset_status="deleting",
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: (_ for _ in ()).throw(RuntimeError("retry preflight failed")),
    )
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda *_args, **_kwargs: pytest.fail("durable dataset fence must remain"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: pytest.fail("durable parent intent must remain"),
    )

    with pytest.raises(RuntimeError, match="retry preflight failed"):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config=config,
            resume_worker=lambda: pytest.fail("durable worker must stay stopped"),
        )

    assert events == [("mark", "dataset-preflight")]


def test_no_milvus_asset_manifest_failure_restores_new_deletion_state(
    monkeypatch,
):
    task_id = "sync-no-milvus-manifest"
    user_id = "user-no-milvus-manifest"
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    config.pop("milvus_collection_name")
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )
    monkeypatch.setattr(
        dataset_asset_service,
        "list_assets",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("asset manifest read failed")
        ),
    )

    with pytest.raises(RuntimeError, match="asset manifest read failed"):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert events == [
        ("mark", "dataset-preflight"),
        ("restore", "dataset-preflight", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]


def test_complete_asset_manifest_is_read_before_any_milvus_drop(monkeypatch):
    task_id = "sync-complete-manifest"
    user_id = "user-complete-manifest"
    events = []
    dropped = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    dataset_ids = ("dataset-manifest-a", "dataset-manifest-b")
    datasets = {
        dataset_id: {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "status": "ready",
            "storage_path": None,
            "storage_uri": None,
            "source_task_type": "sync",
            "source_task_id": task_id,
        }
        for dataset_id in dataset_ids
    }
    monkeypatch.setattr(
        dataset_service,
        "list_datasets",
        lambda **_kwargs: (
            [{"dataset_id": dataset_id} for dataset_id in dataset_ids],
            len(dataset_ids),
        ),
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda dataset_id: dict(datasets[dataset_id]),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )

    def list_assets(*, dataset_id):
        events.append(("manifest", dataset_id))
        if dataset_id == "dataset-manifest-b":
            raise RuntimeError("second asset manifest failed")
        return []

    monkeypatch.setattr(dataset_asset_service, "list_assets", list_assets)
    monkeypatch.setattr(
        dataset_lineage_service,
        "delete_edges_for_dataset",
        lambda _dataset_id: True,
    )
    monkeypatch.setattr(
        dataset_asset_service,
        "delete_assets_for_dataset",
        lambda _dataset_id: True,
    )
    monkeypatch.setattr(
        dataset_service,
        "delete_dataset",
        lambda _dataset_id, **_kwargs: True,
    )
    _install_fake_milvus_client(monkeypatch, dropped=dropped)

    with pytest.raises(RuntimeError, match="second asset manifest failed"):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert dropped == []
    assert ("manifest", "dataset-manifest-a") in events
    assert ("manifest", "dataset-manifest-b") in events
    assert events[-3:] == [
        ("restore", "dataset-manifest-a", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]


@pytest.mark.parametrize("consumer_kind", ["dataset", "generation"])
def test_shared_storage_reference_blocks_cascade_before_physical_cleanup(
    monkeypatch,
    consumer_kind,
):
    task_id = "sync-shared-storage"
    user_id = "user-shared-storage"
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    shared_uri = "s3://bucket/shared/data.jsonl"
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": "dataset-preflight",
            "user_id": user_id,
            "status": "ready",
            "storage_path": None,
            "storage_uri": shared_uri,
            "source_task_type": "sync",
            "source_task_id": task_id,
        },
    )

    def external_consumers(references, *, exclude_dataset_ids):
        assert set(references) == {shared_uri}
        assert set(exclude_dataset_ids) == {"dataset-preflight"}
        return ([
            {
                "dataset_id": "dataset-foreign",
                "reference": shared_uri,
                "source": "asset",
            }
        ] if consumer_kind == "dataset" else [])

    monkeypatch.setattr(
        dataset_service,
        "list_external_storage_reference_consumers",
        external_consumers,
        raising=False,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_artifact_reference_consumers",
        lambda references, *, exclude_task_ids: (
            [
                {
                    "task_id": "generation-foreign",
                    "field": "input_path",
                    "reference": shared_uri,
                }
            ]
            if consumer_kind == "generation"
            else []
        ),
        raising=False,
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: pytest.fail(
            "shared storage must fail before Milvus cleanup"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert events[-3:] == [
        ("restore", "dataset-preflight", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]


def test_source_snapshot_discovers_owner_drift_before_parent_deletion(monkeypatch):
    task_id = "sync-source-owner-drift"
    user_id = "user-owner"
    observed_user_filters = []
    events = []
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": False,
        "status": "deleting_cascade",
        "generation_config": {},
    }

    def list_datasets(*, user_id, **_kwargs):
        observed_user_filters.append(user_id)
        if user_id is not None:
            return [], 0
        return [{"dataset_id": "dataset-owner-drift"}], 1

    monkeypatch.setattr(dataset_service, "list_datasets", list_datasets)
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": "dataset-owner-drift",
            "user_id": "user-foreign",
            "status": "ready",
            "source_task_type": "sync",
            "source_task_id": task_id,
        },
    )
    monkeypatch.setattr(
        dataset_service,
        "mark_deleting",
        lambda *_args, **_kwargs: pytest.fail("owner drift must fail before fencing"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda _task_id, **_kwargs: events.append("cancel") or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: (_ for _ in ()).throw(
            RuntimeError("parent deleted despite owner drift")
        ),
    )

    with pytest.raises(PermissionError, match="owner changed"):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={**config, "status": "idle"},
            resume_worker=lambda: events.append("resume"),
        )

    assert observed_user_filters == [None]
    assert events == ["cancel", "resume"]


def test_partial_dataset_deletion_unlinks_owned_links_before_retry(monkeypatch):
    task_id = "sync-partial-retry"
    user_id = "user-partial"
    collection_name = f"tf_sync_v2_{task_id[:8]}_current"
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": False,
        "status": "deleting_cascade",
        "milvus_collection_name": collection_name,
        "generation_config": {},
    }
    datasets = {
        dataset_id: {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "status": "ready",
            "storage_path": None,
            "storage_uri": None,
            "source_task_type": "sync",
            "source_task_id": task_id,
        }
        for dataset_id in ("dataset-first", "dataset-second")
    }
    links = [
        {
            "collection_name": collection_name,
            "dataset_id": dataset_id,
            "task_id": task_id,
        }
        for dataset_id in datasets
    ]
    failed_second_once = False
    events = []
    _install_fake_milvus_client(monkeypatch, dropped=[])

    def list_datasets(*, source_task_type, source_task_id, **_kwargs):
        assert (source_task_type, source_task_id) == ("sync", task_id)
        rows = [{"dataset_id": dataset_id} for dataset_id in datasets]
        return rows, len(rows)

    def mark_deleting(dataset_id, **_kwargs):
        datasets[dataset_id]["status"] = "deleting"
        return True

    def unlink_dataset(linked_collection, dataset_id):
        assert linked_collection == collection_name
        for index, link in enumerate(links):
            if link["dataset_id"] == dataset_id:
                links.pop(index)
                events.append(("unlink", dataset_id))
                return True
        return False

    def delete_dataset(dataset_id, **_kwargs):
        nonlocal failed_second_once
        if dataset_id == "dataset-second" and not failed_second_once:
            failed_second_once = True
            raise RuntimeError("injected second dataset deletion failure")
        del datasets[dataset_id]
        events.append(("delete", dataset_id))
        return True

    monkeypatch.setattr(dataset_service, "list_datasets", list_datasets)
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda dataset_id: dict(datasets[dataset_id]),
    )
    monkeypatch.setattr(dataset_service, "mark_deleting", mark_deleting)
    monkeypatch.setattr(dataset_service, "delete_dataset", delete_dataset)
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda *_args, **_kwargs: pytest.fail("physical deletion keeps durable intent"),
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_dataset_links",
        lambda dataset_ids: [
            dict(link) for link in links if link["dataset_id"] in set(dataset_ids)
        ],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "get_linked_datasets",
        lambda requested_collection: [
            dict(link)
            for link in links
            if link["collection_name"] == requested_collection
        ],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "unlink_dataset",
        unlink_dataset,
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "delete_collections",
        lambda names, **_kwargs: events.append(
            ("finalize", tuple(sorted(names)))
        )
        or True,
    )
    monkeypatch.setattr(
        dataset_asset_service,
        "list_assets",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        dataset_asset_service,
        "delete_assets_for_dataset",
        lambda _dataset_id: True,
    )
    monkeypatch.setattr(
        dataset_lineage_service,
        "delete_edges_for_dataset",
        lambda _dataset_id: True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: pytest.fail("physical failure keeps parent intent"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "finalize_task_deletion",
        lambda _task_id, **_kwargs: events.append("parent-delete") or True,
    )

    with pytest.raises(
        RuntimeError,
        match="injected second dataset deletion failure",
    ):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={**config, "status": "idle"},
        )

    assert "dataset-first" not in datasets
    # Unlink before deleting metadata: if metadata deletion fails or the
    # process exits between calls, the retry cannot inherit a dangling link.
    assert links == []

    result = sync_routes._perform_sync_task_deletion(
        task_id,
        cascade=True,
        config=config,
        original_config=config,
    )

    assert result == {
        "message": "Sync task deleted",
        "deleted_datasets": ["dataset-second"],
    }
    assert links == []
    assert events[-2:] == [
        ("finalize", (collection_name,)),
        "parent-delete",
    ]


def test_external_dataset_link_conflict_never_unlinks_any_link(monkeypatch):
    task_id = "sync-external-link"
    user_id = "user-external-link"
    dataset_id = "dataset-linked"
    owned_collection = f"tf_sync_v2_{task_id[:8]}_current"
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "status": "deleting_cascade",
        "milvus_collection_name": owned_collection,
        "generation_config": {},
    }
    monkeypatch.setattr(
        dataset_service,
        "list_datasets",
        lambda **_kwargs: ([{"dataset_id": dataset_id}], 1),
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "status": "ready",
            "source_task_type": "sync",
            "source_task_id": task_id,
        },
    )
    monkeypatch.setattr(dataset_service, "mark_deleting", lambda *_a, **_k: True)
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_dataset_links",
        lambda _ids: [
            {"collection_name": owned_collection, "dataset_id": dataset_id},
            {"collection_name": "customer_collection", "dataset_id": dataset_id},
        ],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "unlink_dataset",
        lambda *_a, **_k: pytest.fail("external conflict must not unlink anything"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_a, **_k: True,
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={**config, "status": "idle"},
        )

    assert exc_info.value.status_code == 409


@pytest.mark.parametrize("child_kind", ["generation", "training"])
@pytest.mark.parametrize("drift_location", ["tracking", "child"])
def test_foreign_terminal_sync_child_is_rejected_before_parent_intent(
    monkeypatch,
    child_kind,
    drift_location,
):
    task_id = f"sync-foreign-{child_kind}-{drift_location}"
    user_id = "user-parent"
    foreign_user_id = "user-foreign"
    child_id = f"{child_kind}-terminal-child"
    events = []
    tracking = {
        "id": 1,
        "task_id": task_id,
        "user_id": (
            foreign_user_id if drift_location == "tracking" else user_id
        ),
        "status": "completed" if child_kind == "generation" else "succeeded",
    }
    tracking[f"{child_kind}_task_id"] = child_id
    child = {
        "task_id": child_id,
        "user_id": foreign_user_id if drift_location == "child" else user_id,
        "status": tracking["status"],
        "output_dir": "foreign-output",
    }
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: (
            ([tracking], 1) if child_kind == "generation" else ([], 0)
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: (
            ([tracking], 1) if child_kind == "training" else ([], 0)
        ),
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _child_id: child if child_kind == "generation" else None,
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda _child_id: child if child_kind == "training" else None,
    )
    monkeypatch.setattr(
        background_task_admission_service,
        "begin_deletion",
        lambda *_args, **_kwargs: SimpleNamespace(release=lambda: None),
    )
    monkeypatch.setattr(
        sync_routes,
        "_list_sync_training_artifact_conflicts",
        lambda _snapshot: [],
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "foreign child must fail before parent intent"
        ),
    )
    manager = SimpleNamespace(
        running=True,
        start_worker=lambda requested_id: events.append(("resume", requested_id)),
    )

    with pytest.raises(PermissionError, match="owner changed"):
        sync_routes._delete_sync_task_locked(
            task_id,
            cascade=True,
            config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "idle",
                "is_active": True,
            },
            original_config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "idle",
                "is_active": True,
            },
            sync_manager=manager,
        )

    assert events == [("resume", task_id)]


@pytest.mark.parametrize("foreign_scope", ["tenant", "task"])
def test_batch_manifest_rejects_foreign_managed_scope_before_cleanup(
    monkeypatch,
    tmp_path,
    foreign_scope,
):
    from train_factory.sync import sync_worker as sync_worker_module

    task_id = "sync-batch-scope"
    user_id = "user-batch-scope"
    row_user_id = "user-foreign" if foreign_scope == "tenant" else user_id
    row_task_id = "task-foreign" if foreign_scope == "task" else task_id
    root = tmp_path / "sync-root"
    batch_parent = root / row_user_id / row_task_id
    batch_parent.mkdir(parents=True)
    batch_path = batch_parent / "batch_scope.jsonl"
    batch_path.write_text("{}\n", encoding="utf-8")
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    config.pop("milvus_collection_name")
    monkeypatch.setattr(sync_worker_module, "SYNC_DATA_DIR", str(root))
    monkeypatch.setattr(dataset_asset_service, "list_assets", lambda **_kwargs: [])
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: (
            [
                {
                    "batch_id": "batch-scope",
                    "task_id": row_task_id,
                    "user_id": row_user_id,
                    "storage_path": str(batch_path),
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid batch manifest must fail before Milvus cleanup"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert batch_path.exists()
    assert events[-3:] == [
        ("restore", "dataset-preflight", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]


def test_late_batch_snapshot_failure_precedes_every_physical_cleanup(monkeypatch):
    task_id = "sync-late-batch-snapshot"
    user_id = "user-late-batch-snapshot"
    events = []
    dropped = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    monkeypatch.setattr(dataset_asset_service, "list_assets", lambda **_kwargs: [])
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )

    def list_batches(*, offset, **_kwargs):
        if offset:
            raise RuntimeError("late batch listing failed")
        return (
            [
                {
                    "batch_id": "batch-first-page",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": "not-reached.jsonl",
                }
            ],
            2,
        )

    monkeypatch.setattr(external_sync_service, "list_batches", list_batches)
    _install_fake_milvus_client(monkeypatch, dropped=dropped)

    with pytest.raises(RuntimeError, match="late batch listing failed"):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert dropped == []
    assert events[-3:] == [
        ("restore", "dataset-preflight", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]


def test_non_cascade_batch_rejects_foreign_tracking_before_unlink(monkeypatch):
    task_id = "sync-noncascade-foreign-batch"
    user_id = "user-noncascade-batch"
    events = []
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: (
            [
                {
                    "batch_id": "batch-foreign",
                    "task_id": "sync-foreign",
                    "user_id": user_id,
                    "storage_path": "foreign.jsonl",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        sync_routes,
        "_safe_delete_path",
        lambda *_args, **_kwargs: pytest.fail(
            "foreign batch must not be unlinked"
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: pytest.fail("parent row must remain"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda _task_id, **kwargs: events.append(
            ("cancel", kwargs["status"], kwargs["is_active"])
        )
        or True,
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=False,
            config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "deleting",
            },
            original_config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert events == [("cancel", "idle", True), "resume"]


def test_non_cascade_late_batch_snapshot_failure_precedes_unlink(monkeypatch):
    task_id = "sync-noncascade-late-batch"
    user_id = "user-noncascade-late-batch"
    cleanup_calls = []

    def list_batches(*, offset, **_kwargs):
        if offset:
            raise RuntimeError("noncascade late batch listing failed")
        return (
            [
                {
                    "batch_id": "batch-first-page",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": "not-reached.jsonl",
                }
            ],
            2,
        )

    monkeypatch.setattr(external_sync_service, "list_batches", list_batches)
    monkeypatch.setattr(
        sync_routes,
        "_safe_delete_path",
        lambda *_args, **_kwargs: cleanup_calls.append(True),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: pytest.fail("parent row must remain"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "existing noncascade intent must remain durable"
        ),
    )

    with pytest.raises(RuntimeError, match="noncascade late batch listing failed"):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=False,
            config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "deleting",
            },
            original_config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "deleting",
                "is_active": False,
            },
        )

    assert cleanup_calls == []


def test_preexisting_dataset_fence_keeps_new_parent_intent_on_preflight_failure(
    monkeypatch,
):
    task_id = "sync-existing-dataset-fence"
    user_id = "user-existing-dataset-fence"
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
        dataset_status="deleting",
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: (_ for _ in ()).throw(
            RuntimeError("preflight failed after existing child fence")
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "existing child fence requires durable parent intent"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="preflight failed after existing child fence",
    ):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: pytest.fail(
                "worker must remain stopped with an existing child fence"
            ),
        )

    assert events == [("mark", "dataset-preflight")]


def test_preexisting_milvus_fence_keeps_new_parent_intent_on_preflight_failure(
    monkeypatch,
):
    task_id = "sync-existing-milvus-fence"
    user_id = "user-existing-milvus-fence"
    collection_name = f"tf_sync_v2_{task_id[:8]}_current"
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "status": "deleting_cascade",
        "is_active": False,
        "milvus_collection_name": collection_name,
        "generation_config": {},
    }
    monkeypatch.setattr(dataset_service, "list_datasets", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(external_sync_service, "list_batches", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(external_sync_service, "get_task_raw", lambda _task_id: config)
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_fences",
        lambda **_kwargs: [collection_name],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        lambda **_kwargs: (
            [
                {
                    "collection_id": "existing-milvus-fence-id",
                    "collection_name": collection_name,
                    "user_id": user_id,
                    "status": "deleting",
                    "sync_task_id": task_id,
                    "deletion_owner": f"sync:{task_id}",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        lambda *_args, **_kwargs: SimpleNamespace(
            newly_fenced=(),
            created_placeholders=(),
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_active_collection_consumers",
        lambda _names: (_ for _ in ()).throw(
            RuntimeError("Milvus retry preflight failed")
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "existing Milvus fence requires durable parent intent"
        ),
    )
    _install_fake_milvus_client(monkeypatch)

    with pytest.raises(RuntimeError, match="Milvus retry preflight failed"):
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: pytest.fail(
                "worker must remain stopped with an existing Milvus fence"
            ),
        )


def test_training_child_output_is_validated_before_milvus_cleanup(
    monkeypatch,
    tmp_path,
):
    task_id = "sync-training-output-preflight"
    user_id = "user-training-output-preflight"
    child_id = "training-output-child"
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "status": "deleting_cascade",
        "milvus_collection_name": f"tf_sync_v2_{task_id[:8]}_current",
        "generation_config": {},
    }
    tracking = {
        "id": 1,
        "task_id": task_id,
        "user_id": user_id,
        "training_task_id": child_id,
        "status": "succeeded",
    }
    child = {
        "task_id": child_id,
        "user_id": user_id,
        "status": "succeeded",
        "output_dir": str(tmp_path / "outside-managed-output"),
    }
    monkeypatch.setattr(dataset_service, "list_datasets", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(dataset_lineage_service, "get_edges_by_task", lambda _id: [])
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(training_task_service, "get_task", lambda _id: child)
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(milvus_collection_service, "list_dataset_links", lambda _ids: [])
    monkeypatch.setattr(external_sync_service, "list_batches", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(external_sync_service, "get_task_raw", lambda _id: config)
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: pytest.fail(
            "training output must be validated before Milvus cleanup"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={**config, "status": "idle"},
            training_snapshot=[tracking],
        )

    assert exc_info.value.status_code == 400


def test_sync_cascade_removes_unregistered_generation_child_artifacts(
    monkeypatch,
    tmp_path,
):
    task_id = "sync-generation-child-artifacts"
    user_id = "user-generation-child-artifacts"
    child_id = "generation-artifact-child"
    output_root = tmp_path / "generation-output"
    artifact_path = output_root / child_id / "output.jsonl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("GENERATION_OUTPUT_DIR", str(output_root))
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "status": "deleting_cascade",
        "generation_config": {},
    }
    tracking = {
        "id": 1,
        "task_id": task_id,
        "user_id": user_id,
        "generation_task_id": child_id,
        "status": "completed",
    }
    child = {
        "task_id": child_id,
        "user_id": user_id,
        "status": "completed",
        "output_path": str(artifact_path),
        "qa_output_path": None,
        "qa_filtered_path": None,
        "deep_eval_path": None,
    }
    events = []
    monkeypatch.setattr(dataset_service, "list_datasets", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(dataset_lineage_service, "get_edges_by_task", lambda _id: [])
    monkeypatch.setattr(generation_task_service, "get_task", lambda _id: child)
    monkeypatch.setattr(
        generation_task_service,
        "delete_task",
        lambda child_task_id: events.append(("child-delete", child_task_id)) or True,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(milvus_collection_service, "list_dataset_links", lambda _ids: [])
    monkeypatch.setattr(external_sync_service, "list_batches", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(external_sync_service, "get_task_raw", lambda _id: config)
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        external_sync_service,
        "finalize_task_deletion",
        lambda parent_task_id, **_kwargs: events.append(
            ("parent-delete", parent_task_id)
        )
        or True,
    )

    sync_routes._perform_sync_task_deletion(
        task_id,
        cascade=True,
        config=config,
        original_config={**config, "status": "idle"},
        generation_snapshot=[tracking],
    )

    assert not artifact_path.exists()
    assert events == [
        ("child-delete", child_id),
        ("parent-delete", task_id),
    ]


@pytest.mark.parametrize("consumer_kind", ["training", "evaluation"])
def test_active_path_consumer_of_child_artifact_blocks_before_cleanup(
    monkeypatch,
    tmp_path,
    consumer_kind,
):
    task_id = f"sync-child-path-{consumer_kind}"
    user_id = "user-child-path-consumer"
    generation_child_id = "generation-owned-child"
    training_child_id = "training-owned-child"
    generation_root = tmp_path / "generation-output"
    generation_artifact = generation_root / generation_child_id / "output.jsonl"
    generation_artifact.parent.mkdir(parents=True)
    generation_artifact.write_text("{}\n", encoding="utf-8")
    generation_artifact_raw = str(
        generation_artifact.parent / "unused" / ".." / generation_artifact.name
    )
    training_output = sync_routes.get_settings().get_task_output_dir(
        training_child_id
    )
    monkeypatch.setenv("GENERATION_OUTPUT_DIR", str(generation_root))
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    generation_tracking = {
        "id": 1,
        "task_id": task_id,
        "user_id": user_id,
        "generation_task_id": generation_child_id,
        "status": "completed",
    }
    training_tracking = {
        "id": 2,
        "task_id": task_id,
        "user_id": user_id,
        "training_task_id": training_child_id,
        "status": "succeeded",
    }
    monkeypatch.setattr(dataset_lineage_service, "get_edges_by_task", lambda _id: [])
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _id: {
            "task_id": generation_child_id,
            "user_id": user_id,
            "status": "completed",
            "output_path": generation_artifact_raw,
            "qa_output_path": None,
            "qa_filtered_path": None,
            "deep_eval_path": None,
        },
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda _id: {
            "task_id": training_child_id,
            "user_id": user_id,
            "status": "succeeded",
            "output_dir": str(training_output),
        },
    )

    observed_consumer_paths = {}

    def training_consumers(paths, *, exclude_task_ids=()):
        observed_consumer_paths["training"] = tuple(paths)
        assert exclude_task_ids == (training_child_id,)
        return (
            ["training-external"]
            if consumer_kind == "training" and generation_artifact_raw in paths
            else []
        )

    def evaluation_consumers(dataset_ids, paths):
        observed_consumer_paths["evaluation"] = tuple(paths)
        assert dataset_ids == ["dataset-preflight"]
        return (
            ["evaluation-external"]
            if consumer_kind == "evaluation" and generation_artifact_raw in paths
            else []
        )

    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        training_consumers,
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        evaluation_consumers,
    )
    monkeypatch.setattr(external_sync_service, "get_task_raw", lambda _id: config)
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: pytest.fail(
            "active path consumer must fail before Milvus cleanup"
        ),
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_resolved_path",
        lambda *_args, **_kwargs: pytest.fail(
            "active path consumer must fail before storage cleanup"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={
                **config,
                "status": "idle",
                "is_active": True,
            },
            generation_snapshot=[generation_tracking],
            training_snapshot=[training_tracking],
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert generation_artifact_raw in observed_consumer_paths[consumer_kind]
    assert generation_artifact.exists()
    assert events[-3:] == [
        ("restore", "dataset-preflight", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]


@pytest.mark.parametrize(
    ("child_kind", "child_state"),
    [
        ("generation", "pending"),
        ("generation", "running"),
        ("generation", "executing"),
        ("training", "pending"),
        ("training", "preparing"),
        ("training", "running"),
        ("training", "evaluating"),
        ("training", "executing"),
        ("training", "process"),
    ],
)
def test_non_cascade_rejects_active_child_before_delete_intent(
    monkeypatch,
    child_kind,
    child_state,
):
    task_id = f"sync-noncascade-{child_kind}-{child_state}"
    child_id = f"{child_kind}-{child_state}"
    user_id = "user-noncascade-child"
    tracking = {
        "id": 1,
        "task_id": task_id,
        "user_id": user_id,
        f"{child_kind}_task_id": child_id,
        "status": "completed" if child_state == "process" else child_state,
    }
    if child_kind == "generation":
        tracking["output_dataset_id"] = None
    child = {
        "task_id": child_id,
        "user_id": user_id,
        "status": (
            "completed"
            if child_kind == "generation" and child_state == "executing"
            else "succeeded"
            if child_kind == "training"
            and child_state in {"executing", "process"}
            else child_state
        ),
    }
    if child_state == "process":
        child.update(process_pid=4321, process_status="running")

    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: ([tracking], 1) if child_kind == "generation" else ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([tracking], 1) if child_kind == "training" else ([], 0),
    )
    monkeypatch.setattr(generation_task_service, "get_task", lambda _id: child)
    monkeypatch.setattr(training_task_service, "get_task", lambda _id: child)
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "active child must fail before parent delete intent"
        ),
    )

    def begin_child_deletion(kind, requested_id):
        assert (kind, requested_id) == (child_kind, child_id)
        if child_state == "executing":
            raise BackgroundTaskAlreadyExecuting("child is executing")
        return SimpleNamespace(release=lambda: None)

    monkeypatch.setattr(
        background_task_admission_service,
        "begin_deletion",
        begin_child_deletion,
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._delete_sync_task_locked(
            task_id,
            cascade=False,
            config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "idle",
                "is_active": False,
            },
            original_config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "idle",
                "is_active": False,
            },
            sync_manager=SimpleNamespace(running=False),
        )

    assert exc_info.value.status_code == 409


@pytest.mark.parametrize(
    "consumer_kind",
    ["dataset", "generation", "training", "evaluation"],
)
def test_non_cascade_shared_batch_path_blocks_before_unlink(
    monkeypatch,
    tmp_path,
    consumer_kind,
):
    task_id = f"sync-shared-batch-{consumer_kind}"
    user_id = "user-shared-batch"
    batch_path = tmp_path / "sync" / user_id / task_id / "batch.jsonl"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text("{}\n", encoding="utf-8")
    cleanup_calls = []
    observed_paths = {}

    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: (
            [
                {
                    "batch_id": "batch-shared",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": str(batch_path),
                }
            ],
            1,
        ),
    )
    from train_factory.sync import sync_worker as sync_worker_module

    monkeypatch.setattr(
        sync_worker_module,
        "_resolve_managed_batch_path",
        lambda *_args, **_kwargs: batch_path,
    )

    def dataset_consumers(paths, **_kwargs):
        observed_paths["dataset"] = tuple(paths)
        return [{"source": "asset"}] if consumer_kind == "dataset" else []

    def generation_consumers(paths, **_kwargs):
        observed_paths["generation"] = tuple(paths)
        return [{"task_id": "generation-consumer"}] if consumer_kind == "generation" else []

    def training_consumers(paths, **_kwargs):
        observed_paths["training"] = tuple(paths)
        return ["training-consumer"] if consumer_kind == "training" else []

    def evaluation_consumers(dataset_ids, paths):
        assert dataset_ids == []
        observed_paths["evaluation"] = tuple(paths)
        return ["evaluation-consumer"] if consumer_kind == "evaluation" else []

    monkeypatch.setattr(
        dataset_service,
        "list_external_storage_reference_consumers",
        dataset_consumers,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_artifact_reference_consumers",
        generation_consumers,
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        training_consumers,
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        evaluation_consumers,
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_resolved_path",
        lambda path: cleanup_calls.append(path),
    )
    monkeypatch.setattr(external_sync_service, "delete_task", lambda _id: True)

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=False,
            config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "deleting",
                "is_active": False,
            },
            original_config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "deleting",
                "is_active": False,
            },
        )

    assert exc_info.value.status_code == 409
    assert str(batch_path) in observed_paths[consumer_kind]
    assert cleanup_calls == []
    assert batch_path.exists()


def test_new_non_cascade_intent_is_rolled_back_on_shared_batch_conflict(
    monkeypatch,
    tmp_path,
):
    task_id = "sync-shared-batch-rollback"
    user_id = "user-shared-batch-rollback"
    batch_path = tmp_path / "sync" / user_id / task_id / "batch.jsonl"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text("{}\n", encoding="utf-8")
    events = []
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: (
            [
                {
                    "batch_id": "batch-shared",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": str(batch_path),
                }
            ],
            1,
        ),
    )
    from train_factory.sync import sync_worker as sync_worker_module

    monkeypatch.setattr(
        sync_worker_module,
        "_resolve_managed_batch_path",
        lambda *_args, **_kwargs: batch_path,
    )
    monkeypatch.setattr(
        dataset_service,
        "list_external_storage_reference_consumers",
        lambda paths, **_kwargs: [{"reference": tuple(paths)[0]}],
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_artifact_reference_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda _id, **kwargs: events.append(("cancel", kwargs)) or True,
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_resolved_path",
        lambda *_args: pytest.fail("shared batch must not be unlinked"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda *_args: pytest.fail("parent must survive preflight conflict"),
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=False,
            config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "deleting",
                "is_active": False,
            },
            original_config={
                "task_id": task_id,
                "user_id": user_id,
                "status": "idle",
                "is_active": True,
            },
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert events[0][0] == "cancel"
    assert events[-1] == "resume"
    assert batch_path.exists()


@pytest.mark.parametrize(
    "consumer_kind",
    ["dataset", "generation", "training", "evaluation"],
)
def test_cascade_shared_batch_path_blocks_before_milvus_or_unlink(
    monkeypatch,
    tmp_path,
    consumer_kind,
):
    task_id = f"sync-cascade-shared-batch-{consumer_kind}"
    user_id = "user-cascade-shared-batch"
    events = []
    config = _patch_single_owned_dataset_preflight(
        monkeypatch,
        events,
        task_id=task_id,
        user_id=user_id,
    )
    batch_path = tmp_path / "sync" / user_id / task_id / "batch.jsonl"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: (
            [
                {
                    "batch_id": "batch-cascade-shared",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": str(batch_path),
                }
            ],
            1,
        ),
    )
    from train_factory.sync import sync_worker as sync_worker_module

    monkeypatch.setattr(
        sync_worker_module,
        "_resolve_managed_batch_path",
        lambda *_args, **_kwargs: batch_path,
    )
    monkeypatch.setattr(dataset_lineage_service, "get_edges_by_task", lambda _id: [])
    monkeypatch.setattr(
        dataset_service,
        "list_external_storage_reference_consumers",
        lambda paths, **_kwargs: (
            [{"reference": str(batch_path)}]
            if consumer_kind == "dataset" and str(batch_path) in paths
            else []
        ),
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_artifact_reference_consumers",
        lambda paths, **_kwargs: (
            [{"task_id": "generation-consumer"}]
            if consumer_kind == "generation" and str(batch_path) in paths
            else []
        ),
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda paths, **_kwargs: (
            ["training-consumer"]
            if consumer_kind == "training" and str(batch_path) in paths
            else []
        ),
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda _ids, paths: (
            ["evaluation-consumer"]
            if consumer_kind == "evaluation" and str(batch_path) in paths
            else []
        ),
    )
    monkeypatch.setattr(external_sync_service, "get_task_raw", lambda _id: config)
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: pytest.fail(
            "shared batch conflict must precede Milvus cleanup"
        ),
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_resolved_path",
        lambda *_args: pytest.fail("shared batch must not be unlinked"),
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config={**config, "status": "idle", "is_active": True},
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert batch_path.exists()
    assert events[-3:] == [
        ("restore", "dataset-preflight", "ready"),
        ("cancel", "idle", True),
        "resume",
    ]
