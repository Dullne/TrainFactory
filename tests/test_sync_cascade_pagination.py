"""Regression tests for complete external-sync cascade snapshots."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from train_factory.api.routes import sync_routes
from train_factory.config.settings import get_settings
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.generation_task_entity import GenerationStatus
from train_factory.storage.services.dataset_asset_service import (
    dataset_asset_service,
)
from train_factory.storage.services.dataset_lineage_service import (
    dataset_lineage_service,
)
from train_factory.storage.services.dataset_service import dataset_service
from train_factory.storage.services.dataset_service import (
    DatasetDeletionOwnerConflictError,
)
from train_factory.storage.services.deep_evaluation_task_service import (
    deep_evaluation_task_service,
)
from train_factory.storage.services.external_sync_service import (
    external_sync_service,
)
from train_factory.storage.services.evaluation_task_service import (
    evaluation_task_service,
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
from train_factory.sync.sync_manager import sync_manager


@pytest.fixture(autouse=True)
def _no_external_milvus_links(monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "list_external_storage_reference_consumers",
        lambda *_args, **_kwargs: [],
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
        "list_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
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


def test_cascade_milvus_cleanup_drops_base_adapters_and_retains_fences(
    monkeypatch,
):
    from train_factory.generation.clients import milvus_client as milvus_module
    from train_factory.storage.services.milvus_collection_service import (
        milvus_collection_service,
    )

    task_id = "sync-owned-collections"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    current_base = f"tf_sync_v2_{task_id[:8]}_current"
    prior_v3_base = f"tf_sync_v3_{task_hash}_oldfingerprint"
    prior_v3_adapter = f"tf_sync_v3_{task_hash}_aadapter_oldfingerprint"
    unrelated = "tf_sync_v3_unrelated_collection"
    dropped = []
    registry_deleted = []
    closed = []

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self, _config):
            pass

        def connect(self):
            return None

        def list_collections(self):
            return [prior_v3_base, prior_v3_adapter, unrelated]

        def drop_collection(self, name):
            dropped.append(name)
            return name != "missing"

        def close(self):
            closed.append(True)

    monkeypatch.setattr(milvus_module, "MilvusConfig", FakeConfig)
    monkeypatch.setattr(milvus_module, "MilvusClient", FakeClient)
    monkeypatch.setattr(
        milvus_collection_service,
        "delete_collection",
        lambda name, **_kwargs: registry_deleted.append(name) or True,
    )

    deleted = sync_routes._delete_sync_milvus_collections(
        {
            "task_id": task_id,
            "milvus_collection_name": current_base,
            "generation_config": {"embedding_config": {"model": "embedding"}},
        },
        [
            {
                "loaded_adapter_id": "adapter-12345678",
                "loaded_adapter_name": "sync-adapter",
            }
        ],
    )

    legacy_hash = hashlib.md5(  # noqa: S324 - legacy name compatibility
        b"sync-adapter"
    ).hexdigest()[:8]
    expected = sorted(
        {
            current_base,
            prior_v3_base,
            prior_v3_adapter,
            f"tf_sync_{task_id[:8]}_aadapter-_{legacy_hash}",
            f"tf_sync_v2_{task_id[:8]}_aadapter-_{legacy_hash}",
        }
    )
    assert deleted == expected
    assert dropped == expected
    assert registry_deleted == []
    assert unrelated not in dropped
    assert closed == [True]


def test_cascade_milvus_cleanup_includes_registered_v3_collection_missing_remotely(
    monkeypatch,
):
    from train_factory.generation.clients import milvus_client as milvus_module

    task_id = "sync-registered-orphan"
    user_id = "user-1"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    registered_name = f"tf_sync_v3_{task_hash}_aadapter_fingerprint"
    fenced = []
    dropped = []

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeClient:
        def __init__(self, _config):
            pass

        def connect(self):
            return None

        def list_collections(self):
            return []

        def drop_collection(self, name):
            dropped.append(name)

        def close(self):
            return None

    monkeypatch.setattr(milvus_module, "MilvusConfig", FakeConfig)
    monkeypatch.setattr(milvus_module, "MilvusClient", FakeClient)
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        lambda **_kwargs: (
            [
                {
                    "collection_id": "collection-registered-orphan",
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
    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **_kwargs: (
            fenced.extend(sorted(names))
            or SimpleNamespace(
                newly_fenced=tuple(sorted(names)),
                created_placeholders=(),
            )
        ),
    )

    deleted = sync_routes._delete_sync_milvus_collections(
        {
            "task_id": task_id,
            "user_id": user_id,
            "generation_config": {"embedding_config": {"model": "embedding"}},
        },
        [],
    )

    assert deleted == [registered_name]
    assert fenced == [registered_name]
    assert dropped == [registered_name]


def test_cascade_milvus_cleanup_rejects_unclaimed_exact_current_collection(
    monkeypatch,
):
    task_id = "sync-owned-collections"
    collection_name = "customer_managed_collection"
    monkeypatch.setattr(
        milvus_collection_service,
        "list_deletion_registry_snapshot",
        lambda **_kwargs: (
            [
                {
                    "collection_id": "manual-collection-id",
                    "collection_name": collection_name,
                    "user_id": "user-1",
                    "status": "active",
                    "sync_task_id": None,
                    "deletion_owner": None,
                }
            ],
            1,
        ),
        raising=False,
    )

    with pytest.raises(sync_routes.SyncMilvusCollectionConsumerError):
        sync_routes._delete_sync_milvus_collections(
            {
                "task_id": task_id,
                "user_id": "user-1",
                "milvus_collection_name": collection_name,
                "generation_config": {},
            },
            [],
        )


def test_cascade_milvus_cleanup_rejects_collection_shared_by_external_dataset(
    monkeypatch,
):
    from train_factory.generation.clients import milvus_client as milvus_module

    task_id = "sync-shared-collection"
    current_collection = f"tf_sync_v2_{task_id[:8]}_current"
    dropped = []

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeClient:
        def __init__(self, _config):
            pass

        def connect(self):
            return None

        def list_collections(self):
            return []

        def drop_collection(self, name):
            dropped.append(name)

        def close(self):
            return None

    monkeypatch.setattr(milvus_module, "MilvusConfig", FakeConfig)
    monkeypatch.setattr(milvus_module, "MilvusClient", FakeClient)
    monkeypatch.setattr(
        milvus_collection_service,
        "get_linked_datasets",
        lambda _collection_name: [
            {"dataset_id": "dataset-owned"},
            {"dataset_id": "dataset-external"},
        ],
    )

    with pytest.raises(sync_routes.SyncMilvusCollectionSharedError):
        sync_routes._delete_sync_milvus_collections(
            {
                "task_id": task_id,
                "milvus_collection_name": current_collection,
                "generation_config": {},
            },
            [],
            owned_dataset_ids=("dataset-owned",),
        )

    assert dropped == []


def test_cascade_milvus_cleanup_fences_before_active_consumer_preflight(
    monkeypatch,
):
    from train_factory.generation.clients import milvus_client as milvus_module

    task_id = "sync-active-collection-consumer"
    current_collection = f"tf_sync_v2_{task_id[:8]}_current"
    events = []

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeClient:
        def __init__(self, _config):
            pass

        def connect(self):
            events.append("connect")

        def list_collections(self):
            return []

        def drop_collection(self, name):
            pytest.fail(f"active consumer collection must not be dropped: {name}")

        def close(self):
            events.append("close")

    def acquire(names, *, deletion_owner, **_kwargs):
        normalized = tuple(sorted(names))
        events.append(("fence", normalized, deletion_owner))
        return SimpleNamespace(
            newly_fenced=normalized,
            created_placeholders=(),
        )

    def list_consumers(names, *, exclude_task_ids):
        events.append(("consumers", tuple(sorted(names)), exclude_task_ids))
        return ["external-generation"]

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

    monkeypatch.setattr(milvus_module, "MilvusConfig", FakeConfig)
    monkeypatch.setattr(milvus_module, "MilvusClient", FakeClient)
    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        acquire,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_collection_consumers",
        list_consumers,
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "restore_deletion_fences",
        restore,
    )

    with pytest.raises(sync_routes.SyncMilvusCollectionConsumerError):
        sync_routes._delete_sync_milvus_collections(
            {
                "task_id": task_id,
                "user_id": "user-1",
                "milvus_collection_name": current_collection,
                "generation_config": {},
            },
            [],
            owned_generation_task_ids=("owned-generation",),
        )

    assert events == [
        "connect",
        ("fence", (current_collection,), f"sync:{task_id}"),
        (
            "consumers",
            (current_collection,),
            ("owned-generation",),
        ),
        (
            "restore",
            (current_collection,),
            f"sync:{task_id}",
            (),
        ),
        "close",
    ]


def test_cascade_milvus_cleanup_rejects_active_deep_evaluation(
    monkeypatch,
):
    from train_factory.generation.clients import milvus_client as milvus_module

    task_id = "sync-active-deep-evaluation"
    current_collection = f"tf_sync_v2_{task_id[:8]}_current"
    events = []

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeClient:
        def __init__(self, _config):
            pass

        def connect(self):
            events.append("connect")

        def list_collections(self):
            return []

        def drop_collection(self, name):
            pytest.fail(f"active consumer collection must not be dropped: {name}")

        def close(self):
            events.append("close")

    def acquire(names, *, deletion_owner, **_kwargs):
        normalized = tuple(sorted(names))
        events.append(("fence", normalized, deletion_owner))
        return SimpleNamespace(
            newly_fenced=normalized,
            created_placeholders=(),
        )

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

    monkeypatch.setattr(milvus_module, "MilvusConfig", FakeConfig)
    monkeypatch.setattr(milvus_module, "MilvusClient", FakeClient)
    monkeypatch.setattr(
        milvus_collection_service,
        "acquire_deletion_fences",
        acquire,
    )
    monkeypatch.setattr(
        deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda names: events.append(
            ("deep-evaluation", tuple(sorted(names)))
        )
        or ["deep-evaluation-1"],
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "restore_deletion_fences",
        restore,
    )

    with pytest.raises(sync_routes.SyncMilvusCollectionConsumerError):
        sync_routes._delete_sync_milvus_collections(
            {
                "task_id": task_id,
                "user_id": "user-1",
                "milvus_collection_name": current_collection,
                "generation_config": {},
            },
            [],
        )

    assert events == [
        "connect",
        ("fence", (current_collection,), f"sync:{task_id}"),
        ("deep-evaluation", (current_collection,)),
        (
            "restore",
            (current_collection,),
            f"sync:{task_id}",
            (),
        ),
        "close",
    ]


def test_sync_cascade_rejects_foreign_lineage_before_dataset_fencing(
    monkeypatch,
):
    task_id = "sync-foreign-lineage"
    generation_task_id = "generation-owned"
    foreign_dataset_id = "dataset-foreign"

    monkeypatch.setattr(
        dataset_service,
        "list_datasets",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": foreign_dataset_id,
            "user_id": "user-1",
            "status": "ready",
            "source_task_type": "generation",
            "source_task_id": "generation-foreign",
        },
    )
    monkeypatch.setattr(
        dataset_service,
        "mark_deleting",
        lambda *_args, **_kwargs: pytest.fail(
            "foreign lineage must be rejected before dataset fencing"
        ),
    )
    monkeypatch.setattr(
        dataset_lineage_service,
        "get_edges_by_task",
        lambda _task_id: [{"to_dataset_id": foreign_dataset_id}],
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "deleting_cascade",
                "is_active": False,
            },
            original_config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "deleting_cascade",
                "is_active": False,
            },
            generation_snapshot=[
                {
                    "generation_task_id": generation_task_id,
                    "output_dataset_id": None,
                }
            ],
            training_snapshot=[],
        )

    assert exc_info.value.status_code == 409


def test_sync_cascade_validates_all_lineage_before_fencing_any_dataset(
    monkeypatch,
):
    task_id = "sync-mixed-lineage"
    generation_task_id = "generation-owned"
    owned_dataset_id = "dataset-a-owned"
    foreign_dataset_id = "dataset-z-foreign"
    marked = []
    records = {
        owned_dataset_id: {
            "dataset_id": owned_dataset_id,
            "user_id": "user-1",
            "status": "ready",
            "source_task_type": "sync",
            "source_task_id": task_id,
        },
        foreign_dataset_id: {
            "dataset_id": foreign_dataset_id,
            "user_id": "user-1",
            "status": "ready",
            "source_task_type": "generation",
            "source_task_id": "generation-foreign",
        },
    }

    def list_datasets(*, source_task_type, **_kwargs):
        if source_task_type == "sync":
            return ([{"dataset_id": owned_dataset_id}], 1)
        return ([], 0)

    monkeypatch.setattr(dataset_service, "list_datasets", list_datasets)
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda dataset_id: dict(records[dataset_id]),
    )
    monkeypatch.setattr(
        dataset_service,
        "mark_deleting",
        lambda dataset_id, **_kwargs: marked.append(dataset_id) or True,
    )
    monkeypatch.setattr(
        dataset_lineage_service,
        "get_edges_by_task",
        lambda _task_id: [{"to_dataset_id": foreign_dataset_id}],
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "deleting_cascade",
                "is_active": False,
            },
            original_config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "deleting_cascade",
                "is_active": False,
            },
            generation_snapshot=[
                {
                    "generation_task_id": generation_task_id,
                    "output_dataset_id": None,
                }
            ],
            training_snapshot=[],
        )

    assert exc_info.value.status_code == 409
    assert marked == []


def test_sync_cascade_provenance_conflict_restores_new_parent_delete_intent(
    monkeypatch,
):
    task_id = "sync-provenance-rollback"
    generation_task_id = "generation-owned"
    foreign_dataset_id = "dataset-foreign"
    events = []

    monkeypatch.setattr(dataset_service, "list_datasets", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": foreign_dataset_id,
            "user_id": "user-1",
            "status": "ready",
            "source_task_type": "generation",
            "source_task_id": "generation-foreign",
        },
    )
    monkeypatch.setattr(
        dataset_service,
        "mark_deleting",
        lambda *_args, **_kwargs: pytest.fail(
            "foreign provenance must fail before dataset fencing"
        ),
    )
    monkeypatch.setattr(
        dataset_lineage_service,
        "get_edges_by_task",
        lambda _task_id: [{"to_dataset_id": foreign_dataset_id}],
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda requested_id, *, status, is_active, **_kwargs: events.append(
            ("cancel", requested_id, status, is_active)
        )
        or True,
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "deleting_cascade",
                "is_active": False,
            },
            original_config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "idle",
                "is_active": True,
            },
            generation_snapshot=[
                {
                    "generation_task_id": generation_task_id,
                    "output_dataset_id": None,
                }
            ],
            training_snapshot=[],
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert events == [("cancel", task_id, "idle", True), "resume"]


def test_sync_cascade_mark_conflict_restores_earlier_dataset_fence(
    monkeypatch,
):
    task_id = "sync-mark-rollback"
    dataset_ids = ["dataset-a", "dataset-b"]
    events = []

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
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "user_id": "user-1",
            "status": "ready",
            "source_task_type": "sync",
            "source_task_id": task_id,
        },
    )

    def mark_deleting(dataset_id, **_kwargs):
        events.append(("mark", dataset_id))
        if dataset_id == dataset_ids[1]:
            raise DatasetDeletionOwnerConflictError("foreign owner")
        return True

    monkeypatch.setattr(dataset_service, "mark_deleting", mark_deleting)
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda dataset_id, **_kwargs: events.append(("restore", dataset_id)) or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda requested_id, *, status, is_active, **_kwargs: events.append(
            ("cancel", requested_id, status, is_active)
        )
        or True,
    )

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "deleting_cascade",
                "is_active": False,
            },
            original_config={
                "task_id": task_id,
                "user_id": "user-1",
                "status": "idle",
                "is_active": True,
            },
            generation_snapshot=[],
            training_snapshot=[],
            resume_worker=lambda: events.append("resume"),
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("mark", "dataset-a"),
        ("mark", "dataset-b"),
        ("restore", "dataset-a"),
        ("cancel", task_id, "idle", True),
        "resume",
    ]


def test_sync_cascade_finalizes_collection_fences_before_parent_tracking(
    monkeypatch,
):
    task_id = "sync-deferred-collection-fence"
    config = {
        "task_id": task_id,
        "user_id": "user-1",
        "status": "deleting_cascade",
        "is_active": False,
    }
    events = []
    monkeypatch.setattr(
        dataset_service,
        "list_datasets",
        lambda **_kwargs: ([], 0),
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
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: events.append("remote_drop")
        or ["collection-1"],
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_child_tasks",
        lambda *_args: events.append("children"),
    )
    monkeypatch.setattr(
        milvus_collection_service,
        "delete_collections",
        lambda names, **kwargs: events.append(
            (
                "finalize_fences",
                tuple(names),
                kwargs["deletion_owner"],
            )
        )
        or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "finalize_task_deletion",
        lambda _task_id, **_kwargs: events.append("parent") or True,
    )

    result = sync_routes._perform_sync_task_deletion(
        task_id,
        cascade=True,
        config=config,
        original_config=config,
    )

    assert result == {"message": "Sync task deleted", "deleted_datasets": []}
    assert events == [
        "remote_drop",
        "children",
        (
            "finalize_fences",
            ("collection-1",),
            f"sync:{task_id}",
        ),
        "parent",
    ]


def test_sync_cascade_classifies_only_external_milvus_links_as_conflicts():
    task_id = "sync-link-owner"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    adapter_id = "adapter-12345678"
    adapter_name = "sync-adapter"
    adapter_hash = hashlib.md5(  # noqa: S324 - legacy name compatibility
        adapter_name.encode("utf-8")
    ).hexdigest()[:8]
    current_collection = f"tf_sync_v2_{task_id[:8]}_current"
    known_legacy_adapter = (
        f"tf_sync_{task_id[:8]}_a{adapter_id[:8]}_{adapter_hash}"
    )
    links = [
        {"collection_name": current_collection},
        {"collection_name": known_legacy_adapter},
        {"collection_name": f"tf_sync_v3_{task_hash}_base"},
        {"collection_name": f"tf_sync_{task_id[:8]}_lookalike"},
        {"collection_name": f"tf_sync_v2_{task_id[:8]}_lookalike"},
        {"collection_name": "shared_external_collection"},
    ]

    conflicts = sync_routes._external_sync_milvus_links(
        {
            "task_id": task_id,
            "milvus_collection_name": current_collection,
        },
        [
            {
                "loaded_adapter_id": adapter_id,
                "loaded_adapter_name": adapter_name,
            }
        ],
        links,
    )

    assert conflicts == [
        {"collection_name": f"tf_sync_{task_id[:8]}_lookalike"},
        {"collection_name": f"tf_sync_v2_{task_id[:8]}_lookalike"},
        {"collection_name": "shared_external_collection"},
    ]


def test_cascade_deletes_terminal_generic_child_tasks(monkeypatch):
    events = []
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "child-owner",
            "status": "completed",
        },
    )
    monkeypatch.setattr(
        generation_task_service,
        "delete_task",
        lambda task_id: events.append(("generation-delete", task_id)) or True,
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "child-owner",
            "status": "succeeded",
            "output_dir": str(get_settings().get_task_output_dir(task_id)),
        },
    )
    monkeypatch.setattr(
        training_task_service,
        "delete_task",
        lambda task_id: events.append(("training-delete", task_id)) or True,
    )

    monkeypatch.setattr(
        sync_routes,
        "_delete_resolved_path",
        lambda path: events.append(
            ("training-output-delete", str(path), "training-child")
        ),
    )

    sync_routes._delete_sync_child_tasks(
        [
            {
                "task_id": "sync-parent",
                "user_id": "child-owner",
                "generation_task_id": "generation-child",
            }
        ],
        [
            {
                "task_id": "sync-parent",
                "user_id": "child-owner",
                "training_task_id": "training-child",
            }
        ],
        "sync-parent",
        "child-owner",
    )

    assert events == [
        ("generation-delete", "generation-child"),
        (
            "training-output-delete",
            str(get_settings().get_task_output_dir("training-child").resolve()),
            "training-child",
        ),
        ("training-delete", "training-child"),
    ]


def test_dataset_snapshot_pagination_has_a_deterministic_tie_breaker(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DatasetDB.__table__.create(engine)
    created_at = datetime(2026, 8, 10, 12, 0, 0)
    with Session(engine) as session:
        session.add_all(
            [
                DatasetDB(
                    dataset_id="dataset-a",
                    dataset_name="Dataset A",
                    user_id="user-1",
                    source_task_type="sync",
                    source_task_id="sync-1",
                    created_at=created_at,
                ),
                DatasetDB(
                    dataset_id="dataset-z",
                    dataset_name="Dataset Z",
                    user_id="user-1",
                    source_task_type="sync",
                    source_task_id="sync-1",
                    created_at=created_at,
                ),
            ]
        )
        session.commit()
    monkeypatch.setattr(dataset_service, "engine", engine)

    first, total = dataset_service.list_datasets(
        source_task_type="sync",
        source_task_id="sync-1",
        user_id="user-1",
        limit=1,
        offset=0,
    )
    second, _ = dataset_service.list_datasets(
        source_task_type="sync",
        source_task_id="sync-1",
        user_id="user-1",
        limit=1,
        offset=1,
    )

    assert total == 2
    assert [first[0]["dataset_id"], second[0]["dataset_id"]] == [
        "dataset-a",
        "dataset-z",
    ]


def test_cascade_delete_snapshots_every_page_before_removing_tracking(monkeypatch):
    task_id = "sync-many-pages"
    user_id = "user-many-pages"
    total = 10_001
    tail_output_id = "generation-output-tail"
    deleted_dataset_ids = []
    events = []
    dataset_offsets = []
    generation_offsets = []

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda requested_id: {
            "task_id": requested_id,
            "user_id": user_id,
            "is_active": True,
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda requested_id: {
            "task_id": requested_id,
            "user_id": user_id,
            "is_active": False,
            "status": "deleting_cascade",
        },
    )

    def list_generations(*, task_id, limit, offset=0, **_kwargs):
        if not generation_offsets:
            events.append(("generation_snapshot_started", task_id))
        generation_offsets.append(offset)
        end = min(offset + limit, total)
        rows = [
                {
                    "id": index + 1,
                    "task_id": task_id,
                    "user_id": user_id,
                    "generation_task_id": (
                        "tail-generation-task" if index == total - 1 else None
                    ),
                "output_dataset_id": (
                    tail_output_id if index == total - 1 else None
                ),
                "status": "completed",
            }
            for index in range(offset, end)
        ]
        return rows, total

    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        list_generations,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda requested_id, *, cascade, **_kwargs: {
            "task_id": requested_id,
            "user_id": user_id,
            "is_active": False,
            "status": "deleting_cascade" if cascade else "deleting",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "finalize_task_deletion",
        lambda requested_id, **_kwargs: events.append(
            ("tracking_deleted", requested_id)
        )
        or True,
    )

    def list_datasets(*, source_task_type, source_task_id, limit, offset=0, **_kwargs):
        if source_task_type == "generation":
            assert source_task_id == "tail-generation-task"
            rows = (
                [
                    {
                        "dataset_id": tail_output_id,
                        "user_id": user_id,
                        "source_task_type": "generation",
                        "source_task_id": source_task_id,
                    }
                ]
                if offset == 0
                else []
            )
            return rows, 1
        assert (source_task_type, source_task_id) == ("sync", task_id)
        dataset_offsets.append(offset)
        end = min(offset + limit, total)
        return [
            {
                "dataset_id": f"sync-dataset-{index}",
                "user_id": user_id,
                "storage_backend": "local",
                "storage_path": None,
                "source_task_type": "sync",
                "source_task_id": task_id,
            }
            for index in range(offset, end)
        ], total

    monkeypatch.setattr(dataset_service, "list_datasets", list_datasets)
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "storage_backend": "local",
            "storage_path": None,
            "source_task_type": (
                "generation" if dataset_id == tail_output_id else "sync"
            ),
            "source_task_id": (
                "tail-generation-task"
                if dataset_id == tail_output_id
                else task_id
            ),
        },
    )
    monkeypatch.setattr(
        dataset_service,
        "delete_dataset",
        lambda dataset_id, **_kwargs: deleted_dataset_ids.append(dataset_id)
        or True,
    )
    monkeypatch.setattr(
        dataset_service,
        "mark_deleting",
        lambda _dataset_id, *, deletion_owner, user_id: (
            deletion_owner == f"sync:{task_id}"
            and user_id == "user-many-pages"
        ),
    )
    monkeypatch.setattr(
        dataset_lineage_service,
        "delete_edges_for_dataset",
        lambda _dataset_id: True,
    )
    monkeypatch.setattr(
        dataset_lineage_service,
        "get_edges_by_task",
        lambda _task_id: [],
    )
    monkeypatch.setattr(
        dataset_asset_service,
        "delete_assets_for_dataset",
        lambda _dataset_id: True,
    )
    monkeypatch.setattr(
        dataset_asset_service,
        "list_assets",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda _dataset_ids, *, exclude_task_ids=(): [],
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda _dataset_paths, *, exclude_task_ids=(): [],
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda _dataset_ids, _dataset_paths: [],
    )

    async def stop_worker(_task_id):
        events.append(("worker_stopped", _task_id))

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)

    result = asyncio.run(
        sync_routes.delete_sync_task(
            task_id,
            cascade=True,
            current_user={"user_id": user_id, "is_admin": False},
        )
    )

    assert len(deleted_dataset_ids) == total + 1
    assert tail_output_id in deleted_dataset_ids
    assert len(set(deleted_dataset_ids)) == total + 1
    assert len(dataset_offsets) > 1
    assert len(generation_offsets) > 1
    assert events[0] == ("worker_stopped", task_id)
    assert events[-1] == ("tracking_deleted", task_id)
    assert result["deleted_datasets"][-1] == tail_output_id


def test_cascade_conflict_restores_an_active_sync_worker(monkeypatch):
    task_id = "sync-active-child"
    user_id = "user-active-child"
    events = []

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": True,
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: (
            [
                {
                    "id": 1,
                    "task_id": task_id,
                    "user_id": user_id,
                    "generation_task_id": "generation-active",
                    "status": "pending",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: pytest.fail("tracking must not be deleted on conflict"),
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _task_id: {"status": "pending", "user_id": user_id},
    )

    async def stop_worker(_task_id):
        events.append(("stop", _task_id))

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(
        sync_manager,
        "start_worker",
        lambda _task_id: events.append(("start", _task_id)),
    )
    monkeypatch.setattr(sync_manager, "_running", True)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.delete_sync_task(
                task_id,
                cascade=True,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [("stop", task_id), ("start", task_id)]


@pytest.mark.parametrize(
    "child_status",
    [
        GenerationStatus.STOPPING,
        GenerationStatus.RECOVERING,
        GenerationStatus.RESTARTING,
    ],
)
@pytest.mark.parametrize("cascade", [False, True])
def test_sync_delete_rejects_transitional_generation_child(
    monkeypatch,
    child_status,
    cascade,
):
    from train_factory.storage.services.background_task_admission_service import (
        background_task_admission_service,
    )

    task_id = f"sync-{child_status}-{cascade}"
    user_id = "user-active-generation"
    events = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": True,
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: (
            [
                {
                    "id": 1,
                    "task_id": task_id,
                    "user_id": user_id,
                    "generation_task_id": "generation-active",
                    "status": child_status,
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "delete intent must not be written while a child is active"
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: pytest.fail(
            "sync tracking must not be deleted while a child is active"
        ),
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda _task_id: {"status": child_status, "user_id": user_id},
    )
    monkeypatch.setattr(
        background_task_admission_service,
        "begin_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "child deletion guard must not be acquired while the child is active"
        ),
    )

    async def stop_worker(_task_id):
        events.append(("stop", _task_id))

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(
        sync_manager,
        "start_worker",
        lambda _task_id: events.append(("start", _task_id)),
    )
    monkeypatch.setattr(sync_manager, "_running", True)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.delete_sync_task(
                task_id,
                cascade=cascade,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [("stop", task_id), ("start", task_id)]


def test_cascade_rejects_training_artifact_dependencies_before_delete_intent(
    monkeypatch,
):
    from train_factory.storage.services.background_task_admission_service import (
        background_task_admission_service,
    )

    task_id = "sync-training-dependency"
    child_id = "sync-training-child"
    user_id = "user-1"
    events = []
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": True,
            "status": "idle",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: (
            [
                {
                    "id": 1,
                    "task_id": task_id,
                    "user_id": user_id,
                    "training_task_id": child_id,
                    "status": "succeeded",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda *_args, **_kwargs: pytest.fail(
            "delete intent must not be written while artifacts are referenced"
        ),
    )
    monkeypatch.setattr(
        training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": child_id,
            "status": "succeeded",
            "user_id": user_id,
            "output_dir": f"/app/output/{child_id}",
            "final_model_path": f"/app/output/{child_id}/final_model",
        },
    )
    monkeypatch.setattr(
        sync_routes,
        "_list_sync_training_artifact_conflicts",
        lambda _snapshot: [child_id],
        raising=False,
    )
    monkeypatch.setattr(
        background_task_admission_service,
        "begin_deletion",
        lambda kind, requested_id: events.append(("guard", kind, requested_id))
        or SimpleNamespace(release=lambda: events.append(("release", requested_id))),
    )

    async def stop_worker(_task_id):
        events.append(("stop", _task_id))

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(sync_manager, "_running", True)
    monkeypatch.setattr(
        sync_manager,
        "start_worker",
        lambda _task_id: events.append(("start", _task_id)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.delete_sync_task(
                task_id,
                cascade=True,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("stop", task_id),
        ("guard", "training", child_id),
        ("release", child_id),
        ("start", task_id),
    ]


def test_delete_holds_task_operation_lock_and_persists_intent_before_cleanup(
    monkeypatch,
):
    from train_factory.sync import sync_worker as sync_worker_module

    task_id = "sync-delete-locked"
    user_id = "user-delete-locked"
    events = []

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": True,
            "status": "idle",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda _task_id, *, cascade, **_kwargs: events.append(("intent", cascade))
        or {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": False,
            "status": "deleting",
        },
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: (
            [
                {
                    "batch_id": "batch-locked",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": "managed-batch.jsonl",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "finalize_task_deletion",
        lambda _task_id, **_kwargs: events.append(("delete", _task_id)) or True,
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_resolved_path",
        lambda *_args, **_kwargs: events.append(("cleanup", task_id)),
    )
    monkeypatch.setattr(
        sync_worker_module,
        "_resolve_managed_batch_path",
        lambda *_args, **_kwargs: Path("managed-batch.jsonl"),
    )

    async def stop_worker(_task_id):
        events.append(("stop", _task_id))

    @asynccontextmanager
    async def operation_lock(_task_id):
        events.append(("lock_enter", _task_id))
        try:
            yield
        finally:
            events.append(("lock_exit", _task_id))

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(sync_manager, "task_operation_lock", operation_lock)

    asyncio.run(
        sync_routes.delete_sync_task(
            task_id,
            cascade=False,
            current_user={"user_id": user_id, "is_admin": False},
        )
    )

    assert events == [
        ("stop", task_id),
        ("lock_enter", task_id),
        ("intent", False),
        ("cleanup", task_id),
        ("delete", task_id),
        ("lock_exit", task_id),
    ]


def test_cleanup_failure_keeps_durable_delete_intent(monkeypatch):
    from train_factory.sync import sync_worker as sync_worker_module

    task_id = "sync-delete-retry"
    user_id = "user-delete-retry"
    events = []

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": True,
            "status": "idle",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda _task_id, *, cascade, **_kwargs: events.append("intent")
        or {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": False,
            "status": "deleting",
        },
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: (
            [
                {
                    "batch_id": "batch-retry",
                    "task_id": task_id,
                    "user_id": user_id,
                    "storage_path": "managed-batch.jsonl",
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: pytest.fail("DB tracking must remain for retry"),
    )

    def fail_cleanup(*_args, **_kwargs):
        events.append("cleanup_failed")
        raise OSError("simulated unlink failure")

    async def stop_worker(_task_id):
        events.append("stop")

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    monkeypatch.setattr(sync_routes, "_delete_resolved_path", fail_cleanup)
    monkeypatch.setattr(
        sync_worker_module,
        "_resolve_managed_batch_path",
        lambda *_args, **_kwargs: Path("managed-batch.jsonl"),
    )
    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(sync_manager, "task_operation_lock", operation_lock)
    monkeypatch.setattr(
        sync_manager,
        "start_worker",
        lambda _task_id: events.append("unexpected_restart"),
    )
    monkeypatch.setattr(sync_manager, "_running", True)

    with pytest.raises(OSError, match="simulated unlink failure"):
        asyncio.run(
            sync_routes.delete_sync_task(
                task_id,
                cascade=False,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )

    assert events == ["stop", "intent", "cleanup_failed"]


def test_safe_delete_path_propagates_unlink_failure(monkeypatch, tmp_path):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    target = managed_root / "batch.jsonl"
    target.write_text("{}\n", encoding="utf-8")

    def fail_remove(_path):
        raise OSError("unlink denied")

    monkeypatch.setattr(sync_routes.os, "remove", fail_remove)

    with pytest.raises(OSError, match="unlink denied"):
        sync_routes._safe_delete_path(
            str(target),
            trusted_roots=[str(managed_root)],
        )


def test_safe_delete_path_rejects_outside_managed_roots(tmp_path):
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside trusted roots"):
        sync_routes._safe_delete_path(
            str(outside),
            trusted_roots=[str(managed_root)],
        )


def test_cascade_preserves_intent_when_generic_consumer_is_active(monkeypatch):
    task_id = "sync-generic-consumer"
    user_id = "user-generic-consumer"
    events = []

    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": True,
        "status": "idle",
        "milvus_collection_name": "tf_sync_sync-gen-consumer_base",
    }
    monkeypatch.setattr(external_sync_service, "get_task", lambda _task_id: config)
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: config,
    )
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda _task_id, *, cascade, **_kwargs: {
            **config,
            "is_active": False,
            "status": "deleting_cascade",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "delete_task",
        lambda _task_id: pytest.fail("parent tracking must remain"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda _task_id, *, status, is_active, **_kwargs: events.append(
            ("cancel", status, is_active)
        )
        or True,
        raising=False,
    )
    monkeypatch.setattr(
        dataset_service,
        "list_datasets",
        lambda **_kwargs: ([{"dataset_id": "dataset-consumed"}], 1),
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": "dataset-consumed",
            "user_id": user_id,
            "status": "ready",
            "storage_backend": "local",
            "storage_path": "managed-dataset.jsonl",
            "source_task_type": "sync",
            "source_task_id": task_id,
        },
    )
    monkeypatch.setattr(
        dataset_service,
        "mark_deleting",
        lambda dataset_id, *, deletion_owner, user_id: (
            deletion_owner == f"sync:{task_id}"
            and events.append(("mark", dataset_id)) is None
        ),
        raising=False,
    )
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda dataset_id, *, deletion_owner, status, user_id: (
            deletion_owner == f"sync:{task_id}"
            and events.append(("restore", dataset_id, status)) is None
        ),
        raising=False,
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda dataset_ids, *, exclude_task_ids=(): ["generic-generation"],
        raising=False,
    )
    monkeypatch.setattr(
        training_task_service,
        "list_active_dataset_consumers",
        lambda dataset_paths, *, exclude_task_ids=(): [],
        raising=False,
    )
    monkeypatch.setattr(
        evaluation_task_service,
        "list_active_dataset_consumers",
        lambda dataset_ids, dataset_paths: [],
    )
    monkeypatch.setattr(
        dataset_lineage_service,
        "get_edges_by_task",
        lambda _task_id: [],
    )
    monkeypatch.setattr(
        sync_routes,
        "_delete_sync_milvus_collections",
        lambda *_args, **_kwargs: pytest.fail(
            "Milvus collections must survive a consumer preflight conflict"
        ),
    )
    monkeypatch.setattr(
        sync_routes,
        "_safe_delete_path",
        lambda *_args, **_kwargs: pytest.fail("active input must not be unlinked"),
    )

    async def stop_worker(_task_id):
        events.append(("stop", _task_id))

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(sync_manager, "_running", True)
    monkeypatch.setattr(
        sync_manager,
        "start_worker",
        lambda _task_id: events.append(("start", _task_id)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.delete_sync_task(
                task_id,
                cascade=True,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("stop", task_id),
        ("mark", "dataset-consumed"),
        ("restore", "dataset-consumed", "ready"),
        ("cancel", "idle", True),
        ("start", task_id),
    ]


def test_delete_retries_an_existing_matching_intent(monkeypatch):
    task_id = "sync-delete-retry-existing"
    user_id = "user-delete-retry-existing"
    events = []
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": False,
        "status": "deleting",
    }

    monkeypatch.setattr(external_sync_service, "get_task", lambda _task_id: config)
    monkeypatch.setattr(
        external_sync_service,
        "begin_task_deletion",
        lambda _task_id, *, cascade, **_kwargs: events.append(("intent", cascade))
        or config,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "finalize_task_deletion",
        lambda _task_id, **_kwargs: events.append(("delete", _task_id)) or True,
    )

    async def stop_worker(_task_id):
        events.append(("stop", _task_id))

    @asynccontextmanager
    async def operation_lock(_task_id):
        events.append(("lock", _task_id))
        yield

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(sync_manager, "task_operation_lock", operation_lock)

    result = asyncio.run(
        sync_routes.delete_sync_task(
            task_id,
            cascade=False,
            current_user={"user_id": user_id, "is_admin": False},
        )
    )

    assert result == {"message": "Sync task deleted"}
    assert events == [
        ("stop", task_id),
        ("lock", task_id),
        ("intent", False),
        ("delete", task_id),
    ]


def test_delete_uses_lock_scoped_snapshot_for_existing_durable_intent(
    monkeypatch,
):
    task_id = "sync-delete-stale-prelock-snapshot"
    user_id = "user-delete-stale-prelock-snapshot"
    prelock = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": True,
        "status": "idle",
    }
    locked = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": False,
        "status": "deleting_cascade",
    }
    snapshots = iter((prelock, locked))
    captured = {}

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: next(snapshots),
    )

    async def stop_worker(_task_id):
        return None

    @asynccontextmanager
    async def operation_lock(_task_id):
        yield

    def delete_locked(_task_id, **kwargs):
        captured.update(kwargs)
        return {"message": "Sync task deleted"}

    monkeypatch.setattr(sync_manager, "stop_worker", stop_worker)
    monkeypatch.setattr(sync_manager, "task_operation_lock", operation_lock)
    monkeypatch.setattr(sync_routes, "_delete_sync_task_locked", delete_locked)

    result = asyncio.run(
        sync_routes.delete_sync_task(
            task_id,
            cascade=True,
            current_user={"user_id": user_id, "is_admin": False},
        )
    )

    assert result == {"message": "Sync task deleted"}
    assert captured["config"] is locked
    assert captured["original_config"] is locked


def test_delete_rejects_changing_an_existing_intent_mode(monkeypatch):
    task_id = "sync-delete-mode-change"
    user_id = "user-delete-mode-change"
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": user_id,
            "is_active": False,
            "status": "deleting",
        },
    )

    async def unexpected_stop(_task_id):
        pytest.fail("mode mismatch must be rejected before stopping a worker")

    monkeypatch.setattr(sync_manager, "stop_worker", unexpected_stop)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sync_routes.delete_sync_task(
                task_id,
                cascade=True,
                current_user={"user_id": user_id, "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 409
    assert "mode cannot change" in str(exc_info.value.detail)


def test_existing_delete_intent_is_not_rolled_back_on_consumer_conflict(
    monkeypatch,
):
    task_id = "sync-delete-preserve-existing"
    user_id = "user-delete-preserve-existing"
    config = {
        "task_id": task_id,
        "user_id": user_id,
        "is_active": False,
        "status": "deleting_cascade",
    }

    monkeypatch.setattr(
        dataset_service,
        "list_datasets",
        lambda **_kwargs: ([{"dataset_id": "dataset-existing-intent"}], 1),
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda _dataset_id: {
            "dataset_id": "dataset-existing-intent",
            "user_id": user_id,
            "status": "deleting",
            "storage_path": "managed-existing-intent.jsonl",
        },
    )
    monkeypatch.setattr(dataset_service, "mark_deleting", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        dataset_service,
        "restore_from_deleting",
        lambda *_args, **_kwargs: pytest.fail("durable intent must not be restored"),
    )
    monkeypatch.setattr(
        external_sync_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: pytest.fail("durable intent must not be cancelled"),
    )
    monkeypatch.setattr(
        generation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: ["generation-consumer"],
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

    with pytest.raises(HTTPException) as exc_info:
        sync_routes._perform_sync_task_deletion(
            task_id,
            cascade=True,
            config=config,
            original_config=config,
        )

    assert exc_info.value.status_code == 409


def test_startup_retries_pending_deletions_and_isolates_failures(monkeypatch):
    tasks = [
        {
            "task_id": "delete-normal",
            "status": "deleting",
            "user_id": "user-1",
            "is_active": False,
        },
        {
            "task_id": "delete-cascade",
            "status": "deleting_cascade",
            "user_id": "user-1",
            "is_active": False,
        },
        {
            "task_id": "keep-idle",
            "status": "idle",
            "user_id": "user-1",
            "is_active": True,
        },
    ]
    calls = []

    def list_tasks(*, limit, offset, **_kwargs):
        return tasks[offset : offset + limit], len(tasks)

    monkeypatch.setattr(external_sync_service, "list_tasks", list_tasks)
    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda task_id: next(task for task in tasks if task["task_id"] == task_id),
    )

    @asynccontextmanager
    async def operation_lock(task_id):
        calls.append(("lock", task_id))
        yield

    def resume(task_id, *, cascade, **_kwargs):
        calls.append(("resume", task_id, cascade))
        if task_id == "delete-cascade":
            raise OSError("simulated cleanup failure")

    monkeypatch.setattr(sync_manager, "task_operation_lock", operation_lock)
    monkeypatch.setattr(sync_routes, "_delete_sync_task_locked", resume)

    resumed, failed = asyncio.run(sync_routes.resume_pending_sync_deletions())

    assert (resumed, failed) == (1, 1)
    assert calls == [
        ("lock", "delete-normal"),
        ("resume", "delete-normal", False),
        ("lock", "delete-cascade"),
        ("resume", "delete-cascade", True),
    ]
