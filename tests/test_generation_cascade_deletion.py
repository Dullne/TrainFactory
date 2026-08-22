import asyncio
import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import generation_routes
from train_factory.storage.entities.dataset_asset_entity import DatasetAssetDB
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.dataset_lineage_entity import (
    DatasetLineageEdgeDB,
)
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncGenerationDB,
)

service_module = importlib.import_module(
    "train_factory.storage.services.generation_task_service"
)


USER = {"user_id": "user-1", "username": "alice"}
TASK_ID = "generation-1"
DATASET_ID = "dataset-1"


def test_generation_output_dataset_listing_pages_until_complete(monkeypatch):
    calls = []
    datasets = [
        {"dataset_id": f"dataset-{index}"}
        for index in range(5)
    ]

    def list_datasets(*, limit, offset, **filters):
        calls.append((limit, offset, filters))
        return datasets[offset : offset + limit], len(datasets)

    monkeypatch.setattr(
        generation_routes,
        "dataset_service",
        SimpleNamespace(list_datasets=list_datasets),
    )
    monkeypatch.setattr(
        generation_routes,
        "_GENERATION_DATASET_DELETE_PAGE_SIZE",
        2,
        raising=False,
    )

    result = generation_routes._list_generation_output_datasets(
        TASK_ID,
    )

    assert [dataset["dataset_id"] for dataset in result] == [
        "dataset-0",
        "dataset-1",
        "dataset-2",
        "dataset-3",
        "dataset-4",
    ]
    assert [offset for _limit, offset, _filters in calls] == [0, 2, 4]
    assert all(
        filters
        == {
            "source_task_type": "generation",
            "source_task_id": TASK_ID,
        }
        for _limit, _offset, filters in calls
    )


def test_generation_cascade_detects_source_dataset_owner_drift(monkeypatch):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "owner-drifted storage must not be deleted"
        ),
    )
    foreign_dataset = {
        **dataset,
        "user_id": "another-user",
    }
    monkeypatch.setattr(
        generation_routes.dataset_lineage_service,
        "get_edges_by_task",
        lambda _task_id: [],
    )

    def list_datasets(**filters):
        if filters.get("user_id") == USER["user_id"]:
            return ([], 0)
        return ([dict(foreign_dataset)], 1)

    monkeypatch.setattr(
        generation_routes.dataset_service,
        "list_datasets",
        list_datasets,
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "get_dataset",
        lambda _dataset_id: dict(foreign_dataset),
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "mark_deleting",
        lambda *_args, **_kwargs: pytest.fail(
            "owner drift must be rejected before fencing"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert atomic_deletes == []
    assert "mark_deleting" not in events


def _configure_cascade(
    monkeypatch,
    *,
    storage_cleanup,
    training_consumers=(),
    milvus_links=(),
    status="ready",
    deletion_owner=None,
    source_task_type="generation",
    source_task_id=TASK_ID,
):
    events = []
    dataset = {
        "dataset_id": DATASET_ID,
        "user_id": USER["user_id"],
        "status": status,
        "storage_backend": "local",
        "storage_path": "/app/data/datasets/generation-1/output.jsonl",
        "storage_uri": None,
        "source_task_type": source_task_type,
        "source_task_id": source_task_id,
        "deletion_owner": deletion_owner,
    }

    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: {
            "task_id": TASK_ID,
            "status": GenerationStatus.COMPLETED,
            "user_id": USER["user_id"],
        },
    )
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "dataset_lineage_service",
        SimpleNamespace(
            get_edges_by_task=lambda _task_id: [
                {"to_dataset_id": DATASET_ID}
            ]
        ),
    )

    def mark_deleting(dataset_id, *, deletion_owner, user_id):
        assert dataset_id == DATASET_ID
        assert user_id == USER["user_id"]
        if dataset["deletion_owner"] not in {None, deletion_owner}:
            raise ValueError("Dataset is owned by another deletion operation")
        events.append("mark_deleting")
        dataset["status"] = "deleting"
        dataset["deletion_owner"] = deletion_owner
        return True

    def restore(dataset_id, *, deletion_owner, status, user_id):
        assert dataset_id == DATASET_ID
        assert user_id == USER["user_id"]
        assert deletion_owner == dataset["deletion_owner"]
        events.append(("restore", status))
        dataset["status"] = status
        dataset["deletion_owner"] = None
        return True

    monkeypatch.setattr(
        generation_routes,
        "dataset_service",
        SimpleNamespace(
            list_datasets=lambda **_kwargs: ([dict(dataset)], 1),
            get_dataset=lambda _dataset_id: dict(dataset),
            get_dataset_by_storage_path=lambda *_args, **_kwargs: dict(dataset),
            list_external_storage_reference_consumers=lambda *_args, **_kwargs: [],
            mark_deleting=mark_deleting,
            restore_from_deleting=restore,
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "dataset_asset_service",
        SimpleNamespace(
            list_assets=lambda **_kwargs: [],
        ),
    )

    atomic_deletes = []
    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        SimpleNamespace(
            list_active_dataset_consumers=lambda *_args, **_kwargs: (
                events.append("generation_consumers") or []
            ),
            list_artifact_reference_consumers=lambda *_args, **_kwargs: [],
            delete_task=lambda _task_id: pytest.fail(
                "cascade must use atomic task and dataset metadata deletion"
            ),
            delete_task_with_datasets=lambda task_id, dataset_ids, *, deletion_owner, user_id: (
                atomic_deletes.append(
                    (task_id, tuple(dataset_ids), user_id, deletion_owner)
                )
                or True
            ),
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "training_task_service",
        SimpleNamespace(
            list_active_dataset_consumers=lambda *_args, **_kwargs: (
                events.append("training_consumers") or list(training_consumers)
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        generation_routes,
        "evaluation_task_service",
        SimpleNamespace(
            list_active_dataset_consumers=lambda *_args, **_kwargs: (
                events.append("evaluation_consumers") or []
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        generation_routes,
        "milvus_collection_service",
        SimpleNamespace(
            list_dataset_links=lambda dataset_ids: (
                events.append(("milvus_links", tuple(dataset_ids)))
                or list(milvus_links)
            )
        ),
    )
    monkeypatch.setattr(generation_routes, "_safe_delete_path", storage_cleanup)
    monkeypatch.setattr(
        generation_routes,
        "_validate_generation_cleanup_manifest",
        lambda *_args, **_kwargs: None,
    )

    return events, dataset, atomic_deletes


def test_generation_cascade_rejects_foreign_lineage_before_fencing_or_cleanup(
    monkeypatch,
):
    events, _dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "foreign lineage storage must not be deleted"
        ),
        source_task_id="another-generation",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert "mark_deleting" not in events
    assert atomic_deletes == []


def test_generation_cascade_validates_all_lineage_before_fencing_any_dataset(
    monkeypatch,
):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "foreign lineage storage must not be deleted"
        ),
    )
    owned_id = "dataset-a-owned"
    foreign_id = "dataset-z-foreign"
    records = {
        owned_id: {
            **dataset,
            "dataset_id": owned_id,
        },
        foreign_id: {
            **dataset,
            "dataset_id": foreign_id,
            "source_task_id": "another-generation",
        },
    }
    monkeypatch.setattr(
        generation_routes.dataset_lineage_service,
        "get_edges_by_task",
        lambda _task_id: [
            {"to_dataset_id": owned_id},
            {"to_dataset_id": foreign_id},
        ],
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "list_datasets",
        lambda **_kwargs: ([dict(records[owned_id])], 1),
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: dict(records[dataset_id]),
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "mark_deleting",
        lambda dataset_id, **_kwargs: events.append(("mark", dataset_id)) or True,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert not any(event[0] == "mark" for event in events if isinstance(event, tuple))
    assert atomic_deletes == []


def test_generation_cascade_marks_dataset_before_consumer_scan(monkeypatch):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "storage cleanup ran despite an active consumer"
        ),
        training_consumers=("training-1",),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert events[:4] == [
        "mark_deleting",
        "generation_consumers",
        "training_consumers",
        "evaluation_consumers",
    ]
    assert events[-1] == ("restore", "ready")
    assert dataset["status"] == "ready"
    assert atomic_deletes == []


def test_generation_cascade_restores_fences_when_preflight_raises(monkeypatch):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "storage cleanup must not run after a preflight failure"
        ),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "list_active_dataset_consumers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("consumer lookup unavailable")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 500
    assert "mark_deleting" in events
    assert events[-1] == ("restore", "ready")
    assert dataset["status"] == "ready"
    assert dataset["deletion_owner"] is None
    assert atomic_deletes == []


def test_generation_cascade_rejects_milvus_link_before_storage_cleanup(
    monkeypatch,
):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "storage cleanup ran despite a Milvus dataset link"
        ),
        milvus_links=(
            {
                "collection_name": "orphaned-collection",
                "dataset_id": DATASET_ID,
            },
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert dataset["status"] == "ready"
    assert ("milvus_links", (DATASET_ID,)) in events
    assert events[-1] == ("restore", "ready")
    assert atomic_deletes == []


def test_generation_cascade_rejects_storage_shared_by_external_asset(
    monkeypatch,
):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "shared storage must not be deleted"
        ),
    )
    shared_path = dataset["storage_path"]
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "list_external_storage_reference_consumers",
        lambda references, *, exclude_dataset_ids: [
            {
                "source": "asset",
                "asset_id": "foreign-asset",
                "dataset_id": "foreign-dataset",
                "reference": next(iter(references)),
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert dataset["status"] == "ready"
    assert dataset["deletion_owner"] is None
    assert shared_path
    assert atomic_deletes == []


def test_generation_cascade_rejects_artifact_shared_by_another_generation(
    monkeypatch,
):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "another generation task's artifact must not be deleted"
        ),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "list_artifact_reference_consumers",
        lambda references, *, exclude_task_ids: [
            {
                "task_id": "foreign-generation",
                "field": "qa_output_path",
                "reference": next(iter(references)),
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert dataset["status"] == "ready"
    assert dataset["deletion_owner"] is None
    assert events[-1] == ("restore", "ready")
    assert atomic_deletes == []


@pytest.mark.parametrize("consumer_kind", ["training", "evaluation"])
def test_generation_cascade_rejects_active_consumer_of_unregistered_artifact(
    monkeypatch,
    consumer_kind,
):
    unregistered_path = "/app/data/generation/unregistered-output.jsonl"
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "active consumer artifact must not be deleted"
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: {
            "task_id": TASK_ID,
            "status": GenerationStatus.COMPLETED,
            "user_id": USER["user_id"],
            "qa_output_path": unregistered_path,
        },
    )
    observed_paths = []

    def training_consumers(paths, **_kwargs):
        observed_paths.append(("training", tuple(paths)))
        return (
            ["external-training"]
            if consumer_kind == "training" and unregistered_path in paths
            else []
        )

    def evaluation_consumers(_dataset_ids, paths, **_kwargs):
        observed_paths.append(("evaluation", tuple(paths)))
        return (
            ["external-evaluation"]
            if consumer_kind == "evaluation" and unregistered_path in paths
            else []
        )

    monkeypatch.setattr(
        generation_routes.training_task_service,
        "list_active_dataset_consumers",
        training_consumers,
    )
    monkeypatch.setattr(
        generation_routes.evaluation_task_service,
        "list_active_dataset_consumers",
        evaluation_consumers,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert any(unregistered_path in paths for _kind, paths in observed_paths)
    assert dataset["status"] == "ready"
    assert dataset["deletion_owner"] is None
    assert events[-1] == ("restore", "ready")
    assert atomic_deletes == []


def test_generation_non_cascade_sees_cross_tenant_dataset_path_reference(
    monkeypatch,
):
    path = "/app/data/datasets/foreign/output.jsonl"
    calls = []
    task = {
        "task_id": TASK_ID,
        "status": GenerationStatus.COMPLETED,
        "user_id": USER["user_id"],
        "output_path": path,
    }
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: task,
    )
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id: None,
    )

    def get_reference(_path, *, user_id):
        calls.append(user_id)
        return (
            {"dataset_id": "foreign-dataset", "user_id": "foreign-user"}
            if user_id is None
            else None
        )

    monkeypatch.setattr(
        generation_routes,
        "dataset_service",
        SimpleNamespace(
            get_dataset_by_storage_path=get_reference,
            list_external_storage_reference_consumers=(
                lambda *_args, **_kwargs: []
            ),
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_validate_generation_cleanup_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_safe_delete_path",
        lambda _path: pytest.fail("foreign dataset storage must survive"),
    )
    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        SimpleNamespace(
            list_artifact_reference_consumers=lambda *_args, **_kwargs: [],
            delete_task=lambda *_args, **_kwargs: True,
        ),
    )

    result = asyncio.run(
        generation_routes._delete_task_impl(
            TASK_ID,
            cascade=False,
            current_user=USER,
        )
    )

    assert result == {"status": "deleted"}
    assert calls == [None]


def test_generation_non_cascade_preserves_artifact_shared_by_dataset_asset(
    monkeypatch,
):
    path = "/app/data/datasets/foreign/shared-output.jsonl"
    task = {
        "task_id": TASK_ID,
        "status": GenerationStatus.COMPLETED,
        "user_id": USER["user_id"],
        "output_path": path,
    }
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: task,
    )
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "dataset_service",
        SimpleNamespace(
            get_dataset_by_storage_path=lambda _path, *, user_id: None,
            list_external_storage_reference_consumers=(
                lambda references: [
                    {
                        "source": "asset",
                        "asset_id": "foreign-asset",
                        "dataset_id": "foreign-dataset",
                        "reference": next(iter(references)),
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_validate_generation_cleanup_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_safe_delete_path",
        lambda _path: pytest.fail("shared dataset asset storage must survive"),
    )
    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        SimpleNamespace(
            list_artifact_reference_consumers=lambda *_args, **_kwargs: [],
            delete_task=lambda *_args, **_kwargs: True,
        ),
    )

    result = asyncio.run(
        generation_routes._delete_task_impl(
            TASK_ID,
            cascade=False,
            current_user=USER,
        )
    )

    assert result == {"status": "deleted"}


def test_generation_non_cascade_preserves_artifact_shared_by_generation_task(
    monkeypatch,
):
    path = "/app/data/generation/shared-output.jsonl"
    task = {
        "task_id": TASK_ID,
        "status": GenerationStatus.COMPLETED,
        "user_id": USER["user_id"],
        "output_path": path,
    }
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: task,
    )
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "dataset_service",
        SimpleNamespace(
            get_dataset_by_storage_path=lambda _path, *, user_id: None,
            list_external_storage_reference_consumers=(
                lambda *_args, **_kwargs: []
            ),
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_validate_generation_cleanup_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_safe_delete_path",
        lambda _path: pytest.fail("shared generation artifact must survive"),
    )
    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        SimpleNamespace(
            list_artifact_reference_consumers=(
                lambda references, *, exclude_task_ids: [
                    {
                        "task_id": "foreign-generation",
                        "field": "input_path",
                        "reference": next(iter(references)),
                    }
                ]
            ),
            delete_task=lambda *_args, **_kwargs: True,
        ),
    )

    result = asyncio.run(
        generation_routes._delete_task_impl(
            TASK_ID,
            cascade=False,
            current_user=USER,
        )
    )

    assert result == {"status": "deleted"}


@pytest.mark.parametrize("consumer_kind", ["training", "evaluation"])
def test_generation_non_cascade_preserves_active_consumer_artifact(
    monkeypatch,
    consumer_kind,
):
    path = "/app/data/generation/unregistered-active-output.jsonl"
    task = {
        "task_id": TASK_ID,
        "status": GenerationStatus.COMPLETED,
        "user_id": USER["user_id"],
        "output_path": path,
    }
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: task,
    )
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "dataset_service",
        SimpleNamespace(
            get_dataset_by_storage_path=lambda _path, *, user_id: None,
            list_external_storage_reference_consumers=(
                lambda *_args, **_kwargs: []
            ),
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        SimpleNamespace(
            list_artifact_reference_consumers=lambda *_args, **_kwargs: [],
            delete_task=lambda *_args, **_kwargs: True,
        ),
    )
    monkeypatch.setattr(
        generation_routes.training_task_service,
        "list_active_dataset_consumers",
        lambda paths, **_kwargs: (
            ["external-training"]
            if consumer_kind == "training" and path in paths
            else []
        ),
    )
    monkeypatch.setattr(
        generation_routes.evaluation_task_service,
        "list_active_dataset_consumers",
        lambda _dataset_ids, paths, **_kwargs: (
            ["external-evaluation"]
            if consumer_kind == "evaluation" and path in paths
            else []
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_validate_generation_cleanup_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_safe_delete_path",
        lambda _path: pytest.fail("active consumer artifact must survive"),
    )

    result = asyncio.run(
        generation_routes._delete_task_impl(
            TASK_ID,
            cascade=False,
            current_user=USER,
        )
    )

    assert result == {"status": "deleted"}


def test_generation_cascade_manifest_validation_restores_parent_and_dataset_fences(
    monkeypatch,
):
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "invalid manifest must be rejected before storage cleanup"
        ),
    )
    parent_rollbacks = []
    monkeypatch.setattr(
        generation_routes,
        "_validate_generation_cleanup_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("unmanaged cleanup target")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
                on_preflight_failure=lambda: parent_rollbacks.append(True),
            )
        )

    assert exc_info.value.status_code == 500
    assert dataset["status"] == "ready"
    assert dataset["deletion_owner"] is None
    assert parent_rollbacks == [True]
    assert atomic_deletes == []


def test_generation_cascade_storage_failure_retains_metadata_and_is_retryable(
    monkeypatch,
):
    fail_cleanup = True

    def storage_cleanup(_path):
        if fail_cleanup:
            raise RuntimeError("storage unavailable")

    _events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=storage_cleanup,
    )

    parent_rollbacks = []
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
                on_preflight_failure=lambda: parent_rollbacks.append(True),
            )
        )

    assert exc_info.value.status_code == 500
    assert dataset["status"] == "deleting"
    assert parent_rollbacks == []
    assert atomic_deletes == []

    fail_cleanup = False
    result = asyncio.run(
        generation_routes._delete_task_impl(
            TASK_ID,
            cascade=True,
            current_user=USER,
        )
    )

    assert result == {"status": "deleted", "deleted_datasets": [DATASET_ID]}
    assert atomic_deletes == [
        (
            TASK_ID,
            (DATASET_ID,),
            USER["user_id"],
            f"generation:{TASK_ID}",
        ),
    ]


def test_generation_delete_path_propagates_storage_failure(monkeypatch, tmp_path):
    managed_root = tmp_path / "datasets"
    target = managed_root / "generation-1" / "output.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    monkeypatch.setattr(
        generation_routes,
        "get_settings",
        lambda: SimpleNamespace(
            datasets_dir=managed_root,
            local_cache_dir=tmp_path / "cache",
        ),
    )
    monkeypatch.setenv("GENERATION_OUTPUT_DIR", str(managed_root))
    monkeypatch.setattr(
        generation_routes.os,
        "remove",
        lambda _path: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(RuntimeError, match="storage cleanup failed"):
        generation_routes._safe_delete_path(str(target))


def test_generation_delete_s3_propagates_storage_failure(monkeypatch):
    class FailingStore:
        bucket = "bucket"

        def delete_object(self, _object_key):
            raise OSError("object store unavailable")

    object_store_module = ModuleType("train_factory.storage.object_store")
    object_store_module.get_object_store = lambda: FailingStore()
    object_store_module.uri_to_key = (
        lambda _uri, *, expected_bucket: "output.jsonl"
    )
    monkeypatch.setitem(
        sys.modules,
        "train_factory.storage.object_store",
        object_store_module,
    )

    with pytest.raises(RuntimeError, match="storage cleanup failed"):
        generation_routes._safe_delete_s3_object(
            "s3://bucket/output.jsonl"
        )


def test_generation_parent_delete_intent_blocks_restart_without_output_datasets(
    monkeypatch,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=TASK_ID,
                task_name="generation",
                input_path="/app/data/datasets/source.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.COMPLETED,
                user_id=USER["user_id"],
            )
        )
        session.commit()

    @contextmanager
    def session_scope():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(service_module, "get_session", session_scope)

    intent = service_module.generation_task_service.begin_task_deletion(
        TASK_ID,
        cascade=True,
        user_id=USER["user_id"],
    )

    assert intent is not None
    assert intent.newly_started is True
    assert intent.previous_status == GenerationStatus.COMPLETED
    assert intent.task["status"] == GenerationStatus.DELETING_CASCADE
    assert not service_module.generation_task_service.update_status(
        TASK_ID,
        GenerationStatus.PENDING,
    )


def test_generation_parent_delete_intent_can_roll_back_before_storage_mutation(
    monkeypatch,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=TASK_ID,
                task_name="generation",
                input_path="/app/data/datasets/source.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.FAILED,
                user_id=USER["user_id"],
            )
        )
        session.commit()

    @contextmanager
    def session_scope():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(service_module, "get_session", session_scope)
    intent = service_module.generation_task_service.begin_task_deletion(
        TASK_ID,
        cascade=True,
        user_id=USER["user_id"],
    )

    assert intent is not None and intent.newly_started
    assert service_module.generation_task_service.cancel_task_deletion(
        TASK_ID,
        cascade=True,
        previous_status=intent.previous_status,
        user_id=USER["user_id"],
    )
    with Session(engine) as session:
        assert session.exec(select(GenerationTaskDB)).one().status == GenerationStatus.FAILED


@pytest.mark.parametrize("preexisting_intent", [False, True])
def test_generation_delete_route_rolls_back_only_new_parent_intent(
    monkeypatch,
    preexisting_intent,
):
    events = []
    terminal_task = {
        "task_id": TASK_ID,
        "status": (
            GenerationStatus.DELETING_CASCADE
            if preexisting_intent
            else GenerationStatus.COMPLETED
        ),
        "user_id": USER["user_id"],
    }
    intent = SimpleNamespace(
        task={**terminal_task, "status": GenerationStatus.DELETING_CASCADE},
        newly_started=not preexisting_intent,
        previous_status=(
            None if preexisting_intent else GenerationStatus.COMPLETED
        ),
    )
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: terminal_task,
    )
    monkeypatch.setattr(
        generation_routes.background_task_admission_service,
        "begin_deletion",
        lambda *_args: SimpleNamespace(
            release=lambda: events.append("release")
        ),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "begin_task_deletion",
        lambda *_args, **_kwargs: events.append("begin") or intent,
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "cancel_task_deletion",
        lambda *_args, **_kwargs: events.append("cancel") or True,
    )

    async def fail_preflight(*_args, on_preflight_failure, **_kwargs):
        on_preflight_failure()
        raise HTTPException(status_code=409, detail="preflight conflict")

    monkeypatch.setattr(generation_routes, "_delete_task_impl", fail_preflight)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.delete_task(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert events == (
        ["begin", "release"]
        if preexisting_intent
        else ["begin", "cancel", "release"]
    )


def test_generation_restart_claim_rejects_fenced_output_dataset(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=TASK_ID,
                task_name="generation",
                input_path="/app/data/datasets/source.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.COMPLETED,
                user_id=USER["user_id"],
            )
        )
        session.add(
            DatasetDB(
                dataset_id=DATASET_ID,
                dataset_name="generated-output",
                status="deleting",
                source_task_type="generation",
                source_task_id=TASK_ID,
                user_id=USER["user_id"],
                deletion_owner=f"generation:{TASK_ID}",
            )
        )
        session.commit()

    @contextmanager
    def session_scope():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(service_module, "get_session", session_scope)

    assert not service_module.generation_task_service.update_status(
        TASK_ID,
        GenerationStatus.PENDING,
    )
    with Session(engine) as session:
        assert (
            session.exec(
                select(GenerationTaskDB).where(
                    GenerationTaskDB.task_id == TASK_ID
                )
            ).one().status
            == GenerationStatus.COMPLETED
        )


def test_atomic_generation_cascade_metadata_delete_rolls_back_on_failure(
    monkeypatch,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=TASK_ID,
                task_name="generation",
                input_path="/app/data/datasets/source.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.DELETING_CASCADE,
                user_id=USER["user_id"],
            )
        )
        session.add(
            DatasetDB(
                dataset_id=DATASET_ID,
                dataset_name="generated-output",
                storage_path="/app/data/datasets/generation-1/output.jsonl",
                status="deleting",
                source_task_type="generation",
                source_task_id=TASK_ID,
                user_id=USER["user_id"],
                deletion_owner=f"generation:{TASK_ID}",
            )
        )
        session.add(
            DatasetAssetDB(
                dataset_id=DATASET_ID,
                storage_uri="s3://bucket/output.jsonl",
            )
        )
        session.add(
            DatasetLineageEdgeDB(
                to_dataset_id=DATASET_ID,
                relation_type="generated",
                op_task_type="generation",
                op_task_id=TASK_ID,
            )
        )
        session.add(
            ExternalSyncGenerationDB(
                task_id="sync-task-1",
                generation_task_id=TASK_ID,
                user_id=USER["user_id"],
            )
        )
        session.commit()

    @contextmanager
    def failing_session():
        with Session(engine) as session:
            original_delete = session.delete

            def fail_on_task(instance):
                if isinstance(instance, GenerationTaskDB):
                    raise RuntimeError("injected metadata failure")
                return original_delete(instance)

            session.delete = fail_on_task
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(service_module, "get_session", failing_session)

    with pytest.raises(RuntimeError, match="injected metadata failure"):
        service_module.generation_task_service.delete_task_with_datasets(
            TASK_ID,
            [DATASET_ID],
            deletion_owner=f"generation:{TASK_ID}",
            user_id=USER["user_id"],
        )

    with Session(engine) as session:
        assert session.exec(select(GenerationTaskDB)).all()
        assert session.exec(select(DatasetDB)).all()
        assert session.exec(select(DatasetAssetDB)).all()
        assert session.exec(select(DatasetLineageEdgeDB)).all()
        assert session.exec(select(ExternalSyncGenerationDB)).all()

    @contextmanager
    def normal_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    sync_service_module = importlib.import_module(
        "train_factory.storage.services.external_sync_service"
    )
    monkeypatch.setattr(service_module, "get_session", normal_session)
    monkeypatch.setattr(
        sync_service_module.external_sync_service,
        "delete_generation_tracking",
        lambda _task_id: True,
    )

    assert service_module.generation_task_service.delete_task_with_datasets(
        TASK_ID,
        [DATASET_ID],
        deletion_owner=f"generation:{TASK_ID}",
        user_id=USER["user_id"],
    )

    with Session(engine) as session:
        assert session.exec(select(GenerationTaskDB)).all() == []
        assert session.exec(select(DatasetDB)).all() == []
        assert session.exec(select(DatasetAssetDB)).all() == []
        assert session.exec(select(DatasetLineageEdgeDB)).all() == []
        assert session.exec(select(ExternalSyncGenerationDB)).all() == []


def test_generation_cascade_foreign_owner_returns_conflict_without_restore(
    monkeypatch,
):
    class ForeignDeletionOwnerError(ValueError):
        pass

    monkeypatch.setattr(
        generation_routes,
        "DatasetDeletionOwnerConflictError",
        ForeignDeletionOwnerError,
        raising=False,
    )
    events, dataset, atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=lambda _path: pytest.fail(
            "storage cleanup ran for a foreign deletion owner"
        ),
        status="deleting",
        deletion_owner="sync:other-task",
    )

    def reject_foreign_owner(*_args, **_kwargs):
        raise ForeignDeletionOwnerError(
            "Dataset is owned by another deletion operation"
        )

    monkeypatch.setattr(
        generation_routes.dataset_service,
        "mark_deleting",
        reject_foreign_owner,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert "another deletion operation" in exc_info.value.detail
    assert dataset["status"] == "deleting"
    assert dataset["deletion_owner"] == "sync:other-task"
    assert not any(
        isinstance(event, tuple) and event[0] == "restore" for event in events
    )
    assert atomic_deletes == []


def test_generation_cascade_never_restores_after_destructive_cleanup(
    monkeypatch,
):
    class ForeignDeletionOwnerError(ValueError):
        pass

    monkeypatch.setattr(
        generation_routes,
        "DatasetDeletionOwnerConflictError",
        ForeignDeletionOwnerError,
        raising=False,
    )
    deleted_paths = []
    events, dataset, _atomic_deletes = _configure_cascade(
        monkeypatch,
        storage_cleanup=deleted_paths.append,
    )

    def reject_atomic_delete(*_args, **_kwargs):
        dataset["deletion_owner"] = "sync:other-task"
        raise ForeignDeletionOwnerError(
            "Dataset is owned by another deletion operation"
        )

    def reject_restore(*_args, **_kwargs):
        events.append(("restore", "attempted-after-cleanup"))
        raise ForeignDeletionOwnerError(
            "Dataset is owned by another deletion operation"
        )

    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "delete_task_with_datasets",
        reject_atomic_delete,
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "restore_from_deleting",
        reject_restore,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes._delete_task_impl(
                TASK_ID,
                cascade=True,
                current_user=USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert deleted_paths
    assert dataset["status"] == "deleting"
    assert dataset["deletion_owner"] == "sync:other-task"
    assert not any(
        isinstance(event, tuple) and event[0] == "restore" for event in events
    )


def test_atomic_generation_delete_rejects_foreign_deletion_owner(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=TASK_ID,
                task_name="generation",
                input_path="/app/data/datasets/source.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.DELETING_CASCADE,
                user_id=USER["user_id"],
            )
        )
        session.add(
            DatasetDB(
                dataset_id=DATASET_ID,
                dataset_name="generated-output",
                status="deleting",
                deletion_owner="sync:other-task",
                source_task_type="generation",
                source_task_id=TASK_ID,
                user_id=USER["user_id"],
            )
        )
        session.commit()

    @contextmanager
    def session_scope():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(service_module, "get_session", session_scope)

    with pytest.raises(ValueError, match="another deletion operation"):
        service_module.generation_task_service.delete_task_with_datasets(
            TASK_ID,
            [DATASET_ID],
            deletion_owner=f"generation:{TASK_ID}",
            user_id=USER["user_id"],
        )

    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == DATASET_ID)
        ).one()
        assert dataset.status == "deleting"
        assert dataset.deletion_owner == "sync:other-task"
        assert session.exec(select(GenerationTaskDB)).one().task_id == TASK_ID
