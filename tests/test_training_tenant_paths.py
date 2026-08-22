import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import training_routes
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.services.training_task_service import (
    _persisted_training_dataset_paths,
)
from train_factory.enums import TrainingStatus


CURRENT_USER = {"user_id": "user-1", "username": "alice"}


def _settings(tmp_path: Path):
    output_dir = tmp_path / "output"
    return SimpleNamespace(
        auth_enabled=True,
        datasets_dir=tmp_path / "datasets",
        models_dir=tmp_path / "models",
        minio_bucket="trainfactory",
        output_dir=output_dir,
        get_task_output_dir=lambda task_id: output_dir / task_id,
    )


def _patch_owned_training_inputs(monkeypatch, tmp_path: Path, *, guide_model_id=None):
    settings = _settings(tmp_path)
    model_ids = ["base-model"]
    if guide_model_id:
        model_ids.append(guide_model_id)
    model_paths = {}
    for model_id in model_ids:
        model_path = settings.models_dir / model_id
        model_path.mkdir(parents=True)
        model_paths[model_id] = model_path.resolve()

    dataset_id = "dataset-1"
    dataset_path = settings.datasets_dir / dataset_id
    dataset_path.mkdir(parents=True)

    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes,
        "map_storage_path",
        lambda path: (path, None),
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )

    def get_model_by_path(path, user_id=None):
        requested = Path(path).resolve()
        for model_id, model_path in model_paths.items():
            if requested == model_path:
                return {
                    "model_id": model_id,
                    "model_path": str(model_path),
                    "source_type": "downloaded",
                    "status": "available",
                    "user_id": user_id,
                }
        return None

    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        get_model_by_path,
    )
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": str(dataset_path.resolve()),
            "source_type": "uploaded",
            "status": "ready",
            "user_id": user_id,
        },
    )
    return (
        str(model_paths["base-model"]),
        str(dataset_path.resolve()),
        str(model_paths[guide_model_id]) if guide_model_id else None,
    )


def test_training_local_model_and_dataset_must_be_owned(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda path, user_id=None: None,
    )
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "base_model_path": "/app/models/foreign",
                "dataset_configs": [
                    {"path": "/app/data/datasets/foreign.jsonl", "split": "train"}
                ],
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize("user_id", ["user-1", "admin-user"])
def test_training_rejects_owned_registry_record_that_launders_arbitrary_model_path(
    monkeypatch,
    tmp_path,
    user_id,
):
    settings = _settings(tmp_path)
    arbitrary_path = tmp_path / "arbitrary" / "model"
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda path, user_id=None: {
            "model_id": "model-1",
            "model_path": path,
            "source_type": "trained",
            "status": "available",
            "user_id": user_id,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "base_model_path": str(arbitrary_path),
                "train_dataset_path": "org/dataset",
            },
            user_id,
        )

    assert exc_info.value.status_code == 403


def test_training_accepts_api_managed_downloaded_model(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    model_id = "model-1"
    model_path = settings.models_dir / model_id
    model_path.mkdir(parents=True)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda path, user_id=None: {
            "model_id": model_id,
            "model_path": path,
            "source_type": "downloaded",
            "status": "available",
            "user_id": user_id,
        },
    )

    training_routes._validate_owned_training_resources(
        {
            "base_model_path": str(model_path),
        },
        "user-1",
    )


def test_training_accepts_owned_succeeded_training_model(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    model_path = settings.output_dir / "task-model" / "final_model"
    model_path.mkdir(parents=True)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda path, user_id=None: {
            "model_id": "trained-model",
            "model_path": path,
            "source_type": "trained",
            "source_task_id": "task-model",
            "status": "available",
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "status": "succeeded",
            "final_model_path": str(model_path),
        },
    )

    training_routes._validate_owned_training_resources(
        {
            "base_model_path": str(model_path),
        },
        "user-1",
    )


@pytest.mark.parametrize(
    ("source_type", "registry_status"),
    [("trained", "archived"), ("future-source", "available")],
)
def test_training_rejects_unavailable_or_unknown_training_model_record(
    monkeypatch,
    tmp_path,
    source_type,
    registry_status,
):
    settings = _settings(tmp_path)
    model_path = settings.output_dir / "task-model" / "final_model"
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda path, user_id=None: {
            "model_id": "trained-model",
            "model_path": path,
            "source_type": source_type,
            "source_task_id": "task-model",
            "status": registry_status,
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "status": "succeeded",
            "final_model_path": str(model_path),
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "base_model_path": str(model_path),
                "train_dataset_path": "org/dataset",
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


def test_training_rejects_owned_dataset_record_with_unmanaged_local_path(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    arbitrary_path = tmp_path / "arbitrary" / "train.jsonl"
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": "dataset-1",
            "storage_backend": "local",
            "storage_path": path,
            "source_type": "uploaded",
            "status": "ready",
            "user_id": user_id,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "train_dataset_path": str(arbitrary_path),
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize("source_type", ["uploaded", "huggingface", "modelscope"])
def test_training_accepts_api_managed_local_dataset(
    monkeypatch,
    tmp_path,
    source_type,
):
    settings = _settings(tmp_path)
    dataset_id = "dataset-1"
    dataset_path = settings.datasets_dir / dataset_id
    dataset_path.mkdir(parents=True)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": path,
            "source_type": source_type,
            "status": "ready",
            "user_id": user_id,
        },
    )

    training_routes._validate_owned_training_resources(
        {
            "train_dataset_path": str(dataset_path),
        },
        "user-1",
    )


def test_training_rejects_downloaded_model_symlink_escape(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    model_id = "model-escape"
    outside_path = tmp_path / "outside-model"
    outside_path.mkdir()
    managed_link = settings.models_dir / model_id
    managed_link.parent.mkdir(parents=True)
    try:
        managed_link.symlink_to(outside_path, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda path, user_id=None: {
            "model_id": model_id,
            "model_path": path,
            "source_type": "downloaded",
            "status": "available",
            "user_id": user_id,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "base_model_path": str(managed_link),
                "train_dataset_path": "org/dataset",
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


def test_model_provenance_rejects_managed_link_whose_physical_target_escapes(
    monkeypatch,
):
    from train_factory.auth import resource_provenance

    managed_root = Path("/managed/models")
    requested_path = managed_root / "model-escape"

    def fake_canonical_path(value):
        if Path(value) == managed_root:
            return str(managed_root)
        return "/outside/model-escape"

    monkeypatch.setattr(
        resource_provenance,
        "_canonical_local_path",
        fake_canonical_path,
    )

    with pytest.raises(resource_provenance.ResourceProvenanceError):
        resource_provenance.require_managed_model_provenance(
            {
                "model_id": "model-escape",
                "model_path": str(requested_path),
                "source_type": "downloaded",
                "status": "available",
                "user_id": "user-1",
            },
            str(requested_path),
            user_id="user-1",
            models_dir=managed_root,
            training_output_dir=Path("/managed/output"),
            training_task_lookup=lambda task_id: None,
        )


def test_training_keeps_legacy_resource_compatibility_when_auth_is_disabled(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    settings.auth_enabled = False
    resource_reads = []
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda *args, **kwargs: resource_reads.append("model"),
    )
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda *args, **kwargs: resource_reads.append("dataset"),
    )

    training_routes._validate_owned_training_resources(
        {
            "base_model_path": str(tmp_path / "legacy-model"),
            "train_dataset_path": str(tmp_path / "legacy-dataset.jsonl"),
        },
        "legacy-user",
    )

    assert resource_reads == []


@pytest.mark.parametrize(
    ("config", "expected_detail"),
    [
        (
            {"base_model_path": "Qwen/Qwen3-Embedding-0.6B"},
            "downloaded and registered",
        ),
        (
            {"train_dataset_path": "org/dataset"},
            "uploaded, downloaded, or generated",
        ),
    ],
)
def test_training_rejects_unregistered_repository_ids_when_auth_is_enabled(
    monkeypatch,
    tmp_path,
    config,
    expected_detail,
):
    monkeypatch.setattr(training_routes, "get_settings", lambda: _settings(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(config, "user-1")

    assert exc_info.value.status_code == 403
    assert expected_detail in exc_info.value.detail


def test_create_training_rejects_repo_model_before_task_persistence(
    monkeypatch,
    tmp_path,
):
    task_creations = []
    monkeypatch.setattr(training_routes, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        training_routes,
        "check_idempotency",
        lambda *args, **kwargs: (False, None),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs),
    )
    background_tasks = BackgroundTasks()
    request = training_routes.TrainingRequest(
        base_model_path="Qwen/Qwen3-Embedding-0.6B",
        datasets=[{"path": "org/dataset", "split": "train"}],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.create_training_task(
                request,
                background_tasks,
                CURRENT_USER,
                idempotency_key=None,
            )
        )

    assert exc_info.value.status_code == 403
    assert "downloaded and registered" in exc_info.value.detail
    assert task_creations == []
    assert background_tasks.tasks == []


def test_training_accepts_owned_api_managed_guide_model(monkeypatch, tmp_path):
    _base_model_path, _dataset_path, guide_model_path = _patch_owned_training_inputs(
        monkeypatch,
        tmp_path,
        guide_model_id="guide-model",
    )

    training_routes._validate_owned_training_resources(
        {"loss_config": {"guide_model": guide_model_path}},
        "user-1",
    )


def test_create_training_rejects_unregistered_guide_before_task_persistence(
    monkeypatch,
    tmp_path,
):
    base_model_path, dataset_path, _guide_model_path = _patch_owned_training_inputs(
        monkeypatch,
        tmp_path,
    )
    task_creations = []
    monkeypatch.setattr(
        training_routes,
        "check_idempotency",
        lambda *args, **kwargs: (False, None),
    )
    monkeypatch.setattr(
        training_routes,
        "store_idempotency_response",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "create_task",
        lambda **kwargs: task_creations.append(kwargs)
        or {"task_id": "task-guide", "task_name": "guide test"},
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda *args, **kwargs: True,
    )
    request = training_routes.TrainingRequest(
        base_model_path=base_model_path,
        datasets=[{"path": dataset_path, "split": "train"}],
        embedding_loss_name="GISTEmbedLoss",
        loss_config={"guide_model": "sentence-transformers/all-MiniLM-L6-v2"},
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.create_training_task(
                request,
                background_tasks,
                CURRENT_USER,
                idempotency_key=None,
            )
        )

    assert exc_info.value.status_code == 403
    assert "Guide model" in exc_info.value.detail
    assert task_creations == []
    assert background_tasks.tasks == []


def test_worker_rejects_persisted_unregistered_guide_before_gpu_allocation(
    monkeypatch,
    tmp_path,
):
    base_model_path, dataset_path, _guide_model_path = _patch_owned_training_inputs(
        monkeypatch,
        tmp_path,
    )
    allocation_calls = []
    status_updates = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "status": TrainingStatus.PREPARING.value,
            "run_token": "guide-model-run",
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        lambda *args, **kwargs: allocation_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda *args, **kwargs: True,
    )

    training_routes.run_training_task(
        "task-guide",
        {
            "_run_token": "guide-model-run",
            "user_id": "user-1",
            "base_model_path": base_model_path,
            "train_dataset_path": dataset_path,
            "dataset_configs": [{"path": dataset_path, "split": "train"}],
            "embedding_loss_name": "GISTEmbedLoss",
            "loss_config": {"guide_model": str(tmp_path / "unmanaged-guide")},
        },
    )

    assert allocation_calls == []
    assert any(
        args[1] == "failed" and "guide model" in args[2].lower()
        for args, _kwargs in status_updates
        if len(args) >= 3
    )


def test_resume_training_rejects_repo_resources_before_checkpoint_lookup(
    monkeypatch,
    tmp_path,
):
    task = {
        "task_id": "task-repo",
        "user_id": "user-1",
        "status": "failed",
        "base_model_path": "Qwen/Qwen3-Embedding-0.6B",
        "train_dataset_path": "org/dataset",
        "training_params": {
            "base_model_path": "Qwen/Qwen3-Embedding-0.6B",
            "train_dataset_path": "org/dataset",
        },
    }
    checkpoint_lookups = []
    monkeypatch.setattr(training_routes, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: task,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "find_latest_checkpoint",
        lambda task_id: checkpoint_lookups.append(task_id),
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.resume_task(
                "task-repo",
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert checkpoint_lookups == []
    assert background_tasks.tasks == []


def test_worker_rejects_repo_resources_before_gpu_allocation(monkeypatch, tmp_path):
    allocation_calls = []
    status_updates = []
    monkeypatch.setattr(training_routes, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "status": TrainingStatus.PREPARING.value,
            "run_token": "repo-resource-run",
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        lambda *args, **kwargs: allocation_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda *args, **kwargs: True,
    )

    training_routes.run_training_task(
        "task-repo",
        {
            "_run_token": "repo-resource-run",
            "user_id": "user-1",
            "base_model_path": "Qwen/Qwen3-Embedding-0.6B",
            "train_dataset_path": "org/dataset",
        },
    )

    assert allocation_calls == []
    assert any(
        args[1] == "failed" and "downloaded and registered" in args[2]
        for args, _kwargs in status_updates
        if len(args) >= 3
    )


def test_training_empty_dataset_configs_falls_back_to_top_level_path(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "dataset_configs": [],
                "train_dataset_path": "/app/data/datasets/foreign.jsonl",
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize("owned", [True, False])
def test_training_s3_dataset_requires_same_user_registration(
    monkeypatch,
    tmp_path,
    owned,
):
    settings = _settings(tmp_path)
    dataset_id = "dataset-1"
    storage_uri = f"s3://trainfactory/datasets/{dataset_id}/train.jsonl"
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: (
            {
                "dataset_id": dataset_id,
                "storage_backend": "s3",
                "storage_path": None,
                "storage_uri": path,
                "source_type": "uploaded",
                "status": "ready",
                "user_id": user_id,
            }
            if owned
            else None
        ),
    )
    config = {
        "dataset_configs": [
            {"path": storage_uri, "split": "train"}
        ],
    }

    if owned:
        training_routes._validate_owned_training_resources(config, "user-1")
    else:
        with pytest.raises(HTTPException) as exc_info:
            training_routes._validate_owned_training_resources(config, "user-1")
        assert exc_info.value.status_code == 403


def test_training_rejects_owned_s3_record_outside_dataset_id_prefix(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    storage_uri = "s3://trainfactory/legacy/train.jsonl"
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": "dataset-1",
            "storage_backend": "s3",
            "storage_path": None,
            "storage_uri": path,
            "source_type": "uploaded",
            "status": "ready",
            "user_id": user_id,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "train_dataset_path": storage_uri,
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    "output_filename",
    [
        "generated_abcdef12.jsonl",
        "generated_abcdef12-generation-task.jsonl",
    ],
)
def test_training_accepts_owned_completed_generation_dataset(
    monkeypatch,
    tmp_path,
    output_filename,
):
    from train_factory.storage.services.generation_task_service import (
        generation_task_service,
    )

    settings = _settings(tmp_path)
    task_id = "abcdef12-generation-task"
    dataset_id = "dataset-generated"
    output_path = settings.datasets_dir / output_filename
    output_path.parent.mkdir(parents=True)
    output_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("GENERATION_OUTPUT_DIR", str(settings.datasets_dir))
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": path,
            "source_type": "generated",
            "source_task_type": "generation",
            "source_task_id": task_id,
            "status": "ready",
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda requested_task_id: {
            "task_id": requested_task_id,
            "user_id": "user-1",
            "status": "completed",
            "output_path": str(output_path),
            "output_dataset_id": dataset_id,
        },
    )

    training_routes._validate_owned_training_resources(
        {
            "train_dataset_path": str(output_path),
        },
        "user-1",
    )


def test_training_rejects_generated_dataset_with_foreign_source_task(
    monkeypatch,
    tmp_path,
):
    from train_factory.storage.services.generation_task_service import (
        generation_task_service,
    )

    settings = _settings(tmp_path)
    task_id = "abcdef12-generation-task"
    dataset_id = "dataset-generated"
    output_path = settings.datasets_dir / "generated_abcdef12.jsonl"
    monkeypatch.setenv("GENERATION_OUTPUT_DIR", str(settings.datasets_dir))
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": path,
            "source_type": "generated",
            "source_task_type": "generation",
            "source_task_id": task_id,
            "status": "ready",
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task",
        lambda requested_task_id: {
            "task_id": requested_task_id,
            "user_id": "user-2",
            "status": "completed",
            "output_path": str(output_path),
            "output_dataset_id": dataset_id,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "train_dataset_path": str(output_path),
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


def test_training_accepts_sync_dataset_bound_to_tracked_batch(monkeypatch, tmp_path):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    sync_root = tmp_path / "sync"
    sync_task_id = "12345678-sync-task"
    dataset_id = "dataset-sync"
    dataset_path = (
        sync_root
        / "user-1"
        / "12345678"
        / "merged"
        / "merged_20260809_101112.jsonl"
    )
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": path,
            "source_type": "generated",
            "source_task_type": "sync",
            "source_task_id": sync_task_id,
            "extra_metadata": {"sync_task_id": sync_task_id},
            "status": "ready",
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda task_id: {"task_id": task_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda training_task_id: None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda task_id, status=None, limit=50, offset=0: (
            [
                {
                    "batch_id": "batch-1",
                    "task_id": task_id,
                    "user_id": "user-1",
                    "dataset_id": dataset_id,
                }
            ],
            1,
        ),
    )

    training_routes._validate_owned_training_resources(
        {"train_dataset_path": str(dataset_path)},
        "user-1",
    )


def test_training_accepts_new_sync_attempt_layout(monkeypatch, tmp_path):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    sync_root = tmp_path / "sync"
    sync_task_id = "12345678-sync-task"
    dataset_id = "dataset-sync"
    dataset_path = (
        sync_root
        / "user-1"
        / sync_task_id
        / "merged"
        / "merged_0123456789abcdef0123456789abcdef.jsonl"
    )
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": path,
            "source_type": "generated",
            "source_task_type": "sync",
            "source_task_id": sync_task_id,
            "extra_metadata": {"sync_task_id": sync_task_id},
            "status": "ready",
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda task_id: {"task_id": task_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _training_task_id: None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda task_id, status=None, limit=50, offset=0: (
            [
                {
                    "batch_id": "batch-1",
                    "task_id": task_id,
                    "user_id": "user-1",
                    "dataset_id": dataset_id,
                }
            ],
            1,
        ),
    )

    training_routes._validate_owned_training_resources(
        {"train_dataset_path": str(dataset_path)},
        "user-1",
    )


def test_training_accepts_sync_dataset_listed_by_sync_training(monkeypatch, tmp_path):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    sync_root = tmp_path / "sync"
    sync_task_id = "12345678-sync-task"
    training_task_id = "training-task"
    dataset_id = "dataset-sync"
    dataset_path = (
        sync_root
        / "user-1"
        / "12345678"
        / "merged"
        / "merged_20260809_101112.jsonl"
    )
    monkeypatch.setenv("SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": path,
            "source_type": "generated",
            "source_task_type": "sync",
            "source_task_id": sync_task_id,
            "extra_metadata": {"sync_task_id": sync_task_id},
            "status": "ready",
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda task_id: {"task_id": task_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda requested_task_id: {
            "training_task_id": requested_task_id,
            "task_id": sync_task_id,
            "user_id": "user-1",
            "input_dataset_ids": [dataset_id],
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda *args, **kwargs: ([], 0),
    )

    training_routes._validate_owned_training_resources(
        {
            "task_id": training_task_id,
            "train_dataset_path": str(dataset_path),
        },
        "user-1",
    )


def test_training_rejects_sync_dataset_with_mismatched_training_tracking(
    monkeypatch,
    tmp_path,
):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    sync_root = tmp_path / "sync"
    sync_task_id = "12345678-sync-task"
    dataset_id = "dataset-sync"
    dataset_path = (
        sync_root
        / "user-1"
        / "12345678"
        / "merged"
        / "merged_20260809_101112.jsonl"
    )
    monkeypatch.setenv("SYNC_DATA_DIR", str(sync_root))
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": dataset_id,
            "storage_backend": "local",
            "storage_path": path,
            "source_type": "generated",
            "source_task_type": "sync",
            "source_task_id": sync_task_id,
            "extra_metadata": {"sync_task_id": sync_task_id},
            "status": "ready",
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda task_id: {"task_id": task_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda requested_task_id: {
            "training_task_id": requested_task_id,
            "task_id": sync_task_id,
            "user_id": "user-1",
            "input_dataset_ids": ["other-dataset"],
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_batches",
        lambda *args, **kwargs: (
            [
                {
                    "task_id": sync_task_id,
                    "user_id": "user-1",
                    "dataset_id": dataset_id,
                }
            ],
            1,
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_owned_training_resources(
            {
                "task_id": "training-task",
                "train_dataset_path": str(dataset_path),
            },
            "user-1",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    ("outside_sync_root", "sync_task_owner"),
    [
        (True, "user-1"),
        (False, "user-2"),
    ],
)
def test_sync_dataset_requires_controlled_path_and_owned_source_task(
    tmp_path,
    outside_sync_root,
    sync_task_owner,
):
    from train_factory.auth.resource_provenance import (
        ResourceProvenanceError,
        require_managed_dataset_provenance,
    )

    sync_root = tmp_path / "sync"
    sync_task_id = "12345678-sync-task"
    dataset_id = "dataset-sync"
    path_root = tmp_path / "outside" if outside_sync_root else sync_root
    dataset_path = (
        path_root
        / "user-1"
        / "12345678"
        / "merged"
        / "merged_20260809_101112.jsonl"
    )
    dataset = {
        "dataset_id": dataset_id,
        "storage_backend": "local",
        "storage_path": str(dataset_path),
        "source_type": "generated",
        "source_task_type": "sync",
        "source_task_id": sync_task_id,
        "extra_metadata": {"sync_task_id": sync_task_id},
        "status": "ready",
        "user_id": "user-1",
    }

    with pytest.raises(ResourceProvenanceError):
        require_managed_dataset_provenance(
            dataset,
            str(dataset_path),
            user_id="user-1",
            datasets_dir=tmp_path / "datasets",
            s3_bucket="trainfactory",
            generation_output_dir=tmp_path / "generation",
            generation_task_lookup=lambda task_id: None,
            sync_data_dir=sync_root,
            sync_task_lookup=lambda task_id: {
                "task_id": task_id,
                "user_id": sync_task_owner,
            },
            sync_batch_lookup=lambda task_id: [
                {
                    "task_id": task_id,
                    "user_id": "user-1",
                    "dataset_id": dataset_id,
                }
            ],
        )


def test_training_checkpoint_rejects_foreign_parent_before_path_use(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {"task_id": task_id, "user_id": "user-2"},
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_training_parent_checkpoint(
            "parent-1",
            str(settings.get_task_output_dir("parent-1") / "checkpoint-1"),
            CURRENT_USER,
        )

    assert exc_info.value.status_code == 403


def test_training_checkpoint_must_stay_inside_parent_task_output(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {"task_id": task_id, "user_id": "user-1"},
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_training_parent_checkpoint(
            "parent-1",
            str(settings.get_task_output_dir("other-task") / "checkpoint-1"),
            CURRENT_USER,
        )

    assert exc_info.value.status_code == 403


def test_training_checkpoint_accepts_owned_sync_parent_output(
    monkeypatch,
    tmp_path,
):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    sync_output = (
        settings.output_dir
        / "sync"
        / "sync-config"
        / "target-1"
        / "round_2"
    )
    checkpoint = sync_output / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "output_dir": str(sync_output),
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda task_id: {
            "training_task_id": task_id,
            "task_id": "sync-config",
            "target_id": "target-1",
            "training_round": 2,
            "user_id": "user-1",
        },
    )

    training_routes._validate_training_parent_checkpoint(
        "parent-sync",
        str(checkpoint),
        CURRENT_USER,
    )


def test_training_checkpoint_requires_parent_when_auth_is_enabled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(training_routes, "get_settings", lambda: _settings(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_training_parent_checkpoint(
            None,
            "/app/output/unknown/checkpoint-1",
            CURRENT_USER,
        )

    assert exc_info.value.status_code == 400


def test_task_output_is_always_server_managed(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    config = {"output_dir": str(settings.output_dir / "foreign-task")}

    output_dir = training_routes._set_task_scoped_output("task-1", config)

    assert output_dir == str(settings.output_dir / "task-1")
    assert config["output_dir"] == output_dir


def test_background_training_revalidates_checkpoint_before_gpu_allocation(
    monkeypatch,
    tmp_path,
):
    allocation_calls = []
    status_updates = []
    monkeypatch.setattr(training_routes, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "status": TrainingStatus.PREPARING.value,
            "run_token": "checkpoint-validation-run",
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_process_info",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        lambda *args, **kwargs: allocation_calls.append((args, kwargs)) or "cpu",
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda *args, **kwargs: True,
    )

    training_routes.run_training_task(
        "task-1",
        {
            "_run_token": "checkpoint-validation-run",
            "user_id": "user-1",
            "sft_checkpoint_path": "/app/output/foreign/checkpoint-1",
        },
    )

    assert allocation_calls == []
    assert any(
        args[1] == "failed" and "requires parent_task_id" in args[2]
        for args, _kwargs in status_updates
        if len(args) >= 3
    )


def test_resume_preserves_verified_sync_output_and_uses_resolved_checkpoint(
    monkeypatch,
    tmp_path,
):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    sync_output = (
        settings.output_dir
        / "sync"
        / "sync-config"
        / "target-1"
        / "round_2"
    )
    checkpoint = sync_output / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    task = {
        "task_id": "task-sync",
        "task_name": "sync training",
        "user_id": "user-1",
        "status": "failed",
        "base_model_path": "Qwen/Qwen3-Embedding-0.6B",
        "train_dataset_path": "org/dataset",
        "output_dir": str(sync_output),
        "training_params": {
            "base_model_path": "Qwen/Qwen3-Embedding-0.6B",
            "train_dataset_path": "org/dataset",
            "output_dir": str(sync_output),
        },
    }
    output_updates = []
    lease = SimpleNamespace(release=lambda: None)

    def admit(kind, task_id, admission_user_id, operation, *args, **kwargs):
        assert (kind, task_id, admission_user_id) == (
            "training",
            "task-sync",
            "user-1",
        )
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: task,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "find_latest_checkpoint",
        lambda task_id: str(checkpoint),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda task_id, output_dir, training_params: output_updates.append(
            (task_id, output_dir, dict(training_params))
        )
        or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "reset_for_resume",
        lambda task_id, run_token, **_kwargs: True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda task_id: {
            "training_task_id": task_id,
            "task_id": "sync-config",
            "target_id": "target-1",
            "training_round": 2,
            "user_id": "user-1",
        },
    )

    background_tasks = BackgroundTasks()
    response = asyncio.run(
        training_routes.resume_task(
            "task-sync",
            background_tasks,
            CURRENT_USER,
        )
    )

    resolved_output = str(sync_output.resolve())
    resolved_checkpoint = str(checkpoint.resolve())
    assert response["checkpoint"] == resolved_checkpoint
    assert output_updates == [
        (
            "task-sync",
            resolved_output,
            {
                **task["training_params"],
                "task_id": "task-sync",
                "user_id": "user-1",
                "tuner_type": "full",
                "use_lora": False,
                "lora_config": {"use_lora": False},
                "output_dir": resolved_output,
                "resume_from_checkpoint": resolved_checkpoint,
            },
        )
    ]
    assert background_tasks.tasks[0].func is training_routes.background_task_admission_service.run_sync
    assert background_tasks.tasks[0].kwargs == {}
    assert background_tasks.tasks[0].args[:3] == (
        lease,
        training_routes.run_training_task,
        "task-sync",
    )
    assert background_tasks.tasks[0].args[3]["output_dir"] == resolved_output
    assert (
        background_tasks.tasks[0].args[3]["resume_from_checkpoint"]
        == resolved_checkpoint
    )


def test_background_resume_rejects_checkpoint_outside_task_output_before_gpu(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    task_output = settings.get_task_output_dir("task-1")
    task_output.mkdir(parents=True)
    foreign_checkpoint = settings.output_dir / "foreign" / "checkpoint-1"
    foreign_checkpoint.mkdir(parents=True)
    allocation_calls = []
    status_updates = []
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "output_dir": str(task_output),
            "status": TrainingStatus.PREPARING.value,
            "run_token": "outside-checkpoint-run",
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_process_info",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        lambda *args, **kwargs: allocation_calls.append((args, kwargs)) or "cpu",
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda *args, **kwargs: True,
    )

    training_routes.run_training_task(
        "task-1",
        {
            "_run_token": "outside-checkpoint-run",
            "user_id": "user-1",
            "output_dir": str(task_output),
            "resume_from_checkpoint": str(foreign_checkpoint),
        },
    )

    assert allocation_calls == []
    assert any(
        args[1] == "failed" and "outside the training task output" in args[2]
        for args, _kwargs in status_updates
        if len(args) >= 3
    )


def test_delete_accepts_only_verified_sync_training_output(monkeypatch, tmp_path):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    sync_output = (
        settings.output_dir
        / "sync"
        / "sync-config"
        / "target-1"
        / "round_2"
    )
    sync_output.mkdir(parents=True)
    (sync_output / "adapter.bin").write_bytes(b"adapter")
    unverified_output = settings.output_dir / "sync" / "other" / "target-1" / "round_2"
    unverified_output.mkdir(parents=True)
    (unverified_output / "keep.bin").write_bytes(b"keep")
    task = {
        "task_id": "task-sync",
        "user_id": "user-1",
        "output_dir": str(sync_output),
    }
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda task_id: {
            "training_task_id": task_id,
            "task_id": "sync-config",
            "target_id": "target-1",
            "training_round": 2,
            "user_id": "user-1",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        training_routes._safe_delete_path(str(unverified_output), task)
    assert exc_info.value.status_code == 400
    assert unverified_output.exists()

    training_routes._safe_delete_path(str(sync_output), task)
    assert not sync_output.exists()


def test_training_metrics_reads_standard_server_managed_output_from_params(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    task_id = "task-1"
    task_output = settings.get_task_output_dir(task_id)
    logs_dir = task_output / "logs" / "training" / task_id
    logs_dir.mkdir(parents=True)
    (logs_dir / "loss_history.jsonl").write_text(
        '{"step": 1, "loss": 0.5}\n{"step": 2, "loss": 0.25}\n',
        encoding="utf-8",
    )
    (logs_dir / "training_metrics.json").write_text(
        '{"final_train_loss": 0.25}',
        encoding="utf-8",
    )
    task = {
        "task_id": task_id,
        "user_id": "user-1",
        "output_dir": None,
        "training_params": {"output_dir": str(task_output)},
    }
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda requested_task_id: task,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task_metrics",
        lambda requested_task_id: {"current_step": 2},
    )

    response = asyncio.run(
        training_routes.get_training_metrics(task_id, 1, CURRENT_USER)
    )

    assert response.loss_history == [{"step": 2, "loss": 0.25}]
    assert response.summary == {"final_train_loss": 0.25}
    assert response.current_metrics == {"current_step": 2}
    assert response.has_data is True


def test_training_metrics_reads_authenticated_sync_output(monkeypatch, tmp_path):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    task_id = "task-sync"
    sync_output = (
        settings.output_dir
        / "sync"
        / "sync-config"
        / "target-1"
        / "round_2"
    )
    logs_dir = sync_output / "logs" / "training" / task_id
    logs_dir.mkdir(parents=True)
    (logs_dir / "loss_history.jsonl").write_text(
        '{"step": 4, "loss": 0.125}\n',
        encoding="utf-8",
    )
    task = {
        "task_id": task_id,
        "user_id": "user-1",
        "output_dir": str(sync_output),
        "training_params": {},
    }
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda requested_task_id: task,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task_metrics",
        lambda requested_task_id: None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda requested_task_id: {
            "training_task_id": requested_task_id,
            "task_id": "sync-config",
            "target_id": "target-1",
            "training_round": 2,
            "user_id": "user-1",
        },
    )

    response = asyncio.run(
        training_routes.get_training_metrics(task_id, None, CURRENT_USER)
    )

    assert response.loss_history == [{"step": 4, "loss": 0.125}]
    assert response.has_data is True


@pytest.mark.parametrize("output_kind", ["legacy", "outside"])
def test_training_metrics_rejects_unmanaged_output_before_file_read(
    monkeypatch,
    tmp_path,
    output_kind,
):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    settings = _settings(tmp_path)
    task_id = "task-1"
    configured_output = (
        settings.output_dir / "legacy-task"
        if output_kind == "legacy"
        else tmp_path / "outside-output"
    )
    logs_dir = configured_output / "logs" / "training" / task_id
    logs_dir.mkdir(parents=True)
    (logs_dir / "loss_history.jsonl").write_text(
        '{"secret": "must-not-be-read"}\n',
        encoding="utf-8",
    )
    task = {
        "task_id": task_id,
        "user_id": "user-1",
        "output_dir": None,
        "training_params": {"output_dir": str(configured_output)},
    }
    file_reads = []

    def fail_if_file_is_read(*args, **kwargs):
        file_reads.append((args, kwargs))
        raise AssertionError("unmanaged training metrics file was read")

    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda requested_task_id: task,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task_metrics",
        lambda requested_task_id: None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda requested_task_id: None,
    )
    monkeypatch.setattr("builtins.open", fail_if_file_is_read)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.get_training_metrics(task_id, None, CURRENT_USER)
        )

    assert exc_info.value.status_code == 400
    assert file_reads == []


def test_training_metrics_checks_ownership_before_resolving_output(
    monkeypatch,
):
    task = {
        "task_id": "task-1",
        "user_id": "user-2",
        "output_dir": "C:/private/output",
        "training_params": {},
    }
    resolver_calls = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda requested_task_id: task,
    )
    monkeypatch.setattr(
        training_routes,
        "_resolve_server_managed_task_output",
        lambda *args: resolver_calls.append(args),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.get_training_metrics("task-1", None, CURRENT_USER)
        )

    assert exc_info.value.status_code == 403
    assert resolver_calls == []
def test_persisted_training_dataset_paths_include_secondary_configs():
    task = TrainingTaskDB(
        train_dataset_path="/datasets/primary.jsonl",
        training_params={
            "dataset_configs": [
                {"path": "/datasets/primary.jsonl", "split": "train"},
                {"path": "s3://bucket/datasets/secondary.jsonl", "split": "eval"},
            ]
        },
    )

    assert _persisted_training_dataset_paths(task) == {
        "/datasets/primary.jsonl",
        "s3://bucket/datasets/secondary.jsonl",
    }
