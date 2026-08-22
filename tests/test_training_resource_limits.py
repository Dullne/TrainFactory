import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from train_factory.api.routes import training_routes
from train_factory.enums import TrainingStatus
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskCapacityExceeded,
)


def _request_data():
    return {
        "base_model_path": "/app/models/model-1",
        "datasets": [{"path": "/app/data/train.jsonl", "split": "train"}],
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"num_train_epochs": 0},
        {"num_train_epochs": 1001},
        {"per_device_train_batch_size": 0},
        {"per_device_train_batch_size": 4097},
        {"learning_rate": float("inf")},
        {"warmup_ratio": 1.1},
        {"gradient_accumulation_steps": 0},
        {"logging_steps": 0},
        {"max_length": 131073},
        {"lora_r": 0},
        {"lora_alpha": 65537},
        {"lora_dropout": 1.1},
        {"gpu_ids": list(range(17))},
        {"gpu_ids": [-1]},
        {
            "datasets": [
                {"path": f"/app/data/{index}.jsonl", "split": "train"}
                for index in range(33)
            ]
        },
        {
            "datasets": [
                {
                    "path": "/app/data/train.jsonl",
                    "split": "train",
                    "max_samples": 10_000_001,
                }
            ]
        },
        {"loss_config": {"ranknet_max_pairs_per_batch": 2_000_001}},
        {"rl_config": {"num_iterations": 101}},
        {"rl_config": {"payload": "x" * 65_537}},
    ],
)
def test_training_request_rejects_excessive_resource_values(updates):
    data = _request_data()
    data.update(deepcopy(updates))

    with pytest.raises(ValidationError):
        training_routes.TrainingRequest(**data)


def test_worker_revalidates_stored_limits_before_gpu_allocation(monkeypatch):
    allocations = []
    status_updates = []
    task = {
        "task_id": "oversized-training",
        "user_id": "user-1",
        "status": TrainingStatus.PREPARING.value,
        "run_token": "resource-limit-run",
    }

    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        lambda *args, **kwargs: allocations.append((args, kwargs)),
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda _task_id: True,
    )

    training_routes.run_training_task(
        "oversized-training",
        {
            "_run_token": "resource-limit-run",
            "task_id": "oversized-training",
            "user_id": "user-1",
            "num_train_epochs": 1_000_000,
        },
    )

    assert allocations == []
    assert status_updates
    assert status_updates[-1][0][1] == TrainingStatus.FAILED.value
    assert "num_train_epochs" in status_updates[-1][0][2]


def test_resume_rejects_stored_limits_before_state_reset(monkeypatch):
    task = {
        "task_id": "oversized-resume",
        "user_id": "user-1",
        "status": TrainingStatus.FAILED.value,
        "base_model_path": "/app/models/model-1",
        "train_dataset_path": "/app/data/train.jsonl",
        "training_params": {
            "base_model_path": "/app/models/model-1",
            "train_dataset_path": "/app/data/train.jsonl",
            "num_train_epochs": 1_000_000,
        },
    }
    checkpoint_lookups = []
    resets = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "find_latest_checkpoint",
        lambda task_id: checkpoint_lookups.append(task_id),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "reset_for_resume",
        lambda task_id, run_token: resets.append((task_id, run_token)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.resume_task(
                "oversized-resume",
                BackgroundTasks(),
                {"user_id": "user-1", "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 400
    assert "num_train_epochs" in exc_info.value.detail
    assert checkpoint_lookups == []
    assert resets == []


def test_resume_maps_training_capacity_to_429_before_state_reset(monkeypatch):
    task = {
        "task_id": "capacity-resume",
        "user_id": "user-1",
        "status": TrainingStatus.FAILED.value,
        "base_model_path": "/app/models/model-1",
        "train_dataset_path": "/app/data/train.jsonl",
        "output_dir": "/app/output/capacity-resume",
        "training_params": {
            "base_model_path": "/app/models/model-1",
            "train_dataset_path": "/app/data/train.jsonl",
        },
    }
    resets = []
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "find_latest_checkpoint",
        lambda _task_id: "/app/output/capacity-resume/checkpoint-1",
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "reset_for_resume",
        lambda task_id, run_token: resets.append((task_id, run_token)) or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        training_routes,
        "_normalize_training_config_paths",
        lambda config, _user_id: config,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_training_parent_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_resolve_server_managed_task_output",
        lambda *_args, **_kwargs: "/app/output/capacity-resume",
    )
    monkeypatch.setattr(
        training_routes,
        "_resolve_resume_checkpoint",
        lambda *_args, **_kwargs: "/app/output/capacity-resume/checkpoint-1",
    )
    monkeypatch.setattr(
        training_routes,
        "background_task_admission_service",
        SimpleNamespace(
            admit_execution=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                BackgroundTaskCapacityExceeded("per-user active task limit exceeded")
            )
        ),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.resume_task(
                "capacity-resume",
                BackgroundTasks(),
                {"user_id": "user-1", "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}
    assert resets == []


def test_worker_does_not_fallback_to_cpu_without_operator_opt_in(monkeypatch):
    allocations = []
    status_updates = []
    task = {
        "task_id": "gpu-required-training",
        "user_id": "user-1",
        "status": TrainingStatus.PREPARING.value,
        "run_token": "gpu-required-run",
    }

    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        training_routes,
        "_normalize_training_config_paths",
        lambda config, _user_id: config,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_training_parent_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "get_settings",
        lambda: SimpleNamespace(training_allow_cpu_fallback=False),
    )

    def allocate(_lease_id, requested_device):
        allocations.append(requested_device)
        if requested_device == "cpu":
            pytest.fail("CPU fallback ran without explicit operator opt-in")
        return None

    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        allocate,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda _lease_id: True,
    )

    training_routes.run_training_task(
        "gpu-required-training",
        {
            "_run_token": "gpu-required-run",
            "task_id": "gpu-required-training",
            "user_id": "user-1",
            "base_model_path": "/app/models/model-1",
            "train_dataset_path": "/app/data/train.jsonl",
        },
    )

    assert allocations == ["auto"]
    assert status_updates[-1][0][1] == TrainingStatus.FAILED.value
    assert "GPU allocation failed" in status_updates[-1][0][2]


def test_worker_falls_back_to_cpu_with_operator_opt_in(monkeypatch):
    allocations = []
    persisted_devices = []
    status_updates = []
    task = {
        "task_id": "cpu-fallback-training",
        "user_id": "user-1",
        "status": TrainingStatus.PREPARING.value,
        "run_token": "cpu-fallback-run",
    }

    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_execution_config",
        lambda _task_id, **kwargs: persisted_devices.append(kwargs["device"])
        or False,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_process_info",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        training_routes,
        "_normalize_training_config_paths",
        lambda config, _user_id: config,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_owned_training_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_validate_training_parent_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "get_settings",
        lambda: SimpleNamespace(training_allow_cpu_fallback=True),
    )

    def allocate(_lease_id, requested_device):
        allocations.append(requested_device)
        return "cpu" if requested_device == "cpu" else None

    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        allocate,
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "release_gpus_for_task",
        lambda _lease_id: True,
    )

    training_routes.run_training_task(
        "cpu-fallback-training",
        {
            "_run_token": "cpu-fallback-run",
            "task_id": "cpu-fallback-training",
            "user_id": "user-1",
            "base_model_path": "/app/models/model-1",
            "train_dataset_path": "/app/data/train.jsonl",
        },
    )

    assert allocations == ["auto", "cpu"]
    assert persisted_devices == ["cpu"]
    assert status_updates == []
