import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import deep_evaluation_routes, generation_routes
from train_factory.deep_evaluation import deep_evaluation_runner
from train_factory.generation.pipeline import PipelineConfig
from train_factory.storage.entities.generation_task_entity import GenerationStatus


CURRENT_USER = {"user_id": "user-1", "username": "alice"}


class _GenerationTaskService:
    def __init__(self, task):
        self.task = task
        self.status_updates = []
        self.reset_calls = []

    def get_task(self, _task_id):
        return self.task

    def get_task_raw(self, _task_id):
        return self.task

    def update_status(
        self,
        task_id,
        status,
        error_message=None,
        *,
        expected_run_token=None,
    ):
        self.status_updates.append((task_id, status, error_message))
        return True

    def reset_progress(self, task_id):
        self.reset_calls.append(task_id)


def _generation_task(status=GenerationStatus.FAILED):
    return {
        "task_id": "generation-task-1",
        "status": status,
        "user_id": "user-1",
        "source_dataset_id": "dataset-1",
        "input_path": "/app/data/datasets/dataset-1/train.jsonl",
        "milvus_collection": "shared-name",
        "llm_config": None,
        "run_token": "generation-attempt-1",
    }


def test_generation_restart_rejects_collection_rebound_to_another_tenant(
    monkeypatch,
):
    task = _generation_task()
    service = _GenerationTaskService(task)
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )
    monkeypatch.setattr(
        generation_routes,
        "resolve_managed_local_dataset",
        lambda *_args, **_kwargs: {"storage_path": task["input_path"]},
    )
    monkeypatch.setattr(
        generation_routes.milvus_collection_service,
        "get_by_name",
        lambda _name: {
            "collection_name": "shared-name",
            "user_id": "user-2",
        },
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.restart_task(
                task["task_id"],
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert service.status_updates == []
    assert service.reset_calls == []
    assert background_tasks.tasks == []


def test_generation_worker_rejects_collection_rebound_before_pipeline(
    monkeypatch,
):
    task = _generation_task(status=GenerationStatus.PENDING)
    service = _GenerationTaskService(task)
    pipeline_constructions = []

    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "get_settings",
        lambda: SimpleNamespace(auth_enabled=True),
    )
    monkeypatch.setattr(
        generation_routes,
        "resolve_managed_local_dataset",
        lambda *_args, **_kwargs: {"storage_path": task["input_path"]},
    )
    monkeypatch.setattr(
        generation_routes.milvus_collection_service,
        "get_by_name",
        lambda _name: {
            "collection_name": "shared-name",
            "user_id": "user-2",
        },
    )

    class ForbiddenPipeline:
        def __init__(self, *args, **kwargs):
            pipeline_constructions.append((args, kwargs))
            raise AssertionError("pipeline must not use another tenant's collection")

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        ForbiddenPipeline,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task["task_id"],
            PipelineConfig(
                input_path=task["input_path"],
                existing_collection_name="shared-name",
            ),
        )
    )

    assert pipeline_constructions == []
    assert service.status_updates[-1][1] == GenerationStatus.FAILED


class _DeepTaskService:
    def __init__(self, task):
        self.task = task
        self.reset_calls = []
        self.status_updates = []

    def get_task(self, _task_id, **_kwargs):
        return self.task

    def reset_for_resume(self, task_id):
        self.reset_calls.append(task_id)

    def update_status(self, *args):
        self.status_updates.append(args)

    def complete_task(self, task_id, status, **kwargs):
        self.status_updates.append((task_id, status, kwargs.get("error_message")))
        return True


def _deep_task(status="cancelled"):
    return {
        "task_id": "deep-task-1",
        "status": status,
        "user_id": "user-1",
        "dataset_configs": [{"dataset_id": "dataset-1"}],
        "model_configs": [{"group_name": "group-1", "embedding": {}}],
        "metrics": ["mrr"],
        "results_summary": {},
        "model_progress": {},
        "worker_groups": {
            "retrieval_mode": "online",
            "milvus_collection": "shared-name",
            "retrieval_embedding_config": {
                "endpoint": "https://example.com/v1",
                "model_name": "embedding-model",
            },
        },
    }


def test_deep_evaluation_resume_rejects_collection_rebound_to_another_tenant(
    monkeypatch,
):
    task = _deep_task()
    service = _DeepTaskService(task)
    monkeypatch.setattr(
        deep_evaluation_routes,
        "deep_evaluation_task_service",
        service,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: [{"dataset_id": "dataset-1"}],
    )
    monkeypatch.setattr(
        deep_evaluation_routes.milvus_collection_service,
        "get_by_name",
        lambda _name: {
            "collection_name": "shared-name",
            "user_id": "user-2",
        },
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.resume_deep_evaluation_task(
                task["task_id"],
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert service.reset_calls == []
    assert background_tasks.tasks == []


def test_deep_evaluation_resume_rejects_collection_being_deleted(
    monkeypatch,
):
    task = _deep_task()
    service = _DeepTaskService(task)
    monkeypatch.setattr(
        deep_evaluation_routes,
        "deep_evaluation_task_service",
        service,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: [{"dataset_id": "dataset-1"}],
    )
    monkeypatch.setattr(
        deep_evaluation_routes.milvus_collection_service,
        "get_by_name",
        lambda _name: {
            "collection_name": "shared-name",
            "user_id": "user-1",
            "status": "deleting",
        },
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: pytest.fail(
            "a deleting collection must be rejected before task admission"
        ),
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.resume_deep_evaluation_task(
                task["task_id"],
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert service.reset_calls == []
    assert background_tasks.tasks == []


def test_deep_evaluation_worker_rejects_collection_rebound_before_dataset_read(
    monkeypatch,
):
    task = _deep_task(status="pending")
    service = _DeepTaskService(task)
    dataset_reads = []

    monkeypatch.setattr(
        deep_evaluation_runner,
        "deep_evaluation_task_service",
        service,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "milvus_collection_service",
        SimpleNamespace(
            get_by_name=lambda _name: {
                "collection_name": "shared-name",
                "user_id": "user-2",
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_resolve_task_dataset_records",
        lambda *_args, **_kwargs: dataset_reads.append("read") or [],
    )

    deep_evaluation_runner.run_deep_evaluation_task(task["task_id"])

    assert dataset_reads == []
    assert service.status_updates[-1][1] == "failed"
