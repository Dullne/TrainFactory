"""Resource and restart-claim regressions for background API jobs."""

from __future__ import annotations

import asyncio
import importlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError
from sqlmodel import Session, SQLModel, create_engine

from train_factory.api.routes import deep_evaluation_routes, generation_routes
from train_factory.deep_evaluation import deep_evaluation_runner
from train_factory.generation.pipeline import PipelineConfig
from train_factory.storage.entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationStatus,
    EvaluationTaskDB,
)
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)


DEEP_MAX_GROUPS = 16
DEEP_MAX_DATASETS = 32
DEEP_MAX_COMBINATIONS = 128
DEEP_MAX_SAMPLES = 1_000_000
DEEP_MAX_METRICS = 16
DEEP_MAX_MODEL_CONCURRENCY = 64
DEEP_MAX_LLM_CONCURRENCY = 32
DEEP_MAX_MODEL_WORKERS = 16

GEN_MAX_ENDPOINTS = 16
GEN_MAX_LLM_CONCURRENCY = 50
GEN_MAX_EMBEDDING_CONCURRENCY = 100
GEN_MAX_RERANK_CONCURRENCY = 50
GEN_MAX_BATCH_SIZE = 256
GEN_MAX_TOKENS = 131_072
GEN_MAX_TIMEOUT = 600
GEN_MAX_RETRIES = 10


@pytest.fixture(autouse=True)
def _isolate_route_admission(monkeypatch):
    lease = SimpleNamespace(release=lambda: None)

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        result = operation(*args, **kwargs)
        return result, lease if result is not False and result is not None else None

    monkeypatch.setattr(
        generation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )


def _deep_payload() -> dict:
    return {
        "model_configs": [
            {
                "group_name": "group-1",
                "embedding": {"config_id": "embedding-1"},
            }
        ],
        "dataset_configs": [{"dataset_id": "dataset-1"}],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_configs", []),
        (
            "model_configs",
            [
                {
                    "group_name": f"group-{index}",
                    "embedding": {"config_id": "embedding-1"},
                }
                for index in range(DEEP_MAX_GROUPS + 1)
            ],
        ),
        ("dataset_configs", []),
        (
            "dataset_configs",
            [
                {"dataset_id": f"dataset-{index}"}
                for index in range(DEEP_MAX_DATASETS + 1)
            ],
        ),
        ("max_samples", 0),
        ("max_samples", DEEP_MAX_SAMPLES + 1),
        ("model_workers", DEEP_MAX_MODEL_WORKERS + 1),
        ("metrics", [f"metric-{index}" for index in range(DEEP_MAX_METRICS + 1)]),
    ),
)
def test_deep_request_rejects_unbounded_top_level_resources(field, value):
    payload = _deep_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        deep_evaluation_routes.CreateDeepEvaluationTaskRequest(**payload)


def test_deep_request_caps_model_dataset_cartesian_product():
    payload = _deep_payload()
    payload["model_configs"] = [
        {
            "group_name": f"group-{index}",
            "embedding": {"config_id": "embedding-1"},
        }
        for index in range(DEEP_MAX_GROUPS)
    ]
    payload["dataset_configs"] = [
        {"dataset_id": f"dataset-{index}"}
        for index in range((DEEP_MAX_COMBINATIONS // DEEP_MAX_GROUPS) + 1)
    ]

    with pytest.raises(ValidationError, match="combinations"):
        deep_evaluation_routes.CreateDeepEvaluationTaskRequest(**payload)


@pytest.mark.parametrize(
    ("model", "field", "value"),
    (
        (deep_evaluation_routes.ModelConfigModel, "concurrency", DEEP_MAX_MODEL_CONCURRENCY + 1),
        (deep_evaluation_routes.LLMConfigModel, "concurrency", DEEP_MAX_LLM_CONCURRENCY + 1),
        (deep_evaluation_routes.LLMConfigModel, "max_tokens", GEN_MAX_TOKENS + 1),
        (deep_evaluation_routes.LLMConfigModel, "timeout", GEN_MAX_TIMEOUT + 1),
        (deep_evaluation_routes.LLMConfigModel, "max_retries", GEN_MAX_RETRIES + 1),
        (deep_evaluation_routes.LLMConfigModel, "temperature", 2.1),
        (deep_evaluation_routes.LLMConfigModel, "top_p", 1.1),
        (deep_evaluation_routes.LLMConfigModel, "top_k", 201),
    ),
)
def test_deep_request_rejects_unbounded_model_resources(model, field, value):
    with pytest.raises(ValidationError):
        model(**{field: value})


@pytest.mark.parametrize(
    ("model", "field", "value"),
    (
        (generation_routes.LLMConfigModel, "concurrency", GEN_MAX_LLM_CONCURRENCY + 1),
        (generation_routes.LLMConfigModel, "max_tokens", GEN_MAX_TOKENS + 1),
        (generation_routes.LLMConfigModel, "timeout", GEN_MAX_TIMEOUT + 1),
        (generation_routes.LLMConfigModel, "max_retries", GEN_MAX_RETRIES + 1),
        (
            generation_routes.LLMConfigModel,
            "endpoints",
            [
                {"url": "https://example.com/v1", "model": f"model-{index}"}
                for index in range(GEN_MAX_ENDPOINTS + 1)
            ],
        ),
        (
            generation_routes.EmbeddingConfigModel,
            "concurrency",
            GEN_MAX_EMBEDDING_CONCURRENCY + 1,
        ),
        (generation_routes.EmbeddingConfigModel, "batch_size", GEN_MAX_BATCH_SIZE + 1),
        (
            generation_routes.RerankConfigModel,
            "concurrency",
            GEN_MAX_RERANK_CONCURRENCY + 1,
        ),
        (generation_routes.RerankConfigModel, "batch_size", GEN_MAX_BATCH_SIZE + 1),
        (generation_routes.WorkerConfigModel, "timeout_per_doc", GEN_MAX_TIMEOUT + 1),
    ),
)
def test_generation_request_rejects_unbounded_model_resources(model, field, value):
    with pytest.raises(ValidationError):
        model(**{field: value})


@pytest.mark.parametrize(
    ("step", "field", "value"),
    (
        ("qa_gen", "num_qa_per_doc", 11),
        ("qa_gen", "roles_per_doc", 11),
        ("role_gen", "roles_per_doc", 11),
        ("keypoint_gen", "max_keypoints", 11),
        ("pos_neg_extraction", "num_positive", 21),
        ("pos_neg_extraction", "num_negative", 51),
        ("pos_neg_extraction", "roles_per_doc", 11),
    ),
)
def test_generation_request_rejects_unbounded_step_fanout(step, field, value):
    with pytest.raises(ValidationError):
        generation_routes.StepConfigModel(**{step: {field: value}})


def test_deep_worker_rejects_oversized_persisted_config_before_resource_access(
    monkeypatch,
):
    task = {
        "task_id": "deep-limit-task",
        "status": EvaluationStatus.PENDING,
        "user_id": "user-1",
        "model_configs": [
            {
                "group_name": "group-1",
                "embedding": {
                    "endpoint": "https://example.com/v1",
                    "model_name": "embedding",
                    "concurrency": DEEP_MAX_MODEL_CONCURRENCY + 1,
                },
            }
        ],
        "dataset_configs": [{"dataset_id": "dataset-1"}],
        "metrics": ["mrr"],
        "worker_groups": {"retrieval_mode": "offline", "model_workers": 1},
    }
    updates = []
    service = SimpleNamespace(
        get_task=lambda _task_id, **_kwargs: task,
        update_status=lambda *args: updates.append(args) or True,
        complete_task=lambda task_id, status, **kwargs: updates.append(
            (task_id, status, kwargs.get("error_message"))
        )
        or True,
    )
    monkeypatch.setattr(deep_evaluation_runner, "deep_evaluation_task_service", service)
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_require_task_collection_ownership",
        lambda *_args: pytest.fail("invalid task reached collection access"),
    )

    deep_evaluation_runner.run_deep_evaluation_task(task["task_id"])

    assert updates
    assert updates[-1][1] == EvaluationStatus.FAILED
    assert "concurrency" in updates[-1][2]


def test_generation_worker_rejects_oversized_persisted_config_before_resource_access(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text":"hello"}\n', encoding="utf-8")
    task = {
        "task_id": "generation-limit-task",
        "status": GenerationStatus.PENDING,
        "user_id": "user-1",
    }
    run_token = "generation-limit-run-token"
    updates = []

    def get_task_raw(_task_id):
        return {**task, "run_token": run_token}

    def claim_running(_task_id, *, expected_run_token=None):
        if (
            expected_run_token != run_token
            or task["status"] != GenerationStatus.PENDING
        ):
            return None
        task["status"] = GenerationStatus.RUNNING
        return run_token

    def update_status(
        task_id,
        status,
        error_message=None,
        *,
        expected_run_token=None,
        **_kwargs,
    ):
        assert expected_run_token == run_token
        updates.append((task_id, status, error_message))
        task["status"] = status
        return True

    service = SimpleNamespace(
        get_task=lambda _task_id: dict(task),
        get_task_raw=get_task_raw,
        claim_running=claim_running,
        update_status=update_status,
    )
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args: pytest.fail("invalid task reached collection access"),
    )
    monkeypatch.setitem(
        sys.modules,
        "train_factory.sync.level2_handler",
        SimpleNamespace(on_generation_failed=lambda *_args, **_kwargs: None),
    )
    config = PipelineConfig(
        input_path=str(input_path),
        llm_config={
            "endpoint": "https://example.com/v1",
            "model": "llm",
            "concurrency": GEN_MAX_LLM_CONCURRENCY + 1,
        },
        llm_concurrency=GEN_MAX_LLM_CONCURRENCY + 1,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task["task_id"],
            config,
            expected_run_token=run_token,
        )
    )

    assert updates
    assert updates[-1][1] == GenerationStatus.FAILED
    assert "concurrency" in updates[-1][2]


def _generation_restart_task(tmp_path) -> dict:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text":"hello"}\n', encoding="utf-8")
    return {
        "task_id": "generation-restart-task",
        "status": GenerationStatus.FAILED,
        "generation_mode": "qa_extraction",
        "input_path": str(input_path),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(tmp_path / "output.jsonl"),
        "pos_neg_method": "retrieval",
        "llm_config": {
            "endpoint": "https://example.com/v1",
            "model": "llm",
            "concurrency": 1,
        },
        "worker_config": {"timeout_per_doc": 60},
        "steps_config": {"qa_gen": {"enabled": True, "num_qa_per_doc": 1}},
        "post_process_config": {},
        "user_id": "user-1",
    }


def test_generation_restart_returns_conflict_when_terminal_status_claim_is_lost(
    tmp_path,
    monkeypatch,
):
    task = _generation_restart_task(tmp_path)
    run_token = "generation-restart-run-token"
    resets = []

    def reject_update_status(
        _task_id,
        _status,
        _error_message=None,
        *,
        expected_run_token=None,
        **_kwargs,
    ):
        assert expected_run_token == run_token
        return False

    service = SimpleNamespace(
        get_task=lambda _task_id: dict(task),
        get_task_raw=lambda _task_id: {**task, "run_token": run_token},
        claim_running=lambda _task_id, *, expected_run_token=None: (
            run_token if expected_run_token == run_token else None
        ),
        begin_restart=lambda _task_id, **_kwargs: None,
        update_status=reject_update_status,
        reset_progress=lambda task_id: resets.append(task_id),
    )
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: task,
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda _task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "has_staging_products",
        lambda _task_id, *, expected_run_token: False,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.restart_task(
                task["task_id"],
                background_tasks,
                {"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 409
    assert resets == []
    assert background_tasks.tasks == []


def test_deep_resume_returns_conflict_when_terminal_status_claim_is_lost(monkeypatch):
    task = {
        "task_id": "deep-resume-task",
        "status": EvaluationStatus.CANCELLED,
        "user_id": "user-1",
        "model_configs": [
            {
                "group_name": "group-1",
                "embedding": {
                    "endpoint": "https://example.com/v1",
                    "model_name": "embedding",
                    "concurrency": 1,
                },
            }
        ],
        "dataset_configs": [{"dataset_id": "dataset-1"}],
        "metrics": ["mrr"],
        "worker_groups": {"retrieval_mode": "offline", "model_workers": 1},
        "results_summary": {},
        "model_progress": {},
    }
    service = SimpleNamespace(
        get_task=lambda _task_id: task,
        reset_for_resume=lambda _task_id, **_kwargs: False,
    )
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(deep_evaluation_routes, "deep_evaluation_task_service", service)
    monkeypatch.setattr(
        deep_evaluation_routes,
        "_resolve_deep_evaluation_dataset_configs",
        lambda *_args, **_kwargs: task["dataset_configs"],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.resume_deep_evaluation_task(
                task["task_id"],
                background_tasks,
                {"user_id": "user-1"},
            )
        )

    assert exc_info.value.status_code == 409
    assert background_tasks.tasks == []


def test_generation_terminal_to_pending_transition_is_compare_and_set(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(f"sqlite:///{tmp_path / 'generation-cas.db'}")
    SQLModel.metadata.create_all(engine)
    task_id = "generation-cas-task"
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=task_id,
                task_name="CAS",
                input_path="/tmp/input.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.FAILED,
            )
        )
        session.commit()

    service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    service = service_module.GenerationTaskService()

    assert service.update_status(task_id, GenerationStatus.PENDING) is True
    assert service.update_status(task_id, GenerationStatus.PENDING) is False


def test_deep_resume_reset_is_compare_and_set(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'deep-cas.db'}")
    SQLModel.metadata.create_all(engine)
    task_id = "deep-cas-task"
    with Session(engine) as session:
        session.add(
            EvaluationTaskDB(
                task_id=task_id,
                task_name="CAS",
                eval_framework=EvaluationFramework.DEEPEVAL,
                eval_type="embedding",
                model_configs=[],
                dataset_configs=[],
                status=EvaluationStatus.CANCELLED,
                model_progress={},
            )
        )
        session.commit()

    service_module = importlib.import_module(
        "train_factory.storage.services.deep_evaluation_task_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    service = service_module.DeepEvaluationTaskService()

    assert service.reset_for_resume(task_id) is True
    assert service.reset_for_resume(task_id) is False
