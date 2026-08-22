"""Resource-boundary regressions for user-triggered evaluation jobs."""

import importlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from train_factory.api.routes.evaluation_routes import (
    MAX_EVALUATION_BATCH_SIZE,
    MAX_EVALUATION_COMBINATIONS,
    MAX_EVALUATION_DATASETS,
    MAX_EVALUATION_MODELS,
    MAX_EVALUATION_MODEL_WORKERS,
    MAX_EVALUATION_SAMPLES,
    MAX_EVALUATION_WORKERS,
    CreateEvaluationRequest,
)
from train_factory.evaluation import evaluation_runner


def _model(index: int = 0) -> dict:
    return {
        "endpoint": f"https://example.com/{index}",
        "name": f"model-{index}",
    }


def _dataset(index: int = 0) -> dict:
    return {"type": "mteb", "name": f"dataset-{index}"}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_configs", []),
        ("model_configs", [_model(i) for i in range(MAX_EVALUATION_MODELS + 1)]),
        ("dataset_configs", []),
        (
            "dataset_configs",
            [_dataset(i) for i in range(MAX_EVALUATION_DATASETS + 1)],
        ),
        ("max_samples", 0),
        ("max_samples", MAX_EVALUATION_SAMPLES + 1),
        ("batch_size", 0),
        ("batch_size", MAX_EVALUATION_BATCH_SIZE + 1),
        ("workers", 0),
        ("workers", MAX_EVALUATION_WORKERS + 1),
        ("model_workers", 0),
        ("model_workers", MAX_EVALUATION_MODEL_WORKERS + 1),
    ),
)
def test_create_evaluation_request_rejects_unbounded_resources(field, value):
    payload = {
        "model_configs": [_model()],
        "dataset_configs": [_dataset()],
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        CreateEvaluationRequest(**payload)


def test_create_evaluation_request_caps_model_dataset_cartesian_product():
    model_count = min(MAX_EVALUATION_MODELS, MAX_EVALUATION_COMBINATIONS)
    dataset_count = (MAX_EVALUATION_COMBINATIONS // model_count) + 1

    with pytest.raises(ValidationError, match="combinations"):
        CreateEvaluationRequest(
            model_configs=[_model(i) for i in range(model_count)],
            dataset_configs=[_dataset(i) for i in range(dataset_count)],
        )


def test_runner_rejects_oversized_persisted_config_before_network_or_threads(
    monkeypatch,
):
    task = {
        "task_id": "task-1",
        "user_id": "user-1",
        "status": "running",
        "model_configs": [_model(i) for i in range(MAX_EVALUATION_MODELS + 1)],
        "dataset_configs": [_dataset()],
    }
    completions = []
    fake_service = SimpleNamespace(
        get_task=lambda _task_id: task,
        complete_task=lambda *args, **kwargs: completions.append((args, kwargs)),
    )
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    monkeypatch.setattr(service_module, "evaluation_task_service", fake_service)
    monkeypatch.setattr(
        evaluation_runner,
        "validate_user_outbound_url",
        lambda *_args, **_kwargs: pytest.fail("invalid job reached outbound validation"),
    )
    monkeypatch.setattr(
        evaluation_runner,
        "ThreadPoolExecutor",
        lambda *_args, **_kwargs: pytest.fail("invalid job reached thread creation"),
    )

    evaluation_runner.run_evaluation_task(
        "task-1",
        {
            "model_configs": task["model_configs"],
            "dataset_configs": task["dataset_configs"],
        },
    )

    assert completions
    assert completions[0][1]["status"] == "failed"
    assert "model configurations" in completions[0][1]["error_message"]
