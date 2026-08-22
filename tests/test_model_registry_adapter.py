from contextlib import contextmanager
import importlib

import pytest

from train_factory.api.routes.registry_routes import (
    ModelResponse,
    RegisterModelRequest,
    _model_to_response,
)
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.storage.services.model_registry_service import ModelRegistryService


registry_service_module = importlib.import_module(
    "train_factory.storage.services.model_registry_service"
)
training_task_service_module = importlib.import_module(
    "train_factory.storage.services.training_task_service"
)


class _EmptyResult:
    def all(self):
        return []


class _RecordingSession:
    def __init__(self):
        self.added = []

    def exec(self, _statement):
        return _EmptyResult()

    def add(self, value):
        self.added.append(value)

    def flush(self):
        return None

    def refresh(self, _value):
        return None

    def commit(self):
        return None


@pytest.mark.parametrize("is_adapter", [False, True])
def test_register_model_persists_and_exposes_adapter_flag(
    monkeypatch, tmp_path, is_adapter
):
    session = _RecordingSession()

    @contextmanager
    def fake_get_session():
        yield session

    monkeypatch.setattr(registry_service_module, "get_session", fake_get_session)

    kwargs = {"is_adapter": is_adapter} if is_adapter else {}
    model = ModelRegistryService().register_model(
        model_name="adapter-model",
        model_path=str(tmp_path),
        model_type="embedding",
        **kwargs,
    )

    persisted = next(item for item in session.added if isinstance(item, ModelRegistryDB))
    assert persisted.is_adapter is is_adapter
    assert model["is_adapter"] is is_adapter
    assert _model_to_response(model).is_adapter is is_adapter


@pytest.mark.parametrize(
    ("task_is_lora", "params_use_lora", "expected"),
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
        (None, None, False),
    ],
)
def test_register_from_task_derives_adapter_flag(
    monkeypatch, task_is_lora, params_use_lora, expected
):
    task = {
        "task_id": "task-1",
        "task_name": "trained-model",
        "status": "succeeded",
        "final_model_path": "/models/final",
        "model_type": "embedding",
        "base_model_path": "/models/base",
        "description": None,
        "final_metrics": {"loss": 0.1},
        "user_id": "user-1",
        "is_lora": task_is_lora,
        "training_params": {"use_lora": params_use_lora},
    }
    captured = {}
    service = ModelRegistryService()
    monkeypatch.setattr(service, "get_model_by_source_task", lambda *args, **kwargs: None)

    def fake_register_model(**kwargs):
        captured.update(kwargs)
        return {"model_id": "model-1"}

    monkeypatch.setattr(service, "register_model", fake_register_model)
    monkeypatch.setattr(
        training_task_service_module.training_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        training_task_service_module.training_task_service,
        "update_task_trained_model_registry_id",
        lambda *_args: True,
    )

    service.register_from_task("task-1", "trained-model", user_id="user-1")

    assert captured["is_adapter"] is expected


def test_registry_api_adapter_fields_are_backward_compatible():
    assert RegisterModelRequest(
        model_name="model",
        model_path="/models/model",
        model_type="embedding",
    ).is_adapter is False
    assert ModelResponse.model_fields["is_adapter"].default is False
