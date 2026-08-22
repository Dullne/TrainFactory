import os
import types

import pytest
import torch

from train_factory.api.routes import training_routes
from train_factory.core.device_manager import DeviceManager
from train_factory.trainers.encoder import embedding_trainer as embedding_trainer_module
from train_factory.trainers.encoder.embedding_trainer import EmbeddingTrainer
from train_factory.utils.training_metrics import extract_final_loss_metrics


def test_explicit_cpu_device_wins_when_cuda_is_available(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert DeviceManager.get_training_device("cpu") == "cpu"


@pytest.mark.parametrize("requested_device", ["cuda:0", "auto"])
def test_gpu_device_requests_still_use_available_cuda(monkeypatch, requested_device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(total_memory=8 * 1024**3),
    )

    assert DeviceManager.get_training_device(requested_device) == "cuda"


def test_cpu_training_process_starts_with_cuda_hidden(monkeypatch):
    observed = {}

    class RecordingProcess:
        def start(self):
            observed["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cuda_visible = training_routes._device_to_cuda_visible("cpu")

    training_routes._start_training_process(
        RecordingProcess(),
        "task-cpu",
        cuda_visible,
    )

    assert cuda_visible == ""
    assert observed["cuda_visible"] == ""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_embedding_bf16_fallback_precedes_model_load_and_matches_args(
    monkeypatch,
    tmp_path,
):
    captured = {}
    fake_model = types.SimpleNamespace(max_seq_length=32768)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {
        "bf16": True,
        "fp16": False,
        "device": "cuda:0",
    }
    trainer.device_manager = types.SimpleNamespace(
        check_gpu_bf16_support=lambda: False,
    )
    trainer.config_builder = types.SimpleNamespace(
        build_training_config=lambda model_type: dict(trainer.raw_config),
    )
    trainer.model_type = "embedding"

    monkeypatch.setattr(
        trainer,
        "_requires_text_config_hidden_size_fallback",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(trainer, "_apply_lora_if_configured", lambda model: model)
    monkeypatch.setattr(
        trainer,
        "create_training_args",
        lambda config: dict(config),
    )

    def fake_sentence_transformer(model_name_or_path, **kwargs):
        captured["model_kwargs"] = kwargs.get("model_kwargs")
        return fake_model

    monkeypatch.setattr(
        embedding_trainer_module,
        "SentenceTransformer",
        fake_sentence_transformer,
    )

    trainer.initialize_model(str(model_dir))
    training_args = trainer._create_training_args()

    assert captured["model_kwargs"]["dtype"] is torch.float16
    assert training_args["bf16"] is False
    assert training_args["fp16"] is True


def test_extract_final_loss_metrics_accepts_hugging_face_train_loss():
    history = [
        {"loss": 0.9, "step": 10},
        {"eval_loss": 0.8, "step": 10},
        {"train_loss": 0.4, "train_runtime": 12.0},
    ]

    assert extract_final_loss_metrics(history) == {
        "final_train_loss": 0.4,
        "final_eval_loss": 0.8,
    }
