import sys
import types

import pytest
import torch

from train_factory.api.routes.training_routes import _apply_max_length_to_nested_configs
from train_factory.schemas.training_config import TrainingParametersManager
from train_factory.trainers.encoder import embedding_trainer as embedding_trainer_module
from train_factory.trainers.encoder.embedding_trainer import EmbeddingTrainer


def test_embedding_max_length_maps_to_canonical_config_field():
    normalized = _apply_max_length_to_nested_configs(
        {
            "model_type": "embedding",
            "training_method": "sft",
            "max_length": 128,
        }
    )

    assert normalized["max_length"] == 128
    assert normalized["max_seq_length"] == 128
    custom_params = (
        TrainingParametersManager()
        .load_from_config(normalized)
        .get_custom_params_dict()
    )
    assert custom_params["max_seq_length"] == 128


def test_embedding_max_length_does_not_override_explicit_max_seq_length():
    normalized = _apply_max_length_to_nested_configs(
        {
            "model_type": "embedding",
            "training_method": "sft",
            "max_length": 128,
            "max_seq_length": 256,
        }
    )

    assert normalized["max_seq_length"] == 256


def test_initialize_model_applies_requested_max_seq_length(monkeypatch):
    fake_model = types.SimpleNamespace(max_seq_length=32768)
    fake_modelscope = types.ModuleType("modelscope")
    fake_modelscope.snapshot_download = lambda model_name, cache_dir: "/cached/model"
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {"max_seq_length": 128}
    monkeypatch.setattr(trainer, "_load_sentence_transformer", lambda path: fake_model)
    monkeypatch.setattr(trainer, "_apply_lora_if_configured", lambda model: model)

    model = trainer.initialize_model("Qwen/Qwen3-Embedding-0.6B")

    assert model is fake_model
    assert model.max_seq_length == 128


def test_initialize_model_accepts_legacy_max_length(monkeypatch):
    fake_model = types.SimpleNamespace(max_seq_length=32768)
    fake_modelscope = types.ModuleType("modelscope")
    fake_modelscope.snapshot_download = lambda model_name, cache_dir: "/cached/model"
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {"max_length": 64}
    monkeypatch.setattr(trainer, "_load_sentence_transformer", lambda path: fake_model)
    monkeypatch.setattr(trainer, "_apply_lora_if_configured", lambda model: model)

    model = trainer.initialize_model("Qwen/Qwen3-Embedding-0.6B")

    assert model.max_seq_length == 64


def test_initialize_model_loads_existing_local_directory_without_modelscope(
    monkeypatch, tmp_path
):
    fake_model = types.SimpleNamespace(max_seq_length=32768)
    fake_modelscope = types.ModuleType("modelscope")

    def unexpected_snapshot_download(*args, **kwargs):
        pytest.fail("snapshot_download must not be called for an existing local directory")

    fake_modelscope.snapshot_download = unexpected_snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {}
    loaded_paths = []
    monkeypatch.setattr(
        trainer,
        "_load_sentence_transformer",
        lambda path: loaded_paths.append(path) or fake_model,
    )
    monkeypatch.setattr(trainer, "_apply_lora_if_configured", lambda model: model)

    local_model_dir = tmp_path / "qwen3-embedding"
    local_model_dir.mkdir()

    model = trainer.initialize_model(str(local_model_dir))

    assert model is fake_model
    assert loaded_paths == [str(local_model_dir)]


def test_initialize_model_keeps_modelscope_flow_for_remote_repo(monkeypatch):
    fake_model = types.SimpleNamespace(max_seq_length=32768)
    fake_modelscope = types.ModuleType("modelscope")
    download_calls = []

    def snapshot_download(model_name, cache_dir):
        download_calls.append((model_name, cache_dir))
        return "/cached/model"

    fake_modelscope.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {}
    loaded_paths = []
    monkeypatch.setattr(
        trainer,
        "_load_sentence_transformer",
        lambda path: loaded_paths.append(path) or fake_model,
    )
    monkeypatch.setattr(trainer, "_apply_lora_if_configured", lambda model: model)

    model = trainer.initialize_model("Qwen/Qwen3-Embedding-0.6B")

    assert model is fake_model
    assert len(download_calls) == 1
    assert download_calls[0][0] == "Qwen/Qwen3-Embedding-0.6B"
    assert loaded_paths == ["/cached/model"]


@pytest.mark.parametrize(
    ("precision_config", "expected_dtype"),
    [
        ({"bf16": True}, torch.bfloat16),
        ({"fp16": True}, torch.float16),
    ],
)
def test_sentence_transformer_load_uses_requested_mixed_precision(
    monkeypatch, precision_config, expected_dtype
):
    captured = {}
    fake_model = object()
    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {
        **precision_config,
        "model_kwargs": {"attn_implementation": "eager"},
    }
    monkeypatch.setattr(
        trainer,
        "_requires_text_config_hidden_size_fallback",
        lambda *args, **kwargs: False,
    )

    def fake_sentence_transformer(model_name_or_path, **kwargs):
        captured["path"] = model_name_or_path
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(
        embedding_trainer_module,
        "SentenceTransformer",
        fake_sentence_transformer,
    )

    result = trainer._load_sentence_transformer("/models/qwen3")

    assert result is fake_model
    assert captured["path"] == "/models/qwen3"
    assert captured["kwargs"]["model_kwargs"] == {
        "attn_implementation": "eager",
        "dtype": expected_dtype,
    }


def test_sentence_transformer_load_preserves_explicit_dtype(monkeypatch):
    captured = {}
    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {"bf16": True}
    monkeypatch.setattr(
        trainer,
        "_requires_text_config_hidden_size_fallback",
        lambda *args, **kwargs: False,
    )

    def fake_sentence_transformer(model_name_or_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        embedding_trainer_module,
        "SentenceTransformer",
        fake_sentence_transformer,
    )

    trainer._load_sentence_transformer(
        "/models/qwen3",
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
    )

    assert captured["model_kwargs"] == {
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
    }


@pytest.mark.parametrize(
    "raw_config",
    [
        {},
        {"bf16": True, "device": "cpu"},
    ],
)
def test_sentence_transformer_load_does_not_force_dtype_without_gpu_mixed_precision(
    monkeypatch, raw_config
):
    captured = {}
    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = raw_config
    monkeypatch.setattr(
        trainer,
        "_requires_text_config_hidden_size_fallback",
        lambda *args, **kwargs: False,
    )

    def fake_sentence_transformer(model_name_or_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        embedding_trainer_module,
        "SentenceTransformer",
        fake_sentence_transformer,
    )

    trainer._load_sentence_transformer(
        "/models/qwen3",
        model_kwargs={"attn_implementation": "eager"},
    )

    assert captured["model_kwargs"] == {"attn_implementation": "eager"}
