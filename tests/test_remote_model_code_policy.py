import sys
import types

import pytest

from train_factory.config.settings import Settings
from train_factory.trainers.decoder import decoder_reranker_trainer as decoder_module
from train_factory.trainers.decoder import llm_trainer as llm_module
from train_factory.trainers.decoder.decoder_reranker_trainer import DecoderRerankerTrainer
from train_factory.trainers.decoder.llm_trainer import LLMTrainer
from train_factory.trainers.encoder import embedding_trainer as embedding_module
from train_factory.trainers.encoder.embedding_trainer import EmbeddingTrainer


class _StopAfterModelLoad(RuntimeError):
    pass


def _install_fake_transformers(monkeypatch, *, tokenizer_loader, model_loader=None):
    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=tokenizer_loader)
    if model_loader is not None:
        module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=model_loader)
    module.TrainingArguments = object
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_remote_model_code_is_disabled_by_default():
    assert Settings.model_fields["allow_model_remote_code"].default is False


def test_llm_loader_ignores_request_remote_code_override(monkeypatch):
    calls = {}
    tokenizer = types.SimpleNamespace(pad_token="pad", eos_token="eos")
    model = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True))

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return tokenizer

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        return model

    _install_fake_transformers(
        monkeypatch,
        tokenizer_loader=load_tokenizer,
        model_loader=load_model,
    )
    monkeypatch.setattr(
        llm_module,
        "settings",
        types.SimpleNamespace(allow_model_remote_code=False),
        raising=False,
    )
    trainer = object.__new__(LLMTrainer)
    trainer.config = {
        "base_model_path": "/models/untrusted",
        "trust_remote_code": True,
        "model_kwargs": {"trust_remote_code": True},
        "bf16": False,
        "fp16": False,
    }

    trainer._load_model_and_tokenizer()

    assert calls["tokenizer"][1]["trust_remote_code"] is False
    assert calls["model"][1]["trust_remote_code"] is False


def test_decoder_sft_loader_ignores_request_remote_code_override(monkeypatch, tmp_path):
    calls = {}
    tokenizer = types.SimpleNamespace(pad_token="pad", eos_token="eos")

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return tokenizer

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        raise _StopAfterModelLoad

    _install_fake_transformers(
        monkeypatch,
        tokenizer_loader=load_tokenizer,
        model_loader=load_model,
    )
    package = types.ModuleType("qwen3_rerank_trainer")
    package.ContrastiveSFTTrainer = object
    package.RerankDataset = object
    package.RerankCollator = object
    monkeypatch.setitem(sys.modules, "qwen3_rerank_trainer", package)
    training_package = types.ModuleType("qwen3_rerank_trainer.training")
    sft_module = types.ModuleType("qwen3_rerank_trainer.training.sft_trainer")
    sft_module.get_yes_no_token_ids = lambda tokenizer: (1, 2)
    monkeypatch.setitem(sys.modules, "qwen3_rerank_trainer.training", training_package)
    monkeypatch.setitem(
        sys.modules,
        "qwen3_rerank_trainer.training.sft_trainer",
        sft_module,
    )
    monkeypatch.setattr(
        decoder_module,
        "settings",
        types.SimpleNamespace(allow_model_remote_code=False),
        raising=False,
    )
    trainer = DecoderRerankerTrainer(
        {
            "base_model_path": "/models/untrusted",
            "output_dir": str(tmp_path),
            "trust_remote_code": True,
            "model_kwargs": {"trust_remote_code": True},
            "bf16": False,
            "fp16": False,
        }
    )

    with pytest.raises(_StopAfterModelLoad):
        trainer._train_sft()

    assert calls["tokenizer"][1]["trust_remote_code"] is False
    assert calls["model"][1]["trust_remote_code"] is False


def test_decoder_rl_passes_policy_to_third_party_loader(monkeypatch, tmp_path):
    calls = {}

    def load_sft_model(sft_model_path, base_model_path, **kwargs):
        calls["load_sft_model"] = (sft_model_path, base_model_path, kwargs)
        raise _StopAfterModelLoad

    _install_fake_transformers(
        monkeypatch,
        tokenizer_loader=lambda *args, **kwargs: None,
    )
    package = types.ModuleType("qwen3_rerank_trainer")
    package.RLTrainer = object
    package.load_sft_model = load_sft_model
    monkeypatch.setitem(sys.modules, "qwen3_rerank_trainer", package)
    training_package = types.ModuleType("qwen3_rerank_trainer.training")
    training_package.RLRerankDataset = object
    training_package.RLCollator = object
    monkeypatch.setitem(sys.modules, "qwen3_rerank_trainer.training", training_package)
    monkeypatch.setattr(
        decoder_module,
        "settings",
        types.SimpleNamespace(allow_model_remote_code=False),
        raising=False,
    )
    trainer = DecoderRerankerTrainer(
        {
            "base_model_path": "/models/untrusted",
            "sft_checkpoint_path": "/models/untrusted-adapter",
            "output_dir": str(tmp_path),
            "trust_remote_code": True,
        }
    )

    with pytest.raises(_StopAfterModelLoad):
        trainer._train_rl()

    assert calls["load_sft_model"][2]["trust_remote_code"] is False


def test_embedding_loader_overrides_top_level_and_nested_remote_code(monkeypatch):
    captured = {}
    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {
        "trust_remote_code": True,
        "model_kwargs": {"trust_remote_code": True},
    }
    monkeypatch.setattr(
        embedding_module,
        "settings",
        types.SimpleNamespace(allow_model_remote_code=False),
        raising=False,
    )
    monkeypatch.setattr(
        trainer,
        "_requires_text_config_hidden_size_fallback",
        lambda *args, **kwargs: False,
    )

    def fake_sentence_transformer(model_name_or_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(embedding_module, "SentenceTransformer", fake_sentence_transformer)

    trainer._load_sentence_transformer(
        "/models/untrusted",
        trust_remote_code=True,
        model_kwargs={"trust_remote_code": True},
    )

    assert captured["trust_remote_code"] is False
    assert "trust_remote_code" not in captured.get("model_kwargs", {})


def test_operations_escape_hatch_is_not_read_from_training_request(monkeypatch):
    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = {
        "allow_model_remote_code": False,
        "trust_remote_code": False,
        "model_kwargs": {"trust_remote_code": False},
    }
    monkeypatch.setattr(
        embedding_module,
        "settings",
        types.SimpleNamespace(allow_model_remote_code=True),
        raising=False,
    )

    load_kwargs = trainer._prepare_sentence_transformer_load_kwargs(
        {"trust_remote_code": False, "model_kwargs": {"trust_remote_code": False}}
    )

    assert load_kwargs["trust_remote_code"] is True
    assert "trust_remote_code" not in load_kwargs.get("model_kwargs", {})
