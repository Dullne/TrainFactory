"""Truthful tuner validation across API, resume, worker, and trainer layers."""

import asyncio
import builtins
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from train_factory import train as train_module
from train_factory.api.routes import training_routes
from train_factory.schemas.training_config import TrainingParametersManager
from train_factory.trainers.base.base_trainer import BaseTrainer
from train_factory.trainers.decoder.llm_trainer import LLMTrainer
from train_factory.trainers.decoder.decoder_reranker_trainer import (
    DecoderRerankerTrainer,
)
from train_factory.trainers.encoder.embedding_trainer import EmbeddingTrainer
from train_factory.tuners import BaseTuner, TunerRegistry
from train_factory.tuners.full_param import FreezeLayersTuner
from train_factory.tuners.qlora import QLoRATuner
from train_factory.tuners.policy import (
    UnsupportedTunerError,
    canonicalize_tuner_config,
    ensure_tuner_supported,
)


def _request(**overrides):
    values = {
        "base_model_path": "model",
        "datasets": [{"path": "dataset.jsonl"}],
    }
    values.update(overrides)
    return training_routes.TrainingRequest(**values)


def test_tuner_type_is_trimmed_and_case_normalized():
    assert _request(tuner_type="  LoRA ").tuner_type == "lora"
    assert _request(tuner_type=" QLoRA ").tuner_type == "qlora"


def test_unknown_tuner_type_is_rejected_by_schema():
    with pytest.raises(ValidationError, match="tuner_type"):
        _request(tuner_type="banana")


def test_qlora_is_recognized_but_explicitly_unsupported():
    with pytest.raises(UnsupportedTunerError, match="4-bit quantization"):
        ensure_tuner_supported("qlora")


def test_freeze_is_recognized_but_explicitly_unsupported():
    with pytest.raises(UnsupportedTunerError, match="layer freezing"):
        ensure_tuner_supported("freeze")


def test_registry_cannot_bypass_disabled_qlora_policy():
    with pytest.raises(NotImplementedError, match="QLoRA"):
        TunerRegistry.create("qlora", {})

    with pytest.raises(UnsupportedTunerError):
        TunerRegistry.get("qlora")
    assert TunerRegistry.is_registered("qlora") is False
    assert "qlora" not in TunerRegistry.list_all()
    assert "freeze" not in TunerRegistry.list_all()
    assert set(TunerRegistry.list_all()) >= {"lora", "full"}


def test_direct_disabled_tuner_classes_fail_closed():
    with pytest.raises(UnsupportedTunerError, match="QLoRA"):
        QLoRATuner({})
    with pytest.raises(UnsupportedTunerError, match="layer freezing"):
        FreezeLayersTuner({"trainable_layers": ["layer"]})


def test_custom_registry_extension_remains_available():
    class CustomTuner(BaseTuner):
        def prepare_model(self, model):
            return model

        def get_trainable_parameters(self, model):
            return 0

        @property
        def tuner_type(self):
            return "custom_test"

    original = TunerRegistry._registry.get("custom_test")
    try:
        TunerRegistry.register("custom_test")(CustomTuner)
        assert isinstance(TunerRegistry.create("custom_test", {}), CustomTuner)
        assert TunerRegistry.get("custom_test") is CustomTuner
        assert TunerRegistry.is_registered("custom_test") is True
        trainer = object.__new__(EmbeddingTrainer)
        model = object()
        assert BaseTrainer._apply_tuner(
            trainer,
            model,
            {"tuner_type": "custom_test"},
        ) is model
    finally:
        if original is None:
            TunerRegistry._registry.pop("custom_test", None)
        else:
            TunerRegistry._registry["custom_test"] = original


def test_base_pipeline_applies_registered_custom_tuner_once():
    calls = []
    trained_model = object()

    class PipelineTuner(BaseTuner):
        def prepare_model(self, model):
            calls.append((model, dict(self.config)))
            return model

        def get_trainable_parameters(self, model):
            return 0

        @property
        def tuner_type(self):
            return "pipeline_custom"

    original = TunerRegistry._registry.get("pipeline_custom")
    try:
        TunerRegistry.register("pipeline_custom")(PipelineTuner)
        trainer = object.__new__(EmbeddingTrainer)
        trainer.raw_config = canonicalize_tuner_config(
            {
                "tuner_type": "pipeline_custom",
                "tuner_config": {"marker": 7},
            },
            allow_registered_custom=True,
        )
        trainer.model_config = {"base_model_path": "model"}
        trainer.initialize_model = lambda _model_name: trained_model
        trainer.device_manager = SimpleNamespace(
            get_training_device=lambda _requested: "cpu",
            prepare_model_for_training=lambda model, _device: model,
        )
        trainer.create_loss_function = lambda model, _dataset: ("loss", model)

        model, loss = trainer._initialize_model_and_loss(object())

        assert model is trained_model
        assert loss == ("loss", trained_model)
        assert calls == [
            (
                trained_model,
                {"marker": 7, "tuner_type": "pipeline_custom"},
            )
        ]
    finally:
        if original is None:
            TunerRegistry._registry.pop("pipeline_custom", None)
        else:
            TunerRegistry._registry["pipeline_custom"] = original


@pytest.mark.parametrize("model_type", ["llm", "decoder_reranker"])
def test_registered_custom_tuner_is_rejected_for_decoder_trainers(model_type):
    class EncoderOnlyTuner(BaseTuner):
        def prepare_model(self, model):
            return model

        def get_trainable_parameters(self, model):
            return 0

        @property
        def tuner_type(self):
            return "encoder_only_custom"

    original = TunerRegistry._registry.get("encoder_only_custom")
    try:
        TunerRegistry.register("encoder_only_custom")(EncoderOnlyTuner)
        with pytest.raises(ValueError, match="not supported for decoder"):
            canonicalize_tuner_config(
                {
                    "model_type": model_type,
                    "tuner_type": "encoder_only_custom",
                },
                allow_registered_custom=True,
            )
    finally:
        if original is None:
            TunerRegistry._registry.pop("encoder_only_custom", None)
        else:
            TunerRegistry._registry["encoder_only_custom"] = original


def test_direct_llm_trainer_cannot_silently_run_qlora_as_lora():
    with pytest.raises(UnsupportedTunerError, match="4-bit quantization"):
        LLMTrainer({"training_method": "sft", "tuner_type": "qlora"})


@pytest.mark.parametrize(
    "trainer_cls",
    [EmbeddingTrainer, DecoderRerankerTrainer],
)
def test_direct_non_llm_trainers_cannot_bypass_disabled_tuners(trainer_cls):
    with pytest.raises(UnsupportedTunerError):
        trainer_cls({"training_method": "sft", "tuner_type": "qlora"})


def test_unified_train_entry_rejects_disabled_tuner_before_other_validation():
    with pytest.raises(UnsupportedTunerError):
        train_module.train_with_config({"tuner_type": "freeze"})


@pytest.mark.parametrize(
    ("config", "expected_tuner"),
    [
        ({"tuner_type": "lora"}, "lora"),
        ({"use_lora": True}, "lora"),
        ({"lora_config": {"use_lora": True}}, "lora"),
        ({"tuner_type": "full"}, "full"),
    ],
)
def test_tuner_config_has_one_canonical_lora_truth(config, expected_tuner):
    canonical = canonicalize_tuner_config(config)

    assert canonical["tuner_type"] == expected_tuner
    expected_lora = expected_tuner == "lora"
    assert canonical["use_lora"] is expected_lora
    assert canonical["lora_config"]["use_lora"] is expected_lora


def test_canonicalizer_preserves_legacy_top_level_lora_parameters():
    canonical = canonicalize_tuner_config(
        {
            "use_lora": True,
            "lora_r": 7,
            "lora_alpha": 21,
            "lora_dropout": 0.25,
            "lora_target_modules": ["q_proj", "v_proj"],
            "lora_config": {"r": 11},
        }
    )
    manager = TrainingParametersManager()
    manager.load_from_config(canonical)
    parsed = manager.get_custom_params_dict()["lora_config"]

    assert canonical["lora_config"] == {
        "use_lora": True,
        "r": 11,
        "lora_alpha": 21,
        "lora_dropout": 0.25,
        "target_modules": ["q_proj", "v_proj"],
    }
    assert parsed == canonical["lora_config"]


@pytest.mark.parametrize(
    "config",
    [
        {"tuner_type": "full", "use_lora": True},
        {"tuner_type": "full", "lora_config": {"use_lora": True}},
    ],
)
def test_full_tuner_rejects_conflicting_lora_flags(config):
    with pytest.raises(ValueError, match="conflicts with LoRA"):
        canonicalize_tuner_config(config)


def test_create_route_rejects_full_tuner_with_lora_enabled(monkeypatch):
    monkeypatch.setattr(
        training_routes,
        "check_idempotency",
        lambda *_args, **_kwargs: (False, None),
    )
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: pytest.fail(
            "admission must not run for conflicting tuner config"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.create_training_task(
                _request(tuner_type="full", use_lora=True),
                BackgroundTasks(),
                {"user_id": "owner", "is_admin": False},
                None,
            )
        )

    assert exc_info.value.status_code == 400
    assert "conflicts with LoRA" in str(exc_info.value.detail)


@pytest.mark.parametrize(
    ("trainer_cls", "config_attribute"),
    [
        (LLMTrainer, "config"),
        (DecoderRerankerTrainer, "config"),
        (EmbeddingTrainer, "raw_config"),
    ],
)
def test_direct_trainers_enable_lora_from_tuner_type(
    trainer_cls,
    config_attribute,
):
    trainer = trainer_cls({"training_method": "sft", "tuner_type": "lora"})
    config = getattr(trainer, config_attribute)

    assert config["use_lora"] is True
    assert config["lora_config"]["use_lora"] is True


def test_unified_train_entry_canonicalizes_lora_before_trainer(monkeypatch):
    observed = {}
    trained_model = object()

    class FakeEmbeddingTrainer:
        def __init__(self, config):
            observed.update(config)

        def train(self, _progress_callback):
            return SimpleNamespace(
                model=trained_model,
                save_dir="saved-model",
                final_metrics={"loss": 0.1},
            )

    monkeypatch.setattr(
        "train_factory.trainers.encoder.embedding_trainer.EmbeddingTrainer",
        FakeEmbeddingTrainer,
    )

    result = train_module.train_with_config(
        {
            "base_model_path": "model",
            "train_dataset_path": "dataset.jsonl",
            "model_type": "embedding",
            "tuner_type": "lora",
        }
    )

    assert result.model is trained_model
    assert result.save_dir == "saved-model"
    assert observed["tuner_type"] == "lora"
    assert observed["use_lora"] is True
    assert observed["lora_config"]["use_lora"] is True


def test_embedding_lora_requires_peft(monkeypatch):
    trainer = object.__new__(EmbeddingTrainer)
    trainer.raw_config = canonicalize_tuner_config({"tuner_type": "lora"})
    real_import = builtins.__import__

    def import_without_peft(name, *args, **kwargs):
        if name == "peft":
            raise ImportError("peft intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_peft)

    with pytest.raises(RuntimeError, match="PEFT is required"):
        trainer._apply_lora_if_configured(object())


def test_tuner_application_errors_are_not_silently_downgraded(monkeypatch):
    class BrokenTuner:
        def prepare_model(self, _model):
            raise RuntimeError("adapter application failed")

    monkeypatch.setattr(
        TunerRegistry,
        "create",
        lambda *_args, **_kwargs: BrokenTuner(),
    )
    trainer = object.__new__(EmbeddingTrainer)

    with pytest.raises(RuntimeError, match="adapter application failed"):
        BaseTrainer._apply_tuner(
            trainer,
            object(),
            {"tuner_type": "lora"},
        )


def test_resume_rejects_legacy_qlora_before_claim_or_checkpoint_lookup(monkeypatch):
    task = {
        "task_id": "legacy-qlora",
        "user_id": "owner",
        "status": "failed",
        "process_pid": None,
        "process_status": None,
        "process_create_time": None,
        "training_params": {
            "tuner_type": "QLoRA ",
            "base_model_path": "model",
            "train_dataset_path": "dataset.jsonl",
        },
    }
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "find_latest_checkpoint",
        lambda _task_id: pytest.fail("checkpoint lookup must not run"),
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

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.resume_task(
                task["task_id"],
                BackgroundTasks(),
                {"user_id": "owner", "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 400
    assert "QLoRA" in str(exc_info.value.detail)


def test_resume_rejects_conflicting_stored_full_tuner(monkeypatch):
    task = {
        "task_id": "legacy-conflicting-full",
        "user_id": "owner",
        "status": "failed",
        "process_pid": None,
        "process_status": None,
        "process_create_time": None,
        "training_params": {
            "tuner_type": "full",
            "use_lora": True,
            "base_model_path": "model",
            "train_dataset_path": "dataset.jsonl",
        },
    }
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "find_latest_checkpoint",
        lambda _task_id: pytest.fail("checkpoint lookup must not run"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_routes.resume_task(
                task["task_id"],
                BackgroundTasks(),
                {"user_id": "owner", "is_admin": False},
            )
        )

    assert exc_info.value.status_code == 400
    assert "conflicts with LoRA" in str(exc_info.value.detail)


def test_worker_rejects_conflicting_tuner_before_gpu_allocation(monkeypatch):
    failures = []
    run_token = "conflicting-worker-run"
    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, _run_token: True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": "conflicting-worker",
            "status": "stopped",
            "run_token": run_token,
        },
    )
    monkeypatch.setattr(
        training_routes,
        "_persist_training_failure",
        lambda task_id, token, message: failures.append(
            (task_id, token, message)
        ),
    )
    monkeypatch.setattr(
        training_routes.gpu_resource_manager,
        "allocate_gpus_for_task",
        lambda *_args, **_kwargs: pytest.fail(
            "GPU allocation must not run for conflicting tuner config"
        ),
    )

    training_routes.run_training_task(
        "conflicting-worker",
        {
            "_run_token": run_token,
            "tuner_type": "full",
            "use_lora": True,
        },
    )

    assert failures == [
        (
            "conflicting-worker",
            run_token,
            "tuner_type='full' conflicts with LoRA being enabled",
        )
    ]
