from types import SimpleNamespace

import pytest

from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.trainers.base.base_trainer import BaseTrainer
from train_factory.trainers.base.training_result import TrainingResult
from train_factory.trainers.decoder.decoder_reranker_trainer import DecoderRerankerTrainer
from train_factory.trainers.decoder.llm_trainer import LLMTrainer


class DummyTrainer(BaseTrainer):
    def __init__(self, training_config, stage_log):
        self._stage_log = stage_log
        super().__init__(training_config)

    def _initialize_components(self):
        self.data_loader = object()

    def _load_datasets(self):
        return {"train": [1, 2], "eval": None, "test": [3]}

    def _prepare_datasets(self, datasets):
        return datasets

    def _record_dataset_stats(self, datasets):
        return None

    def initialize_model(self, model_name: str):
        return SimpleNamespace(
            eval=lambda: None,
            train=lambda: None,
            save=lambda path: None,
        )

    def create_loss_function(self, model, train_dataset):
        return None

    def create_training_args(self, config):
        return {}

    def create_trainer_instance(self, model, args, train_dataset, eval_dataset, loss, evaluator):
        return SimpleNamespace(
            train=lambda **kwargs: None,
            add_callback=lambda callback: None,
            state=SimpleNamespace(log_history=[]),
        )

    def create_evaluator(self, eval_dataset):
        return lambda model, output_path, epoch, steps: {"score": 1.0}

    def _update_task_stage(self, status: str, error_message=None):
        self._stage_log.append(status)

    def _save_model(self, model, save_dir: str):
        return None


def test_training_task_entity_allows_preparing_and_evaluating_flow():
    task = TrainingTaskDB(task_name="unit-task")

    task.update_status("preparing")
    task.update_status("running")
    task.update_status("evaluating")
    task.update_status("succeeded")

    assert task.status == "succeeded"

    with pytest.raises(ValueError):
        task.update_status("running")


def test_base_trainer_reports_stage_flow():
    stage_log = []
    trainer = DummyTrainer(
        {
            "task_id": "task-1",
            "model_type": "embedding",
            "base_model_path": "dummy-model",
            "train_dataset_path": "/tmp/train.jsonl",
            "dataset_configs": [{"path": "/tmp/train.jsonl", "split": "train"}],
            "output_dir": "/tmp/out",
        },
        stage_log,
    )

    trainer.train(progress_callback=None)

    assert stage_log == ["preparing", "running", "evaluating"]


def test_decoder_reranker_trainer_marks_preparing_before_dispatch(monkeypatch):
    stage_log = []
    trainer = DecoderRerankerTrainer({"task_id": "task-1", "training_method": "sft"})

    monkeypatch.setattr(trainer, "_update_task_stage", lambda status, error_message=None: stage_log.append(status))
    monkeypatch.setattr(
        trainer,
        "_train_sft",
        lambda progress_callback=None: TrainingResult(model=None, save_dir="/tmp/decoder"),
    )

    trainer.train()

    assert stage_log == ["preparing"]


def test_decoder_reranker_trainer_marks_running_before_trainer_loop(monkeypatch):
    stage_log = []
    trainer = DecoderRerankerTrainer({"task_id": "task-1", "training_method": "sft"})

    monkeypatch.setattr(trainer, "_update_task_stage", lambda status, error_message=None: stage_log.append(status))

    train_calls = []

    class DummyHFTrainer:
        def train(self, **kwargs):
            train_calls.append(kwargs)

    trainer._run_trainer_train(DummyHFTrainer(), None, "SFT")

    assert stage_log == ["running"]
    assert train_calls == [{}]


def test_llm_trainer_marks_preparing_before_dispatch(monkeypatch):
    stage_log = []
    trainer = LLMTrainer({"task_id": "task-1", "training_method": "dpo"})

    monkeypatch.setattr(trainer, "_update_task_stage", lambda status, error_message=None: stage_log.append(status))
    monkeypatch.setattr(
        trainer,
        "_train_dpo",
        lambda progress_callback=None: TrainingResult(model=None, save_dir="/tmp/llm"),
    )

    trainer.train()

    assert stage_log == ["preparing"]


def test_llm_trainer_marks_running_before_trainer_loop(monkeypatch):
    stage_log = []
    trainer = LLMTrainer({"task_id": "task-1", "training_method": "sft"})

    monkeypatch.setattr(trainer, "_update_task_stage", lambda status, error_message=None: stage_log.append(status))

    train_calls = []

    class DummyHFTrainer:
        def train(self, **kwargs):
            train_calls.append(kwargs)

    trainer._run_trainer_train(DummyHFTrainer(), "checkpoint-1", "SFT")

    assert stage_log == ["running"]
    assert train_calls == [{"resume_from_checkpoint": "checkpoint-1"}]
