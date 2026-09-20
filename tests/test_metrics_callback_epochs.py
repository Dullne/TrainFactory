"""Epoch summaries must describe one epoch, not the last/cumulative log."""

import json
from types import SimpleNamespace

import pytest
from transformers import TrainerControl, TrainerState

from train_factory.monitoring.metrics_callback import MetricsCallback


@pytest.fixture
def training(tmp_path):
    callback = MetricsCallback(str(tmp_path), "epoch-test")
    args = SimpleNamespace(
        num_train_epochs=2, learning_rate=0.001,
        per_device_train_batch_size=2, logging_steps=1,
    )
    state = TrainerState(max_steps=10)
    control = TrainerControl()
    yield callback, args, state, control
    callback.on_train_end(args, state, control)


def summaries(callback):
    payload = json.loads(callback.metrics_logger.training_metrics_file.read_text())
    return payload["epoch_summaries"]


def log(callback, args, state, control, step, epoch, **metrics):
    state.global_step, state.epoch = step, epoch
    state.log_history.append({"step": step, "epoch": epoch, **metrics})
    callback.on_log(args, state, control, logs=metrics)


def test_two_epochs_use_weighted_loss_and_local_steps(training):
    callback, args, state, control = training
    callback.on_train_begin(args, state, control)
    callback.on_epoch_begin(args, state, control)
    log(callback, args, state, control, 2, 0.5, loss=2.0)
    log(callback, args, state, control, 4, 1.0, loss=4.0)
    callback.on_epoch_end(args, state, control)
    callback.on_epoch_begin(args, state, control)
    log(callback, args, state, control, 5, 1.25, loss=4.0)
    log(callback, args, state, control, 8, 2.0, loss=8.0)
    callback.on_epoch_end(args, state, control)
    log(callback, args, state, control, 8, 2.0, train_loss=5.0)
    callback.on_train_end(args, state, control)

    result = summaries(callback)
    assert [row["epoch"] for row in result] == [1, 2]
    assert [row["total_steps_in_epoch"] for row in result] == [4, 4]
    assert [row["avg_train_loss"] for row in result] == [3.0, 7.0]
    assert callback.metrics_logger.get_summary()["final_metrics"]["final_train_loss"] == 5.0


def test_epoch_strategy_logs_after_epoch_end_are_included(training):
    callback, args, state, control = training
    callback.on_train_begin(args, state, control)
    callback.on_epoch_begin(args, state, control)
    state.global_step, state.epoch = 3, 1.0
    callback.on_epoch_end(args, state, control)
    log(callback, args, state, control, 3, 1.0, loss=2.5)
    log(callback, args, state, control, 3, 1.0, eval_loss=1.25)
    callback.on_train_end(args, state, control)

    result = summaries(callback)[0]
    assert result["avg_train_loss"] == 2.5
    assert result["eval_loss"] == 1.25
    assert result["total_steps_in_epoch"] == 3


def test_resumed_partial_epoch_counts_only_new_steps(training):
    callback, args, state, control = training
    state.global_step, state.epoch = 6, 1.5
    state.log_history = [{"step": 6, "epoch": 1.5, "loss": 99.0}]
    callback.on_train_begin(args, state, control)
    callback.on_epoch_begin(args, state, control)
    log(callback, args, state, control, 7, 1.75, loss=2.0)
    log(callback, args, state, control, 8, 2.0, loss=4.0)
    callback.on_epoch_end(args, state, control)
    callback.on_train_end(args, state, control)

    result = summaries(callback)[0]
    assert result["epoch"] == 2
    assert result["total_steps_in_epoch"] == 2
    assert result["avg_train_loss"] == 3.0


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), True, {"nested": 1}])
def test_invalid_loss_does_not_become_a_valid_epoch_average(training, loss):
    callback, args, state, control = training
    callback.on_train_begin(args, state, control)
    callback.on_epoch_begin(args, state, control)
    log(callback, args, state, control, 1, 0.5, loss=loss)
    log(callback, args, state, control, 2, 1.0, loss=4.0)
    callback.on_epoch_end(args, state, control)
    callback.on_train_end(args, state, control)

    assert summaries(callback)[0]["avg_train_loss"] is None


def test_unlogged_tail_and_cross_epoch_interval_are_not_invented(training):
    callback, args, state, control = training
    callback.on_train_begin(args, state, control)
    callback.on_epoch_begin(args, state, control)
    log(callback, args, state, control, 2, 0.5, loss=2.0)
    state.global_step, state.epoch = 3, 1.0
    callback.on_epoch_end(args, state, control)
    callback.on_epoch_begin(args, state, control)
    # This interval covers steps 3 and 4, straddling the epoch boundary.
    log(callback, args, state, control, 4, 1.5, loss=10.0)
    log(callback, args, state, control, 5, 2.0, loss=2.0)
    callback.on_epoch_end(args, state, control)
    callback.on_train_end(args, state, control)

    assert [row["avg_train_loss"] for row in summaries(callback)] == [None, None]
    assert [row["total_steps_in_epoch"] for row in summaries(callback)] == [3, 2]


def test_partial_final_epoch_uses_display_epoch_and_is_not_duplicated(training):
    callback, args, state, control = training
    callback.on_train_begin(args, state, control)
    callback.on_epoch_begin(args, state, control)
    log(callback, args, state, control, 1, 0.25, loss=2.0)
    callback.on_epoch_end(args, state, control)
    callback.on_train_end(args, state, control)
    callback.on_train_end(args, state, control)

    assert len(summaries(callback)) == 1
    assert summaries(callback)[0]["epoch"] == 1
    assert summaries(callback)[0]["avg_train_loss"] == 2.0


@pytest.mark.parametrize("logging_strategy", ["steps", "epoch"])
def test_real_trainer_persists_epoch_loss_after_evaluation(tmp_path, logging_strategy):
    import torch
    from transformers import Trainer, TrainingArguments

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, labels=None):
            values = input_ids.float() * self.weight
            return {"loss": values.mean(), "logits": values}

    data = [{"input_ids": [value], "labels": [value]} for value in (1, 3, 5, 7)]
    callback = MetricsCallback(str(tmp_path), "real-trainer-epoch-test")
    trainer = Trainer(
        model=TinyModel(), train_dataset=data, eval_dataset=data,
        args=TrainingArguments(
            output_dir=str(tmp_path), num_train_epochs=2, learning_rate=0,
            per_device_train_batch_size=2, per_device_eval_batch_size=2,
            logging_steps=1, logging_strategy=logging_strategy,
            eval_strategy="epoch", save_strategy="no", report_to="none",
            use_cpu=True, disable_tqdm=True, dataloader_pin_memory=False,
        ),
        callbacks=[callback],
    )
    trainer.train()

    result = summaries(callback)
    assert [row["epoch"] for row in result] == [1, 2]
    assert [row["total_steps_in_epoch"] for row in result] == [2, 2]
    assert [row["avg_train_loss"] for row in result] == [4.0, 4.0]
    assert [row["eval_loss"] for row in result] == [4.0, 4.0]
