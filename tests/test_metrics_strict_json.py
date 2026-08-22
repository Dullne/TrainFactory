import asyncio
from contextlib import contextmanager
import importlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import training_routes
from train_factory.monitoring.metrics_callback import MetricsCallback
from train_factory.monitoring.training_metrics import TrainingMetricsLogger
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.utils.strict_json import sanitize_json_value


CURRENT_USER = {"user_id": "user-1", "username": "alice", "is_admin": False}


def _strict_loads(payload: str):
    def reject_constant(constant: str):
        raise AssertionError(f"non-standard JSON constant leaked: {constant}")

    return json.loads(payload, parse_constant=reject_constant)


def _assert_strict_json(value) -> None:
    encoded = json.dumps(value, allow_nan=False)
    _strict_loads(encoded)


def _training_service(monkeypatch):
    service_module = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[TrainingTaskDB.__table__])

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    return service_module.TrainingTaskService(), engine


def test_sanitizer_handles_non_finite_mapping_keys():
    payload = {
        float("nan"): "nan-key",
        float("inf"): "positive-infinity-key",
        float("-inf"): {float("nan"): float("-inf")},
    }

    sanitized = sanitize_json_value(payload)

    assert sanitized == {
        "NaN": "nan-key",
        "Infinity": "positive-infinity-key",
        "-Infinity": {"NaN": None},
    }
    _assert_strict_json(sanitized)


@pytest.mark.parametrize(
    "payload",
    [
        {"NaN": "literal-key", float("nan"): "numeric-key"},
        {float("nan"): "first-key", float("nan"): "second-key"},
        {"1": "literal-key", 1: "numeric-key"},
        {"1.5": "literal-key", 1.5: "numeric-key"},
        {"null": "literal-key", None: "none-key"},
        {"true": "literal-key", True: "boolean-key"},
    ],
)
def test_sanitizer_rejects_mapping_key_collisions(payload):
    with pytest.raises(ValueError, match="JSON object key collision"):
        sanitize_json_value(payload)


def test_sanitizer_canonicalizes_supported_mapping_keys_to_strings():
    sanitized = sanitize_json_value(
        {2: "integer", 1.5: "float", None: "none", False: "boolean"}
    )

    assert sanitized == {
        "2": "integer",
        "1.5": "float",
        "null": "none",
        "false": "boolean",
    }


def test_sanitizer_rejects_unsupported_mapping_keys():
    with pytest.raises(TypeError, match="Unsupported JSON object key"):
        sanitize_json_value({("tuple",): "value"})


def test_metrics_logger_recursively_sanitizes_every_json_write(tmp_path):
    logger = TrainingMetricsLogger(str(tmp_path), "strict-json-task")

    logger.save_loss_record(
        1,
        {
            "train_loss": float("nan"),
            "nested": {
                "scores": [float("inf"), -float("inf"), 0.25],
            },
        },
        epoch=float("inf"),
    )
    logger.finalize_epoch(
        1,
        {"nested": {"scores": [float("nan"), 0.5]}},
    )
    summary = logger.finalize_training(
        {"nested": [float("inf"), {"loss": float("nan")}]}
    )
    logger.save_metadata(
        {"nested": {"range": [float("-inf"), float("nan")]}}
    )

    history = [
        _strict_loads(line)
        for line in logger.loss_history_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = _strict_loads(
        logger.training_metrics_file.read_text(encoding="utf-8")
    )
    metadata = _strict_loads(logger.metadata_file.read_text(encoding="utf-8"))

    assert history[0]["train_loss"] is None
    assert history[0]["nested"]["scores"] == [None, None, 0.25]
    assert history[0]["epoch"] is None
    assert persisted["epoch_summaries"][0]["nested"]["scores"] == [None, 0.5]
    assert persisted["final_metrics"]["nested"] == [None, {"loss": None}]
    assert metadata["nested"]["range"] == [None, None]
    _assert_strict_json(summary)


def test_metrics_callback_sanitizes_nested_log_epoch_and_db_payloads():
    saved = []
    epochs = []
    db_updates = []
    callback = MetricsCallback(
        "/unused",
        "callback-task",
        db_update_callback=db_updates.append,
    )
    callback.metrics_logger = SimpleNamespace(
        save_loss_record=lambda **kwargs: saved.append(kwargs),
        finalize_epoch=lambda epoch, metrics: epochs.append((epoch, metrics)),
    )
    args = SimpleNamespace(num_train_epochs=2)
    state = SimpleNamespace(
        global_step=3,
        max_steps=10,
        epoch=1.5,
        log_history=[
            {
                "loss": {"branches": [float("nan"), 0.75]},
                "eval_loss": {"branches": [float("inf"), 0.5]},
            }
        ],
    )

    callback.on_log(
        args,
        state,
        SimpleNamespace(),
        logs={
            "loss": float("nan"),
            "eval_nested": {
                "scores": [float("inf"), -float("inf"), 1.0]
            },
        },
    )
    callback.on_epoch_end(args, state, SimpleNamespace())

    assert saved[0]["metrics"] == {
        "train_loss": None,
        "eval_nested": {"scores": [None, None, 1.0]},
    }
    assert db_updates[0]["train_loss"] is None
    assert db_updates[0]["eval_nested"] == {"scores": [None, None, 1.0]}
    assert epochs == [
        (
            1,
            {
                "total_steps_in_epoch": 3,
                "avg_train_loss": {"branches": [None, 0.75]},
                "eval_loss": {"branches": [None, 0.5]},
            },
        )
    ]
    _assert_strict_json(saved)
    _assert_strict_json(db_updates)
    _assert_strict_json(epochs)


def test_training_service_sanitizes_runtime_and_final_metrics_before_db_write(
    monkeypatch,
):
    service, engine = _training_service(monkeypatch)
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="db-strict-json",
                status="running",
                run_token="run-1",
                training_params={"existing": True},
            )
        )
        session.add_all(
            [
                TrainingTaskDB(
                    task_id="legacy-final-metrics",
                    status="succeeded",
                    final_metrics={"loss": float("nan")},
                ),
                TrainingTaskDB(
                    task_id="legacy-result-writer",
                    status="pending",
                ),
            ]
        )
        session.commit()

    assert service.get_task("legacy-final-metrics")["final_metrics"] == {
        "loss": None
    }
    assert service.update_task_result(
        "legacy-result-writer",
        final_metrics={"loss": float("inf")},
    )
    assert service.get_task("legacy-result-writer")["final_metrics"] == {
        "loss": None
    }

    assert service.update_task_metrics(
        "db-strict-json",
        {
            "train_loss": float("nan"),
            "nested": [float("inf"), {"eval_loss": -float("inf")}],
        },
        run_token="run-1",
    )
    runtime_metrics = service.get_task_metrics("db-strict-json")
    assert runtime_metrics["train_loss"] is None
    assert runtime_metrics["nested"] == [None, {"eval_loss": None}]

    assert service.complete_task(
        "db-strict-json",
        "succeeded",
        final_metrics={
            "loss": float("nan"),
            "nested": {"values": [float("inf"), 0.1]},
        },
        run_token="run-1",
    )
    task = service.get_task("db-strict-json")
    assert task["final_metrics"] == {
        "loss": None,
        "nested": {"values": [None, 0.1]},
    }
    assert task["training_params"]["_runtime_metrics"] == runtime_metrics
    _assert_strict_json(task["final_metrics"])
    _assert_strict_json(task["training_params"])

    with Session(engine) as session:
        stored = session.exec(
            select(TrainingTaskDB).where(
                TrainingTaskDB.task_id == "db-strict-json"
            )
        ).one()
        stored_legacy_writer = session.exec(
            select(TrainingTaskDB).where(
                TrainingTaskDB.task_id == "legacy-result-writer"
            )
        ).one()
        assert stored.final_metrics == task["final_metrics"]
        assert stored.training_params == task["training_params"]
        assert stored_legacy_writer.final_metrics == {"loss": None}
        _assert_strict_json(stored.final_metrics)
        _assert_strict_json(stored.training_params)


def test_metrics_api_sanitizes_legacy_file_and_database_values(
    monkeypatch,
    tmp_path,
):
    task_id = "legacy-metrics"
    logs_dir = tmp_path / "logs" / "training" / task_id
    logs_dir.mkdir(parents=True)
    (logs_dir / "loss_history.jsonl").write_text(
        '{"step": 1, "nested": [NaN, Infinity, -Infinity]}\n',
        encoding="utf-8",
    )
    (logs_dir / "training_metrics.json").write_text(
        '{"final_metrics": {"loss": NaN, "nested": [Infinity]}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "output_dir": str(tmp_path),
            "training_params": {},
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task_metrics",
        lambda _task_id: {
            "train_loss": float("nan"),
            "nested": [float("inf"), -float("inf")],
        },
    )
    monkeypatch.setattr(
        training_routes,
        "_resolve_server_managed_task_output",
        lambda *_args, **_kwargs: tmp_path,
    )

    response = asyncio.run(
        training_routes.get_training_metrics(task_id, 1000, CURRENT_USER)
    )
    payload = response.model_dump()

    assert payload["loss_history"][0]["nested"] == [None, None, None]
    assert payload["summary"]["final_metrics"] == {
        "loss": None,
        "nested": [None],
    }
    assert payload["current_metrics"] == {
        "train_loss": None,
        "nested": [None, None],
    }
    _assert_strict_json(payload)


def test_metrics_api_reads_only_the_bounded_jsonl_tail(monkeypatch, tmp_path):
    task_id = "bounded-tail-metrics"
    logs_dir = tmp_path / "logs" / "training" / task_id
    logs_dir.mkdir(parents=True)
    history = logs_dir / "loss_history.jsonl"
    history.write_text(
        "this invalid prefix must never be parsed\n"
        + "".join(
            json.dumps({"step": step, "loss": step / 1000}) + "\n"
            for step in range(6000)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {
            "task_id": task_id,
            "user_id": "user-1",
            "output_dir": str(tmp_path),
            "training_params": {},
        },
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task_metrics",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        training_routes,
        "_resolve_server_managed_task_output",
        lambda *_args, **_kwargs: tmp_path,
    )

    response = asyncio.run(
        training_routes.get_training_metrics(task_id, 2, CURRENT_USER)
    )

    assert [record["step"] for record in response.loss_history] == [5998, 5999]


def test_metrics_tail_reader_rejects_unbounded_limits(tmp_path):
    history = tmp_path / "loss_history.jsonl"
    history.write_text('{"step": 1}\n', encoding="utf-8")

    for invalid_limit in (0, -1, training_routes.MAX_METRICS_HISTORY_RECORDS + 1):
        with pytest.raises(ValueError, match="limit"):
            training_routes._read_metrics_jsonl_tail(history, invalid_limit)


def test_metrics_tail_reader_ignores_only_an_unterminated_partial_record(tmp_path):
    history = tmp_path / "loss_history.jsonl"
    history.write_bytes(
        b'{"step": 1}\n{"step": 2}\n{"step": 3, "loss":'
    )

    assert training_routes._read_metrics_jsonl_tail(history, 2) == [
        {"step": 1},
        {"step": 2},
    ]


def test_metrics_tail_reader_skips_large_blank_tail_without_losing_records(tmp_path):
    history = tmp_path / "loss_history.jsonl"
    history.write_bytes(
        b'{"step": 1}\n{"step": 2}\n'
        + (b" \n" * (training_routes._METRICS_TAIL_CHUNK_BYTES + 1))
    )

    assert training_routes._read_metrics_jsonl_tail(history, 2) == [
        {"step": 1},
        {"step": 2},
    ]


def test_metrics_summary_reader_rejects_oversized_files(tmp_path):
    summary = tmp_path / "training_metrics.json"
    summary.write_bytes(
        b'{"payload":"'
        + b"x" * training_routes.MAX_METRICS_SUMMARY_BYTES
        + b'"}'
    )

    with pytest.raises(ValueError, match="size limit"):
        training_routes._read_metrics_summary(summary)


@pytest.mark.parametrize("invalid_record", [b"1\n", b"[]\n"])
def test_metrics_tail_reader_rejects_non_object_records(tmp_path, invalid_record):
    history = tmp_path / "loss_history.jsonl"
    history.write_bytes(invalid_record)

    with pytest.raises(ValueError, match="JSON object"):
        training_routes._read_metrics_jsonl_tail(history, 1)
