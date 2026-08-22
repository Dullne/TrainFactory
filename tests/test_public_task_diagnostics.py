import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from train_factory.api.routes import evaluation_routes
from train_factory.api.routes import generation_routes
from train_factory.api.routes import deep_evaluation_routes
from train_factory.api.routes import training_routes


CURRENT_USER = {"user_id": "user-1", "username": "owner"}
RAW_FAILURE = r"worker failed while opening C:\\private\\tenant-a\\secret.jsonl"
PUBLIC_FAILURE = "Task failed. Check server logs for details."


def _quick_evaluation_request():
    return deep_evaluation_routes.EvaluateSampleRequest(
        input="question",
        expected_output="expected",
        actual_output="actual",
        retrieval_context=["context"],
        metrics=["answer_relevancy"],
        llm_config={
            "endpoint": "https://judge.example.com/v1",
            "model": "judge-model",
        },
    )


def _admit_quick_evaluation(monkeypatch):
    lease = SimpleNamespace(release=lambda: None)
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda _kind, _task_id, _user_id, operation: (operation(), lease),
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )


def _training_task() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "task_id": "training-public-error",
        "task_name": "training",
        "model_type": "embedding",
        "training_method": "full",
        "model_architecture": None,
        "base_model_path": "managed-model",
        "user_id": "user-1",
        "status": "failed",
        "progress": 25.0,
        "error_message": RAW_FAILURE,
        "final_model_path": None,
        "train_dataset_path": "managed-dataset",
        "training_params": {},
        "created_at": now,
        "started_at": now,
        "completed_at": now,
        "updated_at": now,
    }


def test_training_response_never_exposes_raw_worker_exception():
    response = training_routes._build_task_response(_training_task())

    assert response.error_message == PUBLIC_FAILURE
    assert "private" not in response.model_dump_json()


def test_generation_list_and_detail_never_expose_raw_worker_exception(monkeypatch):
    task = {
        "task_id": "generation-public-error",
        "task_name": "generation",
        "status": "failed",
        "generation_mode": "doc_to_training",
        "pos_neg_method": "retrieval",
        "progress": 12.0,
        "total_docs": 4,
        "processed_docs": 1,
        "output_sample_count": 0,
        "error_message": RAW_FAILURE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": None,
        "input_path": "managed-input",
        "input_format": "jsonl",
        "content_field": "text",
        "output_path": None,
        "output_format": "jsonl",
        "output_dataset_id": None,
        "auto_register_dataset": True,
        "llm_config": None,
        "eval_llm_config": None,
        "embedding_config": None,
    }
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "get_all_tasks",
        lambda **_kwargs: ([task], 1),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "get_task_stats",
        lambda **_kwargs: {"total": 1, "failed": 1},
    )
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: task,
    )

    listed = asyncio.run(
        generation_routes.list_tasks(
            limit=100,
            offset=0,
            current_user=CURRENT_USER,
        )
    )
    detailed = asyncio.run(
        generation_routes.get_task(task["task_id"], current_user=CURRENT_USER)
    )

    assert listed.tasks[0].error_message == PUBLIC_FAILURE
    assert detailed.error_message == PUBLIC_FAILURE
    assert "private" not in listed.model_dump_json()
    assert "private" not in detailed.model_dump_json()


def test_evaluation_public_payload_redacts_report_paths_and_nested_diagnostics():
    task = {
        "task_id": "evaluation-public-error",
        "task_name": "evaluation",
        "eval_type": "reranking",
        "model_configs": [],
        "dataset_configs": [],
        "status": "failed",
        "progress": 50.0,
        "error_message": RAW_FAILURE,
        "report_path": r"C:\\private\\tenant-a\\report.json",
        "results_path": r"C:\\private\\tenant-a\\results.json",
        "results": {
            "model": {
                "dataset": {
                    "_error": {"error": RAW_FAILURE},
                    "artifact_path": r"C:\\private\\tenant-a\\partial.json",
                    "failure_reason": RAW_FAILURE,
                    "artifact": r"C:\\private\\tenant-a\\artifact.json",
                    r"C:\\private\\tenant-a\\key.json": {"score": 1},
                }
            }
        },
        "current_dataset": r"C:\\private\\tenant-a\\unknown.jsonl",
        "model_progress": {
            "model": {
                "dataset": {
                    "status": "failed",
                    "error_message": RAW_FAILURE,
                    "trace_path": r"C:\\private\\tenant-a\\trace.log",
                }
            }
        },
    }

    public = evaluation_routes._public_evaluation_task(task)
    serialized = json.dumps(public)

    assert public["error_message"] == PUBLIC_FAILURE
    assert public["report_path"] is None
    assert public["results_path"] is None
    assert public["results"]["model"]["dataset"]["_error"] == {
        "error": PUBLIC_FAILURE
    }
    assert public["results"]["model"]["dataset"]["artifact_path"] is None
    assert public["results"]["model"]["dataset"]["failure_reason"] == PUBLIC_FAILURE
    assert public["results"]["model"]["dataset"]["artifact"] is None
    assert str(public["current_dataset"]).startswith("local:sha256:")
    assert public["model_progress"]["model"]["dataset"]["trace_path"] is None
    assert "private" not in serialized
    assert "secret.jsonl" not in serialized


def test_inactive_training_attempt_log_does_not_expose_run_token(monkeypatch, caplog):
    token = "4b02b1cb-34d1-4d32-bfef-a72cb781b3fe"
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {"task_id": "training-log", "status": "stopped"},
    )

    with caplog.at_level(logging.INFO, logger=training_routes.__name__):
        assert not training_routes._finalize_successful_training(
            "training-log",
            token,
        )

    assert "training-log" in caplog.text
    assert token not in caplog.text


def test_training_event_payload_never_exposes_worker_diagnostics(monkeypatch):
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda _task_id: {"task_id": "training-events", "user_id": "user-1"},
    )
    monkeypatch.setattr(
        training_routes.training_task_event_service,
        "list_events",
        lambda *_args, **_kwargs: [
            {
                "event_id": "event-1",
                "task_id": "training-events",
                "user_id": "user-1",
                "event_type": "status_changed",
                "payload": {
                    "from": "running",
                    "to": "failed",
                    "error_message": RAW_FAILURE,
                    "trace_path": r"C:\\private\\tenant-a\\trace.log",
                },
                "created_at": datetime.now(timezone.utc),
            }
        ],
    )

    response = asyncio.run(
        training_routes.list_task_events(
            "training-events",
            limit=200,
            current_user=CURRENT_USER,
        )
    )
    payload = response.model_dump()

    assert payload["events"][0]["payload"]["error_message"] == PUBLIC_FAILURE
    assert payload["events"][0]["payload"]["trace_path"] is None
    assert "private" not in json.dumps(payload)


def test_deep_evaluation_list_and_detail_redact_diagnostics(monkeypatch):
    task = {
        "task_id": "deep-public-error",
        "task_name": "deep evaluation",
        "eval_type": "retrieval",
        "model_configs": [],
        "dataset_configs": [],
        "metrics": ["recall"],
        "status": "failed",
        "progress": 50.0,
        "total_samples": 2,
        "processed_samples": 1,
        "model_progress": {
            "model": {"error_message": RAW_FAILURE},
        },
        "results_summary": {
            "by_group": {
                "group": {
                    "_error": RAW_FAILURE,
                    "artifact": r"C:\\private\\tenant-a\\partial.json",
                },
                "successful": {
                    "status": "completed",
                    "score": 0.9,
                    "reason": "Claims are supported by the retrieved context",
                    "details": {"explanation": "All citations matched"},
                },
                "failed": {
                    "score": 0,
                    "skipped": True,
                    "reason": RAW_FAILURE,
                    "details": {"error": RAW_FAILURE},
                },
            }
        },
        "results_path": r"C:\\private\\tenant-a\\results.json",
        "error_message": RAW_FAILURE,
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "get_all_tasks",
        lambda **_kwargs: ([task], 1),
    )
    monkeypatch.setattr(
        deep_evaluation_routes.deep_evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )

    listed = asyncio.run(
        deep_evaluation_routes.list_deep_evaluation_tasks(
            status=None,
            eval_type=None,
            limit=50,
            offset=0,
            current_user=CURRENT_USER,
        )
    )
    detailed = asyncio.run(
        deep_evaluation_routes.get_deep_evaluation_task(
            task["task_id"],
            current_user=CURRENT_USER,
        )
    )
    listed_payload = listed.model_dump()
    detailed_payload = (
        detailed.model_dump()
        if isinstance(detailed, deep_evaluation_routes.DeepEvaluationTaskResponse)
        else deep_evaluation_routes.DeepEvaluationTaskResponse(**detailed).model_dump()
    )

    for payload in (listed_payload["items"][0], detailed_payload):
        assert payload["error_message"] == PUBLIC_FAILURE
        assert payload["results_path"] is None
        assert payload["results_summary"]["by_group"]["group"]["_error"] == PUBLIC_FAILURE
        assert payload["results_summary"]["by_group"]["group"]["artifact"] is None
        assert payload["results_summary"]["by_group"]["successful"]["reason"] == (
            "Claims are supported by the retrieved context"
        )
        assert payload["results_summary"]["by_group"]["successful"]["details"] == {
            "explanation": "All citations matched"
        }
        assert payload["results_summary"]["by_group"]["failed"]["reason"] == (
            PUBLIC_FAILURE
        )
        assert "private" not in json.dumps(payload)


def test_quick_deep_evaluation_redacts_failed_metric_without_losing_success_reason(
    monkeypatch,
):
    _admit_quick_evaluation(monkeypatch)

    class FakeJudge:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeEvaluator:
        def __init__(self, **_kwargs):
            pass

        async def evaluate(self, _sample, _metrics):
            return SimpleNamespace(
                metric_results={
                    "successful": SimpleNamespace(
                        to_dict=lambda: {
                            "score": 0.9,
                            "reason": "Claims are supported",
                            "details": {"explanation": "Citations matched"},
                        }
                    ),
                    "failed": SimpleNamespace(
                        to_dict=lambda: {
                            "score": 0,
                            "skipped": True,
                            "reason": RAW_FAILURE,
                            "details": {"error": RAW_FAILURE},
                        }
                    ),
                },
                overall_score=0.9,
            )

    monkeypatch.setattr(
        deep_evaluation_routes,
        "create_llm_judge_from_dict",
        lambda _config: FakeJudge(),
    )
    monkeypatch.setattr(deep_evaluation_routes, "DeepEvaluator", FakeEvaluator)

    response = asyncio.run(
        deep_evaluation_routes.evaluate_sample(
            _quick_evaluation_request(),
            CURRENT_USER,
        )
    )

    assert response.results["successful"]["reason"] == "Claims are supported"
    assert response.results["failed"]["reason"] == PUBLIC_FAILURE
    assert response.results["failed"]["details"]["error"] == PUBLIC_FAILURE
    assert "private" not in response.model_dump_json()


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [(ValueError(RAW_FAILURE), 400), (RuntimeError(RAW_FAILURE), 500)],
)
def test_quick_deep_evaluation_never_returns_internal_exception_detail(
    monkeypatch,
    exception,
    expected_status,
):
    _admit_quick_evaluation(monkeypatch)
    monkeypatch.setattr(
        deep_evaluation_routes,
        "create_llm_judge_from_dict",
        lambda _config: (_ for _ in ()).throw(exception),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.evaluate_sample(
                _quick_evaluation_request(),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == expected_status
    assert RAW_FAILURE not in str(exc_info.value.detail)
    assert "private" not in str(exc_info.value.detail)
