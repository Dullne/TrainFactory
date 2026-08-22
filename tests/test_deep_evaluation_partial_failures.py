import asyncio
from copy import deepcopy
from types import SimpleNamespace

from train_factory.deep_evaluation import deep_evaluation_runner
from train_factory.storage.entities.evaluation_task_entity import EvaluationStatus


def _sample(query: str) -> dict:
    return {
        "query": query,
        "documents": [f"{query}-positive", f"{query}-negative"],
        "labels": [1, 0],
    }


class _CountingProgress:
    def __init__(self) -> None:
        self.processed = 0

    async def increment(self, count: int = 1) -> None:
        self.processed += count


class _FakeDeepEvaluationService:
    def __init__(self, task: dict) -> None:
        self.task = deepcopy(task)
        self.completions = []
        self.partial_results = []
        self.progress_updates = []

    def get_task(self, _task_id, **_kwargs):
        return deepcopy(self.task)

    def update_status(self, _task_id, status, *_args, **_kwargs):
        self.task["status"] = status
        return True

    def init_model_progress(self, *_args, **_kwargs):
        return True

    def update_model_progress(self, *_args, **_kwargs):
        return True

    def update_progress(self, _task_id, processed, total):
        self.progress_updates.append((processed, total))
        return True

    def update_results(self, _task_id, results_summary=None, **_kwargs):
        self.partial_results.append(deepcopy(results_summary))
        self.task["results"] = deepcopy(results_summary)
        return True

    def complete_task(self, task_id, status, **kwargs):
        self.completions.append((task_id, status, deepcopy(kwargs)))
        self.task["status"] = status
        if kwargs.get("results") is not None:
            self.task["results"] = deepcopy(kwargs["results"])
        return True

    def cancel_task(self, _task_id):
        self.task["status"] = EvaluationStatus.CANCELLED
        return True


def _task(task_id: str, groups: list[dict]) -> dict:
    return {
        "task_id": task_id,
        "status": EvaluationStatus.PENDING,
        "user_id": "user-1",
        "model_configs": deepcopy(groups),
        "dataset_configs": [{"dataset_id": "dataset-1"}],
        "field_mapping": {},
        "metrics": ["mrr"],
        "worker_groups": {"retrieval_mode": "offline", "model_workers": 2},
        "max_samples": 10,
    }


def _group(group_name: str, model_name: str) -> dict:
    return {
        "group_name": group_name,
        "embedding": {
            "endpoint": "https://embedding.example.test/v1",
            "model_name": model_name,
            "concurrency": 1,
        },
    }


def _successful_existing_group() -> dict:
    return {
        "retrieval": {
            "embedding": {
                "summary": {
                    "datasets": {
                        "dataset-1": {
                            "summary": {
                                "overall": {"mean": 1.0, "min": 1.0, "max": 1.0},
                                "metrics": {
                                    "mrr": {
                                        "mean": 1.0,
                                        "min": 1.0,
                                        "max": 1.0,
                                        "count": 1,
                                    }
                                },
                            }
                        }
                    }
                }
            }
        },
        "llm": {},
    }


def _install_runner_environment(
    monkeypatch,
    tmp_path,
    task: dict,
    *,
    rows: list[dict] | None = None,
):
    service = _FakeDeepEvaluationService(task)
    client_calls = []

    class FakeEmbeddingClient:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

        async def similarity(self, query, _documents):
            client_calls.append((self.config.model, query))
            if self.config.model.startswith("fail") or query.startswith("fail"):
                raise RuntimeError(f"client failure for {self.config.model}:{query}")
            return [0.9, 0.1]

    class FakeRerankClient:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

        async def score(self, query, _documents):
            client_calls.append((self.config.model, query))
            if self.config.model.startswith("fail") or query.startswith("fail"):
                raise RuntimeError(f"client failure for {self.config.model}:{query}")
            return [0.9, 0.1]

    monkeypatch.setattr(
        deep_evaluation_runner,
        "deep_evaluation_task_service",
        service,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "validate_deep_evaluation_resource_config",
        lambda _task: None,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_require_task_collection_ownership",
        lambda _task: None,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_resolve_task_dataset_records",
        lambda *_args, **_kwargs: [
            {
                "dataset_id": "dataset-1",
                "dataset_name": "dataset-1",
                "storage_path": "/managed/dataset-1.jsonl",
            }
        ],
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_resolve_dataset_source",
        lambda *_args, **_kwargs: ("/managed/dataset-1.jsonl", "jsonl"),
    )
    source_rows = rows or [
        {"query": "ok-query", "positives": ["positive"], "negatives": ["negative"]}
    ]
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_iter_rows",
        lambda *_args, **_kwargs: iter(deepcopy(source_rows)),
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(deep_evaluation_runner, "EmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr(deep_evaluation_runner, "RerankClient", FakeRerankClient)
    monkeypatch.setattr(deep_evaluation_runner, "is_cancelled", lambda _task_id: False)
    monkeypatch.setattr(
        deep_evaluation_runner,
        "_cleanup_cancelled",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        deep_evaluation_runner,
        "get_settings",
        lambda: SimpleNamespace(output_dir=tmp_path),
    )
    return service, client_calls


def test_evaluate_samples_counts_one_client_failure_and_one_success(
    monkeypatch,
):
    calls = []

    class FakeEmbeddingClient:
        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

        async def similarity(self, query, _documents):
            calls.append(query)
            if query == "fail-query":
                raise RuntimeError("sample failed")
            return [0.9, 0.1]

    monkeypatch.setattr(deep_evaluation_runner, "EmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr(
        deep_evaluation_runner,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(deep_evaluation_runner, "is_cancelled", lambda _task_id: False)
    progress = _CountingProgress()

    results, skipped_count = asyncio.run(
        deep_evaluation_runner._evaluate_samples(
            "partial-samples",
            "embedding",
            {"endpoint": "https://embedding.example.test/v1", "model_name": "model"},
            [_sample("fail-query"), _sample("ok-query")],
            ["mrr"],
            1,
            progress,
            "user-1",
        )
    )

    assert calls == ["fail-query", "ok-query"]
    assert progress.processed == 2
    assert len(results) == 1
    assert results[0]["query"] == "ok-query"
    assert skipped_count == 1


def test_single_group_with_all_samples_failed_is_failed_group_not_empty_success(
    monkeypatch,
    tmp_path,
):
    task = _task("all-samples-failed", [_group("failed-group", "fail-model")])
    service, _calls = _install_runner_environment(monkeypatch, tmp_path, task)

    deep_evaluation_runner.run_deep_evaluation_task(task["task_id"])

    assert [status for _task_id, status, _kwargs in service.completions] == [
        EvaluationStatus.FAILED
    ]
    assert service.partial_results
    failed_group = service.partial_results[-1]["by_group"]["failed-group"]
    assert "_error" in failed_group
    assert not failed_group.get("retrieval")


def test_one_successful_group_and_one_failed_group_completes_with_both_results(
    monkeypatch,
    tmp_path,
):
    task = _task(
        "partial-groups",
        [_group("successful-group", "ok-model"), _group("failed-group", "fail-model")],
    )
    service, _calls = _install_runner_environment(monkeypatch, tmp_path, task)

    deep_evaluation_runner.run_deep_evaluation_task(task["task_id"])

    assert [status for _task_id, status, _kwargs in service.completions] == [
        EvaluationStatus.COMPLETED
    ]
    results = service.completions[0][2]["results"]["by_group"]
    assert results["successful-group"]["retrieval"]["embedding"]
    assert "_error" in results["failed-group"]


def test_failed_group_partial_metrics_are_excluded_from_aggregate(
    monkeypatch,
    tmp_path,
):
    discarded_group = _group("discarded-group", "ok-embedding")
    discarded_group["rerank"] = {
        "endpoint": "https://rerank.example.test/v1",
        "model_name": "fail-rerank",
        "concurrency": 1,
    }
    task = _task(
        "discard-partial-group-metrics",
        [discarded_group, _group("accepted-group", "ok-model")],
    )
    service, _calls = _install_runner_environment(monkeypatch, tmp_path, task)

    deep_evaluation_runner.run_deep_evaluation_task(task["task_id"])

    assert service.completions[0][1] == EvaluationStatus.COMPLETED
    results = service.completions[0][2]["results"]
    assert "_error" in results["by_group"]["discarded-group"]
    accepted_metric = results["by_group"]["accepted-group"]["retrieval"][
        "embedding"
    ]["summary"]["metrics"]["mrr"]
    assert results["metrics"]["mrr"] == accepted_metric


def test_all_groups_failed_marks_task_failed_without_completed_report(
    monkeypatch,
    tmp_path,
):
    task = _task(
        "all-groups-failed",
        [_group("failed-one", "fail-one"), _group("failed-two", "fail-two")],
    )
    service, _calls = _install_runner_environment(monkeypatch, tmp_path, task)

    deep_evaluation_runner.run_deep_evaluation_task(task["task_id"])

    assert [status for _task_id, status, _kwargs in service.completions] == [
        EvaluationStatus.FAILED
    ]
    assert "report_path" not in service.completions[0][2]
    assert not (tmp_path / "deep_evaluations" / f"{task['task_id']}.json").exists()


def test_resume_with_existing_success_and_new_failure_remains_partial_completed(
    monkeypatch,
    tmp_path,
):
    task = _task(
        "resume-partial-groups",
        [_group("existing-group", "must-not-run"), _group("failed-group", "fail-model")],
    )
    service, calls = _install_runner_environment(monkeypatch, tmp_path, task)
    existing = {"existing-group": _successful_existing_group()}

    deep_evaluation_runner.run_deep_evaluation_task(
        task["task_id"],
        existing_results=existing,
    )

    assert calls == [("fail-model", "ok-query")]
    assert [status for _task_id, status, _kwargs in service.completions] == [
        EvaluationStatus.COMPLETED
    ]
    results = service.completions[0][2]["results"]["by_group"]
    assert results["existing-group"] == existing["existing-group"]
    assert "_error" in results["failed-group"]


def test_historical_error_is_not_successful_and_is_retried_on_resume(
    monkeypatch,
    tmp_path,
):
    assert not deep_evaluation_runner._is_successful_group_result(
        {"_error": "old failure"}
    )

    task = _task("resume-error-group", [_group("retry-group", "ok-model")])
    service, calls = _install_runner_environment(monkeypatch, tmp_path, task)

    deep_evaluation_runner.run_deep_evaluation_task(
        task["task_id"],
        existing_results={"retry-group": {"_error": "old failure"}},
    )

    assert calls == [("ok-model", "ok-query")]
    assert service.completions[0][1] == EvaluationStatus.COMPLETED
    result = service.completions[0][2]["results"]["by_group"]["retry-group"]
    assert "_error" not in result
    assert result["retrieval"]["embedding"]


def test_successful_group_requires_at_least_one_non_skipped_metric_result():
    empty_retrieval_shell = {
        "retrieval": {"embedding": {"summary": {"datasets": {}}}},
        "llm": {},
    }
    all_llm_metrics_skipped = {
        "retrieval": {},
        "llm": {
            "datasets": {
                "dataset-1": {
                    "summary": {
                        "total_samples": 1,
                        "metrics": {
                            "answer_relevancy": {"count": 0, "skipped": 1}
                        },
                    }
                }
            }
        },
    }
    one_valid_llm_metric = deepcopy(all_llm_metrics_skipped)
    one_valid_llm_metric["llm"]["datasets"]["dataset-1"]["summary"][
        "metrics"
    ]["answer_relevancy"] = {
        "count": 1,
        "skipped": 0,
        "mean": 0.8,
        "min": 0.8,
        "max": 0.8,
    }

    assert not deep_evaluation_runner._is_successful_group_result(
        empty_retrieval_shell
    )
    assert not deep_evaluation_runner._is_successful_group_result(
        all_llm_metrics_skipped
    )
    assert deep_evaluation_runner._is_successful_group_result(one_valid_llm_metric)


def test_stale_existing_success_cannot_mask_all_current_groups_failing(
    monkeypatch,
    tmp_path,
):
    task = _task(
        "stale-existing-group",
        [_group("current-failed-group", "fail-model")],
    )
    service, calls = _install_runner_environment(monkeypatch, tmp_path, task)

    deep_evaluation_runner.run_deep_evaluation_task(
        task["task_id"],
        existing_results={"removed-group": _successful_existing_group()},
    )

    assert calls == [("fail-model", "ok-query")]
    assert [status for _task_id, status, _kwargs in service.completions] == [
        EvaluationStatus.FAILED
    ]
    assert service.partial_results
    by_group = service.partial_results[-1]["by_group"]
    assert set(by_group) == {"current-failed-group"}
    assert "_error" in by_group["current-failed-group"]
