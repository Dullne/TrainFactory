import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from train_factory.api.routes import deep_evaluation_routes
from train_factory.deep_evaluation import DeepEvaluator, EvaluationSample
from train_factory.deep_evaluation.evaluator import _ConcurrencyLimitedLLMClient
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskAlreadyExecuting,
    BackgroundTaskCapacityExceeded,
)


CURRENT_USER = {"user_id": "user-1", "username": "alice", "role": "user"}


@pytest.fixture(autouse=True)
def external_inference_catalog(monkeypatch):
    from train_factory.storage.services import inference_authorization_service

    monkeypatch.setattr(
        inference_authorization_service, "registered_shared_models_for_endpoint",
        lambda *_args: (False, set()),
    )


def _quick_request(**overrides):
    payload = {
        "input": "How does admission control work?",
        "expected_output": "It bounds concurrent work.",
        "actual_output": "It rejects work after the configured limit.",
        "retrieval_context": ["Admission control limits active work."],
        "metrics": ["answer_relevancy"],
        "llm_config": {
            "endpoint": "https://judge.example.com/v1",
            "model": "judge-model",
            "concurrency": 2,
        },
    }
    payload.update(overrides)
    return deep_evaluation_routes.EvaluateSampleRequest(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param("input", "x" * 16_385, id="input-chars"),
        pytest.param(
            "expected_output",
            "x" * 65_537,
            id="expected-output-chars",
        ),
        pytest.param("actual_output", "x" * 65_537, id="actual-output-chars"),
        pytest.param(
            "retrieval_context",
            ["x" * 32_769],
            id="context-item-chars",
        ),
        pytest.param("retrieval_context", ["x"] * 201, id="context-items"),
        pytest.param(
            "retrieval_context",
            ["x" * 32_768] * 9,
            id="context-total-chars",
        ),
        pytest.param("metrics", ["metric"] * 17, id="metric-items"),
        pytest.param("metrics", ["m" * 129], id="metric-name-chars"),
    ),
)
def test_quick_evaluation_request_rejects_oversized_payloads(field, value):
    with pytest.raises(ValidationError):
        _quick_request(**{field: value})


def test_quick_evaluation_request_rejects_unknown_chunk_mode():
    with pytest.raises(ValidationError):
        _quick_request(chunk_eval_mode="unbounded")


def test_quick_evaluation_request_rejects_empty_metrics():
    with pytest.raises(ValidationError):
        _quick_request(metrics=[])


def test_quick_evaluation_request_accepts_exact_character_limits():
    contexts = [
        "x" * deep_evaluation_routes.MAX_DEEP_EVALUATION_QUICK_CONTEXT_ITEM_CHARS
    ] * 8

    request = _quick_request(
        input="x" * deep_evaluation_routes.MAX_DEEP_EVALUATION_QUICK_INPUT_CHARS,
        expected_output=(
            "x" * deep_evaluation_routes.MAX_DEEP_EVALUATION_QUICK_OUTPUT_CHARS
        ),
        actual_output=(
            "x" * deep_evaluation_routes.MAX_DEEP_EVALUATION_QUICK_OUTPUT_CHARS
        ),
        retrieval_context=contexts,
        metrics=[
            "m"
            * deep_evaluation_routes.MAX_DEEP_EVALUATION_QUICK_METRIC_NAME_CHARS
        ],
    )

    assert sum(map(len, request.retrieval_context)) == (
        deep_evaluation_routes.MAX_DEEP_EVALUATION_QUICK_CONTEXT_TOTAL_CHARS
    )


def test_quick_evaluation_returns_429_when_admission_capacity_is_exhausted(
    monkeypatch,
):
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BackgroundTaskCapacityExceeded("per-user active task limit exceeded")
        ),
    )
    request = _quick_request(
        llm_config={"model": "judge-model"},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(deep_evaluation_routes.evaluate_sample(request, CURRENT_USER))

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}


def test_quick_evaluation_returns_409_when_reservation_is_already_executing(
    monkeypatch,
):
    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BackgroundTaskAlreadyExecuting("quick evaluation is already executing")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.evaluate_sample(_quick_request(), CURRENT_USER)
        )

    assert exc_info.value.status_code == 409


def test_quick_evaluation_preserves_http_exception_and_releases_lease(monkeypatch):
    releases = []
    lease = SimpleNamespace(release=lambda: releases.append(True))

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    request = _quick_request(
        llm_config={"model": "judge-model"},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(deep_evaluation_routes.evaluate_sample(request, CURRENT_USER))

    assert exc_info.value.status_code == 400
    assert releases == [True]


def test_quick_evaluation_releases_lease_after_unexpected_exception(monkeypatch):
    releases = []
    lease = SimpleNamespace(release=lambda: releases.append(True))

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("resolver unavailable")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deep_evaluation_routes.evaluate_sample(_quick_request(), CURRENT_USER)
        )

    assert exc_info.value.status_code == 500
    assert releases == [True]


def test_quick_evaluation_releases_lease_after_success(monkeypatch):
    releases = []
    admitted = []
    evaluator_options = []
    lease = SimpleNamespace(release=lambda: releases.append(True))

    def admit(kind, task_id, user_id, operation, *args, **kwargs):
        admitted.append((kind, task_id, user_id))
        return operation(*args, **kwargs), lease

    class FakeJudge:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeEvaluator:
        def __init__(self, **kwargs):
            evaluator_options.append(kwargs)

        async def evaluate(self, _sample, _metrics):
            metric_result = SimpleNamespace(
                to_dict=lambda: {"score": 1.0, "reason": "ok", "details": None}
            )
            return SimpleNamespace(
                metric_results={"answer_relevancy": metric_result},
                overall_score=1.0,
            )

    monkeypatch.setattr(
        deep_evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "validate_user_outbound_url",
        lambda url, _user_id: url,
    )
    monkeypatch.setattr(
        deep_evaluation_routes,
        "create_llm_judge_from_dict",
        lambda _config: FakeJudge(),
    )
    monkeypatch.setattr(deep_evaluation_routes, "DeepEvaluator", FakeEvaluator)

    response = asyncio.run(
        deep_evaluation_routes.evaluate_sample(_quick_request(), CURRENT_USER)
    )

    assert response.overall_score == 1.0
    assert admitted == [("evaluation", None, "user-1")]
    assert releases == [True]
    assert evaluator_options[0]["concurrency"] == 2


class _TrackingLLM:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def chat(self, _prompt):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return (
            '{"relevant": true, "score": 1.0, "reason": "ok", '
            '"key_points": [], "covered_points": [], "uncovered_points": [], '
            '"claims": [], "supported_claims": [], "unsupported_claims": [], '
            '"relevancy_scores": []}'
        )


def test_contextual_precision_individual_mode_respects_evaluator_concurrency():
    llm = _TrackingLLM()
    evaluator = DeepEvaluator(
        metrics=["contextual_precision"],
        llm_client=llm,
        concurrency=2,
        chunk_eval_mode="individual",
    )
    sample = EvaluationSample(
        input="query",
        expected_output="answer",
        retrieval_context=[f"context-{index}" for index in range(8)],
    )

    result = asyncio.run(evaluator.evaluate(sample))

    assert result.metric_results["contextual_precision"].score == 1.0
    assert llm.peak == 2


def test_quick_evaluation_metrics_share_one_llm_concurrency_limit():
    llm = _TrackingLLM()
    metrics = [
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "answer_relevancy",
        "faithfulness",
    ]
    evaluator = DeepEvaluator(
        metrics=metrics,
        llm_client=llm,
        concurrency=2,
        chunk_eval_mode="individual",
    )
    sample = EvaluationSample(
        input="query",
        expected_output="answer",
        actual_output="answer",
        retrieval_context=[f"context-{index}" for index in range(8)],
    )

    result = asyncio.run(evaluator.evaluate(sample))

    assert set(result.metric_results) == set(metrics)
    assert all(not item.skipped for item in result.metric_results.values())
    assert llm.peak == 2


def test_individual_batch_samples_share_one_llm_concurrency_limit():
    llm = _TrackingLLM()
    evaluator = DeepEvaluator(
        metrics=["contextual_precision"],
        llm_client=llm,
        concurrency=2,
        chunk_eval_mode="individual",
    )
    samples = [
        EvaluationSample(
            input=f"query-{sample_index}",
            expected_output="answer",
            retrieval_context=[f"context-{index}" for index in range(8)],
        )
        for sample_index in range(3)
    ]

    result = asyncio.run(evaluator.evaluate_batch(samples))

    assert len(result.results) == 3
    assert llm.calls == 24
    assert llm.peak == 2


def test_evaluator_concurrency_limit_is_safe_across_sequential_event_loops():
    llm = _TrackingLLM()
    metrics = [
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "answer_relevancy",
        "faithfulness",
    ]
    evaluator = DeepEvaluator(
        metrics=metrics,
        llm_client=llm,
        concurrency=2,
        chunk_eval_mode="individual",
    )
    sample = EvaluationSample(
        input="query",
        expected_output="answer",
        actual_output="answer",
        retrieval_context=[f"context-{index}" for index in range(8)],
    )

    first = asyncio.run(evaluator.evaluate(sample))
    second = asyncio.run(evaluator.evaluate(sample))

    assert all(not item.skipped for item in first.metric_results.values())
    assert all(not item.skipped for item in second.metric_results.values())
    assert llm.calls == 24


class _ThreadSafeTrackingLLM:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._lock = threading.Lock()

    async def chat(self, prompt):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.05)
            return prompt
        finally:
            with self._lock:
                self.active -= 1


def test_evaluator_concurrency_limit_is_global_across_concurrent_event_loops():
    llm = _ThreadSafeTrackingLLM()
    client = _ConcurrencyLimitedLLMClient(llm, concurrency=2)
    start = threading.Barrier(2)
    errors = []

    async def exercise_loop():
        await asyncio.to_thread(start.wait)
        await asyncio.gather(*(client.chat(index) for index in range(4)))

    def run_loop():
        try:
            asyncio.run(exercise_loop())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_loop, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert llm.calls == 8
    assert llm.peak <= 2


def test_concurrency_limit_does_not_leak_permit_when_waiter_is_cancelled():
    class BlockingLLM:
        def __init__(self):
            self.calls = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, prompt):
            self.calls.append(prompt)
            if prompt == "holder":
                self.started.set()
                await self.release.wait()
            return prompt

    async def exercise():
        llm = BlockingLLM()
        client = _ConcurrencyLimitedLLMClient(llm, concurrency=1)
        holder = asyncio.create_task(client.chat("holder"))
        await llm.started.wait()

        waiter = asyncio.create_task(client.chat("waiter"))
        await asyncio.sleep(0.03)
        assert llm.calls == ["holder"]
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        llm.release.set()
        assert await holder == "holder"
        assert await asyncio.wait_for(client.chat("after-cancel"), timeout=0.5) == (
            "after-cancel"
        )

    asyncio.run(exercise())


def test_concurrency_limit_releases_permit_and_preserves_client_exception():
    expected_error = RuntimeError("judge failed")

    class FailOnceLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise expected_error
            return prompt

    async def exercise():
        llm = FailOnceLLM()
        client = _ConcurrencyLimitedLLMClient(llm, concurrency=1)

        with pytest.raises(RuntimeError) as exc_info:
            await client.chat("first")
        assert exc_info.value is expected_error
        assert await asyncio.wait_for(client.chat("second"), timeout=0.5) == "second"

    asyncio.run(exercise())
