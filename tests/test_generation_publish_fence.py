import asyncio
from contextlib import contextmanager
from datetime import datetime
import importlib
import inspect
import threading
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import generation_routes
from train_factory.generation import pipeline as pipeline_module
from train_factory.generation.pipeline import DatasetGenerationPipeline, PipelineConfig
from train_factory.generation.steps.base import Document, GeneratedSample
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)


class _AsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


def _sample() -> GeneratedSample:
    return GeneratedSample(
        query="question",
        answer="answer",
        positive_chunks=["positive"],
        negative_chunks=["negative"],
        source_doc_id="doc-1",
        metadata={"chunk_content": "source"},
    )


def test_run_does_not_clear_a_stop_latched_before_start(tmp_path, monkeypatch):
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(tmp_path / "output.jsonl"),
            generation_mode="qa_extraction",
        )
    )
    loaded = []
    monkeypatch.setattr(
        pipeline,
        "_load_documents",
        lambda: loaded.append(True) or [Document(content="source")],
    )
    pipeline.stop()

    result = asyncio.run(pipeline.run())

    assert result.success is False
    assert result.error == "Pipeline stopped"
    assert loaded == []


def test_generic_processor_stop_after_await_discards_samples_and_file_write(
    tmp_path,
    monkeypatch,
):
    entered = asyncio.Event()
    release = asyncio.Event()

    class PausingProcessor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def process(self, _document):
            entered.set()
            await release.wait()
            return [_sample()]

    monkeypatch.setattr(pipeline_module, "DocumentProcessor", PausingProcessor)
    output_path = tmp_path / "generic.jsonl"
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(output_path),
            generation_mode="qa_extraction",
            llm_concurrency=1,
        )
    )

    async def race():
        task = asyncio.create_task(
            pipeline._process_documents(
                [Document(content="source", doc_id="doc-1")],
                _AsyncContext(),
            )
        )
        await entered.wait()
        pipeline.stop()
        release.set()
        return await task

    assert asyncio.run(race()) == []
    assert output_path.read_text(encoding="utf-8") == ""


def test_eval_embedding_stop_after_await_discards_search_and_record(
    tmp_path,
):
    entered = asyncio.Event()
    release = asyncio.Event()
    searches = []
    progress_updates = []

    class PausingEmbedding:
        async def embed(self, _texts):
            entered.set()
            await release.wait()
            return [[0.1, 0.2]]

    class Milvus:
        def search_similar(self, *_args, **_kwargs):
            searches.append(True)
            return [[{"chunk_content": "candidate"}]]

    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(tmp_path / "output.jsonl"),
            generation_mode="qa_to_eval",
            embedding_concurrency=1,
        ),
        progress_callback=lambda processed, total: progress_updates.append(
            (processed, total)
        ),
    )

    async def race():
        task = asyncio.create_task(
            pipeline._generate_eval_data_batch(
                [
                    {
                        "query": "question",
                        "answer": "answer",
                        "chunk_id": "doc-1",
                        "chunk_content": "source",
                    }
                ],
                PausingEmbedding(),
                Milvus(),
                "collection",
            )
        )
        await entered.wait()
        pipeline.stop()
        release.set()
        return await task

    assert asyncio.run(race()) == []
    assert searches == []
    assert progress_updates == []


def test_pos_neg_llm_stop_after_await_discards_sample_and_checkpoint(
    tmp_path,
    monkeypatch,
):
    entered = asyncio.Event()
    release = asyncio.Event()

    class PausingProcessor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def process(self, _document):
            entered.set()
            await release.wait()
            return [_sample()]

    monkeypatch.setattr(pipeline_module, "DocumentProcessor", PausingProcessor)
    output_path = tmp_path / "pos-neg.jsonl"
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(output_path),
            pos_neg_method="llm",
            steps={
                "pos_neg_extraction": {
                    "enabled": True,
                    "num_positive": 2,
                    "num_negative": 1,
                }
            },
            llm_concurrency=1,
        )
    )

    async def race():
        task = asyncio.create_task(
            pipeline._generate_pos_neg_batch(
                [
                    {
                        "query": "question",
                        "answer": "answer",
                        "chunk_id": "doc-1",
                        "chunk_content": "source",
                    }
                ],
                _AsyncContext(),
                None,
                None,
                None,
                incremental_output_path=output_path,
            )
        )
        await entered.wait()
        pipeline.stop()
        release.set()
        return await task

    samples, deep_eval = asyncio.run(race())
    assert samples == []
    assert deep_eval == []
    assert not output_path.exists() or output_path.read_text(encoding="utf-8") == ""


def test_bounded_qa_queue_put_rechecks_stop_before_publication(
    tmp_path,
    monkeypatch,
):
    processed = asyncio.Event()
    published = []

    class Processor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def process(self, _document):
            processed.set()
            return [_sample()]

    monkeypatch.setattr(pipeline_module, "DocumentProcessor", Processor)
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(tmp_path / "output.jsonl"),
            task_id="blocked-put",
            llm_concurrency=1,
        ),
        on_qa_complete=lambda *_args: published.append(True),
    )
    monkeypatch.setattr(pipeline, "_create_llm_client", _AsyncContext)

    async def race():
        queue = asyncio.Queue(maxsize=1)
        blocker = object()
        await queue.put(blocker)
        producer = asyncio.create_task(
            pipeline._streaming_qa_producer(
                [Document(content="source", doc_id="doc-1")],
                queue,
            )
        )
        await processed.wait()
        for _ in range(10):
            await asyncio.sleep(0)
        assert producer.done() is False
        pipeline.stop()
        assert await queue.get() is blocker

        drained = []
        while not producer.done():
            item = await asyncio.wait_for(queue.get(), timeout=1)
            drained.append(item)
            if item is None:
                break
        all_qa, _ = await producer
        while not queue.empty():
            drained.append(queue.get_nowait())
        return all_qa, drained

    _all_qa, drained = asyncio.run(race())
    assert not any(isinstance(item, list) for item in drained)
    assert drained[-1] is None
    assert published == []


def test_doc_to_training_consumer_discards_batch_after_stop(tmp_path, monkeypatch):
    generated_batches = []
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(tmp_path / "training.jsonl"),
            generation_mode="doc_to_training",
            pos_neg_method="llm",
        )
    )
    monkeypatch.setattr(
        pipeline,
        "_load_documents",
        lambda: [Document(content="source", doc_id="doc-1")],
    )
    monkeypatch.setattr(pipeline, "_create_llm_client", _AsyncContext)

    async def producer(_documents, queue, *_args, **_kwargs):
        await queue.put(
            [
                {
                    "query": "question",
                    "answer": "answer",
                    "chunk_id": "doc-1",
                    "chunk_content": "source",
                }
            ]
        )
        pipeline.stop()
        await queue.put(None)
        return [], str(tmp_path / "qa.jsonl")

    async def generate(records, *_args, **_kwargs):
        generated_batches.append(records)
        return [_sample()], []

    monkeypatch.setattr(pipeline, "_streaming_qa_producer", producer)
    monkeypatch.setattr(pipeline, "_generate_pos_neg_batch", generate)

    result = asyncio.run(pipeline.run())

    assert result.success is False
    assert result.error == "Pipeline stopped"
    assert generated_batches == []


def test_streaming_qa_stop_during_processor_await_discards_result(
    tmp_path,
    monkeypatch,
):
    processor_entered = asyncio.Event()
    release_processor = asyncio.Event()
    published = []

    class FakeLLMClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class PausingProcessor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def process(self, _document):
            processor_entered.set()
            await release_processor.wait()
            return [
                GeneratedSample(
                    query="question",
                    answer="answer",
                    source_doc_id="doc-1",
                    metadata={"chunk_content": "source"},
                )
            ]

    monkeypatch.setattr(pipeline_module, "DocumentProcessor", PausingProcessor)
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(tmp_path / "output.jsonl"),
            task_id="stopped-at-await",
            llm_concurrency=1,
        ),
        on_qa_complete=lambda path, count: published.append((path, count)),
    )
    monkeypatch.setattr(pipeline, "_create_llm_client", lambda: FakeLLMClient())

    async def run_race():
        queue = asyncio.Queue()
        producer = asyncio.create_task(
            pipeline._streaming_qa_producer(
                [Document(content="source", doc_id="doc-1")],
                queue,
            )
        )
        await processor_entered.wait()
        pipeline.stop()
        release_processor.set()
        all_qa, qa_path = await producer
        queued = []
        while not queue.empty():
            queued.append(queue.get_nowait())
        return all_qa, qa_path, queued

    all_qa, qa_path, queued = asyncio.run(run_race())

    assert all_qa == []
    assert queued == [None]
    assert published == []
    output = tmp_path / "qa_extracted_stopped-at-await.jsonl"
    assert qa_path == str(output)
    assert not output.exists() or output.read_text(encoding="utf-8") == ""


@pytest.fixture
def generation_service(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'generation-publish-fence.db'}")
    SQLModel.metadata.create_all(
        engine,
        tables=[GenerationTaskDB.__table__],
    )
    service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    return service_module.GenerationTaskService(), engine


@pytest.mark.parametrize(
    ("current_status", "expected_update"),
    (
        (GenerationStatus.RUNNING, True),
        (GenerationStatus.STOPPED, False),
    ),
    ids=("running", "stopped"),
)
def test_set_qa_output_is_running_status_fenced(
    generation_service,
    current_status,
    expected_update,
):
    service, engine = generation_service
    task_id = f"qa-fence-{current_status}"
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=task_id,
                task_name=task_id,
                input_path="/managed/input.jsonl",
                llm_config={},
                steps_config={},
                status=current_status,
                started_at=datetime(2026, 1, 2, 3, 4, 5),
            )
        )
        session.commit()

    updated = service.set_qa_output(
        task_id,
        qa_output_path="/managed/qa.jsonl",
        qa_dataset_id="qa-dataset-id",
        expected_status=GenerationStatus.RUNNING,
    )

    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        assert updated is expected_update
        assert task.qa_output_path == (
            "/managed/qa.jsonl" if expected_update else None
        )
        assert task.qa_dataset_id == (
            "qa-dataset-id" if expected_update else None
        )


def test_publication_is_a_durable_non_stoppable_cas_phase(generation_service):
    service, engine = generation_service
    task_id = "durable-publication-phase"
    publishing = getattr(GenerationStatus, "PUBLISHING", None)
    assert publishing == "publishing"
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=task_id,
                task_name=task_id,
                input_path="/managed/input.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.RUNNING,
            )
        )
        session.commit()

    assert service.update_status(task_id, publishing) is True
    assert service.update_status(task_id, GenerationStatus.STOPPED) is False
    assert service.set_qa_output(
        task_id,
        qa_output_path="/managed/qa.jsonl",
        qa_dataset_id="qa-dataset-id",
        expected_status=GenerationStatus.RUNNING,
    ) is False
    assert service.set_output(
        task_id,
        "/managed/output.jsonl",
        1,
        "output-dataset-id",
        expected_status=publishing,
    ) is True
    assert service.update_status(task_id, GenerationStatus.COMPLETED) is True

    task = service.get_task(task_id)
    assert task["status"] == GenerationStatus.COMPLETED
    assert task["output_dataset_id"] == "output-dataset-id"


@pytest.mark.parametrize(
    ("action", "expected_status_code"),
    (("stop", 409), ("restart", 400), ("delete", 400)),
)
def test_publishing_api_is_non_stoppable_non_restartable_and_non_deletable(
    monkeypatch,
    action,
    expected_status_code,
):
    task_id = f"publishing-{action}"
    task = {
        "task_id": task_id,
        "status": GenerationStatus.PUBLISHING,
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: dict(task),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "get_task_raw",
        lambda _task_id: {**task, "run_token": "publishing-run-token"},
    )

    async def invoke():
        if action == "stop":
            return await generation_routes.stop_task(
                task_id,
                current_user={"user_id": "user-1"},
            )
        if action == "restart":
            return await generation_routes.restart_task(
                task_id,
                BackgroundTasks(),
                current_user={"user_id": "user-1"},
            )
        return await generation_routes.delete_task(
            task_id,
            current_user={"user_id": "user-1"},
        )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(invoke())

    assert exc_info.value.status_code == expected_status_code


def test_qa_registration_is_idempotent_within_one_attempt(monkeypatch):
    run_token = "11111111-1111-4111-8111-111111111111"
    task = {
        "task_id": "idempotent-qa-registration",
        "task_name": "idempotent",
        "user_id": "user-1",
    }
    staging_calls = []
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "stage_dataset",
        lambda **kwargs: staging_calls.append(kwargs) or kwargs["dataset_id"],
    )

    dataset_ids = [
        generation_routes._register_qa_dataset(
            task,
            "/managed/idempotent-qa.jsonl",
            3,
            run_token=run_token,
        )
        for _ in range(2)
    ]

    assert dataset_ids[0] == dataset_ids[1]
    assert [call["expected_run_token"] for call in staging_calls] == [
        run_token,
        run_token,
    ]
    assert all(call["relation_type"] == "qa_extracted" for call in staging_calls)
    assert all(
        "generation_run_token" not in call["extra_metadata"]
        for call in staging_calls
    )


def test_qa_registration_requires_attempt_token():
    task = {
        "task_id": "partial-qa-registration",
        "task_name": "partial",
        "user_id": "user-1",
    }

    with pytest.raises(TypeError, match="run_token"):
        generation_routes._register_qa_dataset(
            task,
            "/managed/partial-qa.jsonl",
            3,
        )


def test_generation_routes_have_no_visible_dataset_rollback_entrypoint():
    assert not hasattr(
        generation_routes,
        "_rollback_generation_dataset_registration",
    )


def test_qa_registration_never_repairs_a_visible_dataset_in_place(monkeypatch):
    run_token = "11111111-1111-4111-8111-111111111111"
    task = {
        "task_id": "repair-existing-registration",
        "task_name": "repair",
        "user_id": "user-1",
        "source_dataset_id": "source-dataset",
    }
    staging_calls = []
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda *_args, **_kwargs: pytest.fail(
            "registration bypassed attempt-owned staging"
        ),
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "create_dataset",
        lambda **_kwargs: pytest.fail(
            "registration created a publicly visible dataset"
        ),
    )
    monkeypatch.setattr(
        generation_routes.dataset_lineage_service,
        "create_edge",
        lambda **_kwargs: pytest.fail("registration wrote lineage out of transaction"),
    )
    monkeypatch.setattr(
        generation_routes.dataset_asset_service,
        "create_asset",
        lambda **_kwargs: pytest.fail("registration wrote an asset out of transaction"),
    )
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "stage_dataset",
        lambda **kwargs: staging_calls.append(kwargs) or "staging-dataset",
    )

    dataset_id = generation_routes._register_qa_dataset(
        task,
        "/managed/repair-qa.jsonl",
        3,
        run_token=run_token,
    )

    assert dataset_id == "staging-dataset"
    assert staging_calls[0]["expected_run_token"] == run_token
    assert staging_calls[0]["source_dataset_id"] == "source-dataset"


@pytest.mark.parametrize(
    ("helper_name", "args"),
    (
        ("_auto_register_dataset", ("/managed/output.jsonl", 3)),
        ("_register_qa_dataset", ("/managed/qa.jsonl", 3)),
        (
            "_register_qa_filtered_dataset",
            ("/managed/qa-filtered.jsonl", 2, {"kept": 2}),
        ),
        ("_register_deep_eval_dataset", ("/managed/deep-eval.jsonl", 1)),
    ),
)
def test_registration_helper_never_reports_rejected_staging_as_success(
    monkeypatch,
    helper_name,
    args,
):
    run_token = "22222222-2222-4222-8222-222222222222"
    task = {
        "task_id": f"rejected-staging-{helper_name}",
        "task_name": "rejected staging",
        "user_id": "user-1",
        "source_dataset_id": "source-dataset",
    }
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "stage_dataset",
        lambda **_kwargs: None,
    )

    assert getattr(generation_routes, helper_name)(
        task,
        *args,
        run_token=run_token,
    ) is None


def test_rejected_staging_never_invokes_visible_dataset_cleanup(monkeypatch):
    run_token = "33333333-3333-4333-8333-333333333333"
    task = {
        "task_id": "rejected-staging-cleanup",
        "task_name": "rejected staging cleanup",
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "stage_dataset",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "mark_deleting",
        lambda *_args, **_kwargs: pytest.fail("route marked a visible dataset"),
    )
    monkeypatch.setattr(
        generation_routes.dataset_asset_service,
        "delete_assets_for_dataset",
        lambda *_args, **_kwargs: pytest.fail("route deleted visible assets"),
    )
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "restore_from_deleting",
        lambda *_args, **_kwargs: pytest.fail("route restored a visible dataset"),
    )

    assert generation_routes._register_qa_dataset(
        task,
        "/managed/rejected-staging.jsonl",
        3,
        run_token=run_token,
    ) is None


def test_attempt_rotation_changes_staging_identity_and_owner(monkeypatch):
    old_token = "44444444-4444-4444-8444-444444444444"
    new_token = "55555555-5555-4555-8555-555555555555"
    task = {
        "task_id": "stale-compensation",
        "task_name": "stale compensation",
        "user_id": "user-1",
    }
    staging_calls = []
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "stage_dataset",
        lambda **kwargs: staging_calls.append(kwargs) or kwargs["dataset_id"],
    )

    dataset_ids = [
        generation_routes._register_qa_dataset(
            task,
            "/managed/rotated-qa.jsonl",
            3,
            run_token=run_token,
        )
        for run_token in (old_token, new_token)
    ]

    assert dataset_ids[0] != dataset_ids[1]
    assert [call["expected_run_token"] for call in staging_calls] == [
        old_token,
        new_token,
    ]


class _RouteTaskService:
    def __init__(self, stale_state):
        self.stale_state = stale_state
        self.run_token = f"run-token-{stale_state}"
        self.task = {
            "task_id": f"route-{stale_state}",
            "task_name": f"route-{stale_state}",
            "status": GenerationStatus.PENDING,
            "started_at": None,
            "user_id": "user-1",
            "input_path": "/managed/input.jsonl",
            "llm_config": {},
            "auto_register_dataset": True,
        }
        self.qa_writes = []

    def get_task(self, _task_id):
        return dict(self.task)

    def get_task_raw(self, _task_id):
        return {**self.task, "run_token": self.run_token}

    def claim_running(self, _task_id, *, expected_run_token=None):
        if expected_run_token is not None and expected_run_token != self.run_token:
            return None
        if self.task["status"] != GenerationStatus.PENDING:
            return None
        self.task["status"] = GenerationStatus.RUNNING
        self.task["started_at"] = "2026-01-02T03:04:05"
        return self.run_token

    def update_status(
        self,
        _task_id,
        status,
        _error_message=None,
        *,
        expected_run_token=None,
        **_kwargs,
    ):
        if expected_run_token is not None and expected_run_token != self.run_token:
            return False
        self.task["status"] = status
        return True

    def update_progress(self, *_args, **_kwargs):
        return True

    def set_qa_output(self, *args, **kwargs):
        if kwargs.get("expected_run_token", self.run_token) != self.run_token:
            return False
        self.qa_writes.append((args, kwargs))
        return True

    def set_output(self, *_args, **kwargs):
        return kwargs.get("expected_run_token", self.run_token) == self.run_token

    def complete_publication(
        self,
        task_id,
        *,
        expected_run_token,
        dataset_bindings,
        artifact_updates,
        **_kwargs,
    ):
        if (
            expected_run_token != self.run_token
            or self.task["status"] != GenerationStatus.PUBLISHING
        ):
            return False
        self.task.update(dataset_bindings)
        self.task.update(artifact_updates)
        return self.update_status(
            task_id,
            GenerationStatus.COMPLETED,
            expected_run_token=expected_run_token,
        )


@pytest.fixture(autouse=True)
def isolate_generation_worker_state():
    generation_routes._running_pipelines.clear()
    generation_routes._generation_cancellation_requests.clear()
    task_locks = getattr(generation_routes, "_generation_task_locks", None)
    if task_locks is not None:
        task_locks.clear()
    yield
    generation_routes._running_pipelines.clear()
    generation_routes._generation_cancellation_requests.clear()
    if task_locks is not None:
        task_locks.clear()


def test_old_attempt_cannot_claim_publication_after_restart(monkeypatch):
    service = _RouteTaskService("stale-publication-token")
    task_id = service.task["task_id"]
    old_token = service.run_token
    new_token = "newer-generation-run-token"
    publication_attempts = []
    registrations = []

    original_update_status = service.update_status

    def update_status(_task_id, status, error_message=None, **kwargs):
        if status == GenerationStatus.PUBLISHING:
            # Deterministically model STOPPED -> restart -> RUNNING between the
            # old worker's successful result and its publication CAS.
            service.run_token = new_token
            service.task["status"] = GenerationStatus.RUNNING
            publication_attempts.append(kwargs.get("expected_run_token"))
        return original_update_status(
            _task_id,
            status,
            error_message,
            **kwargs,
        )

    service.update_status = update_status
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "_auto_register_dataset",
        lambda *_args, **_kwargs: registrations.append(True) or "dataset-id",
    )

    class SuccessfulOldPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def stop(self):
            pass

        async def run(self):
            return SimpleNamespace(
                success=True,
                total_docs=0,
                processed_docs=0,
                output_samples=1,
                output_path="/managed/output.jsonl",
                details={},
                error=None,
            )

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        SuccessfulOldPipeline,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
        )
    )

    assert publication_attempts == [old_token]
    assert registrations == []
    assert service.run_token == new_token
    assert service.task["status"] == GenerationStatus.RUNNING


def test_stale_legacy_worker_cannot_adopt_restarted_uuid_attempt(monkeypatch):
    """An explicitly scheduled NULL token belongs only to the legacy attempt."""
    task_id = "stale-legacy-worker"
    new_token = "55555555-5555-4555-8555-555555555555"
    task = {
        "task_id": task_id,
        "status": GenerationStatus.PENDING,
        "run_token": new_token,
        "input_path": "/managed/input.jsonl",
        "user_id": "user-1",
    }
    validation_calls = []
    claim_calls = []

    class RestartedTaskService:
        def get_task_raw(self, _task_id):
            return dict(task)

        def claim_running(self, _task_id, *, expected_run_token=None):
            claim_calls.append(expected_run_token)
            return None

    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        RestartedTaskService(),
    )
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: validation_calls.append(True),
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda current: current["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
            expected_run_token=None,
        )
    )

    assert validation_calls == []
    assert claim_calls == []


def test_stale_legacy_stop_cleanup_cannot_stop_restarted_uuid_attempt(
    monkeypatch,
):
    task_id = "stale-legacy-stop"
    new_token = "66666666-6666-4666-8666-666666666666"
    task = {
        "task_id": task_id,
        "status": GenerationStatus.RUNNING,
        "run_token": new_token,
    }
    updates = []

    class RestartedTaskService:
        def get_task_raw(self, _task_id):
            return dict(task)

        def update_status(self, _task_id, status, **kwargs):
            updates.append((status, kwargs.get("expected_run_token")))
            task["status"] = status
            return True

    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        RestartedTaskService(),
    )

    generation_routes._stop_generation_execution(
        task_id,
        "stale legacy cleanup",
        run_token=None,
    )

    assert updates == []
    assert task["status"] == GenerationStatus.RUNNING


def test_stop_endpoint_does_not_cross_loop_block_during_publication(monkeypatch):
    service = _RouteTaskService("cross-loop-publication")
    task_id = service.task["task_id"]
    attempt_output_path = str(
        generation_routes.resolve_generation_attempt_output_path(
            "/managed/output.jsonl",
            task_id,
            service.run_token,
        )
    )
    service.task["output_path"] = attempt_output_path
    registration_entered = threading.Event()
    release_registration = threading.Event()
    worker_done = threading.Event()
    stop_done = threading.Event()
    stop_loop_ready = threading.Event()
    stop_loop_holder = []
    stop_result = []

    original_update_status = service.update_status

    def update_status(_task_id, status, error_message=None, **kwargs):
        expected_status = kwargs.get("expected_status")
        if expected_status is not None and service.task["status"] != expected_status:
            return False
        return original_update_status(
            _task_id,
            status,
            error_message,
            **kwargs,
        )

    service.update_status = update_status
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: service.get_task(task_id),
    )
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_tracking",
        lambda *_args, **_kwargs: True,
    )

    def blocking_registration(*_args, **_kwargs):
        registration_entered.set()
        assert release_registration.wait(5)
        return "published-dataset"

    monkeypatch.setattr(
        generation_routes,
        "_auto_register_dataset",
        blocking_registration,
    )

    class SuccessfulPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def stop(self):
            pass

        async def run(self):
            return SimpleNamespace(
                success=True,
                total_docs=0,
                processed_docs=0,
                output_samples=1,
                output_path=attempt_output_path,
                details={},
                error=None,
            )

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        SuccessfulPipeline,
    )

    def run_worker():
        try:
            asyncio.run(
                generation_routes._run_generation_task(
                    task_id,
                    PipelineConfig(
                        input_path="/managed/input.jsonl",
                        output_path=attempt_output_path,
                    ),
                )
            )
        finally:
            worker_done.set()

    def run_stop():
        loop = asyncio.new_event_loop()
        stop_loop_holder.append(loop)
        asyncio.set_event_loop(loop)
        stop_loop_ready.set()
        try:
            stop_result.append(
                loop.run_until_complete(
                    generation_routes.stop_task(
                        task_id,
                        current_user={"user_id": "user-1"},
                    )
                )
            )
        except HTTPException as exc:
            stop_result.append(exc.status_code)
        finally:
            loop.close()
            stop_done.set()

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    stop_thread = threading.Thread(target=run_stop, daemon=True)
    worker_thread.start()
    assert registration_entered.wait(5)
    stop_thread.start()
    assert stop_loop_ready.wait(5)

    # PUBLISHING is the durable point of no return, so the real endpoint must
    # reject immediately instead of waiting on an asyncio.Lock owned by a
    # different event loop while registration is blocked.
    completed_before_release = stop_done.wait(1)
    release_registration.set()
    if stop_loop_holder and not stop_loop_holder[0].is_closed():
        stop_loop_holder[0].call_soon_threadsafe(lambda: None)
    worker_thread.join(5)
    stop_thread.join(5)

    assert completed_before_release is True
    assert stop_result == [409]
    assert worker_done.is_set()
    assert stop_done.is_set()


@pytest.mark.parametrize("stale_state", ("stopped", "newer-attempt"))
def test_early_qa_publication_skips_inactive_attempt(
    monkeypatch,
    stale_state,
):
    service = _RouteTaskService(stale_state)
    task_id = service.task["task_id"]
    registrations = []

    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_stopped_generation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_register_qa_dataset",
        lambda *args, **kwargs: registrations.append((args, kwargs))
        or "qa-dataset-id",
    )
    level2_handler = importlib.import_module("train_factory.sync.level2_handler")
    monkeypatch.setattr(
        level2_handler,
        "on_qa_phase_completed",
        lambda **_kwargs: None,
    )

    class StaleCallbackPipeline:
        def __init__(self, *_args, **kwargs):
            self.on_qa_complete = kwargs["on_qa_complete"]

        def stop(self):
            pass

        async def run(self):
            if stale_state == "stopped":
                service.task["status"] = GenerationStatus.STOPPED
            else:
                generation_routes._running_pipelines[task_id] = object()
            callback_result = self.on_qa_complete("/managed/qa.jsonl", 1)
            if inspect.isawaitable(callback_result):
                await callback_result
            service.task["status"] = GenerationStatus.STOPPED
            return SimpleNamespace(success=False, error="stale worker stopped")

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        StaleCallbackPipeline,
    )
    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
        )
    )

    assert registrations == []
    assert service.qa_writes == []


def test_early_qa_callback_persists_only_attempt_fenced_path(monkeypatch):
    service = _RouteTaskService("early-path-only")
    task_id = service.task["task_id"]
    registered = []

    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_stopped_generation",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        generation_routes,
        "_register_qa_dataset",
        lambda *_args, **_kwargs: registered.append(True) or "qa-dataset-id",
    )

    class EarlyPathPipeline:
        def __init__(self, *_args, **kwargs):
            self.on_qa_complete = kwargs["on_qa_complete"]

        def stop(self):
            pass

        async def run(self):
            callback_result = self.on_qa_complete("/managed/qa.jsonl", 1)
            if inspect.isawaitable(callback_result):
                await callback_result
            return SimpleNamespace(success=False, error="stale")

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        EarlyPathPipeline,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
        )
    )

    assert registered == []
    assert len(service.qa_writes) == 1
    _, kwargs = service.qa_writes[0]
    assert kwargs["qa_output_path"] == "/managed/qa.jsonl"
    assert kwargs.get("qa_dataset_id") is None
    assert kwargs["expected_status"] == GenerationStatus.RUNNING
    assert kwargs["expected_run_token"] == service.run_token


def test_stale_worker_cleanup_does_not_clear_newer_attempt_cancellation(
    monkeypatch,
):
    service = _RouteTaskService("stale-cleanup")
    task_id = service.task["task_id"]

    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )

    newer_pipeline = object()

    class StalePipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def stop(self):
            pass

        async def run(self):
            generation_routes._running_pipelines[task_id] = newer_pipeline
            generation_routes._request_generation_cancellation(task_id)
            return SimpleNamespace(success=False, error="stale")

    monkeypatch.setattr(generation_routes, "DatasetGenerationPipeline", StalePipeline)

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
        )
    )

    assert generation_routes._running_pipelines[task_id] is newer_pipeline
    assert generation_routes._is_generation_cancellation_requested(task_id) is True


def test_final_publication_claim_wins_or_stop_wins_before_registration(
    monkeypatch,
):
    publishing = "publishing"

    class PublishingRaceService(_RouteTaskService):
        def __init__(self):
            super().__init__("final-publication-race")
            self.transitions = []
            self.publication_calls = []

        def update_status(
            self,
            _task_id,
            status,
            _error_message=None,
            *,
            expected_run_token=None,
            **_kwargs,
        ):
            if (
                expected_run_token is not None
                and expected_run_token != self.run_token
            ):
                return False
            current = self.task["status"]
            allowed = {
                GenerationStatus.PENDING: {GenerationStatus.RUNNING},
                GenerationStatus.RUNNING: {
                    publishing,
                    GenerationStatus.FAILED,
                    GenerationStatus.STOPPING,
                },
                GenerationStatus.STOPPING: {GenerationStatus.STOPPED},
                publishing: {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                },
            }
            if status not in allowed.get(current, set()):
                return False
            self.task["status"] = status
            self.transitions.append((current, status))
            return True

        def complete_publication(self, task_id, **kwargs):
            self.publication_calls.append(
                {
                    "status": self.task["status"],
                    **kwargs,
                }
            )
            return super().complete_publication(task_id, **kwargs)

    service = PublishingRaceService()
    task_id = service.task["task_id"]
    attempt_output_path = str(
        generation_routes.resolve_generation_attempt_output_path(
            "/managed/output.jsonl",
            task_id,
            service.run_token,
        )
    )
    service.task["output_path"] = attempt_output_path
    registrations = []
    stop_attempts = []

    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "validate_generation_resource_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_require_task_collection_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_refresh_owned_pipeline_model_configs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "_revalidate_task_model_config_ownership",
        lambda _task: {},
    )
    monkeypatch.setattr(
        generation_routes,
        "get_generation_input_allowed_dirs",
        lambda: ["/managed"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_tracking",
        lambda *_args, **_kwargs: True,
    )

    def register(*_args, **_kwargs):
        registrations.append(True)
        stop_attempts.append(
            service.update_status(task_id, GenerationStatus.STOPPED)
        )
        return "output-dataset-id"

    monkeypatch.setattr(generation_routes, "_auto_register_dataset", register)

    class SuccessfulPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def stop(self):
            pass

        async def run(self):
            return SimpleNamespace(
                success=True,
                total_docs=1,
                processed_docs=1,
                output_samples=1,
                output_path=attempt_output_path,
                details={},
                error=None,
            )

    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        SuccessfulPipeline,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(
                input_path="/managed/input.jsonl",
                output_path=attempt_output_path,
            ),
        )
    )

    assert registrations == [True]
    assert stop_attempts == [False]
    assert service.transitions[-2:] == [
        (GenerationStatus.RUNNING, publishing),
        (publishing, GenerationStatus.COMPLETED),
    ]
    assert service.task["status"] == GenerationStatus.COMPLETED
    assert service.publication_calls[0]["status"] == publishing
    assert service.publication_calls[0]["expected_run_token"] == service.run_token
    assert service.publication_calls[0]["artifact_updates"]["output_path"] == (
        attempt_output_path
    )


def test_startup_recovery_marks_orphan_publishing_failed(monkeypatch):
    from train_factory.api import server
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )
    from train_factory.storage.services.generation_task_service import (
        generation_task_service,
    )
    from train_factory.storage.services.generation_publication_service import (
        generation_publication_service,
    )

    queried = []
    claims = []
    compensations = []
    updates = []
    run_token = "startup-publishing-run-token"
    recovery_token = "startup-recovery-run-token"

    def get_all_tasks(*, status, limit, offset):
        queried.append((status, limit, offset))
        if status == "publishing":
            return ([{"task_id": "publishing-task", "status": status}], 1)
        return ([], 0)

    monkeypatch.setattr(generation_task_service, "get_all_tasks", get_all_tasks)
    monkeypatch.setattr(
        generation_task_service,
        "get_task_raw",
        lambda _task_id: {
            "task_id": "publishing-task",
            "status": GenerationStatus.PUBLISHING,
            "run_token": run_token,
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        generation_task_service,
        "claim_orphan_recovery",
        lambda task_id, **kwargs: claims.append(
            (
                task_id,
                kwargs.get("expected_status"),
                kwargs.get("expected_run_token"),
            )
        )
        or {"run_token": recovery_token},
    )
    monkeypatch.setattr(
        generation_task_service,
        "finish_orphan_recovery",
        lambda task_id, **kwargs: updates.append(
            (
                task_id,
                kwargs.get("terminal_status"),
                kwargs.get("error_message"),
                kwargs.get("expected_run_token"),
            )
        )
        or True,
    )
    monkeypatch.setattr(
        generation_publication_service,
        "compensate_attempt",
        lambda **kwargs: compensations.append(
            (
                kwargs.get("task_id"),
                kwargs.get("expected_run_token"),
                kwargs.get("recovery_run_token"),
                kwargs.get("user_id"),
            )
        )
        or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_generation_and_restore_batches",
        lambda *_args, **_kwargs: {"tracking_found": False, "recovered": False},
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )

    server.cleanup_orphan_generation_tasks()

    assert ("publishing", 1000, 0) in queried
    assert claims == [
        (
            "publishing-task",
            GenerationStatus.PUBLISHING,
            run_token,
        )
    ]
    assert compensations == [
        (
            "publishing-task",
            run_token,
            recovery_token,
            "user-1",
        )
    ]
    assert updates == [
        (
            "publishing-task",
            GenerationStatus.FAILED,
            "Dataset publication was interrupted by server restart. Restart the task to retry safely.",
            recovery_token,
        )
    ]
