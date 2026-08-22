import asyncio
from contextlib import contextmanager
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import threading
from uuid import UUID

import pytest
from sqlalchemy import update
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.api.routes import generation_routes
from train_factory.enums.dataset_status import DatasetStatus
from train_factory.generation import pipeline as pipeline_module
from train_factory.generation.pipeline import (
    DatasetGenerationPipeline,
    PipelineConfig,
    PipelineResult,
)
from train_factory.generation.steps.base import Document, GeneratedSample
from train_factory.storage.entities.dataset_asset_entity import DatasetAssetDB
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.dataset_lineage_entity import DatasetLineageEdgeDB
from train_factory.storage.entities.external_sync_entity import ExternalSyncTaskDB
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.milvus_collection_entity import (
    CollectionDatasetLinkDB,
    MilvusCollectionDB,
)
from train_factory.storage.services.dataset_service import (
    DatasetConsumptionUnavailableError,
    DatasetService,
    lock_datasets_for_consumption,
)


@pytest.mark.parametrize(
    "helper_name",
    (
        "_auto_register_dataset",
        "_register_qa_dataset",
        "_register_qa_filtered_dataset",
        "_register_deep_eval_dataset",
    ),
)
def test_generation_registration_helpers_require_attempt_token(helper_name):
    helper = getattr(generation_routes, helper_name)
    assert (
        inspect.signature(helper).parameters["run_token"].default
        is inspect.Parameter.empty
    )


class _AsyncClient:
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


def test_attempt_rotation_during_processor_await_prevents_artifact_mutation(
    tmp_path,
    monkeypatch,
):
    entered = asyncio.Event()
    release = asyncio.Event()
    current_token = {"value": "old-run-token"}
    progress = []

    class PausingProcessor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def process(self, _document):
            entered.set()
            await release.wait()
            return [_sample()]

    output_path = tmp_path / "attempt-output.jsonl"
    config = PipelineConfig(
        input_path=str(tmp_path / "input.jsonl"),
        output_path=str(output_path),
        generation_mode="qa_extraction",
        llm_concurrency=1,
    )
    # Wished-for PipelineConfig contract. setattr keeps this RED focused on
    # behavior until the fields are implemented.
    config.run_token = "old-run-token"
    config.attempt_is_valid = (
        lambda: current_token["value"] == config.run_token
    )
    pipeline = DatasetGenerationPipeline(
        config,
        progress_callback=lambda *args: progress.append(args),
    )
    monkeypatch.setattr(pipeline_module, "DocumentProcessor", PausingProcessor)

    async def race():
        task = asyncio.create_task(
            pipeline._process_documents(
                [Document(content="source", doc_id="doc-1")],
                _AsyncClient(),
            )
        )
        await entered.wait()
        current_token["value"] = "new-run-token"
        release.set()
        return await task

    assert asyncio.run(race()) == []
    assert output_path.read_text(encoding="utf-8") == ""
    assert progress == []


def test_attempt_rotated_after_worker_snapshot_never_starts_pipeline(
    tmp_path,
    monkeypatch,
):
    current_token = {"value": "old-run-token"}
    loaded = []
    config = PipelineConfig(
        input_path=str(tmp_path / "input.jsonl"),
        output_path=str(tmp_path / "output.jsonl"),
    )
    config.run_token = "old-run-token"
    config.attempt_is_valid = (
        lambda: current_token["value"] == config.run_token
    )
    pipeline = DatasetGenerationPipeline(config)
    monkeypatch.setattr(
        pipeline,
        "_load_documents",
        lambda: loaded.append(True) or [],
    )

    # Model the durable task snapshot being current, then a restart rotating
    # ownership immediately before the old coroutine is scheduled.
    current_token["value"] = "new-run-token"
    result = asyncio.run(pipeline.run())

    assert result.success is False
    assert result.error == "Pipeline stopped"
    assert loaded == []


@pytest.mark.parametrize(
    "resolver_name",
    (
        "_resolve_output_path",
        "_get_qa_output_path",
        "_get_qa_filtered_path",
        "_get_deep_eval_path",
    ),
)
def test_attempt_token_changes_every_generation_artifact_path(
    tmp_path,
    resolver_name,
):
    paths = []
    for run_token in ("old-run-token", "new-run-token"):
        config = PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(tmp_path / "generated_task.jsonl"),
            task_id="same-task",
        )
        config.run_token = run_token
        pipeline = DatasetGenerationPipeline(config)
        paths.append(str(getattr(pipeline, resolver_name)()))

    assert paths[0] != paths[1]


def test_restart_reads_previous_output_checkpoint_without_mutating_it(
    tmp_path,
    monkeypatch,
):
    previous_output = tmp_path / "previous-attempt.jsonl"
    current_output = tmp_path / "current-attempt.jsonl"
    previous_output.write_text(
        '{"query":"q1","answer":"a1","positives":["source-1"],'
        '"negatives":["old-neg"],"metadata":{"source_doc_id":"doc-1"}}\n',
        encoding="utf-8",
    )
    previous_bytes = previous_output.read_bytes()
    processed_queries = []

    class Processor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def process(self, document):
            query = document.metadata["query"]
            processed_queries.append(query)
            return [
                GeneratedSample(
                    query=query,
                    answer=document.metadata["answer"],
                    positive_chunks=[document.content],
                    negative_chunks=[f"negative-{query}"],
                    source_doc_id=document.doc_id,
                )
            ]

    monkeypatch.setattr(pipeline_module, "DocumentProcessor", Processor)
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(current_output),
            resume_output_path=str(previous_output),
            output_format="universal",
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
    pipeline._incremental_output_path = current_output

    samples, _deep_eval = asyncio.run(
        pipeline._generate_pos_neg_batch(
            [
                {
                    "query": "q1",
                    "answer": "a1",
                    "chunk_id": "doc-1",
                    "chunk_content": "source-1",
                },
                {
                    "query": "q2",
                    "answer": "a2",
                    "chunk_id": "doc-2",
                    "chunk_content": "source-2",
                },
            ],
            _AsyncClient(),
            None,
            None,
            None,
            incremental_output_path=current_output,
        )
    )
    pipeline._save_output(samples)

    assert processed_queries == ["q2"]
    assert previous_output.read_bytes() == previous_bytes
    current_records = [
        json.loads(line)
        for line in current_output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["query"] for record in current_records] == ["q1", "q2"]


@pytest.mark.parametrize(
    "artifact_field",
    (
        "output_path",
        "qa_output_path",
        "qa_filtered_path",
        "deep_eval_path",
    ),
)
def test_publication_rejects_artifacts_outside_current_attempt_before_staging(
    tmp_path,
    monkeypatch,
    artifact_field,
):
    task_id = "publication-path-fence"
    run_token = "17171717-1717-4717-8717-171717171717"
    task = {
        "task_id": task_id,
        "generation_mode": "doc_to_training",
        "auto_register_dataset": True,
        "progress": 90.0,
        "user_id": "user-1",
        "output_path": str(
            pipeline_module.resolve_generation_attempt_output_path(
                str(tmp_path / "generated.jsonl"),
                task_id,
                run_token,
            )
        ),
    }
    bad_path = str(tmp_path / "previous-attempt" / f"{artifact_field}.jsonl")
    details = {}
    output_path = None
    if artifact_field == "output_path":
        output_path = bad_path
    else:
        details[artifact_field] = bad_path
    result = PipelineResult(
        success=True,
        total_docs=1,
        processed_docs=1,
        output_samples=1,
        output_path=output_path,
        details=details,
    )
    staging_calls = []
    completion_calls = []
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "get_task",
        lambda _task_id: dict(task),
    )
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "complete_publication",
        lambda *_args, **kwargs: completion_calls.append(kwargs) or True,
    )
    for helper_name in (
        "_auto_register_dataset",
        "_register_qa_dataset",
        "_register_qa_filtered_dataset",
        "_register_deep_eval_dataset",
    ):
        monkeypatch.setattr(
            generation_routes,
            helper_name,
            lambda *_args, _helper=helper_name, **_kwargs: (
                staging_calls.append(_helper) or f"{_helper}-dataset"
            ),
        )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_tracking",
        lambda *_args, **_kwargs: True,
    )

    assert generation_routes._publish_generation_result(
        task_id,
        run_token,
        result,
    ) is False
    assert staging_calls == []
    assert completion_calls == []


@pytest.mark.parametrize(
    "save_method",
    ("_save_qa_output", "_save_qa_intermediate"),
)
def test_qa_savers_use_attempt_scoped_output_dir(
    tmp_path,
    save_method,
):
    config = PipelineConfig(
        input_path=str(tmp_path / "input.jsonl"),
        output_path=str(tmp_path / "unscoped" / "generated.jsonl"),
        task_id="qa-save-scope",
        run_token="18181818-1818-4818-8818-181818181818",
    )
    pipeline = DatasetGenerationPipeline(config)
    expected_dir = pipeline._get_output_dir()

    saved_path = getattr(pipeline, save_method)(
        [
            {
                "query": "question",
                "answer": "answer",
                "chunk_id": "chunk-1",
                "chunk_content": "source",
            }
        ]
    )

    assert Path(saved_path).parent == expected_dir


@pytest.fixture
def generation_database(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'generation-attempt-publication.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[
            DatasetDB.__table__,
            DatasetAssetDB.__table__,
            DatasetLineageEdgeDB.__table__,
            ExternalSyncTaskDB.__table__,
            GenerationTaskDB.__table__,
            MilvusCollectionDB.__table__,
            CollectionDatasetLinkDB.__table__,
        ],
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


def _add_generation_task(
    engine,
    task_id,
    status,
    *,
    run_token,
    user_id="user-1",
):
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id=task_id,
                task_name=task_id,
                input_path="/managed/input.jsonl",
                llm_config={},
                steps_config={},
                status=status,
                run_token=run_token,
                user_id=user_id,
            )
        )
        session.commit()
        # SQLAlchemy's Python-side UUID default can replace an explicit None
        # during INSERT.  Legacy-row tests need a real durable SQL NULL.
        if run_token is None:
            session.exec(
                update(GenerationTaskDB)
                .where(GenerationTaskDB.task_id == task_id)
                .values(run_token=None)
            )
            session.commit()


def test_running_stop_requires_worker_ack_before_restart(generation_database):
    service, engine = generation_database
    task_id = "stop-handshake"
    old_token = "11111111-1111-4111-8111-111111111111"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.RUNNING,
        run_token=old_token,
    )

    assert service.update_status(
        task_id,
        "stopping",
        expected_run_token=old_token,
    ) is True
    assert service.update_status(
        task_id,
        GenerationStatus.PENDING,
        expected_run_token=old_token,
    ) is False
    assert service.get_task_raw(task_id)["run_token"] == old_token
    assert service.update_status(
        task_id,
        GenerationStatus.STOPPED,
        expected_run_token=old_token,
    ) is True
    assert service.update_status(
        task_id,
        GenerationStatus.PENDING,
        expected_run_token=old_token,
    ) is True
    assert service.get_task_raw(task_id)["run_token"] != old_token


def test_orphan_recovery_preserves_existing_immutable_attempt_token(
    generation_database,
):
    service, engine = generation_database
    task_id = "immutable-orphan-token"
    run_token = "10101010-1010-4010-8010-101010101010"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )

    claim = service.claim_orphan_recovery(
        task_id,
        expected_status=GenerationStatus.PUBLISHING,
        expected_run_token=run_token,
    )

    assert claim == {
        "run_token": run_token,
        "source_status": GenerationStatus.PUBLISHING,
    }
    assert service.get_task_raw(task_id)["run_token"] == run_token


def test_staging_dataset_is_rejected_by_shared_consumption_lock(
    generation_database,
):
    _service, engine = generation_database
    dataset_id = "staging-dataset"
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id=dataset_id,
                dataset_name=dataset_id,
                storage_path="/managed/staging.jsonl",
                status="staging",
                source_type="generated",
                source_task_type="generation",
                source_task_id="generation-task",
                user_id="user-1",
            )
        )
        session.commit()

    with Session(engine) as session, pytest.raises(
        DatasetConsumptionUnavailableError
    ):
        lock_datasets_for_consumption(
            session,
            dataset_ids=(dataset_id,),
            require_all_dataset_ids=True,
        )


def test_visible_dataset_rollback_entrypoint_is_not_available():
    assert not hasattr(
        generation_routes,
        "_rollback_generation_dataset_registration",
    )




def test_public_dataset_reads_and_searches_hide_generation_staging(
    generation_database,
):
    _generation_service, engine = generation_database
    dataset_id = "hidden-staging-dataset"
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id=dataset_id,
                dataset_name="hidden staging",
                storage_path="/managed/hidden-staging.jsonl",
                status="staging",
                source_type="generated",
                source_task_type="generation",
                source_task_id="hidden-staging-task",
                user_id="user-1",
            )
        )
        session.commit()

    service = DatasetService()
    service.engine = engine

    assert service.get_dataset(dataset_id) is None
    assert service.get_dataset_by_name("hidden staging", "user-1") is None
    assert (
        service.get_dataset_by_storage_path(
            "/managed/hidden-staging.jsonl",
            "user-1",
        )
        is None
    )
    rows, total = service.list_datasets(user_id="user-1")
    assert rows == []
    assert total == 0
    assert service.search_datasets("hidden", user_id="user-1") == []


def test_real_stop_endpoint_keeps_running_attempt_stopping_until_worker_ack(
    generation_database,
    monkeypatch,
):
    service, engine = generation_database
    task_id = "real-stop-handshake"
    run_token = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.RUNNING,
        run_token=run_token,
    )
    finalizations = []
    monkeypatch.setattr(
        generation_routes,
        "generation_task_service",
        service,
    )
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda *_args, **_kwargs: service.get_task(task_id),
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_stopped_generation",
        lambda stopped_task_id, reason: finalizations.append(
            (stopped_task_id, reason)
        ),
    )

    try:
        response = asyncio.run(
            generation_routes.stop_task(
                task_id,
                current_user={"user_id": "user-1"},
            )
        )
        assert response == {"status": "stopping"}
        stopping = service.get_task_raw(task_id)
        assert stopping["status"] == "stopping"
        assert stopping["run_token"] == run_token
        assert service.update_status(
            task_id,
            GenerationStatus.PENDING,
            expected_run_token=run_token,
        ) is False
        assert finalizations == []

        generation_routes._stop_generation_execution(
            task_id,
            "worker acknowledged durable stop",
            run_token=run_token,
        )

        stopped = service.get_task_raw(task_id)
        assert stopped["status"] == GenerationStatus.STOPPED
        assert stopped["run_token"] == run_token
        assert finalizations == [
            (task_id, "worker acknowledged durable stop")
        ]
    finally:
        generation_routes._clear_generation_cancellation(
            task_id,
            run_token=run_token,
        )


def test_failed_publication_compensation_keeps_task_repairable_publishing(
    generation_database,
    monkeypatch,
):
    service, engine = generation_database
    task_id = "publication-compensation-repair"
    run_token = "acacacac-acac-4cac-8cac-acacacacacac"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PENDING,
        run_token=run_token,
    )
    failure_tracking = []

    class SuccessfulPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def stop(self):
            pass

        async def run(self):
            return PipelineResult(
                success=True,
                total_docs=1,
                processed_docs=1,
                output_samples=1,
                output_path="/managed/current-attempt/output.jsonl",
                details={},
            )

    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "DatasetGenerationPipeline",
        SuccessfulPipeline,
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
        lambda task: task["input_path"],
    )
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_input_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        generation_routes,
        "_publish_generation_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("publication failed")
        ),
    )
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "compensate_attempt",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        generation_routes,
        "_finalize_sync_generation_failure_tracking",
        lambda *_args, **_kwargs: failure_tracking.append(True) or True,
    )

    asyncio.run(
        generation_routes._run_generation_task(
            task_id,
            PipelineConfig(input_path="/managed/input.jsonl"),
            expected_run_token=run_token,
        )
    )

    current = service.get_task_raw(task_id)
    assert current["status"] == GenerationStatus.PUBLISHING
    assert current["run_token"] == run_token
    assert failure_tracking == []


def _add_staging_publication_rows(engine, task_id, run_token):
    output_id = "output-staging"
    qa_id = "qa-staging"
    source_id = "source-ready"
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id=source_id,
                dataset_name=source_id,
                storage_path="/managed/source.jsonl",
                status=DatasetStatus.READY.value,
                user_id="user-1",
            )
        )
        for dataset_id, path in (
            (output_id, "/managed/output.jsonl"),
            (qa_id, "/managed/qa.jsonl"),
        ):
            dataset = DatasetDB(
                dataset_id=dataset_id,
                dataset_name=dataset_id,
                storage_path=path,
                status="staging",
                source_type="generated",
                source_dataset_id=source_id,
                source_task_type="generation",
                source_task_id=task_id,
                user_id="user-1",
                extra_metadata={"generation_run_token": run_token},
            )
            dataset.generation_run_token = run_token
            session.add(dataset)
            session.add(
                DatasetAssetDB(
                    dataset_id=dataset_id,
                    storage_uri=path,
                    file_format="jsonl",
                )
            )
            session.add(
                DatasetLineageEdgeDB(
                    from_dataset_id=source_id,
                    to_dataset_id=dataset_id,
                    relation_type=(
                        "training_generated"
                        if dataset_id == output_id
                        else "qa_extracted"
                    ),
                    op_task_type="generation",
                    op_task_id=task_id,
                )
            )
        session.commit()
    return source_id, output_id, qa_id


def _publication_service_for_engine(engine, monkeypatch):
    module_name = (
        "train_factory.storage.services.generation_publication_service"
    )
    assert importlib.util.find_spec(module_name) is not None
    publication_module = importlib.import_module(module_name)

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(publication_module, "get_session", test_session)
    return publication_module, publication_module.GenerationPublicationService()


def test_staged_generation_lineage_does_not_persist_raw_attempt_token(
    generation_database,
    monkeypatch,
):
    _generation_service, engine = generation_database
    _publication_module, publication_service = _publication_service_for_engine(
        engine,
        monkeypatch,
    )
    task_id = "lineage-token-storage"
    run_token = "92929292-9292-4292-8292-929292929292"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )

    assert publication_service.stage_dataset(
        task_id=task_id,
        expected_run_token=run_token,
        dataset_id="lineage-token-output",
        dataset_name="lineage-token-output",
        storage_path="/managed/lineage-token-output.jsonl",
        dataset_type="embedding_universal",
        usage="train",
        model_type=["embedding"],
        source_dataset_id=None,
        relation_type="training_generated",
        file_format="jsonl",
        num_rows=1,
        file_size=10,
        user_id="user-1",
    ) == "lineage-token-output"

    with Session(engine) as session:
        edge = session.exec(
            select(DatasetLineageEdgeDB).where(
                DatasetLineageEdgeDB.to_dataset_id == "lineage-token-output"
            )
        ).one()
        assert edge.op_params is None
        assert run_token not in json.dumps(edge.to_dict(), sort_keys=True)


def test_blocked_old_registration_cannot_publish_after_recovery_and_restart(
    generation_database,
    monkeypatch,
):
    generation_service, engine = generation_database
    publication_module, publication_service = _publication_service_for_engine(
        engine,
        monkeypatch,
    )
    del publication_module
    task_id = "blocked-old-registration"
    old_token = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=old_token,
    )
    entered = threading.Event()
    release = threading.Event()
    results = []

    def old_registration():
        entered.set()
        assert release.wait(5)
        results.append(
            publication_service.stage_dataset(
                task_id=task_id,
                expected_run_token=old_token,
                dataset_id="old-staging-output",
                dataset_name="old-staging-output",
                storage_path="/managed/old-staging-output.jsonl",
                dataset_type="embedding_universal",
                usage="train",
                model_type=["embedding", "rerank"],
                source_dataset_id=None,
                relation_type="training_generated",
                file_format="jsonl",
                num_rows=1,
                file_size=10,
                user_id="user-1",
            )
        )

    worker = threading.Thread(target=old_registration)
    worker.start()
    assert entered.wait(5)

    recovery = generation_service.claim_orphan_recovery(
        task_id,
        expected_status=GenerationStatus.PUBLISHING,
        expected_run_token=old_token,
    )
    assert recovery is not None
    recovery_token = recovery["run_token"]
    assert generation_service.finish_orphan_recovery(
        task_id,
        expected_run_token=recovery_token,
        terminal_status=GenerationStatus.FAILED,
        error_message="publication worker was orphaned",
    ) is True
    assert generation_service.update_status(
        task_id,
        GenerationStatus.PENDING,
        expected_run_token=recovery_token,
    ) is True
    restarted = generation_service.get_task_raw(task_id)
    assert restarted["run_token"] not in {old_token, recovery_token}

    release.set()
    worker.join(5)
    assert worker.is_alive() is False
    assert results == [None]

    with Session(engine) as session:
        assert session.exec(
            select(DatasetDB).where(
                DatasetDB.source_task_id == task_id,
                DatasetDB.generation_run_token == old_token,
            )
        ).all() == []
        assert session.exec(select(DatasetAssetDB)).all() == []
        assert session.exec(select(DatasetLineageEdgeDB)).all() == []
        assert session.exec(
            select(MilvusCollectionDB).where(
                MilvusCollectionDB.generation_run_token == old_token
            )
        ).all() == []
        assert session.exec(
            select(CollectionDatasetLinkDB).where(
                CollectionDatasetLinkDB.generation_run_token == old_token
            )
        ).all() == []


def test_staging_compensation_failure_never_restores_ready_partial(
    generation_database,
    monkeypatch,
):
    _generation_service, engine = generation_database
    publication_module, publication_service = _publication_service_for_engine(
        engine,
        monkeypatch,
    )
    task_id = "staging-compensation-failure"
    run_token = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )
    _source_id, output_id, qa_id = _add_staging_publication_rows(
        engine,
        task_id,
        run_token,
    )
    real_delete = publication_module.delete

    def fail_lineage_delete(entity):
        if entity is DatasetLineageEdgeDB:
            raise RuntimeError("lineage cleanup failed")
        return real_delete(entity)

    monkeypatch.setattr(publication_module, "delete", fail_lineage_delete)

    assert publication_service.compensate_attempt(
        task_id=task_id,
        expected_run_token=run_token,
        user_id="user-1",
    ) is False

    with Session(engine) as session:
        datasets = session.exec(
            select(DatasetDB).where(
                DatasetDB.dataset_id.in_((output_id, qa_id))
            )
        ).all()
        assert {dataset.status for dataset in datasets} == {"staging"}
        assert len(
            session.exec(
                select(DatasetAssetDB).where(
                    DatasetAssetDB.dataset_id.in_((output_id, qa_id))
                )
            ).all()
        ) == 2
        assert len(
            session.exec(
                select(DatasetLineageEdgeDB).where(
                    DatasetLineageEdgeDB.to_dataset_id.in_((output_id, qa_id))
                )
            ).all()
        ) == 2


def test_publication_activates_all_datasets_links_and_completion_atomically(
    generation_database,
):
    service, engine = generation_database
    task_id = "atomic-publication"
    run_token = "33333333-3333-4333-8333-333333333333"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )
    source_id, output_id, qa_id = _add_staging_publication_rows(
        engine,
        task_id,
        run_token,
    )

    complete_publication = getattr(service, "complete_publication", None)
    assert complete_publication is not None
    assert complete_publication(
        task_id,
        expected_run_token=run_token,
        dataset_bindings={
            "output_dataset_id": output_id,
            "qa_dataset_id": qa_id,
        },
        milvus_registration={
            "collection_name": "attempt-collection",
            "dataset_id": source_id,
            "dataset_name": "source-ready",
            "user_id": "user-1",
        },
    ) is True

    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        datasets = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id.in_((output_id, qa_id)))
        ).all()
        collection = session.exec(
            select(MilvusCollectionDB).where(
                MilvusCollectionDB.collection_name == "attempt-collection"
            )
        ).one()
        link = session.exec(
            select(CollectionDatasetLinkDB).where(
                CollectionDatasetLinkDB.collection_name == "attempt-collection",
                CollectionDatasetLinkDB.dataset_id == source_id,
            )
        ).one()

        assert task.status == GenerationStatus.COMPLETED
        assert task.output_dataset_id == output_id
        assert task.qa_dataset_id == qa_id
        assert {dataset.status for dataset in datasets} == {
            DatasetStatus.READY.value
        }
        assert collection.generation_run_token == run_token
        assert link.generation_run_token == run_token


def test_publication_missing_asset_leaves_every_product_staging(
    generation_database,
):
    service, engine = generation_database
    task_id = "atomic-publication-missing-asset"
    run_token = "44444444-4444-4444-8444-444444444444"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )
    _source_id, output_id, qa_id = _add_staging_publication_rows(
        engine,
        task_id,
        run_token,
    )
    with Session(engine) as session:
        missing_asset = session.exec(
            select(DatasetAssetDB).where(DatasetAssetDB.dataset_id == qa_id)
        ).one()
        session.delete(missing_asset)
        session.commit()

    complete_publication = getattr(service, "complete_publication", None)
    assert complete_publication is not None
    assert complete_publication(
        task_id,
        expected_run_token=run_token,
        dataset_bindings={
            "output_dataset_id": output_id,
            "qa_dataset_id": qa_id,
        },
    ) is False

    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        datasets = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id.in_((output_id, qa_id)))
        ).all()
        assert task.status == GenerationStatus.PUBLISHING
        assert task.output_dataset_id is None
        assert task.qa_dataset_id is None
        assert {dataset.status for dataset in datasets} == {"staging"}


@pytest.mark.parametrize(
    ("deletion_owner", "sync_task_id"),
    (
        ("delete-owner", None),
        (None, "sync-owner"),
    ),
)
def test_publication_rejects_active_collection_with_exclusive_owner(
    generation_database,
    deletion_owner,
    sync_task_id,
):
    service, engine = generation_database
    task_id = f"owned-active-collection-{deletion_owner or sync_task_id}"
    run_token = "77777777-7777-4777-8777-777777777777"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )
    source_id, output_id, qa_id = _add_staging_publication_rows(
        engine,
        task_id,
        run_token,
    )
    with Session(engine) as session:
        session.add(
            MilvusCollectionDB(
                collection_name="owned-active-collection",
                status="active",
                deletion_owner=deletion_owner,
                sync_task_id=sync_task_id,
                user_id="user-1",
            )
        )
        session.commit()

    assert service.complete_publication(
        task_id,
        expected_run_token=run_token,
        dataset_bindings={
            "output_dataset_id": output_id,
            "qa_dataset_id": qa_id,
        },
        milvus_registration={
            "collection_name": "owned-active-collection",
            "dataset_id": source_id,
            "user_id": "user-1",
        },
    ) is False

    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id
            )
        ).one()
        staged = session.exec(
            select(DatasetDB).where(
                DatasetDB.dataset_id.in_((output_id, qa_id))
            )
        ).all()
        assert task.status == GenerationStatus.PUBLISHING
        assert {dataset.status for dataset in staged} == {
            DatasetStatus.STAGING.value
        }
        assert session.exec(
            select(CollectionDatasetLinkDB).where(
                CollectionDatasetLinkDB.collection_name
                == "owned-active-collection"
            )
        ).first() is None


def test_publication_reuses_existing_link_without_seizing_its_provenance(
    generation_database,
):
    service, engine = generation_database
    task_id = "link-owner-conflict"
    run_token = "88888888-8888-4888-8888-888888888888"
    old_token = "99999999-9999-4999-8999-999999999999"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )
    source_id, output_id, qa_id = _add_staging_publication_rows(
        engine,
        task_id,
        run_token,
    )
    with Session(engine) as session:
        session.add(
            MilvusCollectionDB(
                collection_name="attempt-owned-active-collection",
                status="active",
                generation_task_id="old-generation-attempt",
                generation_run_token=old_token,
                user_id="user-1",
            )
        )
        session.add(
            CollectionDatasetLinkDB(
                collection_name="attempt-owned-active-collection",
                dataset_id=source_id,
                task_id="old-generation-attempt",
                generation_run_token=old_token,
            )
        )
        session.commit()

    assert service.complete_publication(
        task_id,
        expected_run_token=run_token,
        dataset_bindings={
            "output_dataset_id": output_id,
            "qa_dataset_id": qa_id,
        },
        milvus_registration={
            "collection_name": "attempt-owned-active-collection",
            "dataset_id": source_id,
            "user_id": "user-1",
        },
    ) is True

    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task_id
            )
        ).one()
        link = session.exec(
            select(CollectionDatasetLinkDB).where(
                CollectionDatasetLinkDB.collection_name
                == "attempt-owned-active-collection",
                CollectionDatasetLinkDB.dataset_id == source_id,
            )
        ).one()
        assert task.status == GenerationStatus.COMPLETED
        assert {
            row.status
            for row in session.exec(
                select(DatasetDB).where(
                    DatasetDB.dataset_id.in_((output_id, qa_id))
                )
            ).all()
        } == {DatasetStatus.READY.value}
        assert link.task_id == "old-generation-attempt"
        assert link.generation_run_token == old_token


def test_recovery_compensates_products_committed_by_blocked_old_callback(
    generation_database,
    monkeypatch,
):
    generation_service, engine = generation_database
    _publication_module, publication_service = _publication_service_for_engine(
        engine,
        monkeypatch,
    )
    task_id = "post-commit-blocked-registration"
    run_token = "abababab-abab-4bab-8bab-abababababab"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.PUBLISHING,
        run_token=run_token,
    )
    staged = threading.Event()
    release = threading.Event()
    results = []

    def old_callback():
        results.append(
            publication_service.stage_dataset(
                task_id=task_id,
                expected_run_token=run_token,
                dataset_id="post-commit-old-staging",
                dataset_name="post-commit-old-staging",
                storage_path="/managed/post-commit-old-staging.jsonl",
                dataset_type="embedding_universal",
                usage="train",
                model_type=["embedding", "rerank"],
                source_dataset_id=None,
                relation_type="training_generated",
                file_format="jsonl",
                num_rows=1,
                file_size=10,
                user_id="user-1",
            )
        )
        staged.set()
        assert release.wait(5)

    worker = threading.Thread(target=old_callback)
    worker.start()
    assert staged.wait(5)
    assert results == ["post-commit-old-staging"]

    recovery = generation_service.claim_orphan_recovery(
        task_id,
        expected_status=GenerationStatus.PUBLISHING,
        expected_run_token=run_token,
    )
    assert recovery is not None
    assert publication_service.compensate_attempt(
        task_id=task_id,
        expected_run_token=run_token,
        recovery_run_token=recovery["run_token"],
        user_id="user-1",
    ) is True
    release.set()
    worker.join(5)
    assert not worker.is_alive()

    with Session(engine) as session:
        assert session.exec(
            select(DatasetDB).where(
                DatasetDB.generation_run_token == run_token
            )
        ).all() == []
        assert session.exec(
            select(DatasetAssetDB).where(
                DatasetAssetDB.dataset_id == "post-commit-old-staging"
            )
        ).all() == []
        assert session.exec(
            select(DatasetLineageEdgeDB).where(
                DatasetLineageEdgeDB.to_dataset_id
                == "post-commit-old-staging"
            )
        ).all() == []
        assert session.exec(
            select(CollectionDatasetLinkDB).where(
                CollectionDatasetLinkDB.dataset_id
                == "post-commit-old-staging"
            )
        ).all() == []


def test_restart_cleanup_is_exactly_scoped_to_previous_attempt_staging(
    generation_database,
    monkeypatch,
):
    service, engine = generation_database
    _publication_module, publication_service = _publication_service_for_engine(
        engine,
        monkeypatch,
    )
    task_id = "restart-staging-cleanup"
    previous_token = "55555555-5555-4555-8555-555555555555"
    other_token = "66666666-6666-4666-8666-666666666666"
    _add_generation_task(
        engine,
        task_id,
        GenerationStatus.FAILED,
        run_token=previous_token,
    )
    _source_id, output_id, qa_id = _add_staging_publication_rows(
        engine,
        task_id,
        previous_token,
    )
    with Session(engine) as session:
        foreign = DatasetDB(
            dataset_id="foreign-attempt-staging",
            dataset_name="foreign-attempt-staging",
            storage_path="/managed/foreign-attempt-staging.jsonl",
            status=DatasetStatus.STAGING.value,
            source_type="generated",
            source_task_type="generation",
            source_task_id=task_id,
            user_id="user-1",
        )
        foreign.generation_run_token = other_token
        session.add(foreign)
        session.commit()

    assert publication_service.has_staging_products(
        task_id,
        expected_run_token=previous_token,
    ) is True
    assert publication_service.has_staging_products(
        task_id,
        expected_run_token="absent-attempt-token",
    ) is False
    assert publication_service.compensate_attempt(
        task_id=task_id,
        expected_run_token=previous_token,
        user_id="user-1",
    ) is True
    assert publication_service.has_staging_products(
        task_id,
        expected_run_token=previous_token,
    ) is False
    assert publication_service.has_staging_products(
        task_id,
        expected_run_token=other_token,
    ) is True
    # A different orphaned attempt remains a separate repair fence: exact
    # cleanup must neither delete it nor silently rotate past it.
    assert service.update_status(
        task_id,
        GenerationStatus.PENDING,
        expected_run_token=previous_token,
    ) is False

    with Session(engine) as session:
        assert session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == output_id)
        ).first() is None
        assert session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == qa_id)
        ).first() is None
        foreign = session.exec(
            select(DatasetDB).where(
                DatasetDB.dataset_id == "foreign-attempt-staging"
            )
        ).one()
        session.delete(foreign)
        session.commit()

    assert service.update_status(
        task_id,
        GenerationStatus.PENDING,
        expected_run_token=previous_token,
    ) is True


@pytest.mark.parametrize(
    "status",
    (
        GenerationStatus.PENDING,
        GenerationStatus.RUNNING,
        GenerationStatus.PUBLISHING,
    ),
)
def test_legacy_null_orphan_recovery_has_exactly_one_atomic_claim_winner(
    generation_database,
    status,
):
    service, engine = generation_database
    task_id = f"legacy-null-{status}"
    _add_generation_task(engine, task_id, status, run_token=None)
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def claim():
        try:
            barrier.wait(timeout=5)
            claim_recovery = getattr(service, "claim_orphan_recovery", None)
            results.append(
                claim_recovery(
                    task_id,
                    expected_status=status,
                    expected_run_token=None,
                )
                if claim_recovery
                else None
            )
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert errors == []
    winners = [result for result in results if result]
    assert len(winners) == 1
    claimed_token = winners[0]["run_token"]
    assert str(UUID(claimed_token)) == claimed_token
    assert winners[0]["source_status"] == status
    task = service.get_task_raw(task_id)
    assert task["run_token"] == claimed_token
    assert task["status"] == "recovering"


@pytest.mark.parametrize(
    "status",
    (
        GenerationStatus.PENDING,
        GenerationStatus.RUNNING,
        GenerationStatus.PUBLISHING,
    ),
)
def test_nonnull_orphan_attempt_has_one_claim_winner_without_token_rotation(
    generation_database,
    status,
):
    service, engine = generation_database
    task_id = f"nonnull-orphan-{status}"
    run_token = "12121212-1212-4212-8212-121212121212"
    _add_generation_task(engine, task_id, status, run_token=run_token)
    barrier = threading.Barrier(2)
    results = []

    def claim():
        barrier.wait(timeout=5)
        results.append(
            service.claim_orphan_recovery(
                task_id,
                expected_status=status,
                expected_run_token=run_token,
            )
        )

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert [result for result in results if result] == [
        {"run_token": run_token, "source_status": status}
    ]
    task = service.get_task_raw(task_id)
    assert task["status"] == GenerationStatus.RECOVERING
    assert task["run_token"] == run_token


def test_startup_retries_recovering_publication_until_cleanup_succeeds(
    monkeypatch,
):
    from train_factory.api import server
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )
    from train_factory.storage.services.generation_publication_service import (
        generation_publication_service,
    )
    from train_factory.storage.services.generation_task_service import (
        generation_task_service,
    )

    task_id = "retry-recovering-publication"
    run_token = "13131313-1313-4313-8313-131313131313"
    state = {"status": GenerationStatus.RECOVERING}
    compensations = []
    finishes = []

    def get_all_tasks(*, status, limit, offset):
        del limit, offset
        if status == state["status"] == GenerationStatus.RECOVERING:
            return ([{"task_id": task_id, "status": status}], 1)
        return ([], 0)

    monkeypatch.setattr(
        generation_task_service,
        "get_all_tasks",
        get_all_tasks,
    )
    monkeypatch.setattr(
        generation_task_service,
        "get_task_raw",
        lambda _task_id: {
            "task_id": task_id,
            "status": state["status"],
            "run_token": run_token,
            "user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        generation_task_service,
        "claim_orphan_recovery",
        lambda *_args, **_kwargs: pytest.fail(
            "an already-RECOVERING task must be resumed, not reclaimed"
        ),
    )

    def finish(_task_id, **kwargs):
        finishes.append(kwargs)
        state["status"] = kwargs["terminal_status"]
        return True

    monkeypatch.setattr(
        generation_task_service,
        "finish_orphan_recovery",
        finish,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_generation_and_restore_batches",
        lambda *_args, **_kwargs: {
            "tracking_found": False,
            "recovered": False,
        },
    )

    def compensate(**kwargs):
        compensations.append(kwargs)
        return len(compensations) > 1

    monkeypatch.setattr(
        generation_publication_service,
        "compensate_attempt",
        compensate,
    )

    server.cleanup_orphan_generation_tasks()
    assert state["status"] == GenerationStatus.RECOVERING
    assert finishes == []

    server.cleanup_orphan_generation_tasks()
    assert state["status"] == GenerationStatus.FAILED
    assert len(compensations) == 2
    assert all(
        call["expected_run_token"] == run_token
        and call["recovery_run_token"] == run_token
        for call in compensations
    )
    assert finishes == [
        {
            "expected_run_token": run_token,
            "terminal_status": GenerationStatus.FAILED,
            "error_message": (
                "Dataset publication was interrupted by server restart. "
                "Restart the task to retry safely."
            ),
        }
    ]


def test_startup_fails_restarting_attempt_after_exact_directory_cleanup(
    tmp_path,
    monkeypatch,
):
    from train_factory.api import server
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )
    from train_factory.storage.services.generation_publication_service import (
        generation_publication_service,
    )
    from train_factory.storage.services.generation_task_service import (
        generation_task_service,
    )

    task_id = "startup-restarting-attempt"
    run_token = "19191919-1919-4919-8919-191919191919"
    old_refs = {
        "output_path": str(tmp_path / "old-attempt" / "generated.jsonl"),
        "output_dataset_id": "old-output-dataset",
        "qa_output_path": str(tmp_path / "old-attempt" / "qa.jsonl"),
        "qa_dataset_id": "old-qa-dataset",
    }
    state = {
        "task_id": task_id,
        "status": "restarting",
        "run_token": run_token,
        "user_id": "user-1",
        **old_refs,
    }
    attempt_dir = (
        tmp_path
        / f"generation_{task_id}"
        / generation_routes.generation_attempt_key(task_id, run_token)
    )
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "checkpoint.tmp").write_text("partial", encoding="utf-8")
    sibling_dir = tmp_path / f"generation_{task_id}" / "other-attempt"
    sibling_dir.mkdir(parents=True)
    (sibling_dir / "keep.txt").write_text("keep", encoding="utf-8")
    claims = []
    finishes = []
    route_finish_results = []

    def get_all_tasks(*, status, limit, offset):
        del limit, offset
        if status == state["status"] == "restarting":
            return ([dict(state)], 1)
        return ([], 0)

    monkeypatch.setattr(generation_routes, "GENERATION_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(generation_task_service, "get_all_tasks", get_all_tasks)
    monkeypatch.setattr(
        generation_task_service,
        "get_task_raw",
        lambda _task_id: dict(state),
    )
    def claim_restart(_task_id, *, expected_status, expected_run_token):
        claims.append((expected_status, expected_run_token))
        assert state["status"] == "restarting"
        state["status"] = GenerationStatus.RECOVERING
        route_finish_results.append(state["status"] == "restarting")
        return {
            "run_token": run_token,
            "source_status": "restarting",
        }

    monkeypatch.setattr(
        generation_task_service,
        "claim_orphan_recovery",
        claim_restart,
    )

    def finish_recovery(
        _task_id,
        *,
        expected_run_token,
        terminal_status,
        error_message,
    ):
        finishes.append(
            (expected_run_token, terminal_status, error_message)
        )
        assert state["status"] == GenerationStatus.RECOVERING
        state["status"] = terminal_status
        return True

    monkeypatch.setattr(
        generation_task_service,
        "finish_orphan_recovery",
        finish_recovery,
    )
    monkeypatch.setattr(
        generation_task_service,
        "fail_restart",
        lambda *_args, **_kwargs: pytest.fail(
            "restart cleanup must be owned through RECOVERING"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        external_sync_service,
        "fail_generation_and_restore_batches",
        lambda *_args, **_kwargs: pytest.fail(
            "restart checkpoint cleanup is not sync publication recovery"
        ),
    )
    monkeypatch.setattr(
        generation_publication_service,
        "compensate_attempt",
        lambda **_kwargs: pytest.fail(
            "RESTARTING has no publication staging to compensate"
        ),
    )

    server.cleanup_orphan_generation_tasks()

    assert state["status"] == GenerationStatus.FAILED
    assert claims == [("restarting", run_token)]
    assert route_finish_results == [False]
    assert len(finishes) == 1
    assert finishes[0][0] == run_token
    assert finishes[0][1] == GenerationStatus.FAILED
    assert "restart" in finishes[0][2].lower()
    assert all(state[field] == value for field, value in old_refs.items())
    assert not attempt_dir.exists()
    assert (sibling_dir / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_startup_restarting_snapshot_loser_never_deletes_live_attempt(
    tmp_path,
    monkeypatch,
):
    from train_factory.api import server
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )
    from train_factory.storage.services.generation_task_service import (
        generation_task_service,
    )

    task_id = "startup-restarting-lost-race"
    run_token = "20202020-2020-4020-8020-202020202020"
    state = {"status": "restarting"}
    stale_snapshot = {
        "task_id": task_id,
        "status": "restarting",
        "run_token": run_token,
        "user_id": "user-1",
    }
    attempt_dir = (
        tmp_path
        / f"generation_{task_id}"
        / generation_routes.generation_attempt_key(task_id, run_token)
    )
    attempt_dir.mkdir(parents=True)
    checkpoint = attempt_dir / "generated.jsonl"
    checkpoint.write_text("live", encoding="utf-8")
    claims = []

    def get_all_tasks(*, status, limit, offset):
        del limit, offset
        if status == "restarting":
            return ([dict(stale_snapshot)], 1)
        return ([], 0)

    def get_task_raw(_task_id):
        # The route wins after startup's list snapshot but before its claim.
        state["status"] = GenerationStatus.PENDING
        return dict(stale_snapshot)

    def lose_claim(_task_id, *, expected_status, expected_run_token):
        claims.append((expected_status, expected_run_token))
        assert state["status"] == GenerationStatus.PENDING
        return None

    monkeypatch.setattr(generation_routes, "GENERATION_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(generation_task_service, "get_all_tasks", get_all_tasks)
    monkeypatch.setattr(generation_task_service, "get_task_raw", get_task_raw)
    monkeypatch.setattr(
        generation_task_service,
        "claim_orphan_recovery",
        lose_claim,
    )
    monkeypatch.setattr(
        generation_task_service,
        "fail_restart",
        lambda *_args, **_kwargs: pytest.fail(
            "a startup claim loser cannot mutate the live restart"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_pending_generations",
        lambda **_kwargs: ([], 0),
    )

    server.cleanup_orphan_generation_tasks()

    assert state["status"] == GenerationStatus.PENDING
    assert claims == [("restarting", run_token)]
    assert checkpoint.read_text(encoding="utf-8") == "live"
