from contextlib import contextmanager
import importlib
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncGenerationDB,
)
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.milvus_collection_entity import (
    MilvusCollectionDB,
)


@pytest.fixture
def generation_service(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'generation-run-token.db'}")
    SQLModel.metadata.create_all(
        engine,
        tables=[
            DatasetDB.__table__,
            ExternalSyncGenerationDB.__table__,
            GenerationTaskDB.__table__,
            MilvusCollectionDB.__table__,
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


def _create_task(service, task_id: str):
    return service.create_task(
        task_id=task_id,
        task_name=task_id,
        input_path="/managed/input.jsonl",
        llm_config={},
        steps_config={},
    )


def _raw_task(engine, task_id: str) -> GenerationTaskDB:
    with Session(engine) as session:
        return session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()


def _valid_uuid(value: str) -> str:
    return str(UUID(value))


def test_create_persists_uuid_run_token_but_public_dict_hides_it(generation_service):
    service, engine = generation_service

    public_task = _create_task(service, "new-token")
    raw_task = _raw_task(engine, "new-token")

    assert "run_token" not in public_task
    assert _valid_uuid(raw_task.run_token) == raw_task.run_token
    assert "run_token" not in raw_task.to_dict()


def test_legacy_pending_claim_generates_and_returns_durable_token(generation_service):
    service, engine = generation_service
    with Session(engine) as session:
        session.add(
            GenerationTaskDB(
                task_id="legacy-claim",
                task_name="legacy-claim",
                input_path="/managed/input.jsonl",
                llm_config={},
                steps_config={},
                status=GenerationStatus.PENDING,
                run_token=None,
            )
        )
        session.commit()

    claimed_token = service.claim_running("legacy-claim")

    raw_task = _raw_task(engine, "legacy-claim")
    assert _valid_uuid(claimed_token) == claimed_token
    assert raw_task.status == GenerationStatus.RUNNING
    assert raw_task.run_token == claimed_token


def test_explicit_legacy_null_token_cannot_claim_uuid_attempt(generation_service):
    service, engine = generation_service
    _create_task(service, "uuid-claim")

    assert service.claim_running(
        "uuid-claim",
        expected_run_token=None,
    ) is None
    raw_task = _raw_task(engine, "uuid-claim")
    assert raw_task.status == GenerationStatus.PENDING


def test_stop_preserves_token_and_restart_rotates_it_atomically(generation_service):
    service, engine = generation_service
    _create_task(service, "rotate-token")
    first_token = service.claim_running("rotate-token")

    assert service.update_status(
        "rotate-token",
        GenerationStatus.STOPPING,
        expected_run_token=first_token,
    ) is True
    assert service.update_status(
        "rotate-token",
        GenerationStatus.STOPPED,
        expected_run_token=first_token,
    ) is True
    assert _raw_task(engine, "rotate-token").run_token == first_token

    assert service.update_status(
        "rotate-token",
        GenerationStatus.PENDING,
        expected_run_token=first_token,
    ) is True
    restarted = _raw_task(engine, "rotate-token")
    second_token = restarted.run_token
    assert restarted.status == GenerationStatus.PENDING
    assert _valid_uuid(second_token) == second_token
    assert second_token != first_token

    assert service.claim_running("rotate-token") == second_token


def test_restart_atomically_clears_previous_attempt_artifacts(generation_service):
    service, engine = generation_service
    _create_task(service, "clear-artifacts")
    first_token = _raw_task(engine, "clear-artifacts").run_token
    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == "clear-artifacts"
            )
        ).one()
        task.status = GenerationStatus.STOPPED
        task.output_path = "/managed/old-output.jsonl"
        task.output_dataset_id = "old-output-dataset"
        task.qa_output_path = "/managed/old-qa.jsonl"
        task.qa_filtered_path = "/managed/old-filtered.jsonl"
        task.qa_dataset_id = "old-qa-dataset"
        task.qa_filtered_dataset_id = "old-filtered-dataset"
        task.deep_eval_path = "/managed/old-deep.jsonl"
        task.deep_eval_dataset_id = "old-deep-dataset"
        task.filter_stats = {"kept": 4}
        task.progress = 80.0
        task.total_docs = 10
        task.processed_docs = 8
        task.output_sample_count = 4
        task.error_message = "old attempt failed"
        task.started_at = task.created_at
        task.completed_at = task.created_at
        task.milvus_collection = "preserved-collection"
        session.add(
            MilvusCollectionDB(
                collection_name="preserved-collection",
                user_id=None,
                status="active",
            )
        )
        session.add(task)
        session.commit()

    assert service.update_status(
        "clear-artifacts",
        GenerationStatus.PENDING,
        expected_run_token=first_token,
    )

    restarted = _raw_task(engine, "clear-artifacts")
    assert restarted.run_token != first_token
    assert restarted.output_path is None
    assert restarted.output_dataset_id is None
    assert restarted.qa_output_path is None
    assert restarted.qa_filtered_path is None
    assert restarted.qa_dataset_id is None
    assert restarted.qa_filtered_dataset_id is None
    assert restarted.deep_eval_path is None
    assert restarted.deep_eval_dataset_id is None
    assert restarted.filter_stats is None
    assert restarted.progress == 0
    assert restarted.total_docs == 0
    assert restarted.processed_docs == 0
    assert restarted.output_sample_count == 0
    assert restarted.error_message is None
    assert restarted.started_at is None
    assert restarted.completed_at is None
    assert restarted.milvus_collection == "preserved-collection"


def test_restart_reservation_preserves_artifacts_and_fences_consumers_until_finish(
    generation_service,
):
    service, engine = generation_service
    task_id = "restart-reservation"
    _create_task(service, task_id)
    old_token = _raw_task(engine, task_id).run_token
    planned_token = str(uuid4())
    dataset_refs = {
        "output_dataset_id": "old-output-dataset",
        "qa_dataset_id": "old-qa-dataset",
        "qa_filtered_dataset_id": "old-filtered-dataset",
        "deep_eval_dataset_id": "old-deep-dataset",
    }
    path_refs = {
        "output_path": "/managed/old/generated.jsonl",
        "qa_output_path": "/managed/old/qa.jsonl",
        "qa_filtered_path": "/managed/old/qa-filtered.jsonl",
        "deep_eval_path": "/managed/old/deep.jsonl",
    }
    with Session(engine) as session:
        session.add_all(
            [
                DatasetDB(
                    dataset_id=dataset_id,
                    dataset_name=f"dataset-{index}",
                    status="ready",
                    source_task_type="generation",
                    source_task_id=task_id,
                )
                for index, dataset_id in enumerate(dataset_refs.values())
            ]
        )
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        task.status = GenerationStatus.FAILED
        for field, value in {**dataset_refs, **path_refs}.items():
            setattr(task, field, value)
        session.add(task)
        session.commit()

    reserved_token = service.begin_restart(
        task_id,
        expected_run_token=old_token,
        new_run_token=planned_token,
    )

    assert reserved_token == planned_token
    reserved = _raw_task(engine, task_id)
    assert reserved.status == GenerationStatus.RESTARTING
    assert reserved.run_token == planned_token
    for field, value in {**dataset_refs, **path_refs}.items():
        assert getattr(reserved, field) == value
    for dataset_id in dataset_refs.values():
        assert service.list_active_dataset_consumers([dataset_id]) == [task_id]

    new_output_path = "/managed/current/generated.jsonl"
    assert service.finish_restart(
        task_id,
        expected_run_token=planned_token,
        output_path=new_output_path,
    ) is True
    restarted = _raw_task(engine, task_id)
    assert restarted.status == GenerationStatus.PENDING
    assert restarted.run_token == planned_token
    assert restarted.output_path == new_output_path
    assert restarted.output_dataset_id is None
    assert restarted.qa_output_path is None
    assert restarted.qa_dataset_id is None
    assert restarted.qa_filtered_path is None
    assert restarted.qa_filtered_dataset_id is None
    assert restarted.deep_eval_path is None
    assert restarted.deep_eval_dataset_id is None


def test_restart_first_locks_all_ready_artifact_rows_without_status_filter(
    generation_service,
    monkeypatch,
):
    service, engine = generation_service
    task_id = "restart-first-artifact-lock"
    _create_task(service, task_id)
    old_token = _raw_task(engine, task_id).run_token
    artifact_ids = [f"artifact-{index}" for index in range(4)]
    with Session(engine) as session:
        session.add_all(
            [
                DatasetDB(
                    dataset_id=dataset_id,
                    dataset_name=dataset_id,
                    status="ready",
                    source_task_type="generation",
                    source_task_id=task_id,
                )
                for dataset_id in artifact_ids
            ]
        )
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        task.status = GenerationStatus.FAILED
        (
            task.output_dataset_id,
            task.qa_dataset_id,
            task.qa_filtered_dataset_id,
            task.deep_eval_dataset_id,
        ) = artifact_ids
        session.add(task)
        session.commit()

    statements = []
    service_module = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )

    class RecordingSession:
        def __init__(self, session):
            self._session = session

        def __getattr__(self, name):
            return getattr(self._session, name)

        def exec(self, statement, *args, **kwargs):
            statements.append(statement)
            return self._session.exec(statement, *args, **kwargs)

    @contextmanager
    def recording_session():
        with Session(engine) as session:
            yield RecordingSession(session)

    monkeypatch.setattr(service_module, "get_session", recording_session)

    assert service.begin_restart(
        task_id,
        expected_run_token=old_token,
        new_run_token=str(uuid4()),
    ) is not None

    dataset_locks = [
        statement
        for statement in statements
        if "FROM datasets" in str(statement)
        and getattr(statement, "_for_update_arg", None) is not None
    ]
    assert dataset_locks
    lock_sql = str(dataset_locks[0])
    where_sql = lock_sql[lock_sql.index("WHERE ") :]
    assert "datasets.status" not in where_sql
    assert "datasets.dataset_id IN" in lock_sql
    assert "datasets.source_task_type" in lock_sql
    assert "datasets.source_task_id" in lock_sql


def test_delete_first_fence_rejects_restart_without_rotating_attempt(
    generation_service,
):
    service, engine = generation_service
    task_id = "delete-first-artifact-fence"
    dataset_id = "deleting-output-dataset"
    _create_task(service, task_id)
    old_token = _raw_task(engine, task_id).run_token
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id=dataset_id,
                dataset_name=dataset_id,
                status="deleting",
                deletion_owner=f"dataset:{dataset_id}",
                source_task_type="generation",
                source_task_id=task_id,
            )
        )
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        task.status = GenerationStatus.FAILED
        task.output_dataset_id = dataset_id
        session.add(task)
        session.commit()

    assert service.begin_restart(
        task_id,
        expected_run_token=old_token,
        new_run_token=str(uuid4()),
    ) is None
    unchanged = _raw_task(engine, task_id)
    assert unchanged.status == GenerationStatus.FAILED
    assert unchanged.run_token == old_token


def test_startup_claim_fences_restarting_attempt_before_cleanup(generation_service):
    service, engine = generation_service
    task_id = "restart-startup-claim"
    _create_task(service, task_id)
    old_token = _raw_task(engine, task_id).run_token
    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(GenerationTaskDB.task_id == task_id)
        ).one()
        task.status = GenerationStatus.FAILED
        task.output_path = "/managed/old/generated.jsonl"
        session.add(task)
        session.commit()
    planned_token = str(uuid4())
    assert service.begin_restart(
        task_id,
        expected_run_token=old_token,
        new_run_token=planned_token,
    ) == planned_token

    claim = service.claim_orphan_recovery(
        task_id,
        expected_status=GenerationStatus.RESTARTING,
        expected_run_token=planned_token,
    )

    assert claim == {
        "run_token": planned_token,
        "source_status": GenerationStatus.RESTARTING,
    }
    assert _raw_task(engine, task_id).status == GenerationStatus.RECOVERING
    assert service.finish_restart(
        task_id,
        expected_run_token=planned_token,
        output_path="/managed/current/generated.jsonl",
    ) is False
    assert service.finish_orphan_recovery(
        task_id,
        expected_run_token=planned_token,
        terminal_status=GenerationStatus.FAILED,
        error_message="restart recovery completed",
    ) is True
    recovered = _raw_task(engine, task_id)
    assert recovered.status == GenerationStatus.FAILED
    assert recovered.output_path == "/managed/old/generated.jsonl"


@pytest.mark.parametrize(
    "active_status",
    [
        GenerationStatus.STOPPING,
        GenerationStatus.RECOVERING,
        GenerationStatus.RESTARTING,
    ],
)
def test_delete_task_rejects_transitional_attempt(
    generation_service,
    active_status,
):
    service, engine = generation_service
    _create_task(service, f"delete-{active_status}")
    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == f"delete-{active_status}"
            )
        ).one()
        task.status = active_status
        session.add(task)
        session.commit()

    assert service.delete_task(f"delete-{active_status}") is False
    assert _raw_task(engine, f"delete-{active_status}").status == active_status


def test_recovery_owner_requires_finish_primitive_for_terminal_transition(
    generation_service,
):
    service, engine = generation_service
    _create_task(service, "recovery-owner")
    run_token = _raw_task(engine, "recovery-owner").run_token
    with Session(engine) as session:
        task = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == "recovery-owner"
            )
        ).one()
        task.status = GenerationStatus.RECOVERING
        session.add(task)
        session.commit()

    assert service.update_status(
        "recovery-owner",
        GenerationStatus.FAILED,
        expected_run_token=run_token,
    ) is False
    assert _raw_task(engine, "recovery-owner").status == GenerationStatus.RECOVERING
    assert service.finish_orphan_recovery(
        "recovery-owner",
        expected_run_token=run_token,
        terminal_status=GenerationStatus.FAILED,
        error_message="startup recovery completed",
    ) is True
    assert _raw_task(engine, "recovery-owner").status == GenerationStatus.FAILED


def test_old_token_cannot_mutate_restarted_running_attempt(generation_service):
    service, engine = generation_service
    _create_task(service, "stale-mutations")
    old_token = service.claim_running("stale-mutations")
    assert service.update_status(
        "stale-mutations",
        GenerationStatus.STOPPING,
        expected_run_token=old_token,
    )
    assert service.update_status(
        "stale-mutations",
        GenerationStatus.STOPPED,
        expected_run_token=old_token,
    )
    assert service.update_status(
        "stale-mutations",
        GenerationStatus.PENDING,
        expected_run_token=old_token,
    )
    new_token = service.claim_running("stale-mutations")
    assert new_token != old_token

    assert service.update_progress(
        "stale-mutations",
        8,
        10,
        3,
        expected_status=GenerationStatus.RUNNING,
        expected_run_token=old_token,
    ) is False
    assert service.set_output(
        "stale-mutations",
        "/managed/stale-output.jsonl",
        3,
        "stale-output-dataset",
        expected_status=GenerationStatus.RUNNING,
        expected_run_token=old_token,
    ) is False
    assert service.set_qa_output(
        "stale-mutations",
        "/managed/stale-qa.jsonl",
        "stale-qa-dataset",
        expected_status=GenerationStatus.RUNNING,
        expected_run_token=old_token,
    ) is False
    assert service.set_filter_results(
        "stale-mutations",
        "stale-collection",
        {"kept": 3},
        expected_status=GenerationStatus.RUNNING,
        expected_run_token=old_token,
    ) is False
    assert service.set_deep_eval_output(
        "stale-mutations",
        "/managed/stale-eval.jsonl",
        "stale-eval-dataset",
        expected_status=GenerationStatus.RUNNING,
        expected_run_token=old_token,
    ) is False
    assert service.update_status(
        "stale-mutations",
        GenerationStatus.PUBLISHING,
        expected_run_token=old_token,
    ) is False

    current = _raw_task(engine, "stale-mutations")
    assert current.status == GenerationStatus.RUNNING
    assert current.run_token == new_token
    assert current.processed_docs == 0
    assert current.output_path is None
    assert current.qa_output_path is None
    assert current.filter_stats is None
    assert current.deep_eval_path is None


def test_current_token_can_publish_and_complete(generation_service):
    service, engine = generation_service
    _create_task(service, "current-publication")
    run_token = service.claim_running("current-publication")

    assert service.update_status(
        "current-publication",
        GenerationStatus.PUBLISHING,
        expected_run_token=run_token,
    ) is True
    assert service.set_output(
        "current-publication",
        "/managed/output.jsonl",
        2,
        "output-dataset",
        expected_status=GenerationStatus.PUBLISHING,
        expected_run_token=run_token,
    ) is True
    assert service.update_status(
        "current-publication",
        GenerationStatus.COMPLETED,
        expected_run_token=run_token,
    ) is True

    task = _raw_task(engine, "current-publication")
    assert task.status == GenerationStatus.COMPLETED
    assert task.run_token == run_token
    assert task.output_dataset_id == "output-dataset"
