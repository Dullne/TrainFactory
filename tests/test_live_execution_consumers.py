from contextlib import contextmanager
import importlib
from types import SimpleNamespace

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from train_factory.storage.entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationTaskDB,
)
from train_factory.storage.entities.generation_task_entity import GenerationTaskDB
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.services.background_task_admission_service import (
    BackgroundTaskAdmissionService,
)


deep_module = importlib.import_module(
    "train_factory.storage.services.deep_evaluation_task_service"
)
evaluation_module = importlib.import_module(
    "train_factory.storage.services.evaluation_task_service"
)
generation_module = importlib.import_module(
    "train_factory.storage.services.generation_task_service"
)
training_module = importlib.import_module(
    "train_factory.storage.services.training_task_service"
)


def _engine_for(*tables):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=list(tables))
    return engine


def _patch_session(monkeypatch, module, engine):
    @contextmanager
    def get_test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(module, "get_session", get_test_session)


def _patch_executing_ids(monkeypatch, module, ids_by_kind):
    monkeypatch.setattr(
        module,
        "background_task_admission_service",
        SimpleNamespace(
            get_executing_task_ids=lambda task_kind: set(
                ids_by_kind.get(task_kind, ())
            )
        ),
        raising=False,
    )


def test_get_executing_task_ids_returns_a_kind_scoped_snapshot(monkeypatch):
    service = BackgroundTaskAdmissionService(global_limit=8, per_user_limit=8)
    monkeypatch.setattr(service, "_active_task_owners", lambda: {})

    _, generation_lease = service.admit_execution(
        "generation",
        "generation-1",
        "user-1",
        lambda: True,
    )
    _, evaluation_lease = service.admit_execution(
        "evaluation",
        "evaluation-1",
        "user-1",
        lambda: True,
    )

    generation_ids = service.get_executing_task_ids("generation")

    assert generation_ids == {"generation-1"}
    assert service.get_executing_task_ids("evaluation") == {"evaluation-1"}
    generation_lease.release()
    assert generation_ids == {"generation-1"}
    assert service.get_executing_task_ids("generation") == set()
    evaluation_lease.release()


def test_generation_consumers_include_terminal_tasks_with_live_execution(monkeypatch):
    engine = _engine_for(GenerationTaskDB.__table__)
    with Session(engine) as session:
        session.add_all(
            [
                GenerationTaskDB(
                    task_id="generation-live",
                    task_name="live",
                    input_path="input.jsonl",
                    llm_config={},
                    steps_config={},
                    source_dataset_id="dataset-1",
                    milvus_collection="collection-1",
                    status="stopped",
                ),
                GenerationTaskDB(
                    task_id="generation-history",
                    task_name="history",
                    input_path="input.jsonl",
                    llm_config={},
                    steps_config={},
                    source_dataset_id="dataset-1",
                    milvus_collection="collection-1",
                    status="completed",
                ),
            ]
        )
        session.commit()
    _patch_session(monkeypatch, generation_module, engine)
    _patch_executing_ids(
        monkeypatch,
        generation_module,
        {"generation": {"generation-live"}},
    )

    service = generation_module.GenerationTaskService()

    assert service.list_active_dataset_consumers(["dataset-1"]) == [
        "generation-live"
    ]
    assert service.list_active_collection_consumers(["collection-1"]) == [
        "generation-live"
    ]


def test_evaluation_consumers_include_cancelled_task_with_live_execution(monkeypatch):
    engine = _engine_for(EvaluationTaskDB.__table__)
    with Session(engine) as session:
        session.add_all(
            [
                EvaluationTaskDB(
                    task_id="evaluation-live",
                    eval_framework=EvaluationFramework.MTEB,
                    dataset_configs=[{"dataset_id": "dataset-1"}],
                    status="cancelled",
                ),
                EvaluationTaskDB(
                    task_id="evaluation-history",
                    eval_framework=EvaluationFramework.MTEB,
                    dataset_configs=[{"dataset_id": "dataset-1"}],
                    status="completed",
                ),
            ]
        )
        session.commit()
    _patch_session(monkeypatch, evaluation_module, engine)
    _patch_executing_ids(
        monkeypatch,
        evaluation_module,
        {"evaluation": {"evaluation-live"}},
    )

    assert evaluation_module.EvaluationTaskService().list_active_dataset_consumers(
        ["dataset-1"], []
    ) == ["evaluation-live"]


def test_deep_evaluation_consumers_include_only_deepeval_live_execution(monkeypatch):
    engine = _engine_for(EvaluationTaskDB.__table__)
    worker_groups = {
        "retrieval_mode": "online",
        "milvus_collection": "collection-1",
    }
    with Session(engine) as session:
        session.add_all(
            [
                EvaluationTaskDB(
                    task_id="deep-live",
                    eval_framework=EvaluationFramework.DEEPEVAL,
                    worker_groups=worker_groups,
                    status="cancelled",
                ),
                EvaluationTaskDB(
                    task_id="mteb-live",
                    eval_framework=EvaluationFramework.MTEB,
                    worker_groups=worker_groups,
                    status="cancelled",
                ),
                EvaluationTaskDB(
                    task_id="deep-history",
                    eval_framework=EvaluationFramework.DEEPEVAL,
                    worker_groups=worker_groups,
                    status="completed",
                ),
            ]
        )
        session.commit()
    _patch_session(monkeypatch, deep_module, engine)
    _patch_executing_ids(
        monkeypatch,
        deep_module,
        {"evaluation": {"deep-live", "mteb-live"}},
    )

    assert deep_module.DeepEvaluationTaskService().list_active_collection_consumers(
        ["collection-1"]
    ) == ["deep-live"]


def test_training_consumers_include_stopped_task_with_live_execution(monkeypatch):
    engine = _engine_for(TrainingTaskDB.__table__)
    with Session(engine) as session:
        session.add_all(
            [
                TrainingTaskDB(
                    task_id="training-live",
                    task_name="live",
                    train_dataset_path="/datasets/train.jsonl",
                    status="stopped",
                ),
                TrainingTaskDB(
                    task_id="training-history",
                    task_name="history",
                    train_dataset_path="/datasets/train.jsonl",
                    status="completed",
                ),
            ]
        )
        session.commit()
    _patch_session(monkeypatch, training_module, engine)
    _patch_executing_ids(
        monkeypatch,
        training_module,
        {"training": {"training-live"}},
    )

    assert training_module.TrainingTaskService().list_active_dataset_consumers(
        ["/datasets/train.jsonl"]
    ) == ["training-live"]


def test_training_consumers_detect_nested_path_overlap(monkeypatch):
    engine = _engine_for(TrainingTaskDB.__table__)
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="training-nested",
                task_name="nested",
                train_dataset_path="/datasets/owned/splits/train.jsonl",
                status="running",
            )
        )
        session.commit()
    _patch_session(monkeypatch, training_module, engine)
    _patch_executing_ids(monkeypatch, training_module, {})

    assert training_module.TrainingTaskService().list_active_dataset_consumers(
        ["/datasets/owned"]
    ) == ["training-nested"]


def test_training_consumers_include_terminal_task_with_durable_process_lease(
    monkeypatch,
):
    engine = _engine_for(TrainingTaskDB.__table__)
    with Session(engine) as session:
        session.add(
            TrainingTaskDB(
                task_id="training-durable-live",
                task_name="durable-live",
                train_dataset_path="/datasets/durable/train.jsonl",
                status="stopped",
                process_pid=4321,
                process_status="running",
            )
        )
        session.commit()
    _patch_session(monkeypatch, training_module, engine)
    _patch_executing_ids(monkeypatch, training_module, {})

    assert training_module.TrainingTaskService().list_active_dataset_consumers(
        ["/datasets/durable/train.jsonl"]
    ) == ["training-durable-live"]


def test_evaluation_consumers_detect_nested_path_overlap(monkeypatch):
    engine = _engine_for(EvaluationTaskDB.__table__)
    with Session(engine) as session:
        session.add(
            EvaluationTaskDB(
                task_id="evaluation-nested",
                eval_framework=EvaluationFramework.MTEB,
                dataset_configs=[
                    {"path": "/datasets/owned/splits/eval.jsonl"}
                ],
                status="running",
            )
        )
        session.commit()
    _patch_session(monkeypatch, evaluation_module, engine)
    _patch_executing_ids(monkeypatch, evaluation_module, {})

    assert evaluation_module.EvaluationTaskService().list_active_dataset_consumers(
        [], ["/datasets/owned"]
    ) == ["evaluation-nested"]
