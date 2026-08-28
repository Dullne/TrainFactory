from contextlib import contextmanager
import importlib
import os

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.generation_task_entity import (
    GenerationStatus,
    GenerationTaskDB,
)
from train_factory.storage.entities.model_artifact_membership_gate_entity import (
    ModelArtifactMembershipGateDB,
)
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.services.deep_evaluation_task_service import (
    DeepEvaluationTaskService,
)
from train_factory.storage.services.evaluation_task_service import EvaluationTaskService
from train_factory.storage.services.generation_task_service import GenerationTaskService
from train_factory.storage.services.training_task_service import TrainingTaskService
from train_factory.utils.path_utils import hash_path


@pytest.fixture
def isolated_task_database(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'dataset-consumers.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(ModelArtifactMembershipGateDB(gate_id=1))
        session.commit()

    @contextmanager
    def test_session():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    deep_evaluation_task_service = importlib.import_module(
        "train_factory.storage.services.deep_evaluation_task_service"
    )
    evaluation_task_service = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    generation_task_service = importlib.import_module(
        "train_factory.storage.services.generation_task_service"
    )
    training_task_event_service = importlib.import_module(
        "train_factory.storage.services.training_task_event_service"
    )
    training_task_service = importlib.import_module(
        "train_factory.storage.services.training_task_service"
    )

    for module in (
        generation_task_service,
        training_task_service,
        evaluation_task_service,
        deep_evaluation_task_service,
    ):
        monkeypatch.setattr(module, "get_session", test_session)
    monkeypatch.setattr(
        training_task_event_service.training_task_event_service,
        "log_event",
        lambda **_kwargs: None,
    )
    return engine


def _add_dataset(engine, *, status="deleting"):
    storage_path = "/managed/datasets/dataset-1/data.jsonl"
    with Session(engine) as session:
        session.add(
            DatasetDB(
                dataset_id="dataset-1",
                dataset_name="dataset-1",
                storage_path=storage_path,
                storage_path_hash=hash_path(storage_path),
                status=status,
                user_id="user-1",
            )
        )
        session.commit()
    return storage_path


def _mark_dataset_deleting(engine):
    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == "dataset-1")
        ).one()
        dataset.status = "deleting"
        session.add(dataset)
        session.commit()


def _delete_dataset_row(engine):
    with Session(engine) as session:
        dataset = session.exec(
            select(DatasetDB).where(DatasetDB.dataset_id == "dataset-1")
        ).one()
        session.delete(dataset)
        session.commit()


def test_generation_create_rejects_dataset_being_deleted(isolated_task_database):
    _add_dataset(isolated_task_database)

    with pytest.raises(ValueError, match="deletion"):
        GenerationTaskService().create_task(
            task_name="generation",
            input_path="/managed/datasets/dataset-1/data.jsonl",
            llm_config={},
            steps_config={},
            source_dataset_id="dataset-1",
            user_id="user-1",
        )


def test_generation_create_rejects_dataset_deleted_after_route_validation(
    isolated_task_database,
):
    with pytest.raises(ValueError, match="no longer exists"):
        GenerationTaskService().create_task(
            task_name="generation",
            input_path="/managed/datasets/dataset-1/data.jsonl",
            llm_config={},
            steps_config={},
            source_dataset_id="dataset-1",
            user_id="user-1",
            require_source_dataset=True,
        )


def test_training_create_rejects_dataset_being_deleted(isolated_task_database):
    storage_path = _add_dataset(isolated_task_database)

    with pytest.raises(ValueError, match="deletion"):
        TrainingTaskService().create_task(
            task_name="training",
            train_dataset_path=storage_path,
            training_params={"dataset_configs": [{"path": storage_path}]},
            user_id="user-1",
        )


def test_training_create_rejects_dataset_deleted_after_route_validation(
    isolated_task_database,
):
    storage_path = "/managed/datasets/dataset-1/data.jsonl"

    with pytest.raises(ValueError, match="no longer exists"):
        TrainingTaskService().create_task(
            task_name="training",
            train_dataset_path=storage_path,
            training_params={"dataset_configs": [{"path": storage_path}]},
            user_id="user-1",
            require_managed_datasets=True,
        )


@pytest.mark.parametrize(
    ("service", "extra_kwargs"),
    [
        (EvaluationTaskService(), {}),
        (DeepEvaluationTaskService(), {"eval_type": "embedding"}),
    ],
)
def test_evaluation_create_rejects_dataset_being_deleted(
    isolated_task_database,
    service,
    extra_kwargs,
):
    _add_dataset(isolated_task_database)

    with pytest.raises(ValueError, match="deletion"):
        service.create_task(
            task_name="evaluation",
            dataset_configs=[{"dataset_id": "dataset-1"}],
            user_id="user-1",
            **extra_kwargs,
        )


@pytest.mark.parametrize(
    ("service", "extra_kwargs"),
    [
        (EvaluationTaskService(), {}),
        (DeepEvaluationTaskService(), {"eval_type": "embedding"}),
    ],
)
def test_evaluation_create_rejects_dataset_deleted_after_route_validation(
    isolated_task_database,
    service,
    extra_kwargs,
):
    with pytest.raises(ValueError, match="no longer exists"):
        service.create_task(
            task_name="evaluation",
            dataset_configs=[{"dataset_id": "dataset-1"}],
            user_id="user-1",
            **extra_kwargs,
        )


def test_generation_restart_cannot_race_dataset_deletion(isolated_task_database):
    _add_dataset(isolated_task_database, status="ready")
    service = GenerationTaskService()
    task = service.create_task(
        task_name="generation",
        input_path="/managed/datasets/dataset-1/data.jsonl",
        llm_config={},
        steps_config={},
        source_dataset_id="dataset-1",
        user_id="user-1",
    )
    with Session(isolated_task_database) as session:
        row = session.exec(
            select(GenerationTaskDB).where(
                GenerationTaskDB.task_id == task["task_id"]
            )
        ).one()
        row.status = GenerationStatus.FAILED
        session.add(row)
        session.commit()
    _mark_dataset_deleting(isolated_task_database)

    assert service.update_status(task["task_id"], GenerationStatus.PENDING) is False


def test_training_resume_cannot_race_dataset_deletion(isolated_task_database):
    storage_path = _add_dataset(isolated_task_database, status="ready")
    service = TrainingTaskService()
    task = service.create_task(
        task_name="training",
        train_dataset_path=storage_path,
        training_params={"dataset_configs": [{"path": storage_path}]},
        user_id="user-1",
    )
    with Session(isolated_task_database) as session:
        row = session.exec(
            select(TrainingTaskDB).where(TrainingTaskDB.task_id == task["task_id"])
        ).one()
        row.status = "failed"
        session.add(row)
        session.commit()
    _mark_dataset_deleting(isolated_task_database)

    assert service.reset_for_resume(task["task_id"], "new-run-token") is False


def test_training_artifact_consumers_include_parent_and_checkpoint_links(
    isolated_task_database,
):
    artifact_root = os.path.join(os.sep, "app", "output", "parent")
    with Session(isolated_task_database) as session:
        session.add_all(
            [
                TrainingTaskDB(
                    task_id="parent",
                    task_name="parent",
                    output_dir=artifact_root,
                    user_id="user-1",
                ),
                TrainingTaskDB(
                    task_id="child-by-parent",
                    task_name="child-by-parent",
                    parent_task_id="parent",
                    user_id="user-1",
                ),
                TrainingTaskDB(
                    task_id="child-by-path",
                    task_name="child-by-path",
                    sft_checkpoint_path=os.path.join(
                        artifact_root,
                        ".",
                        "checkpoint-10",
                    ),
                    user_id="user-1",
                ),
                TrainingTaskDB(
                    task_id="unrelated",
                    task_name="unrelated",
                    sft_checkpoint_path=os.path.join(
                        f"{artifact_root}-other",
                        "checkpoint-10",
                    ),
                    user_id="user-1",
                ),
            ]
        )
        session.commit()

    consumers = TrainingTaskService().list_artifact_consumers(
        "parent",
        artifact_root + os.sep,
    )

    assert {consumer["task_id"] for consumer in consumers} == {
        "child-by-parent",
        "child-by-path",
    }


@pytest.mark.parametrize(
    ("service", "framework"),
    [
        (EvaluationTaskService(), "mteb"),
        (DeepEvaluationTaskService(), "deepeval"),
    ],
)
def test_evaluation_resume_cannot_race_dataset_deletion(
    isolated_task_database,
    service,
    framework,
):
    _add_dataset(isolated_task_database, status="ready")
    task = service.create_task(
        task_name="evaluation",
        dataset_configs=[{"dataset_id": "dataset-1"}],
        user_id="user-1",
    )
    with Session(isolated_task_database) as session:
        row = session.exec(
            select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task["task_id"])
        ).one()
        assert row.eval_framework == framework
        row.status = "failed"
        session.add(row)
        session.commit()
    _mark_dataset_deleting(isolated_task_database)

    assert service.reset_for_resume(task["task_id"]) is False


@pytest.mark.parametrize(
    ("service", "framework"),
    [
        (EvaluationTaskService(), "mteb"),
        (DeepEvaluationTaskService(), "deepeval"),
    ],
)
def test_evaluation_resume_rejects_explicit_dataset_deleted_after_validation(
    isolated_task_database,
    service,
    framework,
):
    _add_dataset(isolated_task_database, status="ready")
    task = service.create_task(
        task_name="evaluation",
        dataset_configs=[{"dataset_id": "dataset-1"}],
        user_id="anonymous",
    )
    with Session(isolated_task_database) as session:
        row = session.exec(
            select(EvaluationTaskDB).where(EvaluationTaskDB.task_id == task["task_id"])
        ).one()
        assert row.eval_framework == framework
        row.status = "failed"
        session.add(row)
        session.commit()
    _delete_dataset_row(isolated_task_database)

    assert service.reset_for_resume(task["task_id"]) is False
