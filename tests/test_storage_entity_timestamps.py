"""Storage entity timestamps follow the application timezone clock."""

from datetime import datetime

from train_factory.core import time_utils
from train_factory.storage.entities import dataset_entity
from train_factory.storage.entities import model_registry_entity
from train_factory.storage.entities import training_task_entity
from train_factory.storage.entities.audit_log_entity import AuditLogDB
from train_factory.storage.entities.dataset_asset_entity import DatasetAssetDB
from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.dataset_lineage_entity import DatasetLineageEdgeDB
from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.evaluation_task_entity import EvaluationTaskDB
from train_factory.storage.entities.generation_task_entity import GenerationTaskDB
from train_factory.storage.entities.loaded_adapter_entity import LoadedAdapterDB
from train_factory.storage.entities.milvus_collection_entity import (
    CollectionDatasetLinkDB,
    MilvusCollectionDB,
)
from train_factory.storage.entities.model_config_entity import ModelConfigDB
from train_factory.storage.entities.model_registry_entity import (
    ModelRegistryDB,
    ModelVersionDB,
)
from train_factory.storage.entities.training_task_entity import TrainingTaskDB
from train_factory.storage.entities.training_task_event_entity import TrainingTaskEventDB
from train_factory.storage.entities.user_entity import UserDB


ENTITY_TIMESTAMP_FIELDS = (
    (AuditLogDB, ("created_at",)),
    (DatasetAssetDB, ("created_at",)),
    (DatasetDB, ("created_at", "updated_at")),
    (DatasetLineageEdgeDB, ("created_at",)),
    (DeploymentDB, ("created_at", "updated_at")),
    (EvaluationTaskDB, ("created_at", "updated_at")),
    (GenerationTaskDB, ("created_at",)),
    (LoadedAdapterDB, ("loaded_at",)),
    (MilvusCollectionDB, ("created_at", "updated_at")),
    (CollectionDatasetLinkDB, ("linked_at",)),
    (ModelConfigDB, ("created_at", "updated_at")),
    (ModelRegistryDB, ("created_at", "updated_at")),
    (ModelVersionDB, ("created_at",)),
    (TrainingTaskDB, ("created_at", "updated_at")),
    (TrainingTaskEventDB, ("created_at",)),
    (UserDB, ("created_at", "updated_at")),
)


def test_local_entity_timestamp_defaults_use_app_timezone_clock(monkeypatch):
    fixed = datetime(2026, 8, 8, 15, 30, 45)
    monkeypatch.setattr(time_utils, "now_aware", lambda: fixed.astimezone())

    for entity, field_names in ENTITY_TIMESTAMP_FIELDS:
        for field_name in field_names:
            factory = entity.model_fields[field_name].default_factory
            assert factory is time_utils.now_naive
            assert factory() == fixed


def test_training_task_lifecycle_uses_same_naive_clock(monkeypatch):
    fixed = datetime(2026, 8, 8, 16, 0, 0)
    monkeypatch.setattr(training_task_entity, "now_naive", lambda: fixed)
    task = TrainingTaskDB(task_name="timezone-test")

    task.update_status("running")

    assert task.updated_at == fixed
    assert task.started_at == fixed


def test_registry_and_dataset_updates_use_same_naive_clock(monkeypatch):
    fixed = datetime(2026, 8, 8, 16, 30, 0)
    monkeypatch.setattr(model_registry_entity, "now_naive", lambda: fixed)
    monkeypatch.setattr(dataset_entity, "now_naive", lambda: fixed)
    model = ModelRegistryDB(
        model_name="timezone-model",
        model_type="embedding",
        model_path="/tmp/timezone-model",
    )
    dataset = DatasetDB(dataset_name="timezone-dataset")

    model.update_status("available")
    dataset.update_status("ready")

    assert model.updated_at == fixed
    assert dataset.updated_at == fixed
