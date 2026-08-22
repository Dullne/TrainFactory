import pytest

from train_factory.storage.entities.dataset_entity import DatasetDB
from train_factory.storage.entities.model_registry_entity import ModelRegistryDB
from train_factory.enums.sync_status import (
    BatchStatus,
    SyncGenerationStatus,
    SyncStatus,
    SyncTrainingStatus,
)
from train_factory.storage.services.external_sync_service import ExternalSyncService


def test_dataset_status_transitions():
    dataset = DatasetDB(dataset_name="unit-dataset")

    dataset.update_status("uploading")
    dataset.update_status("ready")
    dataset.update_status("archived")
    dataset.update_status("ready")

    with pytest.raises(ValueError):
        dataset.update_status("processing")

    with pytest.raises(ValueError):
        dataset.update_status("not_a_status")


def test_dataset_deletion_is_a_terminal_status():
    dataset = DatasetDB(dataset_name="unit-dataset-delete", status="ready")

    dataset.update_status("deleting")
    assert dataset.status == "deleting"

    with pytest.raises(ValueError):
        dataset.update_status("ready")


def test_model_status_transitions():
    model = ModelRegistryDB(
        model_name="unit-model",
        model_type="embedding",
        model_path="/tmp/unit-model",
    )

    model.update_status("available")
    model.update_status("archived")
    model.update_status("available")

    with pytest.raises(ValueError):
        model.update_status("registered")

    with pytest.raises(ValueError):
        model.update_status("not_a_status")


def test_sync_status_transition_validation():
    service = ExternalSyncService()

    # task
    service._validate_transition(
        SyncStatus.IDLE,
        SyncStatus.SYNCING,
        service._TASK_STATUS_TRANSITIONS,
        "sync task",
    )
    with pytest.raises(ValueError):
        service._validate_transition(
            SyncStatus.GENERATING,
            SyncStatus.SYNCING,
            service._TASK_STATUS_TRANSITIONS,
            "sync task",
        )

    # batch
    service._validate_transition(
        BatchStatus.FETCHED,
        BatchStatus.GENERATION_QUEUED,
        service._BATCH_STATUS_TRANSITIONS,
        "sync batch",
    )
    with pytest.raises(ValueError):
        service._validate_transition(
            BatchStatus.GENERATION_DONE,
            BatchStatus.GENERATION_QUEUED,
            service._BATCH_STATUS_TRANSITIONS,
            "sync batch",
        )

    # generation
    service._validate_transition(
        SyncGenerationStatus.PENDING,
        SyncGenerationStatus.COMPLETED,
        service._GENERATION_STATUS_TRANSITIONS,
        "sync generation",
    )
    with pytest.raises(ValueError):
        service._validate_transition(
            SyncGenerationStatus.COMPLETED,
            SyncGenerationStatus.PENDING,
            service._GENERATION_STATUS_TRANSITIONS,
            "sync generation",
        )

    # training
    service._validate_transition(
        SyncTrainingStatus.PENDING,
        SyncTrainingStatus.COMPLETED,
        service._TRAINING_STATUS_TRANSITIONS,
        "sync training",
    )
    with pytest.raises(ValueError):
        service._validate_transition(
            SyncTrainingStatus.PENDING,
            SyncTrainingStatus.ADAPTER_UNLOADED,
            service._TRAINING_STATUS_TRANSITIONS,
            "sync training",
        )
