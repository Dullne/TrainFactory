"""Sync source identity, collection fingerprint, and public redaction tests."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from train_factory.storage.entities.deployment_entity import DeploymentDB
from train_factory.storage.entities.external_api_config_entity import (
    ExternalApiConfigDB,
)
from train_factory.storage.entities.external_sync_entity import (
    ExternalSyncBatchDB,
    ExternalSyncGenerationDB,
    ExternalSyncTaskDB,
    ExternalSyncTrainingDB,
    ExternalSyncTrainingTargetDB,
)
from train_factory.storage.services.external_api_config_service import (
    ExternalApiConfigService,
)
from train_factory.storage.services.external_sync_service import ExternalSyncService
from train_factory.sync.sync_worker import _get_sync_targets


def _services():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        ExternalApiConfigDB.__table__,
        DeploymentDB.__table__,
        ExternalSyncTaskDB.__table__,
        ExternalSyncBatchDB.__table__,
        ExternalSyncGenerationDB.__table__,
        ExternalSyncTrainingDB.__table__,
        ExternalSyncTrainingTargetDB.__table__,
    ):
        table.create(engine)
    sync_service = ExternalSyncService()
    sync_service.engine = engine
    api_service = ExternalApiConfigService()
    api_service.engine = engine
    return sync_service, api_service, engine


def test_public_sync_dtos_recursively_redact_secrets_but_raw_worker_reads_them():
    service, _api_service, engine = _services()
    with Session(engine) as session:
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-secret",
                task_name="secret",
                user_id="user-1",
                external_api_url="https://source.example/data",
                external_auth_config={
                    "headers": {"Authorization": "Bearer source-secret"},
                },
                generation_config={
                    "embedding_config": {
                        "api_key": "embedding-secret",
                        "max_tokens": 512,
                        "tokenizer": "qwen-tokenizer",
                        "nested": [
                            {"client-secret": "client-secret-value"},
                            {"model": "embedding-model"},
                        ],
                    }
                },
                training_config={
                    "hub_token": "hub-secret",
                    "callbacks": [{"password": "callback-secret"}],
                },
            )
        )
        session.add(
            ExternalSyncTrainingTargetDB(
                target_id="target-secret",
                task_id="sync-secret",
                target_name="Embedding",
                training_config={
                    "apiKey": "target-secret-value",
                    "nested": [{"private_key": "private-value"}],
                    "learning_rate": 0.001,
                },
            )
        )
        session.commit()

    public_task = service.get_task("sync-secret")
    raw_task = service.get_task_raw("sync-secret")
    active_task = service.list_active_tasks()[0]
    public_target = service.get_training_target("target-secret")
    raw_target = service.get_training_target_raw("target-secret")

    assert public_task["external_auth_config"]["headers"]["Authorization"] == "***"
    assert (
        public_task["generation_config"]["embedding_config"]["api_key"]
        == "***"
    )
    assert (
        public_task["generation_config"]["embedding_config"]["nested"][0][
            "client-secret"
        ]
        == "***"
    )
    assert public_task["generation_config"]["embedding_config"]["max_tokens"] == 512
    assert (
        public_task["generation_config"]["embedding_config"]["tokenizer"]
        == "qwen-tokenizer"
    )
    assert public_task["training_config"]["hub_token"] == "***"
    assert public_task["training_config"]["callbacks"][0]["password"] == "***"
    assert (
        raw_task["generation_config"]["embedding_config"]["api_key"]
        == "embedding-secret"
    )
    assert raw_task["training_config"]["hub_token"] == "hub-secret"
    assert active_task["external_auth_config"]["headers"]["Authorization"] == (
        "Bearer source-secret"
    )
    assert (
        active_task["generation_config"]["embedding_config"]["api_key"]
        == "embedding-secret"
    )
    assert public_target["training_config"]["apiKey"] == "***"
    assert public_target["training_config"]["nested"][0]["private_key"] == "***"
    assert public_target["training_config"]["learning_rate"] == 0.001
    assert raw_target["training_config"]["apiKey"] == "target-secret-value"


def test_sync_source_identity_can_change_only_before_ingestion():
    service, _api_service, engine = _services()
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTaskDB(
                    task_id="sync-empty",
                    task_name="empty",
                    user_id="user-1",
                    external_api_url="https://one.example/data",
                ),
                ExternalSyncTaskDB(
                    task_id="sync-used",
                    task_name="used",
                    user_id="user-1",
                    external_api_url="https://one.example/data",
                    last_sync_at=datetime(2026, 8, 11, 0, 0, 0),
                    last_sync_boundary_ids=["a" * 64],
                    total_record_count=1,
                ),
            ]
        )
        session.commit()

    updated = service.update_task(
        "sync-empty",
        external_api_url="https://two.example/data",
    )
    assert updated["external_api_url"] == "https://two.example/data"

    with pytest.raises(ValueError, match="source"):
        service.update_task(
            "sync-used",
            external_api_url="https://two.example/data",
        )
    assert service.get_task_raw("sync-used")["external_api_url"] == (
        "https://one.example/data"
    )


def test_referenced_external_api_url_cannot_change_in_place():
    sync_service, api_service, engine = _services()
    with Session(engine) as session:
        session.add(
            ExternalApiConfigDB(
                config_id="api-config-1",
                config_name="source",
                user_id="user-1",
                api_url="https://one.example/data",
                auth_config={"token": "secret"},
            )
        )
        session.add(
            ExternalSyncTaskDB(
                task_id="sync-referenced",
                task_name="referenced",
                user_id="user-1",
                external_api_config_id="api-config-1",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="referenced"):
        api_service.update_config(
            "api-config-1",
            api_url="https://two.example/data",
        )

    rotated = api_service.update_config(
        "api-config-1",
        auth_config={"token": "rotated"},
    )
    assert rotated["api_url"] == "https://one.example/data"
    assert sync_service.get_task("sync-referenced") is not None


def test_embedding_collection_fingerprint_changes_with_endpoint_and_config_identity():
    config = {
        "task_id": "sync-fingerprint",
        "user_id": "user-1",
        "external_api_config_id": "source-1",
    }
    baseline = _get_sync_targets(
        config,
        {
            "config_id": "embedding-config-1",
            "endpoint": "https://embedding-one.example/v1",
            "model": "qwen3-embedding-0.6b",
            "api_key": "first-secret",
            "max_tokens": 512,
            "tokenizer": "qwen-tokenizer-v1",
        },
    )[0]["collection_name"]
    rotated_secret = _get_sync_targets(
        config,
        {
            "config_id": "embedding-config-1",
            "endpoint": "https://embedding-one.example/v1",
            "model": "qwen3-embedding-0.6b",
            "api_key": "rotated-secret",
            "max_tokens": 512,
            "tokenizer": "qwen-tokenizer-v1",
        },
    )[0]["collection_name"]
    changed_endpoint = _get_sync_targets(
        config,
        {
            "config_id": "embedding-config-1",
            "endpoint": "https://embedding-two.example/v1",
            "model": "qwen3-embedding-0.6b",
            "api_key": "first-secret",
            "max_tokens": 512,
            "tokenizer": "qwen-tokenizer-v1",
        },
    )[0]["collection_name"]
    changed_config = _get_sync_targets(
        config,
        {
            "config_id": "embedding-config-2",
            "endpoint": "https://embedding-one.example/v1",
            "model": "qwen3-embedding-0.6b",
            "api_key": "first-secret",
            "max_tokens": 512,
            "tokenizer": "qwen-tokenizer-v1",
        },
    )[0]["collection_name"]
    changed_max_tokens = _get_sync_targets(
        config,
        {
            "config_id": "embedding-config-1",
            "endpoint": "https://embedding-one.example/v1",
            "model": "qwen3-embedding-0.6b",
            "api_key": "first-secret",
            "max_tokens": 1024,
            "tokenizer": "qwen-tokenizer-v1",
        },
    )[0]["collection_name"]
    changed_tokenizer = _get_sync_targets(
        config,
        {
            "config_id": "embedding-config-1",
            "endpoint": "https://embedding-one.example/v1",
            "model": "qwen3-embedding-0.6b",
            "api_key": "first-secret",
            "max_tokens": 512,
            "tokenizer": "qwen-tokenizer-v2",
        },
    )[0]["collection_name"]

    assert baseline == rotated_secret
    assert changed_endpoint != baseline
    assert changed_config != baseline
    assert changed_max_tokens != baseline
    assert changed_tokenizer != baseline
