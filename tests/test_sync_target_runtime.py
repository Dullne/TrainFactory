from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from train_factory.api.routes import sync_routes
from train_factory.enums.sync_status import SyncTrainingStatus
from train_factory.storage.entities.external_sync_entity import ExternalSyncTrainingDB
from train_factory.storage.services.external_sync_service import (
    ExternalSyncService,
    external_sync_service,
)
from train_factory.sync import post_training_handler, sync_worker


CURRENT_USER = {"user_id": "user-1", "username": "owner"}
adapter_service_module = importlib.import_module(
    "train_factory.deployment.adapter_service"
)


def _deployment(deployment_id: str) -> dict[str, Any]:
    return {
        "deployment_id": deployment_id,
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "status": "degraded",
        "config": {},
    }


def _install_unload_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    adapters_by_binding: dict[tuple[str, str | None], list[dict[str, Any]]],
) -> tuple[list[tuple[str, str, str | None]], list[dict[str, Any]]]:
    target_rows = {target["target_id"]: target for target in targets}
    adapter_calls: list[tuple[str, str, str | None]] = []
    mark_calls: list[dict[str, Any]] = []
    runtime_adapters: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    adapter_rows: dict[str, dict[str, Any]] = {}
    training_rows: dict[str, dict[str, Any]] = {}
    for binding, adapters in adapters_by_binding.items():
        runtime_adapters[binding] = []
        for raw_adapter in adapters:
            adapter = dict(raw_adapter)
            owner = next(
                (
                    target
                    for target in targets
                    if target.get("current_adapter_id") == adapter.get("adapter_id")
                    or (
                        not target.get("current_adapter_id")
                        and target.get("current_adapter_name")
                        == adapter.get("adapter_name")
                    )
                ),
                None,
            )
            adapter.setdefault("deployment_id", binding[0])
            adapter.setdefault("deployment_replica_id", binding[1])
            adapter.setdefault("user_id", config.get("user_id"))
            adapter.setdefault("status", "loaded")
            if owner is not None:
                adapter.setdefault("source_task_id", owner.get("current_training_id"))
            runtime_adapters[binding].append(adapter)
            adapter_id = adapter.get("adapter_id")
            if adapter_id:
                adapter_rows[adapter_id] = adapter
            source_task_id = adapter.get("source_task_id")
            if source_task_id and owner is not None:
                training_rows[source_task_id] = {
                    "task_id": config["task_id"],
                    "training_task_id": source_task_id,
                    "target_id": owner.get("target_id"),
                    "loaded_adapter_id": adapter.get("adapter_id"),
                    "loaded_adapter_name": adapter.get("adapter_name"),
                    "status": SyncTrainingStatus.ADAPTER_LOADED,
                    "target_config_snapshot": {
                        "base_deployment_id": binding[0],
                        "base_deployment_replica_id": binding[1],
                    },
                }

    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: dict(config),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target_raw",
        lambda target_id: dict(target_rows[target_id])
        if target_id in target_rows
        else None,
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda _task_id, **_kwargs: [dict(row) for row in target_rows.values()],
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda training_id: dict(training_rows[training_id])
        if training_id in training_rows
        else None,
    )

    task_updates: list[dict[str, Any]] = []

    def update_task(_task_id: str, **kwargs: Any) -> dict[str, Any]:
        task_updates.append(kwargs)
        config.update(kwargs)
        return dict(config)

    def update_target(target_id: str, **kwargs: Any) -> dict[str, Any]:
        target_rows[target_id].update(kwargs)
        return dict(target_rows[target_id])

    monkeypatch.setattr(external_sync_service, "update_task", update_task)
    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        update_target,
    )
    monkeypatch.setattr(
        external_sync_service,
        "mark_all_trainings_adapter_unloaded",
        lambda *_args, **kwargs: mark_calls.append(kwargs) or 0,
    )

    deployment_service = importlib.import_module(
        "train_factory.deployment.deployment_service"
    ).deployment_service
    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda deployment_id: _deployment(deployment_id),
    )

    class RecordingAdapterService:
        def sync_loaded_adapters(
            self,
            deployment_id: str,
            *,
            deployment_replica_id: str | None,
            user_id: str | None,
        ) -> int:
            assert user_id == "user-1"
            adapter_calls.append(("sync", deployment_id, deployment_replica_id))
            return 0

        def list_loaded_adapters(
            self,
            deployment_id: str,
            *,
            deployment_replica_id: str | None,
            user_id: str | None,
        ) -> list[dict[str, Any]]:
            assert user_id == "user-1"
            adapter_calls.append(("list", deployment_id, deployment_replica_id))
            return [
                dict(adapter)
                for adapter in runtime_adapters.get(
                    (deployment_id, deployment_replica_id), []
                )
            ]

        def get_adapter(self, adapter_id: str) -> dict[str, Any] | None:
            adapter = adapter_rows.get(adapter_id)
            return dict(adapter) if adapter is not None else None

        def unload_adapter(
            self,
            deployment_id: str,
            adapter_name: str,
            *,
            deployment_replica_id: str | None,
            user_id: str | None,
        ) -> bool:
            assert user_id == "user-1"
            adapter_calls.append(
                (f"unload:{adapter_name}", deployment_id, deployment_replica_id)
            )
            binding = (deployment_id, deployment_replica_id)
            runtime_adapters[binding] = [
                adapter
                for adapter in runtime_adapters.get(binding, [])
                if adapter.get("adapter_name") != adapter_name
            ]
            return True

    monkeypatch.setattr(
        adapter_service_module,
        "AdapterService",
        RecordingAdapterService,
    )
    return adapter_calls, mark_calls


def test_unload_explicit_target_only_clears_its_runtime_and_preserves_other_target(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-1",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
        "current_adapter_name": "adapter-one",
        "current_adapter_id": "adapter-1",
    }
    targets = [
        {
            "target_id": "target-1",
            "task_id": "sync-1",
            "base_deployment_id": "deployment-1",
            "base_deployment_replica_id": "replica-1",
            "current_adapter_name": "adapter-one",
            "current_adapter_id": "adapter-1",
            "current_training_id": "training-1",
        },
        {
            "target_id": "target-2",
            "task_id": "sync-1",
            "base_deployment_id": "deployment-2",
            "base_deployment_replica_id": "replica-2",
            "current_adapter_name": "adapter-two",
            "current_adapter_id": "adapter-2",
            "current_training_id": "training-2",
        },
    ]
    adapter_calls, mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-1", "replica-1"): [
                {"adapter_name": "adapter-one", "adapter_id": "adapter-1"}
            ],
        },
    )

    post_training_handler.unload_current_adapter("sync-1", target_id="target-1")

    assert {call[1:] for call in adapter_calls} == {
        ("deployment-1", "replica-1")
    }
    assert mark_calls == [
        {
            "loaded_adapter_id": "adapter-1",
            "loaded_adapter_name": "adapter-one",
        }
    ]
    assert config["current_adapter_name"] == "adapter-two"
    assert config["current_adapter_id"] == "adapter-2"
    assert config["current_training_id"] == "training-2"


def test_unload_shared_adapter_rejects_non_source_target_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-shared",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-one",
        "current_adapter_id": "adapter-1",
    }
    targets = [
        {
            "target_id": "target-1",
            "task_id": "sync-shared",
            "base_deployment_id": "deployment-shared",
            "base_deployment_replica_id": "replica-shared",
            "current_adapter_name": "adapter-one",
            "current_adapter_id": "adapter-1",
            "current_training_id": "training-1",
        },
        {
            "target_id": "target-2",
            "task_id": "sync-shared",
            "base_deployment_id": "deployment-shared",
            "base_deployment_replica_id": "replica-shared",
            "current_adapter_name": "adapter-one",
            "current_adapter_id": "adapter-1",
            "current_training_id": "training-1",
        },
    ]
    adapter_calls, mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-shared", "replica-shared"): [
                {"adapter_name": "adapter-one", "adapter_id": "adapter-1"},
            ],
        },
    )

    post_training_handler.unload_current_adapter(
        "sync-shared", target_id="target-1"
    )

    assert not any(call[0].startswith("unload:") for call in adapter_calls)
    assert mark_calls == []
    assert targets[0]["current_adapter_id"] is None
    assert targets[0]["current_training_id"] is None
    assert targets[1]["current_adapter_id"] == "adapter-1"
    assert config["current_adapter_id"] == "adapter-1"
    assert config["current_training_id"] == "training-1"

    runtime_call_count = len(adapter_calls)
    with pytest.raises(ValueError, match="target ownership"):
        post_training_handler.unload_current_adapter(
            "sync-shared", target_id="target-2"
        )

    assert len(adapter_calls) == runtime_call_count
    assert not any(call[0].startswith("unload:") for call in adapter_calls)
    assert mark_calls == []
    assert targets[1]["current_adapter_id"] == "adapter-1"
    assert config["current_adapter_id"] == "adapter-1"


def test_unload_only_removes_owned_adapter_from_three_party_shared_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-owner-a",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-a",
        "current_adapter_id": "adapter-a-id",
        "current_training_id": "training-a",
    }
    targets = [
        {
            "target_id": "target-owner-a",
            "task_id": "sync-owner-a",
            "base_deployment_id": "deployment-shared-three",
            "base_deployment_replica_id": "replica-shared-three",
            "current_adapter_name": "adapter-a",
            "current_adapter_id": "adapter-a-id",
            "current_training_id": "training-a",
        }
    ]
    other_task = {
        "task_id": "sync-owner-b",
        "current_adapter_name": "adapter-b",
        "current_adapter_id": "adapter-b-id",
        "current_training_id": "training-b",
    }
    adapter_calls, mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-shared-three", "replica-shared-three"): [
                {"adapter_name": "adapter-a", "adapter_id": "adapter-a-id"},
                {
                    "adapter_name": "adapter-b",
                    "adapter_id": "adapter-b-id",
                    "source_task_id": "training-b",
                },
                {
                    "adapter_name": "manual-adapter",
                    "adapter_id": "manual-adapter-id",
                    "source_model_id": "manual-model",
                },
            ]
        },
    )

    post_training_handler.unload_current_adapter(
        "sync-owner-a",
        target_id="target-owner-a",
    )

    assert [
        call for call in adapter_calls if call[0].startswith("unload:")
    ] == [
        (
            "unload:adapter-a",
            "deployment-shared-three",
            "replica-shared-three",
        )
    ]
    assert mark_calls == [
        {
            "loaded_adapter_id": "adapter-a-id",
            "loaded_adapter_name": "adapter-a",
        }
    ]
    assert other_task == {
        "task_id": "sync-owner-b",
        "current_adapter_name": "adapter-b",
        "current_adapter_id": "adapter-b-id",
        "current_training_id": "training-b",
    }


def test_unload_fails_closed_when_persisted_adapter_binding_mismatches_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-mismatch",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-mismatch",
        "current_adapter_id": "adapter-mismatch-id",
        "current_training_id": "training-mismatch",
    }
    targets = [
        {
            "target_id": "target-mismatch",
            "task_id": "sync-mismatch",
            "base_deployment_id": "deployment-expected",
            "base_deployment_replica_id": "replica-expected",
            "current_adapter_name": "adapter-mismatch",
            "current_adapter_id": "adapter-mismatch-id",
            "current_training_id": "training-mismatch",
        }
    ]
    adapter_calls, mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-expected", "replica-expected"): [
                {
                    "adapter_name": "adapter-mismatch",
                    "adapter_id": "adapter-mismatch-id",
                    "deployment_id": "deployment-other",
                    "deployment_replica_id": "replica-other",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="binding"):
        post_training_handler.unload_current_adapter(
            "sync-mismatch",
            target_id="target-mismatch",
        )

    assert not any(call[0].startswith("unload:") for call in adapter_calls)
    assert mark_calls == []
    assert targets[0]["current_adapter_id"] == "adapter-mismatch-id"
    assert config["current_adapter_id"] == "adapter-mismatch-id"


def test_unload_fails_closed_when_runtime_sync_cannot_verify_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-runtime-unverified",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-unverified",
        "current_adapter_id": "adapter-unverified-id",
        "current_training_id": "training-unverified",
    }
    targets = [
        {
            "target_id": "target-runtime-unverified",
            "task_id": "sync-runtime-unverified",
            "base_deployment_id": "deployment-unverified",
            "base_deployment_replica_id": "replica-unverified",
            "current_adapter_name": "adapter-unverified",
            "current_adapter_id": "adapter-unverified-id",
            "current_training_id": "training-unverified",
        }
    ]
    adapter_calls, mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-unverified", "replica-unverified"): [
                {
                    "adapter_name": "adapter-unverified",
                    "adapter_id": "adapter-unverified-id",
                }
            ]
        },
    )

    def fail_sync(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("runtime unavailable")

    monkeypatch.setattr(
        adapter_service_module.AdapterService,
        "sync_loaded_adapters",
        fail_sync,
    )

    with pytest.raises(RuntimeError, match="runtime unavailable"):
        post_training_handler.unload_current_adapter(
            "sync-runtime-unverified",
            target_id="target-runtime-unverified",
        )

    assert not any(call[0].startswith("unload:") for call in adapter_calls)
    assert mark_calls == []
    assert targets[0]["current_adapter_id"] == "adapter-unverified-id"
    assert config["current_adapter_id"] == "adapter-unverified-id"


@pytest.mark.parametrize(
    ("training_target_id", "snapshot_task_id", "snapshot_target_id"),
    [
        ("target-other", "sync-exact-owner", "target-selected"),
        ("target-selected", "sync-other", "target-selected"),
        ("target-selected", "sync-exact-owner", "target-other"),
    ],
)
def test_unload_fails_closed_before_runtime_for_non_exact_target_source(
    monkeypatch: pytest.MonkeyPatch,
    training_target_id: str,
    snapshot_task_id: str,
    snapshot_target_id: str,
):
    config = {
        "task_id": "sync-exact-owner",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-owner",
        "current_adapter_id": "adapter-owner-id",
        "current_training_id": "training-owner",
    }
    targets = [
        {
            "target_id": "target-selected",
            "task_id": "sync-exact-owner",
            "base_deployment_id": "deployment-shared",
            "base_deployment_replica_id": "replica-shared",
            "current_adapter_name": "adapter-owner",
            "current_adapter_id": "adapter-owner-id",
            "current_training_id": "training-owner",
        }
    ]
    adapter_calls, mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-shared", "replica-shared"): [
                {
                    "adapter_name": "adapter-owner",
                    "adapter_id": "adapter-owner-id",
                    "source_task_id": "training-owner",
                }
            ]
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _training_id: {
            "task_id": "sync-exact-owner",
            "training_task_id": "training-owner",
            "target_id": training_target_id,
            "loaded_adapter_id": "adapter-owner-id",
            "loaded_adapter_name": "adapter-owner",
            "status": SyncTrainingStatus.ADAPTER_LOADED,
            "target_config_snapshot": {
                "task_id": snapshot_task_id,
                "target_id": snapshot_target_id,
                "base_deployment_id": "deployment-shared",
                "base_deployment_replica_id": "replica-shared",
            },
        },
    )

    with pytest.raises(ValueError, match="ownership"):
        post_training_handler.unload_current_adapter(
            "sync-exact-owner",
            target_id="target-selected",
        )

    assert adapter_calls == []
    assert mark_calls == []
    assert targets[0]["current_adapter_id"] == "adapter-owner-id"


def test_name_only_adapter_source_rejects_wrong_target_without_runtime_unload(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-name-owner",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-name-owner",
        "current_adapter_id": None,
        "current_training_id": "training-name-owner",
    }
    targets = [
        {
            "target_id": "target-name-selected",
            "task_id": "sync-name-owner",
            "base_deployment_id": "deployment-name-shared",
            "base_deployment_replica_id": "replica-name-shared",
            "current_adapter_name": "adapter-name-owner",
            "current_adapter_id": None,
            "current_training_id": "training-name-owner",
        }
    ]
    adapter_calls, mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-name-shared", "replica-name-shared"): [
                {
                    "adapter_name": "adapter-name-owner",
                    "adapter_id": "adapter-name-owner-id",
                    "source_task_id": "training-name-owner",
                }
            ]
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _training_id: {
            "task_id": "sync-name-owner",
            "training_task_id": "training-name-owner",
            "target_id": "target-other",
            "loaded_adapter_id": "adapter-name-owner-id",
            "loaded_adapter_name": "adapter-name-owner",
            "status": SyncTrainingStatus.ADAPTER_LOADED,
            "target_config_snapshot": {
                "task_id": "sync-name-owner",
                "target_id": "target-other",
                "base_deployment_id": "deployment-name-shared",
                "base_deployment_replica_id": "replica-name-shared",
            },
        },
    )

    with pytest.raises(ValueError, match="target ownership"):
        post_training_handler.unload_current_adapter(
            "sync-name-owner",
            target_id="target-name-selected",
        )

    assert any(call[0] == "sync" for call in adapter_calls)
    assert not any(call[0].startswith("unload:") for call in adapter_calls)
    assert mark_calls == []


def test_unload_uses_persisted_training_snapshot_after_legacy_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-unload-snapshot",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-snapshot",
        "current_adapter_id": "adapter-snapshot-id",
        "current_training_id": "training-snapshot",
    }
    targets = [
        {
            "target_id": "target-unload-snapshot",
            "task_id": "sync-unload-snapshot",
            "base_deployment_id": "deployment-drifted",
            "base_deployment_replica_id": "replica-drifted",
            "current_adapter_name": "adapter-snapshot",
            "current_adapter_id": "adapter-snapshot-id",
            "current_training_id": "training-snapshot",
        }
    ]
    adapter_calls, _mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={
            ("deployment-original", "replica-original"): [
                {
                    "adapter_name": "adapter-snapshot",
                    "adapter_id": "adapter-snapshot-id",
                }
            ]
        },
    )

    post_training_handler.unload_current_adapter(
        "sync-unload-snapshot",
        target_id="target-unload-snapshot",
    )

    assert {call[1:] for call in adapter_calls} == {
        ("deployment-original", "replica-original")
    }
    assert [
        call for call in adapter_calls if call[0].startswith("unload:")
    ] == [
        (
            "unload:adapter-snapshot",
            "deployment-original",
            "replica-original",
        )
    ]


def test_replace_only_unloads_owned_adapter_from_three_party_shared_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-replace",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "old-one",
        "current_adapter_id": "old-1",
        "total_trainings": 1,
    }
    training = {
        "task_id": "sync-replace",
        "training_task_id": "training-new-1",
        "target_id": "target-1",
        "training_round": 2,
        "target_config_snapshot": {
            "target_id": "target-1",
            "task_id": "sync-replace",
            "base_deployment_id": "deployment-1",
            "base_deployment_replica_id": "replica-1",
        },
    }
    old_training = {
        "task_id": "sync-replace",
        "training_task_id": "training-old-1",
        "target_id": "target-1",
        "training_round": 1,
        "loaded_adapter_name": "old-one",
        "loaded_adapter_id": "old-1",
        "status": SyncTrainingStatus.ADAPTER_LOADED,
        "target_config_snapshot": {
            "target_id": "target-1",
            "task_id": "sync-replace",
            "base_deployment_id": "deployment-1",
            "base_deployment_replica_id": "replica-1",
        },
    }
    targets = [
        {
            "target_id": "target-1",
            "task_id": "sync-replace",
            "model_type": "embedding",
            "base_deployment_id": "deployment-1",
            "base_deployment_replica_id": "replica-1",
            "current_adapter_name": "old-one",
            "current_adapter_id": "old-1",
            "current_training_id": "training-old-1",
        },
        {
            "target_id": "target-2",
            "task_id": "sync-replace",
            "model_type": "embedding",
            "base_deployment_id": "deployment-2",
            "base_deployment_replica_id": "replica-2",
            "current_adapter_name": "other-two",
            "current_adapter_id": "other-2",
            "current_training_id": "training-old-2",
        },
    ]
    target_rows = {target["target_id"]: dict(target) for target in targets}
    target_updates: list[tuple[str, dict[str, Any]]] = []
    mark_calls: list[dict[str, Any]] = []
    unload_calls: list[str] = []
    other_task = {
        "task_id": "sync-other-runtime-owner",
        "current_adapter_name": "adapter-b",
        "current_adapter_id": "adapter-b-id",
    }

    monkeypatch.setattr(external_sync_service, "get_task_raw", lambda _id: dict(config))
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda training_id: dict(
            old_training if training_id == "training-old-1" else training
        ),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target_raw",
        lambda target_id: dict(target_rows[target_id]),
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda _id, **_kwargs: [dict(row) for row in target_rows.values()],
    )
    monkeypatch.setattr(external_sync_service, "update_task", lambda *_a, **_k: None)
    monkeypatch.setattr(
        external_sync_service,
        "update_training_status",
        lambda *_a, **_k: None,
    )

    def update_target(target_id: str, **kwargs: Any) -> dict[str, Any]:
        target_updates.append((target_id, kwargs))
        target_rows[target_id].update(kwargs)
        return dict(target_rows[target_id])

    monkeypatch.setattr(external_sync_service, "update_training_target", update_target)
    monkeypatch.setattr(
        external_sync_service,
        "mark_all_trainings_adapter_unloaded",
        lambda *_args, **kwargs: mark_calls.append(kwargs) or 1,
    )
    deployment_service = importlib.import_module(
        "train_factory.deployment.deployment_service"
    ).deployment_service
    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda deployment_id: _deployment(deployment_id),
    )

    class RecordingAdapterService:
        def sync_loaded_adapters(self, *_args: Any, **_kwargs: Any) -> int:
            return 0

        def list_loaded_adapters(self, *_args: Any, **_kwargs: Any):
            return [
                {
                    "adapter_name": "old-one",
                    "adapter_id": "old-1",
                    "deployment_id": "deployment-1",
                    "deployment_replica_id": "replica-1",
                    "source_task_id": "training-old-1",
                    "status": "loaded",
                    "user_id": "user-1",
                },
                {
                    "adapter_name": "adapter-b",
                    "adapter_id": "adapter-b-id",
                    "deployment_id": "deployment-1",
                    "deployment_replica_id": "replica-1",
                    "source_task_id": "training-b",
                    "status": "loaded",
                    "user_id": "user-1",
                },
                {
                    "adapter_name": "manual-adapter",
                    "adapter_id": "manual-adapter-id",
                    "deployment_id": "deployment-1",
                    "deployment_replica_id": "replica-1",
                    "source_model_id": "manual-model",
                    "status": "loaded",
                    "user_id": "user-1",
                },
            ]

        def get_adapter(self, adapter_id: str) -> dict[str, Any] | None:
            if adapter_id != "old-1":
                return None
            return {
                "adapter_name": "old-one",
                "adapter_id": "old-1",
                "deployment_id": "deployment-1",
                "deployment_replica_id": "replica-1",
                "source_task_id": "training-old-1",
                "status": "loaded",
                "user_id": "user-1",
            }

        def unload_adapter(
            self,
            _deployment_id: str,
            adapter_name: str,
            **_kwargs: Any,
        ) -> bool:
            unload_calls.append(adapter_name)
            return True

        def load_adapter(self, **_kwargs: Any) -> dict[str, str]:
            return {"adapter_id": "new-1"}

    monkeypatch.setattr(
        adapter_service_module,
        "AdapterService",
        RecordingAdapterService,
    )

    post_training_handler.load_adapter_for_training(
        "sync-replace",
        "training-new-1",
        "/models/new-1",
        target_id="target-1",
    )

    assert mark_calls == [
        {
            "exclude_training_task_id": "training-new-1",
            "loaded_adapter_id": "old-1",
            "loaded_adapter_name": "old-one",
        }
    ]
    assert unload_calls == ["old-one"]
    assert all(target_id != "target-2" for target_id, _kwargs in target_updates)
    assert target_rows["target-2"]["current_adapter_id"] == "other-2"
    assert other_task["current_adapter_id"] == "adapter-b-id"


def test_completed_training_loads_adapter_on_claim_snapshot_binding(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-binding-snapshot",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "base_deployment_replica_id": None,
        "total_trainings": 0,
    }
    training = {
        "task_id": "sync-binding-snapshot",
        "training_task_id": "training-binding-snapshot",
        "target_id": "target-binding-snapshot",
        "training_round": 1,
        "target_config_snapshot": {
            "target_id": "target-binding-snapshot",
            "task_id": "sync-binding-snapshot",
            "base_deployment_id": "deployment-original",
            "base_deployment_replica_id": "replica-original",
        },
    }
    drifted_target = {
        "target_id": "target-binding-snapshot",
        "task_id": "sync-binding-snapshot",
        "model_type": "embedding",
        "base_deployment_id": "deployment-drifted",
        "base_deployment_replica_id": "replica-drifted",
    }
    load_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda _task_id: dict(config),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_by_task_id",
        lambda _training_id: dict(training),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target_raw",
        lambda _target_id: dict(drifted_target),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda _task_id, **updates: config.update(updates) or dict(config),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        lambda _target_id, **updates: {**drifted_target, **updates},
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_status",
        lambda *_args, **_kwargs: None,
    )

    deployment_service = importlib.import_module(
        "train_factory.deployment.deployment_service"
    ).deployment_service
    monkeypatch.setattr(
        deployment_service,
        "get_deployment",
        lambda deployment_id: _deployment(deployment_id),
    )

    class RecordingAdapterService:
        def sync_loaded_adapters(self, *_args: Any, **_kwargs: Any) -> int:
            return 0

        def load_adapter(self, **kwargs: Any) -> dict[str, str]:
            load_calls.append(kwargs)
            return {"adapter_id": "adapter-snapshot"}

    monkeypatch.setattr(
        adapter_service_module,
        "AdapterService",
        RecordingAdapterService,
    )

    post_training_handler.load_adapter_for_training(
        "sync-binding-snapshot",
        "training-binding-snapshot",
        "/models/snapshot",
        replace=False,
        target_id="target-binding-snapshot",
    )

    assert load_calls[0]["deployment_id"] == "deployment-original"
    assert load_calls[0]["deployment_replica_id"] == "replica-original"


def test_unload_without_target_rejects_ambiguous_multi_target_task(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-ambiguous",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
        "current_adapter_name": "adapter-two",
        "current_adapter_id": "adapter-2",
    }
    targets = [
        {
            "target_id": f"target-{index}",
            "task_id": "sync-ambiguous",
            "base_deployment_id": f"deployment-{index}",
            "base_deployment_replica_id": f"replica-{index}",
            "current_adapter_name": f"adapter-{index}",
            "current_adapter_id": f"adapter-{index}",
        }
        for index in (1, 2)
    ]
    adapter_calls, _mark_calls = _install_unload_fakes(
        monkeypatch,
        config=config,
        targets=targets,
        adapters_by_binding={},
    )

    with pytest.raises(ValueError, match="target_id is required"):
        post_training_handler.unload_current_adapter("sync-ambiguous")

    assert adapter_calls == []


def test_unload_route_forwards_explicit_target_id(monkeypatch: pytest.MonkeyPatch):
    config = {
        "task_id": "sync-route",
        "user_id": "user-1",
        "status": "idle",
    }
    target = {"target_id": "target-route", "task_id": "sync-route"}
    calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        external_sync_service,
        "get_task",
        lambda _task_id: dict(config),
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_training_target",
        lambda _target_id: dict(target),
    )
    monkeypatch.setattr(
        post_training_handler,
        "unload_current_adapter",
        lambda task_id, target_id=None: calls.append((task_id, target_id)),
    )

    response = asyncio.run(
        sync_routes.unload_current_adapter(
            "sync-route",
            target_id="target-route",
            current_user=CURRENT_USER,
        )
    )

    assert response == {"message": "Adapter unloaded"}
    assert calls == [("sync-route", "target-route")]


def _install_sync_target_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_replica_id: str,
) -> list[tuple[str, str | None, str | None]]:
    target = {
        "target_id": "target-only",
        "task_id": "sync-index",
        "base_deployment_id": "deployment-target",
        "base_deployment_replica_id": target_replica_id,
        "current_adapter_name": "target-adapter",
        "current_adapter_id": "adapter-target",
    }
    adapter_calls: list[tuple[str, str | None, str | None]] = []

    monkeypatch.setattr(
        external_sync_service,
        "list_training_targets_raw",
        lambda _task_id, **_kwargs: [dict(target)],
    )
    monkeypatch.setattr(
        external_sync_service,
        "list_trainings",
        lambda **_kwargs: (
            [
                {
                    "training_task_id": "training-target",
                    "target_id": "target-only",
                    "loaded_adapter_name": "target-adapter",
                    "loaded_adapter_id": "adapter-target",
                }
            ],
            1,
        ),
    )

    deployment_service = importlib.import_module(
        "train_factory.deployment.deployment_service"
    ).deployment_service

    def resolve_replica_selection(
        deployment_id: str,
        replica_id: str | None,
        *,
        user_id: str | None,
        require_healthy: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert deployment_id == "deployment-target"
        assert user_id == "user-1"
        assert require_healthy is True
        return _deployment(deployment_id), {
            "replica_id": replica_id,
            "deployment_id": deployment_id,
            "status": "running",
            "health_status": "HEALTHY",
        }

    monkeypatch.setattr(
        deployment_service,
        "resolve_replica_selection",
        resolve_replica_selection,
    )

    class RecordingAdapterService:
        def list_loaded_adapters(
            self,
            deployment_id: str,
            *,
            deployment_replica_id: str | None,
            user_id: str | None,
        ) -> list[dict[str, Any]]:
            adapter_calls.append(
                (deployment_id, deployment_replica_id, user_id)
            )
            return [
                {
                    "adapter_id": "adapter-target",
                    "adapter_name": "target-adapter",
                    "source_task_id": "training-target",
                    "status": "loaded",
                }
            ]

    monkeypatch.setattr(
        adapter_service_module,
        "AdapterService",
        RecordingAdapterService,
    )
    return adapter_calls


def test_sync_targets_include_target_only_adapter_and_binding_in_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "task_id": "sync-index",
        "user_id": "user-1",
        "external_api_config_id": "tenant-1",
        "base_deployment_id": None,
    }
    embedding_config = {"model": "embedding-base"}

    calls = _install_sync_target_fakes(
        monkeypatch,
        target_replica_id="replica-one",
    )
    first = sync_worker._get_sync_targets(config, embedding_config)
    first_adapter = next(target for target in first if target["type"] == "adapter")

    calls_two = _install_sync_target_fakes(
        monkeypatch,
        target_replica_id="replica-two",
    )
    second = sync_worker._get_sync_targets(config, embedding_config)
    second_adapter = next(target for target in second if target["type"] == "adapter")

    assert first[0]["type"] == "base"
    assert first_adapter["target_id"] == "target-only"
    assert first_adapter["deployment_id"] == "deployment-target"
    assert first_adapter["deployment_replica_id"] == "replica-one"
    assert first_adapter["fingerprint"] != second_adapter["fingerprint"]
    assert calls == [("deployment-target", "replica-one", "user-1")]
    assert calls_two == [("deployment-target", "replica-two", "user-1")]


def test_scoped_training_unload_does_not_touch_other_target():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ExternalSyncTrainingDB.__table__.create(engine)
    service = ExternalSyncService()
    service.engine = engine
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTrainingDB(
                    task_id="sync-1",
                    training_task_id="training-target-1",
                    user_id="user-1",
                    target_id="target-1",
                    status=SyncTrainingStatus.ADAPTER_LOADED,
                ),
                ExternalSyncTrainingDB(
                    task_id="sync-1",
                    training_task_id="training-target-2",
                    user_id="user-1",
                    target_id="target-2",
                    status=SyncTrainingStatus.ADAPTER_LOADED,
                ),
                ExternalSyncTrainingDB(
                    task_id="sync-1",
                    training_task_id="training-base",
                    user_id="user-1",
                    target_id=None,
                    status=SyncTrainingStatus.ADAPTER_LOADED,
                ),
            ]
        )
        session.commit()

    updated = service.mark_all_trainings_adapter_unloaded(
        "sync-1",
        target_ids=("target-1",),
    )

    with Session(engine) as session:
        statuses = {
            row.training_task_id: row.status
            for row in session.exec(select(ExternalSyncTrainingDB)).all()
        }

    assert updated == 1
    assert statuses == {
        "training-target-1": SyncTrainingStatus.ADAPTER_UNLOADED,
        "training-target-2": SyncTrainingStatus.ADAPTER_LOADED,
        "training-base": SyncTrainingStatus.ADAPTER_LOADED,
    }


def test_exact_training_unload_does_not_touch_other_adapter_same_target():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ExternalSyncTrainingDB.__table__.create(engine)
    service = ExternalSyncService()
    service.engine = engine
    with Session(engine) as session:
        session.add_all(
            [
                ExternalSyncTrainingDB(
                    task_id="sync-exact",
                    training_task_id="training-adapter-a",
                    user_id="user-1",
                    target_id="target-exact",
                    loaded_adapter_name="adapter-a",
                    loaded_adapter_id="adapter-a-id",
                    status=SyncTrainingStatus.ADAPTER_LOADED,
                ),
                ExternalSyncTrainingDB(
                    task_id="sync-exact",
                    training_task_id="training-adapter-b",
                    user_id="user-1",
                    target_id="target-exact",
                    loaded_adapter_name="adapter-b",
                    loaded_adapter_id="adapter-b-id",
                    status=SyncTrainingStatus.ADAPTER_LOADED,
                ),
            ]
        )
        session.commit()

    updated = service.mark_all_trainings_adapter_unloaded(
        "sync-exact",
        loaded_adapter_id="adapter-a-id",
        loaded_adapter_name="adapter-a",
    )

    with Session(engine) as session:
        statuses = {
            row.training_task_id: row.status
            for row in session.exec(select(ExternalSyncTrainingDB)).all()
        }

    assert updated == 1
    assert statuses == {
        "training-adapter-a": SyncTrainingStatus.ADAPTER_UNLOADED,
        "training-adapter-b": SyncTrainingStatus.ADAPTER_LOADED,
    }
