import asyncio
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from train_factory.api.routes import milvus_routes
from train_factory.storage.services.milvus_collection_service import (
    MilvusCollectionDeletionOwnerConflictError,
    MilvusCollectionUnavailableError,
)


CURRENT_USER = {"user_id": "user-1", "username": "alice", "is_admin": False}
AUTH_DISABLED_USER = {
    "user_id": "anonymous",
    "username": "anonymous",
    "is_admin": True,
}


def _owned_collection(name):
    return {"collection_name": name, "user_id": "user-1"}


@pytest.fixture(autouse=True)
def _no_existing_manual_creation(monkeypatch):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_manual_creation_reservation",
        lambda _name, *, user_id: None,
        raising=False,
    )


def test_milvus_client_reports_preexisting_inner_name_race(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(
            Collection=object,
            CollectionSchema=object,
            FieldSchema=object,
            DataType=object,
        ),
    )
    client = object.__new__(milvus_routes.MilvusClient)
    monkeypatch.setattr(client, "collection_exists", lambda _name: True)

    assert client.create_collection("raced_collection", 768) is False


def test_manual_collection_create_rejects_durable_sync_namespace_before_remote_create(
    monkeypatch,
):
    events = []
    reservation = {
        "collection_name": "tf_sync_v3_reserved_base",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **kwargs: events.append(("reserve", kwargs["collection_name"]))
        or reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda names: events.append(("namespace", tuple(names)))
        or ["sync-task-1"],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda name, **kwargs: events.append(
            ("cancel", name, kwargs["collection_id"], kwargs["user_id"])
        )
        or True,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: pytest.fail("reserved sync namespace must not reach Milvus"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.create_collection(
                milvus_routes.CreateCollectionRequest(
                    name="tf_sync_v3_reserved_base",
                    dim=768,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("reserve", "tf_sync_v3_reserved_base"),
        ("namespace", ("tf_sync_v3_reserved_base",)),
        (
            "cancel",
            "tf_sync_v3_reserved_base",
            "reservation-id",
            "user-1",
        ),
    ]


def test_manual_collection_create_activates_reserved_registry_after_remote_create(
    monkeypatch,
):
    events = []
    reservation = {
        "collection_name": "manual-collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **kwargs: events.append(("reserve", kwargs["collection_name"]))
        or reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda names: events.append(("namespace", tuple(names))) or [],
    )

    class FakeClient:
        def collection_exists(self, name):
            events.append(("exists", name))
            return False

        def create_collection(self, name, dim, **_kwargs):
            events.append(("create", name, dim))

        def get_collection_info(self, name):
            events.append(("info", name))
            return {
                "name": name,
                "dim": 768,
                "schema": [
                    {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                    {"name": "chunk_content", "dtype": "VARCHAR"},
                    {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
                    {"name": "metadata", "dtype": "JSON"},
                ],
                "indexes": [
                    {"field_name": "vector", "metric_type": "COSINE"}
                ],
                "hybrid_enabled": False,
            }

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FakeClient)
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "activate_manual_collection",
        lambda name, **kwargs: events.append(
            ("activate", name, kwargs["collection_id"], kwargs["user_id"])
        )
        or {**reservation, "status": "active"},
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "register_collection",
        lambda **_kwargs: pytest.fail(
            "manual create must not use post-create generic registration"
        ),
    )

    result = asyncio.run(
        milvus_routes.create_collection(
            milvus_routes.CreateCollectionRequest(
                name="manual-collection",
                dim=768,
            ),
            CURRENT_USER,
        )
    )

    assert result["collection_id"] == "reservation-id"
    assert events == [
        ("reserve", "manual_collection"),
        ("namespace", ("manual_collection",)),
        ("exists", "manual_collection"),
        ("create", "manual_collection", 768),
        ("info", "manual_collection"),
        ("activate", "manual_collection", "reservation-id", "user-1"),
        "close",
    ]


def test_manual_collection_create_retains_reservation_after_remote_mutation_starts(
    monkeypatch,
):
    reservation = {
        "collection_name": "manual_collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **_kwargs: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "an uncertain remote create must retain its durable reservation"
        ),
        raising=False,
    )

    class FailingClient:
        def collection_exists(self, _name):
            return False

        def create_collection(self, _name, _dim, **_kwargs):
            raise RuntimeError("remote create failed after mutation began")

        def close(self):
            return None

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FailingClient)

    with pytest.raises(RuntimeError, match="remote create failed"):
        asyncio.run(
            milvus_routes.create_collection(
                milvus_routes.CreateCollectionRequest(
                    name="manual_collection",
                    dim=768,
                ),
                CURRENT_USER,
            )
        )


def test_manual_collection_retry_reconciles_compatible_remote_collection(
    monkeypatch,
):
    events = []
    reservation = {
        "collection_name": "manual_collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
        "dim": 768,
        "metric_type": "COSINE",
        "hybrid_enabled": True,
        "_newly_created": False,
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_manual_creation_reservation",
        lambda _name, *, user_id: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **_kwargs: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )

    class ExistingClient:
        def collection_exists(self, name):
            events.append(("exists", name))
            return True

        def get_collection_info(self, name):
            events.append(("info", name))
            return {
                "name": name,
                "dim": 768,
                "schema": [
                    {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                    {"name": "chunk_content", "dtype": "VARCHAR"},
                    {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
                    {"name": "metadata", "dtype": "JSON"},
                    {"name": "sparse_vector", "dtype": "SPARSE_FLOAT_VECTOR"},
                ],
                "indexes": [
                    {
                        "field_name": "vector",
                        "index_type": "AUTOINDEX",
                        "metric_type": "COSINE",
                    },
                    {
                        "field_name": "sparse_vector",
                        "index_type": "AUTOINDEX",
                        "metric_type": "BM25",
                    },
                ],
                "hybrid_enabled": True,
            }

        def create_collection(self, *_args, **_kwargs):
            pytest.fail("a compatible remote collection must not be recreated")

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", ExistingClient)
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "activate_manual_collection",
        lambda name, **kwargs: events.append(
            ("activate", name, kwargs["collection_id"], kwargs["user_id"])
        )
        or {**reservation, "status": "active"},
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "a durable retry reservation must not be cancelled"
        ),
        raising=False,
    )

    result = asyncio.run(
        milvus_routes.create_collection(
            milvus_routes.CreateCollectionRequest(
                name="manual_collection",
                dim=768,
                metric_type="COSINE",
                # The durable reservation, not a changed retry payload, wins.
                enable_hybrid=False,
            ),
            CURRENT_USER,
        )
    )

    assert result["collection_id"] == "reservation-id"
    assert events == [
        ("exists", "manual_collection"),
        ("info", "manual_collection"),
        ("activate", "manual_collection", "reservation-id", "user-1"),
        "close",
    ]


@pytest.mark.parametrize(
    "current_config",
    [
        None,
        {
            "config_id": "config-1",
            "model_type": "embedding",
            "model_name": "changed-model",
            "api_endpoint": "https://changed.invalid/v1",
            "user_id": "user-1",
        },
    ],
    ids=["deleted-config", "changed-config"],
)
def test_manual_collection_retry_uses_durable_schema_when_remote_is_missing(
    monkeypatch,
    current_config,
):
    events = []
    reservation = {
        "collection_name": "manual_collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
        "dim": 768,
        "metric_type": "COSINE",
        "hybrid_enabled": True,
        "embedding_config_id": "config-1",
        "embedding_model": "original-model",
        "embedding_endpoint": "https://original.invalid/v1",
        "_newly_created": False,
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_manual_creation_reservation",
        lambda _name, *, user_id: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **kwargs: events.append(
            ("reserve-hybrid", kwargs["hybrid_enabled"])
        )
        or reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    from train_factory.storage.services.model_config_service import (
        model_config_service,
    )

    config_reads = []
    monkeypatch.setattr(
        model_config_service,
        "get_config",
        lambda config_id: config_reads.append(config_id) or current_config,
    )

    class MissingClient:
        def collection_exists(self, _name):
            return False

        def create_collection(self, name, dim, **kwargs):
            events.append(
                (
                    "create",
                    name,
                    dim,
                    kwargs["metric_type"],
                    kwargs["enable_hybrid"],
                )
            )

        def get_collection_info(self, name):
            events.append(("info", name))
            return {
                "name": name,
                "dim": 768,
                "schema": [
                    {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                    {"name": "chunk_content", "dtype": "VARCHAR"},
                    {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
                    {"name": "metadata", "dtype": "JSON"},
                    {"name": "sparse_vector", "dtype": "SPARSE_FLOAT_VECTOR"},
                ],
                "indexes": [
                    {"field_name": "vector", "metric_type": "COSINE"},
                    {"field_name": "sparse_vector", "metric_type": "BM25"},
                ],
                "hybrid_enabled": True,
            }

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", MissingClient)
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "activate_manual_collection",
        lambda name, **_kwargs: events.append(("activate", name))
        or {**reservation, "status": "active"},
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "a durable retry reservation must not be cancelled"
        ),
        raising=False,
    )

    asyncio.run(
        milvus_routes.create_collection(
            milvus_routes.CreateCollectionRequest(
                name="manual_collection",
                dim=768,
                metric_type="COSINE",
                embedding_config_id="config-1",
                enable_hybrid=False,
            ),
            CURRENT_USER,
        )
    )

    assert events == [
        ("reserve-hybrid", False),
        ("create", "manual_collection", 768, "COSINE", True),
        ("info", "manual_collection"),
        ("activate", "manual_collection"),
        "close",
    ]
    assert config_reads == []


@pytest.mark.parametrize(
    "remote_info",
    [
        {
            "dim": 1536,
            "schema": [{"name": "vector", "dim": 1536}],
            "indexes": [{"field_name": "vector", "metric_type": "COSINE"}],
            "hybrid_enabled": False,
        },
        {
            "dim": 768,
            "schema": [{"name": "vector", "dim": 768}],
            "indexes": [{"field_name": "vector", "metric_type": "L2"}],
            "hybrid_enabled": False,
        },
        {
            "dim": 768,
            "schema": [
                {"name": "vector", "dim": 768},
                {"name": "sparse_vector"},
            ],
            "indexes": [{"field_name": "vector", "metric_type": "COSINE"}],
            "hybrid_enabled": True,
        },
        {
            "dim": 768,
            "schema": [{"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768}],
            "indexes": [{"field_name": "vector", "metric_type": "COSINE"}],
            "hybrid_enabled": False,
        },
        {
            "_expected_hybrid": True,
            "dim": 768,
            "schema": [
                {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                {"name": "chunk_content", "dtype": "VARCHAR"},
                {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
                {"name": "metadata", "dtype": "JSON"},
                {"name": "sparse_vector", "dtype": "SPARSE_FLOAT_VECTOR"},
            ],
            "indexes": [{"field_name": "vector", "metric_type": "COSINE"}],
            "hybrid_enabled": True,
        },
        {
            "_expected_hybrid": True,
            "dim": 768,
            "schema": [
                {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                {"name": "chunk_content", "dtype": "VARCHAR"},
                {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
                {"name": "metadata", "dtype": "JSON"},
                {"name": "sparse_vector", "dtype": "SPARSE_FLOAT_VECTOR"},
            ],
            "indexes": [
                {"field_name": "vector", "metric_type": "COSINE"},
                {"field_name": "sparse_vector", "metric_type": "L2"},
            ],
            "hybrid_enabled": True,
        },
        {
            "_expected_hybrid": True,
            "dim": 768,
            "schema": [
                {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                {"name": "chunk_content", "dtype": "VARCHAR"},
                {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
                {"name": "metadata", "dtype": "JSON"},
                {"name": "sparse_vector", "dtype": "FLOAT_VECTOR"},
            ],
            "indexes": [
                {"field_name": "vector", "metric_type": "COSINE"},
                {"field_name": "sparse_vector", "metric_type": "BM25"},
            ],
            "hybrid_enabled": True,
        },
    ],
    ids=[
        "dimension",
        "metric",
        "hybrid-schema",
        "base-schema",
        "missing-sparse-index",
        "wrong-sparse-metric",
        "wrong-sparse-dtype",
    ],
)
def test_manual_collection_retry_rejects_incompatible_remote_collection(
    monkeypatch,
    remote_info,
):
    expected_hybrid = bool(remote_info.get("_expected_hybrid", False))
    reservation = {
        "collection_name": "manual_collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
        "dim": 768,
        "metric_type": "COSINE",
        "hybrid_enabled": expected_hybrid,
        "_newly_created": False,
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **_kwargs: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )

    class IncompatibleClient:
        def collection_exists(self, _name):
            return True

        def get_collection_info(self, name):
            return {
                "name": name,
                **{
                    key: value
                    for key, value in remote_info.items()
                    if not key.startswith("_")
                },
            }

        def close(self):
            return None

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", IncompatibleClient)
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "activate_manual_collection",
        lambda *_args, **_kwargs: pytest.fail(
            "an incompatible remote collection must not be activated"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "an existing durable reservation must not be cancelled"
        ),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.create_collection(
                milvus_routes.CreateCollectionRequest(
                    name="manual_collection",
                    dim=768,
                    metric_type="COSINE",
                    enable_hybrid=expected_hybrid,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert "does not match the durable creation reservation" in str(
        exc_info.value.detail
    )


def test_manual_collection_retry_keeps_durable_reservation_on_preflight_error(
    monkeypatch,
):
    reservation = {
        "collection_name": "manual_collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
        "_newly_created": False,
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **_kwargs: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: ["sync-task-1"],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "an existing durable reservation must survive preflight errors"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: pytest.fail("namespace conflict must stop before Milvus"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.create_collection(
                milvus_routes.CreateCollectionRequest(
                    name="manual_collection",
                    dim=768,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409


def test_manual_collection_create_revalidates_after_concurrent_name_appearance(
    monkeypatch,
):
    events = []
    reservation = {
        "collection_name": "manual_collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
        "dim": 768,
        "metric_type": "COSINE",
        "hybrid_enabled": False,
        "_newly_created": True,
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **_kwargs: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )

    class ConcurrentClient:
        def collection_exists(self, _name):
            return False

        def create_collection(self, name, _dim, **_kwargs):
            # Mirrors MilvusClient's current silent skip if another actor wins
            # the inner has_collection race.
            events.append(("create-silently-skipped", name))

        def get_collection_info(self, name):
            events.append(("info", name))
            return {
                "name": name,
                "dim": 1536,
                "schema": [
                    {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                    {"name": "chunk_content", "dtype": "VARCHAR"},
                    {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 1536},
                    {"name": "metadata", "dtype": "JSON"},
                ],
                "indexes": [
                    {"field_name": "vector", "metric_type": "COSINE"}
                ],
                "hybrid_enabled": False,
            }

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", ConcurrentClient)
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "activate_manual_collection",
        lambda *_args, **_kwargs: pytest.fail(
            "a concurrently-created incompatible collection must not activate"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "post-create uncertainty must retain the durable reservation"
        ),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.create_collection(
                milvus_routes.CreateCollectionRequest(
                    name="manual_collection",
                    dim=768,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("create-silently-skipped", "manual_collection"),
        ("info", "manual_collection"),
        "close",
    ]


def test_manual_collection_create_rejects_compatible_concurrent_name_owner(
    monkeypatch,
):
    events = []
    reservation = {
        "collection_name": "manual_collection",
        "collection_id": "reservation-id",
        "status": "creating",
        "user_id": "user-1",
        "dim": 768,
        "metric_type": "COSINE",
        "hybrid_enabled": False,
        "_newly_created": True,
    }
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "reserve_manual_collection",
        lambda **_kwargs: reservation,
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )

    class ConcurrentCompatibleClient:
        def collection_exists(self, _name):
            return False

        def create_collection(self, name, _dim, **_kwargs):
            events.append(("inner-preexisting", name))
            return False

        def get_collection_info(self, name):
            events.append(("info", name))
            return {
                "name": name,
                "dim": 768,
                "schema": [
                    {"name": "chunk_id", "dtype": "VARCHAR", "is_primary": True},
                    {"name": "chunk_content", "dtype": "VARCHAR"},
                    {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
                    {"name": "metadata", "dtype": "JSON"},
                ],
                "indexes": [
                    {"field_name": "vector", "metric_type": "COSINE"}
                ],
                "hybrid_enabled": False,
            }

        def close(self):
            events.append("close")

    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        ConcurrentCompatibleClient,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "activate_manual_collection",
        lambda *_args, **_kwargs: pytest.fail(
            "a compatible but foreign collection must not be adopted"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "cancel_manual_collection_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "post-create ownership uncertainty must retain the reservation"
        ),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.create_collection(
                milvus_routes.CreateCollectionRequest(
                    name="manual_collection",
                    dim=768,
                ),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("inner-preexisting", "manual_collection"),
        "close",
    ]


def test_link_collection_rejects_foreign_dataset_before_mutation(monkeypatch):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        _owned_collection,
    )
    monkeypatch.setattr(
        milvus_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {"dataset_id": dataset_id, "user_id": "user-2"},
    )
    mutations = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "link_dataset",
        lambda **kwargs: mutations.append(kwargs),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.link_dataset_to_collection(
                "owned_collection",
                milvus_routes.LinkDatasetRequest(dataset_id="foreign"),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert mutations == []


def test_link_collection_uses_server_dataset_metadata(monkeypatch):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        _owned_collection,
    )
    monkeypatch.setattr(
        milvus_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "dataset_id": dataset_id,
            "dataset_name": "server-name",
            "user_id": "user-1",
        },
    )
    mutations = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "link_dataset",
        lambda **kwargs: mutations.append(kwargs) or kwargs,
    )

    asyncio.run(
        milvus_routes.link_dataset_to_collection(
            "owned_collection",
            milvus_routes.LinkDatasetRequest(
                dataset_id="owned-dataset",
                dataset_name="spoofed-name",
                task_id="foreign-task",
                chunk_count=10,
            ),
            CURRENT_USER,
        )
    )

    assert mutations == [
        {
            "collection_name": "owned_collection",
            "dataset_id": "owned-dataset",
            "dataset_name": "server-name",
            "chunk_count": 10,
            "task_id": None,
        }
    ]


def test_link_collection_maps_deletion_fence_race_to_conflict(monkeypatch):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        _owned_collection,
    )
    monkeypatch.setattr(
        milvus_routes.dataset_service,
        "get_dataset",
        lambda dataset_id: {"dataset_id": dataset_id, "user_id": "user-1"},
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "link_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(
            MilvusCollectionUnavailableError("collection is deleting")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.link_dataset_to_collection(
                "owned_collection",
                milvus_routes.LinkDatasetRequest(dataset_id="owned-dataset"),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 409


def test_manual_collection_delete_fences_before_consumer_preflight(monkeypatch):
    events = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "collection-id-1",
            "user_id": "user-1",
            "status": "active",
        },
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **kwargs: events.append(
            ("fence", tuple(names), kwargs["deletion_owner"])
        )
        or SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=(),
        ),
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda names: events.append(("consumers", tuple(names)))
        or ["generation-1"],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "restore_deletion_fences",
        lambda names, **kwargs: events.append(
            ("restore", tuple(names), kwargs["deletion_owner"])
        )
        or True,
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: pytest.fail("remote drop must follow consumer preflight"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("fence", ("owned_collection",), "manual:collection-id-1"),
        ("consumers", ("owned_collection",)),
        ("restore", ("owned_collection",), "manual:collection-id-1"),
    ]


def test_manual_collection_delete_can_abort_exact_creating_reservation(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "reservation-id",
            "user_id": "user-1",
            "status": "creating",
        },
    )

    def acquire(names, **kwargs):
        events.append(
            (
                "fence",
                tuple(names),
                kwargs["deletion_owner"],
                kwargs["allow_manual_creating"],
            )
        )
        return SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=(),
            previous_creating=tuple(names),
        )

    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        acquire,
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )

    class FakeClient:
        def drop_collection(self, name):
            events.append(("drop", name))

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FakeClient)
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "delete_collection",
        lambda name, **kwargs: events.append(
            ("finalize", name, kwargs["deletion_owner"])
        )
        or True,
    )

    result = asyncio.run(
        milvus_routes.delete_collection("partial_collection", CURRENT_USER)
    )

    assert "已删除" in result["message"]
    assert events == [
        ("fence", ("partial_collection",), "manual:reservation-id", True),
        ("drop", "partial_collection"),
        ("finalize", "partial_collection", "manual:reservation-id"),
        "close",
    ]


def test_manual_collection_delete_rejects_active_deep_evaluation(monkeypatch):
    events = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "collection-id-1",
            "user_id": "user-1",
            "status": "active",
        },
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **kwargs: events.append(
            ("fence", tuple(names), kwargs["deletion_owner"])
        )
        or SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=(),
        ),
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda names: events.append(("generation", tuple(names))) or [],
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda names: events.append(("deep-evaluation", tuple(names)))
        or ["deep-evaluation-1"],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "restore_deletion_fences",
        lambda names, **kwargs: events.append(
            ("restore", tuple(names), kwargs["deletion_owner"])
        )
        or True,
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: pytest.fail("remote drop must follow consumer preflight"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("fence", ("owned_collection",), "manual:collection-id-1"),
        ("generation", ("owned_collection",)),
        ("deep-evaluation", ("owned_collection",)),
        ("restore", ("owned_collection",), "manual:collection-id-1"),
    ]


def test_manual_collection_delete_rejects_active_sync_writer(monkeypatch):
    events = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "collection-id-1",
            "user_id": "user-1",
            "status": "active",
        },
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **kwargs: events.append(("fence", tuple(names)))
        or SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=(),
        ),
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda names: events.append(("sync", tuple(names))) or ["sync-1"],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "restore_deletion_fences",
        lambda names, **_kwargs: events.append(("restore", tuple(names)))
        or True,
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: pytest.fail("active sync collection must not be dropped"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )

    assert exc_info.value.status_code == 409
    assert events == [
        ("fence", ("owned_collection",)),
        ("sync", ("owned_collection",)),
        ("restore", ("owned_collection",)),
    ]


@pytest.mark.parametrize(
    "failure_stage",
    ["generation", "deep-evaluation", "sync", "connect"],
)
def test_manual_collection_delete_restores_new_fence_before_first_drop_on_error(
    monkeypatch,
    failure_stage,
):
    events = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "collection-id-1",
            "user_id": "user-1",
            "status": "active",
        },
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **_kwargs: SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=(),
        ),
    )

    def stage_result(stage):
        events.append(stage)
        if failure_stage == stage:
            raise RuntimeError(f"{stage} failed")
        return []

    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda _names: stage_result("generation"),
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda _names: stage_result("deep-evaluation"),
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: stage_result("sync"),
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "restore_deletion_fences",
        lambda names, **kwargs: events.append(
            (
                "restore",
                tuple(names),
                kwargs["deletion_owner"],
                tuple(kwargs["created_placeholders"]),
            )
        )
        or True,
    )

    def connect():
        events.append("connect")
        if failure_stage == "connect":
            raise RuntimeError("connect failed")
        pytest.fail("a pre-drop failure must stop before the remote client")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", connect)

    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )

    assert events[-1] == (
        "restore",
        ("owned_collection",),
        "manual:collection-id-1",
        (),
    )


def test_manual_collection_delete_does_not_restore_existing_durable_intent(
    monkeypatch,
):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "collection-id-1",
            "user_id": "user-1",
            "status": "deleting",
        },
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_deletion_fence_owner",
        lambda _name: "manual:collection-id-1",
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        lambda _names, **_kwargs: SimpleNamespace(
            newly_fenced=(),
            created_placeholders=(),
        ),
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "restore_deletion_fences",
        lambda *_args, **_kwargs: pytest.fail(
            "a retried durable deletion intent must never be restored"
        ),
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: (_ for _ in ()).throw(RuntimeError("connect failed")),
    )

    with pytest.raises(RuntimeError, match="connect failed"):
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )


def test_manual_collection_delete_retains_fence_once_remote_drop_starts(
    monkeypatch,
):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "collection-id-1",
            "user_id": "user-1",
            "status": "active",
        },
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        lambda names, **_kwargs: SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=(),
        ),
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "restore_deletion_fences",
        lambda *_args, **_kwargs: pytest.fail(
            "an uncertain remote drop must retain the durable fence"
        ),
    )

    class FailingDropClient:
        def drop_collection(self, _name):
            raise RuntimeError("drop failed after mutation began")

        def close(self):
            return None

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FailingDropClient)

    with pytest.raises(RuntimeError, match="drop failed"):
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )


@pytest.mark.parametrize(
    ("registry", "expected_owner_user_id"),
    [
        (None, None),
        (
            {
                "collection_name": "legacy_collection",
                "collection_id": "legacy-id",
                "user_id": "legacy-user",
                "status": "active",
            },
            "legacy-user",
        ),
    ],
)
def test_auth_disabled_manual_delete_supports_unregistered_and_legacy_owner(
    monkeypatch,
    registry,
    expected_owner_user_id,
):
    events = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda _name: registry,
    )

    def acquire(
        names,
        *,
        deletion_owner,
        user_id,
        expected_collection_ids,
    ):
        assert user_id == expected_owner_user_id
        assert deletion_owner.startswith("manual:")
        assert len(deletion_owner) <= 128
        assert expected_collection_ids == {
            "legacy_collection": (
                registry.get("collection_id") if registry else None
            )
        }
        events.append(("fence", tuple(names), deletion_owner))
        return SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=tuple(names) if registry is None else (),
        )

    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        acquire,
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )

    class FakeClient:
        def drop_collection(self, name):
            events.append(("drop", name))

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FakeClient)
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "delete_collection",
        lambda name, *, deletion_owner: events.append(
            ("finalize", name, deletion_owner)
        )
        or True,
    )

    result = asyncio.run(
        milvus_routes.delete_collection(
            "legacy_collection",
            AUTH_DISABLED_USER,
        )
    )

    assert result["message"]
    assert events[1] == ("drop", "legacy_collection")
    assert events[-1] == "close"


def test_manual_collection_delete_rejects_foreign_deletion_owner(monkeypatch):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "collection-id-1",
            "user_id": "user-1",
            "status": "deleting",
        },
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_deletion_fence_owner",
        lambda _name: "sync:task-1",
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        lambda *_args, **_kwargs: pytest.fail(
            "a foreign deletion owner must be rejected before acquisition"
        ),
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: pytest.fail("a foreign deletion owner must not reach Milvus"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )

    assert exc_info.value.status_code == 409


def test_manual_collection_delete_passes_registry_snapshot_to_fence(monkeypatch):
    events = []
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {
            "collection_name": name,
            "collection_id": "old-collection-id",
            "user_id": "user-1",
            "status": "active",
        },
    )

    def acquire(
        names,
        *,
        deletion_owner,
        user_id,
        expected_collection_ids=None,
    ):
        events.append((tuple(names), expected_collection_ids))
        if expected_collection_ids == {
            "owned_collection": "old-collection-id"
        }:
            raise MilvusCollectionDeletionOwnerConflictError(
                "collection identity changed"
            )
        return SimpleNamespace(
            newly_fenced=tuple(names),
            created_placeholders=(),
        )

    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "acquire_deletion_fences",
        acquire,
    )
    monkeypatch.setattr(
        milvus_routes.generation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.deep_evaluation_task_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes.external_sync_service,
        "list_active_collection_consumers",
        lambda _names: [],
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: pytest.fail("an ABA conflict must not reach Milvus"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            milvus_routes.delete_collection("owned_collection", CURRENT_USER)
        )

    assert exc_info.value.status_code == 409
    assert events == [
        (
            ("owned_collection",),
            {"owned_collection": "old-collection-id"},
        )
    ]
