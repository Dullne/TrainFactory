import asyncio

import pytest
from fastapi import HTTPException

from train_factory.api.routes import milvus_routes


CURRENT_USER = {"user_id": "user-1", "username": "alice", "is_admin": False}


def test_registry_inventory_pages_until_reported_total(monkeypatch):
    rows = [
        {
            "collection_id": f"id-{index}",
            "collection_name": f"collection-{index:04d}",
        }
        for index in range(5001)
    ]
    calls = []

    def list_page(*, user_id=None, limit=100, offset=0, **_kwargs):
        calls.append((user_id, limit, offset))
        return rows[offset : offset + limit], len(rows)

    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "list_collections",
        list_page,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_linked_datasets",
        lambda _name: [],
    )

    registry = milvus_routes._build_registry_map(user_id="user-1")

    assert len(registry) == 5001
    assert set(registry) == {row["collection_name"] for row in rows}
    assert len(calls) > 1
    assert calls[0][2] == 0
    assert calls[-1][2] < len(rows)


def test_registry_inventory_does_not_return_partial_page_on_failure(
    monkeypatch,
):
    calls = []

    def list_page(*, user_id=None, limit=100, offset=0, **_kwargs):
        calls.append((user_id, limit, offset))
        if offset:
            raise RuntimeError("registry pagination failed")
        return [
            {
                "collection_id": f"id-{index}",
                "collection_name": f"collection-{index}",
            }
            for index in range(limit)
        ], limit + 1

    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "list_collections",
        list_page,
    )
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_linked_datasets",
        lambda _name: [],
    )

    with pytest.raises(RuntimeError, match="registry pagination failed"):
        milvus_routes._build_registry_map(user_id="user-1")

    assert len(calls) == 2


def test_registry_inventory_does_not_return_partial_links_on_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "list_collections",
        lambda **_kwargs: (
            [
                {"collection_id": "id-1", "collection_name": "first"},
                {"collection_id": "id-2", "collection_name": "second"},
            ],
            2,
        ),
    )

    def get_links(name):
        if name == "second":
            raise RuntimeError("registry links failed")
        return []

    monkeypatch.setattr(
        milvus_routes.milvus_collection_service,
        "get_linked_datasets",
        get_links,
    )

    with pytest.raises(RuntimeError, match="registry links failed"):
        milvus_routes._build_registry_map(user_id="user-1")


def test_regular_user_status_counts_all_owned_collections_over_old_cap(
    monkeypatch,
):
    names = [f"collection-{index:04d}" for index in range(5001)]

    class FakeClient:
        def list_collections(self):
            return names

        def close(self):
            return None

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FakeClient)
    monkeypatch.setattr(
        milvus_routes,
        "_build_registry_map",
        lambda user_id=None: {name: {"user_id": user_id} for name in names},
    )

    response = asyncio.run(milvus_routes.get_milvus_status(CURRENT_USER))

    assert response == {"connected": True, "collection_count": 5001}


@pytest.mark.parametrize(
    "unexpected_error",
    [
        RuntimeError("status programming bug"),
        PermissionError("status permission denied"),
        HTTPException(status_code=401, detail="status unauthorized"),
        HTTPException(status_code=503, detail="status configuration invalid"),
        HTTPException(status_code=503, detail="Milvus service is unavailable"),
    ],
)
def test_status_does_not_rewrite_non_connection_errors(
    monkeypatch,
    unexpected_error,
):
    events = []

    class FakeClient:
        def list_collections(self):
            raise unexpected_error

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FakeClient)

    with pytest.raises(type(unexpected_error)) as raised:
        asyncio.run(milvus_routes.get_milvus_status(CURRENT_USER))

    if isinstance(unexpected_error, HTTPException):
        assert raised.value.status_code == unexpected_error.status_code
    else:
        assert str(raised.value) == str(unexpected_error)
    assert events == ["close"]


def test_status_returns_generic_disconnected_for_runtime_transport_failure(
    monkeypatch,
):
    events = []

    class FakeClient:
        def list_collections(self):
            raise ConnectionRefusedError("secret endpoint")

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FakeClient)

    response = asyncio.run(milvus_routes.get_milvus_status(CURRENT_USER))

    assert response == {"connected": False, "error": "Milvus is unavailable"}
    assert events == ["close"]


@pytest.mark.parametrize(
    "connection_error",
    [ConnectionRefusedError("refused"), TimeoutError("timed out")],
)
def test_connection_failures_are_generic_503_and_partial_client_is_closed(
    monkeypatch,
    connection_error,
):
    events = []

    class FakeClient:
        def __init__(self, _config):
            events.append("init")

        def connect(self):
            events.append("connect")
            raise connection_error

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "MilvusClient", FakeClient)

    with pytest.raises(HTTPException) as raised:
        milvus_routes._get_milvus_client()

    assert raised.value.status_code == 503
    assert raised.value.detail == "Milvus service is unavailable"
    assert "docker compose" not in raised.value.detail.lower()
    assert events == ["init", "connect", "close"]


@pytest.mark.parametrize(
    "unexpected_error",
    [
        RuntimeError("programming bug"),
        PermissionError("authentication failed"),
        HTTPException(status_code=401, detail="invalid credentials"),
    ],
)
def test_non_connection_errors_are_not_rewritten_as_503_and_client_is_closed(
    monkeypatch,
    unexpected_error,
):
    events = []

    class FakeClient:
        def __init__(self, _config):
            events.append("init")

        def connect(self):
            events.append("connect")
            raise unexpected_error

        def close(self):
            events.append("close")

    monkeypatch.setattr(milvus_routes, "MilvusClient", FakeClient)

    with pytest.raises(type(unexpected_error)) as raised:
        milvus_routes._get_milvus_client()

    if isinstance(unexpected_error, HTTPException):
        assert raised.value.status_code == 401
    else:
        assert str(raised.value) == str(unexpected_error)
    assert events == ["init", "connect", "close"]


def test_pymilvus_connect_code_is_distinct_from_permission_and_config_errors():
    milvus_exception = type(
        "MilvusException",
        (Exception,),
        {"__module__": "pymilvus.exceptions"},
    )
    config_exception = type(
        "ConnectionConfigException",
        (Exception,),
        {"__module__": "pymilvus.exceptions"},
    )
    connect_failure = milvus_exception()
    connect_failure.code = 2
    permission_failure = milvus_exception()
    permission_failure.code = 3
    bad_config = config_exception()
    bad_config.code = 1

    assert milvus_routes._is_milvus_connection_error(connect_failure) is True
    assert milvus_routes._is_milvus_connection_error(permission_failure) is False
    assert milvus_routes._is_milvus_connection_error(bad_config) is False


@pytest.mark.parametrize(
    ("status_name", "expected"),
    [
        ("UNAVAILABLE", True),
        ("DEADLINE_EXCEEDED", True),
        ("PERMISSION_DENIED", False),
        ("UNAUTHENTICATED", False),
        ("INVALID_ARGUMENT", False),
    ],
)
def test_pymilvus_grpc_status_codes_only_classify_transport_failures(
    status_name,
    expected,
):
    milvus_exception = type(
        "MilvusException",
        (Exception,),
        {"__module__": "pymilvus.exceptions"},
    )
    failure = milvus_exception()
    failure.code = type("StatusCode", (), {"name": status_name})()

    assert milvus_routes._is_milvus_connection_error(failure) is expected


@pytest.mark.parametrize(
    ("status_name", "expected"),
    [
        ("DEADLINE_EXCEEDED", True),
        ("UNAVAILABLE", True),
        ("PERMISSION_DENIED", False),
    ],
)
def test_direct_grpc_rpc_errors_only_classify_transport_failures(
    status_name,
    expected,
):
    status = type("StatusCode", (), {"name": status_name})()
    rpc_error_type = type(
        "_InactiveRpcError",
        (Exception,),
        {
            "__module__": "grpc._channel",
            "code": lambda self: status,
        },
    )

    assert milvus_routes._is_milvus_connection_error(rpc_error_type()) is expected
