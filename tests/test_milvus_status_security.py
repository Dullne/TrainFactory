import asyncio
from types import SimpleNamespace

from fastapi import HTTPException

from train_factory.api.routes import milvus_routes


CURRENT_USER = {"user_id": "user-1", "username": "alice", "is_admin": False}


def test_regular_user_milvus_status_hides_operator_endpoint_and_global_count(
    monkeypatch,
):
    class FakeClient:
        def list_collections(self):
            return ["owned", "foreign"]

        def close(self):
            return None

    monkeypatch.setattr(
        milvus_routes,
        "MilvusConfig",
        lambda: SimpleNamespace(host="milvus.internal", port=19530),
    )
    monkeypatch.setattr(milvus_routes, "_get_milvus_client", FakeClient)
    monkeypatch.setattr(
        milvus_routes,
        "_build_registry_map",
        lambda user_id=None: {"owned": {"user_id": user_id}},
    )

    response = asyncio.run(milvus_routes.get_milvus_status(CURRENT_USER))

    assert response == {"connected": True, "collection_count": 1}


def test_regular_user_milvus_status_sanitizes_connection_error(monkeypatch):
    monkeypatch.setattr(
        milvus_routes,
        "MilvusConfig",
        lambda: SimpleNamespace(host="milvus.internal", port=19530),
    )
    monkeypatch.setattr(
        milvus_routes,
        "_get_milvus_client",
        lambda: (_ for _ in ()).throw(
            HTTPException(
                status_code=503,
                detail="Milvus service is unavailable",
            )
        ),
    )

    response = asyncio.run(milvus_routes.get_milvus_status(CURRENT_USER))

    assert response == {"connected": False, "error": "Milvus is unavailable"}
