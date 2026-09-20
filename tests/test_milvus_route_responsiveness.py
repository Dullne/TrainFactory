"""Milvus request work must leave the API event loop available to other requests."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from train_factory.api.routes import milvus_routes


USER = {"user_id": "user-1", "username": "alice", "is_admin": False}


def _app():
    app = FastAPI()
    app.include_router(milvus_routes.router, prefix="/milvus")

    async def user():
        return USER

    app.dependency_overrides[milvus_routes.get_current_user] = user

    @app.get("/ping")
    async def ping():
        return {"available": True}

    return app


async def _assert_responsive(app, entered, release, method, path, payload=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        request = asyncio.create_task(client.request(method, path, json=payload))
        try:
            assert await asyncio.to_thread(entered.wait, 3), "blocking work never started"
            assert not request.done(), "Milvus work blocked the event loop until it finished"
            response = await asyncio.wait_for(client.get("/ping"), timeout=1)
            assert response.json() == {"available": True}
        finally:
            release.set()
            response = await request
        return response


@pytest.mark.parametrize("method,path,payload,boundary", (
    ("GET", "/status", None, "client"),
    ("GET", "/collections", None, "client"),
    ("GET", "/collections/owned", None, "ownership"),
    ("POST", "/collections", {"name": "owned", "dim": 8}, "reservation"),
    ("DELETE", "/collections/owned", None, "ownership"),
    ("GET", "/collections/owned/entities", None, "ownership"),
    ("POST", "/collections/owned/search", {"query_text": "hello", "search_mode": "sparse"}, "ownership"),
    ("POST", "/collections/owned/load", None, "ownership"),
    ("POST", "/collections/owned/release", None, "ownership"),
    ("GET", "/collections/owned/datasets", None, "ownership"),
    ("POST", "/collections/owned/datasets", {"dataset_id": "dataset-1"}, "ownership"),
    ("DELETE", "/collections/owned/datasets/dataset-1", None, "ownership"),
))
def test_every_milvus_route_offloads_its_initial_io(
    monkeypatch, method, path, payload, boundary,
):
    entered = threading.Event()
    release = threading.Event()

    def blocked(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=1.5), "API event loop could not release blocked I/O"
        raise HTTPException(status_code=503, detail="test boundary unavailable")

    if boundary == "reservation":
        monkeypatch.setattr(milvus_routes.milvus_collection_service, "get_manual_creation_reservation", blocked)
    else:
        name = "_get_milvus_client" if boundary == "client" else "_get_owned_collection"
        monkeypatch.setattr(milvus_routes, name, blocked)

    response = asyncio.run(_assert_responsive(
        _app(), entered, release, method, f"/milvus{path}", payload,
    ))
    assert response.status_code == 503


@pytest.mark.parametrize("operation", ("browse", "search"))
@pytest.mark.parametrize("stage", ("connect", "rpc", "close"))
def test_milvus_client_lifecycle_does_not_block_other_requests(monkeypatch, operation, stage):
    entered = threading.Event()
    release = threading.Event()
    closed = []

    def block():
        entered.set()
        assert release.wait(timeout=1.5), "API event loop could not release blocked I/O"

    def query(*_args, **_kwargs):
        if stage == "rpc":
            block()
        return ([], 0) if operation == "browse" else []

    def close():
        if stage == "close":
            block()
        closed.append(True)

    fake = SimpleNamespace(
        collection_exists=lambda _name: True,
        supports_hybrid=lambda _name: True,
        query_entities=query, hybrid_search=query, close=close,
    )

    def connect():
        if stage == "connect":
            block()
        return fake

    monkeypatch.setattr(milvus_routes, "_get_owned_collection", lambda *_args: {"user_id": "user-1"})
    monkeypatch.setattr(milvus_routes, "_get_milvus_client", connect)
    method = "GET" if operation == "browse" else "POST"
    suffix = "entities" if operation == "browse" else "search"
    payload = None if operation == "browse" else {"query_text": "hello", "search_mode": "sparse"}
    response = asyncio.run(_assert_responsive(
        _app(), entered, release, method, f"/milvus/collections/owned/{suffix}", payload,
    ))
    assert response.status_code == 200
    assert closed == [True]


def test_dense_search_embedding_client_setup_leaves_api_event_loop_available(monkeypatch):
    from train_factory.generation.clients.embedding_client import EmbeddingClient
    from train_factory.storage.services.model_config_service import model_config_service

    entered = threading.Event()
    release = threading.Event()
    closed = []

    async def close_embedding():
        closed.append("embedding")

    def create_http_client(_self):
        # Real client setup synchronously resolves endpoint policy/DB/DNS.
        entered.set()
        assert release.wait(timeout=1.5), "API event loop could not release blocked I/O"
        return SimpleNamespace(aclose=close_embedding)

    async def embed_batch(_self, _texts):
        return [[0.1, 0.2]]

    monkeypatch.setattr(EmbeddingClient, "_new_http_client", create_http_client)
    monkeypatch.setattr(EmbeddingClient, "_embed_batch", embed_batch)
    monkeypatch.setattr(model_config_service, "get_config", lambda _id: {
        "user_id": "user-1", "model_type": "embedding", "model_name": "model",
        "api_endpoint": "https://example.com", "api_key": None,
    })
    monkeypatch.setattr(milvus_routes, "_get_owned_collection", lambda *_args: {"user_id": "user-1"})
    monkeypatch.setattr(milvus_routes, "_get_milvus_client", lambda: SimpleNamespace(
        collection_exists=lambda _name: True,
        search_similar=lambda **_kwargs: [[{"chunk_id": "hit"}]],
        close=lambda: closed.append("milvus"),
    ))
    response = asyncio.run(_assert_responsive(
        _app(), entered, release, "POST", "/milvus/collections/owned/search",
        {"query_text": "hello", "search_mode": "dense", "embedding_config_id": "embedding-1"},
    ))
    assert response.status_code == 200
    assert response.json()["results"] == [{"chunk_id": "hit"}]
    assert closed == ["embedding", "milvus"]
