import asyncio

import numpy as np
import pytest
from fastapi import HTTPException

from train_factory.api.routes import model_config_routes


CONFIG = {
    "api_endpoint": "https://embedding.example.com/v1",
    "api_key": None,
    "model_name": "embedding-model",
    "model_type": "embedding",
    "user_id": "user-1",
}
CURRENT_USER = {"user_id": "user-1", "username": "alice", "is_admin": False}


@pytest.mark.parametrize(
    "body",
    [
        {"sentence1": ["a"] * 101, "sentence2": ["b"] * 100},
        {"sentence1": ["a" * 8193], "sentence2": ["b"]},
        {"sentence1": ["a"], "sentence2": ["b"], "batch_size": 0},
        {"sentence1": ["a"], "sentence2": ["b"], "batch_size": 257},
    ],
)
def test_embedding_similarity_rejects_oversized_work_before_network(
    monkeypatch,
    body,
):
    class UnexpectedClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Embedding client created before request bounds validation")

    monkeypatch.setattr(model_config_routes, "EmbeddingClient", UnexpectedClient)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(model_config_routes._compute_embedding_similarity(CONFIG, body))

    assert exc_info.value.status_code == 400


def test_embedding_similarity_uses_server_timeout_and_retry_policy(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, config):
            captured["config"] = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def embed(self, texts):
            return np.ones((len(texts), 1))

    monkeypatch.setattr(model_config_routes, "EmbeddingClient", FakeClient)

    result = asyncio.run(
        model_config_routes._compute_embedding_similarity(
            CONFIG,
            {
                "sentence1": ["a"],
                "sentence2": ["b"],
                "timeout": 999999,
                "max_retries": 999999,
                "batch_size": 64,
            },
        )
    )

    assert result["similarity"] == [[1.0]]
    assert captured["config"].batch_size == 64
    assert captured["config"].timeout == model_config_routes.TEST_PROXY_TIMEOUT_SECONDS
    assert captured["config"].max_retries == model_config_routes.TEST_PROXY_MAX_RETRIES


@pytest.mark.parametrize(
    "body",
    [
        {"queries": ["q"] * 21, "collection_name": "owned"},
        {"queries": ["q"], "collection_name": "owned", "top_k": 0},
        {"queries": ["q"], "collection_name": "owned", "top_k": 101},
        {"queries": ["q"], "collection_name": "owned", "batch_size": 257},
    ],
)
def test_embedding_recall_rejects_oversized_work_before_milvus(monkeypatch, body):
    monkeypatch.setattr(
        model_config_routes,
        "_get_milvus_client_from_env",
        lambda: (_ for _ in ()).throw(
            AssertionError("Milvus client created before request bounds validation")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            model_config_routes._compute_embedding_recall(CONFIG, body, CURRENT_USER)
        )

    assert exc_info.value.status_code == 400


def test_embedding_recall_rejects_foreign_collection_before_milvus(monkeypatch):
    monkeypatch.setattr(
        model_config_routes.milvus_collection_service,
        "get_by_name",
        lambda name: {"collection_name": name, "user_id": "user-2"},
    )
    monkeypatch.setattr(
        model_config_routes,
        "_get_milvus_client_from_env",
        lambda: (_ for _ in ()).throw(
            AssertionError("Milvus client created before ownership validation")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            model_config_routes._compute_embedding_recall(
                CONFIG,
                {"queries": ["q"], "collection_name": "foreign"},
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403


def test_embedding_recall_requires_registered_collection_when_authenticated(
    monkeypatch,
):
    monkeypatch.setattr(
        model_config_routes.milvus_collection_service,
        "get_by_name",
        lambda name: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            model_config_routes._compute_embedding_recall(
                CONFIG,
                {"queries": ["q"], "collection_name": "missing"},
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 404
