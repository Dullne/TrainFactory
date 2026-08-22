"""Milvus failures must not be reported as successful embedding pre-indexing."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from train_factory.generation.steps.embedding_filter_step import EmbeddingFilterStep


class _EmbeddingClient:
    async def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)


class _FailingMilvusClient:
    def get_vectors_by_ids(self, _collection_name, _chunk_ids):
        return {}

    def collection_exists(self, _collection_name):
        return False

    def create_collection(self, _collection_name, _dimension):
        return None

    def insert_vectors(self, **_kwargs):
        raise ConnectionError("Milvus is unavailable")


def _step():
    return EmbeddingFilterStep(
        embedding_client=_EmbeddingClient(),
        milvus_client=_FailingMilvusClient(),
        threshold=1.0,
        collection_name="sync_collection",
    )


def test_pre_index_propagates_milvus_insert_failure_and_clears_local_cache():
    step = _step()
    documents = [SimpleNamespace(doc_id="doc-1", content="first document")]

    with pytest.raises(ConnectionError, match="Milvus is unavailable"):
        asyncio.run(step.pre_index_all_chunks(documents))

    assert "doc-1" not in step._chunk_emb_cache
    assert step._milvus_inserted == 0


def test_streaming_filter_propagates_milvus_insert_failure():
    step = _step()
    records = [
        {
            "query": "question",
            "chunk_id": "doc-1",
            "chunk_content": "first document",
        }
    ]

    with pytest.raises(ConnectionError, match="Milvus is unavailable"):
        asyncio.run(step.execute_batch(records))

    assert "doc-1" not in step._chunk_emb_cache
    assert step._milvus_inserted == 0
