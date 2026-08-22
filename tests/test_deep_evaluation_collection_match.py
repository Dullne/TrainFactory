from train_factory.api.routes.deep_evaluation_routes import (
    _detect_collection_model_mismatch,
)


def test_detect_collection_model_mismatch_by_config_id():
    collection = {
        "collection_name": "test_collection",
        "embedding_config_id": "cfg-a",
        "embedding_model": "model-a",
    }
    retrieval_embedding = {
        "config_id": "cfg-b",
        "model_name": "model-a",
    }

    reason = _detect_collection_model_mismatch(collection, retrieval_embedding)
    assert reason is not None
    assert "cfg-a" in reason
    assert "cfg-b" in reason


def test_detect_collection_model_mismatch_by_model_name():
    collection = {
        "collection_name": "test_collection",
        "embedding_model": "BAAI/bge-m3",
    }
    retrieval_embedding = {
        "model_name": "BAAI/bge-base-zh-v1.5",
    }

    reason = _detect_collection_model_mismatch(collection, retrieval_embedding)
    assert reason is not None
    assert "bge-m3" in reason
    assert "bge-base-zh-v1.5" in reason


def test_detect_collection_model_match_returns_none():
    collection = {
        "collection_name": "test_collection",
        "embedding_config_id": "cfg-a",
        "embedding_model": "BAAI/bge-m3",
    }
    retrieval_embedding = {
        "config_id": "cfg-a",
        "model_name": "BAAI/BGE-M3",  # case-insensitive
    }

    reason = _detect_collection_model_mismatch(collection, retrieval_embedding)
    assert reason is None
