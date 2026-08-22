import pytest
from fastapi import HTTPException

from train_factory.api.routes import model_config_routes


@pytest.mark.parametrize(
    ("model_type", "path"),
    [
        ("embedding", "/v1/embeddings"),
        ("rerank", "/v1/rerank"),
        ("reranker", "/v1/rerank"),
        ("llm", "/v1/chat/completions"),
        ("llm", "/v1/completions"),
    ],
)
def test_model_config_proxy_allows_only_inference_paths(model_type, path):
    assert model_config_routes._validate_test_proxy_path(path, model_type) == path


@pytest.mark.parametrize(
    "path",
    [
        "/v1/models",
        "/v1/models/",
        "/v1/%6dodels",
        "/v1%2fmodels",
        "//v1/models",
        "/v1/models?model=owned",
        "/v1/models#fragment",
        "/v1\\models",
        "http://xinference:9997/v1/models",
        "/v1/chat/completions/../models",
    ],
)
def test_model_config_proxy_rejects_management_and_ambiguous_paths(path):
    with pytest.raises(HTTPException) as exc_info:
        model_config_routes._validate_test_proxy_path(path, "llm")

    assert exc_info.value.status_code == 400


def test_model_config_proxy_rejects_inference_path_for_wrong_model_type():
    with pytest.raises(HTTPException) as exc_info:
        model_config_routes._validate_test_proxy_path(
            "/v1/chat/completions",
            "embedding",
        )

    assert exc_info.value.status_code == 400
