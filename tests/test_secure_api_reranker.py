import pytest

from train_factory.evaluation import secure_api_reranker as secure_module


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_secure_reranker_uses_user_scoped_pinned_transport(monkeypatch):
    calls = []

    def request(method, url, user_id, **kwargs):
        calls.append((method, url, user_id, kwargs))
        return _Response(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.25},
                    {"index": 1, "relevance_score": 0.75},
                ]
            }
        )

    monkeypatch.setattr(secure_module, "request_user_outbound", request)
    reranker = secure_module.SecureAPIReranker(
        "https://rerank.example.test",
        model="reranker",
        user_id="user-1",
    )

    ranking, scores = reranker.rerank("query", ["first", "second"])

    assert ranking == [1, 0]
    assert scores == {0: 0.25, 1: 0.75}
    assert calls == [
        (
            "POST",
            "https://rerank.example.test/v1/rerank",
            "user-1",
            {
                "headers": {"Content-Type": "application/json"},
                "json": {
                    "model": "reranker",
                    "query": "query",
                    "documents": ["first", "second"],
                },
                "timeout": 30,
            },
        )
    ]


def test_secure_reranker_batch_preserves_input_order(monkeypatch):
    def request(method, url, user_id, **kwargs):  # noqa: ARG001
        query = kwargs["json"]["query"]
        score = 0.9 if query == "high" else 0.1
        return _Response({"results": [{"index": 0, "score": score}]})

    monkeypatch.setattr(secure_module, "request_user_outbound", request)
    reranker = secure_module.SecureAPIReranker(
        "http://8.8.8.8:8000",
        user_id="user-1",
        max_concurrency=2,
    )

    results = reranker.rerank_batch(
        [("low", ["a"]), ("high", ["b"])],
        show_progress=False,
    )

    assert results == [([0], {0: 0.1}), ([0], {0: 0.9})]


@pytest.mark.parametrize(
    ("framework", "endpoint", "expected"),
    [
        ("vllm", "https://rerank.test", "https://rerank.test/rerank"),
        ("vllm", "https://rerank.test/v1", "https://rerank.test/v1/rerank"),
        ("vllm", "https://rerank.test/v2/", "https://rerank.test/v2/rerank"),
        ("vllm", "https://rerank.test/api", "https://rerank.test/api/rerank"),
        (
            "vllm",
            "https://rerank.test/v1/rerank/",
            "https://rerank.test/v1/rerank",
        ),
        ("", "https://rerank.test", "https://rerank.test/v1/rerank"),
        ("", "https://rerank.test/v1", "https://rerank.test/v1/rerank"),
        ("", "https://rerank.test/v2/", "https://rerank.test/v2/rerank"),
        ("", "https://rerank.test/api", "https://rerank.test/api/v1/rerank"),
        (
            "",
            "https://rerank.test/v2/rerank/",
            "https://rerank.test/v2/rerank",
        ),
    ],
)
def test_rerank_endpoint_normalization_preserves_explicit_versions(
    framework,
    endpoint,
    expected,
):
    reranker = secure_module.SecureAPIReranker(
        endpoint,
        inference_framework=framework,
    )

    assert reranker.endpoint == expected
