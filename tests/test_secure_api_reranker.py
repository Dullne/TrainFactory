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


def test_secure_vllm_reranker_sends_instruction_with_raw_cohere_payload(monkeypatch):
    calls = []

    def request(method, url, user_id, **kwargs):
        calls.append((method, url, user_id, kwargs))
        return _Response({"results": [{"index": 0, "score": 0.5}]})

    monkeypatch.setattr(secure_module, "request_user_outbound", request)
    reranker = secure_module.SecureAPIReranker(
        "https://rerank.example.test/v1",
        model="Qwen3-Reranker-4B",
        inference_framework="vllm",
        instruction="do not embed this in the query",
        user_id="user-1",
    )

    reranker.rerank("raw query", ["raw document"])

    assert calls[0][3]["json"] == {
        "model": "Qwen3-Reranker-4B",
        "query": "raw query",
        "documents": ["raw document"],
        "instruction": "do not embed this in the query",
    }


def test_secure_sglang_reranker_uses_instruct_field(monkeypatch):
    calls = []

    def request(method, url, user_id, **kwargs):
        calls.append((method, url, user_id, kwargs))
        return _Response({"results": [{"index": 0, "score": 0.5}]})

    monkeypatch.setattr(secure_module, "request_user_outbound", request)
    reranker = secure_module.SecureAPIReranker(
        "https://rerank.example.test/v1",
        model="Qwen3-Reranker-4B",
        inference_framework="sglang",
        instruction="use the fixed SGLang protocol",
        user_id="user-1",
    )

    reranker.rerank("raw query", ["raw document"])

    assert calls[0][3]["json"] == {
        "model": "Qwen3-Reranker-4B",
        "query": "raw query",
        "documents": ["raw document"],
        "instruct": "use the fixed SGLang protocol",
    }


@pytest.mark.parametrize(
    ("framework", "instruction_key"),
    [("vllm", "instruction"), ("sglang", "instruct")],
)
def test_secure_reranker_batch_preserves_raw_framework_payload(
    monkeypatch,
    framework,
    instruction_key,
):
    calls = []

    def request(method, url, user_id, **kwargs):
        calls.append((method, url, user_id, kwargs))
        return _Response({"results": [{"index": 0, "score": 0.5}]})

    monkeypatch.setattr(secure_module, "request_user_outbound", request)
    reranker = secure_module.SecureAPIReranker(
        "https://rerank.example.test/v1",
        model="Qwen3-Reranker-4B",
        inference_framework=framework,
        instruction="framework-owned instruction",
        user_id="user-1",
    )

    reranker.rerank_batch(
        [("raw batch query", ["raw batch document"])],
        show_progress=False,
    )

    assert calls[0][3]["json"] == {
        "model": "Qwen3-Reranker-4B",
        "query": "raw batch query",
        "documents": ["raw batch document"],
        instruction_key: "framework-owned instruction",
    }


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
@pytest.mark.parametrize("instruction", ["", "   "])
def test_secure_reranker_omits_blank_instruction(
    monkeypatch,
    framework,
    instruction,
):
    calls = []

    def request(_method, _url, _user_id, **kwargs):
        calls.append(kwargs)
        return _Response({"results": []})

    monkeypatch.setattr(secure_module, "request_user_outbound", request)
    reranker = secure_module.SecureAPIReranker(
        "https://rerank.example.test/v1",
        inference_framework=framework,
        instruction=instruction,
        user_id="user-1",
    )

    reranker.rerank("raw query", ["raw document"])

    assert "instruction" not in calls[0]["json"]
    assert "instruct" not in calls[0]["json"]


def test_secure_reranker_does_not_log_instruction_on_connection_failure(
    monkeypatch,
    caplog,
):
    secret_instruction = "secure-instruction-secret-do-not-log"

    def request(*_args, **_kwargs):
        raise RuntimeError(f"transport echoed {secret_instruction}")

    monkeypatch.setattr(secure_module, "request_user_outbound", request)
    reranker = secure_module.SecureAPIReranker(
        "https://rerank.example.test/v1",
        inference_framework="vllm",
        instruction=secret_instruction,
        user_id="user-1",
    )

    with caplog.at_level("ERROR", logger=secure_module.__name__):
        assert reranker.test_connection() is False

    assert secret_instruction not in caplog.text
    assert "RuntimeError" in caplog.text


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
