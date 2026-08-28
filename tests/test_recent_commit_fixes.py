import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from train_factory.api.routes import generation_routes, milvus_routes, model_config_routes
from train_factory.deployment.docker_deployer import docker_deployer
from train_factory.deployment.deployment_service import (
    ReplicaOperationClaim,
    deployment_service,
)
from train_factory.generation import pipeline as pipeline_module
from train_factory.generation.pipeline import DatasetGenerationPipeline, PipelineConfig
from train_factory.generation.steps.base import Document, GeneratedSample
from train_factory.storage.backends import local_backend
from train_factory.storage.backends.local_backend import LocalStorageBackend
from train_factory.storage.services.model_registry_service import (
    _remove_deployment_container,
)


def _config_payload(**overrides):
    payload = {
        "config_id": "cfg-1",
        "config_name": "test-config",
        "model_type": "llm",
        "provider": "openai",
        "api_endpoint": "https://example.com/v1",
        "api_key": "secret-key",
        "model_name": "model-1",
        "status": "active",
        "is_default": False,
        "user_id": "owner-1",
    }
    payload.update(overrides)
    return payload


def test_model_config_create_uses_authenticated_user(monkeypatch):
    captured = {}

    async def fake_create_config_async(**kwargs):
        captured.update(kwargs)
        return _config_payload(user_id=kwargs["user_id"])

    monkeypatch.setattr(
        model_config_routes.model_config_service,
        "create_config_async",
        fake_create_config_async,
    )
    request = model_config_routes.CreateConfigRequest(
        config_name="test-config",
        model_type="llm",
        provider="openai",
        api_endpoint="https://example.com/v1",
        model_name="model-1",
        user_id="spoofed-user",
    )

    response = asyncio.run(model_config_routes.create_config(request, {"user_id": "owner-1"}))

    assert captured["user_id"] == "owner-1"
    assert response.user_id == "owner-1"


def test_ownerless_model_config_is_read_only_for_authenticated_users():
    config = _config_payload(user_id=None)

    assert model_config_routes._verify_model_config_access(config, {"user_id": "user-1"}, allow_public=True) is config
    with pytest.raises(HTTPException) as exc_info:
        model_config_routes._verify_model_config_access(config, {"user_id": "user-1"})

    assert exc_info.value.status_code == 403


def test_generation_rejects_foreign_collection_and_ownerless_autofill(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        generation_routes._verify_owned_resource({"user_id": "user-2"}, {"user_id": "user-1"}, "Milvus collection")
    assert exc_info.value.status_code == 403

    monkeypatch.setattr(
        generation_routes.model_config_service,
        "get_config",
        lambda config_id: _config_payload(config_id=config_id, model_type="embedding", user_id=None),
    )
    with pytest.raises(HTTPException) as exc_info:
        generation_routes._resolve_model_config(
            "embedding-1",
            {"user_id": "user-1"},
            {"embedding"},
            allow_public=False,
        )
    assert exc_info.value.status_code == 403


def test_generation_task_access_rejects_ownerless_legacy_task(monkeypatch):
    monkeypatch.setattr(
        generation_routes.generation_task_service,
        "get_task",
        lambda task_id: {"task_id": task_id, "user_id": None},
    )

    with pytest.raises(HTTPException) as exc_info:
        generation_routes._verify_task_access(
            "ownerless-task",
            {"user_id": "user-1"},
        )

    assert exc_info.value.status_code == 403

def test_milvus_list_only_returns_authenticated_users_collections(monkeypatch):
    class FakeClient:
        closed = False

        def list_collections(self):
            return ["owned", "foreign", "unregistered"]

        def get_collection_info(self, name):
            return {"name": name, "num_entities": 1}

        def close(self):
            self.closed = True

    client = FakeClient()
    seen_user_ids = []
    monkeypatch.setattr(milvus_routes, "_get_milvus_client", lambda: client)

    def fake_registry_map(user_id=None):
        seen_user_ids.append(user_id)
        return {"owned": {"collection_name": "owned", "user_id": "user-1"}}

    monkeypatch.setattr(milvus_routes, "_build_registry_map", fake_registry_map)

    response = asyncio.run(milvus_routes.list_collections({"user_id": "user-1"}))

    assert seen_user_ids == ["user-1"]
    assert [item["name"] for item in response["collections"]] == ["owned"]
    assert client.closed is True


def test_qa_checkpoint_key_includes_answer_and_normalizes_output_shape():
    first = {"query": "same", "answer": "answer-1", "chunk_id": "chunk-1"}
    second = {"query": "same", "answer": "answer-2", "chunk_id": "chunk-1"}
    universal = {
        "query": "same",
        "answer": "answer-1",
        "metadata": {"source_doc_id": "chunk-1"},
    }

    assert DatasetGenerationPipeline._qa_checkpoint_key(first) != (DatasetGenerationPipeline._qa_checkpoint_key(second))
    assert DatasetGenerationPipeline._qa_checkpoint_key(first) == (
        DatasetGenerationPipeline._qa_checkpoint_key(universal)
    )


def test_container_cleanup_failure_preserves_deployment_retry_handle(monkeypatch):
    deployment = SimpleNamespace(
        deploy_mode="container",
        container_name="model-container",
        deployment_id="deployment-1",
        config={},
        inference_framework="xinference",
    )
    monkeypatch.setattr(
        deployment_service,
        "_is_unmanaged_binding",
        lambda _deployment: False,
    )
    monkeypatch.setattr(
        deployment_service,
        "_require_managed_container",
        lambda _deployment: None,
    )
    monkeypatch.setattr(
        deployment_service,
        "_require_replica_operation_ownership",
        lambda _claim: None,
    )
    monkeypatch.setattr(docker_deployer, "remove_container", lambda name: False)

    with pytest.raises(RuntimeError, match="deployment-1"):
        _remove_deployment_container(
            deployment,
            "model-1",
            replica_operation_claim=ReplicaOperationClaim(
                deployment_id="deployment-1",
                token="delete-token",
                generation=1,
                operation="delete",
                replica_id=None,
            ),
        )


@pytest.mark.parametrize(
    ("split_name", "expected_field"),
    [("validation", "num_eval"), ("test", "num_test")],
)
def test_arrow_directory_does_not_relabel_fallback_split(tmp_path, monkeypatch, split_name, expected_field):
    split_dir = tmp_path / split_name
    split_dir.mkdir()
    (split_dir / "dataset_info.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(local_backend, "_count_file_rows", lambda path, fmt: 7)
    monkeypatch.setattr(local_backend, "_dir_size", lambda path: 70)
    previewed = []

    def fake_preview(path, file_format, limit):
        previewed.append(path)
        return {"columns": [{"name": "text", "type": "str"}], "rows": []}

    monkeypatch.setattr(local_backend, "_load_single_file_preview", fake_preview)

    result = LocalStorageBackend().load_preview(str(tmp_path), "arrow")

    assert result["num_train"] is None
    assert result[expected_field] == 7
    assert result["num_rows"] == 7
    assert result["file_size"] == 70
    assert previewed == [split_dir]


def test_doc_to_training_without_embedding_isolates_batch_failure(tmp_path, monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    output_path = tmp_path / "partial.jsonl"
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.txt"),
            output_path=str(output_path),
            generation_mode="doc_to_training",
            pos_neg_method="llm",
        )
    )
    documents = [Document(content="source", doc_id="doc-1")]
    monkeypatch.setattr(pipeline, "_load_documents", lambda: documents)
    monkeypatch.setattr(pipeline, "_create_llm_client", lambda config=None: FakeClient())

    qa_records = [
        {
            "query": f"query-{index}",
            "answer": f"answer-{index}",
            "chunk_id": "doc-1",
            "chunk_content": "source",
        }
        for index in range(10)
    ]

    async def fake_producer(documents, qa_queue, log_prefix, **kwargs):
        for record in qa_records:
            await qa_queue.put([record])
        await qa_queue.put(None)
        return qa_records, str(tmp_path / "qa.jsonl")

    async def fake_generate(batch, *args, **kwargs):
        record = batch[0]
        if record["query"] == "query-0":
            raise RuntimeError("first batch failed")
        return [
            GeneratedSample(
                query=record["query"],
                answer=record["answer"],
                positive_chunks=[record["chunk_content"]],
                source_doc_id=record["chunk_id"],
            )
        ], []

    monkeypatch.setattr(pipeline, "_streaming_qa_producer", fake_producer)
    monkeypatch.setattr(pipeline, "_generate_pos_neg_batch", fake_generate)

    async def run_with_deadlock_guard():
        return await asyncio.wait_for(pipeline.run(), timeout=2)

    result = asyncio.run(run_with_deadlock_guard())

    assert result.success is False
    assert result.output_samples == 9
    assert result.output_path == str(output_path)
    assert output_path.read_text(encoding="utf-8").count("\n") == 9
    assert result.details["filter_stats"]["batch_failures"] == {
        "batch_count": 1,
        "record_count": 1,
        "errors": ["first batch failed"],
    }


def test_embedding_doc_to_training_reports_monotonic_progress_across_consumer_batches(
    tmp_path, monkeypatch
):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeFilterStep:
        def __init__(self, **kwargs):
            pass

        async def pre_index_all_chunks(self, documents, progress_callback):
            for processed in range(1, len(documents) + 1):
                progress_callback(processed, len(documents))
            return len(documents)

        async def execute_batch(self, batch):
            return batch

        def finalize_stats(self):
            return {}

    progress_updates = []
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.txt"),
            output_path=str(tmp_path / "output.jsonl"),
            output_format="universal",
            generation_mode="doc_to_training",
            pos_neg_method="llm",
            embedding_config={"endpoint": "http://embedding"},
            steps={"qa_gen": {"num_qa_per_doc": 2}},
        ),
        progress_callback=lambda processed, total: progress_updates.append(
            (processed, total)
        ),
    )
    documents = [
        Document(content="source-1", doc_id="doc-1"),
        Document(content="source-2", doc_id="doc-2"),
    ]
    qa_batches = [
        [
            {
                "query": f"query-{batch_index}-{record_index}",
                "answer": "answer",
                "chunk_id": f"doc-{batch_index + 1}",
                "chunk_content": "source",
            }
            for record_index in range(2)
        ]
        for batch_index in range(2)
    ]

    async def fake_producer(documents, qa_queue, log_prefix, **kwargs):
        producer_callback = kwargs.get("progress_callback") or pipeline.progress_callback
        for processed in range(1, len(documents) + 1):
            producer_callback(processed, len(documents))
        for batch in qa_batches:
            await qa_queue.put(batch)
        await qa_queue.put(None)
        return [record for batch in qa_batches for record in batch], str(
            tmp_path / "qa.jsonl"
        )

    async def fake_generate(batch, *args, **kwargs):
        consumer_callback = kwargs.get("progress_callback") or pipeline.progress_callback
        samples = []
        for processed, record in enumerate(batch, start=1):
            consumer_callback(processed, len(batch))
            samples.append(
                GeneratedSample(
                    query=record["query"],
                    answer=record["answer"],
                    positive_chunks=[record["chunk_content"]],
                    source_doc_id=record["chunk_id"],
                )
            )
        return samples, []

    monkeypatch.setattr(pipeline_module, "EmbeddingFilterStep", FakeFilterStep)
    monkeypatch.setattr(pipeline, "_load_documents", lambda: documents)
    monkeypatch.setattr(
        pipeline,
        "_init_phase2_clients",
        lambda: (FakeClient(), None, None, None),
    )
    monkeypatch.setattr(pipeline, "_create_llm_client", lambda config=None: FakeClient())
    monkeypatch.setattr(pipeline, "_streaming_qa_producer", fake_producer)
    monkeypatch.setattr(pipeline, "_generate_pos_neg_batch", fake_generate)

    result = asyncio.run(pipeline.run())

    processed_values = [processed for processed, _ in progress_updates]
    assert result.success is True
    assert processed_values == sorted(processed_values)
    assert progress_updates[-1] == (8, 8)


def test_embedding_filter_failure_advances_failed_batch_and_continues(
    tmp_path,
    monkeypatch,
):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeFilterStep:
        def __init__(self, **kwargs):
            self.calls = 0

        async def pre_index_all_chunks(self, documents, progress_callback):
            for processed in range(1, len(documents) + 1):
                progress_callback(processed, len(documents))
            return len(documents)

        async def execute_batch(self, batch):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("filter batch failed")
            return batch

        def finalize_stats(self):
            return {}

    progress_updates = []
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.txt"),
            output_path=str(tmp_path / "output.jsonl"),
            output_format="universal",
            generation_mode="doc_to_training",
            pos_neg_method="llm",
            embedding_config={"endpoint": "http://embedding"},
            steps={"qa_gen": {"num_qa_per_doc": 2}},
        ),
        progress_callback=lambda processed, total: progress_updates.append(
            (processed, total)
        ),
    )
    documents = [
        Document(content="source-1", doc_id="doc-1"),
        Document(content="source-2", doc_id="doc-2"),
    ]
    qa_batches = [
        [
            {
                "query": f"query-{batch_index}-{record_index}",
                "answer": "answer",
                "chunk_id": f"doc-{batch_index + 1}",
                "chunk_content": "source",
            }
            for record_index in range(2)
        ]
        for batch_index in range(2)
    ]

    async def fake_producer(documents, qa_queue, log_prefix, **kwargs):
        callback = kwargs.get("progress_callback") or pipeline.progress_callback
        for processed in range(1, len(documents) + 1):
            callback(processed, len(documents))
        for batch in qa_batches:
            await qa_queue.put(batch)
        await qa_queue.put(None)
        return [record for batch in qa_batches for record in batch], str(
            tmp_path / "qa.jsonl"
        )

    async def fake_generate(batch, *args, **kwargs):
        callback = kwargs.get("progress_callback") or pipeline.progress_callback
        samples = []
        for processed, record in enumerate(batch, start=1):
            callback(processed, len(batch))
            samples.append(GeneratedSample(
                query=record["query"],
                answer=record["answer"],
                positive_chunks=[record["chunk_content"]],
                source_doc_id=record["chunk_id"],
            ))
        return samples, []

    monkeypatch.setattr(pipeline_module, "EmbeddingFilterStep", FakeFilterStep)
    monkeypatch.setattr(pipeline, "_load_documents", lambda: documents)
    monkeypatch.setattr(
        pipeline,
        "_init_phase2_clients",
        lambda: (FakeClient(), None, None, None),
    )
    monkeypatch.setattr(pipeline, "_create_llm_client", lambda config=None: FakeClient())
    monkeypatch.setattr(pipeline, "_streaming_qa_producer", fake_producer)
    monkeypatch.setattr(pipeline, "_generate_pos_neg_batch", fake_generate)

    result = asyncio.run(pipeline.run())

    assert result.success is False
    assert result.output_samples == 2
    assert progress_updates[4] == (6, 8)
    assert [processed for processed, _ in progress_updates] == sorted(
        processed for processed, _ in progress_updates
    )
    assert progress_updates[-1] == (8, 8)


def test_doc_to_training_without_embedding_reports_monotonic_streaming_progress(
    tmp_path,
    monkeypatch,
):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    progress_updates = []
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.txt"),
            output_path=str(tmp_path / "output.jsonl"),
            output_format="universal",
            generation_mode="doc_to_training",
            pos_neg_method="llm",
            steps={"qa_gen": {"num_qa_per_doc": 2}},
        ),
        progress_callback=lambda processed, total: progress_updates.append(
            (processed, total)
        ),
    )
    documents = [
        Document(content="source-1", doc_id="doc-1"),
        Document(content="source-2", doc_id="doc-2"),
    ]
    qa_batches = [
        [
            {
                "query": f"query-{batch_index}-{record_index}",
                "answer": "answer",
                "chunk_id": f"doc-{batch_index + 1}",
                "chunk_content": "source",
            }
            for record_index in range(2)
        ]
        for batch_index in range(2)
    ]

    async def fake_producer(documents, qa_queue, log_prefix, **kwargs):
        callback = kwargs.get("progress_callback") or pipeline.progress_callback
        for processed in range(1, len(documents) + 1):
            callback(processed, len(documents))
        for batch in qa_batches:
            await qa_queue.put(batch)
        await qa_queue.put(None)
        return [record for batch in qa_batches for record in batch], str(
            tmp_path / "qa.jsonl"
        )

    async def fake_generate(batch, *args, **kwargs):
        callback = kwargs.get("progress_callback") or pipeline.progress_callback
        samples = []
        for processed, record in enumerate(batch, start=1):
            callback(processed, len(batch))
            samples.append(GeneratedSample(
                query=record["query"],
                answer=record["answer"],
                positive_chunks=[record["chunk_content"]],
                source_doc_id=record["chunk_id"],
            ))
        return samples, []

    monkeypatch.setattr(pipeline, "_load_documents", lambda: documents)
    monkeypatch.setattr(pipeline, "_create_llm_client", lambda config=None: FakeClient())
    monkeypatch.setattr(pipeline, "_streaming_qa_producer", fake_producer)
    monkeypatch.setattr(pipeline, "_generate_pos_neg_batch", fake_generate)

    result = asyncio.run(pipeline.run())

    processed_values = [processed for processed, _ in progress_updates]
    assert result.success is True
    assert processed_values == sorted(processed_values)
    assert progress_updates[-1] == (6, 6)
