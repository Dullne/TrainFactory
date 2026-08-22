import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import BackgroundTasks, HTTPException

from train_factory.api.routes import generation_routes
from train_factory.auth.resource_provenance import ResourceProvenanceError
from train_factory.config.settings import get_settings
from train_factory.core.ssrf import SSRFError
from train_factory.generation import pipeline as pipeline_module
from train_factory.generation.pipeline import DatasetGenerationPipeline, PipelineConfig
from train_factory.storage.entities.generation_task_entity import GenerationStatus


CURRENT_USER = {"user_id": "user-1", "username": "alice"}


@pytest.fixture(autouse=True)
def _isolate_route_admission(monkeypatch):
    lease = SimpleNamespace(release=lambda: None)

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        result = operation(*args, **kwargs)
        return result, lease if result is not False and result is not None else None

    monkeypatch.setattr(
        generation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )


class _RecordingTaskService:
    def __init__(self):
        self.created = []
        self.outputs = []
        self.status_updates = []
        self.task = None
        self.run_token = "generation-security-run-token-1"
        self._run_number = 1

    def create_task(self, **kwargs):
        self.created.append(kwargs)
        self.task = {
            "task_id": "generation-security-task",
            "task_name": kwargs["task_name"],
            "status": GenerationStatus.PENDING,
            "generation_mode": kwargs["generation_mode"],
            "pos_neg_method": kwargs["pos_neg_method"],
            "progress": 0.0,
            "total_docs": 0,
            "processed_docs": 0,
            "output_sample_count": 0,
        }
        return dict(self.task)

    def get_task(self, _task_id):
        return dict(self.task) if self.task else None

    def get_task_raw(self, task_id):
        task = self.get_task(task_id)
        if task is None:
            return None
        return {**task, "run_token": self.run_token}

    def _assert_attempt(self, expected_run_token, expected_status=None):
        assert expected_run_token == self.run_token
        if expected_status is not None:
            assert self.task is not None
            assert self.task["status"] == expected_status

    def set_output(self, *args, **kwargs):
        self._assert_attempt(
            kwargs.get("expected_run_token"),
            kwargs.get("expected_status"),
        )
        self.outputs.append((args, kwargs))
        if self.task is not None:
            self.task["output_path"] = args[1]
            self.task["output_sample_count"] = args[2]
        return True

    def update_status(
        self,
        task_id,
        status,
        error_message=None,
        *,
        expected_run_token=None,
        **_kwargs,
    ):
        self._assert_attempt(expected_run_token)
        self.status_updates.append((task_id, status, error_message))
        if status == GenerationStatus.PENDING:
            self._run_number += 1
            self.run_token = f"generation-security-run-token-{self._run_number}"
        if self.task is not None:
            self.task["status"] = status
        return True

    def claim_running(self, task_id, *, expected_run_token=None):
        if expected_run_token != self.run_token:
            return None
        if self.task is None or self.task.get("status") != GenerationStatus.PENDING:
            return None
        self.status_updates.append((task_id, GenerationStatus.RUNNING, None))
        self.task["status"] = GenerationStatus.RUNNING
        return self.run_token

    def update_progress(self, *_args, **kwargs):
        self._assert_attempt(
            kwargs.get("expected_run_token"),
            kwargs.get("expected_status"),
        )
        return True

    def set_qa_output(self, *_args, **kwargs):
        self._assert_attempt(
            kwargs.get("expected_run_token"),
            kwargs.get("expected_status"),
        )
        return True

    def set_filter_results(self, *_args, **kwargs):
        self._assert_attempt(
            kwargs.get("expected_run_token"),
            kwargs.get("expected_status"),
        )
        return True

    def set_deep_eval_output(self, *_args, **kwargs):
        self._assert_attempt(
            kwargs.get("expected_run_token"),
            kwargs.get("expected_status"),
        )
        return True


def _allowed_input_file(name: str = "input.txt") -> Path:
    path = Path(get_settings().datasets_dir) / "generation-security" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("safe input", encoding="utf-8")
    return path


def _create_request(input_path: Path, endpoint: str):
    return generation_routes.CreateTaskRequest(
        task_name="security-check",
        input_path=str(input_path),
        input_format="txt",
        generation_mode="qa_extraction",
        llm_config={"endpoint": endpoint, "model": "test-model"},
    )


def _bind_owned_dataset(monkeypatch, request, input_path: Path) -> None:
    request.dataset_id = "dataset-1"
    dataset = {
        "dataset_id": "dataset-1",
        "dataset_name": "owned-input",
        "storage_path": str(input_path),
        "storage_backend": "local",
        "source_type": "uploaded",
        "status": "ready",
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "get_dataset",
        lambda _dataset_id: dataset,
    )
    monkeypatch.setattr(
        generation_routes,
        "resolve_managed_local_dataset",
        lambda _dataset_id, *, user_id: dataset,
    )


def test_generation_rejects_dataset_with_deletion_in_progress(monkeypatch):
    dataset = {
        "dataset_id": "dataset-deleting",
        "dataset_name": "deleting-input",
        "storage_path": str(_allowed_input_file("deleting.jsonl")),
        "storage_backend": "local",
        "source_type": "generated",
        "status": "deleting",
        "user_id": "user-1",
    }
    monkeypatch.setattr(
        generation_routes.dataset_service,
        "get_dataset",
        lambda _dataset_id: dataset,
    )

    with pytest.raises(HTTPException) as exc_info:
        generation_routes._resolve_owned_generation_dataset(
            "dataset-deleting",
            {"user_id": "anonymous", "username": "anonymous"},
        )

    assert exc_info.value.status_code == 409


def test_create_generation_rejects_private_direct_llm_before_persistence(monkeypatch):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)

    with pytest.raises(SSRFError) as exc_info:
        asyncio.run(
            generation_routes.create_task(
                _create_request(_allowed_input_file(), "http://127.0.0.1:8000/v1"),
                BackgroundTasks(),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert service.created == []


def test_create_generation_normalizes_direct_llm_before_persistence(monkeypatch):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    endpoint_checks = []

    def validate_for_user(url, user_id):
        endpoint_checks.append((url, user_id))
        return "http://93.184.216.34:8080/v1"

    monkeypatch.setattr(
        generation_routes,
        "validate_user_outbound_url",
        validate_for_user,
        raising=False,
    )

    input_path = _allowed_input_file("normalized.txt")
    request = _create_request(input_path, " 93.184.216.34:8080/v1/ ")
    _bind_owned_dataset(monkeypatch, request, input_path)

    asyncio.run(
        generation_routes.create_task(
            request,
            BackgroundTasks(),
            CURRENT_USER,
        )
    )

    assert service.created[0]["llm_config"]["endpoint"] == (
        "http://93.184.216.34:8080/v1"
    )
    assert endpoint_checks == [(" 93.184.216.34:8080/v1/ ", "user-1")]
    assert Path(service.outputs[0][0][1]).name == (
        "generated_generation-security-task.jsonl"
    )


def test_pipeline_intermediate_outputs_use_complete_task_id(tmp_path):
    task_id = "abcdef12-generation-task"
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path=str(tmp_path / "input.jsonl"),
            output_path=str(tmp_path / f"generated_{task_id}.jsonl"),
            task_id=task_id,
        )
    )

    assert pipeline._get_qa_output_path().name == f"qa_extracted_{task_id}.jsonl"
    assert pipeline._get_qa_filtered_path().name == f"qa_filtered_{task_id}.jsonl"
    assert pipeline._get_deep_eval_path().name == f"deep_eval_{task_id}.jsonl"


@pytest.mark.parametrize("config_field", ["llm_config", "eval_llm_config"])
def test_create_generation_rejects_private_endpoint_in_every_llm_config(
    monkeypatch,
    config_field,
):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    request_data = {
        "task_name": "all-llm-endpoints",
        "input_path": str(_allowed_input_file(f"{config_field}.txt")),
        "input_format": "txt",
        "generation_mode": "qa_extraction",
        "llm_config": {
            "endpoint": "https://93.184.216.34/v1",
            "model": "test-model",
        },
    }
    if config_field == "llm_config":
        request_data["llm_config"] = {
            "endpoints": [
                {
                    "url": "https://93.184.216.34/v1",
                    "model": "public-model",
                },
                {
                    "url": "http://10.0.0.5:8000/v1",
                    "model": "private-model",
                },
            ]
        }
    else:
        request_data["eval_llm_config"] = {
            "endpoint": "http://169.254.169.254/latest/meta-data",
            "model": "metadata-model",
        }

    with pytest.raises(SSRFError):
        asyncio.run(
            generation_routes.create_task(
                generation_routes.CreateTaskRequest(**request_data),
                BackgroundTasks(),
                CURRENT_USER,
            )
        )

    assert service.created == []


@pytest.mark.parametrize(
    ("config_field", "private_endpoint"),
    [
        ("embedding_config", "http://127.0.0.1:9997/v1"),
        ("rerank_config", "http://169.254.169.254/latest/meta-data"),
    ],
)
def test_create_generation_rejects_private_embedding_and_rerank_before_persistence(
    monkeypatch,
    config_field,
    private_endpoint,
):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    request_data = {
        "task_name": "all-model-endpoints",
        "input_path": str(_allowed_input_file(f"{config_field}.txt")),
        "input_format": "txt",
        "generation_mode": "qa_extraction",
        "llm_config": {
            "endpoint": "https://93.184.216.34/v1",
            "model": "test-model",
        },
        config_field: {
            "endpoint": private_endpoint,
            "model": "private-model",
        },
    }

    with pytest.raises(SSRFError):
        asyncio.run(
            generation_routes.create_task(
                generation_routes.CreateTaskRequest(**request_data),
                BackgroundTasks(),
                CURRENT_USER,
            )
        )

    assert service.created == []


def test_create_generation_persists_owned_model_config_id_for_worker_recheck(
    monkeypatch,
):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes.model_config_service,
        "get_config",
        lambda config_id: {
            "config_id": config_id,
            "model_type": "llm",
            "api_endpoint": "https://93.184.216.34/v1",
            "model_name": "owned-model",
            "api_key": "owned-secret",
            "user_id": "user-1",
        },
    )

    input_path = _allowed_input_file("owned-config.txt")
    request = generation_routes.CreateTaskRequest(
        task_name="owned-config-id",
        input_path=str(input_path),
        input_format="txt",
        generation_mode="qa_extraction",
        llm_config={"config_id": "owned-llm"},
    )
    _bind_owned_dataset(monkeypatch, request, input_path)

    asyncio.run(
        generation_routes.create_task(
            request,
            BackgroundTasks(),
            CURRENT_USER,
        )
    )

    assert service.created[0]["llm_config"]["config_id"] == "owned-llm"


def test_generation_config_id_cannot_bypass_owner_check_with_direct_endpoints(
    monkeypatch,
):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes.model_config_service,
        "get_config",
        lambda config_id: {
            "config_id": config_id,
            "model_type": "llm",
            "api_endpoint": "https://93.184.216.34/v1",
            "model_name": "foreign-model",
            "api_key": "foreign-secret",
            "user_id": "user-2",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.create_task(
                generation_routes.CreateTaskRequest(
                    task_name="mixed-config-source",
                    input_path=str(_allowed_input_file("mixed-config.txt")),
                    input_format="txt",
                    generation_mode="qa_extraction",
                    llm_config={
                        "config_id": "foreign-llm",
                        "endpoints": [
                            {
                                "url": "https://93.184.216.34/v1",
                                "model": "direct-model",
                            }
                        ],
                    },
                ),
                BackgroundTasks(),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert service.created == []


def test_create_generation_rejects_input_outside_allowed_storage(monkeypatch, tmp_path):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    outside = tmp_path / "outside.txt"
    outside.write_text("not project data", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.create_task(
                _create_request(outside, "http://93.184.216.34:8080/v1"),
                BackgroundTasks(),
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 400
    assert service.created == []


def test_authenticated_generation_rejects_direct_managed_root_path(monkeypatch):
    service = _RecordingTaskService()
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.create_task(
                _create_request(
                    _allowed_input_file("cross-tenant-path.txt"),
                    "http://93.184.216.34:8080/v1",
                ),
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert service.created == []
    assert background_tasks.tasks == []


def test_generation_input_empty_allowlist_does_not_fall_back_to_defaults():
    with pytest.raises(HTTPException) as exc_info:
        generation_routes._resolve_generation_input_path(
            str(_allowed_input_file("empty-allowlist.txt")),
            allowed_dirs=[],
        )

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "non_data_path",
    ["/app/models/private-config.json", "/app/output/training-secret.json"],
)
def test_generation_input_default_policy_excludes_non_data_storage_roots(
    non_data_path,
):
    with pytest.raises(HTTPException) as exc_info:
        generation_routes._resolve_generation_input_path(non_data_path)

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("loader_name", ["_load_documents", "_load_qa_input_records"])
def test_pipeline_revalidates_symlink_target_immediately_before_read(
    tmp_path,
    loader_name,
):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside = tmp_path / "secret.jsonl"
    outside.write_text(
        '{"content":"outside","query":"q","chunk_content":"outside"}\n',
        encoding="utf-8",
    )
    symlink = allowed_dir / "input.jsonl"
    try:
        symlink.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    config = PipelineConfig(input_path=str(symlink), input_format="jsonl")
    config.allowed_input_dirs = [str(allowed_dir)]
    pipeline = DatasetGenerationPipeline(config)

    with pytest.raises(HTTPException) as exc_info:
        getattr(pipeline, loader_name)()

    assert exc_info.value.status_code == 400


def test_pipeline_validates_llm_endpoint_before_creating_pinned_client(monkeypatch):
    validated = []

    def fake_validate(url, user_id):
        validated.append((url, user_id))
        return url.rstrip("/")

    async def fake_pool_chat(self, *args, **kwargs):
        return "ok"

    monkeypatch.setattr(
        pipeline_module,
        "validate_user_outbound_url",
        fake_validate,
        raising=False,
    )
    monkeypatch.setattr(pipeline_module.LLMClientPool, "chat", fake_pool_chat)
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path="unused",
            llm_config={
                "endpoint": "https://93.184.216.34/v1/",
                "model": "test-model",
            },
        )
    )
    pipeline.config.user_id = "user-1"

    client = pipeline._create_llm_client()
    assert len(validated) == 1
    assert client.configs[0].endpoint == "https://93.184.216.34/v1"

    assert asyncio.run(client.chat("hello")) == "ok"
    assert len(validated) == 1
    assert all(call[1] == "user-1" for call in validated)


def test_pipeline_pins_every_model_http_client_to_user_scoped_endpoint(monkeypatch):
    validated = []
    pinned = []

    def fake_validate(url, user_id):
        validated.append((url, user_id))
        return url.rstrip("/")

    class FakeResponse:
        def __init__(self, url):
            self.url = str(url)

        def raise_for_status(self):
            return None

        def json(self):
            if self.url.endswith("/chat/completions"):
                return {"choices": [{"message": {"content": "ok"}}]}
            if self.url.endswith("/embeddings"):
                return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
            return {
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            return FakeResponse(request.url)

        async def aclose(self):
            return None

    monkeypatch.setattr(
        pipeline_module,
        "validate_user_outbound_url",
        fake_validate,
        raising=False,
    )
    def fake_pinned_client(endpoint, user_id, **kwargs):
        pinned.append((endpoint, user_id, kwargs))
        return FakeAsyncClient()

    monkeypatch.setattr(
        pipeline_module,
        "create_pinned_async_client",
        fake_pinned_client,
    )
    monkeypatch.delenv("MILVUS_HOST", raising=False)

    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path="unused",
            llm_config={
                "endpoint": "https://93.184.216.34/v1/",
                "model": "llm-model",
            },
            embedding_config={
                "endpoint": "https://93.184.216.35/v1/",
                "model": "embedding-model",
            },
            rerank_config={
                "endpoint": "https://93.184.216.36/v1/",
                "model": "rerank-model",
            },
            user_id="user-1",
        )
    )

    llm_client = pipeline._create_llm_client()
    embedding_client, _, _, rerank_client = pipeline._init_phase2_clients()

    async def exercise_clients():
        async with llm_client, embedding_client, rerank_client:
            assert await llm_client.chat("hello") == "ok"
            await embedding_client.embed("hello")
            await rerank_client.rerank("hello", ["document"])

    asyncio.run(exercise_clients())

    assert [item[:2] for item in pinned] == [
        ("https://93.184.216.34/v1", "user-1"),
        ("https://93.184.216.35/v1", "user-1"),
        ("https://93.184.216.36/v1", "user-1"),
    ]
    assert all(user_id == "user-1" for _, user_id in validated)


def test_pipeline_rejects_private_llm_endpoint_before_client_creation():
    pipeline = DatasetGenerationPipeline(
        PipelineConfig(
            input_path="unused",
            llm_config={
                "endpoint": "http://169.254.169.254/latest/meta-data",
                "model": "test-model",
            },
        )
    )

    with pytest.raises(SSRFError):
        pipeline._create_llm_client()


class _SecretReadGuard(dict):
    def __init__(self, *args, secret_reads, **kwargs):
        super().__init__(*args, **kwargs)
        self._secret_reads = secret_reads

    def get(self, key, default=None):
        if key == "api_key":
            self._secret_reads.append(key)
        return super().get(key, default)


@pytest.mark.parametrize("config_owner", [None, "user-2"])
def test_generation_model_config_default_rejects_unowned_before_secret_read(
    monkeypatch,
    config_owner,
):
    secret_reads = []
    config = _SecretReadGuard(
        {
            "config_id": "config-unowned",
            "model_type": "llm",
            "api_endpoint": "https://93.184.216.34/v1",
            "model_name": "test-model",
            "api_key": "must-not-be-read",
            "user_id": config_owner,
        },
        secret_reads=secret_reads,
    )
    monkeypatch.setattr(
        generation_routes.model_config_service,
        "get_config",
        lambda config_id: config,
    )

    with pytest.raises(HTTPException) as exc_info:
        generation_routes._resolve_model_config(
            "config-unowned",
            CURRENT_USER,
            {"llm"},
        )

    assert exc_info.value.status_code == 403
    assert secret_reads == []


def test_generation_background_rechecks_config_owner_before_pipeline(monkeypatch):
    input_path = _allowed_input_file("background-owner.txt")
    task = {
        "task_id": "generation-security-task",
        "status": GenerationStatus.PENDING,
        "user_id": "user-1",
        "input_path": str(input_path),
        "source_dataset_id": "dataset-1",
        "llm_config": {
            "config_id": "ownerless-llm",
            "endpoint": "https://93.184.216.34/v1",
            "model": "test-model",
            "api_key": "stale-secret",
        },
    }
    service = _RecordingTaskService()
    service.task = task
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "resolve_managed_local_dataset",
        lambda _dataset_id, *, user_id: {"storage_path": str(input_path)},
    )
    monkeypatch.setattr(
        generation_routes.model_config_service,
        "get_config",
        lambda config_id: {
            "config_id": config_id,
            "model_type": "llm",
            "user_id": None,
            "api_key": "ownerless-secret",
        },
    )
    pipeline_constructions = []

    class ForbiddenPipeline:
        def __init__(self, *args, **kwargs):
            pipeline_constructions.append((args, kwargs))
            raise AssertionError("pipeline must not start with an unowned config")

    monkeypatch.setattr(generation_routes, "DatasetGenerationPipeline", ForbiddenPipeline)

    asyncio.run(
        generation_routes._run_generation_task(
            "generation-security-task",
            PipelineConfig(
                input_path=str(input_path),
                llm_config=task["llm_config"],
            ),
            expected_run_token=service.run_token,
        )
    )

    assert pipeline_constructions == []
    assert service.status_updates[-1][1] == GenerationStatus.FAILED


def test_generation_background_refreshes_owned_credentials_after_owner_check(
    monkeypatch,
):
    input_path = _allowed_input_file("background-refresh.txt")
    task = {
        "task_id": "generation-refresh-task",
        "status": GenerationStatus.PENDING,
        "user_id": "user-1",
        "input_path": str(input_path),
        "source_dataset_id": "dataset-1",
        "llm_config": {
            "config_id": "owned-llm",
            "endpoint": "https://93.184.216.34/old-v1",
            "model": "old-model",
            "api_key": "***",
        },
    }
    service = _RecordingTaskService()
    service.task = task
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "resolve_managed_local_dataset",
        lambda _dataset_id, *, user_id: {"storage_path": str(input_path)},
    )
    monkeypatch.setattr(
        generation_routes.model_config_service,
        "get_config",
        lambda config_id: {
            "config_id": config_id,
            "model_type": "llm",
            "user_id": "user-1",
            "api_endpoint": "https://93.184.216.34/current-v1",
            "model_name": "current-model",
            "api_key": "current-secret",
        },
    )
    runtime_configs = []

    class CapturingPipeline:
        def __init__(self, config, **kwargs):
            runtime_configs.append(config.llm_config.copy())
            raise RuntimeError("stop after capturing runtime config")

    monkeypatch.setattr(generation_routes, "DatasetGenerationPipeline", CapturingPipeline)

    asyncio.run(
        generation_routes._run_generation_task(
            "generation-refresh-task",
            PipelineConfig(
                input_path=str(input_path),
                llm_config={
                    **task["llm_config"],
                    "api_key": "stale-secret",
                },
            ),
            expected_run_token=service.run_token,
        )
    )

    assert runtime_configs == [
        {
            "config_id": "owned-llm",
            "endpoint": "https://93.184.216.34/current-v1",
            "model": "current-model",
            "api_key": "current-secret",
        }
    ]


def test_generation_background_rechecks_dataset_owner_before_pipeline(monkeypatch):
    input_path = _allowed_input_file("background-dataset-owner.txt")
    task = {
        "task_id": "generation-dataset-owner-task",
        "status": GenerationStatus.PENDING,
        "user_id": "user-1",
        "source_dataset_id": "dataset-1",
        "input_path": str(input_path),
        "llm_config": None,
    }
    service = _RecordingTaskService()
    service.task = task
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        generation_routes,
        "resolve_managed_local_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ResourceProvenanceError("Dataset owner does not match")
        ),
    )
    pipeline_constructions = []

    class ForbiddenPipeline:
        def __init__(self, *args, **kwargs):
            pipeline_constructions.append((args, kwargs))
            raise AssertionError("pipeline must not read an unowned dataset")

    monkeypatch.setattr(generation_routes, "DatasetGenerationPipeline", ForbiddenPipeline)

    asyncio.run(
        generation_routes._run_generation_task(
            task["task_id"],
            PipelineConfig(input_path=str(input_path)),
            expected_run_token=service.run_token,
        )
    )

    assert pipeline_constructions == []
    assert service.status_updates[-1][1] == GenerationStatus.FAILED


def test_generation_restart_rechecks_dataset_before_reset(monkeypatch):
    from train_factory.storage.services.external_sync_service import (
        external_sync_service,
    )

    input_path = _allowed_input_file("restart-dataset-owner.txt")
    task = {
        "task_id": "generation-restart-task",
        "status": GenerationStatus.FAILED,
        "user_id": "user-1",
        "source_dataset_id": "dataset-1",
        "input_path": str(input_path),
    }
    service = _RecordingTaskService()
    service.task = task
    service.reset_progress = lambda *_args: (_ for _ in ()).throw(
        AssertionError("task must not be reset before dataset validation")
    )
    monkeypatch.setattr(generation_routes, "generation_task_service", service)
    monkeypatch.setattr(
        external_sync_service,
        "get_generation_by_task_id",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        generation_routes,
        "resolve_managed_local_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ResourceProvenanceError("Dataset owner does not match")
        ),
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            generation_routes.restart_task(
                task["task_id"],
                background_tasks,
                CURRENT_USER,
            )
        )

    assert exc_info.value.status_code == 403
    assert service.status_updates == []
    assert background_tasks.tasks == []
