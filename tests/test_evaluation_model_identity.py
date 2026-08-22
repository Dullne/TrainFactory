import asyncio
import hashlib
import importlib
import sys
from copy import deepcopy
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from train_factory.api.routes import evaluation_routes
from train_factory.evaluation import evaluation_runner
from train_factory.storage.entities.evaluation_task_entity import (
    EvaluationFramework,
    EvaluationStatus,
    EvaluationTaskDB,
)


USER = {"user_id": "user-1", "username": "user", "role": "user"}
IDENTITY_FIELD = "_evaluation_identity"
IDENTITY_SCHEMA_VERSION = 2


def _identity(result_key):
    return {"schema_version": IDENTITY_SCHEMA_VERSION, "result_key": result_key}


def _v1_identity(result_key):
    return {"schema_version": 1, "result_key": result_key}


@pytest.fixture
def persisted_evaluation_service(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'evaluation-identity.db'}")
    SQLModel.metadata.create_all(engine, tables=[EvaluationTaskDB.__table__])
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )

    @contextmanager
    def test_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(service_module, "get_session", test_session)
    return service_module.EvaluationTaskService(), engine


def _add_persisted_task(engine, **overrides):
    values = {
        "task_id": "legacy-resume-db",
        "task_name": "legacy-resume-db",
        "eval_framework": EvaluationFramework.MTEB,
        "status": EvaluationStatus.FAILED,
        "run_token": "attempt-1",
        "model_configs": [
            {
                "name": " legacy-model ",
                "model_name": "served-model",
                "endpoint": "https://one.example.test",
            }
        ],
        "dataset_configs": [{"type": "mteb", "name": "T2Reranking"}],
        "results": {
            " legacy-model ": {"T2Reranking": {"NDCG@10": 0.75}}
        },
        "model_progress": {
            " legacy-model ": {
                "T2Reranking": {"progress": 100, "status": "completed"}
            }
        },
    }
    values.update(overrides)
    with Session(engine) as session:
        session.add(EvaluationTaskDB(**values))
        session.commit()


def _request(model_configs):
    return evaluation_routes.CreateEvaluationRequest(
        model_configs=model_configs,
        dataset_configs=[
            evaluation_routes.DatasetConfig(type="mteb", name="T2Reranking")
        ],
        batch_size=1,
        workers=1,
        model_workers=1,
    )


def test_request_returns_422_when_model_identity_is_empty():
    app = FastAPI()

    @app.post("/validate")
    async def validate(request: evaluation_routes.CreateEvaluationRequest):
        return request.model_dump()

    response = TestClient(app).post(
        "/validate",
        json={
            "model_configs": [
                {
                    "endpoint": "https://reranker.example.test",
                    "name": "   ",
                    "model_name": "  ",
                }
            ],
            "dataset_configs": [{"type": "mteb", "name": "T2Reranking"}],
        },
    )

    assert response.status_code == 422


def test_trimmed_duplicate_model_identities_are_rejected_before_admission(monkeypatch):
    request = _request(
        [
            evaluation_routes.ModelConfig(
                endpoint="https://one.example.test",
                name=" duplicate ",
                model_name="served-one",
            ),
            evaluation_routes.ModelConfig(
                endpoint="https://two.example.test",
                name="duplicate",
                model_name="served-two",
            ),
        ]
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: pytest.fail("duplicate identity reached admission"),
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.create_evaluation_task(
                request,
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 400


def test_create_persists_canonical_model_name(monkeypatch):
    creations = []
    lease = SimpleNamespace(release=lambda: None)

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "create_task",
        lambda **kwargs: creations.append(kwargs) or {"task_id": "evaluation-1"},
    )

    request = _request(
        [
            evaluation_routes.ModelConfig(
                endpoint="https://reranker.example.test",
                name="  Display name  ",
                model_name="  served-model  ",
            )
        ]
    )
    asyncio.run(
        evaluation_routes.create_evaluation_task(
            request,
            BackgroundTasks(),
            USER,
        )
    )

    assert creations[0]["model_configs"] == [
        {
            "model_id": None,
            "endpoint": "https://reranker.example.test",
            "model_name": "served-model",
            "name": "Display name",
            "inference_framework": None,
            IDENTITY_FIELD: _identity("Display name"),
        }
    ]
    assert creations[0]["dataset_configs"] == [
        {
            "type": "mteb",
            "name": "T2Reranking",
            IDENTITY_FIELD: _identity("mteb:T2Reranking"),
        }
    ]


def test_route_rejects_normalized_mteb_type_with_unknown_dataset(monkeypatch):
    request = evaluation_routes.CreateEvaluationRequest(
        model_configs=[
            evaluation_routes.ModelConfig(
                endpoint="https://reranker.example.test",
                name="model-one",
                model_name="served-one",
            )
        ],
        dataset_configs=[
            evaluation_routes.DatasetConfig(type=" MTEB ", name="NotAllowlisted")
        ],
        batch_size=1,
        workers=1,
        model_workers=1,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: pytest.fail("invalid MTEB dataset reached admission"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.create_evaluation_task(
                request,
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 400


class _RunnerService:
    def __init__(
        self,
        model_configs,
        dataset_configs=None,
        model_progress=None,
        results=None,
    ):
        self.task = {
            "task_id": "legacy-evaluation",
            "status": "running",
            "run_token": "attempt-1",
            "user_id": None,
            "model_configs": model_configs,
            "dataset_configs": dataset_configs
            or [{"type": "mteb", "name": "T2Reranking"}],
            "max_samples": 1,
            "batch_size": 1,
            "workers": 1,
            "model_workers": 2,
            "model_progress": model_progress,
            "results": results,
        }
        self.completions = []
        self.progress_initializations = []
        self.progress_updates = []
        self.partial_results = []
        self.connection_attempts = 0
        self.evaluation_calls = []

    def get_task(self, _task_id):
        return self.task

    def update_status(self, *_args, **_kwargs):
        return True

    def init_model_progress(self, *args, **kwargs):
        self.progress_initializations.append((args, kwargs))
        return True

    def update_model_progress(self, *args, **kwargs):
        self.progress_updates.append((args, kwargs))
        return True

    def save_model_dataset_result(
        self,
        _task_id,
        model_name,
        dataset_name,
        result,
        **_kwargs,
    ):
        results = self.task.get("results")
        if not isinstance(results, dict):
            results = {}
            self.task["results"] = results
        results.setdefault(model_name, {})[dataset_name] = deepcopy(result)
        progress = self.task.get("model_progress")
        if not isinstance(progress, dict):
            progress = {}
            self.task["model_progress"] = progress
        progress.setdefault(model_name, {})[dataset_name] = {
            "progress": 100,
            "status": "completed",
        }
        return True

    def save_partial_results(self, _task_id, results, **_kwargs):
        self.partial_results.append(deepcopy(results))
        return True

    def complete_task(self, task_id, status, **kwargs):
        self.completions.append((task_id, status, kwargs))
        return True


def _run_legacy_models(
    monkeypatch,
    model_configs,
    *,
    dataset_configs=None,
    existing_results=None,
    model_progress=None,
    eval_existing_results=None,
    constructor_fail_models=None,
    evaluation_failures=None,
):
    service = _RunnerService(
        model_configs,
        dataset_configs,
        model_progress,
        deepcopy(existing_results or {}),
    )
    constructor_fail_models = set(constructor_fail_models or ())
    evaluation_failures = set(evaluation_failures or ())
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    monkeypatch.setattr(service_module, "evaluation_task_service", service)
    monkeypatch.setattr(evaluation_runner, "is_cancelled", lambda _task_id: False)
    monkeypatch.setattr(evaluation_runner, "_cleanup_cancelled", lambda _task_id: None)
    monkeypatch.setattr(
        evaluation_runner,
        "task_requires_dataset_provenance",
        lambda _user_id: False,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "resolve_local_evaluation_dataset_record",
        lambda _dataset_id: None,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "resolve_evaluation_dataset_path",
        lambda path, **_kwargs: path,
    )

    class FakeReranker:
        def __init__(self, *_args, **kwargs):
            self.model = kwargs.get("model", "")
            if self.model in constructor_fail_models:
                raise RuntimeError(f"constructor failed for {self.model}")

        def test_connection(self):
            service.connection_attempts += 1
            return True

    class FakeEvaluator:
        def __init__(self, reranker, *_args, **_kwargs):
            self.model = reranker.model

        def evaluate(self, **_kwargs):
            service.evaluation_calls.append(("mteb", _kwargs["task_name"]))
            if (self.model, _kwargs["task_name"]) in evaluation_failures:
                raise RuntimeError(f"evaluation failed for {_kwargs['task_name']}")
            return {"source": "mteb", "NDCG@10": 1.0}

        def evaluate_local(self, **_kwargs):
            service.evaluation_calls.append(("local", _kwargs["dataset"]))
            if (self.model, _kwargs["dataset"]) in evaluation_failures:
                raise RuntimeError(f"evaluation failed for {_kwargs['dataset']}")
            return {"source": "local", "NDCG@10": 0.5}

    fake_qwen_evaluation = ModuleType("qwen3_rerank_trainer.evaluation")
    fake_qwen_evaluation.MTEBRerankEvaluator = FakeEvaluator
    monkeypatch.setitem(
        sys.modules,
        "qwen3_rerank_trainer.evaluation",
        fake_qwen_evaluation,
    )
    monkeypatch.setattr(evaluation_runner, "SecureAPIReranker", FakeReranker)

    evaluation_runner.run_evaluation_task(
        "legacy-evaluation",
        {
            "existing_results": (
                eval_existing_results
                if eval_existing_results is not None
                else existing_results or {}
            ),
            "existing_model_progress": model_progress or {},
        },
    )
    return service


def test_runner_rejects_normalized_mteb_type_with_unknown_dataset(monkeypatch):
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        dataset_configs=[{"type": " MTEB ", "name": "NotAllowlisted"}],
    )

    assert service.completions[-1][1] == "failed"
    assert "dataset" in service.completions[-1][2]["error_message"].lower()
    assert service.evaluation_calls == []


def test_runner_falls_back_from_legacy_null_names_without_overwriting(monkeypatch):
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": None,
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                "inference_framework": None,
            },
            {
                "name": None,
                "model_name": "served-two",
                "endpoint": "https://two.example.test",
            },
        ],
    )

    assert service.completions[-1][1] == "succeeded"
    assert set(service.completions[-1][2]["results"]) == {
        "served-one",
        "served-two",
    }
    assert "_error" not in service.completions[-1][2]["results"]["served-one"]


def test_resume_maps_invalid_legacy_model_identity_to_400_before_admission(
    monkeypatch,
):
    task = {
        "task_id": "legacy-resume",
        "status": "failed",
        "user_id": USER["user_id"],
        "model_configs": [
            {
                "name": None,
                "model_name": " duplicate ",
                "endpoint": "https://one.example.test",
            },
            {
                "name": "duplicate",
                "model_name": "served-two",
                "endpoint": "https://two.example.test",
            },
        ],
        "dataset_configs": [{"type": "mteb", "name": "T2Reranking"}],
        "results": {},
        "batch_size": 1,
        "workers": 1,
        "model_workers": 1,
    }
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: pytest.fail("invalid identity reached admission"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.resume_evaluation_task(
                "legacy-resume",
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 400


def test_runner_rejects_duplicate_legacy_identity_instead_of_overwriting(monkeypatch):
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": None,
                "model_name": " duplicate ",
                "endpoint": "https://one.example.test",
            },
            {
                "name": "duplicate",
                "model_name": "served-two",
                "endpoint": "https://two.example.test",
            },
        ],
    )

    assert service.completions[-1][1] == "failed"
    assert "duplicate" in service.completions[-1][2]["error_message"].lower()
    assert "results" not in service.completions[-1][2]


def test_same_display_name_datasets_keep_distinct_results_and_progress(monkeypatch):
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        dataset_configs=[
            {"type": "mteb", "name": "T2Reranking"},
            {
                "type": "registered",
                "name": "T2Reranking",
                "dataset_id": "dataset-1",
                "path": "/managed/dataset-1.jsonl",
            },
        ],
    )

    assert service.completions[-1][1] == "succeeded"
    assert service.completions[-1][2]["results"] == {
        "model-one": {
            "mteb:T2Reranking": {"source": "mteb", "NDCG@10": 1.0},
            "dataset:dataset-1": {"source": "local", "NDCG@10": 0.5},
        }
    }
    assert service.progress_initializations[-1][0][2] == [
        "mteb:T2Reranking",
        "dataset:dataset-1",
    ]
    assert service.task["model_progress"] == {
        "model-one": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"},
            "dataset:dataset-1": {"progress": 100, "status": "completed"},
        }
    }


@pytest.mark.parametrize(
    ("model_config", "legacy_model_key", "canonical_model_key"),
    [
        pytest.param(
            {
                "name": " legacy-name ",
                "model_name": "served-name",
                "endpoint": "https://one.example.test",
            },
            " legacy-name ",
            "legacy-name",
            id="spaced-display-name",
        ),
        pytest.param(
            {
                "name": None,
                "model_name": "served-name",
                "endpoint": "https://one.example.test",
            },
            "null",
            "served-name",
            id="null-display-name",
        ),
        pytest.param(
            {
                "model_name": "served-name",
                "endpoint": "https://one.example.test",
            },
            "served-name",
            "served-name",
            id="missing-display-name-fallback",
        ),
    ],
)
def test_legacy_resume_identity_skips_success_without_overwriting(
    monkeypatch,
    model_config,
    legacy_model_key,
    canonical_model_key,
):
    success = {"NDCG@10": 0.75}
    service = _run_legacy_models(
        monkeypatch,
        [model_config],
        existing_results={
            legacy_model_key: {"T2Reranking": success},
        },
        model_progress={
            legacy_model_key: {
                "T2Reranking": {"progress": 100, "status": "completed"}
            }
        },
    )

    assert service.connection_attempts == 0
    assert service.evaluation_calls == []
    assert service.completions[-1][1] == "succeeded"
    assert service.completions[-1][2]["results"] == {
        canonical_model_key: {"mteb:T2Reranking": success}
    }
    init_args, init_kwargs = service.progress_initializations[-1]
    assert init_args[1:] == ([canonical_model_key], ["mteb:T2Reranking"])
    assert init_kwargs["preserve_completed"] is True


@pytest.mark.parametrize("stored_dataset_key", ["T2Reranking", "mteb:T2Reranking"])
def test_resume_state_reads_legacy_and_namespaced_dataset_keys(stored_dataset_key):
    success = {"NDCG@10": 0.75}

    state = evaluation_runner.canonicalize_evaluation_resume_state(
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        [{"type": "mteb", "name": "T2Reranking"}],
        {"model-one": {stored_dataset_key: success}},
        {
            "model-one": {
                stored_dataset_key: {"progress": 100, "status": "completed"}
            }
        },
    )

    assert state["model_configs"][0]["name"] == "model-one"
    assert state["dataset_configs"][0] == {
        "type": "mteb",
        "name": "T2Reranking",
        "result_key": "mteb:T2Reranking",
        IDENTITY_FIELD: _identity("mteb:T2Reranking"),
    }
    assert state["results"] == {
        "model-one": {"mteb:T2Reranking": success}
    }
    assert state["model_progress"] == {
        "model-one": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"}
        }
    }


def test_resume_state_rejects_canonical_looking_legacy_dataset_alias():
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        evaluation_runner.canonicalize_evaluation_resume_state(
            [
                {
                    "name": "model-one",
                    "model_name": "served-one",
                    "endpoint": "https://one.example.test",
                }
            ],
            [
                {"type": "mteb", "name": "T2Reranking"},
                {
                    "type": "registered",
                    "name": "mteb:T2Reranking",
                    "dataset_id": "dataset-1",
                    "path": "/managed/dataset-1.jsonl",
                },
            ],
            {"model-one": {"mteb:T2Reranking": {"NDCG@10": 0.75}}},
            {},
        )


def test_versioned_identity_snapshot_makes_canonical_keys_authoritative():
    initial = evaluation_runner.canonicalize_evaluation_resume_state(
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        [
            {"type": "mteb", "name": "T2Reranking"},
            {
                "type": "registered",
                "name": "mteb:T2Reranking",
                "dataset_id": "dataset-1",
                "path": "/managed/dataset-1.jsonl",
            },
        ],
        {},
        {},
    )

    assert all(
        config[IDENTITY_FIELD]["schema_version"] == IDENTITY_SCHEMA_VERSION
        for config in initial["model_configs"] + initial["dataset_configs"]
    )
    resumed = evaluation_runner.canonicalize_evaluation_resume_state(
        initial["model_configs"],
        initial["dataset_configs"],
        {
            "model-one": {
                "mteb:T2Reranking": {"NDCG@10": 0.75},
                "dataset:dataset-1": {"NDCG@10": 0.5},
            }
        },
        {},
    )

    assert set(resumed["results"]["model-one"]) == {
        "mteb:T2Reranking",
        "dataset:dataset-1",
    }


def test_v1_identity_snapshot_migrates_all_dataset_namespaces_to_v2():
    local_path = "/managed/private/v1-local.jsonl"
    local_v2_key = "local:sha256:" + hashlib.sha256(local_path.encode()).hexdigest()
    state = evaluation_runner.canonicalize_evaluation_resume_state(
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                IDENTITY_FIELD: _v1_identity("model-one"),
            }
        ],
        [
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _v1_identity("mteb:T2Reranking"),
            },
            {
                "type": "registered",
                "name": "Registered",
                "dataset_id": "dataset-1",
                "path": "/managed/dataset-1.jsonl",
                IDENTITY_FIELD: _v1_identity("dataset:dataset-1"),
            },
            {
                "type": "local",
                "name": "Local",
                "path": local_path,
                IDENTITY_FIELD: _v1_identity(f"local:{local_path}"),
            },
        ],
        {
            "model-one": {
                "mteb:T2Reranking": {"NDCG@10": 0.9},
                "dataset:dataset-1": {"NDCG@10": 0.8},
                f"local:{local_path}": {"NDCG@10": 0.7},
            }
        },
        {
            "model-one": {
                "mteb:T2Reranking": {"progress": 100, "status": "completed"},
                "dataset:dataset-1": {"progress": 100, "status": "completed"},
                f"local:{local_path}": {"progress": 100, "status": "completed"},
            }
        },
    )

    assert state["source_identity_schema_version"] == 1
    assert {
        config[IDENTITY_FIELD]["schema_version"]
        for config in state["model_configs"] + state["dataset_configs"]
    } == {IDENTITY_SCHEMA_VERSION}
    assert set(state["results"]["model-one"]) == {
        "mteb:T2Reranking",
        "dataset:dataset-1",
        local_v2_key,
    }
    assert state["model_progress"]["model-one"][local_v2_key] == {
        "progress": 100,
        "status": "completed",
    }


@pytest.mark.parametrize(
    ("dataset_config", "stored_key"),
    [
        (
            {"type": "mteb", "name": "T2Reranking"},
            "mteb:wrong",
        ),
        (
            {
                "type": "registered",
                "name": "Registered",
                "dataset_id": "dataset-1",
                "path": "/managed/dataset-1.jsonl",
            },
            "dataset:wrong",
        ),
        (
            {
                "type": "local",
                "name": "Local",
                "path": "/managed/v1-local.jsonl",
            },
            "local:/managed/wrong.jsonl",
        ),
    ],
)
def test_v1_identity_marker_key_must_match_its_config_exactly(
    dataset_config,
    stored_key,
):
    dataset_config = {
        **dataset_config,
        IDENTITY_FIELD: _v1_identity(stored_key),
    }
    with pytest.raises(ValueError, match="key"):
        evaluation_runner.canonicalize_evaluation_resume_state(
            [
                {
                    "name": "model-one",
                    IDENTITY_FIELD: _v1_identity("model-one"),
                }
            ],
            [dataset_config],
            {},
            {},
        )


@pytest.mark.parametrize(
    ("model_version", "dataset_version"),
    [(1, 2), (99, 99)],
)
def test_versioned_identity_snapshot_rejects_mixed_or_unknown_versions(
    model_version,
    dataset_version,
):
    with pytest.raises(ValueError, match="version"):
        evaluation_runner.canonicalize_evaluation_resume_state(
            [
                {
                    "name": "model-one",
                    IDENTITY_FIELD: {
                        "schema_version": model_version,
                        "result_key": "model-one",
                    },
                }
            ],
            [
                {
                    "type": "mteb",
                    "name": "T2Reranking",
                    IDENTITY_FIELD: {
                        "schema_version": dataset_version,
                        "result_key": "mteb:T2Reranking",
                    },
                }
            ],
            {},
            {},
        )


def test_v1_snapshot_rejects_old_and_v2_result_key_collision():
    local_path = "/managed/v1-local.jsonl"
    v2_key = "local:sha256:" + hashlib.sha256(local_path.encode()).hexdigest()
    with pytest.raises(ValueError, match="Unknown|collision"):
        evaluation_runner.canonicalize_evaluation_resume_state(
            [
                {
                    "name": "model-one",
                    IDENTITY_FIELD: _v1_identity("model-one"),
                }
            ],
            [
                {
                    "type": "local",
                    "name": "Local",
                    "path": local_path,
                    IDENTITY_FIELD: _v1_identity(f"local:{local_path}"),
                }
            ],
            {
                "model-one": {
                    f"local:{local_path}": {"NDCG@10": 0.7},
                    v2_key: {"NDCG@10": 0.8},
                }
            },
            {},
        )


def test_resume_route_admits_fully_migrated_v1_snapshot(monkeypatch):
    local_path = "/managed/private/v1-local.jsonl"
    local_v2_key = "local:sha256:" + hashlib.sha256(local_path.encode()).hexdigest()
    task = {
        "task_id": "v1-resume",
        "status": "failed",
        "user_id": USER["user_id"],
        "model_configs": [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                IDENTITY_FIELD: _v1_identity("model-one"),
            }
        ],
        "dataset_configs": [
            {
                "type": "local",
                "name": "Local",
                "path": local_path,
                IDENTITY_FIELD: _v1_identity(f"local:{local_path}"),
            }
        ],
        "results": {
            "model-one": {f"local:{local_path}": {"NDCG@10": 0.7}}
        },
        "model_progress": {
            "model-one": {
                f"local:{local_path}": {"progress": 100, "status": "completed"}
            }
        },
        "batch_size": 1,
        "workers": 1,
        "model_workers": 1,
    }
    captured = {}
    lease = SimpleNamespace(release=lambda: None)

    def admit(_kind, _task_id, _user_id, _operation, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True, lease

    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "requires_tenant_provenance",
        lambda _current_user: False,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_evaluation_dataset_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )

    asyncio.run(
        evaluation_routes.resume_evaluation_task(
            "v1-resume",
            BackgroundTasks(),
            USER,
        )
    )

    assert captured["args"] == ("v1-resume",)
    assert captured["kwargs"]["model_configs"][0][IDENTITY_FIELD] == _identity(
        "model-one"
    )
    assert captured["kwargs"]["dataset_configs"][0][IDENTITY_FIELD] == _identity(
        local_v2_key
    )
    assert captured["kwargs"]["results"] == {
        "model-one": {local_v2_key: {"NDCG@10": 0.7}}
    }
    assert captured["kwargs"]["model_progress"] == {
        "model-one": {
            local_v2_key: {"progress": 100, "status": "completed"}
        }
    }


def test_resume_rejects_legacy_model_progress_collision_before_admission(monkeypatch):
    task = {
        "task_id": "legacy-resume",
        "status": "failed",
        "user_id": USER["user_id"],
        "model_configs": [
            {
                "name": " legacy-name ",
                "model_name": "served-name",
                "endpoint": "https://one.example.test",
            }
        ],
        "dataset_configs": [{"type": "mteb", "name": "T2Reranking"}],
        "results": {},
        "model_progress": {
            " legacy-name ": {
                "T2Reranking": {"progress": 100, "status": "completed"}
            },
            "legacy-name": {
                "T2Reranking": {"progress": 0, "status": "failed"}
            },
        },
        "batch_size": 1,
        "workers": 1,
        "model_workers": 1,
    }
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        lambda *_args, **_kwargs: pytest.fail("collision reached admission"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            evaluation_routes.resume_evaluation_task(
                "legacy-resume",
                BackgroundTasks(),
                USER,
            )
        )

    assert exc_info.value.status_code == 400
    assert "invalid" in str(exc_info.value.detail).lower()


def test_resume_admission_receives_complete_canonical_snapshot(monkeypatch):
    task = {
        "task_id": "legacy-resume",
        "status": "failed",
        "user_id": USER["user_id"],
        "model_configs": [
            {
                "name": " legacy-name ",
                "model_name": "served-name",
                "endpoint": "https://one.example.test",
            }
        ],
        "dataset_configs": [{"type": "mteb", "name": "T2Reranking"}],
        "results": {
            " legacy-name ": {"T2Reranking": {"NDCG@10": 0.75}}
        },
        "model_progress": {
            " legacy-name ": {
                "T2Reranking": {"progress": 100, "status": "completed"}
            }
        },
        "batch_size": 1,
        "workers": 1,
        "model_workers": 1,
    }
    captured = {}
    lease = SimpleNamespace(release=lambda: None)

    def admit(_kind, _task_id, _user_id, _operation, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True, lease

    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        evaluation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )

    asyncio.run(
        evaluation_routes.resume_evaluation_task(
            "legacy-resume",
            BackgroundTasks(),
            USER,
        )
    )

    assert captured["args"] == ("legacy-resume",)
    assert captured["kwargs"]["model_configs"][0][IDENTITY_FIELD] == _identity(
        "legacy-name"
    )
    assert captured["kwargs"]["dataset_configs"][0][IDENTITY_FIELD] == _identity(
        "mteb:T2Reranking"
    )
    assert captured["kwargs"]["results"] == {
        "legacy-name": {"mteb:T2Reranking": {"NDCG@10": 0.75}}
    }
    assert captured["kwargs"]["model_progress"] == {
        "legacy-name": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"}
        }
    }


def test_reset_for_resume_atomically_persists_identity_snapshot(
    persisted_evaluation_service,
):
    service, engine = persisted_evaluation_service
    _add_persisted_task(engine)
    model_configs = [
        {
            "name": "legacy-model",
            "model_name": "served-model",
            "endpoint": "https://one.example.test",
            IDENTITY_FIELD: _identity("legacy-model"),
        }
    ]
    dataset_configs = [
        {
            "type": "mteb",
            "name": "T2Reranking",
            IDENTITY_FIELD: _identity("mteb:T2Reranking"),
        }
    ]
    results = {
        "legacy-model": {"mteb:T2Reranking": {"NDCG@10": 0.75}}
    }
    model_progress = {
        "legacy-model": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"}
        }
    }

    assert service.reset_for_resume(
        "legacy-resume-db",
        model_configs=model_configs,
        dataset_configs=dataset_configs,
        results=results,
        model_progress=model_progress,
    )
    stored = service.get_task("legacy-resume-db")
    assert stored["status"] == EvaluationStatus.PENDING
    assert stored["model_configs"] == model_configs
    assert stored["dataset_configs"] == dataset_configs
    assert stored["results"] == results
    assert stored["model_progress"] == model_progress
    assert stored["progress"] == 100


def test_reset_for_resume_cas_failure_does_not_partially_persist_snapshot(
    persisted_evaluation_service,
):
    service, engine = persisted_evaluation_service
    _add_persisted_task(engine, status=EvaluationStatus.RUNNING)
    original = service.get_task("legacy-resume-db")

    assert not service.reset_for_resume(
        "legacy-resume-db",
        model_configs=[
            {
                "name": "canonical",
                IDENTITY_FIELD: _identity("canonical"),
            }
        ],
        dataset_configs=[
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _identity("mteb:T2Reranking"),
            }
        ],
        results={"canonical": {}},
        model_progress={"canonical": {}},
    )
    stored = service.get_task("legacy-resume-db")
    assert stored["status"] == EvaluationStatus.RUNNING
    assert stored["model_configs"] == original["model_configs"]
    assert stored["dataset_configs"] == original["dataset_configs"]
    assert stored["results"] == original["results"]
    assert stored["model_progress"] == original["model_progress"]


def test_evaluation_detail_omits_private_identity_metadata(monkeypatch):
    task = {
        "task_id": "evaluation-1",
        "task_name": "evaluation-1",
        "eval_type": "reranker-mteb",
        "status": "succeeded",
        "progress": 100,
        "user_id": USER["user_id"],
        "model_configs": [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                IDENTITY_FIELD: _identity("model-one"),
            }
        ],
        "dataset_configs": [
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _identity("mteb:T2Reranking"),
            }
        ],
    }
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )

    response = asyncio.run(
        evaluation_routes.get_evaluation_task_detail("evaluation-1", USER)
    )

    assert IDENTITY_FIELD not in response.model_configs[0]
    assert IDENTITY_FIELD not in response.dataset_configs[0]


def test_runner_prefers_persisted_versioned_snapshot_over_stale_resume_arguments(
    monkeypatch,
):
    success = {"NDCG@10": 0.75}
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                IDENTITY_FIELD: _identity("model-one"),
            }
        ],
        dataset_configs=[
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _identity("mteb:T2Reranking"),
            }
        ],
        existing_results={
            "model-one": {"mteb:T2Reranking": success},
        },
        eval_existing_results={
            "model-one": {
                "mteb:T2Reranking": {"error": "stale resume argument"}
            }
        },
        model_progress={
            "model-one": {
                "mteb:T2Reranking": {"progress": 100, "status": "completed"}
            }
        },
    )

    assert service.connection_attempts == 0
    assert service.evaluation_calls == []
    assert service.progress_initializations[-1][1]["preserve_completed"] is True
    assert service.completions[-1][2]["results"] == {
        "model-one": {"mteb:T2Reranking": success}
    }


def test_every_partial_snapshot_contains_all_persisted_model_successes(monkeypatch):
    model_configs = [
        {
            "name": name,
            "model_name": f"served-{index}",
            "endpoint": f"https://{index}.example.test",
            IDENTITY_FIELD: _identity(name),
        }
        for index, name in enumerate(("model-one", "model-two"), start=1)
    ]
    dataset_configs = [
        {
            "type": "mteb",
            "name": "T2Reranking",
            IDENTITY_FIELD: _identity("mteb:T2Reranking"),
        }
    ]
    existing_results = {
        name: {"mteb:T2Reranking": {"NDCG@10": score}}
        for name, score in (("model-one", 0.8), ("model-two", 0.7))
    }

    service = _run_legacy_models(
        monkeypatch,
        model_configs,
        dataset_configs=dataset_configs,
        existing_results=existing_results,
    )

    assert service.partial_results
    assert all(
        set(snapshot) == {"model-one", "model-two"}
        for snapshot in service.partial_results
    )
    assert service.completions[-1][2]["results"] == existing_results


def test_future_error_merges_with_success_and_fails_missing_pair(monkeypatch):
    success = {"NDCG@10": 0.75}
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                IDENTITY_FIELD: _identity("model-one"),
            }
        ],
        dataset_configs=[
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _identity("mteb:T2Reranking"),
            },
            {
                "type": "mteb",
                "name": "MMarcoReranking",
                IDENTITY_FIELD: _identity("mteb:MMarcoReranking"),
            },
        ],
        existing_results={
            "model-one": {"mteb:T2Reranking": success},
        },
        model_progress={
            "model-one": {
                "mteb:T2Reranking": {"progress": 100, "status": "completed"},
                "mteb:MMarcoReranking": {"progress": 0, "status": "pending"},
            }
        },
        constructor_fail_models={"served-one"},
    )

    assert service.completions[-1][1] == "failed"
    results = service.completions[-1][2]["results"]["model-one"]
    assert results["mteb:T2Reranking"] == success
    assert "constructor failed" in results["_error"]["error"]
    assert (
        "legacy-evaluation",
        "model-one",
        "mteb:MMarcoReranking",
        100,
        "failed",
    ) in [args for args, _kwargs in service.progress_updates]
    assert service.partial_results[-1]["model-one"]["mteb:T2Reranking"] == success


def test_partial_dataset_failure_never_marks_task_succeeded(monkeypatch):
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        dataset_configs=[
            {"type": "mteb", "name": "T2Reranking"},
            {"type": "mteb", "name": "MMarcoReranking"},
        ],
        evaluation_failures={("served-one", "MMarcoReranking")},
    )

    assert service.completions[-1][1] == "failed"
    results = service.completions[-1][2]["results"]["model-one"]
    assert "error" not in results["mteb:T2Reranking"]
    assert "error" in results["mteb:MMarcoReranking"]


def test_long_local_identity_uses_bounded_display_label_for_current_dataset(
    monkeypatch,
):
    long_path = "/managed/" + ("segment-" * 45) + "dataset.jsonl"
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        dataset_configs=[
            {
                "type": "local",
                "name": "Readable local dataset",
                "path": long_path,
            }
        ],
    )

    running_updates = [
        (args, kwargs)
        for args, kwargs in service.progress_updates
        if args[4] == "running"
    ]
    assert running_updates
    assert running_updates[0][0][2].startswith("local:sha256:")
    assert long_path not in running_updates[0][0][2]
    assert len(running_updates[0][0][2]) <= 255
    assert running_updates[0][1]["current_dataset"] == "Readable local dataset"
    assert len(running_updates[0][1]["current_dataset"]) <= 255


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("dataset_id", "victim-dataset"),
        ("path", "/managed/victim-dataset.jsonl"),
    ],
)
def test_request_rejects_local_only_fields_on_mteb(field_name, field_value):
    app = FastAPI()

    @app.post("/validate")
    async def validate(request: evaluation_routes.CreateEvaluationRequest):
        return request.model_dump()

    dataset = {"type": " MTEB ", "name": "T2Reranking", field_name: field_value}
    response = TestClient(app).post(
        "/validate",
        json={
            "model_configs": [
                {
                    "endpoint": "https://reranker.example.test",
                    "name": "model-one",
                }
            ],
            "dataset_configs": [dataset],
        },
    )

    assert response.status_code == 422


def test_route_rejects_mteb_reference_smuggling_before_resolution(monkeypatch):
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_local_evaluation_dataset_record",
        lambda _dataset_id: pytest.fail("smuggled MTEB dataset ID was resolved"),
    )

    with pytest.raises(HTTPException) as exc_info:
        evaluation_routes._validate_dataset_configs(
            [
                {
                    "type": "mteb",
                    "name": "T2Reranking",
                    "dataset_id": "victim-dataset",
                }
            ],
            {"user_id": None, "username": "anonymous"},
        )

    assert exc_info.value.status_code == 400


def test_runner_rejects_persisted_mteb_reference_smuggling(monkeypatch):
    service = _run_legacy_models(
        monkeypatch,
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        dataset_configs=[
            {
                "type": "mteb",
                "name": "T2Reranking",
                "dataset_id": "victim-dataset",
                "path": "/managed/victim-dataset.jsonl",
            }
        ],
    )

    assert service.completions[-1][1] == "failed"
    assert service.evaluation_calls == []


def test_mteb_smuggled_references_never_lock_or_register_as_consumers(
    persisted_evaluation_service,
    monkeypatch,
):
    service, _engine = persisted_evaluation_service
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    lock_calls = []
    monkeypatch.setattr(
        service_module,
        "lock_datasets_for_consumption",
        lambda _session, **kwargs: lock_calls.append(kwargs),
    )

    created = service.create_task(
        task_name="malicious-mteb-reference",
        model_configs=[],
        dataset_configs=[
            {
                "type": "mteb",
                "name": "T2Reranking",
                "dataset_id": "victim-dataset",
                "path": "/managed/victim-dataset.jsonl",
            }
        ],
        require_managed_datasets=True,
    )

    assert lock_calls[0]["dataset_ids"] == set()
    assert lock_calls[0]["storage_refs"] == set()
    assert service.list_active_dataset_consumers(
        ["victim-dataset"],
        ["/managed/victim-dataset.jsonl"],
    ) == []
    assert created["task_id"]


def test_spoofed_dataset_identity_fields_are_recomputed(monkeypatch):
    monkeypatch.setattr(
        evaluation_routes,
        "requires_tenant_provenance",
        lambda _current_user: False,
    )
    monkeypatch.setattr(
        evaluation_routes,
        "resolve_evaluation_dataset_path",
        lambda path, **_kwargs: path,
    )
    path = "/managed/real-dataset.jsonl"
    expected_key = "local:sha256:" + hashlib.sha256(path.encode()).hexdigest()

    validated = evaluation_routes._validate_dataset_configs(
        [
            {
                "type": "local",
                "name": "Real dataset",
                "path": path,
                "result_key": "dataset:victim",
                IDENTITY_FIELD: _identity("dataset:victim"),
            }
        ],
        {"user_id": None, "username": "anonymous"},
    )

    assert "result_key" not in validated[0]
    assert validated[0][IDENTITY_FIELD] == _identity(expected_key)


def test_local_path_identity_is_stable_and_never_contains_the_path():
    path = "/managed/private/customer-a/eval.jsonl"
    first = evaluation_runner.canonical_evaluation_dataset_key(
        {"type": "local", "name": "Private", "path": path}
    )
    second = evaluation_runner.canonical_evaluation_dataset_key(
        {"type": "local", "name": "Renamed", "path": path}
    )

    assert first == second
    assert first.startswith("local:sha256:")
    assert path not in first


def test_public_identity_contract_exposes_keys_but_never_private_paths(monkeypatch):
    local_path = "/managed/private/customer-a/eval.jsonl"
    local_key = "local:sha256:" + hashlib.sha256(local_path.encode()).hexdigest()
    task = {
        "task_id": "evaluation-public-identity",
        "task_name": "evaluation-public-identity",
        "eval_type": "reranker-mixed",
        "status": "succeeded",
        "progress": 100,
        "user_id": USER["user_id"],
        "model_configs": [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                IDENTITY_FIELD: _identity("model-one"),
            }
        ],
        "dataset_configs": [
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _identity("mteb:T2Reranking"),
            },
            {
                "type": "local",
                "name": "Private dataset",
                "path": local_path,
                IDENTITY_FIELD: _identity(local_key),
            },
        ],
    }
    monkeypatch.setattr(
        evaluation_routes.evaluation_task_service,
        "get_task",
        lambda _task_id: task,
    )

    response = asyncio.run(
        evaluation_routes.get_evaluation_task_detail(
            "evaluation-public-identity",
            USER,
        )
    )
    payload = response.model_dump()

    assert payload["identity_schema_version"] == IDENTITY_SCHEMA_VERSION
    assert payload["dataset_configs"] == [
        {
            "type": "mteb",
            "name": "T2Reranking",
            "result_key": "mteb:T2Reranking",
        },
        {
            "type": "local",
            "name": "Private dataset",
            "result_key": local_key,
        },
    ]
    assert payload["identity_map"] == {
        "models": {"model-one": {"name": "model-one"}},
        "datasets": {
            "mteb:T2Reranking": {"name": "T2Reranking", "type": "mteb"},
            local_key: {"name": "Private dataset", "type": "local"},
        },
    }
    assert local_path not in str(payload)
    assert IDENTITY_FIELD not in str(payload)


def test_resume_progress_is_reconciled_from_successful_results():
    state = evaluation_runner.canonicalize_evaluation_resume_state(
        [
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
            }
        ],
        [
            {"type": "mteb", "name": "T2Reranking"},
            {"type": "mteb", "name": "MMarcoReranking"},
        ],
        {
            "model-one": {
                "T2Reranking": {"NDCG@10": 0.75},
                "MMarcoReranking": {"error": "failed"},
            }
        },
        {
            "model-one": {
                "T2Reranking": {"progress": 0, "status": "pending"},
                "MMarcoReranking": {"progress": 100, "status": "completed"},
            }
        },
    )

    assert state["model_progress"] == {
        "model-one": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"},
            "mteb:MMarcoReranking": {"progress": 0, "status": "pending"},
        }
    }


def test_pair_result_and_completed_progress_commit_atomically(
    persisted_evaluation_service,
):
    service, engine = persisted_evaluation_service
    _add_persisted_task(
        engine,
        status=EvaluationStatus.RUNNING,
        run_token="attempt-atomic",
        model_configs=[
            {
                "name": "model-one",
                IDENTITY_FIELD: _identity("model-one"),
            }
        ],
        dataset_configs=[
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _identity("mteb:T2Reranking"),
            }
        ],
        results={},
        model_progress={
            "model-one": {
                "mteb:T2Reranking": {"progress": 0, "status": "running"}
            }
        },
    )

    assert service.save_model_dataset_result(
        "legacy-resume-db",
        "model-one",
        "mteb:T2Reranking",
        {"NDCG@10": 0.75},
        run_token="attempt-atomic",
    )
    stored = service.get_task("legacy-resume-db")
    assert stored["results"] == {
        "model-one": {"mteb:T2Reranking": {"NDCG@10": 0.75}}
    }
    assert stored["model_progress"] == {
        "model-one": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"}
        }
    }


def test_later_progress_failure_cannot_erase_durable_successful_pair(
    persisted_evaluation_service,
    monkeypatch,
):
    service, engine = persisted_evaluation_service
    _add_persisted_task(
        engine,
        status=EvaluationStatus.RUNNING,
        run_token="attempt-durable",
        model_configs=[
            {
                "name": "model-one",
                "model_name": "served-one",
                "endpoint": "https://one.example.test",
                IDENTITY_FIELD: _identity("model-one"),
            }
        ],
        dataset_configs=[
            {
                "type": "mteb",
                "name": "T2Reranking",
                IDENTITY_FIELD: _identity("mteb:T2Reranking"),
            },
            {
                "type": "mteb",
                "name": "MMarcoReranking",
                IDENTITY_FIELD: _identity("mteb:MMarcoReranking"),
            },
        ],
        results={},
        model_progress={
            "model-one": {
                "mteb:T2Reranking": {"progress": 0, "status": "pending"},
                "mteb:MMarcoReranking": {"progress": 0, "status": "pending"},
            }
        },
    )
    service_module = importlib.import_module(
        "train_factory.storage.services.evaluation_task_service"
    )
    monkeypatch.setattr(service_module, "evaluation_task_service", service)
    monkeypatch.setattr(evaluation_runner, "is_cancelled", lambda _task_id: False)
    monkeypatch.setattr(evaluation_runner, "_cleanup_cancelled", lambda _task_id: None)
    monkeypatch.setattr(
        evaluation_runner,
        "task_requires_dataset_provenance",
        lambda _user_id: False,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    original_update = service.update_model_progress

    def fail_on_second_start(*args, **kwargs):
        if args[2] == "mteb:MMarcoReranking" and args[4] == "running":
            raise RuntimeError("progress store failed after first success")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(service, "update_model_progress", fail_on_second_start)

    class FakeReranker:
        def __init__(self, *_args, **_kwargs):
            pass

        def test_connection(self):
            return True

    class FakeEvaluator:
        def __init__(self, *_args, **_kwargs):
            pass

        def evaluate(self, **_kwargs):
            return {"NDCG@10": 0.75}

    fake_qwen_evaluation = ModuleType("qwen3_rerank_trainer.evaluation")
    fake_qwen_evaluation.MTEBRerankEvaluator = FakeEvaluator
    monkeypatch.setitem(
        sys.modules,
        "qwen3_rerank_trainer.evaluation",
        fake_qwen_evaluation,
    )
    monkeypatch.setattr(evaluation_runner, "SecureAPIReranker", FakeReranker)

    evaluation_runner.run_evaluation_task("legacy-resume-db", {})

    stored = service.get_task("legacy-resume-db")
    assert stored["status"] == EvaluationStatus.FAILED
    assert stored["results"]["model-one"]["mteb:T2Reranking"] == {
        "NDCG@10": 0.75
    }
    assert stored["model_progress"]["model-one"]["mteb:T2Reranking"] == {
        "progress": 100,
        "status": "completed",
    }


def test_progress_initialization_cross_checks_results_instead_of_stored_status(
    persisted_evaluation_service,
):
    service, engine = persisted_evaluation_service
    _add_persisted_task(
        engine,
        status=EvaluationStatus.RUNNING,
        run_token="attempt-reconcile",
        results={
            "model-one": {
                "mteb:T2Reranking": {"NDCG@10": 0.75},
                "mteb:MMarcoReranking": {"error": "failed"},
            }
        },
        model_progress={
            "model-one": {
                "mteb:T2Reranking": {"progress": 0, "status": "pending"},
                "mteb:MMarcoReranking": {
                    "progress": 100,
                    "status": "completed",
                },
            }
        },
    )

    assert service.init_model_progress(
        "legacy-resume-db",
        ["model-one"],
        ["mteb:T2Reranking", "mteb:MMarcoReranking"],
        preserve_completed=True,
        run_token="attempt-reconcile",
    )
    assert service.get_task("legacy-resume-db")["model_progress"] == {
        "model-one": {
            "mteb:T2Reranking": {"progress": 100, "status": "completed"},
            "mteb:MMarcoReranking": {"progress": 0, "status": "pending"},
        }
    }
