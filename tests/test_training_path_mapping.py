import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi import BackgroundTasks

from train_factory.api.routes import training_routes
from train_factory.sync import level2_handler


def _mock_owned_training_resources(monkeypatch):
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "get_model_by_path",
        lambda path, user_id=None: {
            "model_id": "model-owned",
            "model_path": path,
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        training_routes.dataset_service,
        "get_dataset_by_storage_path",
        lambda path, user_id=None: {
            "dataset_id": "dataset-owned",
            "storage_path": path,
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        training_routes,
        "require_managed_model_provenance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        training_routes,
        "require_managed_dataset_provenance",
        lambda *args, **kwargs: None,
    )


def test_normalize_base_model_path_maps_host_path_to_container(monkeypatch):
    captured = {}

    def fake_map_storage_path(path: str):
        assert path == "/workspace/train-factory/models/e16984f4"
        return "/app/models/e16984f4", path

    def fake_validate_storage_path(path: str, resource_type: str = "resource"):
        captured["path"] = path
        captured["resource_type"] = resource_type
        return path

    monkeypatch.setattr(training_routes, "map_storage_path", fake_map_storage_path)
    monkeypatch.setattr(training_routes, "validate_storage_path", fake_validate_storage_path)

    normalized = training_routes._normalize_base_model_path(
        "/workspace/train-factory/models/e16984f4",
        user_id=None,
        model_type="llm",
    )

    assert normalized == "/app/models/e16984f4"
    assert captured == {
        "path": "/app/models/e16984f4",
        "resource_type": "base model",
    }


def test_normalize_base_model_path_keeps_repo_id_unchanged():
    normalized = training_routes._normalize_base_model_path(
        "Qwen/Qwen3-0.6B",
        user_id=None,
        model_type="llm",
    )

    assert normalized == "Qwen/Qwen3-0.6B"


def test_normalize_storage_path_rejects_unmapped_absolute_path():
    try:
        training_routes._normalize_storage_path("/tmp/evil", "dataset")
    except Exception as exc:
        assert exc.status_code == 400
        assert "Invalid dataset path" in exc.detail
    else:
        raise AssertionError("expected invalid dataset path to be rejected")


def test_normalize_storage_path_rejects_relative_local_path_outside_allowed_dirs():
    try:
        training_routes._normalize_storage_path("../../../../tmp/evil.jsonl", "dataset")
    except Exception as exc:
        assert exc.status_code == 400
        assert "Invalid dataset path" in exc.detail
    else:
        raise AssertionError("expected relative local path to be rejected")


def test_normalize_base_model_path_resolves_legacy_models_alias(monkeypatch):
    def fake_search_models(query: str, model_type: str | None = None, user_id=None, limit: int = 100):
        assert query == "Qwen3-0.6B"
        assert model_type == "llm"
        return [{
            "model_name": "Qwen3-0.6B",
            "model_path": "/app/models/e16984f4-72a9-466e-924e-0684f1336013",
        }]

    monkeypatch.setattr(training_routes.model_registry_service, "search_models", fake_search_models)
    monkeypatch.setattr(
        training_routes,
        "map_storage_path",
        lambda path: (path, None),
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )

    normalized = training_routes._normalize_base_model_path(
        "/models/Qwen3-0.6B",
        user_id=None,
        model_type="llm",
    )

    assert normalized == "/app/models/e16984f4-72a9-466e-924e-0684f1336013"


def test_normalize_training_config_paths_updates_datasets_and_train_path(monkeypatch):
    mappings = {
        "/workspace/train-factory/models/e16984f4": "/app/models/e16984f4",
        "/workspace/train-factory/data/dpo.jsonl": "/app/data/dpo.jsonl",
    }

    monkeypatch.setattr(
        training_routes,
        "map_storage_path",
        lambda path: (mappings.get(path, path), path if path in mappings else None),
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )

    normalized = training_routes._normalize_training_config_paths(
        {
            "base_model_path": "/workspace/train-factory/models/e16984f4",
            "model_type": "llm",
            "train_dataset_path": "/workspace/train-factory/data/dpo.jsonl",
            "dataset_configs": [
                {"path": "/workspace/train-factory/data/dpo.jsonl", "split": "train"},
                {"path": "hf-user/dpo-dataset", "split": "eval"},
            ],
            "datasets": [
                {"path": "/workspace/train-factory/data/dpo.jsonl", "split": "train"},
                {"path": "hf-user/dpo-dataset", "split": "eval"},
            ],
        },
        user_id=None,
    )

    assert normalized["base_model_path"] == "/app/models/e16984f4"
    assert normalized["train_dataset_path"] == "/app/data/dpo.jsonl"
    assert normalized["dataset_configs"] == [
        {"path": "/app/data/dpo.jsonl", "split": "train"},
        {"path": "hf-user/dpo-dataset", "split": "eval"},
    ]
    assert normalized["datasets"] == [
        {"path": "/app/data/dpo.jsonl", "split": "train"},
        {"path": "hf-user/dpo-dataset", "split": "eval"},
    ]


def test_normalize_training_config_paths_updates_checkpoint_and_direct_paths(monkeypatch):
    mappings = {
        "/workspace/train-factory/data/dpo.jsonl": "/app/data/dpo.jsonl",
        "/workspace/train-factory/output/task/checkpoint-1": "/app/output/task/checkpoint-1",
        "/workspace/train-factory/output/task": "/app/output/task",
    }

    monkeypatch.setattr(
        training_routes,
        "map_storage_path",
        lambda path: (mappings.get(path, path), path if path in mappings else None),
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )

    normalized = training_routes._normalize_training_config_paths(
        {
            "model_type": "decoder_reranker",
            "train_dataset_path": "/workspace/train-factory/data/dpo.jsonl",
            "sft_checkpoint_path": "/workspace/train-factory/output/task/checkpoint-1",
            "resume_from_checkpoint": "/workspace/train-factory/output/task/checkpoint-1",
            "output_dir": "/workspace/train-factory/output/task",
        },
        user_id=None,
    )

    assert normalized["train_dataset_path"] == "/app/data/dpo.jsonl"
    assert normalized["sft_checkpoint_path"] == "/app/output/task/checkpoint-1"
    assert normalized["resume_from_checkpoint"] == "/app/output/task/checkpoint-1"
    assert normalized["output_dir"] == "/app/output/task"


def test_normalize_training_config_paths_canonicalizes_relative_dataset_entries(
    monkeypatch,
):
    relative_path = "./data/dpo.jsonl"
    mapped_path = "/app/data/dpo.jsonl"
    monkeypatch.setattr(
        training_routes,
        "map_storage_path",
        lambda path: (mapped_path, path) if path == relative_path else (path, None),
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )

    normalized = training_routes._normalize_training_config_paths(
        {
            "train_dataset_path": relative_path,
            "dataset_configs": [{"path": relative_path, "split": "train"}],
            "datasets": [{"path": relative_path, "split": "train"}],
        },
        user_id="user-1",
    )

    assert normalized["train_dataset_path"] == mapped_path
    assert normalized["dataset_configs"][0]["path"] == mapped_path
    assert normalized["datasets"][0]["path"] == mapped_path


def test_create_training_task_normalizes_host_paths_before_persist(monkeypatch):
    mappings = {
        "/workspace/train-factory/models/e16984f4": "/app/models/e16984f4",
        "/workspace/train-factory/data/dpo.jsonl": "/app/data/dpo.jsonl",
    }
    captured = {}
    admissions = []
    lease = SimpleNamespace(release=lambda: None)

    def admit(kind, task_id, admission_user_id, operation, *args, **kwargs):
        admissions.append((kind, task_id, admission_user_id))
        return operation(*args, **kwargs), lease

    def run_admitted(*_args, **_kwargs):
        raise AssertionError("background task should not execute in this unit test")

    admission_service = SimpleNamespace(
        admit_execution=admit,
        run_sync=run_admitted,
    )
    monkeypatch.setattr(
        training_routes,
        "background_task_admission_service",
        admission_service,
        raising=False,
    )

    monkeypatch.setattr(training_routes, "check_idempotency", lambda *args, **kwargs: (False, None))
    monkeypatch.setattr(training_routes, "store_idempotency_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        training_routes,
        "map_storage_path",
        lambda path: (mappings.get(path, path), path if path in mappings else None),
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )
    _mock_owned_training_resources(monkeypatch)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda task_id, output_dir, training_params: captured.update(
            server_output_dir=output_dir,
            updated_training_params=training_params,
        ) or True,
    )

    def fake_create_task(**kwargs):
        captured.update(kwargs)
        captured["create_output_dir"] = kwargs.get("output_dir")
        return {
            "task_id": "task-host-path",
            "task_name": kwargs.get("task_name"),
        }

    monkeypatch.setattr(training_routes.training_task_service, "create_task", fake_create_task)

    request = training_routes.TrainingRequest(
        model_type="llm",
        training_method="dpo",
        task_name="route-host-path",
        base_model_path="/workspace/train-factory/models/e16984f4",
        output_dir="/app/output/task-host-path",
        datasets=[
            {"path": "/workspace/train-factory/data/dpo.jsonl", "split": "train"},
        ],
    )
    background_tasks = BackgroundTasks()

    response = asyncio.run(
        training_routes.create_training_task(
            request=request,
            background_tasks=background_tasks,
            current_user={"user_id": "user-1"},
            idempotency_key=None,
        )
    )

    assert response.task_id == "task-host-path"
    assert captured["model_path"] == "/app/models/e16984f4"
    assert captured["train_dataset_path"] == "/app/data/dpo.jsonl"
    assert captured["training_params"]["base_model_path"] == "/app/models/e16984f4"
    assert captured["training_params"]["dataset_configs"] == [
        {"path": "/app/data/dpo.jsonl", "max_samples": None, "split": "train"},
    ]
    assert captured["training_params"]["train_dataset_path"] == "/app/data/dpo.jsonl"
    assert captured["create_output_dir"] is None
    assert Path(captured["server_output_dir"]).name == "task-host-path"
    assert captured["training_params"]["output_dir"] == captured["server_output_dir"]
    assert admissions == [("training", None, "user-1")]
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is run_admitted
    assert background_tasks.tasks[0].args == (
        lease,
        training_routes.run_training_task,
        "task-host-path",
        captured["training_params"],
    )


def test_create_training_task_resolves_legacy_model_alias_before_persist(monkeypatch):
    captured = {}
    lease = SimpleNamespace(release=lambda: None)

    def admit(kind, task_id, admission_user_id, operation, *args, **kwargs):
        assert (kind, task_id, admission_user_id) == ("training", None, "user-1")
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(training_routes, "check_idempotency", lambda *args, **kwargs: (False, None))
    monkeypatch.setattr(training_routes, "store_idempotency_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        training_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        training_routes.model_registry_service,
        "search_models",
        lambda query, model_type=None, user_id=None, limit=100: [
            {
                "model_name": "Qwen3-0.6B",
                "model_path": "/app/models/e16984f4-72a9-466e-924e-0684f1336013",
            }
        ] if query == "Qwen3-0.6B" else [],
    )
    monkeypatch.setattr(
        training_routes,
        "map_storage_path",
        lambda path: (path, None),
    )
    monkeypatch.setattr(
        training_routes,
        "validate_storage_path",
        lambda path, resource_type="resource": path,
    )
    _mock_owned_training_resources(monkeypatch)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_output_dir",
        lambda task_id, output_dir, training_params: captured.update(
            server_output_dir=output_dir,
            updated_training_params=training_params,
        ) or True,
    )

    def fake_create_task(**kwargs):
        captured.update(kwargs)
        return {
            "task_id": "task-legacy-alias",
            "task_name": kwargs.get("task_name"),
        }

    monkeypatch.setattr(training_routes.training_task_service, "create_task", fake_create_task)

    request = training_routes.TrainingRequest(
        model_type="llm",
        training_method="dpo",
        task_name="route-legacy-alias",
        base_model_path="/models/Qwen3-0.6B",
        output_dir="/app/output/task-legacy-alias",
        datasets=[
            {"path": "/app/data/dpo.jsonl", "split": "train"},
        ],
    )
    background_tasks = BackgroundTasks()

    response = asyncio.run(
        training_routes.create_training_task(
            request=request,
            background_tasks=background_tasks,
            current_user={"user_id": "user-1"},
            idempotency_key=None,
        )
    )

    assert response.task_id == "task-legacy-alias"
    assert captured["model_path"] == "/app/models/e16984f4-72a9-466e-924e-0684f1336013"
    assert captured["training_params"]["base_model_path"] == "/app/models/e16984f4-72a9-466e-924e-0684f1336013"
    assert Path(captured["server_output_dir"]).name == "task-legacy-alias"
    assert captured["training_params"]["output_dir"] == captured["server_output_dir"]
    assert len(background_tasks.tasks) == 1


def test_run_training_task_persists_normalized_execution_config(monkeypatch):
    import train_factory.sync.post_training_handler as post_training_handler

    mappings = {
        "/workspace/train-factory/models/e16984f4": "/app/models/e16984f4",
        "/workspace/train-factory/data/dpo.jsonl": "/app/data/dpo.jsonl",
        "/workspace/train-factory/output/task/checkpoint-1": "/app/output/task/checkpoint-1",
        "/workspace/train-factory/output/task": "/app/output/task",
    }
    captured = {}
    trained_configs = []
    task_state = {"status": "pending", "run_token": None}

    monkeypatch.setattr(
        training_routes,
        "get_settings",
        lambda: SimpleNamespace(
            auth_enabled=True,
            datasets_dir=Path("/app/data"),
            minio_bucket="trainfactory",
            models_dir=Path("/app/models"),
            output_dir=Path("/app/output"),
            get_task_output_dir=lambda task_id: Path("/app/output") / task_id,
        ),
    )
    monkeypatch.setattr(training_routes, "map_storage_path", lambda path: (mappings.get(path, path), path))
    monkeypatch.setattr(training_routes, "validate_storage_path", lambda path, resource_type="resource": path)
    _mock_owned_training_resources(monkeypatch)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "claim_preparing",
        lambda _task_id, run_token: task_state.update(
            status="preparing",
            run_token=run_token,
        )
        or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_status",
        lambda _task_id, status, *_args, **_kwargs: task_state.update(status=status)
        or True,
    )
    monkeypatch.setattr(training_routes.training_task_service, "update_task_progress", lambda *args, **kwargs: True)
    monkeypatch.setattr(training_routes.training_task_service, "update_process_info", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        training_routes.training_task_service,
        "update_task_execution_config",
        lambda task_id, **kwargs: captured.update({"task_id": task_id, **kwargs}) or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "complete_task",
        lambda *args, **kwargs: task_state.update(status="succeeded") or True,
    )
    monkeypatch.setattr(
        training_routes.training_task_service,
        "get_task",
        lambda task_id: {
            "task_name": "normalized-task",
            "user_id": "user-1",
            "status": task_state["status"],
            "run_token": task_state["run_token"],
        },
    )
    monkeypatch.setattr(training_routes.model_registry_service, "register_from_task", lambda **kwargs: None)
    monkeypatch.setattr(post_training_handler, "on_training_completed", lambda **kwargs: None)
    monkeypatch.setattr(training_routes.gpu_resource_manager, "allocate_gpus_for_task", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(training_routes.gpu_resource_manager, "release_gpus_for_task", lambda *args, **kwargs: True)

    def fake_train_with_config(config, progress_callback=None):
        trained_configs.append(dict(config))
        return SimpleNamespace(
            save_dir="/app/output/task/final",
            final_metrics={},
        )

    monkeypatch.setattr(training_routes, "train_with_config", fake_train_with_config)

    class InlineProcess:
        def __init__(self, target, args, name):
            self.target = target
            self.args = args
            self.name = name
            self.pid = 1234
            self.exitcode = None
            self._popen = None
            self._alive = False

        def start(self):
            self._popen = object()
            self._alive = True
            try:
                self.target(*self.args)
                self.exitcode = 0
            finally:
                self._alive = False

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

    class InlineContext:
        Process = InlineProcess

    monkeypatch.setattr(
        training_routes.multiprocessing,
        "get_context",
        lambda method: InlineContext(),
    )

    training_routes.run_training_task(
        "task-run",
        {
            "_run_token": "path-mapping-run",
            "user_id": "user-1",
            "model_type": "decoder_reranker",
            "training_method": "dpo",
            "base_model_path": "/workspace/train-factory/models/e16984f4",
            "train_dataset_path": "/workspace/train-factory/data/dpo.jsonl",
            "dataset_configs": [
                {"path": "/workspace/train-factory/data/dpo.jsonl", "split": "train"},
            ],
            "parent_task_id": "task",
            "sft_checkpoint_path": "/workspace/train-factory/output/task/checkpoint-1",
            "output_dir": "/workspace/train-factory/output/task",
        },
    )

    assert captured["task_id"] == "task-run"
    assert captured["model_path"] == "/app/models/e16984f4"
    assert captured["train_dataset_path"] == "/app/data/dpo.jsonl"
    assert captured["sft_checkpoint_path"] == "/app/output/task/checkpoint-1"
    assert captured["output_dir"] == "/app/output/task"
    assert captured["training_params"]["base_model_path"] == "/app/models/e16984f4"
    assert captured["training_params"]["train_dataset_path"] == "/app/data/dpo.jsonl"
    assert captured["training_params"]["sft_checkpoint_path"] == "/app/output/task/checkpoint-1"
    assert trained_configs[0]["base_model_path"] == "/app/models/e16984f4"
    assert trained_configs[0]["train_dataset_path"] == "/app/data/dpo.jsonl"
    assert trained_configs[0]["sft_checkpoint_path"] == "/app/output/task/checkpoint-1"


def test_build_sync_target_output_dir_uses_configured_output_root():
    output_dir = level2_handler._build_sync_target_output_dir(
        "sync-config",
        "target-1",
        2,
        output_root=Path("/app/output"),
    )

    assert output_dir == "/app/output/sync/sync-config/target-1/round_2"


def test_trigger_training_for_target_persists_allowed_output_dir(monkeypatch):
    import importlib
    import threading

    settings_module = importlib.import_module("train_factory.config.settings")
    from train_factory.storage.services.dataset_service import dataset_service
    from train_factory.storage.services.external_sync_service import external_sync_service
    from train_factory.storage.services.training_task_service import training_task_service

    captured = {}
    lease = SimpleNamespace(release=lambda: None)

    def admit(kind, admitted_task_id, admission_user_id, operation, *args, **kwargs):
        assert kind == "training"
        assert admission_user_id == "user-1"
        assert admitted_task_id == kwargs["task_id"]
        return operation(*args, **kwargs), lease

    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: SimpleNamespace(output_dir=Path("/app/output")),
    )
    monkeypatch.setattr(
        level2_handler.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(training_routes, "map_storage_path", lambda path: (path, None))
    monkeypatch.setattr(training_routes, "validate_storage_path", lambda path, resource_type="resource": path)
    monkeypatch.setattr(
        external_sync_service,
        "get_all_completed_generation_datasets",
        lambda config_id: [{"dataset_id": "dataset-1", "sample_count": 12}],
    )
    monkeypatch.setattr(
        external_sync_service,
        "claim_training_target",
        lambda config_id, target_id=None, require_threshold=True: {
            "target_id": target_id,
            "target_name": "decoder-target",
            "model_type": "decoder_reranker",
            "training_method": "dpo",
            "training_config": {
                "rl_config": {"beta": 0.2},
                "parent_task_id": "sft",
                "sft_checkpoint_path": "/app/output/sft/checkpoint-1",
            },
            "base_model_path": "/app/models/base",
            "data_phase": "final",
            "total_trainings": 1,
            "_claim_previous_status": "ready",
            "_claim_previous_training_id": None,
            "_claim_pending_samples": 12,
        },
    )
    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda dataset_id: {
            "storage_path": None,
            "storage_uri": "s3://trainfactory/datasets/dpo.jsonl",
        },
    )
    monkeypatch.setattr(
        external_sync_service,
        "get_task_raw",
        lambda config_id: {"task_id": config_id, "user_id": "user-1", "status": "idle"},
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_training_target",
        lambda target_id, **kwargs: captured.setdefault("target_updates", []).append((target_id, kwargs)),
    )
    monkeypatch.setattr(
        external_sync_service,
        "update_task",
        lambda config_id, **kwargs: captured.setdefault("task_updates", []).append((config_id, kwargs)),
    )
    monkeypatch.setattr(
        external_sync_service,
        "create_training_with_claim",
        lambda **kwargs: captured.update({"sync_training": kwargs}) or True,
    )
    monkeypatch.setattr(
        external_sync_service,
        "reset_target_pending_samples",
        lambda target_id: captured.update({"reset_target": target_id}),
    )
    monkeypatch.setattr(
        training_task_service,
        "create_task",
        lambda **kwargs: captured.update({"create_task": kwargs})
        or {"task_id": kwargs["task_id"]},
    )
    monkeypatch.setattr(
        training_task_service,
        "update_task_output_dir",
        lambda task_id, output_dir, training_params: captured.update({
            "updated_task_id": task_id,
            "output_dir": output_dir,
            "updated_training_params": training_params,
        }),
    )
    monkeypatch.setattr(
        level2_handler,
        "_validate_sync_training_parent_checkpoint",
        lambda parent_task_id, checkpoint_path, sync_user_id: captured.update(
            {
                "parent_validation": (
                    parent_task_id,
                    checkpoint_path,
                    sync_user_id,
                )
            }
        ),
    )

    class FakeThread:
        def __init__(self, target, args, daemon):
            captured["thread"] = {
                "target": target,
                "args": args,
                "daemon": daemon,
            }

        def start(self):
            captured["thread_started"] = True

    monkeypatch.setattr(threading, "Thread", FakeThread)

    level2_handler._trigger_training_for_target(
        "sync-config",
        {
            "target_id": "target-1",
            "target_name": "decoder-target",
            "model_type": "decoder_reranker",
            "training_method": "dpo",
            "training_config": {
                "rl_config": {"beta": 0.2},
                "parent_task_id": "sft",
                "sft_checkpoint_path": "/app/output/sft/checkpoint-1",
            },
            "base_model_path": "/app/models/base",
            "data_phase": "final",
            "total_trainings": 1,
        },
    )

    assert captured["create_task"]["model_path"] == "/app/models/base"
    assert captured["create_task"]["train_dataset_path"] == "s3://trainfactory/datasets/dpo.jsonl"
    assert captured["updated_training_params"]["dataset_configs"][0]["path"] == (
        "s3://trainfactory/datasets/dpo.jsonl"
    )
    assert captured["create_task"]["rl_config"] == {"beta": 0.2}
    assert captured["create_task"]["parent_task_id"] == "sft"
    assert captured["create_task"]["sft_checkpoint_path"] == "/app/output/sft/checkpoint-1"
    assert captured["parent_validation"] == (
        "sft",
        "/app/output/sft/checkpoint-1",
        "user-1",
    )
    assert captured["output_dir"] == "/app/output/sync/sync-config/target-1/round_2"
    assert captured["updated_training_params"]["output_dir"] == captured["output_dir"]
    created_task_id = captured["create_task"]["task_id"]
    assert captured["updated_training_params"]["task_id"] == created_task_id
    assert captured["thread"]["target"] is level2_handler.background_task_admission_service.run_sync
    assert captured["thread"]["args"] == (
        lease,
        training_routes.run_training_task,
        created_task_id,
        captured["updated_training_params"],
    )
    assert captured["thread_started"] is True
    assert "reset_target" not in captured
