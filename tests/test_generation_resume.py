import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_factory.api.routes import generation_routes
from train_factory.generation.pipeline import DatasetGenerationPipeline, PipelineConfig
from train_factory.generation.steps.base import Document
from train_factory.storage.entities.generation_task_entity import GenerationStatus
from train_factory.sync.sync_worker import _split_docs_for_targets


@pytest.fixture(autouse=True)
def _isolate_resume_mechanics_from_input_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(
        generation_routes,
        "_resolve_generation_task_input",
        lambda task: task["input_path"],
    )
    lease = SimpleNamespace(release=lambda: None)

    def admit(_kind, _task_id, _user_id, operation, *args, **kwargs):
        result = operation(*args, **kwargs)
        return result, lease if result is not False and result is not None else None

    monkeypatch.setattr(
        generation_routes.background_task_admission_service,
        "admit_execution",
        admit,
    )
    monkeypatch.setattr(
        generation_routes,
        "_reject_pending_sync_tracking",
        lambda _task_id, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_routes.generation_publication_service,
        "has_staging_products",
        lambda _task_id, *, expected_run_token: False,
    )
    monkeypatch.setattr(
        generation_routes,
        "GENERATION_OUTPUT_DIR",
        str(tmp_path / "generation-output"),
    )


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_task_stages_pos_neg_resume_supported_by_output_format(tmp_path):
    output_path = tmp_path / "generated.jsonl"
    _write_jsonl(output_path, [{"query": "q", "positives": ["p"], "negatives": ["n"]}])

    base_task = {
        "generation_mode": "doc_to_training",
        "status": GenerationStatus.COMPLETED,
        "output_path": str(output_path),
        "embedding_config": {"endpoint": "http://e", "model": "m"},
    }

    universal_task = {**base_task, "output_format": "universal"}
    triplet_task = {**base_task, "output_format": "triplet"}

    universal_stage = next(
        s for s in generation_routes._build_task_stages(universal_task)
        if s["stage"] == "pos_neg_generation"
    )
    triplet_stage = next(
        s for s in generation_routes._build_task_stages(triplet_task)
        if s["stage"] == "pos_neg_generation"
    )

    assert universal_stage["resume_supported"] is True
    assert universal_stage["resume_ready"] is True
    assert triplet_stage["resume_supported"] is False
    assert triplet_stage["resume_ready"] is False


class _DummyTaskService:
    def __init__(self):
        self.update_status_calls = []
        self.begin_restart_calls = []
        self.finish_restart_calls = []
        self.reset_progress_calls = []
        self.find_completed_task_calls = 0
        self.task = None
        self.run_token = "resume-run-token-1"
        self._run_number = 1

    def get_task(self, _task_id):
        return dict(self.task) if self.task else None

    def get_task_raw(self, task_id):
        task = self.get_task(task_id)
        if task is None:
            return None
        return {**task, "run_token": self.run_token}

    def update_status(
        self,
        task_id,
        status,
        _error_message=None,
        *,
        expected_run_token=None,
        **_kwargs,
    ):
        assert expected_run_token == self.run_token
        self.update_status_calls.append((task_id, status))
        if status == GenerationStatus.PENDING:
            self._run_number += 1
            self.run_token = f"resume-run-token-{self._run_number}"
            if self.task is not None:
                for field in (
                    "output_path",
                    "output_dataset_id",
                    "filter_stats",
                    "qa_output_path",
                    "qa_dataset_id",
                    "qa_filtered_path",
                    "qa_filtered_dataset_id",
                    "deep_eval_path",
                    "deep_eval_dataset_id",
                ):
                    self.task[field] = None
        if self.task is not None:
            self.task["status"] = status
        return True

    def claim_running(self, task_id, *, expected_run_token=None):
        if expected_run_token != self.run_token:
            return None
        if self.task is None or self.task.get("status") != GenerationStatus.PENDING:
            return None
        self.task["status"] = GenerationStatus.RUNNING
        return self.run_token

    def begin_restart(
        self,
        task_id,
        *,
        expected_run_token,
        new_run_token,
        **_kwargs,
    ):
        assert expected_run_token == self.run_token
        assert self.task is not None
        assert self.task["status"] in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.STOPPED,
        }
        self.begin_restart_calls.append((task_id, expected_run_token, new_run_token))
        self.run_token = new_run_token
        self.task["status"] = "restarting"
        return new_run_token

    def finish_restart(self, task_id, *, expected_run_token, output_path):
        assert expected_run_token == self.run_token
        assert self.task is not None and self.task["status"] == "restarting"
        self.finish_restart_calls.append((task_id, expected_run_token, output_path))
        for field in (
            "output_dataset_id",
            "filter_stats",
            "qa_output_path",
            "qa_dataset_id",
            "qa_filtered_path",
            "qa_filtered_dataset_id",
            "deep_eval_path",
            "deep_eval_dataset_id",
        ):
            self.task[field] = None
        self.task["output_path"] = output_path
        self.task["status"] = GenerationStatus.PENDING
        return True

    def fail_restart(self, task_id, *, expected_run_token, error_message):
        del task_id, error_message
        if expected_run_token != self.run_token or self.task is None:
            return False
        if self.task["status"] != "restarting":
            return False
        self.task["status"] = GenerationStatus.FAILED
        return True

    def set_output(
        self,
        _task_id,
        output_path,
        output_count,
        *,
        expected_status,
        expected_run_token,
    ):
        if (
            self.task is None
            or self.task.get("status") != expected_status
            or self.run_token != expected_run_token
        ):
            return False
        self.task["output_path"] = output_path
        self.task["output_count"] = output_count
        return True

    def reset_progress(self, task_id):
        self.reset_progress_calls.append(task_id)

    def find_completed_task(self, **kwargs):
        self.find_completed_task_calls += 1
        return None


class _DummyBackgroundTasks:
    def __init__(self):
        self.calls = []

    def add_task(self, fn, *args, **kwargs):
        self.calls.append((fn, kwargs))


def test_restart_task_uses_stage_checkpoint_for_qa_to_training(tmp_path, monkeypatch):
    qa_filtered_path = (
        tmp_path / "generation-output" / "legacy" / "qa_filtered.jsonl"
    )
    _write_jsonl(qa_filtered_path, [{"query": "q", "answer": "a", "chunk_id": "c"}])

    task = {
        "task_id": "task-qa-training",
        "status": GenerationStatus.FAILED,
        "generation_mode": "qa_to_training",
        "input_path": str(tmp_path / "qa_input.jsonl"),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(tmp_path / "generated.jsonl"),
        "pos_neg_method": "retrieval",
        "embedding_config": {"endpoint": "http://emb", "model": "emb"},
        "qa_filtered_path": str(qa_filtered_path),
        "llm_config": {"endpoint": "http://llm", "model": "llm", "api_key": "k"},
        "worker_config": {"concurrency": 7, "timeout_per_doc": 123},
        "steps_config": {"pos_neg_extraction": {"enabled": True}},
        "post_process_config": {},
    }

    svc = _DummyTaskService()
    svc.task = task
    bg = _DummyBackgroundTasks()

    monkeypatch.setattr(generation_routes, "_verify_task_access", lambda task_id, current_user: task)
    monkeypatch.setattr(generation_routes, "generation_task_service", svc)

    resp = asyncio.run(generation_routes.restart_task("task-qa-training", bg, {"user_id": "u1"}))

    assert resp["status"] == "restarted"
    assert svc.update_status_calls == []
    assert len(svc.begin_restart_calls) == 1
    assert len(svc.finish_restart_calls) == 1
    assert svc.reset_progress_calls == []
    assert len(bg.calls) == 1
    _, payload = bg.calls[0]
    cfg = payload["config"]
    assert isinstance(cfg, PipelineConfig)
    assert cfg.resume_qa_filtered_path != str(qa_filtered_path)
    assert Path(cfg.resume_qa_filtered_path).read_bytes() == qa_filtered_path.read_bytes()
    assert cfg.resume_qa_path is None
    assert cfg.llm_concurrency == 7


def test_restart_task_uses_stage_checkpoint_for_doc_to_eval(tmp_path, monkeypatch):
    qa_output_path = (
        tmp_path / "generation-output" / "legacy" / "qa_extracted.jsonl"
    )
    _write_jsonl(qa_output_path, [{"query": "q", "answer": "a", "chunk_id": "c"}])

    task = {
        "task_id": "task-doc-eval",
        "status": GenerationStatus.STOPPED,
        "generation_mode": "doc_to_eval",
        "input_path": str(tmp_path / "docs.jsonl"),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(tmp_path / "deep_eval.jsonl"),
        "pos_neg_method": "retrieval",
        "embedding_config": {"endpoint": "http://emb", "model": "emb"},
        "qa_output_path": str(qa_output_path),
        "llm_config": {"endpoint": "http://llm", "model": "llm", "api_key": "k", "concurrency": 3},
        "worker_config": {"concurrency": 11, "timeout_per_doc": 90},
        "steps_config": {"qa_gen": {"enabled": True}},
        "post_process_config": {},
    }

    svc = _DummyTaskService()
    svc.task = task
    bg = _DummyBackgroundTasks()

    monkeypatch.setattr(generation_routes, "_verify_task_access", lambda task_id, current_user: task)
    monkeypatch.setattr(generation_routes, "generation_task_service", svc)

    resp = asyncio.run(generation_routes.restart_task("task-doc-eval", bg, {"user_id": "u1"}))

    assert resp["status"] == "restarted"
    assert len(bg.calls) == 1
    _, payload = bg.calls[0]
    cfg = payload["config"]
    assert isinstance(cfg, PipelineConfig)
    assert cfg.resume_qa_path != str(qa_output_path)
    assert Path(cfg.resume_qa_path).read_bytes() == qa_output_path.read_bytes()
    assert cfg.resume_qa_filtered_path is None
    assert cfg.llm_concurrency == 3


def test_restart_snapshots_old_checkpoints_before_clearing_attempt_artifacts(
    tmp_path,
    monkeypatch,
):
    old_directory = tmp_path / "generation-output" / "old"
    old_output_path = old_directory / "generated.jsonl"
    old_filtered_path = old_directory / "qa_filtered.jsonl"
    old_deep_path = old_directory / "deep_eval.jsonl"
    _write_jsonl(
        old_output_path,
        [{"query": "q", "positives": ["p"], "negatives": ["n"]}],
    )
    _write_jsonl(
        old_filtered_path,
        [{"query": "q", "answer": "a", "chunk_id": "c"}],
    )
    _write_jsonl(old_deep_path, [{"query": "q", "expected_output": "a"}])
    task = {
        "task_id": "task-attempt-artifact-reset",
        "status": GenerationStatus.FAILED,
        "generation_mode": "doc_to_training",
        "input_path": str(tmp_path / "docs.jsonl"),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(old_output_path),
        "output_dataset_id": "old-output-dataset",
        "pos_neg_method": "retrieval",
        "embedding_config": {"endpoint": "http://emb", "model": "emb"},
        "qa_output_path": str(old_directory / "qa.jsonl"),
        "qa_dataset_id": "old-qa-dataset",
        "qa_filtered_path": str(old_filtered_path),
        "qa_filtered_dataset_id": "old-filtered-dataset",
        "deep_eval_path": str(old_deep_path),
        "deep_eval_dataset_id": "old-deep-dataset",
        "filter_stats": {"kept": 1},
        "llm_config": {"endpoint": "http://llm", "model": "llm"},
        "worker_config": {"concurrency": 2, "timeout_per_doc": 60},
        "steps_config": {},
        "post_process_config": {},
    }
    service = _DummyTaskService()
    service.task = task
    background_tasks = _DummyBackgroundTasks()
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda _task_id, _current_user: task,
    )
    monkeypatch.setattr(generation_routes, "generation_task_service", service)

    response = asyncio.run(
        generation_routes.restart_task(
            task["task_id"],
            background_tasks,
            {"user_id": "u1"},
        )
    )

    assert response["status"] == "restarted"
    _, payload = background_tasks.calls[0]
    config = payload["config"]
    assert config.resume_output_path == config.output_path
    assert config.resume_output_path != str(old_output_path)
    assert config.resume_qa_filtered_path != str(old_filtered_path)
    assert Path(config.resume_output_path).read_bytes() == old_output_path.read_bytes()
    assert (
        Path(config.resume_qa_filtered_path).read_bytes()
        == old_filtered_path.read_bytes()
    )
    assert config.run_token == service.run_token
    assert service.task["output_path"] == config.output_path
    assert service.task["output_path"] != str(old_output_path)
    assert service.task["output_dataset_id"] is None
    assert service.task["qa_output_path"] is None
    assert service.task["qa_dataset_id"] is None
    assert service.task["qa_filtered_path"] is None
    assert service.task["qa_filtered_dataset_id"] is None
    assert service.task["deep_eval_path"] is None
    assert service.task["deep_eval_dataset_id"] is None
    assert service.task["filter_stats"] is None


def test_restart_clones_checkpoints_into_new_attempt_before_scheduling(
    tmp_path,
    monkeypatch,
):
    old_output_path = tmp_path / "old-attempt" / "generated.jsonl"
    old_qa_path = tmp_path / "old-attempt" / "qa.jsonl"
    old_filtered_path = tmp_path / "old-attempt" / "qa_filtered.jsonl"
    output_rows = [{"query": "q", "positives": ["p"], "negatives": ["n"]}]
    qa_rows = [{"query": "q", "answer": "a", "chunk_id": "c"}]
    _write_jsonl(old_output_path, output_rows)
    _write_jsonl(old_qa_path, qa_rows)
    _write_jsonl(old_filtered_path, qa_rows)
    task = {
        "task_id": "task-clone-restart-checkpoints",
        "status": GenerationStatus.FAILED,
        "generation_mode": "doc_to_training",
        "input_path": str(tmp_path / "docs.jsonl"),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(old_output_path),
        "output_dataset_id": "old-output-dataset",
        "pos_neg_method": "retrieval",
        "embedding_config": {"endpoint": "http://emb", "model": "emb"},
        "qa_output_path": str(old_qa_path),
        "qa_dataset_id": "old-qa-dataset",
        "qa_filtered_path": str(old_filtered_path),
        "qa_filtered_dataset_id": "old-filtered-dataset",
        "deep_eval_path": str(tmp_path / "old-attempt" / "deep.jsonl"),
        "deep_eval_dataset_id": "old-deep-dataset",
        "filter_stats": {"kept": 1},
        "llm_config": {"endpoint": "http://llm", "model": "llm"},
        "worker_config": {"concurrency": 2, "timeout_per_doc": 60},
        "steps_config": {},
        "post_process_config": {},
    }
    service = _DummyTaskService()
    service.task = task
    background_tasks = _DummyBackgroundTasks()
    monkeypatch.setattr(generation_routes, "GENERATION_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        generation_routes,
        "_verify_task_access",
        lambda _task_id, _current_user: task,
    )
    monkeypatch.setattr(generation_routes, "generation_task_service", service)

    response = asyncio.run(
        generation_routes.restart_task(
            task["task_id"],
            background_tasks,
            {"user_id": "u1"},
        )
    )

    assert response["status"] == "restarted"
    assert len(service.begin_restart_calls) == 1
    assert len(service.finish_restart_calls) == 1
    _, payload = background_tasks.calls[0]
    config = payload["config"]
    attempt_key = generation_routes.generation_attempt_key(
        task["task_id"],
        config.run_token,
    )
    expected_dir = tmp_path / f"generation_{task['task_id']}" / attempt_key
    assert Path(config.output_path).parent == expected_dir
    assert config.resume_output_path == config.output_path
    assert Path(config.resume_qa_path).parent == expected_dir
    assert Path(config.resume_qa_filtered_path).parent == expected_dir
    assert json.loads(Path(config.resume_output_path).read_text(encoding="utf-8"))
    assert json.loads(Path(config.resume_qa_path).read_text(encoding="utf-8"))
    assert json.loads(
        Path(config.resume_qa_filtered_path).read_text(encoding="utf-8")
    )
    assert old_output_path.read_text(encoding="utf-8") == (
        json.dumps(output_rows[0], ensure_ascii=False) + "\n"
    )
    assert old_qa_path.read_text(encoding="utf-8") == (
        json.dumps(qa_rows[0], ensure_ascii=False) + "\n"
    )


def test_restart_task_force_mode_skips_all_checkpoints_and_dedup(tmp_path, monkeypatch):
    qa_output_path = tmp_path / "qa_extracted.jsonl"
    qa_filtered_path = tmp_path / "qa_filtered.jsonl"
    _write_jsonl(qa_output_path, [{"query": "q", "answer": "a", "chunk_id": "c"}])
    _write_jsonl(qa_filtered_path, [{"query": "q", "answer": "a", "chunk_id": "c"}])

    task = {
        "task_id": "task-force-restart",
        "status": GenerationStatus.COMPLETED,
        "generation_mode": "doc_to_training",
        "input_path": str(tmp_path / "docs.jsonl"),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(tmp_path / "generated.jsonl"),
        "pos_neg_method": "retrieval",
        "embedding_config": {"endpoint": "http://emb", "model": "emb"},
        "embedding_config_id": "emb-cfg-1",
        "source_dataset_id": "dataset-1",
        "qa_output_path": str(qa_output_path),
        "qa_filtered_path": str(qa_filtered_path),
        "llm_config": {"endpoint": "http://llm", "model": "llm", "api_key": "k"},
        "worker_config": {"concurrency": 5, "timeout_per_doc": 120},
        "steps_config": {"qa_gen": {"enabled": True}},
        "post_process_config": {"dedup": {"enabled": True}},
        "similarity_threshold": 0.9,
        "user_id": "u1",
    }

    svc = _DummyTaskService()
    svc.task = task
    bg = _DummyBackgroundTasks()

    monkeypatch.setattr(generation_routes, "_verify_task_access", lambda task_id, current_user: task)
    monkeypatch.setattr(generation_routes, "generation_task_service", svc)

    resp = asyncio.run(
        generation_routes.restart_task(
            "task-force-restart",
            bg,
            {"user_id": "u1"},
            force=True,
        )
    )

    assert resp["status"] == "restarted"
    assert len(bg.calls) == 1
    _, payload = bg.calls[0]
    cfg = payload["config"]
    assert isinstance(cfg, PipelineConfig)
    assert cfg.resume_qa_path is None
    assert cfg.resume_qa_filtered_path is None
    assert cfg.skip_filter is False
    assert svc.find_completed_task_calls == 0


def test_restart_task_force_skips_checkpoints_for_doc_to_eval(tmp_path, monkeypatch):
    """force=True 在 doc_to_eval 模式下也不使用 checkpoint。"""
    qa_output_path = tmp_path / "qa_extracted.jsonl"
    _write_jsonl(qa_output_path, [{"query": "q", "answer": "a", "chunk_id": "c"}])

    task = {
        "task_id": "task-eval-force",
        "status": GenerationStatus.STOPPED,
        "generation_mode": "doc_to_eval",
        "input_path": str(tmp_path / "docs.jsonl"),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(tmp_path / "deep_eval.jsonl"),
        "pos_neg_method": "retrieval",
        "embedding_config": {"endpoint": "http://emb", "model": "emb"},
        "qa_output_path": str(qa_output_path),
        "llm_config": {"endpoint": "http://llm", "model": "llm", "api_key": "k", "concurrency": 3},
        "worker_config": {"concurrency": 11, "timeout_per_doc": 90},
        "steps_config": {"qa_gen": {"enabled": True}},
        "post_process_config": {},
    }

    svc = _DummyTaskService()
    svc.task = task
    bg = _DummyBackgroundTasks()

    monkeypatch.setattr(generation_routes, "_verify_task_access", lambda task_id, current_user: task)
    monkeypatch.setattr(generation_routes, "generation_task_service", svc)

    resp = asyncio.run(
        generation_routes.restart_task("task-eval-force", bg, {"user_id": "u1"}, force=True)
    )

    assert resp["status"] == "restarted"
    _, payload = bg.calls[0]
    cfg = payload["config"]
    assert cfg.resume_qa_path is None
    assert cfg.resume_qa_filtered_path is None


def test_restart_task_running_status_rejected(tmp_path, monkeypatch):
    """正在运行（running）的任务不允许重启，无论 force 值。"""
    task = {
        "task_id": "task-running",
        "status": GenerationStatus.RUNNING,
        "generation_mode": "doc_to_training",
        "input_path": str(tmp_path / "docs.jsonl"),
        "input_format": "jsonl",
        "output_format": "universal",
        "output_path": str(tmp_path / "generated.jsonl"),
        "pos_neg_method": "retrieval",
        "embedding_config": {"endpoint": "http://emb", "model": "emb"},
        "llm_config": {"endpoint": "http://llm", "model": "llm", "api_key": "k"},
        "worker_config": {"concurrency": 5, "timeout_per_doc": 60},
        "steps_config": {},
        "post_process_config": {},
    }

    svc = _DummyTaskService()
    svc.task = task
    bg = _DummyBackgroundTasks()

    monkeypatch.setattr(generation_routes, "_verify_task_access", lambda task_id, current_user: task)
    monkeypatch.setattr(generation_routes, "generation_task_service", svc)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            generation_routes.restart_task("task-running", bg, {"user_id": "u1"})
        )
    assert "只能重启" in str(exc_info.value.detail)
    assert len(svc.update_status_calls) == 0


class _DummyEmbeddingClient:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3]]


class _DummyMilvusClient:
    def __init__(self):
        self.calls = []

    def search_similar(self, collection_name, query_emb, top_k=10, exclude_chunk_ids=None):
        self.calls.append({
            "collection_name": collection_name,
            "top_k": top_k,
            "exclude_chunk_ids": exclude_chunk_ids or [],
        })
        return [[
            {"chunk_content": "ctx-1"},
            {"chunk_content": "ctx-2"},
        ]]


def test_generate_eval_data_batch_resume_skips_completed_queries(tmp_path):
    output_path = tmp_path / "generated.jsonl"
    cfg = PipelineConfig(
        input_path=str(tmp_path / "input.jsonl"),
        input_format="jsonl",
        output_path=str(output_path),
        generation_mode="doc_to_eval",
        task_id="abc12345-task",
        embedding_config={"retrieval_top_k": 5},
        steps={},
    )
    pipeline = DatasetGenerationPipeline(cfg)

    # 预写 checkpoint：1 条命中当前任务 query，1 条无关键（应被忽略）
    deep_eval_path = pipeline._get_deep_eval_path()
    _write_jsonl(deep_eval_path, [
        {
            "query": "q1",
            "expected_output": "a1",
            "retrieval_context": ["old-ctx"],
            "source_chunk_id": "c1",
            "source_chunk_content": "src-1",
        },
        {
            "query": "other-q",
            "expected_output": "x",
            "retrieval_context": ["x"],
            "source_chunk_id": "other-c",
            "source_chunk_content": "x",
        },
    ])

    records = [
        {"query": "q1", "answer": "a1", "chunk_id": "c1", "chunk_content": "src-1"},
        {"query": "q2", "answer": "a2", "chunk_id": "c2", "chunk_content": "src-2"},
    ]

    emb = _DummyEmbeddingClient()
    milvus = _DummyMilvusClient()
    results = asyncio.run(pipeline._generate_eval_data_batch(records, emb, milvus, "col_a"))

    # q1 从 checkpoint 恢复，不应重复检索；q2 需要执行一次检索
    assert len(emb.calls) == 1
    assert len(milvus.calls) == 1
    assert milvus.calls[0]["exclude_chunk_ids"] == ["c2"]

    # 返回结果应仅包含：恢复的 q1 + 新生成的 q2
    assert len(results) == 2
    assert any(r.get("query") == "q1" and r.get("source_chunk_id") == "c1" for r in results)
    assert any(r.get("query") == "q2" and r.get("source_chunk_id") == "c2" for r in results)
    assert not any(r.get("query") == "other-q" for r in results)


def _build_sync_docs(n: int):
    docs = []
    for i in range(n):
        docs.append(
            Document(
                content=f"content-{i}",
                doc_id=f"doc-{i}",
                metadata={
                    "chunk_id": f"chunk-{i}",
                    "external_id": f"ext-{i}",
                },
            )
        )
    return docs


def test_split_docs_for_targets_replicate_mode():
    config = {"generation_config": {"adapter_request_balancing": False}}
    targets = [
        {"type": "base", "collection_name": "col_base", "label": "base"},
        {"type": "adapter", "collection_name": "col_a1", "label": "a1", "adapter_id": "a1"},
        {"type": "adapter", "collection_name": "col_a2", "label": "a2", "adapter_id": "a2"},
    ]
    docs = _build_sync_docs(10)

    result = _split_docs_for_targets(config, docs, targets)

    assert len(result["col_base"]) == 10
    assert len(result["col_a1"]) == 10
    assert len(result["col_a2"]) == 10


def test_split_docs_for_targets_balancing_mode():
    config = {
        "generation_config": {
            "adapter_request_balancing": True,
            "adapter_request_weights": {"a1": 1.0, "a2": 1.0},
        }
    }
    targets = [
        {"type": "base", "collection_name": "col_base", "label": "base"},
        {"type": "adapter", "collection_name": "col_a1", "label": "a1", "adapter_id": "a1"},
        {"type": "adapter", "collection_name": "col_a2", "label": "a2", "adapter_id": "a2"},
    ]
    docs = _build_sync_docs(100)

    result = _split_docs_for_targets(config, docs, targets)

    assert len(result["col_base"]) == 100
    a1 = len(result["col_a1"])
    a2 = len(result["col_a2"])
    assert a1 + a2 == 100
    assert abs(a1 - a2) <= 30
