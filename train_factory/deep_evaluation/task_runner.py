"""Deep evaluation task runner."""

import asyncio
import csv
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..evaluation.dataset_access import (
    resolve_local_evaluation_dataset_record,
    resolve_managed_local_evaluation_dataset,
    task_requires_dataset_provenance,
)
from ..storage.services.deep_evaluation_task_service import deep_evaluation_task_service
from ..storage.entities.evaluation_task_entity import EvaluationStatus
from .evaluator import DeepEvaluator
from .metrics import EvaluationSample

DeepEvaluationStatus = EvaluationStatus

logger = logging.getLogger(__name__)

# Track cancelled tasks (thread-safe)
_cancelled_tasks: set = set()
_cancelled_lock = threading.Lock()
# Cache DB-based cancellation checks with TTL (task_id -> last_check_time)
_cancel_check_cache: Dict[str, float] = {}
_CANCEL_CHECK_TTL = 5.0


def _normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _get_row_value(row: Dict[str, Any], key: Optional[str], fallback_keys: List[str]) -> Any:
    if key and key in row:
        return row.get(key)
    for fallback in fallback_keys:
        if fallback in row:
            return row.get(fallback)
    return None


def _build_deep_eval_sample(
    row: Dict[str, Any],
    mapping: Dict[str, Any],
) -> Optional[Tuple[str, List[str], List[str]]]:
    query_key = mapping.get("query")
    positives_key = mapping.get("positives")
    negatives_key = mapping.get("negatives")

    query = _get_row_value(row, query_key, ["query", "question", "input", "prompt"])
    positives = _get_row_value(row, positives_key, ["positives", "positive", "pos", "relevant"])
    negatives = _get_row_value(row, negatives_key, ["negatives", "negative", "neg", "irrelevant"])

    if query is None:
        return None

    positives_list = _normalize_text_list(positives)
    negatives_list = _normalize_text_list(negatives)

    if not positives_list:
        return None

    return str(query), positives_list, negatives_list


def _resolve_metric_map(metrics: List[str]) -> Dict[str, str]:
    mapping = {
        "mrr": "MRR",
        "map": "AP",
        "ndcg@10": "NDCG@10",
        "recall@10": "R@10",
        "precision@10": "P@10",
    }
    return {name: mapping[name] for name in metrics if name in mapping}


def _parse_rerank_results(results: List[Dict[str, Any]], total_docs: int) -> Tuple[List[int], Dict[int, float]]:
    score_map: Dict[int, float] = {}
    for item in results:
        idx = item.get("index")
        if idx is None:
            idx = item.get("ind")
        if idx is None:
            idx = item.get("corpus_id")
        if idx is None:
            continue
        score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        if score is None:
            continue
        try:
            score_map[int(idx)] = float(score)
        except (ValueError, TypeError):
            continue

    for i in range(total_docs):
        if i not in score_map:
            score_map[i] = float("-inf")

    ranking = sorted(score_map.keys(), key=lambda i: score_map[i], reverse=True)
    return ranking, score_map


def _cosine_scores(embeddings: List[List[float]]) -> List[float]:
    import numpy as np

    vectors = np.array(embeddings, dtype=float)
    if vectors.size == 0:
        return []
    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    if doc_vecs.size == 0:
        return []
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    scores = np.dot(doc_norms, query_norm)
    return scores.tolist()


def _get_deep_eval_total_hint(dataset: Dict[str, Any], sample_size: Optional[int]) -> Optional[int]:
    if sample_size:
        return sample_size
    for key in ("num_eval", "num_rows", "num_train", "num_test"):
        value = dataset.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def run_deep_evaluation_task(task_id: str) -> None:
    """Run deep evaluation task in background with multi-model group concurrency.

    DEPRECATED：deep_evaluation/__init__.py 已将入口指向 deep_evaluation_runner；
    本入口的 results_summary schema 与 resume 期望的 by_group 不兼容，
    直接调用会导致 resume 全量重跑。仅保留内部 helper（_iter_rows 等）被复用。
    """
    try:
        task = deep_evaluation_task_service.get_task(task_id, include_secrets=True)
        if not task:
            logger.error("Deep evaluation task not found: %s", task_id)
            return
        if task.get("status") == DeepEvaluationStatus.CANCELLED:
            logger.info("Deep evaluation task already cancelled: %s", task_id)
            return

        model_configs = task.get("model_configs") or []
        dataset_configs = task.get("dataset_configs") or []
        if not model_configs:
            raise ValueError("No model configs provided")
        if not dataset_configs:
            raise ValueError("No dataset configs provided")

        # Resolve datasets
        datasets_info = _resolve_datasets(
            dataset_configs,
            user_id=task.get("user_id"),
        )
        deep_evaluation_task_service.update_status(task_id, DeepEvaluationStatus.RUNNING)

        # Determine model_workers from worker_groups
        worker_groups = task.get("worker_groups") or {}
        model_workers = _get_model_workers(worker_groups, len(model_configs))

        # Initialize model_progress matrix
        model_names = [_get_model_display_name(cfg, i) for i, cfg in enumerate(model_configs)]
        dataset_names = [info["name"] for info in datasets_info]
        deep_evaluation_task_service.init_model_progress(task_id, model_names, dataset_names)

        logger.info(
            "Starting deep evaluation: %d models × %d datasets, model_workers=%d",
            len(model_configs), len(datasets_info), model_workers,
        )

        # Multi-model group concurrent evaluation
        effective_workers = min(model_workers, len(model_configs))
        all_results: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_model_group,
                    task_id, cfg, model_names[i], datasets_info, task,
                ): model_names[i]
                for i, cfg in enumerate(model_configs)
            }

            for future in as_completed(futures):
                model_name = futures[future]
                if is_cancelled(task_id):
                    logger.info("Deep evaluation cancelled, stopping remaining models")
                    for f in futures:
                        f.cancel()
                    break
                try:
                    result = future.result()
                    all_results[model_name] = result
                except Exception as e:
                    logger.error("Model group %s evaluation failed: %s", model_name, e)
                    all_results[model_name] = {"_error": str(e)}
                    # Mark all datasets of this model as failed
                    for ds_name in dataset_names:
                        deep_evaluation_task_service.update_model_progress(
                            task_id, model_name, ds_name, 100, "failed"
                        )

        if is_cancelled(task_id):
            deep_evaluation_task_service.update_status(task_id, DeepEvaluationStatus.CANCELLED)
            return

        # Save results
        deep_evaluation_task_service.update_results(
            task_id,
            results_summary=all_results,
            results_path=None,
        )
        deep_evaluation_task_service.update_status(task_id, DeepEvaluationStatus.COMPLETED)

    except Exception as exc:
        logger.exception("Deep evaluation failed")
        deep_evaluation_task_service.update_status(task_id, DeepEvaluationStatus.FAILED, str(exc))
    finally:
        _cleanup_cancelled(task_id)


def _get_model_display_name(model_cfg: Dict[str, Any], index: int) -> str:
    """Get display name for a model group config."""
    # 新格式：按组配置，使用 group_name
    if "group_name" in model_cfg:
        return model_cfg["group_name"]
    # 旧格式：扁平列表，使用 config_name 或 model_name
    name = model_cfg.get("config_name") or model_cfg.get("model_name")
    if name:
        return name
    return f"model_{index}"


def _get_model_workers(worker_groups: Dict[str, Any], num_models: int) -> int:
    """Determine model_workers from worker_groups config."""
    if not worker_groups:
        return min(2, num_models)

    # 新格式：worker_groups.model_workers
    if "model_workers" in worker_groups:
        return max(1, int(worker_groups["model_workers"]))

    # 旧格式：累加各组的 workers
    total = 0
    for group_cfg in worker_groups.values():
        try:
            if isinstance(group_cfg, dict):
                total += int(group_cfg.get("workers", 1))
        except (ValueError, TypeError, AttributeError):
            total += 1
    return max(1, total) if total > 0 else min(2, num_models)


def _resolve_datasets(
    dataset_configs: List[Dict[str, Any]],
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Resolve dataset configs to dataset info with paths."""
    datasets_info = []
    require_provenance = task_requires_dataset_provenance(user_id)
    for ds_cfg in dataset_configs:
        dataset_id = ds_cfg.get("dataset_id")
        if not dataset_id:
            raise ValueError("dataset_id is required in dataset_configs")
        if require_provenance:
            dataset = resolve_managed_local_evaluation_dataset(
                dataset_id,
                user_id=user_id,
            )
        else:
            dataset = resolve_local_evaluation_dataset_record(dataset_id)
        storage_path = dataset.get("storage_path")
        if not storage_path:
            raise ValueError(f"Dataset has no storage path: {dataset_id}")
        datasets_info.append({
            "dataset_id": dataset_id,
            "name": ds_cfg.get("dataset_name") or dataset.get("dataset_name") or dataset_id,
            "storage_path": storage_path,
            "dataset": dataset,
        })
    if not datasets_info:
        raise ValueError("No valid datasets found")
    return datasets_info


def _evaluate_model_group(
    task_id: str,
    model_cfg: Dict[str, Any],
    group_name: str,
    datasets_info: List[Dict[str, Any]],
    task: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single model group against all datasets.

    Supports both:
    - New format: {"group_name": "...", "embedding": {...}, "rerank": {...}, "llm": {...}}
    - Legacy format: {"model_type": "embedding", "endpoint": "...", ...}

    Returns:
        Dict with per-dataset results:
        {
            "overall": {"mean": 0.85},
            "datasets": {
                "dataset_name": {
                    "overall": {"mean": 0.82},
                    "metrics": {"mrr": {"mean": 0.82}, "ndcg@10": {"mean": 0.88}}
                }
            }
        }
    """
    from qwen3_rerank_trainer import evaluation as rerank_eval

    field_mapping = task.get("field_mapping") or {}
    metrics = task.get("metrics") or ["mrr", "ndcg@10"]
    metric_key_map = _resolve_metric_map(metrics)
    if not metric_key_map:
        raise ValueError("No valid metrics provided")

    sample_limit = task.get("max_samples")

    # 检测格式：新格式有 group_name，旧格式有 model_type
    is_new_format = "group_name" in model_cfg

    if is_new_format:
        # 新格式：组内可能有多种模型
        embedding_cfg = model_cfg.get("embedding")
        rerank_cfg = model_cfg.get("rerank")

        # 优先使用 rerank，其次 embedding
        if rerank_cfg:
            model_type = "rerank"
            active_cfg = rerank_cfg
        elif embedding_cfg:
            model_type = "embedding"
            active_cfg = embedding_cfg
        else:
            raise ValueError(f"Model group '{group_name}' has no embedding or rerank config")

        client, api_model_name, framework = _build_client(
            active_cfg,
            task.get("user_id"),
        )
    else:
        # 旧格式：单个模型配置
        model_type = (model_cfg.get("model_type") or "rerank").lower()
        client, api_model_name, framework = _build_client(
            model_cfg,
            task.get("user_id"),
        )

    datasets_results: Dict[str, Any] = {}
    all_metric_values: Dict[str, List[float]] = {name: [] for name in metric_key_map}

    for ds_info in datasets_info:
        if is_cancelled(task_id):
            break

        ds_name = ds_info["name"]
        storage_path = ds_info["storage_path"]
        dataset = ds_info["dataset"]

        deep_evaluation_task_service.update_model_progress(
            task_id, group_name, ds_name, 0, "running"
        )

        try:
            split = "eval"
            file_path, file_format = _resolve_dataset_source(storage_path, split)
            total_hint = _get_deep_eval_total_hint(dataset, sample_limit)

            processed = 0
            metric_values: Dict[str, List[float]] = {name: [] for name in metric_key_map}
            last_progress_update = 0.0
            progress_update_interval = 2.0
            progress_update_every = 20

            for row in _iter_rows(file_path, file_format, split):
                if is_cancelled(task_id):
                    break

                sample = _build_deep_eval_sample(row, field_mapping)
                if sample is None:
                    continue

                query, positives, negatives = sample
                documents = positives + negatives
                if not documents:
                    continue

                if model_type == "embedding":
                    if framework == "xinference":
                        embeddings = client.embed(api_model_name, [query] + documents)
                    else:
                        embeddings = client.embeddings([query] + documents, model=api_model_name or None)
                    scores = _cosine_scores(embeddings)
                    ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                else:
                    if framework == "xinference":
                        results = client.rerank(api_model_name, query, documents)
                    else:
                        results = client.rerank(query=query, documents=documents, model=api_model_name or None)
                    ranking, _ = _parse_rerank_results(results, len(documents))

                positive_indices = set(range(len(positives)))
                computed = rerank_eval.compute_all_metrics(ranking, positive_indices, ks=[10])

                for metric_name, metric_key in metric_key_map.items():
                    value = computed.get(metric_key)
                    if value is not None:
                        metric_values[metric_name].append(float(value))
                        all_metric_values[metric_name].append(float(value))

                processed += 1
                now = time.monotonic()
                should_update = (
                    processed % progress_update_every == 0
                    or now - last_progress_update >= progress_update_interval
                )
                if should_update:
                    last_progress_update = now
                    progress_pct = 0.0
                    if total_hint and total_hint > 0:
                        progress_pct = min(processed / total_hint * 100, 99.0)
                    deep_evaluation_task_service.update_model_progress(
                        task_id, group_name, ds_name, progress_pct, "running"
                    )

                if sample_limit and processed >= sample_limit:
                    break

            # Compute per-dataset summary
            ds_metrics_summary = {
                name: {"mean": (sum(values) / len(values)) if values else 0.0}
                for name, values in metric_values.items()
            }
            if ds_metrics_summary:
                ds_overall = sum(m["mean"] for m in ds_metrics_summary.values()) / len(ds_metrics_summary)
            else:
                ds_overall = 0.0

            datasets_results[ds_name] = {
                "overall": {"mean": ds_overall},
                "metrics": ds_metrics_summary,
                "processed": processed,
            }

            deep_evaluation_task_service.update_model_progress(
                task_id, group_name, ds_name, 100, "completed"
            )

        except Exception as exc:
            logger.error("Model group %s dataset %s evaluation failed: %s", group_name, ds_name, exc)
            datasets_results[ds_name] = {"_error": str(exc)}
            deep_evaluation_task_service.update_model_progress(
                task_id, group_name, ds_name, 100, "failed"
            )

    # Compute overall summary across all datasets
    overall_metrics = {
        name: {"mean": (sum(values) / len(values)) if values else 0.0}
        for name, values in all_metric_values.items()
    }
    if overall_metrics:
        overall_mean = sum(m["mean"] for m in overall_metrics.values()) / len(overall_metrics)
    else:
        overall_mean = 0.0

    return {
        "overall": {"mean": overall_mean},
        "metrics": overall_metrics,
        "datasets": datasets_results,
    }


def _build_client(
    model_cfg: Dict[str, Any],
    user_id: Optional[str] = None,
):
    """Build inference client from model config."""
    from ..deployment.xinference_client import XinferenceClient
    from ..deployment.vllm_client import VLLMClient
    from ..deployment.sglang_client import SGLangClient

    framework = (model_cfg.get("inference_framework") or "xinference").lower()
    endpoint = model_cfg.get("endpoint") or ""
    model_name = model_cfg.get("model_name") or ""

    if framework == "vllm":
        return VLLMClient(endpoint, user_id=user_id), model_name, framework
    if framework == "sglang":
        return SGLangClient(endpoint, user_id=user_id), model_name, framework
    return XinferenceClient(endpoint, user_id=user_id), model_name, framework


def cancel_deep_evaluation(task_id: str) -> None:
    """Mark a deep evaluation task for cancellation."""
    with _cancelled_lock:
        _cancelled_tasks.add(task_id)


def _cleanup_cancelled(task_id: str) -> None:
    with _cancelled_lock:
        _cancelled_tasks.discard(task_id)
    _cancel_check_cache.pop(task_id, None)


def is_cancelled(task_id: str) -> bool:
    """Check if a task has been cancelled."""
    with _cancelled_lock:
        if task_id in _cancelled_tasks:
            return True

    now = time.monotonic()
    last_check = _cancel_check_cache.get(task_id, 0)
    if now - last_check < _CANCEL_CHECK_TTL:
        return False

    _cancel_check_cache[task_id] = now
    task = deep_evaluation_task_service.get_task(task_id)
    if task and task.get("status") == "cancelled":
        with _cancelled_lock:
            _cancelled_tasks.add(task_id)
        return True
    return False


def _detect_file_format(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    return {
        ".jsonl": "jsonl",
        ".json": "json",
        ".parquet": "parquet",
        ".csv": "csv",
        ".arrow": "arrow",
    }.get(suffix, "jsonl")


def _is_arrow_dataset_dir(dir_path: Path) -> bool:
    return dir_path.is_dir() and (
        (dir_path / "dataset_info.json").exists() or any(dir_path.glob("*.arrow"))
    )


def _resolve_dataset_source(storage_path: str, split: str) -> Tuple[Path, str]:
    path = Path(storage_path)
    split = (split or "eval").lower()

    if path.is_file():
        return path, _detect_file_format(path)

    if not path.is_dir():
        raise ValueError(f"Dataset path not found: {storage_path}")

    if split == "all":
        if _is_arrow_dataset_dir(path):
            return path, "arrow"

        arrow_split_dirs = []
        for name in ["train", "eval", "val", "validation", "dev", "test"]:
            split_dir = path / name
            if _is_arrow_dataset_dir(split_dir):
                arrow_split_dirs.append(split_dir)
        if arrow_split_dirs:
            return path, "arrow_multi"

        return path, "multi"

    # Arrow dataset directory
    if _is_arrow_dataset_dir(path):
        return path, "arrow"

    # Arrow split directories
    split_dir_map = {
        "train": ["train"],
        "eval": ["eval", "val", "validation", "dev"],
        "test": ["test"],
    }
    split_dir_candidates = split_dir_map.get(split, []) + [
        "train",
        "eval",
        "val",
        "validation",
        "dev",
        "test",
    ]
    for name in split_dir_candidates:
        split_dir = path / name
        if _is_arrow_dataset_dir(split_dir):
            return split_dir, "arrow"

    train_patterns = [
        "train.jsonl", "train.json", "train.parquet", "train.csv", "train.arrow",
        "data.jsonl", "data.json", "data.parquet", "data.csv", "data.arrow",
    ]
    eval_patterns = [
        "val.jsonl", "val.json", "validation.jsonl", "validation.json", "dev.jsonl", "dev.json",
        "eval.jsonl", "eval.json",
        "val.parquet", "validation.parquet", "dev.parquet", "eval.parquet",
        "val.csv", "validation.csv", "dev.csv", "eval.csv",
        "val.arrow", "validation.arrow", "dev.arrow", "eval.arrow",
    ]
    test_patterns = [
        "test.jsonl", "test.json", "test.parquet", "test.csv", "test.arrow",
    ]

    if split == "train":
        patterns = train_patterns
    elif split in {"eval", "validation", "val", "dev"}:
        patterns = eval_patterns
    elif split == "test":
        patterns = test_patterns
    else:
        patterns = []

    for pattern in patterns:
        files = list(path.glob(pattern))
        if files:
            return files[0], _detect_file_format(files[0])

    # Fallback to any data file
    for pattern in ["*.jsonl", "*.json", "*.parquet", "*.csv", "*.arrow"]:
        files = list(path.glob(pattern))
        if files:
            return files[0], _detect_file_format(files[0])

    raise ValueError(f"No dataset files found in: {storage_path}")


def _iter_rows(file_path: Path, file_format: str, split: str) -> Iterable[Dict[str, Any]]:
    if file_format == "multi":
        if not file_path.is_dir():
            raise ValueError(f"Dataset path not found: {file_path}")

        patterns = [
            "train.jsonl", "train.json", "train.parquet", "train.csv", "train.arrow",
            "eval.jsonl", "eval.json", "eval.parquet", "eval.csv", "eval.arrow",
            "val.jsonl", "val.json", "val.parquet", "val.csv", "val.arrow",
            "validation.jsonl", "validation.json", "validation.parquet", "validation.csv", "validation.arrow",
            "dev.jsonl", "dev.json", "dev.parquet", "dev.csv", "dev.arrow",
            "test.jsonl", "test.json", "test.parquet", "test.csv", "test.arrow",
            "data.jsonl", "data.json", "data.parquet", "data.csv", "data.arrow",
        ]

        candidates = []
        for pattern in patterns:
            candidates.extend(sorted(file_path.glob(pattern)))
        if not candidates:
            for pattern in ["*.jsonl", "*.json", "*.parquet", "*.csv", "*.arrow"]:
                candidates.extend(sorted(file_path.glob(pattern)))

        priority = {".parquet": 0, ".arrow": 1, ".jsonl": 2, ".json": 3, ".csv": 4}
        selected = {}
        for data_file in candidates:
            ext = data_file.suffix.lower()
            key = data_file.stem
            if key not in selected:
                selected[key] = data_file
                continue
            current = selected[key]
            cur_pri = priority.get(current.suffix.lower(), 99)
            new_pri = priority.get(ext, 99)
            if new_pri < cur_pri:
                selected[key] = data_file

        for data_file in sorted(selected.values()):
            fmt = _detect_file_format(data_file)
            for row in _iter_rows(data_file, fmt, split):
                yield row
        return

    if file_format == "arrow_multi":
        if not file_path.is_dir():
            raise ValueError(f"Dataset path not found: {file_path}")
        try:
            from datasets import load_from_disk, DatasetDict
        except Exception as exc:
            raise ValueError("datasets is required to read arrow datasets") from exc

        split_order = ["train", "eval", "val", "validation", "dev", "test"]
        for name in split_order:
            split_dir = file_path / name
            if not _is_arrow_dataset_dir(split_dir):
                continue
            ds = load_from_disk(str(split_dir))
            if isinstance(ds, DatasetDict):
                for split_ds in ds.values():
                    for row in split_ds:
                        yield row
            else:
                for row in ds:
                    yield row
        return

    if file_format == "jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if file_format == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        yield row
            else:
                raise ValueError("JSON dataset must be a list of objects")
        return

    if file_format == "csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        return

    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise ValueError("pyarrow is required to read parquet datasets") from exc

        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            records = batch.to_pydict()
            keys = list(records.keys())
            for idx in range(len(next(iter(records.values()), []))):
                row = {key: records[key][idx] for key in keys}
                yield row
        return

    if file_format == "arrow":
        try:
            from datasets import load_from_disk, DatasetDict
        except Exception as exc:
            raise ValueError("datasets is required to read arrow datasets") from exc

        load_path = file_path
        if file_path.is_file() and file_path.suffix == ".arrow":
            load_path = file_path.parent

        ds = load_from_disk(str(load_path))
        if isinstance(ds, DatasetDict):
            if split == "all":
                for split_ds in ds.values():
                    for row in split_ds:
                        yield row
                return
            split_key = split
            if split_key not in ds:
                split_key = "validation" if split == "eval" and "validation" in ds else None
            if split_key is None:
                if "train" in ds:
                    split_key = "train"
                else:
                    split_key = list(ds.keys())[0]
            ds = ds[split_key]
        for row in ds:
            yield row
        return

    raise ValueError(f"Unsupported dataset format: {file_format}")


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _coerce_context(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v is not None]
            except Exception:
                pass
        return [line for line in value.splitlines() if line.strip()]
    return [str(value)]


def _build_sample(row: Dict[str, Any], mapping: Dict[str, Any]) -> Optional[EvaluationSample]:
    input_field = mapping.get("input")
    if not input_field:
        raise ValueError("Field mapping must include input")

    input_value = row.get(input_field)
    input_text = _coerce_text(input_value)
    if not input_text or not input_text.strip():
        return None

    expected_field = mapping.get("expected_output")
    actual_field = mapping.get("actual_output")
    context_field = mapping.get("retrieval_context")

    expected_output = _coerce_text(row.get(expected_field)) if expected_field else None
    actual_output = _coerce_text(row.get(actual_field)) if actual_field else None
    retrieval_context = _coerce_context(row.get(context_field)) if context_field else []

    return EvaluationSample(
        input=input_text,
        expected_output=expected_output,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )


async def _evaluate_samples(
    task_id: str,
    evaluator: DeepEvaluator,
    samples: List[EvaluationSample],
    metrics: List[str],
) -> List[Any]:
    semaphore = asyncio.Semaphore(evaluator.concurrency)
    results: List[Any] = []
    total = len(samples)
    completed = 0
    last_update = 0.0

    async def evaluate_one(sample: EvaluationSample):
        nonlocal completed, last_update
        async with semaphore:
            if is_cancelled(task_id):
                raise asyncio.CancelledError()
            result = await evaluator.evaluate(sample, metrics)
            completed += 1
            now = time.monotonic()
            if now - last_update >= 2.0 or completed >= total:
                last_update = now
                deep_evaluation_task_service.update_progress(task_id, completed, total)
            if is_cancelled(task_id):
                raise asyncio.CancelledError()
            return result

    tasks = [asyncio.create_task(evaluate_one(sample)) for sample in samples]
    try:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        raise
    except Exception:
        for t in tasks:
            t.cancel()
        raise
    finally:
        deep_evaluation_task_service.update_progress(task_id, completed, total)

    return results
