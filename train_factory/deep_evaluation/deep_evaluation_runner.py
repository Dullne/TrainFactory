"""Deep evaluation task runner for embedding/reranker models."""

import asyncio
import json
import logging
import math
import time
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config.settings import get_settings
from ..evaluation.dataset_access import (
    resolve_local_evaluation_dataset_record,
    resolve_managed_local_evaluation_dataset,
    task_requires_dataset_provenance,
)
from ..generation.clients import EmbeddingClient, EmbeddingConfig, RerankClient, RerankConfig
from ..storage.services.deep_evaluation_task_service import deep_evaluation_task_service
from ..storage.services.milvus_collection_service import milvus_collection_service
from ..storage.services.model_config_service import model_config_service
from ..storage.services.outbound_endpoint_policy import validate_user_outbound_url
from .metrics import MetricRegistry, EvaluationSample
from .evaluator import DeepEvaluator
from .llm_judge import create_llm_judge_from_dict
from .task_runner import _resolve_dataset_source, _iter_rows, is_cancelled, _cleanup_cancelled

logger = logging.getLogger(__name__)

TRADITIONAL_METRICS = {"mrr", "map", "ndcg@10", "recall@10", "precision@10"}

MAX_DEEP_EVALUATION_MODEL_GROUPS = 16
MAX_DEEP_EVALUATION_DATASETS = 32
MAX_DEEP_EVALUATION_COMBINATIONS = 128
# 在线检索模式会把样本一次性载入内存（query+chunk 全文），1M 行可达 GB 级。
# 降低默认上限以限制内存峰值；分批流式构建样本留待有运行环境时重构。
MAX_DEEP_EVALUATION_SAMPLES = 200_000
MAX_DEEP_EVALUATION_METRICS = 16
MAX_DEEP_EVALUATION_MODEL_CONCURRENCY = 64
MAX_DEEP_EVALUATION_LLM_CONCURRENCY = 32
MAX_DEEP_EVALUATION_MODEL_WORKERS = 16
MAX_DEEP_EVALUATION_LLM_TOKENS = 131_072
MAX_DEEP_EVALUATION_TIMEOUT = 600
MAX_DEEP_EVALUATION_RETRIES = 10
MAX_DEEP_EVALUATION_TOP_K = 200


def _require_bounded_integer(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Deep evaluation {name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Deep evaluation {name} must be between {minimum} and {maximum}"
        )
    return value


def _validate_deep_model_config(
    config: Any,
    *,
    name: str,
    llm: bool = False,
) -> None:
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError(f"Deep evaluation {name} must be an object")

    concurrency_limit = (
        MAX_DEEP_EVALUATION_LLM_CONCURRENCY
        if llm
        else MAX_DEEP_EVALUATION_MODEL_CONCURRENCY
    )
    _require_bounded_integer(
        config.get("concurrency", 2 if llm else 8),
        f"{name}.concurrency",
        1,
        concurrency_limit,
    )
    if not llm:
        return

    _require_bounded_integer(
        config.get("max_tokens", 2048),
        f"{name}.max_tokens",
        1,
        MAX_DEEP_EVALUATION_LLM_TOKENS,
    )
    _require_bounded_integer(
        config.get("timeout", 60),
        f"{name}.timeout",
        1,
        MAX_DEEP_EVALUATION_TIMEOUT,
    )
    _require_bounded_integer(
        config.get("max_retries", 3),
        f"{name}.max_retries",
        0,
        MAX_DEEP_EVALUATION_RETRIES,
    )

    for field_name, minimum, maximum, default in (
        ("temperature", 0.0, 2.0, 0.0),
        ("top_p", 0.0, 1.0, 1.0),
    ):
        value = config.get(field_name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Deep evaluation {name}.{field_name} must be numeric")
        if not minimum <= float(value) <= maximum:
            raise ValueError(
                f"Deep evaluation {name}.{field_name} must be between "
                f"{minimum} and {maximum}"
            )

    top_k = config.get("top_k")
    if top_k is not None:
        _require_bounded_integer(
            top_k,
            f"{name}.top_k",
            0,
            MAX_DEEP_EVALUATION_TOP_K,
        )


def validate_deep_evaluation_resource_config(task: Dict[str, Any]) -> None:
    """Reject persisted deep-evaluation work with excessive fan-out."""
    model_groups = task.get("model_configs")
    datasets = task.get("dataset_configs")
    if not isinstance(model_groups, list) or not (
        1 <= len(model_groups) <= MAX_DEEP_EVALUATION_MODEL_GROUPS
    ):
        raise ValueError(
            "Deep evaluation must contain 1-"
            f"{MAX_DEEP_EVALUATION_MODEL_GROUPS} model groups"
        )
    if not isinstance(datasets, list) or not (
        1 <= len(datasets) <= MAX_DEEP_EVALUATION_DATASETS
    ):
        raise ValueError(
            "Deep evaluation must contain 1-"
            f"{MAX_DEEP_EVALUATION_DATASETS} datasets"
        )
    if len(model_groups) * len(datasets) > MAX_DEEP_EVALUATION_COMBINATIONS:
        raise ValueError(
            "Deep evaluation model/dataset combinations exceed the configured limit"
        )

    metrics = task.get("metrics") or ["mrr", "ndcg@10"]
    if not isinstance(metrics, list) or not (
        1 <= len(metrics) <= MAX_DEEP_EVALUATION_METRICS
    ):
        raise ValueError(
            "Deep evaluation must contain 1-"
            f"{MAX_DEEP_EVALUATION_METRICS} metrics"
        )

    max_samples = task.get("max_samples")
    if max_samples is not None:
        _require_bounded_integer(
            max_samples,
            "max_samples",
            1,
            MAX_DEEP_EVALUATION_SAMPLES,
        )

    worker_groups = task.get("worker_groups") or {}
    if not isinstance(worker_groups, dict):
        raise ValueError("Deep evaluation worker_groups must be an object")
    _require_bounded_integer(
        worker_groups.get("model_workers", 2),
        "model_workers",
        1,
        MAX_DEEP_EVALUATION_MODEL_WORKERS,
    )
    if "retrieval_top_k" in worker_groups:
        _require_bounded_integer(
            worker_groups["retrieval_top_k"],
            "retrieval_top_k",
            1,
            100,
        )
    _validate_deep_model_config(
        worker_groups.get("retrieval_embedding_config"),
        name="retrieval_embedding_config",
    )

    for index, group in enumerate(model_groups):
        if not isinstance(group, dict):
            raise ValueError(f"Deep evaluation model group {index} must be an object")
        _validate_deep_model_config(
            group.get("embedding"),
            name=f"model_configs[{index}].embedding",
        )
        _validate_deep_model_config(
            group.get("rerank"),
            name=f"model_configs[{index}].rerank",
        )
        _validate_deep_model_config(
            group.get("llm"),
            name=f"model_configs[{index}].llm",
            llm=True,
        )


def _resolve_task_dataset_records(
    raw_datasets: List[Dict[str, Any]],
    user_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Re-resolve persisted dataset IDs before a worker reads any files."""
    require_provenance = task_requires_dataset_provenance(user_id)
    resolved = []
    for raw_dataset in raw_datasets:
        dataset_id = raw_dataset.get("dataset_id")
        if not dataset_id:
            raise ValueError("dataset_id is required in dataset_configs")
        if require_provenance:
            dataset = resolve_managed_local_evaluation_dataset(
                dataset_id,
                user_id=user_id,
            )
        else:
            dataset = resolve_local_evaluation_dataset_record(dataset_id)
        resolved.append(dataset)
    return resolved


def _require_task_collection_ownership(task: Dict[str, Any]) -> None:
    """Reject online evaluation if its collection changed tenants."""
    worker_groups = task.get("worker_groups") or {}
    if worker_groups.get("retrieval_mode") != "online":
        return

    if not get_settings().auth_enabled:
        return
    user_id = task.get("user_id")
    if not user_id or user_id == "anonymous":
        raise PermissionError("Authenticated deep evaluation task owner is missing")

    collection_name = worker_groups.get("milvus_collection")
    collection = (
        milvus_collection_service.get_by_name(collection_name)
        if collection_name
        else None
    )
    if not collection or collection.get("user_id") != user_id:
        raise PermissionError(
            "Milvus collection is not owned by deep evaluation task user"
        )


def _save_partial_results(
    task_id: str,
    group_details: Dict[str, Any],
    merged_stats: Dict[str, Dict[str, float]],
) -> None:
    """Save partial results for resume capability on cancel/failure."""
    try:
        metrics_summary = {}
        for name, stats in merged_stats.items():
            if stats["count"] <= 0:
                continue
            metrics_summary[name] = {
                "mean": stats["sum"] / stats["count"],
                "min": stats["min"],
                "max": stats["max"],
                "count": stats["count"],
            }
        overall = None
        if metrics_summary:
            means = [s["mean"] for s in metrics_summary.values()]
            overall = {"mean": sum(means) / len(means), "min": min(means), "max": max(means)}
        summary = {"overall": overall, "metrics": metrics_summary, "by_group": group_details}
        deep_evaluation_task_service.update_results(task_id, results_summary=summary)
        logger.info("Saved partial results for task %s (%d groups)", task_id, len(group_details))
    except Exception:
        logger.warning("Failed to save partial results for task %s", task_id, exc_info=True)


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_candidates(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                return _normalize_candidates(parsed)
            except Exception:
                return [raw]
        return [raw]
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                for key in ("text", "passage", "document", "content"):
                    if key in item and item[key]:
                        items.append(str(item[key]))
                        break
                else:
                    items.append(json.dumps(item, ensure_ascii=False))
            else:
                items.append(str(item))
        return [it for it in items if it and it.strip()]
    return [str(value)]


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


def _get_row_value(row: Dict[str, Any], key: Optional[str], fallback_keys: List[str]) -> Any:
    if key and key in row:
        return row.get(key)
    for fallback in fallback_keys:
        if fallback in row:
            return row.get(fallback)
    return None


def _build_sample(row: Dict[str, Any], mapping: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    query_field = mapping.get("query")
    positives_field = mapping.get("positives")
    negatives_field = mapping.get("negatives")

    query = _coerce_text(_get_row_value(row, query_field, ["query", "question", "input", "prompt"]))
    if not query or not query.strip():
        return None

    positives = _normalize_candidates(
        _get_row_value(
            row,
            positives_field,
            ["positives", "positive", "pos", "relevant", "relevants", "answers", "answer"],
        )
    )
    negatives = _normalize_candidates(
        _get_row_value(
            row,
            negatives_field,
            ["negatives", "negative", "neg", "irrelevant", "irrelevants", "hard_negatives", "hard_negative"],
        )
    )
    if not positives:
        return None

    documents = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    if not documents:
        return None

    return {
        "query": query,
        "documents": documents,
        "labels": labels,
    }


def _build_llm_sample(row: Dict[str, Any], mapping: Dict[str, Any]) -> Optional[EvaluationSample]:
    input_key = mapping.get("input") or mapping.get("query")
    expected_key = mapping.get("expected_output")
    actual_key = mapping.get("actual_output")
    context_key = mapping.get("retrieval_context")

    input_value = _get_row_value(row, input_key, ["input", "query", "question", "prompt"])
    input_text = _coerce_text(input_value)
    if not input_text or not input_text.strip():
        return None

    expected_output = _coerce_text(
        _get_row_value(row, expected_key, ["expected_output", "expected", "answer", "reference", "label"])
    )
    actual_output = _coerce_text(
        _get_row_value(row, actual_key, ["actual_output", "output", "prediction", "response", "answer"])
    )
    retrieval_context = _coerce_context(
        _get_row_value(row, context_key, ["retrieval_context", "context", "contexts", "passages", "documents"])
    )

    return EvaluationSample(
        input=input_text,
        expected_output=expected_output,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )


async def _build_online_samples(
    rows: List[Dict[str, Any]],
    field_mapping: Dict[str, Any],
    milvus_collection: str,
    retrieval_embedding_config: Dict[str, Any],
    retrieval_top_k: int,
    task_id: str,
    user_id: Optional[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """通过 Milvus 在线检索构建评估样本。

    从 QA 数据集行中提取 query 和 source chunk 信息，
    用 embedding 模型向量化 query 后在 Milvus 中检索 top-k 候选，
    source chunk 标记为 positive，其余为 negative。
    """
    from ..generation.clients.milvus_client import MilvusClient, MilvusConfig

    query_field = field_mapping.get("query")
    chunk_id_field = (
        field_mapping.get("chunk_id")
        or field_mapping.get("source_chunk_id")
        or "chunk_id"
    )
    chunk_content_field = (
        field_mapping.get("chunk_content")
        or field_mapping.get("source_chunk_content")
        or "chunk_content"
    )

    # 提取 query 和 ground truth 信息
    entries = []
    for row in rows:
        query = _coerce_text(_get_row_value(row, query_field, ["query", "question", "input"]))
        if not query or not query.strip():
            continue
        chunk_id = (
            row.get(chunk_id_field)
            or row.get("chunk_id")
            or row.get("source_chunk_id")
            or ""
        )
        chunk_content = (
            row.get(chunk_content_field)
            or row.get("chunk_content")
            or row.get("source_chunk_content")
            or ""
        )
        entries.append((query, chunk_id, chunk_content))

    if not entries:
        return []

    # 初始化客户端
    endpoint = validate_user_outbound_url(
        retrieval_embedding_config.get("endpoint", ""),
        user_id,
    )
    emb_config = EmbeddingConfig(
        endpoint=endpoint,
        model=retrieval_embedding_config.get("model_name") or retrieval_embedding_config.get("model", ""),
        api_key=retrieval_embedding_config.get("api_key"),
        user_id=user_id,
    )
    embedding_client = EmbeddingClient(emb_config)

    milvus_client = MilvusClient(MilvusConfig())

    samples: List[Dict[str, Any]] = []
    try:
        milvus_client.connect()
        async with embedding_client:
            BATCH_SIZE = 32
            for batch_start in range(0, len(entries), BATCH_SIZE):
                if is_cancelled(task_id):
                    break

                batch = entries[batch_start:batch_start + BATCH_SIZE]
                batch_queries = [e[0] for e in batch]

                # 采样阶段上报进度（此前整个向量化+检索阶段进度为 0%）
                if progress_callback is not None:
                    progress_callback(
                        min(batch_start + BATCH_SIZE, len(entries)),
                        len(entries),
                    )

                # 向量化 query
                query_vectors = await embedding_client.embed(batch_queries)

                # 逐个搜索 Milvus（search_similar 支持批量）
                for i, (query, gt_chunk_id, gt_content) in enumerate(batch):
                    try:
                        search_results = milvus_client.search_similar(
                            milvus_collection,
                            query_vectors[i:i+1],
                            top_k=retrieval_top_k,
                        )
                        hits = search_results[0] if search_results else []
                    except Exception as e:
                        logger.warning("Milvus search failed for query: %s", e)
                        hits = []

                    documents = []
                    labels = []

                    # 检索到的候选按原顺序作为候选
                    for hit in hits:
                        doc_text = hit.get("chunk_content", "")
                        if not doc_text or not doc_text.strip():
                            continue
                        hit_id = hit.get("chunk_id", "")
                        documents.append(doc_text)
                        if gt_chunk_id and hit_id == gt_chunk_id:
                            labels.append(1)
                        elif not gt_chunk_id and gt_content and doc_text == gt_content:
                            labels.append(1)
                        else:
                            labels.append(0)

                    if documents:
                        samples.append({
                            "query": query,
                            "documents": documents,
                            "labels": labels,
                        })

                logger.info(
                    "Online retrieval progress: %d/%d queries",
                    min(batch_start + BATCH_SIZE, len(entries)),
                    len(entries),
                )
    finally:
        try:
            milvus_client.close()
        except Exception:
            pass

    logger.info("Built %d online samples from %d queries", len(samples), len(entries))
    return samples


def _parse_metric(metric: str) -> Tuple[str, Optional[int]]:
    if "@" in metric:
        name, raw_k = metric.split("@", 1)
        try:
            return name.lower(), int(raw_k)
        except ValueError:
            return name.lower(), None
    return metric.lower(), None


def _dcg(labels: List[int], k: int) -> float:
    score = 0.0
    for idx, rel in enumerate(labels[:k], start=1):
        if rel <= 0:
            continue
        score += (2 ** rel - 1) / math.log2(idx + 1)
    return score


def _compute_metrics(scores: List[float], labels: List[int], metrics: List[str]) -> Dict[str, float]:
    ranked = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    ranked_labels = [label for _, label in ranked]
    num_pos = sum(labels)

    results: Dict[str, float] = {}
    for metric in metrics:
        name, k = _parse_metric(metric)
        if name == "mrr":
            value = 0.0
            for idx, label in enumerate(ranked_labels, start=1):
                if label:
                    value = 1.0 / idx
                    break
            results[metric] = value
            continue

        if name == "map":
            if num_pos == 0:
                results[metric] = 0.0
                continue
            hit = 0
            sum_prec = 0.0
            for idx, label in enumerate(ranked_labels, start=1):
                if label:
                    hit += 1
                    sum_prec += hit / idx
            results[metric] = sum_prec / num_pos
            continue

        if name == "ndcg":
            cutoff = k or len(ranked_labels)
            if cutoff <= 0:
                results[metric] = 0.0
                continue
            dcg = _dcg(ranked_labels, cutoff)
            ideal = sorted(labels, reverse=True)
            idcg = _dcg(ideal, cutoff)
            results[metric] = (dcg / idcg) if idcg > 0 else 0.0
            continue

        if name in {"recall", "precision"}:
            cutoff = k or len(ranked_labels)
            cutoff = min(cutoff, len(ranked_labels))
            if cutoff <= 0:
                results[metric] = 0.0
                continue
            hits = sum(ranked_labels[:cutoff])
            if name == "recall":
                results[metric] = hits / num_pos if num_pos else 0.0
            else:
                results[metric] = hits / cutoff
            continue

        raise ValueError(f"Unsupported metric: {metric}")

    return results


def _summarize_results(results: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Any]:
    metric_summary: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        values = [r["metrics"].get(metric) for r in results if metric in r["metrics"]]
        values = [v for v in values if isinstance(v, (int, float))]
        if not values:
            continue
        metric_summary[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    overall_scores = [r.get("overall") for r in results if isinstance(r.get("overall"), (int, float))]
    overall = None
    if overall_scores:
        overall = {
            "mean": sum(overall_scores) / len(overall_scores),
            "min": min(overall_scores),
            "max": max(overall_scores),
        }

    return {
        "overall": overall,
        "metrics": metric_summary,
    }


def _contains_successful_metric_result(value: Any) -> bool:
    if isinstance(value, dict):
        successful_samples = value.get("successful_samples")
        if (
            isinstance(successful_samples, (int, float))
            and not isinstance(successful_samples, bool)
            and successful_samples > 0
        ):
            return True

        metrics = value.get("metrics")
        if isinstance(metrics, dict) and any(
            isinstance(stat, dict)
            and isinstance(stat.get("count"), (int, float))
            and not isinstance(stat.get("count"), bool)
            and stat["count"] > 0
            for stat in metrics.values()
        ):
            return True

        return any(
            _contains_successful_metric_result(item)
            for key, item in value.items()
            if key not in {"_error", "metrics"}
        )
    if isinstance(value, list):
        return any(_contains_successful_metric_result(item) for item in value)
    return False


def _is_successful_group_result(result: Any) -> bool:
    """Return whether a group contains at least one non-skipped metric result."""
    return bool(
        isinstance(result, dict)
        and "_error" not in result
        and _contains_successful_metric_result(
            {"retrieval": result.get("retrieval"), "llm": result.get("llm")}
        )
    )


def _get_model_workers(worker_groups: Optional[Dict[str, Any]], total_groups: int) -> int:
    if not isinstance(worker_groups, dict):
        return min(2, total_groups) if total_groups > 0 else 1
    value = worker_groups.get("model_workers", 1)
    try:
        value = max(1, int(value))
    except Exception:
        value = 1
    return min(value, MAX_DEEP_EVALUATION_MODEL_WORKERS, max(1, total_groups))


def _normalize_model_type(raw_type: Optional[str]) -> Optional[str]:
    if not raw_type:
        return None
    value = str(raw_type).lower()
    if value == "reranker":
        return "rerank"
    return value


def _model_display_name(config: Dict[str, Any], index: int) -> str:
    return (
        config.get("config_name")
        or config.get("model_name")
        or config.get("model")
        or config.get("config_id")
        or f"model_{index}"
    )


def _resolve_model_config(config_data: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
    config_id = config_data.get("config_id")
    if not config_id:
        resolved = dict(config_data)
        if resolved.get("endpoint"):
            resolved["endpoint"] = validate_user_outbound_url(
                resolved["endpoint"],
                user_id,
            )
        return resolved

    config = model_config_service.get_config(config_id)
    if not config:
        raise ValueError(f"Model config not found: {config_id}")
    if user_id and config.get("user_id") and config.get("user_id") != user_id:
        raise ValueError("Not authorized to access model config")

    resolved = {
        "config_id": config_id,
        "config_name": config.get("config_name"),
        "endpoint": config.get("api_endpoint"),
        "model_name": config.get("model_name"),
        "api_key": config.get("api_key"),
        "inference_framework": config.get("inference_framework"),
        "model_type": config.get("model_type"),
    }
    resolved["endpoint"] = validate_user_outbound_url(
        resolved.get("endpoint") or "",
        user_id,
    )
    return resolved


class _ProgressTracker:
    def __init__(self, task_id: str, total: int):
        self.task_id = task_id
        self.total = max(0, int(total))
        self.processed = 0
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def increment(self, count: int = 1) -> None:
        async with self.lock:
            self.processed += count
            now = time.monotonic()
            if now - self.last_update >= 2.0 or self.processed >= self.total:
                self.last_update = now
                deep_evaluation_task_service.update_progress(self.task_id, self.processed, self.total)


async def _evaluate_samples(
    task_id: str,
    eval_type: str,
    model_config: Dict[str, Any],
    samples: List[Dict[str, Any]],
    metrics: List[str],
    concurrency: int,
    progress: _ProgressTracker,
    user_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], int]:
    results: List[Dict[str, Any]] = []

    endpoint = validate_user_outbound_url(
        model_config.get("endpoint", ""),
        user_id,
    )

    if eval_type == "embedding":
        model_name = model_config.get("model_name") or model_config.get("model")
        client = EmbeddingClient(EmbeddingConfig(
            endpoint=endpoint,
            model=model_name or "",
            api_key=model_config.get("api_key"),
            user_id=user_id,
        ))
    else:
        model_name = model_config.get("model_name") or model_config.get("model")
        client = RerankClient(RerankConfig(
            endpoint=endpoint,
            model=model_name or "",
            api_key=model_config.get("api_key"),
            user_id=user_id,
        ))

    async with client:
        skipped_count = 0

        async def process_one(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            nonlocal skipped_count
            if is_cancelled(task_id):
                raise asyncio.CancelledError()
            query = sample["query"]
            documents = sample["documents"]
            labels = sample["labels"]

            try:
                if eval_type == "embedding":
                    scores = await client.similarity(query, documents)
                else:
                    scores = await client.score(query, documents)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 单样本失败不摧毁整个评估：记录失败样本数并跳过（与 MTEB /
                # LLM 指标路径的容错语义一致）。
                logger.warning(
                    "Deep-eval sample failed (skipped): query=%r err=%s",
                    str(query)[:80],
                    exc,
                )
                skipped_count += 1
                return None

            metric_values = _compute_metrics(scores, labels, metrics)
            overall = None
            if metric_values:
                overall = sum(metric_values.values()) / len(metric_values)

            return {
                "query": query,
                "metrics": metric_values,
                "overall": overall,
                "num_candidates": len(documents),
                "num_positives": sum(labels),
            }

        sample_iterator = iter(samples)

        async def worker() -> None:
            for sample in sample_iterator:
                result = await process_one(sample)
                if result is not None:
                    results.append(result)
                await progress.increment(1)

        worker_count = min(
            max(1, int(concurrency)),
            MAX_DEEP_EVALUATION_MODEL_CONCURRENCY,
            max(1, len(samples)),
        )
        tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
        except Exception:
            for task in tasks:
                task.cancel()
            raise

    return results, skipped_count


def run_deep_evaluation_task(task_id: str, *, existing_results: Optional[Dict[str, Any]] = None) -> None:
    """Run deep evaluation task in background.

    Args:
        task_id: The task ID to run.
        existing_results: Previously completed group results (keyed by group_name).
            When provided, groups that already have successful results will be skipped.
    """

    async def _run() -> None:
        try:
            task = deep_evaluation_task_service.get_task(task_id, include_secrets=True)
            if not task:
                logger.error("Deep evaluation task not found: %s", task_id)
                return
            if task.get("status") == "cancelled":
                logger.info("Deep evaluation task already cancelled: %s", task_id)
                return

            validate_deep_evaluation_resource_config(task)
            _require_task_collection_ownership(task)

            model_groups = task.get("model_configs") or []
            raw_datasets = task.get("dataset_configs") or []
            if not raw_datasets:
                raise ValueError("数据集配置为空")
            if not model_groups:
                raise ValueError("模型组配置为空")

            configured_group_names = {
                group.get("group_name") or f"模型组 {idx + 1}"
                for idx, group in enumerate(model_groups)
            }

            # 检测 resume：只恢复当前配置中仍然存在的成功组。
            completed_group_names: set = set()
            existing_by_group: Dict[str, Any] = {
                gname: gdata
                for gname, gdata in (existing_results or {}).items()
                if gname in configured_group_names
            }
            for gname, gdata in existing_by_group.items():
                if _is_successful_group_result(gdata):
                    completed_group_names.add(gname)
            is_resume = bool(completed_group_names)
            if is_resume:
                logger.info(
                    "Resuming evaluation: %d completed groups: %s",
                    len(completed_group_names), list(completed_group_names),
                )

            resolved_datasets = _resolve_task_dataset_records(
                raw_datasets,
                task.get("user_id"),
            )
            deep_evaluation_task_service.update_status(task_id, "running")

            field_mapping = task.get("field_mapping") or {}
            metrics = task.get("metrics") or ["mrr", "ndcg@10"]

            metric_info = {m["name"]: m for m in MetricRegistry.list_metrics()}
            retrieval_metrics = [m for m in metrics if m in TRADITIONAL_METRICS]
            deepeval_metrics = [m for m in metrics if m in metric_info]
            requires_llm = any(
                metric_info[m].get("requires_llm") for m in deepeval_metrics
            )

            if retrieval_metrics and any(
                not (group.get("embedding") or group.get("rerank")) for group in model_groups
            ):
                raise ValueError("检索指标评估要求每个模型组至少配置 Embedding 或 Rerank")
            if requires_llm and any(not group.get("llm") for group in model_groups):
                raise ValueError("所选 LLM 指标要求每个模型组配置 LLM")

            dataset_info: Dict[str, Dict[str, Any]] = {}
            retrieval_samples_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
            llm_samples_by_dataset: Dict[str, List[EvaluationSample]] = {}
            sample_limit = task.get("max_samples") or MAX_DEEP_EVALUATION_SAMPLES

            # 检测在线检索模式
            worker_groups_cfg = task.get("worker_groups") or {}
            retrieval_mode = worker_groups_cfg.get("retrieval_mode", "offline")

            for dataset in resolved_datasets:
                dataset_id = dataset["dataset_id"]

                storage_path = dataset.get("storage_path")
                file_path, file_format = _resolve_dataset_source(storage_path, "all")

                retrieval_samples: List[Dict[str, Any]] = []
                llm_samples: List[EvaluationSample] = []

                if retrieval_mode == "online" and retrieval_metrics:
                    # 在线模式：从 Milvus 检索构建样本
                    all_rows = list(
                        islice(
                            _iter_rows(file_path, file_format, "all"),
                            sample_limit,
                        )
                    )
                    logger.info(
                        "Online retrieval mode: %d rows, collection=%s",
                        len(all_rows),
                        worker_groups_cfg.get("milvus_collection"),
                    )
                    retrieval_samples = await _build_online_samples(
                        rows=all_rows,
                        field_mapping=field_mapping,
                        milvus_collection=worker_groups_cfg["milvus_collection"],
                        retrieval_embedding_config=worker_groups_cfg["retrieval_embedding_config"],
                        retrieval_top_k=worker_groups_cfg.get("retrieval_top_k", 20),
                        task_id=task_id,
                        user_id=task.get("user_id"),
                        # 采样阶段进度（日志可见；完整进度条由 _ProgressTracker
                        # 在样本就绪后接管）
                        progress_callback=lambda done, total: logger.info(
                            "[deep-eval] Online sampling: %d/%d rows processed",
                            done,
                            total,
                        ),
                    )
                    # 在线模式下也构建 LLM 样本（如果需要）
                    if deepeval_metrics:
                        for row in all_rows:
                            llm_sample = _build_llm_sample(row, field_mapping)
                            if llm_sample is not None:
                                llm_samples.append(llm_sample)
                else:
                    # 离线模式：从数据集中读取预计算的 positives/negatives
                    for row in _iter_rows(file_path, file_format, "all"):
                        if retrieval_metrics:
                            sample = _build_sample(row, field_mapping)
                            if sample is not None:
                                retrieval_samples.append(sample)
                        if deepeval_metrics:
                            llm_sample = _build_llm_sample(row, field_mapping)
                            if llm_sample is not None:
                                llm_samples.append(llm_sample)

                        if sample_limit:
                            retrieval_done = not retrieval_metrics or len(retrieval_samples) >= sample_limit
                            llm_done = not deepeval_metrics or len(llm_samples) >= sample_limit
                            if retrieval_done and llm_done:
                                break

                if retrieval_metrics and not retrieval_samples:
                    raise ValueError(f"No valid retrieval samples found for dataset: {dataset_id}")
                if deepeval_metrics and not llm_samples:
                    raise ValueError(f"No valid deepeval samples found for dataset: {dataset_id}")

                dataset_info[dataset_id] = {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset.get("dataset_name") or dataset_id,
                }
                if retrieval_metrics:
                    retrieval_samples_by_dataset[dataset_id] = retrieval_samples
                if deepeval_metrics:
                    llm_samples_by_dataset[dataset_id] = llm_samples

            group_names = [
                group.get("group_name") or f"模型组 {idx + 1}"
                for idx, group in enumerate(model_groups)
            ]
            dataset_names = [info["dataset_name"] for info in dataset_info.values()]
            deep_evaluation_task_service.init_model_progress(
                task_id, group_names, dataset_names,
                preserve_completed=is_resume,
            )

            total_samples = 0
            completed_samples = 0
            for idx, group in enumerate(model_groups):
                group_name = group.get("group_name") or f"模型组 {idx + 1}"
                group_sample_count = 0
                if retrieval_metrics:
                    if group.get("embedding"):
                        group_sample_count += sum(len(s) for s in retrieval_samples_by_dataset.values())
                    if group.get("rerank"):
                        group_sample_count += sum(len(s) for s in retrieval_samples_by_dataset.values())
                if deepeval_metrics and group.get("llm"):
                    group_sample_count += sum(len(s) for s in llm_samples_by_dataset.values())
                total_samples += group_sample_count
                if group_name in completed_group_names:
                    completed_samples += group_sample_count

            if total_samples == 0:
                raise ValueError("没有可评估的样本")
            deep_evaluation_task_service.update_progress(task_id, completed_samples, total_samples)
            progress_tracker = _ProgressTracker(task_id, total_samples)
            progress_tracker.processed = completed_samples  # 从已完成的位置继续

            model_workers = _get_model_workers(task.get("worker_groups"), len(model_groups))
            chunk_eval_mode = (task.get("worker_groups") or {}).get("chunk_eval_mode", "batch")
            group_semaphore = asyncio.Semaphore(model_workers)

            merged_stats: Dict[str, Dict[str, float]] = {}
            group_details: Dict[str, Any] = {}

            def _merge_metric(metric_name: str, stat: Dict[str, Any]) -> None:
                mean = stat.get("mean")
                count = stat.get("count") or 0
                if mean is None or count <= 0:
                    return
                entry = merged_stats.setdefault(metric_name, {
                    "sum": 0.0,
                    "count": 0.0,
                    "min": None,
                    "max": None,
                })
                entry["sum"] += float(mean) * float(count)
                entry["count"] += float(count)
                min_val = stat.get("min")
                max_val = stat.get("max")
                if min_val is not None:
                    entry["min"] = min_val if entry["min"] is None else min(entry["min"], min_val)
                if max_val is not None:
                    entry["max"] = max_val if entry["max"] is None else max(entry["max"], max_val)

            def _merge_group_metrics(group_result: Dict[str, Any]) -> None:
                retrieval = group_result.get("retrieval") or {}
                for model_type in ("embedding", "rerank"):
                    model_data = retrieval.get(model_type) or {}
                    summary = model_data.get("summary") or {}
                    for dataset_data in (summary.get("datasets") or {}).values():
                        dataset_summary = dataset_data.get("summary") or {}
                        for metric_name, stat in (
                            dataset_summary.get("metrics") or {}
                        ).items():
                            _merge_metric(metric_name, stat)

                llm = group_result.get("llm") or {}
                for dataset_data in (llm.get("datasets") or {}).values():
                    dataset_summary = dataset_data.get("summary") or {}
                    for metric_name, stat in (
                        dataset_summary.get("metrics") or {}
                    ).items():
                        _merge_metric(metric_name, stat)

            # 恢复已完成组的结果到 group_details 和 merged_stats
            for gname in completed_group_names:
                if gname in existing_by_group:
                    group_details[gname] = existing_by_group[gname]
                    _merge_group_metrics(existing_by_group[gname])

            async def _evaluate_retrieval_model(
                model_cfg: Dict[str, Any],
                model_type: str,
            ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                resolved = _resolve_model_config(model_cfg, task.get("user_id"))
                resolved["model_type"] = model_type
                resolved["concurrency"] = model_cfg.get("concurrency") or (8 if model_type == "embedding" else 4)

                all_results: List[Dict[str, Any]] = []
                dataset_results: Dict[str, Any] = {}
                total_samples = 0
                total_skipped = 0
                for dataset_id, samples in retrieval_samples_by_dataset.items():
                    if is_cancelled(task_id):
                        raise asyncio.CancelledError()
                    results, skipped_count = await _evaluate_samples(
                        task_id,
                        model_type,
                        resolved,
                        samples,
                        retrieval_metrics,
                        resolved["concurrency"],
                        progress_tracker,
                        task.get("user_id"),
                    )
                    total_samples += len(samples)
                    total_skipped += skipped_count
                    all_results.extend(results)
                    dataset_summary = _summarize_results(results, retrieval_metrics)
                    dataset_summary.update(
                        {
                            "total_samples": len(samples),
                            "successful_samples": len(results),
                            "skipped_samples": skipped_count,
                        }
                    )
                    dataset_results[dataset_id] = {
                        "dataset_name": dataset_info[dataset_id]["dataset_name"],
                        "summary": dataset_summary,
                    }

                if not all_results:
                    raise RuntimeError(
                        f"All {total_samples} samples failed for {model_type} evaluation"
                    )

                overall_summary = _summarize_results(all_results, retrieval_metrics)
                return {
                    "overall": overall_summary.get("overall"),
                    "metrics": overall_summary.get("metrics"),
                    "datasets": dataset_results,
                    "total_samples": total_samples,
                    "successful_samples": len(all_results),
                    "skipped_samples": total_skipped,
                }, resolved

            async def _evaluate_llm_group(llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
                llm_config = dict(llm_cfg)
                if llm_config.get("config_id") and not llm_config.get("endpoint"):
                    resolved = _resolve_model_config(llm_config, task.get("user_id"))
                    llm_config["endpoint"] = resolved.get("endpoint")
                    llm_config["model"] = resolved.get("model_name") or resolved.get("model")
                    llm_config["api_key"] = resolved.get("api_key")
                    llm_config["config_name"] = resolved.get("config_name")

                llm_config["endpoint"] = validate_user_outbound_url(
                    llm_config.get("endpoint") or "",
                    task.get("user_id"),
                )
                llm_config["user_id"] = task.get("user_id")
                llm_concurrency = llm_config.get("concurrency", 2)
                llm_judge = create_llm_judge_from_dict(llm_config)
                evaluator = DeepEvaluator(
                    metrics=deepeval_metrics,
                    llm_client=llm_judge,
                    concurrency=llm_concurrency,
                    chunk_eval_mode=chunk_eval_mode,
                )

                datasets_summary: Dict[str, Any] = {}

                last_completed = 0
                expected_base = 0

                def progress_callback(completed: int, _total: int) -> None:
                    nonlocal last_completed, expected_base
                    # evaluate_batch 的 completed 是本次数据集内的局部计数
                    # （每数据集从 0 重计）——用 expected_base 换算绝对进度，
                    # 否则第二个数据集起 delta 恒为负、任务进度卡死。
                    absolute = expected_base + completed
                    delta = absolute - last_completed
                    last_completed = absolute
                    if delta > 0:
                        asyncio.create_task(progress_tracker.increment(delta))

                async with llm_judge:
                    for dataset_id, samples in llm_samples_by_dataset.items():
                        if is_cancelled(task_id):
                            raise asyncio.CancelledError()
                        batch_result = await evaluator.evaluate_batch(
                            samples,
                            deepeval_metrics,
                            progress_callback=progress_callback,
                        )
                        # 本数据集完成：累计进 expected_base，供下一个数据集的
                        # 进度回调换算绝对进度
                        expected_base += len(samples)
                        summary = batch_result.summary
                        datasets_summary[dataset_id] = {
                            "dataset_name": dataset_info[dataset_id]["dataset_name"],
                            "summary": summary,
                        }

                return {
                    "datasets": datasets_summary,
                }

            async def run_group(group_cfg: Dict[str, Any], index: int) -> Tuple[str, Dict[str, Any]]:
                async with group_semaphore:
                    group_name = group_cfg.get("group_name") or f"模型组 {index + 1}"

                    # 跳过已完成的组 (resume)
                    if group_name in completed_group_names:
                        logger.info("Skipping completed group: %s (resume)", group_name)
                        return group_name, group_details.get(group_name, {"retrieval": {}, "llm": {}})

                    group_result: Dict[str, Any] = {"retrieval": {}, "llm": {}}
                    dataset_ids = list(dataset_info.keys())
                    dataset_name_map = {ds_id: info["dataset_name"] for ds_id, info in dataset_info.items()}

                    def update_group_progress(progress: float, status: str) -> None:
                        for ds_id in dataset_ids:
                            deep_evaluation_task_service.update_model_progress(
                                task_id,
                                group_name,
                                dataset_name_map[ds_id],
                                progress,
                                status,
                            )

                    needs_retrieval = retrieval_metrics and (group_cfg.get("embedding") or group_cfg.get("rerank"))
                    needs_llm = deepeval_metrics and group_cfg.get("llm")

                    update_group_progress(0, "running")

                    try:
                        if needs_retrieval:
                            if group_cfg.get("embedding"):
                                summary, resolved = await _evaluate_retrieval_model(group_cfg["embedding"], "embedding")
                                group_result["retrieval"]["embedding"] = {
                                    "model": {
                                        "config_id": resolved.get("config_id"),
                                        "config_name": resolved.get("config_name"),
                                        "endpoint": resolved.get("endpoint"),
                                        "model_name": resolved.get("model_name") or resolved.get("model"),
                                    },
                                    "summary": summary,
                                }
                            if group_cfg.get("rerank"):
                                summary, resolved = await _evaluate_retrieval_model(group_cfg["rerank"], "rerank")
                                group_result["retrieval"]["rerank"] = {
                                    "model": {
                                        "config_id": resolved.get("config_id"),
                                        "config_name": resolved.get("config_name"),
                                        "endpoint": resolved.get("endpoint"),
                                        "model_name": resolved.get("model_name") or resolved.get("model"),
                                    },
                                    "summary": summary,
                                }

                            if needs_llm:
                                update_group_progress(50, "running")
                            else:
                                update_group_progress(100, "completed")

                        if needs_llm:
                            group_result["llm"] = await _evaluate_llm_group(group_cfg["llm"])
                            update_group_progress(100, "completed")

                        return group_name, group_result
                    except Exception as group_error:
                        update_group_progress(100, "failed")
                        # 单组失败不摧毁整个任务（也不取消其他健康组）：记录错误
                        # 并返回，由结果的 _error 标记呈现（与 MTEB 的 dataset
                        # 级容错语义一致）。
                        logger.warning(
                            "Deep-eval group %r failed: %s", group_name, group_error
                        )
                        return group_name, {"_error": str(group_error)}

            group_tasks = [
                asyncio.create_task(run_group(group_cfg, idx))
                for idx, group_cfg in enumerate(model_groups)
            ]

            try:
                for coro in asyncio.as_completed(group_tasks):
                    group_name, result = await coro
                    group_details[group_name] = result
                    if (
                        group_name not in completed_group_names
                        and _is_successful_group_result(result)
                    ):
                        _merge_group_metrics(result)
                    if is_cancelled(task_id):
                        raise asyncio.CancelledError()
            except asyncio.CancelledError:
                for task_item in group_tasks:
                    task_item.cancel()
                raise

            if not any(
                _is_successful_group_result(result)
                for result in group_details.values()
            ):
                raise RuntimeError("All deep evaluation model groups failed")

            metrics_summary = {}
            for name, stats in merged_stats.items():
                if stats["count"] <= 0:
                    continue
                metrics_summary[name] = {
                    "mean": stats["sum"] / stats["count"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "count": stats["count"],
                }

            overall_summary = None
            if metrics_summary:
                means = [stat["mean"] for stat in metrics_summary.values()]
                overall_summary = {
                    "mean": sum(means) / len(means),
                    "min": min(means),
                    "max": max(means),
                }

            summary = {
                "overall": overall_summary,
                "metrics": metrics_summary,
                "by_group": group_details,
            }

            output_dir = get_settings().output_dir / "deep_evaluations"
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / f"{task_id}.json"

            def _strip_api_key(model_cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
                if not isinstance(model_cfg, dict):
                    return None
                sanitized = dict(model_cfg)
                sanitized.pop("api_key", None)
                return sanitized

            safe_model_groups = []
            for group in model_groups:
                safe_model_groups.append({
                    "group_name": group.get("group_name"),
                    "embedding": _strip_api_key(group.get("embedding")),
                    "rerank": _strip_api_key(group.get("rerank")),
                    "llm": _strip_api_key(group.get("llm")),
                })

            payload: Dict[str, Any] = {
                "task_id": task_id,
                "metrics": metrics,
                "field_mapping": field_mapping,
                "worker_groups": {"model_workers": model_workers},
                "model_groups": safe_model_groups,
                "datasets": list(dataset_info.values()),
                "results": group_details,
            }

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            if is_cancelled(task_id):
                deep_evaluation_task_service.cancel_task(task_id)
                return

            completed = deep_evaluation_task_service.complete_task(
                task_id,
                "completed",
                results=summary,
                report_path=str(results_path),
            )
            if not completed:
                logger.info(
                    "Discarding completion for inactive deep evaluation task %s",
                    task_id,
                )
                return

        except asyncio.CancelledError:
            # 保存已完成组的结果，供下次 resume 使用
            try:
                if group_details:
                    _save_partial_results(task_id, group_details, merged_stats)
            except NameError:
                pass
            deep_evaluation_task_service.cancel_task(task_id)
            logger.info("Deep evaluation task cancelled: %s", task_id)
        except Exception as exc:
            try:
                if group_details:
                    _save_partial_results(task_id, group_details, merged_stats)
            except NameError:
                pass
            logger.exception("Deep evaluation task failed: %s", task_id)
            completed = deep_evaluation_task_service.complete_task(
                task_id,
                "failed",
                error_message=str(exc),
            )
            if not completed:
                logger.info(
                    "Discarding failure for inactive deep evaluation task %s",
                    task_id,
                )
        finally:
            _cleanup_cancelled(task_id)

    asyncio.run(_run())
