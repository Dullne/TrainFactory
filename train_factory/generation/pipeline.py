"""
数据生成流水线

Worker Pool 模式并发处理文档。
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from ..storage.services.outbound_endpoint_policy import (
    create_pinned_async_client,
    validate_user_outbound_url,
)
from .security import resolve_generation_input_path
from .processor import DocumentProcessor, ProcessorConfig
from .steps.base import Document, GeneratedSample
from .steps.embedding_filter_step import EmbeddingFilterStep
from .clients import LLMClient, LLMClientPool, LLMConfig
from .clients.embedding_client import EmbeddingClient, EmbeddingConfig
from .clients.milvus_client import MilvusClient, MilvusConfig
from .clients.rerank_client import RerankClient, RerankConfig
from ..deep_evaluation.metrics.contextual_precision import ContextualPrecisionMetric
from ..deep_evaluation.metrics.base import EvaluationSample

logger = logging.getLogger(__name__)

MAX_GENERATION_LLM_ENDPOINTS = 16
MAX_GENERATION_LLM_CONCURRENCY = 50
MAX_GENERATION_EMBEDDING_CONCURRENCY = 100
MAX_GENERATION_RERANK_CONCURRENCY = 50
MAX_GENERATION_BATCH_SIZE = 256
MAX_GENERATION_LLM_TOKENS = 131_072
MAX_GENERATION_TIMEOUT = 600
MAX_GENERATION_RETRIES = 10
MAX_GENERATION_RETRIEVAL_TOP_K = 100
MAX_GENERATION_CUSTOM_PROMPTS = 32
MAX_GENERATION_PROMPT_CHARS = 100_000


def generation_attempt_key(task_id: str, run_token: str) -> str:
    """Return an opaque, stable directory key for one generation attempt."""
    return hashlib.sha256(
        f"{task_id}\x1f{run_token}".encode("utf-8")
    ).hexdigest()[:24]


def resolve_generation_attempt_output_path(
    output_path: str,
    task_id: Optional[str],
    run_token: Optional[str],
) -> Path:
    """Scope a generation output path to an immutable opaque attempt."""
    path = Path(output_path)
    if not task_id or not run_token:
        return path
    attempt_key = generation_attempt_key(task_id, run_token)
    attempt_parent = path.parent / f"generation_{task_id}" / attempt_key
    if (
        path.parent.name == attempt_key
        and path.parent.parent.name == f"generation_{task_id}"
    ):
        return path
    return attempt_parent / path.name


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 输入配置
    input_path: str
    input_format: str = "auto"  # auto, jsonl, json, txt
    content_field: Optional[str] = None  # 自定义内容字段名
    generation_mode: str = "doc_to_training"  # doc_to_training, qa_to_training, qa_extraction
    pos_neg_method: str = "retrieval"  # retrieval(向量检索+评估分类) 或 llm(纯LLM生成)

    # 输出配置
    output_path: Optional[str] = None
    output_format: str = "universal"

    # LLM 配置
    llm_config: Dict[str, Any] = field(default_factory=dict)
    eval_llm_config: Optional[Dict[str, Any]] = None  # 独立评估 LLM（可选，为空时复用 llm_config）

    # Worker 配置
    llm_concurrency: int = 10
    embedding_concurrency: int = 20
    timeout_per_doc: int = 300

    # 步骤配置
    steps: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 后处理配置
    post_process: Optional[Dict[str, Any]] = None

    # 自定义 Prompt
    custom_prompts: Optional[Dict[str, str]] = None

    # Embedding + Milvus 配置
    embedding_config: Optional[Dict[str, Any]] = None
    source_dataset_id: Optional[str] = None
    milvus_config: Optional[Dict[str, Any]] = None

    # Rerank 配置（可选，用于 pos_neg_method=retrieval 时重排序检索候选）
    rerank_config: Optional[Dict[str, Any]] = None

    # Embedding 检索参数
    similarity_threshold: float = 0.85
    retrieval_top_k: int = 10

    # 去重：已有的筛选结果（跳过 embedding 筛选时使用）
    skip_filter: bool = False
    previous_filter_stats: Optional[Dict[str, Any]] = None

    # 断点续传：跳过 QA 提取阶段，直接从已有 QA 文件开始
    resume_qa_path: Optional[str] = None  # 已提取的 QA 文件路径
    resume_qa_filtered_path: Optional[str] = None  # 已筛选的 QA 文件路径

    # 任务 ID（用于确定性文件命名，支持增量写入续传）
    task_id: Optional[str] = None

    # Durable execution-attempt fence.  API workers provide both fields; direct
    # library callers may omit them for backward compatibility.
    run_token: Optional[str] = None
    attempt_is_valid: Optional[Callable[[], bool]] = None

    # A restart may read an earlier attempt as a checkpoint, but all writes are
    # always resolved beneath the current attempt directory.
    resume_output_path: Optional[str] = None

    # 使用已有集合（不自动生成集合名称）
    existing_collection_name: Optional[str] = None

    # API/worker-provided local path capability. Direct library callers that do
    # not read user-controlled paths can leave this unset for compatibility.
    allowed_input_dirs: Optional[List[str]] = None
    user_id: Optional[str] = None


def _require_generation_integer(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Generation {name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Generation {name} must be between {minimum} and {maximum}"
        )
    return value


def validate_generation_step_limits(steps: Any) -> None:
    """Bound step settings that multiply model calls or generated records."""
    if not isinstance(steps, dict):
        raise ValueError("Generation steps must be an object")

    limits = {
        "keypoint_gen": {"max_keypoints": (1, 10)},
        "qa_gen": {
            "num_qa_per_doc": (1, 10),
            "roles_per_doc": (1, 10),
        },
        "role_gen": {"roles_per_doc": (1, 10)},
        "pos_neg_extraction": {
            "num_positive": (1, 20),
            "num_negative": (1, 50),
            "roles_per_doc": (1, 10),
        },
    }
    for step_name, field_limits in limits.items():
        step_config = steps.get(step_name)
        if step_config is None:
            continue
        if not isinstance(step_config, dict):
            raise ValueError(f"Generation step {step_name} must be an object")
        for field_name, (minimum, maximum) in field_limits.items():
            if field_name in step_config:
                _require_generation_integer(
                    step_config[field_name],
                    f"steps.{step_name}.{field_name}",
                    minimum,
                    maximum,
                )


def validate_generation_custom_prompts(prompts: Any) -> None:
    if prompts is None:
        return
    if not isinstance(prompts, dict) or len(prompts) > MAX_GENERATION_CUSTOM_PROMPTS:
        raise ValueError(
            "Generation custom prompts must be an object with at most "
            f"{MAX_GENERATION_CUSTOM_PROMPTS} entries"
        )
    for name, prompt in prompts.items():
        if not isinstance(name, str) or not isinstance(prompt, str):
            raise ValueError("Generation custom prompt names and values must be strings")
        if len(name) > 128 or len(prompt) > MAX_GENERATION_PROMPT_CHARS:
            raise ValueError("Generation custom prompt exceeds the configured size limit")


def _validate_generation_llm_config(
    config: Any,
    *,
    name: str,
    default_concurrency: int,
) -> None:
    if not config:
        return
    if not isinstance(config, dict):
        raise ValueError(f"Generation {name} must be an object")

    endpoints = config.get("endpoints")
    if endpoints is not None:
        if not isinstance(endpoints, list) or not (
            1 <= len(endpoints) <= MAX_GENERATION_LLM_ENDPOINTS
        ):
            raise ValueError(
                f"Generation {name}.endpoints must contain 1-"
                f"{MAX_GENERATION_LLM_ENDPOINTS} entries"
            )

    _require_generation_integer(
        config.get("concurrency", default_concurrency),
        f"{name}.concurrency",
        1,
        MAX_GENERATION_LLM_CONCURRENCY,
    )
    _require_generation_integer(
        config.get("max_tokens", 2048),
        f"{name}.max_tokens",
        1,
        MAX_GENERATION_LLM_TOKENS,
    )
    _require_generation_integer(
        config.get("timeout", 60),
        f"{name}.timeout",
        1,
        MAX_GENERATION_TIMEOUT,
    )
    _require_generation_integer(
        config.get("max_retries", 3),
        f"{name}.max_retries",
        0,
        MAX_GENERATION_RETRIES,
    )


def validate_generation_resource_config(config: PipelineConfig) -> None:
    """Reject generation work with excessive concurrency or output fan-out."""
    _require_generation_integer(
        config.llm_concurrency,
        "llm_concurrency",
        1,
        MAX_GENERATION_LLM_CONCURRENCY,
    )
    _require_generation_integer(
        config.embedding_concurrency,
        "embedding_concurrency",
        1,
        MAX_GENERATION_EMBEDDING_CONCURRENCY,
    )
    _require_generation_integer(
        config.timeout_per_doc,
        "timeout_per_doc",
        1,
        MAX_GENERATION_TIMEOUT,
    )
    _require_generation_integer(
        config.retrieval_top_k,
        "retrieval_top_k",
        1,
        MAX_GENERATION_RETRIEVAL_TOP_K,
    )

    _validate_generation_llm_config(
        config.llm_config,
        name="llm_config",
        default_concurrency=config.llm_concurrency,
    )
    _validate_generation_llm_config(
        config.eval_llm_config,
        name="eval_llm_config",
        default_concurrency=config.llm_concurrency,
    )

    embedding_config = config.embedding_config
    if embedding_config is not None:
        if not isinstance(embedding_config, dict):
            raise ValueError("Generation embedding_config must be an object")
        _require_generation_integer(
            embedding_config.get("batch_size", 32),
            "embedding_config.batch_size",
            1,
            MAX_GENERATION_BATCH_SIZE,
        )
        _require_generation_integer(
            embedding_config.get("concurrency", config.embedding_concurrency),
            "embedding_config.concurrency",
            1,
            MAX_GENERATION_EMBEDDING_CONCURRENCY,
        )
        _require_generation_integer(
            embedding_config.get("retrieval_top_k", config.retrieval_top_k),
            "embedding_config.retrieval_top_k",
            1,
            MAX_GENERATION_RETRIEVAL_TOP_K,
        )

    rerank_config = config.rerank_config
    if rerank_config is not None:
        if not isinstance(rerank_config, dict):
            raise ValueError("Generation rerank_config must be an object")
        _require_generation_integer(
            rerank_config.get("top_k", 10),
            "rerank_config.top_k",
            1,
            MAX_GENERATION_RETRIEVAL_TOP_K,
        )
        _require_generation_integer(
            rerank_config.get("batch_size", 64),
            "rerank_config.batch_size",
            1,
            MAX_GENERATION_BATCH_SIZE,
        )
        _require_generation_integer(
            rerank_config.get("concurrency", 10),
            "rerank_config.concurrency",
            1,
            MAX_GENERATION_RERANK_CONCURRENCY,
        )

    validate_generation_step_limits(config.steps)
    validate_generation_custom_prompts(config.custom_prompts)


@dataclass
class PipelineResult:
    """流水线执行结果"""
    success: bool
    total_docs: int = 0
    processed_docs: int = 0
    output_samples: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def _apply_batch_failure_status(
    result: PipelineResult,
    batch_failures: Dict[str, Any],
) -> PipelineResult:
    """Make partial batch failures visible to task status and retry handling."""
    failed_batch_count = int(batch_failures.get("batch_count") or 0)
    if not failed_batch_count:
        return result

    failed_record_count = int(batch_failures.get("record_count") or 0)
    result.success = False
    result.error = (
        f"{failed_batch_count} generation batch(es) failed "
        f"({failed_record_count} record(s)); partial output was preserved for retry"
    )
    result.details.setdefault("filter_stats", {})["batch_failures"] = batch_failures
    return result


def _new_batch_failure_stats() -> Dict[str, Any]:
    return {
        "batch_count": 0,
        "record_count": 0,
        "errors": [],
    }


def _record_batch_failure(
    batch_failures: Dict[str, Any],
    batch: List[Dict[str, Any]],
    batch_error: Exception,
    log_context: str,
) -> None:
    """Record one isolated batch failure while allowing the consumer to drain."""
    batch_failures["batch_count"] += 1
    batch_failures["record_count"] += len(batch)
    if len(batch_failures["errors"]) < 20:
        batch_failures["errors"].append(str(batch_error)[:500])
    logger.exception("%s: batch failed, skipping batch", log_context)


class _ValidatedOutboundHTTPClientMixin:
    """Bind every model client to one user-scoped, DNS-pinned destination."""

    _generation_user_id: Optional[str]

    def _new_http_client(self) -> httpx.AsyncClient:
        return create_pinned_async_client(
            self.config.endpoint,
            self._generation_user_id,
            timeout=self.config.timeout,
        )

    async def __aenter__(self):
        self._client = self._new_http_client()
        return self

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._new_http_client()
        return self._client


class _ValidatedLLMClient(_ValidatedOutboundHTTPClientMixin, LLMClient):
    def __init__(self, config: LLMConfig, user_id: Optional[str]) -> None:
        LLMClient.__init__(self, config)
        self._generation_user_id = user_id
        self.config.user_id = user_id


class _ValidatedEmbeddingClient(_ValidatedOutboundHTTPClientMixin, EmbeddingClient):
    def __init__(self, config: EmbeddingConfig, user_id: Optional[str]) -> None:
        EmbeddingClient.__init__(self, config)
        self._generation_user_id = user_id
        self.config.user_id = user_id


class _ValidatedRerankClient(_ValidatedOutboundHTTPClientMixin, RerankClient):
    def __init__(self, config: RerankConfig, user_id: Optional[str]) -> None:
        RerankClient.__init__(self, config)
        self._generation_user_id = user_id
        self.config.user_id = user_id


class _ValidatedLLMClientPool(LLMClientPool):
    """LLM pool whose clients are pinned at network-client creation."""

    def __init__(
        self,
        configs: List[LLMConfig],
        max_concurrency: int,
        user_id: Optional[str],
    ) -> None:
        super().__init__(configs, max_concurrency=max_concurrency)
        self._generation_user_id = user_id
        self.clients = [
            _ValidatedLLMClient(config, user_id)
            for config in configs
        ]

class DatasetGenerationPipeline:
    """
    数据集生成流水线

    使用 Worker Pool 模式并发处理文档。
    """

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        qa_phase_callback: Optional[Callable[[str, Optional[str]], None]] = None,
        on_qa_complete: Optional[Callable[[str, int], None]] = None,
    ):
        """
        初始化流水线

        Args:
            config: 流水线配置
            progress_callback: 进度回调 (processed, total)
            qa_phase_callback: QA 阶段完成回调 (qa_output_path, qa_filtered_path)
                              用于在 Phase 1 完成后立即持久化中间产物路径
            on_qa_complete: QA 提取完成回调 (qa_output_path, qa_count)
                           在 qa_producer 完成后立即触发，用于提前注册 QA 数据集
        """
        self.config = config
        self.progress_callback = progress_callback
        self.qa_phase_callback = qa_phase_callback
        self.on_qa_complete = on_qa_complete
        self._stop_flag = False

    @property
    def _stop_flag(self) -> bool:
        """Combine local cancellation with the durable attempt predicate.

        Keeping this as a property makes every existing producer/consumer
        cancellation checkpoint also reject an obsolete run token.  Predicate
        failures are fail-closed because publishing an old attempt is less safe
        than stopping it.
        """
        if getattr(self, "_local_stop_requested", False):
            return True
        predicate = getattr(self.config, "attempt_is_valid", None)
        if predicate is None:
            return False
        try:
            return not bool(predicate())
        except Exception:
            logger.exception("Generation attempt validity check failed")
            return True

    @_stop_flag.setter
    def _stop_flag(self, value: bool) -> None:
        self._local_stop_requested = bool(value)

    async def run(self) -> PipelineResult:
        """
        运行流水线

        Returns:
            PipelineResult: 执行结果
        """
        # A stop may be requested after the worker owns this pipeline but before
        # its coroutine is scheduled. Never erase that latched cancellation.
        if self._stop_flag:
            return self._stopped_result()

        # 独立流程分发
        if self.config.generation_mode == "doc_to_training":
            return await self._run_doc_to_training()
        elif self.config.generation_mode == "qa_to_training":
            return await self._run_qa_to_training()
        elif self.config.generation_mode == "doc_to_eval":
            return await self._run_doc_to_eval()
        elif self.config.generation_mode == "qa_to_eval":
            return await self._run_qa_to_eval()

        try:
            # 1. 加载文档
            documents = self._load_documents()
            total_docs = len(documents)

            if total_docs == 0:
                return PipelineResult(
                    success=False,
                    error="No documents found",
                )

            logger.info(f"Loaded {total_docs} documents")

            # 2. 创建 LLM 客户端
            llm_client = self._create_llm_client()

            # 3. 处理文档
            async with llm_client:
                if self._stop_flag:
                    return self._stopped_result(total_docs)
                samples = await self._process_documents(documents, llm_client)

            if self._stop_flag:
                return PipelineResult(
                    success=False,
                    total_docs=total_docs,
                    processed_docs=0,
                    error="Pipeline stopped",
                )

            # 4. 后处理 & 保存
            if self.config.post_process:
                original_count = len(samples)
                samples = await self._post_process(samples)
                if self._stop_flag:
                    return self._stopped_result(total_docs)
                # Rewrite file only if post-processing changed the data
                if len(samples) != original_count:
                    output_path = self._save_output(samples)
                else:
                    output_path = str(getattr(self, "_incremental_output_path", ""))
                    if not output_path:
                        output_path = self._save_output(samples)
            else:
                # No post-processing: incremental file is the final output
                output_path = str(getattr(self, "_incremental_output_path", ""))
                if not output_path:
                    output_path = self._save_output(samples)

            return PipelineResult(
                success=True,
                total_docs=total_docs,
                processed_docs=total_docs,
                output_samples=len(samples),
                output_path=output_path,
            )

        except Exception as e:
            logger.exception("Pipeline execution failed")
            return PipelineResult(
                success=False,
                error=str(e),
            )

    def stop(self):
        """停止流水线"""
        self._stop_flag = True

    def _stopped_result(self, total_docs: int = 0) -> PipelineResult:
        return PipelineResult(
            success=False,
            total_docs=total_docs,
            processed_docs=0,
            error="Pipeline stopped",
        )

    async def _put_queue_data_unless_stopped(
        self,
        queue: asyncio.Queue,
        item: Any,
    ) -> bool:
        """Publish data without letting a bounded put cross a stop fence."""
        while not self._stop_flag:
            try:
                queue.put_nowait(item)
                return True
            except asyncio.QueueFull:
                await asyncio.sleep(0)
        return False

    async def _notify_qa_phase(
        self,
        qa_output_path: str,
        qa_filtered_path: Optional[str],
    ) -> None:
        if self.qa_phase_callback is None or self._stop_flag:
            return
        callback_result = self.qa_phase_callback(
            qa_output_path,
            qa_filtered_path,
        )
        if inspect.isawaitable(callback_result):
            await callback_result

    def _resolve_input_path_for_read(self, path: Path) -> Path:
        """Revalidate a configured input path immediately before opening it."""
        if self.config.allowed_input_dirs is None:
            return path
        return Path(
            resolve_generation_input_path(
                str(path),
                allowed_dirs=self.config.allowed_input_dirs,
            )
        )

    def _load_documents(self) -> List[Document]:
        """加载文档"""
        input_path = self._resolve_input_path_for_read(Path(self.config.input_path))
        documents = []

        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        file_format = self.config.input_format

        # 加载文件
        if input_path.is_file():
            if file_format == "auto":
                file_format = self._detect_format(input_path)
            documents = self._load_file(input_path, file_format)
        else:
            if file_format == "auto":
                for suffix in ("jsonl", "json", "txt"):
                    for file_path in input_path.glob(f"*.{suffix}"):
                        documents.extend(self._load_file(file_path, suffix))
            else:
                for file_path in input_path.glob(f"*.{file_format}"):
                    documents.extend(self._load_file(file_path, file_format))

        return documents

    def _detect_format(self, path: Path) -> str:
        """检测文件格式"""
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".jsonl":
                return "jsonl"
            elif suffix == ".json":
                return "json"
            elif suffix == ".txt":
                return "txt"
        return "jsonl"  # 默认

    def _load_file(self, file_path: Path, file_format: str) -> List[Document]:
        """加载单个文件"""
        file_path = self._resolve_input_path_for_read(file_path)
        documents = []

        try:
            if file_format == "jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        fallback_doc_id = f"{file_path.stem}_{i}"
                        doc_id = self._resolve_doc_id(data, fallback_doc_id)
                        doc = self._data_to_document(data, doc_id)
                        if doc:
                            documents.append(doc)

            elif file_format == "json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            fallback_doc_id = f"{file_path.stem}_{i}"
                            doc_id = self._resolve_doc_id(item, fallback_doc_id)
                            doc = self._data_to_document(item, doc_id)
                            if doc:
                                documents.append(doc)
                    else:
                        doc_id = self._resolve_doc_id(data, file_path.stem)
                        doc = self._data_to_document(data, doc_id)
                        if doc:
                            documents.append(doc)

            elif file_format == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(Document(
                        content=content,
                        doc_id=file_path.stem,
                    ))

        except Exception as e:
            logger.warning(f"Failed to load file {file_path}: {e}")

        return documents

    @staticmethod
    def _resolve_doc_id(data: Dict[str, Any], fallback_doc_id: str) -> str:
        """Resolve deterministic doc_id from payload when available."""
        if not isinstance(data, dict):
            return fallback_doc_id

        candidate = data.get("chunk_id")
        if isinstance(candidate, str) and candidate.strip():
            return DatasetGenerationPipeline._sanitize_doc_id(candidate.strip(), fallback_doc_id)

        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            for key in ("chunk_id", "source_doc_id", "doc_id", "external_id"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return DatasetGenerationPipeline._sanitize_doc_id(value.strip(), fallback_doc_id)

        return fallback_doc_id

    @staticmethod
    def _sanitize_doc_id(doc_id: str, fallback: str) -> str:
        """截断过长的 doc_id 并移除空字节（#9: 防止下游路径/查询注入）。"""
        doc_id = doc_id.replace("\x00", "")
        if len(doc_id) > 512:
            doc_id = doc_id[:512]
        return doc_id if doc_id else fallback

    def _data_to_document(self, data: Dict[str, Any], doc_id: str) -> Optional[Document]:
        """将数据字典转换为 Document"""
        # 优先使用配置的 content_field，否则尝试多种默认字段名
        if self.config.content_field:
            content = data.get(self.config.content_field, "")
        else:
            content = (
                data.get("content") or
                data.get("text") or
                data.get("passage") or
                data.get("document") or
                ""
            )

        if not content and self.config.generation_mode == "qa_to_training":
            # qa_to_training / qa_based 内部模式：content 可以为空，但需要 query 和 answer
            pass
        elif not content:
            return None

        return Document(
            content=content,
            metadata=data,
            doc_id=doc_id,
        )

    def _create_llm_client(self, llm_config: Optional[Dict[str, Any]] = None) -> LLMClientPool:
        """创建 LLM 客户端

        Args:
            llm_config: LLM 配置字典，为空时使用 self.config.llm_config
        """
        llm_config = llm_config or self.config.llm_config

        # 支持多端点配置
        if "endpoints" in llm_config:
            configs = [
                LLMConfig(
                    endpoint=validate_user_outbound_url(
                        ep["url"],
                        self.config.user_id,
                    ),
                    model=ep.get("model", llm_config.get("model", "")),
                    api_key=ep.get("api_key"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 2048),
                    timeout=llm_config.get("timeout", 60),
                    max_retries=llm_config.get("max_retries", 3),
                )
                for ep in llm_config["endpoints"]
            ]
        else:
            configs = [
                LLMConfig(
                    endpoint=validate_user_outbound_url(
                        llm_config.get("endpoint", ""),
                        self.config.user_id,
                    ),
                    model=llm_config.get("model", ""),
                    api_key=llm_config.get("api_key"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 2048),
                    timeout=llm_config.get("timeout", 60),
                    max_retries=llm_config.get("max_retries", 3),
                )
            ]

        concurrency = llm_config.get("concurrency", self.config.llm_concurrency)
        return _ValidatedLLMClientPool(
            configs,
            max_concurrency=concurrency,
            user_id=self.config.user_id,
        )

    async def _process_documents(
        self,
        documents: List[Document],
        llm_client: LLMClientPool,
    ) -> List[GeneratedSample]:
        """并发处理文档，处理完一个即追加写入输出文件"""
        semaphore = asyncio.Semaphore(self.config.llm_concurrency)
        all_samples = []
        processed = 0
        total = len(documents)

        # Prepare incremental output file (append mode)
        output_path = self._resolve_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate if exists (new run)
        output_path.write_text("", encoding="utf-8")
        self._incremental_output_path = output_path
        self._incremental_lock = asyncio.Lock()

        async def process_one(doc: Document) -> List[GeneratedSample]:
            nonlocal processed
            if self._stop_flag:
                return []

            async with semaphore:
                if self._stop_flag:
                    return []
                try:
                    processor = DocumentProcessor(
                        ProcessorConfig(
                            generation_mode=self.config.generation_mode,
                            steps=self.config.steps,
                            custom_prompts=self.config.custom_prompts,
                        ),
                        llm_client=llm_client,
                    )
                    samples = await asyncio.wait_for(
                        processor.process(doc),
                        timeout=self.config.timeout_per_doc,
                    )
                    if self._stop_flag:
                        return []
                    # Incrementally append to output file for live preview
                    if samples:
                        formatted = self._format_samples(samples)
                        async with self._incremental_lock:
                            if self._stop_flag:
                                return []
                            with open(output_path, "a", encoding="utf-8") as f:
                                for item in formatted:
                                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    return samples
                except asyncio.TimeoutError:
                    logger.warning(f"Document {doc.doc_id} processing timeout")
                    return []
                except Exception as e:
                    logger.warning(f"Document {doc.doc_id} processing failed: {e}")
                    return []
                finally:
                    processed += 1
                    if self.progress_callback and not self._stop_flag:
                        self.progress_callback(processed, total)

        # 并发执行
        tasks = [process_one(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        if self._stop_flag:
            return []

        for samples in results:
            all_samples.extend(samples)

        return all_samples

    async def _post_process(self, samples: List[GeneratedSample]) -> List[GeneratedSample]:
        """后处理：按 post_process 配置执行已启用的处理器。"""
        if not samples or not self.config.post_process:
            return samples

        from .post_process import DedupProcessor

        dedup_cfg = self.config.post_process.get("dedup")
        if dedup_cfg and dedup_cfg.get("enabled", False):
            result = await DedupProcessor(dedup_cfg).process(samples)
            if self._stop_flag:
                return []
            if result.removed_count > 0:
                logger.info("后处理去重: %d → %d (去除 %d 条重复)",
                            len(samples), len(result.samples), result.removed_count)
            samples = result.samples

        return samples

    def _is_dedup_enabled(self) -> bool:
        """检查去重是否启用。"""
        if not self.config.post_process:
            return False
        dedup_cfg = self.config.post_process.get("dedup")
        return bool(dedup_cfg and dedup_cfg.get("enabled", False))

    @staticmethod
    def _qa_dedup_hash(record: Dict[str, Any]) -> str:
        """QA 去重指纹：(query, answer, chunk_id)。

        所有 QA 去重点（流式生产、resume 载入、批量去重）共用此指纹，避免仅按
        query 去重而误删「同一 query 来自不同 chunk/答案」的合法样本。
        """
        fingerprint = "\x1f".join([
            (record.get("query") or "").lower().strip(),
            (record.get("answer") or "").lower().strip(),
            record.get("chunk_id") or "",
        ])
        return hashlib.md5(fingerprint.encode()).hexdigest()

    @staticmethod
    def _qa_checkpoint_key(record: Dict[str, Any]) -> str:
        """Build the QA fingerprint from either pipeline or universal output data."""
        metadata = record.get("metadata") or {}
        chunk_id = record.get("chunk_id") or metadata.get("source_doc_id") or ""
        return DatasetGenerationPipeline._qa_dedup_hash({
            "query": record.get("query", ""),
            "answer": record.get("answer", ""),
            "chunk_id": chunk_id,
        })

    @staticmethod
    def _dedup_qa_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对 QA 记录按 (query, answer, chunk_id) 哈希去重（见 _qa_dedup_hash）。"""
        seen: set = set()
        deduped = []
        for r in records:
            h = DatasetGenerationPipeline._qa_dedup_hash(r)
            if h not in seen:
                seen.add(h)
                deduped.append(r)
        return deduped

    def _resolve_output_path(self) -> Path:
        """Resolve and return the output file path."""
        if self.config.output_path:
            return resolve_generation_attempt_output_path(
                self.config.output_path,
                self.config.task_id,
                self.config.run_token,
            )
        output_dir = Path(self.config.input_path).parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return resolve_generation_attempt_output_path(
            str(output_dir / f"generated_{uuid.uuid4().hex[:8]}.jsonl"),
            self.config.task_id,
            self.config.run_token,
        )

    def _save_output(self, samples: List[GeneratedSample]) -> str:
        """保存输出（后处理后重写，或使用增量写入的文件）"""
        # Use the incremental path if available (normal doc processing flow)
        output_path = getattr(self, "_incremental_output_path", None)
        if output_path is None:
            output_path = self._resolve_output_path()

        # 转换为输出格式
        output_data = self._format_output(samples)

        # Rewrite file (needed after dedup/post-processing removes some samples)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(output_data)} samples to {output_path}")
        return str(output_path)

    def _format_samples(self, samples: List[GeneratedSample]) -> List[Dict[str, Any]]:
        """Format a batch of samples for incremental output (same logic as _format_output)."""
        return self._format_output(samples)

    def _format_output(self, samples: List[GeneratedSample]) -> List[Dict[str, Any]]:
        """格式化输出"""
        # qa_extraction 模式使用专用格式
        if self.config.generation_mode == "qa_extraction":
            return self._format_output_qa(samples)

        output_format = self.config.output_format

        if output_format == "universal":
            # Universal 格式：每个样本一行，包含 query, positives, negatives
            return [
                {
                    "query": sample.query,
                    "answer": sample.answer,
                    "positives": sample.positive_chunks,
                    "negatives": sample.negative_chunks,
                    "metadata": {
                        "source_doc_id": sample.source_doc_id,
                        "role_config": sample.role_config.to_dict() if sample.role_config else None,
                    },
                }
                for sample in samples
            ]
        elif output_format == "triplet":
            # Triplet 格式：每个 (query, positive, negative) 组合一行
            output = []
            for sample in samples:
                for pos in sample.positive_chunks:
                    for neg in sample.negative_chunks:
                        output.append({
                            "query": sample.query,
                            "positive": pos,
                            "negative": neg,
                        })
            return output
        elif output_format == "pair":
            # Pair 格式：每个 (query, text, label) 一行
            output = []
            for sample in samples:
                for pos in sample.positive_chunks:
                    output.append({
                        "query": sample.query,
                        "text": pos,
                        "label": 1,
                    })
                for neg in sample.negative_chunks:
                    output.append({
                        "query": sample.query,
                        "text": neg,
                        "label": 0,
                    })
            return output
        else:
            # 默认：原始格式
            return [sample.to_dict() for sample in samples]

    def _format_output_qa(self, samples: List[GeneratedSample]) -> List[Dict[str, Any]]:
        """QA 提取模式输出格式（Phase 1）"""
        return [
            {
                "query": sample.query,
                "answer": sample.answer,
                "chunk_id": sample.source_doc_id,
                "chunk_content": sample.metadata.get("chunk_content", ""),
            }
            for sample in samples
        ]

    # ── 评估数据生成模式 ──────────────────────────────────────

    async def _generate_eval_data_batch(
        self,
        records: List[Dict[str, Any]],
        embedding_client: EmbeddingClient,
        milvus_client: MilvusClient,
        collection_name: str,
        rerank_client: Optional[RerankClient] = None,
    ) -> List[Dict[str, Any]]:
        """
        纯检索模式：为每条 QA 记录执行向量检索 + 可选 Rerank，
        收集 {query, expected_output, retrieval_context} 评估数据。
        不执行 ContextualPrecision 评估、不做 pos/neg 分类。
        """
        eval_records: List[Dict[str, Any]] = []
        retrieval_top_k = (self.config.embedding_config or {}).get("retrieval_top_k", 10)
        semaphore = asyncio.Semaphore(self.config.embedding_concurrency)
        processed = 0
        total = len(records)
        completed_keys: set = set()
        candidate_keys = {
            (r.get("query", ""), r.get("chunk_id", ""))
            for r in records
            if r.get("query")
        }

        # Query-level checkpoint resume for eval modes: reuse existing deep_eval file
        # and skip records already exported in previous run of the same task.
        deep_eval_path = self._get_deep_eval_path()
        if deep_eval_path.exists() and deep_eval_path.stat().st_size > 0:
            try:
                with open(deep_eval_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = (
                            obj.get("query", ""),
                            obj.get("source_chunk_id", ""),
                        )
                        if key[0] and key in candidate_keys:
                            completed_keys.add(key)
                            eval_records.append(obj)
                if completed_keys:
                    logger.info(
                        "[eval_data] Resuming from checkpoint: %d records already exported",
                        len(completed_keys),
                    )
            except Exception as e:
                logger.warning("[eval_data] Failed to load checkpoint: %s", e)
                completed_keys.clear()
                eval_records.clear()

        async def process_one(record: Dict[str, Any]):
            nonlocal processed
            if self._stop_flag:
                return
            key = (record.get("query", ""), record.get("chunk_id", ""))
            if key in completed_keys:
                processed += 1
                if self.progress_callback:
                    self.progress_callback(processed, total)
                return
            async with semaphore:
                if self._stop_flag:
                    return
                try:
                    query_emb = await embedding_client.embed([record["query"]])
                    if self._stop_flag:
                        return
                    search_results = milvus_client.search_similar(
                        collection_name,
                        query_emb,
                        top_k=retrieval_top_k,
                        exclude_chunk_ids=[record.get("chunk_id", "")],
                    )
                    candidates = search_results[0] if search_results else []
                    if not candidates:
                        return

                    contexts = [c["chunk_content"] for c in candidates]

                    if rerank_client:
                        try:
                            rerank_results = await rerank_client.rerank(record["query"], contexts)
                            if self._stop_flag:
                                return
                            rerank_cfg = self.config.rerank_config or {}
                            threshold = rerank_cfg.get("rerank_threshold", 1.0)
                            rerank_results = [r for r in rerank_results if r.score <= threshold]
                            contexts = [r.text for r in rerank_results]
                        except Exception as e:
                            logger.warning(f"[eval_data] Rerank failed: {e}, using embedding order")

                    if contexts:
                        eval_records.append({
                            "query": record["query"],
                            "expected_output": record.get("answer", ""),
                            "retrieval_context": contexts,
                            "source_chunk_id": record.get("chunk_id", ""),
                            "source_chunk_content": record.get("chunk_content", ""),
                        })
                except Exception as e:
                    logger.warning(f"[eval_data] Failed for query: {e}")
                finally:
                    processed += 1
                    if self.progress_callback and not self._stop_flag:
                        self.progress_callback(processed, total)

        await asyncio.gather(*[process_one(r) for r in records])
        if self._stop_flag:
            return []
        return eval_records

    async def _run_qa_to_eval(self) -> PipelineResult:
        """QA → 评估数据：从 QA 数据集生成深度评估数据集（纯检索，无 LLM 分类）"""
        embedding_client = None
        milvus_client = None
        collection_name = None
        rerank_client = None

        try:
            qa_records = self._load_qa_input_records()
            total = len(qa_records)
            if total == 0:
                return PipelineResult(success=False, error="No QA records found in input")

            if self._is_dedup_enabled():
                before = len(qa_records)
                qa_records = self._dedup_qa_records(qa_records)
                if len(qa_records) < before:
                    logger.info("[qa_to_eval] 去重: %d → %d", before, len(qa_records))

            embedding_client, milvus_client, collection_name, rerank_client = self._init_phase2_clients()
            if not milvus_client or not collection_name:
                return PipelineResult(success=False, error="Milvus not available, eval mode requires vector store")

            emb_cfg = self.config.embedding_config or {}
            filter_step = EmbeddingFilterStep(
                embedding_client=embedding_client,
                milvus_client=milvus_client,
                threshold=emb_cfg.get("similarity_threshold", 0.85),
                collection_name=collection_name,
                default_metadata=self._build_chunk_metadata(),
            )
            async with embedding_client:
                if self._stop_flag:
                    return self._stopped_result(total)
                filtered_records, filter_stats = await filter_step.execute(qa_records)
                if self._stop_flag:
                    return self._stopped_result(total)

                if not filtered_records:
                    return PipelineResult(
                        success=True, total_docs=total, processed_docs=total,
                        output_samples=0,
                        details={"filter_stats": filter_stats} if filter_stats else {},
                    )

                eval_records = await self._generate_eval_data_batch(
                    filtered_records, embedding_client, milvus_client, collection_name, rerank_client,
                )

            if self._stop_flag:
                return self._stopped_result(total)

            deep_eval_path = self._save_deep_eval_data(eval_records, "[qa_to_eval]")

            details: Dict[str, Any] = {}
            if filter_stats:
                filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                details["filter_stats"] = filter_stats
                details["milvus_collection"] = filter_stats.get("milvus_collection")
            if deep_eval_path:
                details["deep_eval_path"] = deep_eval_path
                details["deep_eval_count"] = len(eval_records)

            return PipelineResult(
                success=True, total_docs=total, processed_docs=total,
                output_samples=len(eval_records), output_path=deep_eval_path, details=details,
            )
        except Exception as e:
            logger.exception("[qa_to_eval] Pipeline execution failed")
            return PipelineResult(success=False, error=str(e))
        finally:
            await self._cleanup_clients(milvus_client, embedding_client, rerank_client)

    async def _run_doc_to_eval(self) -> PipelineResult:
        """文档 → 评估数据：QA 提取 + 向量检索生成深度评估数据集"""
        embedding_client = None
        milvus_client = None
        collection_name = None
        rerank_client = None

        try:
            documents = self._load_documents()
            total_docs = len(documents)
            if total_docs == 0:
                return PipelineResult(success=False, error="No documents found")

            logger.info(f"[doc_to_eval] Loaded {total_docs} documents")

            # Phase 1: QA 提取（支持从 checkpoint 续传）
            if self.config.resume_qa_path and os.path.exists(self.config.resume_qa_path):
                all_qa = self._load_qa_records(self.config.resume_qa_path)
                qa_output_path = self.config.resume_qa_path
                logger.info(
                    "[doc_to_eval] Resumed from QA checkpoint: %d records",
                    len(all_qa),
                )
            else:
                llm_client = self._create_llm_client()
                qa_queue: asyncio.Queue = asyncio.Queue()
                async with llm_client:
                    if self._stop_flag:
                        return self._stopped_result(total_docs)
                    all_qa, qa_output_path = await self._streaming_qa_producer(
                        documents, qa_queue, "[doc_to_eval]",
                    )

            if self._stop_flag:
                return self._stopped_result(total_docs)

            if not all_qa:
                return PipelineResult(success=False, error="No QA pairs generated from documents")

            logger.info(f"[doc_to_eval] Generated {len(all_qa)} QA pairs, starting eval data collection")

            # Phase 2: Embedding 入库 + 向量检索
            embedding_client, milvus_client, collection_name, rerank_client = self._init_phase2_clients()
            if not milvus_client or not collection_name:
                return PipelineResult(success=False, error="Milvus not available, eval mode requires vector store")

            emb_cfg = self.config.embedding_config or {}
            filter_step = EmbeddingFilterStep(
                embedding_client=embedding_client,
                milvus_client=milvus_client,
                threshold=emb_cfg.get("similarity_threshold", 0.85),
                collection_name=collection_name,
                default_metadata=self._build_chunk_metadata(),
            )
            async with embedding_client:
                if self._stop_flag:
                    return self._stopped_result(total_docs)
                filtered_records, filter_stats = await filter_step.execute(all_qa)
                if self._stop_flag:
                    return self._stopped_result(total_docs)

                if not filtered_records:
                    return PipelineResult(
                        success=True, total_docs=total_docs, processed_docs=total_docs,
                        output_samples=0,
                        details={
                            "qa_output_path": qa_output_path,
                            "qa_total": len(all_qa),
                            "filter_stats": filter_stats,
                        },
                    )

                eval_records = await self._generate_eval_data_batch(
                    filtered_records, embedding_client, milvus_client, collection_name, rerank_client,
                )

            if self._stop_flag:
                return self._stopped_result(total_docs)

            deep_eval_path = self._save_deep_eval_data(eval_records, "[doc_to_eval]")

            details: Dict[str, Any] = {
                "qa_output_path": qa_output_path,
                "qa_total": len(all_qa),
            }
            if filter_stats:
                filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                details["filter_stats"] = filter_stats
                details["milvus_collection"] = filter_stats.get("milvus_collection")
            if deep_eval_path:
                details["deep_eval_path"] = deep_eval_path
                details["deep_eval_count"] = len(eval_records)

            return PipelineResult(
                success=True, total_docs=total_docs, processed_docs=total_docs,
                output_samples=len(eval_records), output_path=deep_eval_path, details=details,
            )
        except Exception as e:
            logger.exception("[doc_to_eval] Pipeline execution failed")
            return PipelineResult(success=False, error=str(e))
        finally:
            await self._cleanup_clients(milvus_client, embedding_client, rerank_client)

    # ── 训练数据生成模式 ──────────────────────────────────────

    async def _run_qa_to_training(self) -> PipelineResult:
        """
        QA → 训练数据：从 QA 数据集生成正负例训练数据

        有 embedding → 筛选+入库 → pos/neg
        无 embedding → 直接 pos/neg
        """
        embedding_client = None
        milvus_client = None
        collection_name = None
        rerank_client = None

        try:
            qa_records = self._load_qa_input_records()
            total_pairs = len(qa_records)
            if total_pairs == 0:
                return PipelineResult(success=False, error="No QA records found in input")

            if self._is_dedup_enabled():
                qa_records = self._dedup_qa_records(qa_records)
                if len(qa_records) < total_pairs:
                    logger.info("[qa_to_training] 去重: %d → %d", total_pairs, len(qa_records))

            has_embedding = bool(self.config.embedding_config)
            use_retrieval = self.config.pos_neg_method == "retrieval"
            logger.info(
                f"[qa_to_training] {total_pairs} QA pairs, "
                f"embedding={has_embedding}, pos_neg_method={self.config.pos_neg_method}"
            )

            filter_stats = None
            filtered_records = qa_records
            qa_filtered_path = None

            if has_embedding:
                embedding_client, milvus_client, collection_name, rerank_client = self._init_phase2_clients()

                if (
                    self.config.resume_qa_filtered_path
                    and os.path.exists(self.config.resume_qa_filtered_path)
                ):
                    filtered_records = self._load_qa_records(self.config.resume_qa_filtered_path)
                    qa_filtered_path = self.config.resume_qa_filtered_path
                    filter_stats = {
                        "resumed_from_checkpoint": True,
                        "checkpoint_path": self.config.resume_qa_filtered_path,
                        "total_pairs": total_pairs,
                        "kept": len(filtered_records),
                        "filtered_out": max(total_pairs - len(filtered_records), 0),
                    }
                    await embedding_client.__aenter__()
                    if self._stop_flag:
                        return self._stopped_result(total_pairs)
                    logger.info(
                        "[qa_to_training] Resumed filtered QA from %s (%d records)",
                        qa_filtered_path,
                        len(filtered_records),
                    )
                elif self.config.skip_filter and self.config.previous_filter_stats:
                    prev_stats = self.config.previous_filter_stats
                    kept_indices = prev_stats.get("filtered_indices", [])
                    filtered_records = (
                        [qa_records[i] for i in kept_indices if i < len(qa_records)]
                        if kept_indices else qa_records
                    )
                    filter_stats = {**prev_stats, "skipped_dedup": True}
                    logger.info(f"[qa_to_training] Skipped filtering (dedup), using {len(filtered_records)} pairs")
                    await embedding_client.__aenter__()
                    if self._stop_flag:
                        return self._stopped_result(total_pairs)
                else:
                    _emb_cfg = self.config.embedding_config or {}
                    filter_step = EmbeddingFilterStep(
                        embedding_client=embedding_client,
                        milvus_client=milvus_client,
                        threshold=_emb_cfg.get("similarity_threshold", 0.85),
                        collection_name=collection_name,
                        default_metadata=self._build_chunk_metadata(),
                    )
                    async with embedding_client:
                        if self._stop_flag:
                            return self._stopped_result(total_pairs)
                        filtered_records, filter_stats = await filter_step.execute(qa_records)
                    if self._stop_flag:
                        return self._stopped_result(total_pairs)

                if not filtered_records:
                    if filter_stats:
                        filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                    return PipelineResult(
                        success=True, total_docs=total_pairs, processed_docs=total_pairs,
                        output_samples=0, details={"filter_stats": filter_stats} if filter_stats else {},
                    )

                qa_filtered_path = self._save_qa_intermediate(filtered_records, suffix="qa_filtered")

            # Pre-resolve output path for incremental pos/neg checkpoint resume.
            _output_path = self._resolve_output_path()
            _output_path.parent.mkdir(parents=True, exist_ok=True)
            if not _output_path.exists():
                _output_path.write_text("", encoding="utf-8")
            self._incremental_output_path = _output_path

            # pos/neg 生成
            llm_client = self._create_llm_client()
            eval_llm_client = (
                self._create_llm_client(self.config.eval_llm_config)
                if self.config.eval_llm_config else None
            )
            if eval_llm_client:
                async with llm_client, eval_llm_client:
                    if self._stop_flag:
                        return self._stopped_result(total_pairs)
                    all_samples, deep_eval_records = await self._generate_pos_neg_batch(
                        filtered_records, llm_client,
                        embedding_client if use_retrieval else None,
                        milvus_client if use_retrieval else None,
                        collection_name if use_retrieval else None,
                        rerank_client if use_retrieval else None,
                        eval_llm_client=eval_llm_client,
                    )
            else:
                async with llm_client:
                    if self._stop_flag:
                        return self._stopped_result(total_pairs)
                    all_samples, deep_eval_records = await self._generate_pos_neg_batch(
                        filtered_records, llm_client,
                        embedding_client if use_retrieval else None,
                        milvus_client if use_retrieval else None,
                        collection_name if use_retrieval else None,
                        rerank_client if use_retrieval else None,
                        eval_llm_client=eval_llm_client,
                    )

            if self._stop_flag:
                return self._stopped_result(total_pairs)

            output_path = self._save_output(all_samples)
            details: Dict[str, Any] = {}
            if filter_stats:
                filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                details["filter_stats"] = filter_stats
                details["milvus_collection"] = filter_stats.get("milvus_collection")
            if use_retrieval:
                details["retrieval_top_k"] = (self.config.embedding_config or {}).get("retrieval_top_k", 10)
            if qa_filtered_path:
                details["qa_filtered_path"] = qa_filtered_path
                details["qa_filtered_count"] = len(filtered_records)

            # 保存深度评估数据
            deep_eval_path = self._save_deep_eval_data(deep_eval_records, "[qa_to_training]")
            if deep_eval_path:
                details["deep_eval_path"] = deep_eval_path
                details["deep_eval_count"] = len(deep_eval_records)

            return PipelineResult(
                success=True, total_docs=total_pairs, processed_docs=total_pairs,
                output_samples=len(all_samples), output_path=output_path, details=details,
            )
        except Exception as e:
            logger.exception("[qa_to_training] Pipeline execution failed")
            return PipelineResult(success=False, error=str(e))
        finally:
            await self._cleanup_clients(milvus_client, embedding_client, rerank_client)

    async def _run_doc_to_training(self) -> PipelineResult:
        """
        文档 → 训练数据：从原始文档提取 QA 再生成正负例训练数据

        有 embedding → Phase 0 (chunk 预索引到 Milvus) → 流式 QA提取 ‖ filter+pos/neg
        无 embedding → QA提取 ‖ pos/neg 流水线

        Phase 0 预先将所有文档 chunk embed 并入库 Milvus，消除了旧架构中
        Phase 1→2 的 barrier 等待。Phase 0 完成后 QA 提取和正负例生成可以流式并行。

        支持断点续传：如果 resume_qa_path 或 resume_qa_filtered_path 存在，
        跳过 QA 提取阶段，直接从已有 QA 文件开始正负例生成。
        """
        embedding_client = None
        milvus_client = None
        collection_name = None
        rerank_client = None

        try:
            documents = self._load_documents()
            total_docs = len(documents)
            if total_docs == 0:
                return PipelineResult(success=False, error="No documents found")

            has_embedding = bool(self.config.embedding_config)
            use_retrieval = self.config.pos_neg_method == "retrieval"

            # Pre-resolve output path for incremental writes during pos/neg phase
            _output_path = self._resolve_output_path()
            _output_path.parent.mkdir(parents=True, exist_ok=True)
            # Create empty file so preview endpoint can find it immediately
            if not _output_path.exists():
                _output_path.write_text("", encoding="utf-8")
            self._incremental_output_path = _output_path

            logger.info(
                f"[doc_to_training] {total_docs} documents, "
                f"embedding={has_embedding}, pos_neg_method={self.config.pos_neg_method}"
            )

            # ── 断点续传：检查是否可以跳过 QA 提取阶段 ──
            resumed = False
            all_qa = []
            filtered_records = []
            qa_output_path = None
            qa_filtered_path = None
            filter_stats = {}

            # 确定性路径（增量写入使用），用于区分旧系统 UUID 文件和增量文件
            _incremental_qa_path = str(self._get_qa_output_path())
            _incremental_filtered_path = str(self._get_qa_filtered_path())

            # 判断 resume_qa_path 是否为增量文件（由 _streaming_qa_producer 自行处理）
            _is_incremental = (
                self.config.resume_qa_path
                and self.config.resume_qa_path == _incremental_qa_path
            )

            if _is_incremental:
                # 增量文件：跳过 legacy 路径，让正常流程中的 _streaming_qa_producer 自动 resume
                logger.info("[doc_to_training] Incremental QA file detected, delegating to streaming producer")

            elif has_embedding and self.config.resume_qa_filtered_path and os.path.exists(self.config.resume_qa_filtered_path):
                # 最优情况：QA 提取 + 筛选都已完成（旧系统 UUID 文件）
                filtered_records = self._load_qa_records(self.config.resume_qa_filtered_path)
                qa_filtered_path = self.config.resume_qa_filtered_path
                if self.config.resume_qa_path and os.path.exists(self.config.resume_qa_path):
                    all_qa = self._load_qa_records(self.config.resume_qa_path)
                    qa_output_path = self.config.resume_qa_path
                resumed = True
                logger.info(
                    f"[doc_to_training] Resumed from filtered QA: {len(filtered_records)} records "
                    f"(skipped QA extraction + filtering)"
                )

            elif self.config.resume_qa_path and os.path.exists(self.config.resume_qa_path):
                # QA 提取已完成，但筛选未完成（旧系统 UUID 文件）
                all_qa = self._load_qa_records(self.config.resume_qa_path)
                qa_output_path = self.config.resume_qa_path
                if has_embedding:
                    # 需要重新筛选
                    embedding_client, milvus_client, collection_name, rerank_client = self._init_phase2_clients()
                    _emb_cfg2 = self.config.embedding_config or {}
                    filter_step = EmbeddingFilterStep(
                        embedding_client=embedding_client,
                        milvus_client=milvus_client,
                        threshold=_emb_cfg2.get("similarity_threshold", 0.85),
                        collection_name=collection_name,
                        default_metadata=self._build_chunk_metadata(),
                    )
                    async with embedding_client:
                        if self._stop_flag:
                            return self._stopped_result(total_docs)
                        filtered_records = await filter_step.execute_batch(all_qa)
                    if self._stop_flag:
                        return self._stopped_result(total_docs)
                    filter_stats = filter_step.finalize_stats()
                    qa_filtered_path = self._save_qa_intermediate(filtered_records, suffix="qa_filtered") if filtered_records else None
                else:
                    filtered_records = all_qa
                resumed = True
                logger.info(
                    f"[doc_to_training] Resumed from QA output: {len(all_qa)} records "
                    f"(skipped QA extraction, re-filtered to {len(filtered_records)})"
                )

            if resumed:
                # 直接进入 Phase 2: 正负例生成（Phase 1 已跳过）
                # 合并进度：Phase 1 (total_docs) 视为已完成，Phase 2 从 total_docs 偏移开始
                phase2_records = filtered_records if has_embedding else all_qa
                actual_combined = total_docs + len(phase2_records)
                original_callback = self.progress_callback
                if original_callback and phase2_records:
                    # 立即报告 Phase 1 完成的偏移量，让前端看到"QA 阶段已跳过"
                    original_callback(total_docs, actual_combined)
                    self.progress_callback = lambda p, t: original_callback(
                        total_docs + p, actual_combined
                    )

                if has_embedding:
                    if not embedding_client:
                        embedding_client, milvus_client, collection_name, rerank_client = self._init_phase2_clients()

                    if not filtered_records:
                        filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                        self.progress_callback = original_callback
                        return PipelineResult(
                            success=True, total_docs=total_docs, processed_docs=total_docs,
                            output_samples=0, details={
                                "filter_stats": filter_stats,
                                "qa_output_path": qa_output_path, "qa_total": len(all_qa),
                                "qa_filtered_path": None, "qa_filtered_count": 0,
                            },
                        )

                    llm_client = self._create_llm_client()
                    eval_llm_client = (
                        self._create_llm_client(self.config.eval_llm_config)
                        if self.config.eval_llm_config else None
                    )
                    if eval_llm_client:
                        async with llm_client, eval_llm_client:
                            if self._stop_flag:
                                return self._stopped_result(total_docs)
                            all_samples, deep_eval_records = await self._generate_pos_neg_batch(
                                filtered_records, llm_client,
                                embedding_client if use_retrieval else None,
                                milvus_client if use_retrieval else None,
                                collection_name if use_retrieval else None,
                                rerank_client if use_retrieval else None,
                                eval_llm_client=eval_llm_client,
                            )
                    else:
                        async with llm_client:
                            if self._stop_flag:
                                return self._stopped_result(total_docs)
                            all_samples, deep_eval_records = await self._generate_pos_neg_batch(
                                filtered_records, llm_client,
                                embedding_client if use_retrieval else None,
                                milvus_client if use_retrieval else None,
                                collection_name if use_retrieval else None,
                                rerank_client if use_retrieval else None,
                                eval_llm_client=eval_llm_client,
                            )

                    if self._stop_flag:
                        return self._stopped_result(total_docs)
                    self.progress_callback = original_callback
                    output_path = self._save_output(all_samples)
                    filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                    details: Dict[str, Any] = {
                        "filter_stats": filter_stats,
                        "qa_output_path": qa_output_path, "qa_total": len(all_qa),
                        "qa_filtered_path": qa_filtered_path, "qa_filtered_count": len(filtered_records),
                    }
                    if use_retrieval:
                        details["milvus_collection"] = filter_stats.get("milvus_collection")
                        details["retrieval_top_k"] = (self.config.embedding_config or {}).get("retrieval_top_k", 10)

                    # 保存深度评估数据
                    deep_eval_path = self._save_deep_eval_data(deep_eval_records, "[doc_to_training:resume]")
                    if deep_eval_path:
                        details["deep_eval_path"] = deep_eval_path
                        details["deep_eval_count"] = len(deep_eval_records)

                    return PipelineResult(
                        success=True, total_docs=total_docs, processed_docs=total_docs,
                        output_samples=len(all_samples), output_path=output_path, details=details,
                    )
                else:
                    # 无 embedding，直接生成正负例
                    llm_client = self._create_llm_client()
                    eval_llm_client = (
                        self._create_llm_client(self.config.eval_llm_config)
                        if self.config.eval_llm_config else None
                    )
                    if eval_llm_client:
                        async with llm_client, eval_llm_client:
                            if self._stop_flag:
                                return self._stopped_result(total_docs)
                            all_samples, deep_eval_records = await self._generate_pos_neg_batch(
                                all_qa, llm_client, None, None, None, None,
                                eval_llm_client=eval_llm_client,
                            )
                    else:
                        async with llm_client:
                            if self._stop_flag:
                                return self._stopped_result(total_docs)
                            all_samples, deep_eval_records = await self._generate_pos_neg_batch(
                                all_qa, llm_client, None, None, None, None,
                                eval_llm_client=eval_llm_client,
                            )

                    if self._stop_flag:
                        return self._stopped_result(total_docs)
                    self.progress_callback = original_callback
                    output_path = self._save_output(all_samples)
                    details: Dict[str, Any] = {
                        "qa_output_path": qa_output_path, "qa_total": len(all_qa),
                    }

                    # 保存深度评估数据
                    deep_eval_path = self._save_deep_eval_data(deep_eval_records, "[doc_to_training:resume]")
                    if deep_eval_path:
                        details["deep_eval_path"] = deep_eval_path
                        details["deep_eval_count"] = len(deep_eval_records)

                    return PipelineResult(
                        success=True, total_docs=total_docs, processed_docs=total_docs,
                        output_samples=len(all_samples), output_path=output_path,
                        details=details,
                    )

            # ── 正常流程（非续传）──
            qa_queue: asyncio.Queue = asyncio.Queue(maxsize=8)

            if has_embedding:
                # ═══ Phase 0 预索引 + 流式 Phase 1+2（无 barrier）═══
                embedding_client, milvus_client, collection_name, rerank_client = self._init_phase2_clients()
                _emb_cfg2 = self.config.embedding_config or {}
                filter_step = EmbeddingFilterStep(
                    embedding_client=embedding_client,
                    milvus_client=milvus_client,
                    threshold=_emb_cfg2.get("similarity_threshold", 0.85),
                    collection_name=collection_name,
                    default_metadata=self._build_chunk_metadata(),
                )

                # 进度分配：Phase 0 (chunk 入库) + Phase 1+2 (QA+filter+pos/neg)
                pairs_per_doc = (self.config.steps or {}).get("qa_gen", {}).get("num_qa_per_doc", 1)
                estimated_phase12 = total_docs * (1 + pairs_per_doc)
                phase0_weight = total_docs
                total_combined = phase0_weight + estimated_phase12
                original_callback = self.progress_callback
                last_reported_progress = 0

                def report_combined_progress(processed: int) -> None:
                    nonlocal last_reported_progress
                    if not original_callback or self._stop_flag:
                        return
                    last_reported_progress = max(
                        last_reported_progress,
                        min(processed, total_combined),
                    )
                    original_callback(last_reported_progress, total_combined)

                # ═══ Phase 0: 预索引所有文档 chunk 到 Milvus ═══
                logger.info(f"[doc_to_training] Phase 0: pre-indexing {total_docs} chunks into Milvus")
                async with embedding_client:
                    if self._stop_flag:
                        return self._stopped_result(total_docs)
                    inserted = await filter_step.pre_index_all_chunks(
                        documents,
                        progress_callback=lambda p, t: report_combined_progress(p),
                    )
                if self._stop_flag:
                    return self._stopped_result(total_docs)
                logger.info(f"[doc_to_training] Phase 0 complete: {inserted} new chunks indexed")

                # 提前确定文件路径并持久化
                qa_filtered_file_path = self._get_qa_filtered_path()
                qa_file_path = self._get_qa_output_path()
                await self._notify_qa_phase(
                    str(qa_file_path),
                    str(qa_filtered_file_path),
                )
                if self._stop_flag:
                    return self._stopped_result(total_docs)

                # Phase 1+2 进度偏移
                # ═══ Phase 1+2: 流式 QA提取 + filter + 正负例 ═══
                async def filter_and_pos_neg_consumer() -> Tuple[
                    List[GeneratedSample],
                    List[Dict[str, Any]],
                    List[Dict[str, Any]],
                    Dict[str, Any],
                ]:
                    all_samples: List[GeneratedSample] = []
                    all_deep_eval: List[Dict[str, Any]] = []
                    all_filtered: List[Dict[str, Any]] = []
                    batch_failures = _new_batch_failure_stats()
                    consumer_processed = 0

                    # 加载 checkpoint（仅一次，consumer 级别）
                    _completed_qa_keys: set = set()
                    _inc_path = getattr(self, "_incremental_output_path", None)
                    checkpoint_supported = self.config.output_format == "universal"
                    if (
                        checkpoint_supported
                        and _inc_path
                        and Path(_inc_path).exists()
                        and Path(_inc_path).stat().st_size > 0
                    ):
                        try:
                            with open(_inc_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        obj = json.loads(line)
                                        cid = (obj.get("metadata") or {}).get("source_doc_id") or ""
                                        _completed_qa_keys.add(self._qa_checkpoint_key(obj))
                                        all_samples.append(GeneratedSample(
                                            query=obj.get("query", ""),
                                            answer=obj.get("answer", ""),
                                            positive_chunks=obj.get("positives", []),
                                            negative_chunks=obj.get("negatives", []),
                                            source_doc_id=cid,
                                        ))
                                    except json.JSONDecodeError:
                                        continue
                            if _completed_qa_keys:
                                logger.info(
                                    f"[doc_to_training] Consumer checkpoint: "
                                    f"{len(_completed_qa_keys)} records resumed"
                                )
                        except Exception as e:
                            logger.warning(f"[doc_to_training] Consumer checkpoint load failed: {e}")
                            _completed_qa_keys.clear()
                    elif _inc_path and not checkpoint_supported and Path(_inc_path).exists():
                        logger.info(
                            "[doc_to_training] Output format '%s' does not support pos/neg checkpoint resume",
                            self.config.output_format,
                        )

                    llm_client = self._create_llm_client()
                    eval_llm_client = (
                        self._create_llm_client(self.config.eval_llm_config)
                        if self.config.eval_llm_config else None
                    )

                    # 重建过滤文件（producer 会 replay 已有记录，重建避免重复）
                    if qa_filtered_file_path.exists():
                        qa_filtered_file_path.unlink()

                    ctx_managers = [embedding_client, llm_client]
                    if eval_llm_client:
                        ctx_managers.append(eval_llm_client)

                    async with AsyncExitStack() as stack:
                        for mgr in ctx_managers:
                            await stack.enter_async_context(mgr)
                            if self._stop_flag:
                                break

                        while True:
                            batch = await qa_queue.get()
                            if batch is None:
                                break
                            if self._stop_flag:
                                continue

                            filtered: List[Dict[str, Any]] = []
                            try:
                                # Step 1: 过滤（chunk 已缓存，只需 embed query + cosine sim）
                                filtered = await filter_step.execute_batch(batch)
                                if self._stop_flag:
                                    continue
                                if not filtered:
                                    continue
                                all_filtered.extend(filtered)

                                # 增量写入 qa_filtered_path
                                with open(qa_filtered_file_path, "a", encoding="utf-8") as f:
                                    for r in filtered:
                                        f.write(json.dumps({
                                            "query": r.get("query", ""),
                                            "answer": r.get("answer", ""),
                                            "chunk_id": r.get("chunk_id", ""),
                                            "chunk_content": r.get("chunk_content", ""),
                                        }, ensure_ascii=False) + "\n")

                                # Step 2: 跳过已完成的记录（checkpoint resume）
                                new_records = [
                                    r for r in filtered
                                    if self._qa_checkpoint_key(r) not in _completed_qa_keys
                                ]
                                skipped_records = len(batch) - len(new_records)
                                if not new_records:
                                    continue

                                # Step 3: 立即生成正负例（Milvus 已就绪）
                                completed_before_batch = consumer_processed + skipped_records
                                samples, deep_eval = await self._generate_pos_neg_batch(
                                    new_records, llm_client,
                                    embedding_client if use_retrieval else None,
                                    milvus_client if use_retrieval else None,
                                    collection_name if use_retrieval else None,
                                    rerank_client if use_retrieval else None,
                                    eval_llm_client=eval_llm_client,
                                    _skip_checkpoint=True,
                                    progress_callback=lambda p, t, completed=completed_before_batch: (
                                        report_combined_progress(
                                            phase0_weight + total_docs + completed + p
                                        )
                                    ),
                                )
                                if self._stop_flag:
                                    continue
                                all_samples.extend(samples)
                                all_deep_eval.extend(deep_eval)
                            except Exception as batch_error:
                                # Isolate per-batch failures so a single bad batch
                                # doesn't kill the consumer (leaving the producer
                                # blocked on a full queue and _save_output unrun).
                                # asyncio.CancelledError is not an Exception subclass,
                                # so task cancellation still propagates correctly.
                                _record_batch_failure(
                                    batch_failures,
                                    batch,
                                    batch_error,
                                    "doc_to_training",
                                )
                                continue
                            finally:
                                consumer_processed += len(batch)
                                report_combined_progress(
                                    phase0_weight + total_docs + consumer_processed
                                )

                    return (
                        all_samples,
                        all_filtered,
                        all_deep_eval,
                        batch_failures,
                    )

                (all_qa, qa_output_path), (
                    all_samples,
                    filtered_records_list,
                    deep_eval_records,
                    batch_failures,
                ) = (
                    await asyncio.gather(
                        asyncio.create_task(
                            self._streaming_qa_producer(
                                documents,
                                qa_queue,
                                "[doc_to_training]",
                                progress_callback=lambda p, t: report_combined_progress(
                                    phase0_weight + p
                                ),
                            )
                        ),
                        asyncio.create_task(filter_and_pos_neg_consumer()),
                    )
                )
                if self._stop_flag:
                    return self._stopped_result(total_docs)
                filter_stats = filter_step.finalize_stats()
                report_combined_progress(total_combined)

                logger.info(
                    f"[doc_to_training] QA: {len(all_qa)} records, "
                    f"filtered: {len(filtered_records_list)} kept, "
                    f"PosNeg: {len(all_samples)} samples"
                )

                # 恢复原始回调
                self.progress_callback = original_callback

                if not filtered_records_list:
                    filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                    return _apply_batch_failure_status(
                        PipelineResult(
                            success=True,
                            total_docs=total_docs,
                            processed_docs=total_docs,
                            output_samples=0,
                            details={
                                "filter_stats": filter_stats,
                                "qa_output_path": qa_output_path,
                                "qa_total": len(all_qa),
                                "qa_filtered_path": None,
                                "qa_filtered_count": 0,
                            },
                        ),
                        batch_failures,
                    )

                output_path = self._save_output(all_samples)
                filter_stats["dedup_enabled"] = self._is_dedup_enabled()
                details: Dict[str, Any] = {
                    "filter_stats": filter_stats,
                    "qa_output_path": qa_output_path, "qa_total": len(all_qa),
                    "qa_filtered_path": str(qa_filtered_file_path),
                    "qa_filtered_count": len(filtered_records_list),
                }
                if use_retrieval:
                    details["milvus_collection"] = filter_stats.get("milvus_collection")
                    details["retrieval_top_k"] = (self.config.embedding_config or {}).get("retrieval_top_k", 10)

                # 保存深度评估数据
                deep_eval_path = self._save_deep_eval_data(deep_eval_records, "[doc_to_training]")
                if deep_eval_path:
                    details["deep_eval_path"] = deep_eval_path
                    details["deep_eval_count"] = len(deep_eval_records)

                return _apply_batch_failure_status(
                    PipelineResult(
                        success=True,
                        total_docs=total_docs,
                        processed_docs=total_docs,
                        output_samples=len(all_samples),
                        output_path=output_path,
                        details=details,
                    ),
                    batch_failures,
                )

            else:
                # 无 embedding → QA提取 ‖ pos/neg 流水线（无 barrier）
                pairs_per_doc = (self.config.steps or {}).get(
                    "qa_gen", {}
                ).get("num_qa_per_doc", 1)
                total_combined = total_docs + (total_docs * pairs_per_doc)
                original_callback = self.progress_callback
                last_reported_progress = 0

                def report_combined_progress(processed: int) -> None:
                    nonlocal last_reported_progress
                    if not original_callback or self._stop_flag:
                        return
                    last_reported_progress = max(
                        last_reported_progress,
                        min(processed, total_combined),
                    )
                    original_callback(last_reported_progress, total_combined)

                # 提前持久化 QA 文件路径
                qa_file_path = self._get_qa_output_path()
                await self._notify_qa_phase(str(qa_file_path), None)
                if self._stop_flag:
                    return self._stopped_result(total_docs)

                async def pos_neg_consumer() -> Tuple[
                    List[GeneratedSample],
                    List[Dict[str, Any]],
                    Dict[str, Any],
                ]:
                    llm_client = self._create_llm_client()
                    eval_llm_client = (
                        self._create_llm_client(self.config.eval_llm_config)
                        if self.config.eval_llm_config else None
                    )
                    all_samples: List[GeneratedSample] = []
                    all_deep_eval: List[Dict[str, Any]] = []
                    batch_failures = _new_batch_failure_stats()
                    consumer_processed = 0

                    ctx_managers = [llm_client]
                    if eval_llm_client:
                        ctx_managers.append(eval_llm_client)

                    async with AsyncExitStack() as stack:
                        for manager in ctx_managers:
                            await stack.enter_async_context(manager)
                            if self._stop_flag:
                                break

                        while True:
                            batch = await qa_queue.get()
                            if batch is None:
                                break
                            if self._stop_flag:
                                continue

                            try:
                                completed_before_batch = consumer_processed
                                samples, deep_eval_records = await self._generate_pos_neg_batch(
                                    batch, llm_client, None, None, None, None,
                                    eval_llm_client=eval_llm_client,
                                    _skip_checkpoint=True,
                                    progress_callback=lambda p, t, completed=completed_before_batch: (
                                        report_combined_progress(
                                            total_docs + completed + p
                                        )
                                    ),
                                )
                                if self._stop_flag:
                                    continue
                                all_samples.extend(samples)
                                all_deep_eval.extend(deep_eval_records)
                            except Exception as batch_error:
                                _record_batch_failure(
                                    batch_failures,
                                    batch,
                                    batch_error,
                                    "doc_to_training",
                                )
                                continue
                            finally:
                                consumer_processed += len(batch)
                                report_combined_progress(
                                    total_docs + consumer_processed
                                )

                    return all_samples, all_deep_eval, batch_failures

                (all_qa, qa_output_path), (
                    all_samples,
                    deep_eval_records,
                    batch_failures,
                ) = await asyncio.gather(
                    asyncio.create_task(
                        self._streaming_qa_producer(
                            documents,
                            qa_queue,
                            "[doc_to_training]",
                            progress_callback=lambda p, t: report_combined_progress(p),
                        )
                    ),
                    asyncio.create_task(pos_neg_consumer()),
                )
                if self._stop_flag:
                    return self._stopped_result(total_docs)
                report_combined_progress(total_combined)

                logger.info(
                    f"[doc_to_training] QA: {len(all_qa)} records, PosNeg: {len(all_samples)} samples"
                )

                output_path = self._save_output(all_samples)
                details: Dict[str, Any] = {
                    "qa_output_path": qa_output_path, "qa_total": len(all_qa),
                }

                # 保存深度评估数据
                deep_eval_path = self._save_deep_eval_data(deep_eval_records, "[doc_to_training]")
                if deep_eval_path:
                    details["deep_eval_path"] = deep_eval_path
                    details["deep_eval_count"] = len(deep_eval_records)

                return _apply_batch_failure_status(
                    PipelineResult(
                        success=True, total_docs=total_docs, processed_docs=total_docs,
                        output_samples=len(all_samples), output_path=output_path,
                        details=details,
                    ),
                    batch_failures,
                )
        except Exception as e:
            logger.exception("[doc_to_training] Pipeline execution failed")
            return PipelineResult(success=False, error=str(e))
        finally:
            await self._cleanup_clients(milvus_client, embedding_client, rerank_client)

    @staticmethod
    def _calculate_average_precision(binary_labels: List[int]) -> float:
        """Average Precision (AP) for a ranked binary relevance list.

        AP = (1/R) * sum_{k where y_k=1} Precision@k, with R = #relevant.
        Returns 0.0 if there is no relevant item.
        """
        if not binary_labels:
            return 0.0
        relevant = 0
        sum_prec = 0.0
        for k, y in enumerate(binary_labels, start=1):
            if y:
                relevant += 1
                sum_prec += relevant / k
        if relevant == 0:
            return 0.0
        return sum_prec / relevant

    @staticmethod
    def _check_training_value(
        ap: float,
        skip_perfect: bool = True,
        skip_zero: bool = True,
        epsilon: float = 1e-6,
    ) -> Tuple[bool, str]:
        """判断样本是否有训练价值

        - AP ≈ 1: 完美排序，排序模型已经能正确排序，无需训练
        - AP ≈ 0: 无正例，候选中没有相关内容，无法构建有效训练样本
        - 0 < AP < 1: 有训练价值
        """
        if skip_perfect and ap >= 1.0 - epsilon:
            return False, "AP=1 (perfect ranking, no training value)"
        if skip_zero and ap <= epsilon:
            return False, "AP=0 (no positives in candidates)"
        return True, ""

    async def _generate_pos_neg_batch(
        self,
        filtered_records: List[Dict[str, Any]],
        llm_client: LLMClientPool,
        embedding_client: Optional[EmbeddingClient],
        milvus_client: Optional[MilvusClient],
        collection_name: Optional[str],
        rerank_client: Optional[RerankClient] = None,
        eval_llm_client: Optional[LLMClientPool] = None,
        incremental_output_path: Optional[Path] = None,
        _skip_checkpoint: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[GeneratedSample], List[Dict[str, Any]]]:
        """
        barrier 后的并发 pos/neg 生成

        流程（retrieval 模式）：
        embedding 检索 → [可选] rerank 重排序 → ContextualPrecisionMetric 评估
        → AP 训练价值过滤 → 按位置分类正负例（very_hard/hard/medium）
        → [可选] statement 模式难负例提取 → [可选] LLM 增广

        Args:
            incremental_output_path: If provided, each sample is appended
                to this file as soon as it completes, enabling live preview.
                Falls back to self._incremental_output_path if not given.

        Returns:
            Tuple of (generated samples, deep eval records for deep evaluation)
        """
        all_samples: List[GeneratedSample] = []
        deep_eval_records: List[Dict[str, Any]] = []
        resolved_progress_callback = progress_callback or self.progress_callback
        # Writes always belong to the current attempt.  A restart may read the
        # previous attempt's immutable checkpoint, but must never append to it.
        _inc_path = incremental_output_path or getattr(self, "_incremental_output_path", None)
        _checkpoint_path = (
            Path(self.config.resume_output_path)
            if self.config.resume_output_path
            else Path(_inc_path) if _inc_path else None
        )
        _incremental_lock = asyncio.Lock() if _inc_path else None

        # ── Phase 2 checkpoint: skip records already in output file ──
        _completed_qa_keys: set = set()
        _resumed_samples: List[GeneratedSample] = []
        checkpoint_supported = self.config.output_format == "universal"
        if (
            not _skip_checkpoint
            and checkpoint_supported
            and _checkpoint_path
            and _checkpoint_path.exists()
            and _checkpoint_path.stat().st_size > 0
        ):
            try:
                with open(_checkpoint_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # Extract chunk_id from the output format
                            cid = (obj.get("metadata") or {}).get("source_doc_id") or ""
                            _completed_qa_keys.add(self._qa_checkpoint_key(obj))
                            # Reconstruct GeneratedSample for final _save_output
                            _resumed_samples.append(GeneratedSample(
                                query=obj.get("query", ""),
                                answer=obj.get("answer", ""),
                                positive_chunks=obj.get("positives", []),
                                negative_chunks=obj.get("negatives", []),
                                source_doc_id=cid,
                            ))
                        except json.JSONDecodeError:
                            continue
                if _completed_qa_keys:
                    all_samples.extend(_resumed_samples)
                    logger.info(
                        f"[pos_neg] Resuming: {len(_completed_qa_keys)} records "
                        f"already in output, will skip"
                    )
            except Exception as e:
                logger.warning(f"[pos_neg] Failed to load checkpoint: {e}")
                _completed_qa_keys.clear()
                _resumed_samples.clear()
        elif (
            not _skip_checkpoint
            and _checkpoint_path
            and not checkpoint_supported
            and _checkpoint_path.exists()
        ):
            logger.info(
                "[pos_neg] Output format '%s' does not support checkpoint resume",
                self.config.output_format,
            )
        emb_cfg = self.config.embedding_config or {}
        retrieval_top_k = emb_cfg.get("retrieval_top_k", 10)
        pos_neg_cfg = self.config.steps.get("pos_neg_extraction", {})
        if not pos_neg_cfg.get("enabled", True):
            logger.info("[pos_neg] pos_neg_extraction disabled, skip pos/neg generation")
            return [], []
        num_pos = pos_neg_cfg.get("num_positive", 5)
        num_neg = pos_neg_cfg.get("num_negative", 20)
        augment = pos_neg_cfg.get("augment", False)
        skip_perfect_ap = pos_neg_cfg.get("skip_perfect_ap", True)
        skip_zero_ap = pos_neg_cfg.get("skip_zero_ap", True)
        neg_detection_mode = pos_neg_cfg.get("neg_detection_mode", "chunk")
        supplement_positives = pos_neg_cfg.get("supplement_positives", True)
        confirm_positives = pos_neg_cfg.get("confirm_positives", False)
        confirm_negatives = pos_neg_cfg.get("confirm_negatives", False)
        answer_rewrite = pos_neg_cfg.get("answer_rewrite", False)
        rerank_score_classification = pos_neg_cfg.get("rerank_score_classification", False)
        evidence_removal = pos_neg_cfg.get("evidence_removal", False)
        evidence_pruning = pos_neg_cfg.get("evidence_pruning", False)
        neg_chunk_scoring = pos_neg_cfg.get("neg_chunk_scoring", False)
        skip_easy_negatives = pos_neg_cfg.get("skip_easy_negatives", True)
        chunk_eval_mode = pos_neg_cfg.get("chunk_eval_mode", "batch")
        use_retrieval = bool(milvus_client and collection_name and embedding_client)
        semaphore = asyncio.Semaphore(self.config.llm_concurrency)
        processed = 0
        skipped_ap = 0
        total = len(filtered_records)

        # hard negative 提取器（statement 模式 + chunk 模式的分层/证据操作共用）
        hard_neg_extractor = None
        needs_extractor = (
            (neg_detection_mode == "statement" and use_retrieval)
            or neg_chunk_scoring
            or evidence_removal
            or evidence_pruning
        )
        if needs_extractor:
            try:
                from .steps.hard_neg_extraction import HardNegativeExtractionStep
                hard_neg_extractor = HardNegativeExtractionStep()
            except ImportError:
                logger.warning("[pos_neg] hard_neg_extraction module not found, disabling dependent features")
                if neg_detection_mode == "statement":
                    neg_detection_mode = "chunk"
                neg_chunk_scoring = False
                evidence_removal = False
                evidence_pruning = False

        async def generate_pos_neg(record: Dict[str, Any]) -> Optional[GeneratedSample]:
            nonlocal processed, skipped_ap
            if self._stop_flag:
                return None

            # Skip only the exact QA already completed in the checkpoint.
            if self._qa_checkpoint_key(record) in _completed_qa_keys:
                processed += 1
                if self.progress_callback and not self._stop_flag:
                    self.progress_callback(processed, total)
                return None

            async with semaphore:
                if self._stop_flag:
                    return None
                try:
                    source_content = record.get("chunk_content", "")
                    answer_text = record.get("answer", "")
                    negatives = []

                    if neg_detection_mode == "statement" and answer_text.strip():
                        # statement 模式：answer 为首正例（可选 answer 重述）
                        first_positive = answer_text
                        if answer_rewrite and hard_neg_extractor:
                            try:
                                rewritten = await hard_neg_extractor.rewrite_answer(
                                    record["query"], answer_text, llm_client,
                                )
                                if self._stop_flag:
                                    return None
                                if rewritten:
                                    first_positive = rewritten
                            except Exception as e:
                                logger.debug(f"[pos_neg] Answer rewrite failed: {e}")
                        positives = [first_positive]
                        seen_contents = {first_positive.strip(), source_content.strip()}
                        # 收集正例 chunks 用于可选的正例语句提取
                        positive_chunks_for_extraction = [source_content]
                        seen_chunks = {source_content.strip()}
                    else:
                        # chunk 模式（或 answer 为空时回退）：源 chunk 为首正例
                        positives = [source_content]
                        seen_contents = {source_content.strip()}
                        # chunk 模式下也收集正例 chunk（用于证据去除/裁剪）
                        positive_chunks_for_extraction = [source_content] if (evidence_removal or evidence_pruning) else []
                        seen_chunks = {source_content.strip()} if (evidence_removal or evidence_pruning) else set()

                    # 向量检索 + [可选 rerank] + 评估分类
                    rerank_score_map: Dict[str, float] = {}
                    if use_retrieval:
                        try:
                            query_emb = await embedding_client.embed([record["query"]])
                            if self._stop_flag:
                                return None
                            search_results = milvus_client.search_similar(
                                collection_name,
                                query_emb,
                                top_k=retrieval_top_k,
                                exclude_chunk_ids=[record.get("chunk_id", "")],
                            )
                            candidate_chunks = search_results[0] if search_results else []

                            if candidate_chunks:
                                contexts = [c["chunk_content"] for c in candidate_chunks]

                                # 可选：rerank 重排序
                                if rerank_client:
                                    try:
                                        rerank_results = await rerank_client.rerank(
                                            record["query"], contexts,
                                        )
                                        if self._stop_flag:
                                            return None
                                        # rerank 相似度过滤（大于阈值的视为近似重复，过滤掉）
                                        rerank_cfg = self.config.rerank_config or {}
                                        rerank_threshold = rerank_cfg.get("rerank_threshold", 1.0)
                                        before_count = len(rerank_results)
                                        rerank_results = [r for r in rerank_results if r.score <= rerank_threshold]
                                        if before_count != len(rerank_results):
                                            logger.debug(
                                                f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                                                f"rerank filtered {before_count - len(rerank_results)} "
                                                f"candidates above threshold {rerank_threshold}"
                                            )
                                        # 按 rerank 分数重排序 contexts，保存分数映射
                                        rerank_score_map = {r.text.strip(): r.score for r in rerank_results}
                                        contexts = [r.text for r in rerank_results]
                                        logger.debug(
                                            f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                                            f"reranked {len(rerank_results)} candidates"
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"[pos_neg] Rerank failed for "
                                            f"{record.get('chunk_id', '')}: {e}, using embedding order"
                                        )

                                # chunk 评估（优先使用独立评估 LLM）
                                _eval_client = eval_llm_client or llm_client
                                eval_sample = EvaluationSample(
                                    input=record["query"],
                                    expected_output=answer_text,
                                    retrieval_context=contexts,
                                )
                                metric = ContextualPrecisionMetric()
                                eval_result = await metric.evaluate(eval_sample, llm_client=_eval_client, mode=chunk_eval_mode)
                                if self._stop_flag:
                                    return None
                                relevant_indices = set(
                                    eval_result.details.get("relevant_indices", [])
                                    if eval_result.details else []
                                )

                                # 收集 深度评估数据（AP 过滤之前，保留完整样本）
                                deep_eval_records.append({
                                    "query": record["query"],
                                    "expected_output": answer_text,
                                    "retrieval_context": list(contexts),
                                    "source_chunk_id": record.get("chunk_id", ""),
                                    "source_chunk_content": record.get("chunk_content", ""),
                                })

                                # 计算 per-chunk verdicts 和 AP
                                chunk_verdicts = [1 if i in relevant_indices else 0 for i in range(len(contexts))]
                                ap = self._calculate_average_precision(chunk_verdicts)

                                has_any_negative = any(v == 0 for v in chunk_verdicts)

                                # 训练价值过滤（仅在候选中有正有负时才有意义）
                                if has_any_negative and (skip_perfect_ap or skip_zero_ap):
                                    has_value, skip_reason = self._check_training_value(
                                        ap, skip_perfect=skip_perfect_ap, skip_zero=skip_zero_ap,
                                    )
                                    if not has_value:
                                        logger.debug(
                                            f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                                            f"skipped ({skip_reason}, AP={ap:.4f})"
                                        )
                                        skipped_ap += 1
                                        return None

                                # 按排名位置分类正负例
                                # 找到第一个和最后一个正例位置
                                first_pos_idx = -1
                                last_pos_idx = -1
                                for i, v in enumerate(chunk_verdicts):
                                    if v == 1:
                                        if first_pos_idx == -1:
                                            first_pos_idx = i
                                        last_pos_idx = i

                                very_hard_negs = []  # 排在所有正例之前
                                hard_negs = []       # 夹在正例之间
                                medium_negs = []     # 排在所有正例之后

                                for i, ctx in enumerate(contexts):
                                    ctx_key = ctx.strip()
                                    if ctx_key in seen_contents:
                                        continue

                                    if chunk_verdicts[i] == 1:
                                        if neg_detection_mode == "statement":
                                            # statement 模式：收集正例 chunk 用于后续语句提取
                                            if ctx_key not in seen_chunks:
                                                seen_chunks.add(ctx_key)
                                                positive_chunks_for_extraction.append(ctx)
                                        elif supplement_positives:
                                            # chunk 模式 + 补充正例：检索正例 chunk 加入 positives
                                            seen_contents.add(ctx_key)
                                            positives.append(ctx)
                                            # 收集正例 chunk 用于证据操作
                                            if (evidence_removal or evidence_pruning) and ctx_key not in seen_chunks:
                                                seen_chunks.add(ctx_key)
                                                positive_chunks_for_extraction.append(ctx)
                                    else:
                                        seen_contents.add(ctx_key)
                                        # 按位置分类负例
                                        if first_pos_idx == -1:
                                            very_hard_negs.append(ctx)
                                        elif i < first_pos_idx:
                                            very_hard_negs.append(ctx)
                                        elif i > last_pos_idx:
                                            medium_negs.append(ctx)
                                        else:
                                            hard_negs.append(ctx)

                                # 按难度优先级填充负例：very_hard > hard > medium
                                # skip_easy_negatives: 排在所有正例之后的负例模型已能区分，可跳过
                                negatives = very_hard_negs + hard_negs + ([] if skip_easy_negatives else medium_negs)

                                # 计算正例 chunk 的分数区间（供分层/筛选使用，支持 rerank / embedding）
                                best_pos = None
                                worst_pos = None
                                if relevant_indices and (neg_chunk_scoring or rerank_score_classification):
                                    pos_scores: List[float] = []
                                    if rerank_score_map:
                                        # 优先使用 rerank 分数（已在重排序时计算）
                                        for idx in relevant_indices:
                                            if idx < len(contexts):
                                                s = rerank_score_map.get(contexts[idx].strip())
                                                if s is not None:
                                                    pos_scores.append(float(s))
                                    elif embedding_client:
                                        # fallback: 使用 embedding 相似度
                                        try:
                                            all_sim = await embedding_client.similarity(
                                                record["query"], contexts,
                                            )
                                            if self._stop_flag:
                                                return None
                                            for idx in relevant_indices:
                                                if idx < len(all_sim):
                                                    pos_scores.append(float(all_sim[idx]))
                                        except Exception as e:
                                            logger.debug(f"[pos_neg] Embedding similarity for pos scores failed: {e}")
                                    if pos_scores:
                                        best_pos = max(pos_scores)
                                        worst_pos = min(pos_scores)

                                # 统一无负例检查（在语句提取前快速退出，避免无谓 LLM 开销）
                                # 覆盖场景：全部正例 / 完美排序+skip_easy_negatives 等
                                if not negatives and not augment:
                                    logger.debug(
                                        f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                                        f"skipped (no negatives after classification, AP={ap:.4f})"
                                    )
                                    skipped_ap += 1
                                    return None

                                # 特性 6: 负例 chunk 按 rerank 分数分层
                                auto_accept_negs = negatives
                                need_llm_negs: List[str] = []
                                scoring_client_available = rerank_client or embedding_client
                                if neg_chunk_scoring and scoring_client_available and hard_neg_extractor and worst_pos is not None:
                                    try:
                                        auto_accept_negs, need_llm_negs = await hard_neg_extractor.score_negative_chunks(
                                            record["query"], negatives, worst_pos,
                                            rerank_client=rerank_client, embedding_client=embedding_client,
                                        )
                                        if self._stop_flag:
                                            return None
                                        if neg_detection_mode != "statement":
                                            # chunk 模式：只保留高分 chunk（确定的难负例）
                                            negatives = auto_accept_negs
                                            logger.debug(
                                                f"[pos_neg] chunk mode neg_chunk_scoring: "
                                                f"kept {len(auto_accept_negs)}, dropped {len(need_llm_negs)} low-score chunks"
                                            )
                                    except Exception as e:
                                        logger.debug(f"[pos_neg] Neg chunk scoring failed: {e}")
                                        auto_accept_negs = negatives
                                        need_llm_negs = []

                                # 证据裁剪（正例 chunk → 更干净正例）
                                if evidence_pruning and hard_neg_extractor and positive_chunks_for_extraction:
                                    for pchunk in positive_chunks_for_extraction:
                                        try:
                                            pruned = await hard_neg_extractor.prune_evidence(
                                                record["query"], answer_text, pchunk, llm_client,
                                            )
                                            if self._stop_flag:
                                                return None
                                            if pruned and pruned.strip() not in seen_contents:
                                                seen_contents.add(pruned.strip())
                                                positives.append(pruned)
                                        except Exception as e:
                                            logger.debug(f"[pos_neg] Evidence pruning failed: {e}")

                                # statement 模式：先检查是否有可处理的负例 chunk，再做正例提取
                                # 避免 neg_chunk_scoring 清空后仍浪费 LLM 调用
                                has_neg_chunks = bool(auto_accept_negs) if neg_chunk_scoring else bool(negatives)
                                if neg_detection_mode == "statement" and not has_neg_chunks and not augment:
                                    # 证据去除也无法拯救（statement 模式下证据去除的结果也需要提取语句）
                                    if not evidence_removal:
                                        logger.debug(
                                            f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                                            f"skipped (no viable neg chunks after scoring, AP={ap:.4f})"
                                        )
                                        skipped_ap += 1
                                        return None

                                # statement 模式 + 补充正例：从正例 chunks 中提取支持性语句
                                if neg_detection_mode == "statement" and supplement_positives and hard_neg_extractor and positive_chunks_for_extraction:
                                    try:
                                        pos_statements = await hard_neg_extractor.extract_positives(
                                            query=record["query"],
                                            answer=answer_text,
                                            positive_chunks=positive_chunks_for_extraction,
                                            llm_client=llm_client,
                                            confirm=confirm_positives,
                                        )
                                        if self._stop_flag:
                                            return None
                                        # rerank 评分筛选 + 智能确认
                                        if pos_statements and rerank_score_classification and scoring_client_available and best_pos is not None and worst_pos is not None:
                                            try:
                                                scored_pos = await hard_neg_extractor.score_and_classify_statements(
                                                    record["query"], pos_statements, best_pos, worst_pos, mode="positive",
                                                    rerank_client=rerank_client, embedding_client=embedding_client,
                                                )
                                                if self._stop_flag:
                                                    return None
                                                accept_pos = [s["text"] for s in scored_pos if s.get("stmt_type") in ("gt_best", "ge_worst")]
                                                low_pos = [s for s in scored_pos if s.get("stmt_type") == "lt_worst"]

                                                # lt_worst 正例（分数 < 最差正例 chunk）：LLM 二次确认
                                                if low_pos:
                                                    low_stmts = [{"text": s["text"], "score": s["score"]} for s in low_pos]
                                                    # 取第一个正例 chunk 作为确认上下文
                                                    confirm_chunk = positive_chunks_for_extraction[0] if positive_chunks_for_extraction else ""
                                                    confirmed = await hard_neg_extractor._confirm_pos_statements(
                                                        record["query"], answer_text, confirm_chunk,
                                                        low_stmts, llm_client,
                                                    )
                                                    if self._stop_flag:
                                                        return None
                                                    accept_pos.extend([s["text"] for s in confirmed])

                                                pos_statements = accept_pos
                                            except Exception as e:
                                                logger.debug(f"[pos_neg] Positive statement scoring failed: {e}")

                                        if pos_statements:
                                            for stmt in pos_statements:
                                                stmt_key = stmt.strip()
                                                if stmt_key not in seen_contents:
                                                    seen_contents.add(stmt_key)
                                                    positives.append(stmt)
                                    except Exception as e:
                                        logger.warning(
                                            f"[pos_neg] Positive statement extraction failed for "
                                            f"{record.get('chunk_id', '')}: {e}"
                                        )

                                # statement 模式：从负例 chunks 中提取迷惑性语句（支持分层处理）
                                if neg_detection_mode == "statement" and hard_neg_extractor and negatives:
                                    try:
                                        all_neg_statements: List[str] = []

                                        # auto_accept chunks（分数高，确认是难负例）
                                        if auto_accept_negs:
                                            stmts = await hard_neg_extractor.extract(
                                                query=record["query"],
                                                answer=answer_text,
                                                negative_chunks=auto_accept_negs,
                                                llm_client=llm_client,
                                                rerank_client=rerank_client if confirm_negatives else None,
                                                embedding_client=embedding_client if (confirm_negatives and not rerank_client) else None,
                                                confirm=confirm_negatives and not neg_chunk_scoring,
                                            )
                                            if self._stop_flag:
                                                return None
                                            all_neg_statements.extend(stmts or [])

                                        # need_llm chunks（分数低）：直接丢弃，与 chunk 模式一致
                                        # 低分 chunk 模型已能区分，提取语句训练价值低

                                        # rerank 评分筛选负例语句
                                        if rerank_score_classification and scoring_client_available and best_pos is not None and worst_pos is not None and all_neg_statements:
                                            try:
                                                scored_neg = await hard_neg_extractor.score_and_classify_statements(
                                                    record["query"], all_neg_statements, best_pos, worst_pos, mode="negative",
                                                    rerank_client=rerank_client, embedding_client=embedding_client,
                                                )
                                                if self._stop_flag:
                                                    return None
                                                all_neg_statements = [s["text"] for s in scored_neg if s.get("stmt_type") in ("very_hard", "hard")]
                                            except Exception as e:
                                                logger.debug(f"[pos_neg] Negative statement scoring failed: {e}")

                                        if all_neg_statements:
                                            negatives = []
                                            for stmt in all_neg_statements:
                                                stmt_key = stmt.strip()
                                                if stmt_key not in seen_contents:
                                                    seen_contents.add(stmt_key)
                                                    negatives.append(stmt)
                                    except Exception as e:
                                        logger.warning(
                                            f"[pos_neg] Hard neg extraction failed for "
                                            f"{record.get('chunk_id', '')}: {e}, using chunk negatives"
                                        )

                                # 证据去除（正例 chunk → 硬负例变体）
                                # 放在 statement 提取之后，避免被 negatives 重置覆盖
                                if evidence_removal and hard_neg_extractor and positive_chunks_for_extraction:
                                    for pchunk in positive_chunks_for_extraction:
                                        try:
                                            removed = await hard_neg_extractor.remove_evidence(
                                                record["query"], answer_text, pchunk, llm_client,
                                            )
                                            if self._stop_flag:
                                                return None
                                            if removed and removed.strip() not in seen_contents:
                                                seen_contents.add(removed.strip())
                                                negatives.append(removed)
                                        except Exception as e:
                                            logger.debug(f"[pos_neg] Evidence removal failed: {e}")

                                logger.debug(
                                    f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                                    f"searched {len(candidate_chunks)}, AP={ap:.4f}, "
                                    f"pos={len(relevant_indices)}, "
                                    f"neg_very_hard={len(very_hard_negs)}, neg_hard={len(hard_negs)}, "
                                    f"neg_medium={len(medium_negs)}"
                                    f"{f', best_pos={best_pos:.4f}, worst_pos={worst_pos:.4f}' if best_pos is not None else ''}"
                                )

                                # 语句提取后二次检查：提取可能产生空结果（去重/过滤后为空）
                                if not negatives and not augment:
                                    logger.debug(
                                        f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                                        f"skipped (no negatives after statement extraction, AP={ap:.4f})"
                                    )
                                    skipped_ap += 1
                                    return None
                        except Exception as e:
                            logger.warning(
                                f"[pos_neg] Vector search/eval failed for "
                                f"{record.get('chunk_id', '')}: {e}"
                            )

                    # 检索后无负例兜底：检索返回 0 候选 / 搜索异常等场景
                    if use_retrieval and not negatives and not augment:
                        logger.debug(
                            f"[pos_neg] chunk={record.get('chunk_id', '')[:8]}: "
                            f"skipped (no negatives from retrieval, augment=False)"
                        )
                        skipped_ap += 1
                        return None

                    # LLM 生成正负例：
                    # - pos_neg_method=llm 时：LLM 是主要生成方式，必须执行
                    # - pos_neg_method=retrieval 时：仅 augment=True 时作为补充
                    need_llm = (
                        not use_retrieval  # llm 模式：LLM 是唯一来源
                        or (augment and (len(positives) < num_pos or len(negatives) < num_neg))  # retrieval 模式：增广补充
                    )
                    if need_llm and (len(positives) < num_pos or len(negatives) < num_neg):
                        try:
                            doc = Document(
                                content=record.get("chunk_content", ""),
                                metadata=record,
                                doc_id=record.get("chunk_id", ""),
                            )
                            processor = DocumentProcessor(
                                ProcessorConfig(
                                    generation_mode="qa_based",
                                    steps={
                                        **self.config.steps,
                                        "qa_gen": {"enabled": False},
                                        "doc_quality": {"enabled": False},
                                        "keypoint_gen": {"enabled": False},
                                        "role_gen": self.config.steps.get("role_gen", {"enabled": False}),
                                        "pos_neg_extraction": {
                                            **self.config.steps.get("pos_neg_extraction", {}),
                                            "num_positive": max(0, num_pos - len(positives)),
                                            "num_negative": max(0, num_neg - len(negatives)),
                                        },
                                    },
                                    custom_prompts=self.config.custom_prompts,
                                ),
                                llm_client=llm_client,
                            )

                            llm_samples = await asyncio.wait_for(
                                processor.process(doc),
                                timeout=self.config.timeout_per_doc,
                            )
                            if self._stop_flag:
                                return None
                            if llm_samples:
                                for p in llm_samples[0].positive_chunks:
                                    p_key = p.strip()
                                    if p_key not in seen_contents:
                                        seen_contents.add(p_key)
                                        positives.append(p)
                                for n in llm_samples[0].negative_chunks:
                                    n_key = n.strip()
                                    if n_key not in seen_contents:
                                        seen_contents.add(n_key)
                                        negatives.append(n)
                        except Exception as e:
                            logger.warning(
                                f"[pos_neg] LLM fallback failed for "
                                f"{record.get('chunk_id', '')}: {e}"
                            )

                    # llm 模式或 augment 模式：截断到目标数量
                    # retrieval 模式且不增广：使用检索实际结果
                    should_truncate = not use_retrieval or augment
                    final_pos = positives[:num_pos] if should_truncate else positives
                    final_neg = negatives[:num_neg] if should_truncate else negatives

                    sample = GeneratedSample(
                        query=record["query"],
                        answer=answer_text,
                        positive_chunks=final_pos,
                        negative_chunks=final_neg,
                        source_doc_id=record.get("chunk_id"),
                    )

                    # Incremental write for live preview
                    if _inc_path and _incremental_lock:
                        if self._stop_flag:
                            return None
                        formatted = self._format_samples([sample])
                        async with _incremental_lock:
                            if self._stop_flag:
                                return None
                            with open(_inc_path, "a", encoding="utf-8") as f:
                                for item in formatted:
                                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    return sample
                except Exception as e:
                    logger.warning(f"Pos/Neg generation failed for {record.get('chunk_id')}: {e}")
                    return None
                finally:
                    processed += 1
                    if resolved_progress_callback and not self._stop_flag:
                        resolved_progress_callback(processed, total)

        tasks = [generate_pos_neg(r) for r in filtered_records]
        results = await asyncio.gather(*tasks)

        if self._stop_flag:
            return [], []

        for sample in results:
            if sample:
                all_samples.append(sample)

        if skipped_ap > 0:
            logger.info(
                f"[pos_neg] Skipped {skipped_ap}/{total} records due to AP filter "
                f"(skip_perfect={skip_perfect_ap}, skip_zero={skip_zero_ap})"
            )

        if deep_eval_records:
            logger.info(f"[pos_neg] Collected {len(deep_eval_records)} deep eval records")

        return all_samples, deep_eval_records

    def _load_qa_input_records(self) -> List[Dict[str, Any]]:
        """加载 QA 数据集记录（从 config.input_path 读取原始输入）"""
        input_path = self._resolve_input_path_for_read(Path(self.config.input_path))

        if not input_path.exists():
            raise FileNotFoundError(f"QA dataset not found: {input_path}")

        file_format = self.config.input_format
        if file_format == "auto":
            suffix = input_path.suffix.lower()
            if suffix == ".json":
                file_format = "json"
            elif suffix == ".jsonl":
                file_format = "jsonl"
            else:
                file_format = "jsonl"

        records: List[Dict[str, Any]] = []
        if file_format == "json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                candidates = data
            else:
                candidates = [data]
            for record in candidates:
                if isinstance(record, dict) and "query" in record and "chunk_content" in record:
                    records.append(record)
                else:
                    logger.debug(
                        f"Skipping QA record missing query/chunk_content: "
                        f"{list(record.keys()) if isinstance(record, dict) else type(record)}"
                    )
            return records

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # 验证必需字段
                if "query" in record and "chunk_content" in record:
                    records.append(record)
                else:
                    logger.debug(f"Skipping QA record missing query/chunk_content: {list(record.keys())}")

        return records

    # ==================== 公共辅助方法 ====================

    async def _streaming_qa_producer(
        self,
        documents: List[Document],
        qa_queue: asyncio.Queue,
        log_prefix: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        流式 QA 提取 producer：并发处理文档，每个文档完成后立即将其 QA 记录推入队列。

        支持增量写入和文档级断点续传：
        - 每个文档的 QA 记录写入后立即追加到 JSONL 文件
        - 每条记录附带 _doc_idx 标记源文档索引
        - Resume 时根据已有文件跳过已处理的文档

        Returns:
            (all_qa_records, qa_output_path_str)
        """
        all_qa: List[Dict[str, Any]] = []
        llm_client = self._create_llm_client()
        semaphore = asyncio.Semaphore(self.config.llm_concurrency)
        lock = asyncio.Lock()
        batch: List[Dict[str, Any]] = []
        processed = 0
        total = len(documents)
        resolved_progress_callback = progress_callback or self.progress_callback
        dedup_enabled = self._is_dedup_enabled()
        seen_hashes: set = set()

        # 确定性文件路径
        qa_file_path = self._get_qa_output_path()

        # ── 断点续传检测 ──
        completed_doc_idxs: set = set()
        if qa_file_path.exists() and qa_file_path.stat().st_size > 0:
            _, loaded_hashes, existing_records = self._load_resume_state(str(qa_file_path))
            all_qa.extend(existing_records)
            seen_hashes.update(loaded_hashes)
            completed_doc_idxs = {
                r.get("_doc_idx", -1)
                for r in existing_records
                if r.get("_doc_idx", -1) >= 0
            }
            logger.info(
                "%s Resuming: loaded %d QA records, %d docs already completed "
                "(skip by exact index set, not a max watermark)",
                log_prefix, len(existing_records), len(completed_doc_idxs),
            )
            # Replay 已有记录到 queue（供下游 filter/pos_neg consumer 处理）
            replay_clean = [
                {k: v for k, v in r.items() if k != "_doc_idx"}
                for r in existing_records
            ]
            for i in range(0, len(replay_clean), 50):
                if not await self._put_queue_data_unless_stopped(
                    qa_queue,
                    replay_clean[i : i + 50],
                ):
                    break
            # 进度起点 = 已完成文档数
            processed = len(completed_doc_idxs)
            if resolved_progress_callback and not self._stop_flag:
                resolved_progress_callback(processed, total)

        async def process_one(doc: Document, doc_idx: int):
            nonlocal processed
            if self._stop_flag:
                return
            # 跳过已完成的文档：按具体索引集合，而非最大值水位线。
            # 并发乱序完成时，低于最大值但未真正完成的文档必须重跑，否则永久丢失。
            if doc_idx in completed_doc_idxs:
                return
            async with semaphore:
                if self._stop_flag:
                    return
                try:
                    processor = DocumentProcessor(
                        ProcessorConfig(
                            generation_mode="qa_extraction",
                            steps=self.config.steps,
                            custom_prompts=self.config.custom_prompts,
                        ),
                        llm_client=llm_client,
                    )
                    samples = await asyncio.wait_for(
                        processor.process(doc),
                        timeout=self.config.timeout_per_doc,
                    )
                    if self._stop_flag:
                        return
                    records = [self._sample_to_qa_record(s) for s in samples]
                    records = [r for r in records if r]

                    if records:
                        # 为每条记录添加文档索引
                        for r in records:
                            r["_doc_idx"] = doc_idx

                        batches_to_send: List[List[Dict[str, Any]]] = []
                        async with lock:
                            if self._stop_flag:
                                return
                            if dedup_enabled:
                                unique_records = []
                                for r in records:
                                    h = self._qa_dedup_hash(r)
                                    if h not in seen_hashes:
                                        seen_hashes.add(h)
                                        unique_records.append(r)
                                records = unique_records
                            if self._stop_flag:
                                return
                            all_qa.extend(records)
                            # 增量写入 JSONL 文件
                            if self._stop_flag:
                                return
                            with open(qa_file_path, "a", encoding="utf-8") as f:
                                for r in records:
                                    f.write(json.dumps({
                                        "query": r.get("query", ""),
                                        "answer": r.get("answer", ""),
                                        "chunk_id": r.get("chunk_id", ""),
                                        "chunk_content": r.get("chunk_content", ""),
                                        "_doc_idx": r.get("_doc_idx", -1),
                                    }, ensure_ascii=False) + "\n")
                            # 发送到 queue 时剥离 _doc_idx
                            clean = [{k: v for k, v in r.items() if k != "_doc_idx"} for r in records]
                            batch.extend(clean)
                            while len(batch) >= 50:
                                batches_to_send.append(batch[:50])
                                del batch[:50]
                        for b in batches_to_send:
                            if not await self._put_queue_data_unless_stopped(
                                qa_queue,
                                b,
                            ):
                                return
                except asyncio.TimeoutError:
                    logger.warning(f"{log_prefix} QA extraction timeout for doc {doc.doc_id}")
                except Exception as e:
                    logger.warning(f"{log_prefix} QA extraction failed for doc {doc.doc_id}: {e}")
                finally:
                    processed += 1
                    if resolved_progress_callback and not self._stop_flag:
                        resolved_progress_callback(processed, total)

        try:
            async with llm_client:
                if self._stop_flag:
                    return all_qa, str(qa_file_path)
                doc_tasks = [process_one(doc, idx) for idx, doc in enumerate(documents)]
                await asyncio.gather(*doc_tasks)
                # 刷出剩余不足 50 条的 batch
                if not self._stop_flag:
                    async with lock:
                        pending_batch = list(batch)
                    if pending_batch and await self._put_queue_data_unless_stopped(
                        qa_queue,
                        pending_batch,
                    ):
                        batch.clear()
        finally:
            await qa_queue.put(None)  # 哨兵

        # QA 提取完成，文件已写完，立即触发注册回调
        # 此时 consumer 仍在处理 queue 中的剩余 batch
        if self.on_qa_complete and all_qa and not self._stop_flag:
            try:
                callback_result = self.on_qa_complete(
                    str(qa_file_path),
                    len(all_qa),
                )
                if inspect.isawaitable(callback_result):
                    await callback_result
            except Exception as e:
                logger.warning(f"on_qa_complete callback failed: {e}")

        return all_qa, str(qa_file_path)

    def _build_chunk_metadata(self) -> Optional[Dict[str, Any]]:
        """构建插入 Milvus 时附带的 chunk 元数据"""
        meta: Dict[str, Any] = {}
        if self.config.source_dataset_id:
            meta["dataset_id"] = self.config.source_dataset_id
        if self.config.task_id:
            meta["task_id"] = self.config.task_id
        return meta or None

    def _init_phase2_clients(self) -> Tuple[EmbeddingClient, Optional[MilvusClient], Optional[str], Optional[RerankClient]]:
        """初始化 embedding/milvus/rerank 客户端"""
        emb_cfg = self.config.embedding_config or {}
        embedding_endpoint = validate_user_outbound_url(
            emb_cfg.get("endpoint", ""),
            self.config.user_id,
        )
        embedding_client = _ValidatedEmbeddingClient(EmbeddingConfig(
            endpoint=embedding_endpoint,
            model=emb_cfg.get("model", ""),
            api_key=emb_cfg.get("api_key"),
            batch_size=emb_cfg.get("batch_size", 32),
            concurrency=emb_cfg.get("concurrency", self.config.embedding_concurrency),
        ), self.config.user_id)

        milvus_client = None
        collection_name = None
        milvus_cfg = self.config.milvus_config or {}
        if milvus_cfg.get("host") or os.environ.get("MILVUS_HOST"):
            milvus_client = MilvusClient(MilvusConfig(
                host=milvus_cfg.get("host", ""),
                port=milvus_cfg.get("port", 19530),
                token=milvus_cfg.get("token"),
            ))
            milvus_client.connect()

            if self.config.existing_collection_name:
                # Use the pre-existing collection directly
                collection_name = self.config.existing_collection_name
            else:
                # Auto-generate collection name from dataset + embedding hash
                dataset_name = self.config.source_dataset_id or Path(self.config.input_path).stem[:16]
                emb_model = emb_cfg.get("model", "unknown")
                emb_hash = hashlib.md5(emb_model.encode()).hexdigest()[:8]
                collection_name = MilvusClient.sanitize_collection_name(
                    f"tf_{dataset_name[:8]}_{emb_hash}"
                )

        rerank_client = None
        rerank_cfg = self.config.rerank_config
        if rerank_cfg:
            rerank_endpoint = validate_user_outbound_url(
                rerank_cfg.get("endpoint", ""),
                self.config.user_id,
            )
            rerank_client = _ValidatedRerankClient(RerankConfig(
                endpoint=rerank_endpoint,
                model=rerank_cfg.get("model", ""),
                api_key=rerank_cfg.get("api_key"),
                top_k=rerank_cfg.get("top_k", 10),
                batch_size=rerank_cfg.get("batch_size", 64),
                concurrency=rerank_cfg.get("concurrency", 10),
            ), self.config.user_id)

        return embedding_client, milvus_client, collection_name, rerank_client

    @staticmethod
    def _sample_to_qa_record(sample: GeneratedSample) -> Optional[Dict[str, Any]]:
        """GeneratedSample → QA dict"""
        if not sample.query:
            return None
        return {
            "query": sample.query,
            "answer": sample.answer or "",
            "chunk_id": sample.source_doc_id or "",
            "chunk_content": sample.metadata.get("chunk_content", "") if sample.metadata else "",
        }

    # ==================== 增量写入辅助方法 ====================

    def _get_output_dir(self) -> Path:
        """获取输出目录"""
        if self.config.output_path:
            output_dir = self._resolve_output_path().parent
        else:
            fallback = (
                Path(self.config.input_path).parent
                / "output"
                / "generated.jsonl"
            )
            output_dir = resolve_generation_attempt_output_path(
                str(fallback),
                self.config.task_id,
                self.config.run_token,
            ).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_qa_output_path(self) -> Path:
        """基于 task_id 的确定性 QA 输出路径"""
        tag = self.config.task_id if self.config.task_id else uuid.uuid4().hex
        return self._get_output_dir() / f"qa_extracted_{tag}.jsonl"

    def _get_qa_filtered_path(self) -> Path:
        """基于 task_id 的确定性过滤输出路径"""
        tag = self.config.task_id if self.config.task_id else uuid.uuid4().hex
        return self._get_output_dir() / f"qa_filtered_{tag}.jsonl"

    def _get_deep_eval_path(self) -> Path:
        """基于 task_id 的深度评估数据输出路径"""
        tag = self.config.task_id if self.config.task_id else uuid.uuid4().hex
        return self._get_output_dir() / f"deep_eval_{tag}.jsonl"

    def _save_deep_eval_data(self, records: List[Dict[str, Any]], label: str = "") -> Optional[str]:
        """保存深度评估数据到 JSONL 文件，返回文件路径。无数据时返回 None。"""
        if not records:
            return None
        deep_eval_path = str(self._get_deep_eval_path())
        with open(deep_eval_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"{label} Saved {len(records)} deep eval records to {deep_eval_path}")
        return deep_eval_path

    @staticmethod
    def _load_resume_state(qa_path: str) -> Tuple[int, set, List[Dict[str, Any]]]:
        """从增量 JSONL 加载 resume 状态。

        Returns:
            (记录数, seen_hashes, records)
        """
        records: List[Dict[str, Any]] = []
        seen_hashes: set = set()
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # 崩溃时最后一行可能不完整，跳过
                    logger.warning("Skipping malformed line in %s: %s", qa_path, line[:80])
                    continue
                records.append(record)
                if record.get("query"):
                    seen_hashes.add(DatasetGenerationPipeline._qa_dedup_hash(record))
        return len(records), seen_hashes, records

    def _save_qa_output(self, qa_records: List[Dict[str, Any]]) -> str:
        """保存 QA 中间数据集"""
        output_dir = self._get_output_dir()
        qa_path = output_dir / f"qa_extracted_{uuid.uuid4().hex[:8]}.jsonl"

        with open(qa_path, "w", encoding="utf-8") as f:
            for record in qa_records:
                f.write(json.dumps({
                    "query": record.get("query", ""),
                    "answer": record.get("answer", ""),
                    "chunk_id": record.get("chunk_id", ""),
                    "chunk_content": record.get("chunk_content", ""),
                }, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(qa_records)} QA records to {qa_path}")
        return str(qa_path)

    @staticmethod
    def _load_qa_records(qa_path: str) -> List[Dict[str, Any]]:
        """从 JSONL 文件加载 QA 记录（用于断点续传），自动剥离内部 _doc_idx 字段"""
        records = []
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record.pop("_doc_idx", None)
                records.append(record)
        logger.info(f"Loaded {len(records)} QA records from {qa_path}")
        return records

    @staticmethod
    async def _cleanup_clients(
        milvus_client: Optional[MilvusClient],
        embedding_client: Optional[EmbeddingClient],
        rerank_client: Optional[RerankClient] = None,
    ):
        """清理客户端连接"""
        if milvus_client:
            try:
                milvus_client.close()
            except Exception:
                pass
        if embedding_client:
            try:
                await embedding_client.close()
            except Exception:
                pass
        if rerank_client:
            try:
                await rerank_client.close()
            except Exception:
                pass

    # ==================== 公共辅助方法（QA 中间数据集）====================

    def _save_qa_intermediate(
        self,
        qa_records: List[Dict[str, Any]],
        suffix: str = "qa",
    ) -> str:
        """保存中间 QA 数据集为 JSONL（区分全量 / 筛选后）"""
        output_dir = self._get_output_dir()

        output_path = output_dir / f"{suffix}_{uuid.uuid4().hex[:8]}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for record in qa_records:
                f.write(json.dumps({
                    "query": record.get("query", ""),
                    "answer": record.get("answer", ""),
                    "chunk_id": record.get("chunk_id", ""),
                    "chunk_content": record.get("chunk_content", ""),
                }, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(qa_records)} QA records to {output_path}")
        return str(output_path)
