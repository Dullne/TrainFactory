"""Milvus 向量库管理 API 路由"""

import hashlib
import logging
import socket
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...auth.dependencies import get_current_user
from ...generation.clients.milvus_client import MilvusClient, MilvusConfig
from ...storage.services.dataset_service import (
    DatasetConsumptionUnavailableError,
    dataset_service,
)
from ...storage.services.generation_task_service import generation_task_service
from ...storage.services.deep_evaluation_task_service import (
    deep_evaluation_task_service,
)
from ...storage.services.external_sync_service import external_sync_service
from ...storage.services.milvus_collection_service import (
    MilvusCollectionDeletionOwnerConflictError,
    MilvusCollectionUnavailableError,
    milvus_collection_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic Models ──────────────────────────────────────────

class LinkedDatasetInfo(BaseModel):
    dataset_id: str
    dataset_name: Optional[str] = None
    chunk_count: int = 0
    task_id: Optional[str] = None


class CollectionSummary(BaseModel):
    name: str
    num_entities: int = 0
    dim: Optional[int] = None
    index_type: Optional[str] = None
    metric_type: Optional[str] = None
    load_state: str = "NotLoad"
    description: Optional[str] = None
    hybrid_enabled: bool = False
    metadata_enabled: bool = False
    # Registry fields
    collection_id: Optional[str] = None
    display_name: Optional[str] = None
    embedding_config_id: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    linked_datasets: List[LinkedDatasetInfo] = []
    status: Optional[str] = None
    # Backward compat (derived from first linked dataset)
    source_dataset_id: Optional[str] = None
    source_dataset_name: Optional[str] = None
    associated_task_id: Optional[str] = None


class CollectionDetail(CollectionSummary):
    schema_fields: List[Dict[str, Any]] = []
    indexes: List[Dict[str, Any]] = []


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Collection 名称")
    dim: int = Field(..., ge=1, le=4096, description="向量维度")
    metric_type: str = Field(default="COSINE", description="距离度量: COSINE, L2, IP")
    description: Optional[str] = Field(default=None, max_length=500)
    embedding_config_id: Optional[str] = Field(default=None, description="绑定的 Embedding 模型配置 ID")
    display_name: Optional[str] = Field(default=None, max_length=255, description="显示名称")
    enable_hybrid: bool = Field(default=False, description="启用混合搜索（BM25 + 向量），创建后不可更改")


class SearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1, max_length=8192, description="查询文本")
    embedding_config_id: Optional[str] = Field(default=None, description="Embedding 模型配置 ID（sparse 模式可省略）")
    top_k: int = Field(default=10, ge=1, le=100)
    search_mode: str = Field(default="dense", description="搜索模式: dense, sparse, hybrid")
    ranker: str = Field(default="rrf", description="融合策略: rrf, weighted（仅 hybrid 模式）")
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Dense 权重（weighted 融合）")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Sparse 权重（weighted 融合）")
    filter_expr: Optional[str] = Field(default=None, max_length=4096, description="元数据过滤表达式，如 metadata[\"dataset_id\"] == \"ds_123\"")


# ── Helpers ──────────────────────────────────────────────────

def _is_milvus_connection_error(exc: Exception) -> bool:
    """Classify transport failures without hiding auth/config/program bugs."""
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            socket.gaierror,
        ),
    ):
        return True

    exception_type = type(exc)
    exception_module = exception_type.__module__
    is_pymilvus_error = exception_module.startswith("pymilvus")
    is_grpc_rpc_error = (
        exception_module == "grpc" or exception_module.startswith("grpc.")
    ) and exception_type.__name__.endswith("RpcError")
    if not is_pymilvus_error and not is_grpc_rpc_error:
        return False
    if is_pymilvus_error and exception_type.__name__ in {
        "ConnectError",
        "MilvusUnavailableException",
    }:
        return True
    # Pymilvus wraps gRPC channel-readiness failures in MilvusException with
    # client Status.CONNECT_FAILED (2). Newer releases can preserve the gRPC
    # status enum, while deadline failures can escape directly as RpcError.
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            return False
    if is_pymilvus_error and code == 2:
        return True
    status_name = getattr(code, "name", None)
    if not isinstance(status_name, str) and isinstance(code, str):
        status_name = code.rsplit(".", 1)[-1]
    return isinstance(status_name, str) and status_name.upper() in {
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
    }


def _close_partial_milvus_client(client: MilvusClient) -> None:
    try:
        client.close()
    except Exception as close_error:
        logger.warning(
            "Failed to close partially initialized Milvus client (%s)",
            type(close_error).__name__,
        )


def _get_milvus_client() -> MilvusClient:
    """Create a client and expose only confirmed transport failures as 503."""
    client = MilvusClient(MilvusConfig())
    try:
        client.connect()
    except Exception as exc:
        _close_partial_milvus_client(client)
        if _is_milvus_connection_error(exc):
            logger.warning(
                "Milvus connection failed (%s)",
                type(exc).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="Milvus service is unavailable",
            ) from exc
        raise
    return client


class LinkDatasetRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=64, description="数据集 ID")
    dataset_name: Optional[str] = Field(default=None, max_length=255, description="数据集名称")
    chunk_count: int = Field(default=0, ge=0, le=2_147_483_647, description="chunk 数量")
    task_id: Optional[str] = Field(default=None, max_length=64, description="关联的任务 ID")


def _is_auth_disabled(current_user: Dict[str, Any]) -> bool:
    return current_user.get("user_id") in (None, "anonymous")


def _verify_owned_resource(
    resource: Optional[Dict[str, Any]],
    current_user: Dict[str, Any],
    resource_name: str,
) -> Dict[str, Any]:
    if not resource:
        raise HTTPException(status_code=404, detail=f"{resource_name} not found")
    if _is_auth_disabled(current_user):
        return resource
    if resource.get("user_id") != current_user.get("user_id"):
        raise HTTPException(
            status_code=403,
            detail=f"Not authorized to access this {resource_name.lower()}",
        )
    return resource


def _get_owned_collection(
    collection_name: str,
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    registered = milvus_collection_service.get_by_name(collection_name)
    if not registered and _is_auth_disabled(current_user):
        return {"collection_name": collection_name, "user_id": None}
    return _verify_owned_resource(registered, current_user, "Collection")


def _validate_manual_collection_reconciliation(
    collection_name: str,
    remote_info: Dict[str, Any],
    *,
    expected_dim: int,
    expected_metric_type: str,
    expected_hybrid_enabled: bool,
) -> None:
    """Require a pre-existing remote collection to match a durable intent."""
    mismatches: List[str] = []
    schema = remote_info.get("schema")
    schema_fields = schema if isinstance(schema, list) else []
    fields_by_name = {
        field.get("name"): field
        for field in schema_fields
        if isinstance(field, dict) and isinstance(field.get("name"), str)
    }
    vector_field = next(
        (
            field
            for field in schema_fields
            if isinstance(field, dict) and field.get("name") == "vector"
        ),
        None,
    )
    remote_dim = (
        vector_field.get("dim")
        if vector_field is not None and vector_field.get("dim") is not None
        else remote_info.get("dim")
    )
    try:
        dimension_matches = int(remote_dim) == expected_dim
    except (TypeError, ValueError):
        dimension_matches = False
    if not dimension_matches:
        mismatches.append("dimension")
    if schema_fields and vector_field is None:
        mismatches.append("vector schema")
    if schema_fields:
        required_types = {
            "chunk_id": "VARCHAR",
            "chunk_content": "VARCHAR",
            "vector": "FLOAT_VECTOR",
            "metadata": "JSON",
        }
        base_schema_matches = all(
            str(fields_by_name.get(name, {}).get("dtype") or "").upper()
            == expected_type
            for name, expected_type in required_types.items()
        ) and fields_by_name.get("chunk_id", {}).get("is_primary") is True
        if not base_schema_matches:
            mismatches.append("base schema")
    else:
        mismatches.append("base schema")

    remote_metric = remote_info.get("metric_type")
    indexes = remote_info.get("indexes")
    index_rows = indexes if isinstance(indexes, list) else []
    vector_index = None
    sparse_index = None
    if isinstance(indexes, list):
        vector_index = next(
            (
                index
                for index in indexes
                if isinstance(index, dict)
                and index.get("field_name") == "vector"
            ),
            None,
        )
        sparse_index = next(
            (
                index
                for index in index_rows
                if isinstance(index, dict)
                and index.get("field_name") == "sparse_vector"
            ),
            None,
        )
        if vector_index is not None:
            remote_metric = vector_index.get("metric_type")
            if remote_metric is None and isinstance(
                vector_index.get("params"), dict
            ):
                remote_metric = vector_index["params"].get("metric_type")
    if str(remote_metric or "").strip().upper() != str(
        expected_metric_type
    ).strip().upper():
        mismatches.append("metric")

    remote_hybrid: Optional[bool] = None
    if isinstance(remote_info.get("hybrid_enabled"), bool):
        remote_hybrid = remote_info["hybrid_enabled"]
    elif schema_fields:
        remote_hybrid = any(
            isinstance(field, dict) and field.get("name") == "sparse_vector"
            for field in schema_fields
        )
    sparse_field = fields_by_name.get("sparse_vector")
    if expected_hybrid_enabled:
        sparse_metric = None
        if sparse_index is not None:
            sparse_metric = sparse_index.get("metric_type")
            if sparse_metric is None and isinstance(
                sparse_index.get("params"), dict
            ):
                sparse_metric = sparse_index["params"].get("metric_type")
        hybrid_complete = (
            remote_hybrid is True
            and sparse_field is not None
            and str(sparse_field.get("dtype") or "").upper()
            == "SPARSE_FLOAT_VECTOR"
            and str(sparse_metric or "").upper() == "BM25"
        )
    else:
        hybrid_complete = (
            remote_hybrid is False
            and sparse_field is None
            and sparse_index is None
        )
    if not hybrid_complete:
        mismatches.append("hybrid schema")

    if mismatches:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Remote collection '{collection_name}' does not match the "
                "durable creation reservation: " + ", ".join(mismatches)
            ),
        )


def _compute_recall_summary(
    results: List[Dict[str, Any]],
    expected_chunk_ids: List[str],
    top_k: int,
) -> Dict[str, Any]:
    """Compute recall against unique, non-empty expected chunk IDs."""
    gold_ids: List[str] = []
    seen_gold = set()
    for raw_id in expected_chunk_ids:
        chunk_id = str(raw_id or "").strip()
        if chunk_id and chunk_id not in seen_gold:
            seen_gold.add(chunk_id)
            gold_ids.append(chunk_id)

    retrieved_ids = {
        str(result.get("chunk_id") or "").strip()
        for result in results[:max(int(top_k), 0)]
        if str(result.get("chunk_id") or "").strip()
    }
    hit_ids = [chunk_id for chunk_id in gold_ids if chunk_id in retrieved_ids]
    miss_ids = [chunk_id for chunk_id in gold_ids if chunk_id not in retrieved_ids]

    return {
        "gold_count": len(gold_ids),
        "hit_count": len(hit_ids),
        "recall_at_k": len(hit_ids) / len(gold_ids) if gold_ids else 0.0,
        "hit_chunk_ids": hit_ids,
        "miss_chunk_ids": miss_ids,
    }


def _build_registry_map(user_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    构建 collection_name -> registry info 映射（从集合注册表读取）。
    包含 linked_datasets 和 embedding 信息。
    """
    registry: Dict[str, Dict[str, Any]] = {}
    page_size = 500
    offset = 0
    while True:
        collections, total = milvus_collection_service.list_collections(
            user_id=user_id,
            limit=page_size,
            offset=offset,
        )
        if not collections:
            break
        for coll in collections:
            coll_name = coll["collection_name"]
            linked = milvus_collection_service.get_linked_datasets(coll_name)
            registry[coll_name] = {
                **coll,
                "linked_datasets": linked,
            }
        offset += len(collections)
        if offset >= total:
            break
    return registry


def _info_to_summary(
    info: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
) -> CollectionSummary:
    """将 Milvus get_collection_info 结果 + 注册表数据合并为 CollectionSummary"""
    name = info["name"]
    reg = registry.get(name, {})

    index_type = None
    metric_type = None
    if info.get("indexes"):
        idx = info["indexes"][0]
        index_type = idx.get("index_type")
        metric_type = idx.get("metric_type")

    # Build linked datasets
    linked_datasets = []
    for link in reg.get("linked_datasets", []):
        linked_datasets.append(LinkedDatasetInfo(
            dataset_id=link["dataset_id"],
            dataset_name=link.get("dataset_name"),
            chunk_count=link.get("chunk_count", 0),
            task_id=link.get("task_id"),
        ))

    # Backward compat: first linked dataset as source
    first_link = linked_datasets[0] if linked_datasets else None

    return CollectionSummary(
        name=name,
        num_entities=info.get("num_entities", 0),
        dim=info.get("dim"),
        index_type=index_type,
        metric_type=metric_type or reg.get("metric_type"),
        load_state=info.get("load_state", "NotLoad"),
        description=reg.get("description") or info.get("description"),
        hybrid_enabled=info.get("hybrid_enabled", False),
        metadata_enabled=info.get("metadata_enabled", False),
        collection_id=reg.get("collection_id"),
        display_name=reg.get("display_name"),
        embedding_config_id=reg.get("embedding_config_id"),
        embedding_model=reg.get("embedding_model"),
        embedding_endpoint=reg.get("embedding_endpoint"),
        linked_datasets=linked_datasets,
        status=reg.get("status"),
        source_dataset_id=first_link.dataset_id if first_link else None,
        source_dataset_name=first_link.dataset_name if first_link else None,
        associated_task_id=first_link.task_id if first_link else None,
    )


# ── API Endpoints ────────────────────────────────────────────

@router.get("/status")
async def get_milvus_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取 Milvus 连接状态"""
    config = MilvusConfig()
    client = None
    show_operator_details = _is_auth_disabled(current_user) or bool(
        current_user.get("is_admin")
    )
    try:
        try:
            client = _get_milvus_client()
        except HTTPException as exc:
            if (
                exc.status_code != 503
                or exc.detail != "Milvus service is unavailable"
            ):
                raise
            logger.warning("Milvus status check failed (HTTPException)")
            response = {
                "connected": False,
                "error": "Milvus is unavailable",
            }
            if show_operator_details:
                response.update({"host": config.host, "port": config.port})
            return response

        try:
            collections = client.list_collections()
        except Exception as exc:
            if not _is_milvus_connection_error(exc):
                raise
            logger.warning(
                "Milvus status check failed (%s)",
                type(exc).__name__,
            )
            response = {
                "connected": False,
                "error": "Milvus is unavailable",
            }
            if show_operator_details:
                response.update({"host": config.host, "port": config.port})
            return response

        if not show_operator_details:
            registry = _build_registry_map(user_id=current_user.get("user_id"))
            collections = [name for name in collections if name in registry]
        response = {
            "connected": True,
            "collection_count": len(collections),
        }
        if show_operator_details:
            response.update({"host": config.host, "port": config.port})
        return response
    finally:
        if client is not None:
            client.close()


@router.get("/collections")
async def list_collections(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """列出所有 Milvus 集合（合并注册表元数据 + Milvus 实时状态）"""
    client = _get_milvus_client()
    try:
        names = client.list_collections()
        user_id = None if _is_auth_disabled(current_user) else current_user.get("user_id")
        registry = _build_registry_map(user_id=user_id)
        if user_id:
            names = [name for name in names if name in registry]

        collections: List[Dict[str, Any]] = []
        for name in sorted(names):
            try:
                info = client.get_collection_info(name)
                if info:
                    collections.append(
                        _info_to_summary(info, registry).model_dump()
                    )
            except Exception as e:
                logger.warning("Failed to get info for collection '%s': %s", name, e)
                collections.append(CollectionSummary(name=name).model_dump())

        return {"collections": collections, "total": len(collections)}
    finally:
        client.close()


@router.get("/collections/{collection_name}")
async def get_collection(
    collection_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取集合详情（注册表 + Milvus 实时状态）"""
    reg = _get_owned_collection(collection_name, current_user)
    client = _get_milvus_client()
    try:
        info = client.get_collection_info(collection_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' 不存在")

        # Get registry data for this collection
        linked = milvus_collection_service.get_linked_datasets(collection_name)

        index_type = None
        metric_type = None
        if info.get("indexes"):
            idx = info["indexes"][0]
            index_type = idx.get("index_type")
            metric_type = idx.get("metric_type")

        linked_datasets = [
            LinkedDatasetInfo(
                dataset_id=link["dataset_id"],
                dataset_name=link.get("dataset_name"),
                chunk_count=link.get("chunk_count", 0),
                task_id=link.get("task_id"),
            )
            for link in linked
        ]
        first_link = linked_datasets[0] if linked_datasets else None

        return CollectionDetail(
            name=info["name"],
            num_entities=info.get("num_entities", 0),
            dim=info.get("dim"),
            index_type=index_type,
            metric_type=metric_type or reg.get("metric_type"),
            load_state=info.get("load_state", "NotLoad"),
            description=reg.get("description") or info.get("description"),
            hybrid_enabled=info.get("hybrid_enabled", False),
            metadata_enabled=info.get("metadata_enabled", False),
            schema_fields=info.get("schema", []),
            indexes=info.get("indexes", []),
            collection_id=reg.get("collection_id"),
            display_name=reg.get("display_name"),
            embedding_config_id=reg.get("embedding_config_id"),
            embedding_model=reg.get("embedding_model"),
            embedding_endpoint=reg.get("embedding_endpoint"),
            linked_datasets=linked_datasets,
            status=reg.get("status"),
            source_dataset_id=first_link.dataset_id if first_link else None,
            source_dataset_name=first_link.dataset_name if first_link else None,
            associated_task_id=first_link.task_id if first_link else None,
        ).model_dump()
    finally:
        client.close()


@router.post("/collections")
async def create_collection(
    request: CreateCollectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """创建新集合并注册到数据库"""
    from ...storage.services.model_config_service import model_config_service

    sanitized_name = MilvusClient.sanitize_collection_name(request.name)
    reservation_user_id = current_user.get("user_id")
    try:
        durable_reservation = (
            milvus_collection_service.get_manual_creation_reservation(
                sanitized_name,
                user_id=reservation_user_id,
            )
        )
    except MilvusCollectionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if durable_reservation is not None:
        # Recovery must not depend on a mutable/deleted model-config row.
        embedding_model = durable_reservation.get("embedding_model")
        embedding_endpoint = durable_reservation.get("embedding_endpoint")
    else:
        embedding_model = None
        embedding_endpoint = None
        if request.embedding_config_id:
            config = _verify_owned_resource(
                model_config_service.get_config(request.embedding_config_id),
                current_user,
                "Embedding config",
            )
            if config.get("model_type") != "embedding":
                raise HTTPException(
                    status_code=400,
                    detail="Config is not an embedding config",
                )
            embedding_model = config.get("model_name")
            embedding_endpoint = config.get("api_endpoint")

    try:
        reservation = milvus_collection_service.reserve_manual_collection(
            collection_name=sanitized_name,
            embedding_config_id=request.embedding_config_id,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
            dim=request.dim,
            metric_type=request.metric_type,
            hybrid_enabled=request.enable_hybrid,
            display_name=request.display_name or sanitized_name,
            description=request.description,
            user_id=current_user.get("user_id"),
        )
    except MilvusCollectionUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    reservation_id = reservation["collection_id"]
    reservation_newly_created = bool(
        reservation.get("_newly_created", True)
    )
    reservation_dim = reservation.get("dim", request.dim)
    reservation_metric = reservation.get(
        "metric_type", request.metric_type
    )
    reservation_hybrid = reservation.get("hybrid_enabled")
    if not isinstance(reservation_hybrid, bool):
        if not reservation_newly_created:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Collection '{sanitized_name}' has an ambiguous legacy "
                    "hybrid creation intent"
                ),
            )
        reservation_hybrid = request.enable_hybrid
    client = None
    remote_mutation_started = False
    try:
        namespace_consumers = (
            external_sync_service.list_active_collection_consumers(
                [sanitized_name]
            )
        )
        if namespace_consumers:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Collection '{sanitized_name}' is reserved by an "
                    "external sync task"
                ),
            )

        client = _get_milvus_client()
        remote_exists = client.collection_exists(sanitized_name)
        if remote_exists:
            if reservation_newly_created:
                raise HTTPException(
                    status_code=409,
                    detail=f"Collection '{sanitized_name}' already exists",
                )
        else:
            remote_mutation_started = True
            created_remotely = client.create_collection(
                sanitized_name,
                int(reservation_dim),
                metric_type=str(reservation_metric),
                enable_hybrid=reservation_hybrid,
            )
            if created_remotely is False:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Collection '{sanitized_name}' appeared concurrently; "
                        "the durable reservation was retained for cleanup"
                    ),
                )

        # Always re-read after create: the client may silently skip when a
        # concurrent actor wins its inner existence check, and a failed RPC
        # may leave only a partial schema/index build behind.
        remote_info = client.get_collection_info(sanitized_name)
        if not remote_info:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Remote collection '{sanitized_name}' could not be "
                    "reconciled with its durable creation reservation"
                ),
            )
        _validate_manual_collection_reconciliation(
            sanitized_name,
            remote_info,
            expected_dim=int(reservation_dim),
            expected_metric_type=str(reservation_metric),
            expected_hybrid_enabled=reservation_hybrid,
        )

        reg = milvus_collection_service.activate_manual_collection(
            sanitized_name,
            collection_id=reservation_id,
            user_id=reservation_user_id,
        )

        return {
            "name": sanitized_name,
            "collection_id": reg.get("collection_id"),
            "message": f"Collection '{sanitized_name}' 创建成功",
        }
    except BaseException:
        # Only this request's pre-mutation reservation is safe to cancel. A
        # reused row is durable recovery state from an earlier uncertain call.
        if reservation_newly_created and not remote_mutation_started:
            milvus_collection_service.cancel_manual_collection_creation(
                sanitized_name,
                collection_id=reservation_id,
                user_id=reservation_user_id,
            )
        raise
    finally:
        if client is not None:
            client.close()


@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """删除 Milvus 集合并清除注册记录"""
    registry = _get_owned_collection(collection_name, current_user)
    collection_id = registry.get("collection_id")
    collection_hash = hashlib.sha256(
        collection_name.encode("utf-8")
    ).hexdigest()
    unregistered_owner = f"manual:unregistered:{collection_hash}"
    deletion_owner = (
        f"manual:{collection_id}"
        if collection_id
        else unregistered_owner
    )
    if registry.get("status") == "deleting":
        fenced_owner = milvus_collection_service.get_deletion_fence_owner(
            collection_name
        )
        allowed_owners = {deletion_owner, unregistered_owner}
        if fenced_owner not in allowed_owners:
            raise HTTPException(
                status_code=409,
                detail="Collection is being deleted by another operation",
            )
        deletion_owner = fenced_owner
    owner_user_id = (
        registry.get("user_id")
        if _is_auth_disabled(current_user)
        else current_user.get("user_id")
    )
    creating_abort_options = (
        {"allow_manual_creating": True}
        if registry.get("status") == "creating"
        else {}
    )
    try:
        acquisition = milvus_collection_service.acquire_deletion_fences(
            [collection_name],
            deletion_owner=deletion_owner,
            user_id=owner_user_id,
            expected_collection_ids={collection_name: collection_id},
            **creating_abort_options,
        )
    except (
        MilvusCollectionDeletionOwnerConflictError,
        MilvusCollectionUnavailableError,
        PermissionError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    client = None
    remote_drop_started = False

    def restore_new_fence() -> None:
        if not acquisition.newly_fenced:
            return
        milvus_collection_service.restore_deletion_fences(
            acquisition.newly_fenced,
            deletion_owner=deletion_owner,
            created_placeholders=acquisition.created_placeholders,
            previous_creating=getattr(
                acquisition,
                "previous_creating",
                (),
            ),
        )

    try:
        active_consumers = (
            generation_task_service.list_active_collection_consumers(
                [collection_name]
            )
        )
        if not active_consumers:
            active_consumers = (
                deep_evaluation_task_service.list_active_collection_consumers(
                    [collection_name]
                )
            )
        if not active_consumers:
            active_consumers = (
                external_sync_service.list_active_collection_consumers(
                    [collection_name]
                )
            )
        if active_consumers:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete a collection used by active tasks",
            )

        client = _get_milvus_client()
        remote_drop_started = True
        client.drop_collection(collection_name)
        milvus_collection_service.delete_collection(
            collection_name,
            deletion_owner=deletion_owner,
        )
        return {"message": f"Collection '{collection_name}' 已删除"}
    except BaseException:
        if not remote_drop_started:
            restore_new_fence()
        raise
    finally:
        if client is not None:
            client.close()


@router.get("/collections/{collection_name}/entities")
async def browse_entities(
    collection_name: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    include_vector: bool = Query(False),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """分页浏览集合中的实体"""
    _get_owned_collection(collection_name, current_user)
    client = _get_milvus_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' 不存在"
            )

        output_fields = None
        if include_vector:
            output_fields = None  # 返回所有字段，query_entities 内部处理
            from pymilvus import Collection
            collection = Collection(collection_name, using=client._alias)
            output_fields = [f.name for f in collection.schema.fields]

        entities, total = client.query_entities(
            collection_name,
            offset=offset,
            limit=limit,
            output_fields=output_fields,
        )

        # 如果包含向量，截断为前 5 维预览
        if include_vector:
            for entity in entities:
                if "vector" in entity and isinstance(entity["vector"], list):
                    full_dim = len(entity["vector"])
                    entity["vector_preview"] = entity["vector"][:5]
                    entity["vector_dim"] = full_dim
                    del entity["vector"]

        return {"entities": entities, "total": total}
    finally:
        client.close()


@router.post("/collections/{collection_name}/search")
async def search_collection(
    collection_name: str,
    request: SearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """在集合中搜索，支持 dense / sparse / hybrid 三种模式"""
    import numpy as np
    from ...storage.services.model_config_service import model_config_service
    from ...generation.clients.embedding_client import EmbeddingClient, EmbeddingConfig

    _get_owned_collection(collection_name, current_user)
    mode = request.search_mode
    if mode not in ("dense", "sparse", "hybrid"):
        raise HTTPException(status_code=400, detail=f"不支持的搜索模式: {mode}")

    client = _get_milvus_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' 不存在",
            )

        # sparse / hybrid 需要集合支持混合搜索
        if mode in ("sparse", "hybrid") and not client.supports_hybrid(collection_name):
            if mode == "hybrid":
                # 降级为 dense
                mode = "dense"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="该集合不支持关键词搜索，请在创建时启用混合搜索",
                )

        # dense / hybrid 需要向量化
        query_vector: Optional[np.ndarray] = None
        if mode in ("dense", "hybrid"):
            if not request.embedding_config_id:
                raise HTTPException(
                    status_code=400,
                    detail="dense/hybrid 搜索需要指定 embedding_config_id",
                )
            config = _verify_owned_resource(
                model_config_service.get_config(request.embedding_config_id),
                current_user,
                "Embedding config",
            )
            if not config:
                raise HTTPException(status_code=404, detail="Embedding 配置不存在")
            if config.get("model_type") != "embedding":
                raise HTTPException(status_code=400, detail="指定的配置不是 Embedding 类型")

            emb_config = EmbeddingConfig(
                endpoint=config["api_endpoint"],
                model=config["model_name"],
                api_key=config.get("api_key"),
                user_id=current_user.get("user_id"),
            )
            try:
                async with EmbeddingClient(emb_config) as embedding_client:
                    query_vector = await embedding_client.embed(request.query_text)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Embedding 失败: {e}")

        # 执行搜索
        try:
            if mode == "dense":
                results = client.search_similar(
                    collection_name=collection_name,
                    query_vectors=query_vector,
                    top_k=request.top_k,
                    filter_expr=request.filter_expr,
                )
                hits = results[0] if results else []
            else:
                # sparse 或 hybrid → 走 hybrid_search
                hits = client.hybrid_search(
                    collection_name=collection_name,
                    search_mode=mode,
                    query_vectors=query_vector,
                    query_text=request.query_text,
                    top_k=request.top_k,
                    ranker=request.ranker,
                    dense_weight=request.dense_weight,
                    sparse_weight=request.sparse_weight,
                    filter_expr=request.filter_expr,
                )
        except HTTPException:
            raise
        except Exception as e:
            msg = str(e).lower()
            filter_expr_error_markers = (
                "parse expression",
                "invalid expression",
                "syntax error",
                "query plan",
                "cannot parse",
            )
            if request.filter_expr and any(marker in msg for marker in filter_expr_error_markers):
                raise HTTPException(
                    status_code=400,
                    detail=f"过滤表达式格式错误，请检查 filter_expr: {e}",
                )
            raise HTTPException(status_code=500, detail=f"Milvus 搜索失败: {e}")

        return {
            "results": hits,
            "query_text": request.query_text,
            "search_mode": mode,
            "total": len(hits),
        }
    finally:
        client.close()


@router.post("/collections/{collection_name}/load")
async def load_collection(
    collection_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """加载集合到内存"""
    _get_owned_collection(collection_name, current_user)
    client = _get_milvus_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' 不存在"
            )
        client.load_collection(collection_name)
        return {"message": f"Collection '{collection_name}' 已加载"}
    finally:
        client.close()


@router.post("/collections/{collection_name}/release")
async def release_collection(
    collection_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """从内存释放集合"""
    _get_owned_collection(collection_name, current_user)
    client = _get_milvus_client()
    try:
        if not client.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' 不存在"
            )
        client.release_collection(collection_name)
        return {"message": f"Collection '{collection_name}' 已释放"}
    finally:
        client.close()


# ── Dataset Link Endpoints ───────────────────────────────────

@router.get("/collections/{collection_name}/datasets")
async def get_linked_datasets(
    collection_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取集合关联的数据集列表"""
    _get_owned_collection(collection_name, current_user)
    links = milvus_collection_service.get_linked_datasets(collection_name)
    return {"datasets": links, "total": len(links)}


@router.post("/collections/{collection_name}/datasets")
async def link_dataset_to_collection(
    collection_name: str,
    request: LinkDatasetRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """手动关联数据集到集合"""
    # Verify collection is registered
    reg = _get_owned_collection(collection_name, current_user)
    if not reg:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' 未注册")

    dataset = _verify_owned_resource(
        dataset_service.get_dataset(request.dataset_id),
        current_user,
        "Dataset",
    )

    try:
        link = milvus_collection_service.link_dataset(
            collection_name=collection_name,
            dataset_id=dataset["dataset_id"],
            dataset_name=(
                dataset.get("dataset_name")
                or dataset.get("display_name")
                or dataset["dataset_id"]
            ),
            chunk_count=request.chunk_count,
            task_id=None,
        )
    except (
        DatasetConsumptionUnavailableError,
        MilvusCollectionUnavailableError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return link


@router.delete("/collections/{collection_name}/datasets/{dataset_id}")
async def unlink_dataset_from_collection(
    collection_name: str,
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """解除数据集与集合的关联"""
    _get_owned_collection(collection_name, current_user)
    success = milvus_collection_service.unlink_dataset(collection_name, dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="关联不存在")
    return {"message": "已解除关联"}
