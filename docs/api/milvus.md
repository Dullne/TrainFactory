# 向量库管理 API

向量库管理模块提供 Milvus 向量集合的独立管理能力，支持集合与数据集的多对多关联。

## 核心概念

- **集合 (Collection)**: Milvus 中的向量集合，作为独立实体管理，绑定唯一的 Embedding 模型
- **集合注册表**: 本地数据库记录集合元数据（embedding 绑定、显示名称等），Milvus 提供实时状态（实体数、加载状态）
- **多对多关联**: 一个集合可关联多个数据集，一个数据集也可关联多个集合（使用不同 embedding 模型）

## 端点概览

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/milvus/status` | 获取 Milvus 连接状态 |
| GET | `/api/milvus/collections` | 列出所有集合 |
| GET | `/api/milvus/collections/{name}` | 获取集合详情 |
| POST | `/api/milvus/collections` | 创建新集合 |
| DELETE | `/api/milvus/collections/{name}` | 删除集合 |
| GET | `/api/milvus/collections/{name}/entities` | 浏览集合数据 |
| POST | `/api/milvus/collections/{name}/search` | 语义搜索 |
| POST | `/api/milvus/collections/{name}/load` | 加载集合到内存 |
| POST | `/api/milvus/collections/{name}/release` | 从内存释放集合 |
| GET | `/api/milvus/collections/{name}/datasets` | 获取关联数据集 |
| POST | `/api/milvus/collections/{name}/datasets` | 关联数据集 |
| DELETE | `/api/milvus/collections/{name}/datasets/{dataset_id}` | 解除数据集关联 |

---

## GET /api/milvus/status

获取 Milvus 连接状态。

**响应**:
```json
{
  "connected": true,
  "host": "milvus",
  "port": 19530,
  "collection_count": 7
}
```

---

## GET /api/milvus/collections

列出所有集合，合并注册表元数据与 Milvus 实时状态。

**响应**:
```json
{
  "collections": [
    {
      "name": "tf_022fa895_1d1780cb",
      "num_entities": 20,
      "dim": 1024,
      "index_type": "IVF_FLAT",
      "metric_type": "COSINE",
      "load_state": "Loaded",
      "description": "TrainFactory chunk vectors",
      "collection_id": "5d91516d-d58b-48cb-a1aa-11ee08ea9743",
      "display_name": "tf_022fa895_1d1780cb",
      "embedding_config_id": "92166c52-182d-445d-b537-2f105bbd8fcc",
      "embedding_model": "Qwen3-Embedding-0.6B",
      "embedding_endpoint": "http://embedding-service.example.com:9997",
      "linked_datasets": [
        {
          "dataset_id": "022fa895-3ab7-49a8-bc2d-f850d04b3e46",
          "dataset_name": "test-dataset",
          "chunk_count": 0,
          "task_id": "8e50e0af-22de-49db-a4c5-21e728bf4a30"
        }
      ],
      "status": "active",
      "source_dataset_id": "022fa895-3ab7-49a8-bc2d-f850d04b3e46",
      "source_dataset_name": "test-dataset",
      "associated_task_id": "8e50e0af-22de-49db-a4c5-21e728bf4a30"
    }
  ],
  "total": 7
}
```

---

## POST /api/milvus/collections

创建新集合并注册到数据库。

**请求**:
```json
{
  "name": "my_collection",
  "dim": 1024,
  "metric_type": "COSINE",
  "description": "示例集合",
  "embedding_config_id": "92166c52-...",
  "display_name": "我的集合"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 集合名称（字母、数字、下划线） |
| `dim` | int | 是 | 向量维度 (1-4096) |
| `metric_type` | string | 否 | 距离度量: COSINE, L2, IP（默认 COSINE） |
| `description` | string | 否 | 描述 |
| `embedding_config_id` | string | 否 | 绑定的 Embedding 模型配置 ID |
| `display_name` | string | 否 | 显示名称 |

**响应**:
```json
{
  "name": "my_collection",
  "collection_id": "uuid-...",
  "message": "Collection 'my_collection' 创建成功"
}
```

---

## DELETE /api/milvus/collections/{name}

删除 Milvus 集合并清除注册记录。

**响应**:
```json
{
  "message": "Collection 'my_collection' 已删除"
}
```

---

## GET /api/milvus/collections/{name}

获取集合详情（注册表 + Milvus 实时状态），包含 schema 和索引信息。

**响应**:
```json
{
  "name": "tf_022fa895_1d1780cb",
  "num_entities": 20,
  "dim": 1024,
  "index_type": "IVF_FLAT",
  "metric_type": "COSINE",
  "load_state": "Loaded",
  "collection_id": "5d91516d-...",
  "embedding_model": "Qwen3-Embedding-0.6B",
  "linked_datasets": [...],
  "schema_fields": [
    {"name": "chunk_id", "dtype": "VarChar", "is_primary": true},
    {"name": "chunk_content", "dtype": "VarChar", "is_primary": false},
    {"name": "embedding", "dtype": "FloatVector", "dim": 1024}
  ],
  "indexes": [
    {"field_name": "embedding", "index_type": "IVF_FLAT", "metric_type": "COSINE"}
  ]
}
```

---

## GET /api/milvus/collections/{name}/entities

浏览集合中的实体数据。

**查询参数**:

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `offset` | int | 0 | 起始偏移 |
| `limit` | int | 20 | 返回数量 |
| `include_vector` | bool | false | 是否包含向量数据 |

**响应**:
```json
{
  "total": 20,
  "entities": [
    {
      "chunk_id": "abc123",
      "chunk_content": "这是一段文档内容...",
      "doc_id": "doc-001"
    }
  ]
}
```

---

## POST /api/milvus/collections/{name}/search

语义搜索，使用指定 Embedding 模型将查询文本向量化后在集合中检索。

**请求**:
```json
{
  "query_text": "如何配置模型训练",
  "embedding_config_id": "92166c52-...",
  "top_k": 10
}
```

**响应**:
```json
{
  "results": [
    {
      "chunk_id": "abc123",
      "chunk_content": "模型训练配置包括...",
      "distance": 0.85
    }
  ],
  "query_text": "如何配置模型训练",
  "total": 5
}
```

---

## POST /api/milvus/collections/{name}/load

加载集合到 Milvus 内存（搜索前需先加载）。

---

## POST /api/milvus/collections/{name}/release

从 Milvus 内存释放集合。

---

## GET /api/milvus/collections/{name}/datasets

获取集合关联的数据集列表。

**响应**:
```json
{
  "datasets": [
    {
      "collection_name": "tf_022fa895_1d1780cb",
      "dataset_id": "022fa895-...",
      "dataset_name": "test-dataset",
      "chunk_count": 20,
      "task_id": "8e50e0af-..."
    }
  ],
  "total": 1
}
```

---

## POST /api/milvus/collections/{name}/datasets

手动关联数据集到集合。

**请求**:
```json
{
  "dataset_id": "022fa895-...",
  "dataset_name": "test-dataset",
  "chunk_count": 20,
  "task_id": "8e50e0af-..."
}
```

---

## DELETE /api/milvus/collections/{name}/datasets/{dataset_id}

解除数据集与集合的关联。

---

## 自动注册机制

当数据生成任务完成时，系统会自动：
1. 将任务使用的 Milvus 集合注册到注册表
2. 关联源数据集到该集合
3. 记录 embedding 模型绑定信息

生成任务创建时可通过 `milvus_collection_name` 参数选择已有集合，此时会自动从注册表获取 embedding 配置。
