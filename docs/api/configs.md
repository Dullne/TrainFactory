# 模型配置 API

模型配置用于管理 API 端点（如 OpenAI、Azure、本地部署等），供评估和推理使用。

## 验证配置

```http
POST /api/configs/validate
```

**请求体**:
```json
{
  "provider": "openai",
  "api_endpoint": "https://api.openai.com/v1",
  "api_key": "sk-xxx",
  "model_name": "text-embedding-3-large"
}
```

**响应**:
```json
{
  "valid": true,
  "latency_ms": 150.5,
  "models": ["text-embedding-3-large", "text-embedding-3-small"],
  "message": "Connection successful"
}
```

## 创建配置

```http
POST /api/configs
```

**请求体**:
```json
{
  "config_name": "openai-embedding",
  "model_type": "embedding",
  "provider": "openai",
  "api_endpoint": "https://api.openai.com/v1",
  "api_key": "sk-xxx",
  "model_name": "text-embedding-3-large",
  "is_default": true,
  "validate_api": true
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config_name` | string | 是 | 配置名称 |
| `model_type` | string | 是 | 模型类型: embedding, rerank, llm |
| `provider` | string | 是 | 提供商: openai, azure, xinference, ollama, custom |
| `api_endpoint` | string | 是 | API 端点 |
| `api_key` | string | 否 | API 密钥 |
| `model_name` | string | 否 | 模型名称 |
| `is_default` | bool | 否 | 设为默认配置 |
| `validate_api` | bool | 否 | 创建前验证连通性 |

## 列出配置

```http
GET /api/configs?model_type=embedding
```

## 获取配置详情

```http
GET /api/configs/{config_id}
```

## 更新配置

```http
PUT /api/configs/{config_id}
```

## 删除配置

```http
DELETE /api/configs/{config_id}
```

## 获取模板

```http
GET /api/configs/templates
```

返回各提供商的配置模板（OpenAI, Azure, Xinference, Ollama 等）。

## 获取默认配置

```http
GET /api/configs/default/{model_type}
```

获取指定模型类型的默认配置。

## 检查连通性

```http
POST /api/configs/{config_id}/check
```

检查单个配置的连通性。返回模型列表包含详细信息：

```json
{
  "valid": true,
  "latency_ms": 50.2,
  "models": [
    {
      "id": "model-name",
      "model_type": "embedding",
      "parent": null,
      "embedding_dim": 1024,
      "context_length": 8192
    }
  ]
}
```

其中 `parent` 字段用于识别 LoRA adapter：
- 基座模型: `parent` 为 `null`
- LoRA adapter: `parent` 为基座模型名称（如 `/app/models/xxx`）

## 批量检查

```http
POST /api/configs/check-all?model_type=embedding
```

检查所有指定类型配置的连通性。

## API 测试代理

```http
POST /api/configs/{config_id}/test
```

通过后端转发 API 请求到推理服务，支持多种测试模式。

**请求体**:
```json
{
  "path": "/v1/embeddings",
  "body": {
    "input": "Hello world",
    "model": "model-name"
  }
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | API 路径（如 `/v1/embeddings`、`/v1/rerank`） |
| `body` | object | 是 | 请求体，根据测试模式不同而不同 |

### 测试模式

**向量模式**: 发送标准 embedding 请求，返回向量结果。

```json
{
  "path": "/v1/embeddings",
  "body": {
    "input": "测试文本",
    "model": "my-lora-adapter"
  }
}
```

**相似度模式**: 计算两段文本的 embedding 余弦相似度。

```json
{
  "path": "/v1/embeddings",
  "body": {
    "mode": "embedding_similarity",
    "sentence1": "机器学习是人工智能的分支",
    "sentence2": "ML is a subset of AI",
    "model": "my-lora-adapter"
  }
}
```

**召回模式**: 使用当前 embedding 模型 embed 查询文本，在指定 Milvus 向量库中搜索 top-k 结果。

```json
{
  "path": "/v1/embeddings",
  "body": {
    "mode": "embedding_recall",
    "queries": ["什么是机器学习？", "如何训练模型？"],
    "collection_name": "tf_sync_a1b2c3d4_f5e6d7c8",
    "top_k": 10,
    "model": "my-lora-adapter"
  }
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `mode` | string | 是 | 固定 `"embedding_recall"` |
| `queries` | array | 是 | 查询文本列表（每条单独搜索） |
| `collection_name` | string | 是 | Milvus 向量库名称 |
| `top_k` | int | 否 | 召回数量，默认 10 |
| `model` | string | 否 | 模型/adapter 名称 |
| `batch_size` | int | 否 | Embedding 批次大小，默认 32 |

**响应**:

```json
{
  "success": true,
  "status_code": 200,
  "latency_ms": 350.5,
  "data": {
    "collection_name": "tf_sync_a1b2c3d4_f5e6d7c8",
    "model": "my-lora-adapter",
    "top_k": 10,
    "results": [
      {
        "query": "什么是机器学习？",
        "hits": [
          {"id": "doc_001", "score": 0.923, "content": "机器学习是..."},
          {"id": "doc_002", "score": 0.891, "content": "ML是一种..."}
        ]
      }
    ]
  }
}
```

### Adapter 覆盖

在 `body.model` 中指定 adapter 名称，即可使用已加载的 LoRA adapter 进行测试。不指定时默认使用配置的基座模型名称。

前端 API 测试弹窗提供 Adapter 选择器下拉框，自动获取部署上已加载的 adapter 列表。

## 向量库匹配查询

```http
GET /api/configs/{config_id}/collections
```

列出所有已注册的 Milvus 向量库，标注与当前模型配置的匹配关系。用于 embedding 召回测试中的向量库选择。

**响应**:

```json
{
  "collections": [
    {
      "name": "tf_sync_a1b2c3d4_f5e6d7c8",
      "match_type": "exact",
      "embedding_model": "bge-large-zh",
      "embedding_config_id": "config-uuid",
      "dim": 1024,
      "status": "ready"
    },
    {
      "name": "other_collection",
      "match_type": "mismatch",
      "embedding_model": "text-embedding-3-large",
      "embedding_config_id": "other-config-uuid",
      "dim": 1536,
      "status": "ready"
    }
  ]
}
```

**匹配类型**:

| `match_type` | 说明 |
|--------------|------|
| `exact` | `embedding_config_id` 与当前 config_id 完全匹配 |
| `model_match` | `embedding_model` 与当前 model_name 匹配 |
| `mismatch` | 不匹配（使用不同模型入库，召回结果可能不准确） |

结果按匹配度排序：exact > model_match > mismatch。
