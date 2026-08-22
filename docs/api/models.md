# 模型注册 API

## 注册模型

```http
POST /api/models
```

**请求体**:
```json
{
  "model_name": "my-embedding-model",
  "model_path": "/app/output/final_model",
  "model_type": "embedding",
  "version": "v1.0.0",
  "source_task_id": "550e8400-e29b-41d4-a716-446655440000",
  "description": "Fine-tuned embedding model",
  "tags": ["embedding", "chinese"],
  "metrics": {
    "accuracy": 0.95,
    "f1": 0.93
  }
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 是 | 模型名称 |
| `model_path` | string | 是 | 模型路径 |
| `model_type` | string | 是 | 模型类型: embedding, reranker, decoder_reranker, llm |
| `version` | string | 否 | 版本号 |
| `source_task_id` | string | 否 | 来源训练任务 ID |
| `description` | string | 否 | 描述 |
| `tags` | array | 否 | 标签 |
| `metrics` | object | 否 | 评估指标 |

## 从训练任务注册

```http
POST /api/models/from-task/{task_id}
```

**请求体**:
```json
{
  "model_name": "my-model",
  "version": "v1.0.0",
  "description": "Model from training task"
}
```

## 列出模型

```http
GET /api/models?model_type=embedding&latest_only=true
```

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_type` | string | 按模型类型过滤: embedding, reranker, decoder_reranker, llm |
| `status` | string | 按状态过滤: registered, available, archived |
| `latest_only` | bool | 只返回最新版本 |
| `page` | int | 页码（默认 1） |
| `page_size` | int | 每页数量（默认 10） |

## 搜索模型

```http
GET /api/models/search?query=embedding&limit=10
```

## 添加版本

```http
POST /api/models/{model_id}/versions
```

**请求体**:
```json
{
  "version": "v1.1.0",
  "model_path": "/app/output/new_model",
  "changelog": "Improved accuracy",
  "metrics": {"accuracy": 0.97},
  "set_as_latest": true
}
```

## 对比模型

```http
POST /api/models/compare
```

**请求体**:
```json
{
  "model_ids": ["model-1", "model-2", "model-3"]
}
```

## 获取模型详情

```http
GET /api/models/{model_id}
```

## 删除模型

```http
DELETE /api/models/{model_id}
```
