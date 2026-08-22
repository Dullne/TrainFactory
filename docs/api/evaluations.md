# 评估管理 API

## 创建评估任务

```http
POST /api/evaluations
```

**请求体**:
```json
{
  "task_name": "reranker-evaluation",
  "model_configs": [
    {
      "name": "xinference-reranker",
      "endpoint": "http://xinference:9997",
      "model_name": "Qwen3-Reranker-0.6B",
      "inference_framework": "xinference"
    },
    {
      "name": "vllm-reranker",
      "endpoint": "http://localhost:10001",
      "model_name": "Qwen3-Reranker-0.6B",
      "inference_framework": "vllm"
    },
    {
      "name": "sglang-reranker",
      "endpoint": "http://localhost:10002",
      "model_name": "Qwen3-Reranker-0.6B",
      "inference_framework": "sglang"
    }
  ],
  "dataset_configs": [
    {"name": "T2Reranking", "type": "mteb"},
    {"name": "CMedQAv2-reranking", "type": "mteb"},
    {"name": "my-dataset", "type": "local", "path": "/data/datasets/eval.jsonl"}
  ],
  "max_samples": 200,
  "batch_size": 50,
  "workers": 8,
  "model_workers": 2
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `task_name` | string | 否 | null | 任务名称 |
| `model_configs` | array | 是 | - | 待评估模型配置列表 |
| `model_configs[].name` | string | 否 | null | 模型显示名称 |
| `model_configs[].endpoint` | string | 是 | - | 推理服务端点 |
| `model_configs[].model_name` | string | 否 | null | 服务中的模型名称 |
| `model_configs[].inference_framework` | string | 否 | null | 推理框架: xinference, vllm, sglang |
| `dataset_configs` | array | 是 | - | 数据集配置列表 |
| `dataset_configs[].name` | string | 是 | - | 数据集名称 |
| `dataset_configs[].type` | string | 是 | - | 类型: mteb, local, registered |
| `dataset_configs[].path` | string | 否 | null | 本地数据集路径 (type=local 时必填) |
| `max_samples` | int | 否 | null | 每个数据集最大样本数 |
| `batch_size` | int | 否 | 50 | API 调用批次大小 |
| `workers` | int | 否 | 8 | 并发请求数 |
| `model_workers` | int | 否 | 2 | 模型并行评估数 |

**响应**:
```json
{
  "task_id": "e22c2d74-ca5f-406f-99e1-e4db7a7f2500",
  "message": "评估任务已创建"
}
```

## 列出评估任务

```http
GET /api/evaluations/tasks?status=running&limit=10
```

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `status` | string | 按状态过滤 |
| `limit` | int | 返回数量限制 |
| `offset` | int | 偏移量 |

## 获取评估任务详情

```http
GET /api/evaluations/tasks/{task_id}
```

**响应**:
```json
{
  "task_id": "e22c2d74-ca5f-406f-99e1-e4db7a7f2500",
  "task_name": "reranker-evaluation",
  "status": "succeeded",
  "progress": 100.0,
  "model_progress": {
    "xinference-reranker": {
      "T2Reranking": {"status": "completed", "progress": 100},
      "CMedQAv2-reranking": {"status": "completed", "progress": 100}
    }
  },
  "results": {
    "xinference-reranker": {
      "T2Reranking": {
        "NDCG@10": 0.7420,
        "MRR": 0.7867,
        "AP": 0.6739
      }
    }
  },
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T11:30:00Z"
}
```

## 取消评估任务

```http
POST /api/evaluations/tasks/{task_id}/cancel
```

## 恢复评估任务

```http
POST /api/evaluations/tasks/{task_id}/resume
```

从上次中断的位置继续评估，跳过已完成的模型-数据集组合。

**响应**:
```json
{
  "message": "任务已重新启动",
  "task_id": "e22c2d74-ca5f-406f-99e1-e4db7a7f2500",
  "skipped_evaluations": 3
}
```

## 删除评估任务

```http
DELETE /api/evaluations/tasks/{task_id}
```

## 获取可用数据集

```http
GET /api/evaluations/datasets
```

返回 MTEB 评估数据集列表及其分组。

**响应**:
```json
{
  "datasets": {
    "T2Reranking": {"split": "dev", "lang": "zh", "description": "中文通用重排序"},
    "MMarcoReranking": {"split": "dev", "lang": "zh", "description": "中文 MS MARCO"}
  },
  "groups": {
    "chinese": ["T2Reranking", "MMarcoReranking"],
    "english": ["AskUbuntuDupQuestions", "MindSmallReranking"],
    "all": ["..."]
  }
}
```

---

# 深度评估 API

深度评估支持对 Embedding 和 Reranker 模型进行详细的指标评估。

## 获取支持的评估类型

```http
GET /api/deep-evaluation/eval-types
```

**响应**:
```json
{
  "rerank": {
    "metrics": ["mrr", "map", "ndcg@10", "recall@10", "precision@10"],
    "description": "Reranker 模型评估",
    "default_metrics": ["mrr", "ndcg@10"]
  },
  "embedding": {
    "metrics": ["mrr", "map", "ndcg@10", "recall@10", "precision@10"],
    "description": "Embedding 模型评估",
    "default_metrics": ["mrr", "ndcg@10"]
  }
}
```

## 获取评估指标

```http
# LLM 指标（默认）
GET /api/deep-evaluation/metrics

# 传统 IR 指标
GET /api/deep-evaluation/metrics?category=traditional
```

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `category` | string | 指标类别: traditional/retrieval (传统检索指标); 不传时返回 LLM 指标 |

**响应**（示例）:
```json
[
  {
    "name": "answer_relevancy",
    "description": "答案相关性评估",
    "category": "llm",
    "requires_llm": true,
    "requires_expected_output": false,
    "requires_actual_output": true,
    "requires_retrieval_context": true
  }
]
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 指标名称 |
| `description` | string | 指标描述 |
| `category` | string | 类别: retrieval (传统检索) / llm (LLM-as-Judge) |
| `requires_llm` | boolean | 是否需要 LLM 模型 |
| `requires_expected_output` | boolean | 是否需要期望输出字段 |
| `requires_actual_output` | boolean | 是否需要实际输出字段 |
| `requires_retrieval_context` | boolean | 是否需要检索上下文字段 |

## 创建深度评估任务

```http
POST /api/deep-evaluation/tasks
```

深度评估支持多模型组并行评估，每个组可包含 Embedding、Rerank 和 LLM 三类模型。

**请求体**:
```json
{
  "task_name": "multi-model-evaluation",
  "description": "对比多个模型的检索效果",
  "model_configs": [
    {
      "group_name": "baseline",
      "embedding": {
        "config_id": "config-uuid-1",
        "concurrency": 8
      },
      "rerank": {
        "config_id": "config-uuid-2",
        "concurrency": 8
      }
    },
    {
      "group_name": "optimized",
      "embedding": {
        "endpoint": "http://localhost:9997",
        "model_name": "bge-large-zh",
        "concurrency": 8
      },
      "rerank": {
        "endpoint": "http://localhost:9998",
        "model_name": "bge-reranker-large",
        "inference_framework": "xinference",
        "concurrency": 8
      },
      "llm": {
        "config_id": "llm-config-uuid",
        "temperature": 0.0,
        "max_tokens": 2048,
        "concurrency": 2
      }
    }
  ],
  "dataset_configs": [
    {"dataset_id": "dataset-uuid-1"},
    {"dataset_id": "dataset-uuid-2"}
  ],
  "max_samples": 1000,
  "field_mapping": {
    "query": "question",
    "positives": "positive_passages",
    "negatives": "negative_passages",
    "input": "input",
    "expected_output": "expected_output",
    "actual_output": "actual_output",
    "retrieval_context": "retrieval_context"
  },
  "metrics": ["mrr", "ndcg@10", "answer_relevancy", "faithfulness"],
  "model_workers": 2
}
```

**说明**:
- 选择 **LLM 指标**时，每个模型组必须配置 `llm`
- 选择 **IR 指标**时，每个模型组至少配置 `embedding` 或 `rerank`

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `task_name` | string | 否 | null | 任务名称 |
| `description` | string | 否 | null | 任务描述 |
| `model_configs` | array | 是 | - | 模型组配置列表 |
| `model_configs[].group_name` | string | 是 | - | 组名称，用于标识和结果展示 |
| `model_configs[].embedding` | object | 否 | null | Embedding 模型配置 |
| `model_configs[].rerank` | object | 否 | null | Rerank 模型配置 |
| `model_configs[].llm` | object | 否 | null | LLM 模型配置（用于 LLM-as-Judge 指标） |
| `dataset_configs` | array | 是 | - | 数据集配置列表 |
| `dataset_configs[].dataset_id` | string | 是 | - | 数据集 ID |
| `max_samples` | int | 否 | null | 每个数据集最大样本数（空表示全量） |
| `field_mapping` | object | 否 | null | 字段映射 |
| `metrics` | array | 否 | ["mrr", "ndcg@10"] | 评估指标列表 |
| `model_workers` | int | 否 | 2 | 模型组并发数（同时评估几个组） |
| `retrieval_mode` | string | 否 | null | 检索模式: `online`（在线从 Milvus 检索） |
| `milvus_collection` | string | 否 | null | Milvus 集合名称（`online` 模式必填） |
| `retrieval_embedding_config` | object | 否 | null | 检索 Embedding 模型配置（`online` 模式必填，不填时自动从 collection 注册信息解析） |
| `retrieval_top_k` | int | 否 | 20 | 在线检索返回的候选数量 (1-100) |
| `allow_collection_model_mismatch` | bool | 否 | false | 允许检索模型与集合入库模型不一致 |

**模型配置字段（embedding/rerank）**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `config_id` | string | 否* | null | 已注册模型配置 ID |
| `endpoint` | string | 否* | null | 模型 API 端点 |
| `model_name` | string | 否* | null | 模型名称 |
| `api_key` | string | 否 | null | API 密钥 |
| `inference_framework` | string | 否 | null | 推理框架: vllm, sglang, xinference |
| `concurrency` | int | 否 | 8 | 请求并发数 |

*注：`config_id` 或 `endpoint + model_name` 二选一

**LLM 配置字段**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `config_id` | string | 否* | null | 已注册模型配置 ID |
| `endpoint` | string | 否* | null | LLM API 端点 |
| `model` | string | 否* | null | 模型名称 |
| `api_key` | string | 否 | null | API 密钥 |
| `temperature` | float | 否 | 0.0 | 温度参数 |
| `top_p` | float | 否 | 1.0 | Top-p 采样 |
| `top_k` | int | 否 | null | Top-k 采样 |
| `max_tokens` | int | 否 | 2048 | 最大 token 数 |
| `timeout` | int | 否 | 60 | 超时时间（秒） |
| `max_retries` | int | 否 | 3 | 最大重试次数 |
| `concurrency` | int | 否 | 2 | LLM 请求并发数 |

**字段映射**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | "query" | 查询字段名 |
| `positives` | string | "positives" | 正例字段名 |
| `negatives` | string | "negatives" | 负例字段名 |
| `input` | string | "input" | 输入字段名（LLM 评估） |
| `expected_output` | string | "expected_output" | 期望输出字段名 |
| `actual_output` | string | "actual_output" | 实际输出字段名 |
| `retrieval_context` | string | "retrieval_context" | 检索上下文字段名 |

**支持的评估指标**:

| 指标 | 类型 | 需要 LLM | 说明 |
|------|------|----------|------|
| `mrr` | 传统检索 | 否 | Mean Reciprocal Rank - 平均倒数排名 |
| `map` | 传统检索 | 否 | Mean Average Precision - 平均精度均值 |
| `ndcg@10` | 传统检索 | 否 | Normalized DCG at 10 - 归一化折损累积增益 |
| `recall@10` | 传统检索 | 否 | Recall at 10 - 前10召回率 |
| `precision@10` | 传统检索 | 否 | Precision at 10 - 前10精确率 |
| `answer_relevancy` | LLM-as-Judge | 是 | 答案相关性 |
| `faithfulness` | LLM-as-Judge | 是 | 答案忠实度 |
| `contextual_precision` | LLM-as-Judge | 是 | 上下文精确度 |
| `contextual_recall` | LLM-as-Judge | 是 | 上下文召回率 |
| `contextual_relevancy` | LLM-as-Judge | 是 | 上下文相关性 |

**响应**:
```json
{
  "task_id": "f1a2b3c4-d5e6-7890-abcd-ef1234567890",
  "message": "Deep evaluation task created"
}
```

## 列出深度评估任务

```http
GET /api/deep-evaluation/tasks?status=running&eval_type=embedding&limit=10
```

## 获取深度评估任务详情

```http
GET /api/deep-evaluation/tasks/{task_id}
```

**响应**:
```json
{
  "task_id": "f1a2b3c4-d5e6-7890-abcd-ef1234567890",
  "task_name": "multi-model-evaluation",
  "description": "对比多个模型的检索效果",
  "eval_type": "multi",
  "model_configs": [
    {
      "group_name": "baseline",
      "embedding": {
        "config_id": "config-uuid-1",
        "config_name": "bge-base-zh",
        "endpoint": "http://localhost:9997",
        "model_name": "bge-base-zh",
        "concurrency": 8
      },
      "rerank": {
        "config_id": "config-uuid-2",
        "config_name": "bge-reranker-base",
        "endpoint": "http://localhost:9998",
        "model_name": "bge-reranker-base",
        "concurrency": 8
      }
    }
  ],
  "dataset_configs": [
    {"dataset_id": "dataset-uuid-1", "dataset_name": "test-dataset"}
  ],
  "max_samples": 1000,
  "metrics": ["mrr", "ndcg@10"],
  "worker_groups": { "model_workers": 2 },
  "status": "completed",
  "progress": 100.0,
  "total_samples": 1000,
  "processed_samples": 1000,
  "model_progress": {
    "baseline": {
      "test-dataset": {
        "status": "completed",
        "progress": 100
      }
    }
  },
  "results_summary": {
    "overall": { "mean": 0.71, "min": 0.64, "max": 0.78 },
    "metrics": {
      "mrr": { "mean": 0.7523, "min": 0.71, "max": 0.78, "count": 1000 }
    },
    "by_group": {
      "baseline": {
        "retrieval": {
          "embedding": {
            "summary": { "overall": { "mean": 0.71 } }
          }
        }
      }
    }
  },
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T11:15:30Z"
}
```

**eval_type 说明**:
- `embedding`: 仅包含 Embedding 模型
- `rerank`: 仅包含 Rerank 模型
- `llm`: 仅包含 LLM 模型
- `multi`: 包含多种类型模型

## 取消深度评估任务

```http
POST /api/deep-evaluation/tasks/{task_id}/cancel
```

## 恢复深度评估任务

```http
POST /api/deep-evaluation/tasks/{task_id}/resume
```

## 删除深度评估任务

```http
DELETE /api/deep-evaluation/tasks/{task_id}
```

## 在线评估（单样本）

```http
POST /api/deep-evaluation/evaluate
```

**请求体**:
```json
{
  "input": "什么是机器学习？",
  "expected_output": "机器学习是人工智能的一个分支...",
  "actual_output": "机器学习是一种让计算机从数据中学习的方法...",
  "retrieval_context": ["上下文1", "上下文2"],
  "metrics": ["answer_relevancy", "faithfulness"],
  "llm_config": {
    "config_id": "llm-config-id",
    "temperature": 0,
    "top_p": 1,
    "top_k": null,
    "max_tokens": 2048,
    "timeout": 60,
    "max_retries": 3
  }
}
```

**响应**:
```json
{
  "results": {
    "answer_relevancy": { "score": 0.82, "reason": "..." }
  },
  "overall_score": 0.82
}
```

---

## 在线评估（单样本测试）

用于快速测试深度评估，无需创建完整的评估任务。

```http
POST /api/deep-evaluation/evaluate
```

**请求体**:
```json
{
  "input": "什么是机器学习？",
  "expected_output": "机器学习是人工智能的一个分支...",
  "actual_output": "机器学习是一种让计算机从数据中学习的技术...",
  "retrieval_context": [
    "机器学习（Machine Learning）是人工智能的核心...",
    "深度学习是机器学习的一个子领域..."
  ],
  "metrics": ["answer_relevancy", "faithfulness"],
  "llm_config": {
    "config_id": "llm-config-uuid"
  }
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input` | string | 是 | - | 用户问题/Query |
| `expected_output` | string | 否 | null | 期望的答案 |
| `actual_output` | string | 否 | null | 实际生成的答案 |
| `retrieval_context` | array | 否 | [] | 检索到的上下文列表 |
| `metrics` | array | 否 | ["answer_relevancy"] | 要使用的评估指标 |
| `llm_config` | object | 是 | - | LLM 配置 |

**响应**:
```json
{
  "results": {
    "answer_relevancy": {
      "score": 0.85,
      "reason": "答案与问题高度相关，准确解释了机器学习的概念"
    },
    "faithfulness": {
      "score": 0.92,
      "reason": "答案内容忠实于检索到的上下文"
    }
  },
  "overall_score": 0.885
}
```

---

## 集合-模型 Mismatch 检测

在线检索模式（`retrieval_mode: "online"`）下，系统自动检测检索 embedding 模型与 Milvus 集合入库模型是否一致：

1. 比对 `embedding_config_id`（精确匹配）
2. 比对 `embedding_model` 名称（模糊匹配）

**不一致时的行为**:
- `allow_collection_model_mismatch=false`（默认）: 返回 400 错误，提示用户更换模型或显式允许 mismatch
- `allow_collection_model_mismatch=true`: 允许创建任务，但 `worker_groups` 中标记 `evaluation_mode: "fallback"`、`model_mismatch: true`，提示结果仅供参考

> 不同模型产生的向量空间不一致，跨模型检索的召回质量无法保证。此检测旨在防止用户误用错误的模型-集合组合。
