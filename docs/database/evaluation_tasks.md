# evaluation_tasks - 评估任务

统一的评估任务表，支持 MTEB（有标签评估）和 DeepEval（LLM-as-Judge 评估）两种框架。

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT | 自增主键 |
| `task_id` | VARCHAR(36) | 任务 UUID（唯一索引） |
| `task_name` | VARCHAR(255) | 任务名称 |
| `description` | TEXT | 任务描述 |
| `eval_framework` | VARCHAR(32) | 评估框架: mteb, deepeval |
| `eval_type` | VARCHAR(32) | 评估类型（见下方说明） |
| `model_configs` | JSON | 模型配置列表（见下方说明） |
| `dataset_configs` | JSON | 数据集配置列表（见下方说明） |
| `field_mapping` | JSON | 字段映射 (DeepEval): {query, positives, negatives, input, expected_output, ...} |
| `metrics` | JSON | 评估指标列表 (DeepEval): ["mrr", "ndcg@10", "answer_relevancy", ...] |
| `llm_config` | JSON | LLM 配置 (DeepEval): {endpoint, model, api_key, ...} |
| `worker_groups` | JSON | 工作组配置 (DeepEval): {model_workers: 2} |
| `max_samples` | INT | 每数据集最大样本数 |
| `batch_size` | INT | API 批次大小 (MTEB) |
| `workers` | INT | 并发请求数 |
| `model_workers` | INT | 模型并行数 (MTEB) |
| `status` | VARCHAR(50) | 状态: pending, running, succeeded/completed, failed, cancelled |
| `progress` | FLOAT | 总进度 (0-100) |
| `current_model` | VARCHAR(255) | 当前评估模型 |
| `current_dataset` | VARCHAR(255) | 当前评估数据集 |
| `total_samples` | INT | 总样本数 (DeepEval) |
| `processed_samples` | INT | 已处理样本数 (DeepEval) |
| `model_progress` | JSON | 每模型每数据集进度 (MTEB) |
| `results` | JSON | 评估结果 |
| `results_path` | VARCHAR(1024) | 详细结果文件路径 |
| `error_message` | TEXT | 错误信息 |
| `user_id` | VARCHAR(64) | 用户 ID（用户隔离） |
| `created_at` | DATETIME | 创建时间 |
| `updated_at` | DATETIME | 更新时间 |
| `started_at` | DATETIME | 开始时间 |
| `completed_at` | DATETIME | 完成时间 |

**eval_framework 说明**:
- `mteb`: MTEB 框架，有标签数据评估，支持 NDCG、MRR、MAP 等指标
- `deepeval`: DeepEval 框架，LLM-as-Judge 评估，支持 answer_relevancy、faithfulness 等指标

**eval_type 说明**:
- MTEB: `single` / `multi_model` / `multi_dataset`
- DeepEval: `embedding` / `rerank` / `llm` / `multi`

**model_configs 格式**:

MTEB 格式:
```json
[{"name": "model-a", "endpoint": "http://...", "model_name": "...", "inference_framework": "xinference"}]
```

DeepEval 格式（多模型组）:
```json
[
  {
    "group_name": "baseline",
    "embedding": {"config_id": "...", "endpoint": "...", "model_name": "...", "concurrency": 8},
    "rerank": {"config_id": "...", "endpoint": "...", "model_name": "...", "concurrency": 8},
    "llm": {"config_id": "...", "endpoint": "...", "model": "...", "temperature": 0.0, "concurrency": 2}
  }
]
```

**dataset_configs 格式**:

MTEB 格式:
```json
[{"type": "mteb", "name": "T2Reranking"}, {"type": "local", "name": "...", "path": "..."}]
```

DeepEval 格式:
```json
[{"dataset_id": "uuid-1", "dataset_name": "test-dataset"}]
```

## 相关文档

- 模块设计: [评估模块](../modules/evaluation.md)
- API 文档: [评估 API](../api/evaluations.md)
