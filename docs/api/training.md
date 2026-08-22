# 训练管理 API

## 创建训练任务

```http
POST /api/train
```

**请求体**:
```json
{
  "model_name_or_path": "BAAI/bge-base-zh-v1.5",
  "dataset_name_or_path": "sentence-transformers/all-nli",
  "task_name": "my-embedding-task",
  "description": "Training embedding model",

  "model_type": "embedding",
  "training_method": "sft",

  "num_train_epochs": 3,
  "per_device_train_batch_size": 16,
  "learning_rate": 2e-5,
  "warmup_ratio": 0.1,

  "use_lora": false,
  "bf16": true,

  "loss_config": {
    "name": "MultipleNegativesRankingLoss",
    "params": {"scale": 20.0}
  }
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model_name_or_path` | string | 是 | - | 模型名称或路径 |
| `dataset_name_or_path` | string | 是 | - | 数据集名称或路径 |
| `task_name` | string | 否 | null | 任务名称 |
| `model_type` | string | 否 | embedding | 模型类型: embedding, reranker, decoder_reranker, llm |
| `training_method` | string | 否 | sft | 训练方法: sft, dpo, grpo, dapo, orpo, reinforce, two_stage |
| `num_train_epochs` | int | 否 | 3 | 训练轮数 |
| `per_device_train_batch_size` | int | 否 | 16 | 每设备批次大小 |
| `learning_rate` | float | 否 | 2e-5 | 学习率 |
| `warmup_ratio` | float | 否 | 0.1 | 预热比例 |
| `use_lora` | bool | 否 | false | 是否使用 LoRA |
| `bf16` | bool | 否 | false | 是否使用 bf16 混合精度 |
| `loss_config` | object | 否 | null | 损失函数配置 |
| `rl_config` | object | 否 | null | RL 训练配置 |

`rl_config` 对象参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `beta` | float | 否 | 0.1 | DPO/ORPO 的 beta 参数，控制 KL 散度惩罚强度。取值范围：0.01-10 |
| `rankings_direction` | string | 否 | `auto` | 排序方向，用于 responses+rankings 格式。可选值：`auto`（自动推断）、`higher_is_better`（分数越大越好）、`lower_is_better`（排名越小越好） |

**响应**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_name": "my-embedding-task",
  "status": "pending",
  "message": "Training task created successfully"
}
```

## 获取任务状态

```http
GET /api/train/{task_id}
```

**响应**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_name": "my-embedding-task",
  "train_type": "embedding",
  "model_type": "embedding",
  "training_method": "sft",
  "model_architecture": "encoder",
  "status": "running",
  "progress": 45.5,
  "error_message": null,
  "final_model_path": null,
  "dataset_configs": [
    {
      "path": "/app/data/datasets/all-nli",
      "max_samples": null,
      "split": "train",
      "num_rows": 12500
    }
  ],
  "gpu_ids": [0, 1],
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": null
}
```

> **`dataset_configs`**: 训练使用的数据集配置列表，支持多数据集训练。每项包含 `path`（数据集路径）、`max_samples`（最大样本数，null 表示全量）、`split`（数据集分片）、`num_rows`（行数，由后端补充）。

> **`gpu_ids` 归一化**: 响应中的 `gpu_ids` 总是返回 `List[int]` 格式。后端接受多种输入格式（列表、整数、逗号分隔字符串如 `"0,1"`），统一归一化为整数列表返回。

## 列出训练任务

```http
GET /api/train?status=running&model_type=embedding&limit=10
```

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `status` | string | 按状态过滤 |
| `model_type` | string | 按模型类型过滤 |
| `training_method` | string | 按训练方法过滤 |
| `limit` | int | 返回数量限制 |
| `offset` | int | 偏移量 |

## 停止任务

```http
POST /api/train/{task_id}/stop
```

## 删除任务

```http
DELETE /api/train/{task_id}
```
