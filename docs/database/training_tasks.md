# training_tasks - 训练任务

存储所有训练任务的配置、状态和结果。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | VARCHAR(36) | 任务 UUID |
| `task_name` | VARCHAR(255) | 任务名称 |
| `model_type` | VARCHAR(32) | 模型类型: embedding, reranker, decoder_reranker, llm |
| `training_method` | VARCHAR(32) | 训练方法: sft, dpo, grpo, reinforce |
| `model_architecture` | VARCHAR(32) | 模型架构: encoder, decoder |
| `base_model_path` | VARCHAR(1024) | 基础模型路径 |
| `final_model_path` | VARCHAR(1024) | 最终模型路径 |
| `dataset_name_or_path` | VARCHAR(1024) | 数据集路径 |
| `is_lora` | BOOLEAN | 是否 LoRA 微调 |
| `status` | VARCHAR(50) | 状态: pending, running, succeeded, failed, stopped |
| `progress` | FLOAT | 进度 (0-100) |
| `training_params` | JSON | 训练参数（含 `dataset_configs`） |
| `final_metrics` | JSON | 最终指标 |
| `rl_config` | JSON | RL 训练配置 |
| `loss_config` | JSON | 损失函数配置 |
| `parent_task_id` | VARCHAR(36) | 父任务 ID (两阶段训练) |
| `trained_model_registry_id` | VARCHAR(36) | 训练产出的注册模型 ID |

## dataset_configs 结构

`training_params.dataset_configs` 记录训练数据集的配置和实际统计：

```json
[
  {
    "path": "/data/datasets/train.jsonl",
    "split": "train",
    "max_samples": null,
    "num_rows": 514
  },
  {
    "path": "/data/datasets/eval.jsonl",
    "split": "eval",
    "max_samples": 100,
    "num_rows": 100
  }
]
```

`num_rows` 由 `BaseTrainer._record_dataset_stats()` 在训练开始时写入，记录各 split 的实际样本数量。

## 相关文档

- 模块设计: [训练模块](../modules/training.md)
- API 文档: [训练 API](../api/training.md)
