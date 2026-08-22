# external_sync_tasks - 外部数据同步任务

管理外部 API 增量同步任务，支持两级阈值自动触发流水线（数据→生成→训练→加载）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | VARCHAR(36) | 任务 UUID |
| `task_name` | VARCHAR(255) | 任务名称 |
| `user_id` | VARCHAR(64) | 用户 ID |
| `external_api_config_id` | VARCHAR(36) | 外部 API 配置 ID |
| `external_api_url` | VARCHAR(1024) | 外部 API 地址 |
| `external_auth_config` | JSON | 认证配置 (username/password 或 token) |
| `sync_interval_seconds` | INT | 同步间隔（秒），默认 300 |
| `boundary_rollback_seconds` | INT | 边界回退秒数（默认 5），防止同时间戳记录遗漏 |
| `last_sync_at` | DATETIME | 上次同步时间（外部 API 的 created_at） |
| `last_sync_boundary_ids` | JSON | 边界去重键列表 (id:session_id:doc_id) |
| `generation_threshold` | INT | Level 1 生成阈值，默认 500 |
| `generation_mode` | VARCHAR(32) | 生成模式，默认 doc_to_training |
| `generation_config` | JSON | 生成配置 (LLM/Embedding/Rerank 配置) |
| `training_threshold` | INT | Level 2 训练阈值，默认 1000 |
| `training_config` | JSON | 训练配置（含 model_type、loss 参数等，见 [sync API](../api/sync.md)） |
| `base_deployment_id` | VARCHAR(36) | 关联的部署 ID（用于 LoRA 热加载，为空时自动发现） |
| `pending_record_count` | INT | 待生成记录数 |
| `pending_training_samples` | INT | 待训练样本数 |
| `total_record_count` | INT | 累计记录数 |
| `total_training_samples` | INT | 累计训练样本数 |
| `total_trainings` | INT | 累计训练轮次 |
| `current_adapter_name` | VARCHAR(255) | 当前加载的 adapter 名称 |
| `current_adapter_id` | VARCHAR(36) | 当前加载的 adapter ID |
| `current_training_id` | VARCHAR(36) | 当前训练任务 ID |
| `is_active` | BOOLEAN | 是否启用 |
| `status` | VARCHAR(32) | 状态: idle, syncing, generating, training, loading_adapter, error |
| `error_message` | TEXT | 错误信息 |

## training_config JSON 结构

`training_config` 字段的内容根据 `model_type` 不同而包含不同的 loss 配置：

```json
{
  "model_type": "embedding",
  "training_method": "sft",
  "base_model_path": "/app/models/bge-base-zh-v1.5",
  "lora_r": 16,
  "lora_alpha": 32,
  "num_train_epochs": 3,
  "learning_rate": 2e-5,
  "gpu_ids": [0],
  "embedding_loss_name": "MultipleNegativesRankingLoss",
  "bf16": true
}
```

## 相关文档

- 模块设计: [数据同步](../modules/sync.md)
- API 文档: [数据同步 API](../api/sync.md)
