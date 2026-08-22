# external_sync_trainings - 同步训练任务追踪

追踪同步触发的模型训练任务及 adapter 加载状态。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | VARCHAR(36) | 关联同步任务 ID |
| `training_task_id` | VARCHAR(36) | 关联训练任务 ID (唯一) |
| `user_id` | VARCHAR(64) | 用户 ID |
| `input_dataset_ids` | JSON | 输入数据集 ID 列表 |
| `total_samples` | INT | 总样本数 |
| `training_round` | INT | 训练轮次 |
| `output_adapter_path` | VARCHAR(1024) | 输出 adapter 路径 |
| `output_model_registry_id` | VARCHAR(36) | 输出模型注册 ID |
| `loaded_adapter_name` | VARCHAR(255) | 已加载 adapter 名称 |
| `loaded_adapter_id` | VARCHAR(36) | 已加载 adapter ID |
| `previous_training_task_id` | VARCHAR(36) | 上一轮训练任务 ID |
| `status` | VARCHAR(32) | 状态（见下方状态说明） |
| `created_at` | DATETIME | 创建时间 |
| `completed_at` | DATETIME | 完成时间 |

## 状态说明

| 状态 | 说明 |
|------|------|
| `pending` | 训练进行中 |
| `completed` | 训练完成，等待 adapter 加载 |
| `failed` | 训练本身失败 |
| `adapter_loaded` | Adapter 已成功加载到部署 |
| `adapter_unloaded` | Adapter 已卸载（被替换或手动卸载） |
| `adapter_load_failed` | Adapter 加载失败（可重试） |
| `adapter_failed` | Adapter 操作失败 |

> `adapter_load_failed` 与 `failed` 的区别：前者表示训练成功但 adapter 加载到推理服务时失败，后者表示训练本身失败。

## 相关文档

- 模块设计: [数据同步](../modules/sync.md)
- API 文档: [数据同步 API](../api/sync.md)
