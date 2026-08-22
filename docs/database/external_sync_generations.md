# external_sync_generations - 同步生成任务追踪

追踪同步触发的数据生成任务。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | VARCHAR(36) | 关联同步任务 ID |
| `generation_task_id` | VARCHAR(36) | 关联生成任务 ID (唯一) |
| `user_id` | VARCHAR(64) | 用户 ID |
| `input_batch_ids` | JSON | 输入批次 ID 列表 |
| `input_record_count` | INT | 输入记录数 |
| `output_dataset_id` | VARCHAR(36) | 输出数据集 ID |
| `output_sample_count` | INT | 输出样本数 |
| `status` | VARCHAR(32) | 状态: pending, completed, failed |
| `disabled` | BOOLEAN | 是否停用（默认 false）。停用后不参与训练合并，关联批次重置为 fetched |
| `created_at` | DATETIME | 创建时间 |
| `completed_at` | DATETIME | 完成时间 |

## 相关文档

- 模块设计: [数据同步](../modules/sync.md)
- API 文档: [数据同步 API](../api/sync.md)
