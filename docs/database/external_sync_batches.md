# external_sync_batches - 同步批次

记录每次增量拉取的数据批次。

| 字段 | 类型 | 说明 |
|------|------|------|
| `batch_id` | VARCHAR(36) | 批次 UUID |
| `task_id` | VARCHAR(36) | 关联同步任务 ID |
| `user_id` | VARCHAR(64) | 用户 ID |
| `record_count` | INT | 记录数 |
| `storage_path` | VARCHAR(1024) | JSONL 文件路径 |
| `dataset_id` | VARCHAR(36) | 关联数据集 ID |
| `since_time` | DATETIME | 拉取起始时间 |
| `until_time` | DATETIME | 拉取截止时间 |
| `fetched_at` | DATETIME | 拉取时间 |
| `status` | VARCHAR(32) | 状态: fetched, registered, generation_queued, generation_done |
| `generation_task_id` | VARCHAR(36) | 关联的生成任务 ID |

## 相关文档

- 模块设计: [数据同步](../modules/sync.md)
- API 文档: [数据同步 API](../api/sync.md)
