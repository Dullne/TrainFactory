# collection_dataset_links - 集合数据集关联

记录集合与数据集的多对多关系。

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT | 自增主键 |
| `collection_name` | VARCHAR(255) | 集合名称 |
| `dataset_id` | VARCHAR(36) | 数据集 ID |
| `dataset_name` | VARCHAR(255) | 数据集名称 |
| `chunk_count` | INT | chunk 数量 |
| `task_id` | VARCHAR(36) | 关联的生成任务 ID |
| `linked_at` | DATETIME | 关联时间 |

**唯一约束**: `(collection_name, dataset_id)`

## 相关文档

- 模块设计: [向量库](../modules/vectordb.md)
- API 文档: [向量库 API](../api/milvus.md)
