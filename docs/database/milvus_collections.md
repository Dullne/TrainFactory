# milvus_collections - 向量集合注册表

独立管理 Milvus 向量集合，每个集合绑定唯一的 Embedding 模型。

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT | 自增主键 |
| `collection_id` | VARCHAR(36) | 集合 UUID（唯一索引） |
| `collection_name` | VARCHAR(255) | Milvus 集合名称（唯一索引） |
| `display_name` | VARCHAR(255) | 显示名称 |
| `description` | TEXT | 描述 |
| `embedding_config_id` | VARCHAR(36) | 绑定的 Embedding 模型配置 ID |
| `embedding_model` | VARCHAR(255) | Embedding 模型名称 |
| `embedding_endpoint` | VARCHAR(512) | Embedding API 端点 |
| `dim` | INT | 向量维度 |
| `metric_type` | VARCHAR(32) | 距离度量: COSINE, L2, IP |
| `status` | VARCHAR(32) | 状态: active, archived |
| `user_id` | VARCHAR(64) | 用户 ID |
| `created_at` | DATETIME | 创建时间 |
| `updated_at` | DATETIME | 更新时间 |

## 相关文档

- 模块设计: [向量库](../modules/vectordb.md)
- API 文档: [向量库 API](../api/milvus.md)
