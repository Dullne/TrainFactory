# model_versions - 模型版本

记录模型的版本历史，每个版本对应一个独立的模型路径和可选的评估指标。

## 字段定义

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT (PK) | 自增主键 |
| `version_id` | VARCHAR(36) UNIQUE | 版本唯一标识 (UUID) |
| `model_id` | VARCHAR(36) | 关联模型 ID (model_registry) |
| `version` | VARCHAR(50) | 版本号 |
| `model_path` | VARCHAR(1024) | 该版本的模型文件路径 |
| `changelog` | TEXT | 变更说明 |
| `metrics` | JSON | 评估指标 |
| `created_at` | DATETIME | 创建时间 |

## 约束与索引

- **唯一约束**: `(model_id, version)` — 同一模型不允许重复版本号
- **索引**: `(model_id, created_at)` — 按模型查询版本历史

## 相关文档

- 模块设计: [模型注册](../modules/models.md)
- API 文档: [模型 API](../api/models.md)
- 关联表: [model_registry](model_registry.md)
