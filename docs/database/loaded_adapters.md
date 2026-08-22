# loaded_adapters - 已加载的 LoRA Adapter

追踪在部署上加载的 LoRA adapter 状态。

## 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT | 自增主键 |
| `adapter_id` | VARCHAR(36) | Adapter UUID（业务主键） |
| `deployment_id` | VARCHAR(36) | 关联部署 ID |
| `adapter_name` | VARCHAR(255) | Adapter 名称（推理请求中的 model 字段） |
| `adapter_path` | VARCHAR(1024) | Adapter 权重文件路径 |
| `source_task_id` | VARCHAR(36) | 来源训练任务 ID |
| `source_model_id` | VARCHAR(36) | 来源模型注册 ID |
| `status` | VARCHAR(32) | 状态: loading, loaded, unloading, unloaded, failed |
| `error_message` | TEXT | 错误信息 |
| `user_id` | VARCHAR(64) | 用户 ID（多租户隔离） |
| `loaded_at` | DATETIME | 加载时间 |
| `unloaded_at` | DATETIME | 卸载时间 |

## 索引

| 索引名 | 字段 | 说明 |
|--------|------|------|
| `idx_adapter_deployment` | `deployment_id` | 按部署查询 |
| `idx_adapter_source_task` | `source_task_id` | 按来源训练任务查询 |
| `idx_adapter_status` | `status` | 按状态过滤 |

## 状态流转

```
loading → loaded → unloading → unloaded
   ↓                  ↓
 failed            failed → loading (重试)
```

- `loading`: 正在加载到推理服务
- `loaded`: 已成功加载，可用于推理
- `unloading`: 正在从推理服务卸载
- `unloaded`: 已卸载（终态）
- `failed`: 加载/卸载失败，可重试加载

## 相关文档

- 模块设计: [模型部署](../modules/deployment.md)
- API 文档: [部署 API - Adapter 管理](../api/deployments.md#lora-adapter-管理)
- Entity: `train_factory/storage/entities/loaded_adapter_entity.py`
- Service: `train_factory/deployment/adapter_service.py`
