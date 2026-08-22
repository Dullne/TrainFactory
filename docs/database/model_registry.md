# model_registry - 模型注册

存储已注册的模型信息。

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_id` | VARCHAR(36) | 模型 UUID |
| `model_name` | VARCHAR(255) | 模型名称 |
| `display_name` | VARCHAR(255) | 显示名称（可选） |
| `version` | VARCHAR(50) | 版本号 |
| `model_type` | VARCHAR(50) | 模型类型: embedding, reranker, decoder_reranker, llm |
| `model_path` | VARCHAR(1024) | 模型存储路径 |
| `base_model_path` | VARCHAR(1024) | 基础模型路径（可选） |
| `source_task_id` | VARCHAR(36) | 来源训练任务 ID |
| `source_type` | VARCHAR(32) | 来源: trained, downloaded, uploaded, external_bind |
| `is_adapter` | BOOLEAN | 是否为 LoRA adapter |
| `embedding_dim` | INT | Embedding 维度 |
| `metrics` | JSON | 性能指标 |
| `status` | VARCHAR(50) | 状态: registered, available, archived |
| `is_latest` | BOOLEAN | 是否最新版本 |
| `download_source` | VARCHAR(32) | 下载来源: modelscope, huggingface |
| `remote_repo` | VARCHAR(512) | 远程仓库地址 |
| `download_status` | VARCHAR(32) | 下载状态: pending, downloading, completed, failed |
| `download_progress` | INT | 下载进度 (0-100) |
| `download_error` | TEXT | 下载失败错误信息 |
| `extra_metadata` | JSON | 扩展元数据 |

## 状态生命周期

| 状态 | 说明 |
|------|------|
| `registered` | 模型元数据已注册，但文件未就绪（下载中或等待下载） |
| `available` | 模型文件已存在且可用，可直接用于部署或训练 |
| `archived` | 模型已归档，不再用于部署和训练 |

下载过程通过 `download_status` / `download_progress` / `download_error` 独立跟踪，不影响主状态。下载失败时模型保持 `registered` 状态，下载成功后自动变为 `available`。

## 相关文档

- 模块设计: [模型注册](../modules/models.md)
- API 文档: [模型 API](../api/models.md)
