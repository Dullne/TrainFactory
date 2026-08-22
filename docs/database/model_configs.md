# model_configs - 模型配置

存储外部 API 模型配置。

| 字段 | 类型 | 说明 |
|------|------|------|
| `config_id` | VARCHAR(36) | 配置 UUID |
| `config_name` | VARCHAR(255) | 配置名称 |
| `source_type` | VARCHAR(32) | 来源: external_api, local_deployed |
| `model_type` | VARCHAR(50) | 模型类型: llm, embedding, rerank |
| `provider` | VARCHAR(50) | 提供商: openai, azure, xinference, ollama |
| `api_endpoint` | VARCHAR(1024) | API 端点 |
| `api_key` | VARCHAR(512) | API 密钥 (加密存储) |
| `model_name` | VARCHAR(255) | 模型名称 |
| `embedding_dim` | INT | Embedding 维度 |
| `is_default` | BOOLEAN | 是否默认配置 |

## 相关文档

- 模块设计: [模型配置](../modules/configs.md)
- API 文档: [模型配置 API](../api/configs.md)
