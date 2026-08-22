# deployments - 部署记录

存储模型部署信息。

| 字段 | 类型 | 说明 |
|------|------|------|
| `deployment_id` | VARCHAR(36) | 部署 UUID |
| `model_id` | VARCHAR(36) | 关联模型 ID |
| `model_uid` | VARCHAR(255) | 推理服务中的模型 UID |
| `deployment_name` | VARCHAR(255) | 部署名称 |
| `xinference_endpoint` | VARCHAR(512) | 推理服务端点 |
| `inference_framework` | VARCHAR(32) | 推理框架: xinference, vllm, sglang |
| `deploy_mode` | VARCHAR(32) | 部署模式: external, container |
| `container_name` | VARCHAR(255) | Docker 容器名 |
| `gpu_id` | INT | GPU 编号 |
| `port` | INT | 服务端口 |
| `enable_lora` | BOOLEAN | 是否启用 LoRA 热加载 |
| `max_loras` | INT | 最大同时加载的 LoRA adapter 数量 |
| `max_lora_rank` | INT | 最大 LoRA 秩 |
| `health_status` | VARCHAR(32) | 健康状态: HEALTHY, UNHEALTHY, UNKNOWN |
| `status` | VARCHAR(50) | 状态: pending, deploying, running, stopped, failed |

## LoRA 热加载

当 `enable_lora=true` 时，部署支持运行时加载/卸载 LoRA adapter（仅 vLLM 和 SGLang 框架）。

加载的 adapter 记录存储在 [`loaded_adapters`](loaded_adapters.md) 表中。

## 相关文档

- 模块设计: [模型部署](../modules/deployment.md)
- API 文档: [部署 API](../api/deployments.md)
- Adapter 记录: [loaded_adapters](loaded_adapters.md)
