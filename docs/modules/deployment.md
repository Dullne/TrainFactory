# 模型部署模块

## 概述

部署模块支持多种推理框架，通过 Docker 容器部署模型，支持 LoRA adapter 热加载。

## 推理框架

| 框架 | 说明 | 适用模型 | LoRA 热加载 |
|------|------|----------|-------------|
| Xinference | 通用推理框架 | Embedding, Reranker, LLM | 不支持 |
| vLLM | 高性能推理 | LLM, Embedding, Decoder Reranker | 支持（含 Embedding LoRA） |
| SGLang | 高性能 LLM 推理 | LLM, Decoder Reranker | 支持（仅 decode 模式） |

> **注意**: vLLM 是唯一同时支持 Embedding 模型 + LoRA 热加载的框架（通过 `--task pooling --enable-lora` 参数）。SGLang 的 LoRA 仅支持 decode 模式（LLM 生成），不支持 Embedding/Reranker。

## 核心组件

```
train_factory/deployment/
├── deployment_service.py    # 部署管理（多框架支持）
├── docker_deployer.py       # Docker 容器部署器
├── adapter_service.py       # LoRA Adapter 管理（加载/卸载/同步）
├── xinference_client.py     # Xinference 客户端
├── vllm_client.py           # vLLM 客户端
└── sglang_client.py         # SGLang 客户端
```

## LoRA Adapter 热加载

### 工作原理

1. 创建部署时设置 `enable_lora=True`、`max_loras`、`max_lora_rank`
2. 部署启动后，通过 Adapter API 将训练好的 LoRA 权重加载到运行中的推理服务
3. 请求时将 `model` 字段设为 adapter 名称即可使用 adapter 模型
4. 可随时卸载 adapter，释放 LoRA 槽位

### Adapter 来源

- **训练任务**: 已完成的 LoRA 训练任务（`is_lora=True` 且 `status=succeeded`）
- **模型注册**: 注册为 adapter 的模型（`is_adapter=True`）

### 自动发现

前端 AdapterManager 组件提供"发现可用 Adapter"功能，自动匹配与部署基座模型相同 `base_model_path` 的可用 adapter。

### Adapter 状态流转

```
loading → loaded → unloading → unloaded
   ↓                  ↓
 failed            failed → loading (重试)
```

### 连接测试中的 Adapter 信息

通过 `/v1/models` 端点可以识别已加载的 adapter：
- 基座模型: `parent` 字段为 `null`
- LoRA adapter: `parent` 字段指向基座模型名称

前端连接测试弹窗中，adapter 模型会显示蓝色 "LoRA" 标签。

## 相关文档

- 数据库表: [deployments](../database/deployments.md)、[loaded_adapters](../database/loaded_adapters.md)
- API 文档: [部署 API](../api/deployments.md)
- 前端组件: [AdapterManager](../architecture/frontend.md)
