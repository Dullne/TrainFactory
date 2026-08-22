# 模型配置模块

## 概述

管理外部 LLM API 配置（如 OpenAI、本地部署的模型），用于数据生成、深度评估等需要调用 LLM 的场景。支持健康检测和 API 测试。

## 核心功能

| 功能 | 说明 |
|------|------|
| 配置管理 | 创建、编辑、删除 LLM API 配置 |
| 健康检测 | 自动检测 API 端点可用性 |
| API 测试 | 前端支持直接测试对话 |

## 核心组件

```
train_factory/api/model_config_routes.py   # API 路由
train_factory/storage/model_config_*.py    # Entity + Service
```

## 相关文档

- 数据库表: [model_configs](../database/model_configs.md)
- API 文档: [模型配置 API](../api/configs.md)
