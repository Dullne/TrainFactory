# 系统架构

## 项目概述

TrainFactory 是一个统一的模型训练平台，提供完整的 **后端 API + 前端 UI** 解决方案，支持多种模型类型和训练方法。

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户浏览器                            │
│                    (React + Ant Design)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST
┌─────────────────────────▼───────────────────────────────────┐
│                      FastAPI 后端                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ 训练API │  │ 模型API │  │ 部署API │  │ 评估API │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       └────────────┴────────────┴────────────┘              │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │                    核心训练引擎                         │  │
│  │  Trainers │ Losses │ Tuners │ Evaluation │ Monitoring │  │
│  └───────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      基础设施层                              │
│  MySQL │ Xinference │ vLLM │ SGLang │ GPU Cluster          │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术选型

### 后端

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM 训练框架 | transformers Trainer | HuggingFace 官方，生态完善 |
| RL 训练框架 | TRL (Transformer RL) | HuggingFace RL 库，支持 DPO/GRPO/PPO |
| 微调方法 | PEFT | Parameter-Efficient Fine-Tuning |
| 量化 | BitsAndBytes | 4-bit/8-bit 量化支持 |
| API 框架 | FastAPI | 高性能异步 API |
| ORM | SQLModel | Pydantic + SQLAlchemy |
| 实验追踪 | SwanLab | 训练可视化 |

### 前端

详见 [前端架构](frontend.md#技术栈)（React 18 + TypeScript + Vite + Ant Design 5 + ECharts + i18n）

## 详细文档

- [目录结构与设计原则](directory.md)
- [前端架构与 i18n](frontend.md)
- [API 端点概览](api-overview.md)
- [状态系统统一方案](status-system.md)

## 相关文档

- [模块设计文档](../modules/README.md) — 各板块的设计文档
- [数据库文档](../database/README.md) — 表结构定义
- [API 详细文档](../api/README.md) — REST API 接口说明
