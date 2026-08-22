# 模块总览

## 模块职责

### 后端模块 (`train_factory/`)

| 模块 | 职责 | 实现状态 |
|------|------|----------|
| `api/` | REST API 接口层，处理 HTTP 请求 | ✅ 完整 |
| `trainers/` | 训练器核心逻辑，按模型架构分层 | ⚠️ encoder 完整，LLM + decoder reranker 已实现，其他 decoder 待实现 |
| `losses/` | 损失函数库，支持注册机制 | ⚠️ contrastive 完整，其他待集成 |
| `data/` | 数据加载、格式化、预处理 | ✅ 完整 |
| `evaluation/` | 评估：MTEB 基准测试、多模型并行评估 | ✅ 完整 |
| `monitoring/` | 训练指标记录、生命周期、资源监控 | ✅ 完整 |
| `storage/` | 数据库 ORM 实体和业务服务层 | ✅ 完整 |
| `deployment/` | 模型部署 (Xinference/vLLM/SGLang) | ✅ 完整 |
| `tuners/` | 微调方法（LoRA/QLoRA/Full） | ✅ 完整 |
| `core/` | 核心基础设施（设备管理、配置构建） | ✅ 完整 |
| `enums/` | 类型枚举定义 | ✅ 完整 |
| `schemas/` | Pydantic 数据模型 | ✅ 完整 |
| `utils/` | 工具函数（通用工具、训练配方、兼容性修复） | ✅ 完整 |

### 前端模块 (`web/`)

| 模块 | 职责 | 实现状态 |
|------|------|----------|
| `pages/training/` | 训练任务管理页面（列表、创建、详情） | ✅ 完整 |
| `pages/datasets/` | 数据集管理页面（列表、创建） | ✅ 完整 |
| `pages/models/` | 模型注册管理页面 | ✅ 完整 |
| `pages/deployments/` | 部署管理页面 | ✅ 完整 |
| `pages/evaluations/` | 评估管理页面（创建、详情、结果对比） | ✅ 完整 |
| `pages/configs/` | 模型配置管理页面 | ✅ 完整 |
| `components/` | 公共组件（Layout 等） | ✅ 完整 |
| `services/` | API 请求封装 | ✅ 完整 |
| `types/` | TypeScript 类型定义 | ✅ 完整 |

## 详细文档

### 后端模块
- [训练模块](training.md)
- [评估模块](evaluation.md)
- [数据生成](generation.md)
- [模型部署](deployment.md)
- [数据集管理](datasets.md)
- [模型注册](models.md)
- [模型配置](configs.md)
- [向量库](vectordb.md)
- [资源监控](resources.md)
- [数据同步](sync.md)

### 前端
- [前端架构](../architecture/frontend.md)
