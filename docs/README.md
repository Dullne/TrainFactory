# TrainFactory 文档

## 文档目录

| 目录 | 说明 |
|------|------|
| [architecture/](./architecture/README.md) | 系统架构：架构图、技术选型、目录结构、前端架构 |
| [modules/](./modules/README.md) | 模块设计：各板块的设计文档（训练、评估、部署等） |
| [database/](./database/README.md) | 数据库：表结构定义、迁移管理、Entity/Service |
| [api/](./api/README.md) | API 接口：REST API 详细说明 |

## 快速开始

### 启动服务

```bash
cd docker
docker compose up -d
```

### 访问 API

- API 文档: http://localhost:18000/docs
- 健康检查: http://localhost:18000/health

### 创建训练任务

```bash
curl -X POST http://localhost:18000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name_or_path": "BAAI/bge-base-zh-v1.5",
    "dataset_name_or_path": "sentence-transformers/all-nli",
    "model_type": "embedding",
    "training_method": "sft",
    "num_train_epochs": 3
  }'
```

## 功能概览

### 训练

| 模型类型 | 训练方法 | 状态 |
|----------|----------|------|
| Embedding | SFT | ✅ 可用 |
| Reranker (CrossEncoder) | SFT | ✅ 可用 |
| Decoder Reranker (Qwen3) | SFT/DPO/GRPO/DAPO/DR_GRPO | ✅ 可用 |
| LLM | SFT/DPO/ORPO | ✅ 可用 |

### 推理部署

| 框架 | 用途 | 状态 |
|------|------|------|
| Xinference | Embedding/Reranker 部署 | ✅ 可用 |
| vLLM | LLM/Reranker 部署 | ✅ 可用 |
| SGLang | LLM/Reranker 部署 | ✅ 可用 |

### 评估

| 功能 | 说明 | 状态 |
|------|------|------|
| MTEB 基准测试 | 标准 Reranking 数据集评估 | ✅ 可用 |
| 本地数据集评估 | 自定义数据集评估 | ✅ 可用 |
| 多模型对比 | 多模型并行评估+结果对比 | ✅ 可用 |
| 断点续评 | 中断后从上次位置继续 | ✅ 可用 |
| 深度评估 | IR + LLM 指标评估，按模型组配置 | ✅ 可用 |

### 数据生成

| 功能 | 说明 | 状态 |
|------|------|------|
| 文档生成 QA | 从文档批量生成问答对 | ✅ 可用 |
| 正负例生成 | 使用 Embedding + LLM 生成正负例 | ✅ 可用 |
| 多格式输出 | 支持 Universal/Triplet/Pair 等训练格式 | ✅ 可用 |
| 并发处理 | 支持并发调用 LLM 加速生成 | ✅ 可用 |

## 维护规范

### 文件组织

docs/ 下四大目录对应四个维度：

- `architecture/` — 系统设计（架构图、目录结构、前端架构）
- `modules/` — 板块设计（每个后端模块一个文件）
- `database/` — 表结构（每张数据库表一个文件）
- `api/` — API 接口文档（已有独立体系）

### 新增板块

1. `modules/` 新增 `<module>.md`
2. `database/` 新增对应表文件
3. `api/` 新增 API 文档（如有）
4. 更新 `modules/README.md` 和 `database/README.md` 索引

### 新增数据库表

1. `database/` 新增 `<table_name>.md`
2. 更新 `database/README.md` 表概览索引

### 三层对应关系

```
modules/training.md    ↔ database/training_tasks.md    ↔ api/training.md
modules/deployment.md  ↔ database/deployments.md       ↔ api/deployments.md
modules/evaluation.md  ↔ database/evaluation_tasks.md  ↔ api/evaluations.md
modules/generation.md  ↔ database/generation_tasks.md  ↔ api/generation.md
modules/datasets.md    ↔ database/datasets.md          ↔ api/datasets.md
modules/models.md      ↔ database/model_registry.md    ↔ api/models.md
modules/configs.md     ↔ database/model_configs.md     ↔ api/configs.md
modules/vectordb.md    ↔ database/milvus_collections.md↔ api/milvus.md
modules/sync.md        ↔ database/external_sync_*.md   ↔ api/sync.md
modules/resources.md   ↔                               ↔ api/resources.md
```

## 相关链接

- [项目 README](../README.md)
- [Docker 配置](../docker/)
