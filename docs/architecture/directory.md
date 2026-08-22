# 目录结构

```
TrainFactory/
├── train_factory/                    # 主代码包
│   ├── __init__.py
│   ├── __main__.py                   # 入口点
│   ├── cli.py                        # CLI 命令行工具
│   ├── train.py                      # 训练主入口
│   │
│   ├── api/                          # REST API 层
│   │   ├── server.py                 # FastAPI 服务器
│   │   └── routes/                   # API 路由
│   │       ├── training_routes.py    # 训练任务 API
│   │       ├── registry_routes.py    # 模型注册 API
│   │       ├── deployment_routes.py  # 部署管理 API
│   │       ├── adapter_routes.py     # LoRA Adapter 管理 API
│   │       ├── model_config_routes.py # 模型配置 API
│   │       ├── dataset_routes.py     # 数据集管理 API
│   │       ├── evaluation_routes.py  # 评估管理 API (MTEB)
│   │       ├── deep_evaluation_routes.py # 深度评估 API (DeepEval)
│   │       ├── generation_routes.py  # 数据生成 API
│   │       ├── sync_routes.py        # 外部同步 API
│   │       ├── external_api_config_routes.py # 外部 API 配置
│   │       ├── milvus_routes.py      # Milvus 向量集合管理
│   │       ├── resource_routes.py    # GPU/资源监控 API
│   │       └── auth_routes.py        # 用户认证 API
│   │
│   ├── config/                       # 配置管理
│   │   └── settings.py               # 全局设置
│   │
│   ├── core/                         # 核心基础设施
│   │   ├── config_builder.py         # 配置构建器
│   │   ├── device_manager.py         # 设备管理（GPU/CPU）
│   │   └── gpu_resource_manager.py   # GPU 资源管理
│   │
│   ├── trainers/                     # 训练器（按架构分层）
│   │   ├── factory.py                # 训练器工厂
│   │   ├── base/                     # 基类
│   │   │   ├── base_trainer.py       # 抽象基类
│   │   │   └── training_result.py    # 训练结果容器
│   │   ├── encoder/                  # Encoder 模型 (仅 SFT)
│   │   │   ├── embedding_trainer.py  # Embedding 训练器
│   │   │   └── reranker_trainer.py   # CrossEncoder Reranker
│   │   └── decoder/                  # Decoder 模型 (SFT + RL: DPO, GRPO, KTO...)
│   │       ├── __init__.py
│   │       └── decoder_reranker_trainer.py  # Decoder Reranker (Qwen3-Reranker)
│   │
│   ├── losses/                       # 损失函数库
│   │   ├── registry.py               # 损失函数注册器
│   │   ├── contrastive/              # 对比学习损失
│   │   │   └── dynamic_negatives_loss.py
│   │   ├── ranking/                  # 排序损失（待集成）
│   │   │   └── __init__.py
│   │   ├── rl/                       # RL 损失（待集成）
│   │   │   └── __init__.py
│   │   └── similarity/               # 相似度损失（待集成）
│   │       └── __init__.py
│   │
│   ├── data/                         # 数据处理
│   │   ├── data_loader.py            # 数据加载器
│   │   ├── formats/                  # 数据格式定义
│   │   │   ├── base.py               # DataFormatType, DataSample 基类
│   │   │   ├── embedding.py          # Pair, Triplet, QPN 格式
│   │   │   ├── sft.py                # Instruction, ShareGPT 格式
│   │   │   └── dpo.py                # Preference 格式
│   │   └── preprocessors/            # 数据预处理器
│   │       ├── base.py               # BasePreprocessor 基类
│   │       ├── sft_preprocessor.py   # SFT 数据预处理
│   │       └── dpo_preprocessor.py   # DPO 数据预处理（待实现）
│   │
│   ├── evaluation/                   # 评估模块
│   │   ├── __init__.py               # 评估模块入口
│   │   ├── evaluation_runner.py      # 评估任务执行器 (MTEB + 多模型并行)
│   │   ├── evaluator.py              # UnifiedEvaluator 统一评估器
│   │   ├── metric_registry.py        # MetricRegistry 指标注册表
│   │   └── result.py                 # EvaluationResult 结果处理
│   │
│   ├── deep_evaluation/               # 深度评估模块 (检索/生成指标)
│   │   ├── deep_evaluation_runner.py # 深度评估任务执行器
│   │   ├── evaluator.py              # LLM-as-Judge 评估器
│   │   └── metrics/                  # DeepEval 指标
│   │
│   ├── generation/                   # 数据生成模块
│   │   ├── pipeline.py               # 生成流水线 (Phase 0 预索引 + 流式 QA/正负例)
│   │   └── steps/                    # 生成步骤
│   │       ├── base.py               # 步骤基类
│   │       ├── doc_quality.py        # 文档质量评估
│   │       ├── qa_gen.py             # QA 提取
│   │       ├── embedding_filter_step.py # 相似度过滤 + Milvus 入库 + 向量缓存
│   │       ├── pos_neg_gen.py        # 正负例生成 (LLM)
│   │       ├── hard_neg_extraction.py # 硬负例提取
│   │       ├── keypoint_gen.py       # 关键点生成
│   │       ├── role_gen.py           # 角色生成
│   │       └── validation.py         # 数据验证
│   │
│   ├── monitoring/                   # 监控模块
│   │   ├── training_metrics.py       # TrainingMetricsLogger 训练指标记录
│   │   ├── lifecycle_manager.py      # LifecycleManager 生命周期管理
│   │   └── resource_monitor.py       # ResourceMonitor 资源监控
│   │
│   ├── storage/                      # 数据持久化层
│   │   ├── database.py               # 数据库连接
│   │   ├── backends/                 # 存储后端抽象层
│   │   │   ├── base.py               # StorageBackend 抽象基类
│   │   │   ├── __init__.py           # get_storage_backend() 工厂 (单例缓存)
│   │   │   ├── local_backend.py      # 本地文件系统后端
│   │   │   └── s3_backend.py         # S3/MinIO 对象存储后端
│   │   ├── entities/                 # ORM 实体
│   │   │   ├── training_task_entity.py
│   │   │   ├── training_task_event_entity.py
│   │   │   ├── model_registry_entity.py
│   │   │   ├── deployment_entity.py
│   │   │   ├── loaded_adapter_entity.py
│   │   │   ├── model_config_entity.py
│   │   │   ├── dataset_entity.py
│   │   │   ├── dataset_asset_entity.py   # 数据集文件资产
│   │   │   ├── dataset_lineage_entity.py # 数据集血缘关系
│   │   │   ├── evaluation_task_entity.py
│   │   │   ├── generation_task_entity.py
│   │   │   ├── milvus_collection_entity.py # Milvus 集合注册
│   │   │   ├── external_sync_entity.py   # 外部同步配置/任务
│   │   │   ├── external_api_config_entity.py # 外部 API 配置
│   │   │   ├── user_entity.py
│   │   │   └── audit_log_entity.py
│   │   ├── services/                 # 业务服务
│   │   │   ├── training_task_service.py
│   │   │   ├── training_task_event_service.py
│   │   │   ├── model_registry_service.py
│   │   │   ├── model_config_service.py
│   │   │   ├── dataset_service.py
│   │   │   ├── dataset_asset_service.py
│   │   │   ├── dataset_lineage_service.py
│   │   │   ├── dataset_download_service.py
│   │   │   ├── dataset_schema.py          # 数据集列结构检测
│   │   │   ├── evaluation_task_service.py
│   │   │   ├── deep_evaluation_task_service.py
│   │   │   ├── generation_task_service.py
│   │   │   ├── milvus_collection_service.py
│   │   │   ├── external_sync_service.py
│   │   │   ├── external_api_config_service.py
│   │   │   ├── model_download_service.py
│   │   │   └── audit_log_service.py
│   │   └── migrations/              # Alembic 数据库迁移
│   │       └── alembic/
│   │
│   ├── deployment/                   # 部署服务
│   │   ├── deployment_service.py     # 部署管理 (多框架支持)
│   │   ├── docker_deployer.py        # Docker 容器部署器
│   │   ├── adapter_service.py        # LoRA Adapter 热加载/卸载
│   │   ├── xinference_client.py      # Xinference 客户端
│   │   ├── vllm_client.py            # vLLM 客户端
│   │   └── sglang_client.py          # SGLang 客户端
│   │
│   ├── tuners/                       # 微调方法
│   │   ├── base.py                   # BaseTuner 抽象基类
│   │   ├── registry.py               # TunerRegistry 注册表
│   │   ├── lora.py                   # LoRA 微调
│   │   ├── qlora.py                  # QLoRA (4-bit/8-bit 量化)
│   │   └── full_param.py             # 全参数微调 / 冻结层
│   │
│   ├── enums/                        # 枚举类型
│   │   ├── training_type.py          # 训练类型
│   │   ├── training_status.py        # 训练状态
│   │   ├── model_architecture.py     # 模型架构/类型/方法
│   │   ├── model_status.py           # 模型状态
│   │   ├── deployment_status.py      # 部署状态
│   │   └── ...
│   │
│   ├── schemas/                      # Pydantic Schema
│   │   └── training_config.py        # 训练配置
│   │
│   ├── sync/                         # 外部同步模块
│   │   ├── sync_manager.py           # SyncManager 同步管理器
│   │   ├── sync_worker.py            # SyncWorker 批次处理/训练触发
│   │   ├── level2_handler.py         # Level 2 训练编排处理
│   │   ├── post_training_handler.py  # 训练后处理 (评估/部署)
│   │   └── external_client.py        # 外部系统 HTTP 客户端
│   │
│   └── utils/                        # 工具函数
│       ├── common_utils.py           # 通用工具 (init_swanlab, setup_logging)
│       ├── peft_workaround.py        # PEFT 兼容性修复
│       └── embedding_recipe.py       # Embedding 训练配方
│
├── web/                              # 前端 (React + TypeScript)
│   ├── src/
│   │   ├── App.tsx                   # 应用入口
│   │   ├── main.tsx                  # React 入口
│   │   ├── components/               # 公共组件
│   │   │   └── Layout.tsx            # 页面布局
│   │   ├── pages/                    # 页面组件
│   │   │   ├── training/             # 训练任务页面
│   │   │   │   ├── TrainingList.tsx
│   │   │   │   ├── TrainingCreate.tsx
│   │   │   │   └── TrainingDetail.tsx
│   │   │   ├── datasets/             # 数据集管理页面
│   │   │   │   ├── DatasetList.tsx
│   │   │   │   └── DatasetCreate.tsx
│   │   │   ├── models/               # 模型管理页面
│   │   │   │   └── ModelList.tsx
│   │   │   ├── deployments/          # 部署管理页面
│   │   │   │   └── DeploymentList.tsx
│   │   │   ├── configs/              # 配置管理页面
│   │   │   │   └── ConfigList.tsx
│   │   │   ├── sync/                 # 外部同步页面
│   │   │   ├── evaluations/          # 评估页面
│   │   │   ├── generation/           # 数据生成页面
│   │   │   ├── resources/            # 资源监控页面
│   │   │   ├── vectordb/             # 向量数据库管理页面
│   │   │   └── rag-evaluation/       # RAG 评估页面
│   │   ├── services/                 # API 服务
│   │   │   └── api.ts
│   │   ├── types/                    # TypeScript 类型定义
│   │   │   └── index.ts
│   │   ├── hooks/                    # 自定义 Hooks
│   │   └── utils/                    # 工具函数
│   ├── package.json
│   ├── vite.config.ts
│   ├── Dockerfile                    # 前端 Docker 配置
│   └── nginx.conf                    # Nginx 配置
│
├── docker/                           # Docker 配置
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── init.sql                      # 数据库初始化脚本
│
├── docs/                             # 文档目录
├── configs/                          # 配置文件
├── scripts/                          # 脚本工具
├── tests/                            # 测试用例
├── models/                           # 模型存储目录
├── output/                           # 训练输出目录
└── data/                             # 数据目录
    └── datasets/
```

## 目录层级设计原则

### 为什么采用多层级目录？

当前采用 **2-3 层目录结构**，主要基于以下考虑：

| 设计原则 | 说明 | 示例 |
|----------|------|------|
| **按职责分层** | 顶层按功能职责划分 | `trainers/`, `losses/`, `data/` |
| **按类型细分** | 第二层按具体类型划分 | `trainers/encoder/`, `trainers/decoder/` |
| **避免过深嵌套** | 控制在 3 层以内 | `data/formats/embedding.py` |

### 对比：扁平 vs 多层级

```
❌ 扁平结构（不推荐）          ✅ 多层级结构（当前采用）
trainers/                      trainers/
├── embedding_trainer.py       ├── base/
├── reranker_trainer.py        │   └── base_trainer.py
├── llm_sft_trainer.py         ├── encoder/        # Encoder (仅 SFT)
├── llm_dpo_trainer.py         │   ├── embedding_trainer.py
├── llm_grpo_trainer.py        │   └── reranker_trainer.py
├── decoder_reranker_sft.py    └── decoder/        # Decoder (SFT + RL)
├── decoder_reranker_dpo.py        ├── sft_trainer.py
└── ...（文件爆炸）                ├── dpo_trainer.py
                                   └── grpo_trainer.py
```

**设计原则：RL 训练仅适用于 Decoder 模型**，因此将 RL 训练器（DPO、GRPO、KTO 等）直接放在 `decoder/` 目录下，而非单独的 `rl/` 目录。

### 层级设计规范

| 层级 | 用途 | 命名规范 |
|------|------|----------|
| **第 1 层** | 功能模块 | 名词复数：`trainers`, `losses`, `enums` |
| **第 2 层** | 架构分组 | 按模型架构：`encoder`, `decoder` |
| **第 3 层** | 具体实现 | 具体文件：`embedding_trainer.py` |

### 扩展示例

新增 LLM KTO 训练时，只需在 `decoder/` 目录添加：
```
trainers/decoder/
├── __init__.py
├── sft_trainer.py
├── cpt_trainer.py
├── dpo_trainer.py
├── grpo_trainer.py
└── kto_trainer.py    # 新增
```

而不是在扁平目录中创建 `llm_kto_trainer.py`，避免与其他文件混淆。

## 相关文档

- [架构首页](README.md)
- [模块职责](../modules/README.md)
