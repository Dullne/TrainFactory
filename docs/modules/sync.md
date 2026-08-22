# 数据同步模块

## 概述

外部数据同步模块，支持从外部系统自动拉取增量数据，经过三阶段流水线（拉取→生成→训练）实现数据驱动的模型持续优化。

## 核心功能

| 功能 | 说明 |
|------|------|
| 增量同步 | 基于时间戳从外部 API 拉取增量数据，三层去重保证无重复 |
| 批次管理 | 跟踪每次同步批次的存储、注册、生成状态 |
| 自动流水线 | 两级阈值驱动：记录数 → 数据生成，训练样本数 → 模型训练 |
| Adapter 生命周期 | 训练后自动加载，支持手动重试/卸载/替换，`adapter_load_failed` 状态可重试 |
| 部署自动发现 | 无需指定 `base_deployment_id`，自动匹配兼容的 running 部署 |
| 生成数据管理 | 停用/启用已完成的生成数据，停用后批次自动重置，支持重新生成 |
| 计数器修复 | 支持一键重算 `total_training_samples`/`pending_training_samples`，修复计数漂移 |

## 三阶段同步周期

每次 worker 轮询执行以下三个阶段：

```
Phase 1: 增量拉取
  外部 API ──since──→ 拉取新数据 ──→ 三层去重 ──→ 创建批次
                                                     │
Phase 2: Level 1 阈值检查                             │
  pending_record_count >= generation_threshold? ──Yes──→ 合并批次 → 数据生成任务
                                                          │
Phase 3: Level 2 阈值检查                                  │
  pending_training_samples >= training_threshold? ──Yes──→ 合并生成数据 → 训练任务
                                                               │
                                                               ▼
                                                         LoRA Adapter 热加载
```

### Phase 1: 增量拉取 + Stage-1 预索引

- 以 `last_sync_at` 为起点向外部 API 发起分页请求
- 回退 5 秒时间窗口，覆盖同时间戳的边界记录
- 三层去重后写入批次文件（JSONL 格式）
- 写入时为每条记录生成 **稳定 chunk_id**（`_build_stable_chunk_id()`），基于 `external_id:session_id:doc_id` 组合键，确保跨同步/生成阶段 doc_id 一致
- 批次初始状态为 **`REGISTERED`**，然后执行 Stage-1 Milvus 预索引

#### 两阶段批次生命周期

```
REGISTERED ──Stage-1 成功──→ FETCHED ──→ generation_queued ──→ generation_done
     │
     └── Stage-1 失败 → 保持 REGISTERED → 下一轮 run_once 重试
```

- `REGISTERED`: 批次文件已写入，但尚未完成 Milvus 预索引
- `FETCHED`: Stage-1 预索引成功，计入 `pending_record_count`，可参与数据生成

#### Stage-1: Milvus 预索引

批次从 REGISTERED 提升到 FETCHED 之前，`_pre_index_to_milvus()` 将文档向量写入 Milvus：

1. 读取 embedding 模型配置（`generation_config.embedding_config`）
2. 确定 **多 collection 同步目标**（`_get_sync_targets()`）：
   - 基座模型 → `tf_sync_{task_id[:8]}_{model_hash}`
   - 每个已加载 adapter → `tf_sync_{task_id[:8]}_{adapter_hash}`
3. 对每个目标 collection，调用 `pre_index_all_chunks()` 将文档 embed 并写入 Milvus
4. 注册 collection 到 `milvus_collection_service`，记录 `embedding_config_id` 和 `embedding_model`
5. 全部成功后返回 `ready_for_generation=True`，失败则保持 REGISTERED 待下轮重试

#### Adapter 历史数据回填

当部署新加载 adapter 或重新加载已卸载的 adapter 时，`_load_all_historical_documents()` 从所有历史批次加载文档（按 `doc_id` 去重），确保新 adapter collection 拥有完整的向量索引。已存在的向量通过 Milvus upsert 跳过，仅索引新增数据。

### Phase 2: 数据生成（Level 1）

- 当 `pending_record_count >= generation_threshold` 时触发
- 合并所有 `fetched` 状态的批次为一个数据集
- 创建数据生成任务（`generation_mode` 决定生成策略）
- 生成完成后累加 `pending_training_samples`

### Phase 3: 模型训练（Level 2）

- 当 `pending_training_samples >= training_threshold` 时触发
- 收集所有已完成生成的训练数据集
- 创建 LoRA 训练任务，训练完成后触发 adapter 热加载
- 训练失败时在下一周期自动重试（重新检查阈值）

### 阈值检查

`_check_thresholds()` 在每轮 `run_once` 结束时统一执行，即使本轮无新数据也会检查（处理上轮生成/训练失败的重试场景）。

## 三层去重机制

防止 API 返回重复或回退数据：

| 层级 | 机制 | 说明 |
|------|------|------|
| 边界键去重 | `last_sync_boundary_ids` | 组合键 `id:session_id:doc_id`，去除上次同步尾部的重叠记录 |
| 时间过滤 | `_item_is_after()` | 验证每条记录的 `created_at` 在查询窗口之后，防止 API 忽略 since 参数。统一转换为 naive UTC 比较，兼容各种时区偏移格式 |
| 批次窗口去重 | `create_batch()` | 同一时间窗口内的重复记录合并，返回 `(batch, is_new)` 元组 |

## 训练配置

### Loss 函数支持

`training_config` 支持以下 loss 相关字段（按 `model_type` 区分）：

| model_type | 字段 | 示例 |
|------------|------|------|
| embedding | `embedding_loss_name` | `"MultipleNegativesRankingLoss"` |
| reranker | `reranker_loss_name` | `"CrossEntropyLoss"` |
| decoder_reranker | `loss_config` | `{"name": "infonce", "n_docs": 8}` |

### 训练方法限制

同步流水线仅支持 **SFT**（Supervised Fine-Tuning）。RL 方法（DPO/GRPO/DAPO/ORPO）需要 `rl_config` 和 `sft_checkpoint_path`，sync level2_handler 不处理这些参数。

### 训练未配置处理

当 `training_config` 为空或缺少 `base_model_path`（且无 `base_deployment_id`）时，level2_handler 记录警告并保持 idle 状态，不会报错中断同步周期。

## 部署自动发现

当未指定 `base_deployment_id` 时，post_training_handler 自动搜索兼容的部署：

1. 查询所有 `running` 状态的部署
2. 匹配 `enable_lora=true` 且使用 vLLM/SGLang 框架
3. 比对部署的基座模型路径与训练使用的模型路径
4. 找到后自动加载 adapter，未找到则跳过加载（记录日志）

## Adapter 生命周期

### 自动加载流程

训练完成后的自动化流程：

1. **卸载旧 adapter**: 如果存在已加载的 adapter，全部从部署卸载，训练记录状态更新为 `adapter_unloaded`
2. **加载新 adapter**: 将新训练的 LoRA adapter 加载到部署
3. **更新状态**: 记录 `current_adapter_name`、`current_adapter_id`、`total_trainings`，训练记录状态更新为 `adapter_loaded`

### 训练记录状态流转

```
pending → completed → adapter_loaded → adapter_unloaded
   │          │              │
   ▼          ▼              ▼
 failed   adapter_load_failed  (被新 adapter 替换或手动卸载)
              │
              ▼
        (可通过 API 重试加载)
```

- `adapter_load_failed`: Adapter 加载失败，与训练本身失败（`failed`）区分。前端显示重试按钮
- `adapter_unloaded`: 加载新 adapter 时，所有旧的 `adapter_loaded` 记录批量更新为此状态

### 手动操作

| 操作 | API | 说明 |
|------|-----|------|
| 重试加载 | `POST /tasks/{id}/trainings/{tid}/retry-adapter-load` | 对 `adapter_load_failed` 等状态的训练记录重试 adapter 加载 |
| 卸载 adapter | `POST /tasks/{id}/unload-adapter` | 从部署卸载当前 adapter，清除 sync 任务的 adapter 信息 |
| 重算计数器 | `POST /tasks/{id}/recalculate-counters` | 以已完成生成记录为准重算训练样本计数，修复异常漂移 |

### 并发控制

使用 per-config 线程锁（`_get_config_lock`）防止同一 sync 任务上的并发 adapter 加载/卸载操作冲突（如自动回调与手动重试同时触发）。

## 生成数据停用与重新生成

已完成的生成数据可以被停用（`disabled=true`），停用后：
1. 关联的批次状态自动重置为 `fetched`
2. `pending_record_count` 恢复对应的记录数
3. 停用的生成数据不再参与后续训练合并

当所有生成数据均被停用时，手动触发生成会自动重新生成全部数据。这可用于更换生成配置（如切换 `neg_detection_mode`）后重新生成。

## 核心组件

```
train_factory/sync/
├── sync_worker.py              # Worker 主循环（三阶段同步）
├── level2_handler.py           # Level 2 训练任务创建
├── post_training_handler.py    # 训练后处理（adapter 加载/卸载/重试）
├── external_api_client.py      # 外部 API 客户端（认证 + 分页）
└── __init__.py

train_factory/enums/
└── sync_status.py              # 状态常量（SyncStatus, BatchStatus, SyncTrainingStatus 等）

train_factory/api/routes/
└── sync_routes.py              # API 路由

train_factory/storage/
├── entities/external_sync_entity.py  # ORM 实体（4 张表）
└── services/external_sync_service.py # 业务服务（含状态机校验）
```

## 相关文档

- 数据库表: [external_sync_tasks](../database/external_sync_tasks.md), [external_sync_batches](../database/external_sync_batches.md), [external_sync_generations](../database/external_sync_generations.md), [external_sync_trainings](../database/external_sync_trainings.md)
- API 文档: [数据同步 API](../api/sync.md)
- 前端页面: [SyncConfigCreate / SyncConfigList / SyncConfigDetail](../architecture/frontend.md)
