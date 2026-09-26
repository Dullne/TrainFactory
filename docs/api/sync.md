# 外部数据同步 API

管理外部 API 增量数据同步，支持两级阈值自动触发数据生成和模型训练。

## 数据流

```
外部 API ──since──→ 增量拉取 ──batch──→ 累积计数
                                          │
                                count >= gen_threshold?
                                          │ Yes
                                          ▼
                                    数据生成任务
                                          │
                                          ▼
                                  训练数据累积计数
                                          │
                              samples >= train_threshold?
                                          │ Yes
                                          ▼
                                    模型训练任务
                                          │
                                          ▼
                                  LoRA Adapter 热加载
```

## 三层去重机制

| 层级 | 机制 | 说明 |
|------|------|------|
| 边界键 | `last_sync_boundary_ids` | 组合键 `id:session_id:doc_id`，去除上次同步尾部的重叠记录 |
| 时间过滤 | `_item_is_after()` | 验证记录 `created_at` 在查询窗口之后，防止 API 忽略 since 参数返回旧数据 |
| 批次窗口 | `create_batch()` | 同一时间窗口内的重复记录合并，返回 `(batch, is_new)` 元组 |

- 每次同步回退 5 秒时间窗口，确保不遗漏同时间戳的记录
- 无论外部 API 返回 `>` 还是 `>=` 语义的数据，均可正确处理

---

## 任务管理

### 创建同步任务

```http
POST /api/sync/tasks
```

**请求体**:

```json
{
  "task_name": "业务系统同步",
  "external_api_url": "http://business-system.example.com:9003",
  "external_auth_config": {
    "username": "admin",
    "password": "secret"
  },
  "sync_interval_seconds": 300,
  "generation_threshold": 500,
  "generation_mode": "doc_to_training",
  "generation_config": {
    "llm_config": { "config_id": "uuid-llm" },
    "embedding_config": { "config_id": "uuid-emb" },
    "rerank_config": { "config_id": "uuid-rerank" }
  },
  "training_threshold": 1000,
  "training_config": {
    "model_type": "llm",
    "training_method": "sft",
    "lora_r": 16,
    "lora_alpha": 32
  },
  "base_deployment_id": "uuid-deployment",
  "is_active": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_name` | string | 是 | 任务名称 |
| `external_api_url` | string | 是 | 外部 API 地址 |
| `external_auth_config` | object | 是 | 认证配置，支持 `{username, password}` 或 `{token}` |
| `sync_interval_seconds` | int | 否 | 同步间隔（秒），默认 300，最小 10 |
| `generation_threshold` | int | 否 | Level 1 生成阈值，默认 500 |
| `generation_mode` | string | 否 | 生成模式: `doc_to_training`, `qa_to_training` |
| `generation_config` | object | 否 | 生成配置（LLM/Embedding/Rerank 模型配置引用） |
| `training_threshold` | int | 否 | Level 2 训练阈值，默认 1000 |
| `training_config` | object | 否 | 训练配置（见下方详细说明） |
| `base_deployment_id` | string | 否 | 部署 ID（用于 LoRA 热加载）；留空时自动发现兼容部署 |
| `is_active` | bool | 否 | 是否立即启用，默认 true |

**`training_config` 详细说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_type` | string | 模型类型: embedding, reranker, decoder_reranker, llm |
| `training_method` | string | 训练方法（仅支持 `sft`，RL 方法在 sync 流水线中不可用） |
| `base_model_path` | string | 基座模型路径（如未指定 `base_deployment_id` 则必填） |
| `lora_r` | int | LoRA rank |
| `lora_alpha` | int | LoRA alpha |
| `num_train_epochs` | int | 训练轮数 |
| `per_device_train_batch_size` | int | 批次大小 |
| `learning_rate` | float | 学习率 |
| `gpu_ids` | list | GPU 编号列表 |
| `embedding_loss_name` | string | Embedding Loss 名称（model_type=embedding 时） |
| `reranker_loss_name` | string | Reranker Loss 名称（model_type=reranker 时） |
| `loss_config` | object | Decoder-Reranker Loss 配置（如 `{"name": "infonce", "n_docs": 8}`） |
| `bf16` / `fp16` | bool | 混合精度训练 |

> **训练方法限制**: 同步流水线的 level2_handler 仅处理 SFT 训练。DPO/GRPO/DAPO/ORPO 等 RL 方法需要 `rl_config` 和 `sft_checkpoint_path`，不在 sync 流程中处理。

> **部署自动发现**: 当 `base_deployment_id` 为空时，训练完成后系统自动搜索 `running` 状态、`enable_lora=true`、使用 vLLM/SGLang 框架且基座模型匹配的部署，自动加载 adapter。未找到兼容部署时跳过加载。

**响应**:

```json
{
  "message": "Sync task created",
  "task": {
    "task_id": "uuid",
    "task_name": "业务系统同步",
    "status": "idle",
    "pending_record_count": 0,
    "...": "..."
  }
}
```

---

### 列出同步任务

```http
GET /api/sync/tasks
```

返回当前用户的所有同步任务。

**响应**:

```json
{
  "tasks": [...],
  "total": 2
}
```

---

### 获取同步任务详情

```http
GET /api/sync/tasks/{task_id}
```

**响应**:

```json
{
  "task": {
    "task_id": "uuid",
    "task_name": "业务系统同步",
    "external_api_url": "http://...",
    "external_auth_config": { "username": "admin", "password": "***" },
    "sync_interval_seconds": 300,
    "last_sync_at": "2026-02-12T10:30:00",
    "generation_threshold": 500,
    "training_threshold": 1000,
    "pending_record_count": 123,
    "pending_training_samples": 456,
    "total_record_count": 5000,
    "total_training_samples": 3200,
    "total_trainings": 2,
    "current_adapter_name": "sync-user1234-r2",
    "status": "idle",
    "is_active": true
  }
}
```

> 注意：`external_auth_config` 中的敏感字段（password、token、secret、api_key）会被脱敏为 `"***"`。

---

### 更新同步任务

```http
PATCH /api/sync/tasks/{task_id}
```

仅需传入要更新的字段。更新同步间隔、API 地址或认证配置时会自动重启 worker。

**请求体**:

```json
{
  "sync_interval_seconds": 600,
  "generation_threshold": 1000
}
```

---

### 删除同步任务

```http
DELETE /api/sync/tasks/{task_id}
```

删除任务并停止关联的后台 worker。

---

## Worker 控制

### 启动同步

```http
POST /api/sync/tasks/{task_id}/start
```

激活任务并启动后台轮询 worker。

**响应**:

```json
{
  "message": "Sync started",
  "worker_status": "running"
}
```

---

### 停止同步

```http
POST /api/sync/tasks/{task_id}/stop
```

停止后台 worker 并将任务设为非活跃。

---

### 立即同步

```http
POST /api/sync/tasks/{task_id}/sync-now
```

手动触发一次同步周期，不等待轮询间隔。

数据源抓取或记录校验失败时返回 `502`，响应为
`{"detail":"Sync source fetch or validation failed"}`，不会返回同步成功。
失败不会推进本轮数据游标；已有批次仍可按阈值进入生成流程。成功重试会清除旧的同步错误信息。

---

### 手动触发生成

```http
POST /api/sync/tasks/{task_id}/trigger-generation
```

忽略 Level 1 阈值，立即合并待处理批次并创建数据生成任务。

---

### 手动触发训练

```http
POST /api/sync/tasks/{task_id}/trigger-training
```

忽略 Level 2 阈值，立即合并所有已完成生成的训练数据并创建训练任务。

---

## 历史查询

### 批次列表

```http
GET /api/sync/tasks/{task_id}/batches?status=fetched&limit=50&offset=0
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `status` | string | 过滤状态: `fetched`, `registered`, `generation_queued`, `generation_done` |
| `limit` | int | 每页数量，默认 50 |
| `offset` | int | 偏移量，默认 0 |

**响应**:

```json
{
  "batches": [
    {
      "batch_id": "uuid",
      "task_id": "uuid",
      "record_count": 150,
      "storage_path": "/app/data/sync/.../batch_20260212_103000_a1b2c3.jsonl",
      "since_time": "2026-02-12T10:25:00",
      "until_time": "2026-02-12T10:29:55",
      "fetched_at": "2026-02-12T10:30:00",
      "status": "fetched",
      "generation_task_id": null
    }
  ],
  "total": 10
}
```

---

### 生成任务列表

```http
GET /api/sync/tasks/{task_id}/generations?limit=50&offset=0
```

**响应**:

```json
{
  "generations": [
    {
      "task_id": "uuid",
      "generation_task_id": "uuid",
      "input_batch_ids": ["batch-1", "batch-2"],
      "input_record_count": 500,
      "output_dataset_id": "uuid",
      "output_sample_count": 1200,
      "status": "completed",
      "disabled": false,
      "created_at": "2026-02-12T11:00:00",
      "completed_at": "2026-02-12T11:45:00"
    }
  ],
  "total": 3
}
```

---

### 训练任务列表

```http
GET /api/sync/tasks/{task_id}/trainings?limit=50&offset=0
```

**响应**:

```json
{
  "trainings": [
    {
      "task_id": "uuid",
      "training_task_id": "uuid",
      "input_dataset_ids": ["ds-1", "ds-2", "ds-3"],
      "total_samples": 3200,
      "training_round": 2,
      "output_adapter_path": "/app/output/sync-r2/final",
      "loaded_adapter_name": "sync-user1234-r2",
      "status": "adapter_loaded",
      "created_at": "2026-02-12T12:00:00",
      "completed_at": "2026-02-12T13:30:00"
    }
  ],
  "total": 2
}
```

---

### 停用/启用生成数据

```http
PATCH /api/sync/tasks/{task_id}/generations/{generation_task_id}/disabled
```

停用已完成的生成数据，使其不再参与后续训练合并。停用后关联批次自动重置为 `fetched`，恢复 `pending_record_count`，可重新触发生成。

**请求体**:

```json
{
  "disabled": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `disabled` | bool | 是 | `true` 停用，`false` 启用 |

**响应**:

```json
{
  "id": 38,
  "task_id": "uuid",
  "generation_task_id": "uuid",
  "status": "completed",
  "disabled": true,
  "reset_batch_count": 1,
  "reset_record_count": 1292
}
```

> 仅 `completed` 状态的生成记录可以停用/启用。停用时 `reset_batch_count` 和 `reset_record_count` 表示被重置的批次数和记录数。

---

### 重算训练样本计数器

```http
POST /api/sync/tasks/{task_id}/recalculate-counters
```

基于 `external_sync_generations` 表中的真实完成记录，重算：

- `total_training_samples`
- `pending_training_samples`

用于修复因中断/异常导致的计数漂移。

**响应**:

```json
{
  "old_total_training_samples": 3200,
  "new_total_training_samples": 3150,
  "old_pending_training_samples": 900,
  "new_pending_training_samples": 850
}
```

---

### 综合状态

```http
GET /api/sync/tasks/{task_id}/status
```

返回配置的综合状态，包含 worker 状态和各计数器。

**响应**:

```json
{
  "task_id": "uuid",
  "status": "idle",
  "is_active": true,
  "worker_status": "running",
  "pending_record_count": 123,
  "generation_threshold": 500,
  "pending_training_samples": 456,
  "training_threshold": 1000,
  "total_record_count": 5000,
  "total_training_samples": 3200,
  "total_trainings": 2,
  "current_adapter_name": "sync-user1234-r2",
  "last_sync_at": "2026-02-12T10:30:00"
}
```

---

## Adapter 管理

### 重试加载 Adapter

```http
POST /api/sync/tasks/{task_id}/trainings/{training_task_id}/retry-adapter-load?replace=true
```

对 adapter 加载失败的训练记录重试加载。后台异步执行，立即返回。

| 参数 | 类型 | 说明 |
|------|------|------|
| `replace` | bool | 是否先卸载所有已加载的 adapter（默认 `true`） |

**允许的训练状态**: `adapter_load_failed`, `adapter_failed`, `completed`, `adapter_unloaded`, `adapter_loaded`

**响应**:

```json
{
  "message": "Adapter loading retry started"
}
```

---

### 卸载当前 Adapter

```http
POST /api/sync/tasks/{task_id}/unload-adapter
```

从同步任务关联的部署中卸载当前 adapter，清除 `current_adapter_name` 和 `current_adapter_id`，并将所有 `adapter_loaded` 状态的训练记录更新为 `adapter_unloaded`。

**响应**:

```json
{
  "message": "Adapter unloaded"
}
```


---

## 认证模式

外部 API 支持两种认证方式：

### 用户名密码

```json
{
  "external_auth_config": {
    "username": "admin",
    "password": "secret123"
  }
}
```

客户端会自动 `POST /api/auth/login` 获取 `bm_auth` cookie，遇到 401 时自动重新登录。

### Token 直连

```json
{
  "external_auth_config": {
    "token": "bm_auth=eyJhbGciOi..."
  }
}
```

直接使用提供的 token，token 过期时会报错（不支持自动续期）。

---

## 状态枚举

### 同步任务状态 (`status`)

| 状态 | 说明 |
|------|------|
| `idle` | 空闲，等待下一次同步 |
| `syncing` | 正在拉取外部数据 |
| `generating` | 正在执行数据生成 |
| `training` | 正在执行模型训练 |
| `loading_adapter` | 正在加载 LoRA adapter |
| `error` | 出错（详见 `error_message`） |

### 批次状态 (`status`)

| 状态 | 说明 |
|------|------|
| `fetched` | 已拉取，等待处理 |
| `registered` | 已注册为数据集 |
| `generation_queued` | 已加入生成队列 |
| `generation_done` | 生成完成 |

### 生成追踪状态 (`status`)

| 状态 | 说明 |
|------|------|
| `pending` | 等待执行 |
| `completed` | 生成完成 |
| `failed` | 生成失败 |

### 训练追踪状态 (`status`)

| 状态 | 说明 |
|------|------|
| `pending` | 等待执行 |
| `completed` | 训练完成，等待 adapter 加载 |
| `failed` | 训练失败 |
| `adapter_loaded` | Adapter 已成功加载到部署 |
| `adapter_unloaded` | Adapter 已卸载（被更新的 adapter 替换或手动卸载） |
| `adapter_load_failed` | Adapter 加载失败（可通过 retry-adapter-load 重试） |
| `adapter_failed` | Adapter 操作失败（通用错误状态） |
