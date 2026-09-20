# 部署管理 API

## 创建部署

```http
POST /api/deployments
```

**请求体**:
```json
{
  "model_id": "550e8400-e29b-41d4-a716-446655440000",
  "xinference_endpoint": "http://xinference:9997",
  "deployment_name": "my-embedding-service",
  "inference_framework": "vllm",
  "replica": 1,
  "gpu_memory_utilization": 0.9,
  "enable_lora": false,
  "max_loras": 4,
  "max_lora_rank": 64,
  "config": {
    "dtype": "auto",
    "enforce_eager": false,
    "attention_backend": "flashinfer"
  }
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model_id` | string | 是 | - | 模型 ID |
| `xinference_endpoint` | string | 否 | 默认端点 | 推理服务端点（Xinference 模式） |
| `deployment_name` | string | 否 | null | 部署名称 |
| `inference_framework` | string | 否 | xinference | 推理框架: xinference, vllm, sglang |
| `replica` | int | 否 | 1 | 副本数 (1-8) |
| `gpu_memory_utilization` | float | 否 | 自动计算 | GPU 显存利用率 (0.05-1.0) |
| `enable_lora` | bool | 否 | false | 启用 LoRA 热加载 |
| `max_loras` | int | 否 | 4 | 最大 LoRA 适配器数量 (1-16) |
| `max_lora_rank` | int | 否 | 64 | 最大 LoRA 秩 (8-256) |
| `config` | object | 否 | null | 高级配置 |

> **注意**: vLLM 和 SGLang 框架会自动使用容器模式部署（独立 Docker 容器），Xinference 使用共享容器模式（多模型共用一个容器）。

### 本机 GPU 准入与故障恢复

训练、独立推理容器和本机共享 Xinference 使用同一套 GPU 排他准入；不同工作负载不能同时预留同一张卡。单个部署内部显式配置的副本 GPU 复用仍按其拓扑配置处理。部署启动与创建会在启动容器前持久化占用；API 重启后会读取部署状态、操作令牌和训练进程证据，防止把仍可能使用中的 GPU 再次分配。

共享 Xinference 的本机身份包括 `xinference`、`${COMPOSE_PROJECT_NAME:-trainfactory}-xinference`、回环地址、`localhost`、`host.docker.internal`，以及 API 进程环境中的 `XINFERENCE_CONTAINER_NAME`。该变量本来用于指定本机共享容器名称；只有确认端点与 API 使用同一台 GPU 主机时，才将自定义本机服务名配置为此值。任意远程域名/IP 的 GPU 编号属于远端调度域，不会自动当成本机 GPU；自定义本机端点应使用上述明确的本机身份，不能依赖 DNS 猜测主机归属。本机共享服务未指定 GPU 时，启动准入会选取可用 GPU 并保存所选编号。

无法确认训练进程退出、部署停止失败或运行中的本机共享服务缺少实际 GPU 编号时，系统保留占用；设备编号无法确定时暂停所有新 GPU 分配。`failed` 状态本身不是释放依据。应先恢复进程检查/Docker/推理服务的可访问性，通过停止或删除部署、确认训练进程退出并重新执行启动清理来解除隔离，不要通过手工修改数据库状态绕过检查。CPU 训练不占用 GPU。

上述准入适用于单个 API 实例、单个 worker 管理同一 GPU 主机的部署方式；数据库中的令牌和状态用于恢复及 fencing，并不构成多 API 实例或跨主机的分布式 GPU 调度器。

### config 高级配置

| 参数 | 框架 | 说明 |
|------|------|------|
| `dtype` | 通用 | 数据类型: auto, float16, bfloat16 |
| `enforce_eager` | vLLM | 禁用 CUDA Graph，使用 PyTorch eager 模式（默认 false） |
| `attention_backend` | SGLang | Attention 后端: flashinfer, torch_native, fa3, triton |

## 容器部署

```http
POST /api/deployments/container
```

用于显式创建容器模式部署，可指定 GPU ID 和端口。

**额外参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `gpu_id` | int | GPU 设备 ID（不指定则自动选择） |
| `port` | int | 暴露端口（不指定则自动分配） |
| `auto_start` | bool | 创建后自动启动（默认 true） |

## 快速部署

```http
POST /api/deployments/from-model/{model_id}
```

**请求体**:
```json
{
  "xinference_endpoint": "http://xinference:9997",
  "auto_start": true
}
```

## 启动部署

```http
POST /api/deployments/{deployment_id}/start
```

## 停止部署

```http
POST /api/deployments/{deployment_id}/stop
```

## 重启部署

```http
POST /api/deployments/{deployment_id}/restart
```

重启部署以清理 GPU 显存缓存。不同框架有不同的重启行为：

- **vLLM/SGLang**（单模型容器）：重启容器
- **Xinference**（多模型共享）：可仅重载模型或重启容器

**请求体**（可选）:
```json
{
  "mode": "auto",
  "reset_gpu": false
}
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | auto | 重启模式 |
| `reset_gpu` | bool | false | 重启前重置 GPU（仅容器模式） |

**mode 重启模式**:

| 模式 | 说明 |
|------|------|
| `auto` | 自动选择最佳策略（推荐）：vLLM/SGLang → 重启容器；Xinference → 重载模型 |
| `model` | 仅重载模型（仅 Xinference），保留同实例中的其他模型 |
| `container` | 强制重启容器，清理所有显存但会影响容器内所有模型 |

## 同步状态

```http
POST /api/deployments/{deployment_id}/sync
```

同步部署状态与实际推理服务状态。

## 列出部署

```http
GET /api/deployments?status=running&limit=10
```

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_id` | string | 按模型过滤 |
| `status` | string | 按状态过滤 |
| `sync` | bool | 返回前同步状态 |
| `limit` | int | 返回数量限制 |
| `offset` | int | 偏移量 |

## 获取部署详情

```http
GET /api/deployments/{deployment_id}
```

## 删除部署

```http
DELETE /api/deployments/{deployment_id}?force=false
```

| 参数 | 说明 |
|------|------|
| `force` | 是否同时删除关联的模型配置 |

## GPU 信息

```http
GET /api/gpus
```

获取所有 GPU 的显存使用情况。

**响应**:
```json
{
  "gpus": [
    {
      "gpu_id": 0,
      "used_mb": 8192,
      "total_mb": 24576,
      "free_mb": 16384,
      "utilization": 0.33
    }
  ]
}
```

## 发现已运行模型

```http
GET /api/discover-models?endpoint=http://localhost:9997&framework=xinference
```

发现推理端点上已运行的模型。

## 绑定已有模型

```http
POST /api/bind-existing
```

将已运行的模型绑定为部署记录（不启动新模型）。

**请求体**:
```json
{
  "endpoint": "http://localhost:9997",
  "model_uid": "my-model",
  "model_type": "embedding",
  "inference_framework": "xinference"
}
```

---

## LoRA Adapter 管理

> 仅 vLLM 和 SGLang 框架支持 adapter 热加载。Xinference 不支持运行时加载 adapter。
> 部署必须在创建时设置 `enable_lora=true` 才能使用以下接口。

### 加载 Adapter

```http
POST /api/deployments/{deployment_id}/adapters
```

**请求体**:
```json
{
  "adapter_name": "my-lora-adapter",
  "adapter_path": "/path/to/adapter/weights",
  "source_task_id": "cc9de486-...",
  "source_model_id": null
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `adapter_name` | string | 是 | Adapter 唯一名称，推理请求的 `model` 字段使用此名称 |
| `adapter_path` | string | 是 | Adapter 权重文件路径 |
| `source_task_id` | string | 否 | 来源训练任务 ID |
| `source_model_id` | string | 否 | 来源模型注册 ID |

**前置条件**:
- 部署状态为 `running`
- 部署已启用 LoRA（`enable_lora=true`）
- 框架为 vLLM 或 SGLang
- 已加载的 adapter 数量未超过 `max_loras`

### 从训练任务加载 Adapter

```http
POST /api/deployments/{deployment_id}/adapters/from-task
```

**请求体**:
```json
{
  "task_id": "cc9de486-1f8a-4fe0-995b-f61637e6b77a",
  "adapter_name": "my-adapter"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | 是 | LoRA 训练任务 ID（须 `is_lora=true` 且 `status=succeeded`） |
| `adapter_name` | string | 否 | 自定义名称，不指定则自动生成 `task-{task_id[:8]}` |

### 卸载 Adapter

```http
DELETE /api/deployments/{deployment_id}/adapters/{adapter_name}
```

### 列出已加载 Adapter

```http
GET /api/deployments/{deployment_id}/adapters?include_unloaded=false
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include_unloaded` | bool | false | 是否包含已卸载/失败的 adapter |

**响应**:
```json
{
  "adapters": [
    {
      "adapter_id": "uuid",
      "deployment_id": "uuid",
      "adapter_name": "my-lora-adapter",
      "adapter_path": "/path/to/weights",
      "source_task_id": "uuid",
      "source_model_id": null,
      "status": "loaded",
      "error_message": null,
      "loaded_at": "2026-02-25T10:00:00",
      "unloaded_at": null
    }
  ]
}
```

### 同步 Adapter 状态

```http
POST /api/deployments/{deployment_id}/adapters/sync
```

将数据库中的 adapter 状态与推理服务的实际状态对齐。如果推理服务中某个 adapter 已不存在（如服务重启导致丢失），则标记为 `unloaded`。

### 查询可用 Adapter

```http
GET /api/adapters/available?base_model_id=xxx
```

搜索可加载的 adapter 来源：
- 已完成的 LoRA 训练任务（`is_lora=true`、`status=succeeded`、有输出路径）
- 模型注册表中标记为 adapter 的模型（`is_adapter=true`）

| 参数 | 类型 | 说明 |
|------|------|------|
| `base_model_id` | string | 按基座模型过滤 |

### 获取 Adapter 详情

```http
GET /api/adapters/{adapter_id}
```
