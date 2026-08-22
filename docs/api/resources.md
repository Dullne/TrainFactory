# 资源管理 API

## 获取资源状态

```http
GET /api/resources/status
```

获取系统资源（GPU、CPU、内存、磁盘）的详细状态。

**响应**:
```json
{
  "gpu": {
    "total_gpus": 2,
    "available_gpus": [0, 1],
    "gpu_memory": {
      "0": {"total": 24576, "used": 8192, "free": 16384},
      "1": {"total": 24576, "used": 0, "free": 24576}
    }
  },
  "system": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "memory_used_mb": 16384,
    "memory_total_mb": 32768,
    "disk_usage_percent": 60.0,
    "disk_used_gb": 500,
    "disk_total_gb": 1000,
    "open_files": 128,
    "thread_count": 64
  }
}
```

## 获取 GPU 列表

```http
GET /api/resources/gpus
```

获取简化的 GPU 信息列表。

**响应**:
```json
{
  "total": 2,
  "gpus": [
    {
      "gpu_id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "used_mb": 8192,
      "total_mb": 24576,
      "free_mb": 16384,
      "utilization": 0.33
    },
    {
      "gpu_id": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "used_mb": 0,
      "total_mb": 24576,
      "free_mb": 24576,
      "utilization": 0.0
    }
  ]
}
```

## 获取 GPU 进程

```http
GET /api/resources/gpus/processes
```

获取按 GPU 分组的进程占用信息，包括 Docker 容器识别。

**响应**:
```json
{
  "total": 2,
  "gpus": [
    {
      "gpu_index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "uuid": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "total_memory_mb": 24576,
      "processes": [
        {
          "pid": 12345,
          "name": "python",
          "cmdline": "python -m vllm.entrypoints.openai.api_server ...",
          "user": "root",
          "gpu_memory_mb": 8192,
          "container_id": "abc123def456",
          "container_name": "vllm-qwen3-reranker-0.6b-xxxx",
          "started_at": "2025-01-29T10:30:00"
        }
      ]
    },
    {
      "gpu_index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "uuid": "GPU-yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy",
      "total_memory_mb": 24576,
      "processes": []
    }
  ]
}
```

**进程字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `pid` | int | 进程 ID |
| `name` | string | 进程名称 |
| `cmdline` | string | 完整命令行 |
| `user` | string | 运行用户 |
| `gpu_memory_mb` | int | 占用显存 (MB) |
| `container_id` | string | Docker 容器 ID（如适用） |
| `container_name` | string | Docker 容器名称（如适用） |
| `started_at` | string | 进程启动时间 (ISO 8601) |

## 清理资源

```http
POST /api/resources/cleanup
```

执行资源清理：垃圾回收、清理过期 GPU 分配。

**响应**:
```json
{
  "gc_collected": 1234,
  "memory_freed_mb": 256,
  "stale_gpu_allocations_cleaned": 2
}
```
