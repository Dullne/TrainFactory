# API 端点概览

各模块 REST API 端点速查，详细文档见 [API 参考](../api/README.md)。

## 训练管理 `/api/train`

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/train` | 创建训练任务 |
| GET | `/train` | 列出训练任务 |
| GET | `/train/{task_id}` | 获取任务状态 |
| POST | `/train/{task_id}/stop` | 停止任务 |
| DELETE | `/train/{task_id}` | 删除任务 |

## 模型注册 `/api/models`

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/models` | 注册模型 |
| GET | `/models` | 列出模型 |
| GET | `/models/{model_id}` | 获取模型详情 |
| PUT | `/models/{model_id}` | 更新模型 |
| DELETE | `/models/{model_id}` | 删除模型 |
| POST | `/models/{model_id}/versions` | 添加版本 |
| POST | `/models/from-task/{task_id}` | 从任务注册 |

## 部署管理 `/api/deployments`

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/deployments` | 创建部署 |
| GET | `/deployments` | 列出部署 |
| GET | `/deployments/{id}` | 获取部署详情 |
| POST | `/deployments/{id}/start` | 启动部署 |
| POST | `/deployments/{id}/stop` | 停止部署 |
| POST | `/deployments/{id}/restart` | 重启部署 |
| POST | `/deployments/{id}/sync` | 同步状态 |
| DELETE | `/deployments/{id}` | 删除部署 |

## Adapter 管理 `/api/deployments/...`

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/deployments/{id}/adapters` | 加载 adapter |
| POST | `/deployments/{id}/adapters/from-task` | 从训练任务加载 adapter |
| GET | `/deployments/{id}/adapters` | 列出已加载 adapter |
| DELETE | `/deployments/{id}/adapters/{name}` | 卸载 adapter |
| POST | `/deployments/{id}/adapters/sync` | 同步 adapter 状态 |
| GET | `/adapters/available` | 查询可用 adapter |
| GET | `/adapters/{adapter_id}` | 获取 adapter 详情 |

## 模型配置 `/api/configs`

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/configs` | 创建配置 |
| GET | `/configs` | 列出配置 |
| GET | `/configs/{config_id}` | 获取配置 |
| PUT | `/configs/{config_id}` | 更新配置 |
| DELETE | `/configs/{config_id}` | 删除配置 |
| POST | `/configs/validate` | 验证配置 |
| POST | `/configs/{id}/check` | 检查连通性 |
| POST | `/configs/{id}/test` | API 测试代理 |

## 数据集管理 `/api/datasets`

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/datasets` | 创建数据集 |
| GET | `/datasets` | 列出数据集 |
| GET | `/datasets/{id}` | 获取数据集详情 |
| PUT | `/datasets/{id}` | 更新数据集 |
| DELETE | `/datasets/{id}` | 删除数据集 |
| GET | `/datasets/{id}/preview` | 预览数据集 |
| GET | `/datasets/types` | 获取支持的类型 |
| POST | `/datasets/download` | 下载远程数据集 |

## 评估管理 `/api/evaluations`

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/evaluations/tasks` | 创建评估任务 |
| GET | `/evaluations/tasks` | 列出评估任务 |
| GET | `/evaluations/tasks/{id}` | 获取评估详情 |
| POST | `/evaluations/tasks/{id}/cancel` | 取消评估任务 |
| POST | `/evaluations/tasks/{id}/resume` | 恢复评估任务 |
| DELETE | `/evaluations/tasks/{id}` | 删除评估任务 |

## 资源管理 `/api/resources`

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/resources/status` | 获取系统资源状态 |
| GET | `/resources/gpus` | 获取 GPU 列表 |
| GET | `/resources/gpus/processes` | 获取 GPU 进程占用 |
| POST | `/resources/cleanup` | 清理系统资源 |

## 相关文档

- [API 详细文档](../api/README.md)
- [架构首页](README.md)
