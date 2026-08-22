# TrainFactory API 参考文档

## 基础信息

- **Base URL**: `http://localhost:18000/api`
- **Content-Type**: `application/json`

## 健康检查

```http
GET /health
```

**响应**:
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

## API 模块

| 模块 | 说明 | 文档 |
|------|------|------|
| 训练管理 | 创建、监控训练任务 | [training.md](training.md) |
| 模型注册 | 模型版本管理 | [models.md](models.md) |
| 部署管理 | 模型部署到推理服务 | [deployments.md](deployments.md) |
| 模型配置 | API 端点配置管理 | [configs.md](configs.md) |
| 数据集管理 | 数据集注册与下载 | [datasets.md](datasets.md) |
| 评估管理 | 模型评估任务、深度评估 | [evaluations.md](evaluations.md) |
| 数据生成 | 数据生成任务与格式 | [generation.md](generation.md) |
| 向量库管理 | Milvus 集合管理与数据集关联 | [milvus.md](milvus.md) |
| 外部数据同步 | 增量同步、两级阈值触发、批次与生成/训练追踪 | [sync.md](sync.md) |
| 资源管理 | GPU/CPU 监控、进程查看 | [resources.md](resources.md) |
| 枚举参考 | 状态码与枚举值 | [reference.md](reference.md) |

## 认证

开发模式下无需认证。生产环境使用 JWT Token：

```http
Authorization: Bearer <token>
```

或通过 httpOnly Cookie 自动携带。
