# 枚举值与错误参考

## 错误响应

所有 API 在出错时返回统一的错误格式：

```json
{
  "detail": "Error message describing what went wrong"
}
```

## HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未认证 |
| 403 | 无权限 |
| 404 | 资源不存在 |
| 409 | 资源冲突 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

## 枚举值

### 训练状态 (TrainingStatus)

| 值 | 说明 |
|----|------|
| `pending` | 等待执行 |
| `running` | 正在运行 |
| `succeeded` | 成功完成 |
| `failed` | 执行失败 |
| `stopped` | 手动停止 |

### 模型类型 (ModelType)

| 值 | 说明 | 训练支持 | 注册支持 |
|----|------|----------|----------|
| `embedding` | Embedding 模型 | ✓ | ✓ |
| `reranker` | CrossEncoder 重排序 | ✓ | ✓ |
| `decoder_reranker` | Decoder 重排序 | ✓ | ✓ |
| `llm` | 大语言模型 | - | ✓ |

### 训练方法 (TrainingMethod)

| 值 | 说明 | 适用模型类型 |
|----|------|-------------|
| `sft` | 监督微调 | 全部 |
| `dpo` | 直接偏好优化 | decoder_reranker |
| `grpo` | 群体相对策略优化 | decoder_reranker |
| `reinforce` | REINFORCE 算法 | decoder_reranker |
| `two_stage` | 两阶段训练 | decoder_reranker |

### 部署状态 (DeploymentStatus)

| 值 | 说明 |
|----|------|
| `pending` | 等待部署 |
| `deploying` | 正在部署 |
| `starting` | 正在启动 |
| `running` | 运行中 |
| `restarting` | 正在重启 |
| `stopping` | 正在停止 |
| `stopped` | 已停止 |
| `failed` | 部署失败 |

### 数据集用途 (DatasetUsage)

| 值 | 说明 |
|----|------|
| `train` | 训练数据集 |
| `eval` | 评估数据集 |

### 推理框架 (InferenceFramework)

| 值 | 说明 | 部署模式 |
|----|------|----------|
| `xinference` | Xinference 服务 | 外部服务 |
| `vllm` | vLLM | 容器部署 |
| `sglang` | SGLang | 容器部署 |

### 评估任务状态

| 值 | 说明 |
|----|------|
| `pending` | 等待执行 |
| `running` | 正在运行 |
| `succeeded` | 成功完成 |
| `failed` | 执行失败 |
| `cancelled` | 已取消 |
