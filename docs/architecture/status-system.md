# 状态系统统一方案

本文档说明 TrainFactory 当前的状态设计原则、各实体状态的语义边界，以及前后端统一展示规范。

目标不是把所有实体强行塞进一套枚举，而是统一“状态维度”和“展示规则”，避免同名状态混义或同义状态散落。

## 1. 设计原则

- 不同实体保留各自状态机，避免为了“统一”破坏业务流程
- 统一的是语义维度，而不是字段名本身
- 后端继续维护原始 `status`，前端基于共享映射做一致展示
- 活跃态、成功态、失败态、非活跃态在 UI 上保持稳定颜色和交互语义

## 2. 四个语义维度

### 2.1 生命周期状态 `lifecycle_status`

描述任务或运行实例正处于哪个阶段。

典型值：

- `pending`
- `preparing`
- `running`
- `evaluating`
- `stopping`
- `stopped`
- `completed`
- `failed`

适用对象：

- 训练任务
- 生成任务
- 部署任务
- 同步编排任务

### 2.2 可用性状态 `availability_status`

描述资源能否被消费。

典型值：

- `available`
- `unavailable`
- `archived`

适用对象：

- 模型注册
- 数据集
- 向量库集合

### 2.3 健康状态 `health_status`

描述外部服务或运行实例是否健康。

典型值：

- `healthy`
- `unhealthy`
- `unknown`

适用对象：

- 部署
- 外部模型配置连通性结果

### 2.4 管理状态 `admin_status`

描述配置是否启用。

典型值：

- `active`
- `inactive`
- `disabled`

适用对象：

- 外部 API 配置
- 模型配置
- 部分同步目标配置

## 3. 当前实体映射

### 3.1 训练任务

原始状态：

- `pending`
- `preparing`
- `running`
- `evaluating`
- `succeeded`
- `failed`
- `stopped`
- `cancelled`

归属：

- 主维度：`lifecycle_status`

说明：

- `preparing`：GPU 分配、数据加载、模型初始化
- `running`：实际训练过程
- `evaluating`：训练后测试集评估与结果汇总

### 3.2 数据集

原始状态：

- `registered`
- `uploading`
- `downloading`
- `processing`
- `ready`
- `error`
- `archived`

归属：

- 主维度：`availability_status`
- 次维度：`lifecycle_status`（上传/下载/处理阶段）

说明：

- `ready` 表示可消费
- `archived` 表示保留但默认不参与训练/生成

### 3.3 模型注册

原始状态：

- `registered`
- `available`
- `archived`

归属：

- 主维度：`availability_status`

### 3.4 部署

原始状态：

- `pending`
- `starting`
- `running`
- `restarting`
- `stopping`
- `stopped`
- `failed`

附加健康状态：

- `HEALTHY`
- `UNHEALTHY`
- `UNKNOWN`

归属：

- 主维度：`lifecycle_status`
- 次维度：`health_status`

### 3.5 模型配置 / 外部 API 配置

原始状态：

- `active`
- `inactive`
- `error`（部分配置对象）

归属：

- 主维度：`admin_status`
- 连通性结果不直接等价于 lifecycle 状态

## 4. 前端展示规范

前端通过共享状态元数据统一渲染，不直接在页面里散写颜色判断。

共享入口：

- `web/src/utils/status.ts`
- `web/src/components/StatusTag.tsx`

### 4.1 颜色语义

- `queued`：黄色，表示等待中但未真正执行
- `in_progress`：蓝色，表示当前正在推进
- `succeeded`：绿色，表示任务完成或资源可用
- `failed`：红色，表示失败或不可用
- `inactive`：灰色/默认色，表示停用、归档、空闲

### 4.2 动画规则

仅对真正推进中的状态显示脉冲点动画：

- `preparing`
- `running`
- `restarting`
- `evaluating`
- `training`
- `deploying`
- `syncing`
- `generating`
- `loading_adapter`

`pending` 不显示动画，避免把排队态误判成执行中。

### 4.3 Tooltip 规则

状态标签默认展示补充说明，用于解释中间态：

- `preparing`：资源准备阶段
- `evaluating`：训练已结束，正在汇总评估指标
- `generating`：同步任务已触发生成，等待生成链路完成

## 5. 当前已落地项

- 后端训练任务已真正落地 `preparing -> running -> evaluating -> succeeded|failed|stopped`
- 服务重启时会清理处于 `pending/preparing/running/evaluating` 的孤儿训练任务
- 前端训练页已区分 `preparing` 和 `evaluating`，不再把它们都粗暴等价成 `running`
- 共享 `StatusTag` 已支持统一颜色、动画与状态说明

## 6. 后续建议

### 6.1 短期

- 继续让更多页面改用共享状态元数据，而不是本地 `statusColorMap`
- 为部署页增加 `lifecycle_status + health_status` 双标签展示

### 6.2 中期

- 对数据集、模型、部署分别增加前端派生字段：
  - `semantic`
  - `description`
  - `isActive`
- 避免每个页面重复判断“是否可停止 / 是否可删除 / 是否可轮询”

### 6.3 长期

- 如果后端后续要开放公共 API 给外部系统，可考虑新增统一派生字段：
  - `display_status`
  - `status_semantic`
  - `status_description`

这样前端与第三方调用方都不需要重新编码状态映射规则。
