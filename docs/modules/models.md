# 模型注册模块

## 概述

管理训练产出和外部导入的模型，支持版本管理、来源追踪。

## 模型状态

| 状态 | 说明 |
|------|------|
| `registered` | 模型元数据已注册，文件未就绪（下载中或等待下载） |
| `available` | 模型文件已就绪，可直接用于部署或训练 |
| `archived` | 模型已归档，不再用于部署和训练 |

下载过程通过独立字段 (`download_status` / `download_progress` / `download_error`) 跟踪，不影响模型主状态。下载失败时保持 `registered`，下载成功自动变为 `available`。

## 模型来源

| 来源 | 说明 |
|------|------|
| `trained` | 训练任务产出 |
| `downloaded` | 从 HuggingFace/ModelScope 下载 |
| `uploaded` | 用户手动上传 |
| `external_bind` | 绑定外部路径 |

## 模型详情弹窗

前端模型列表页提供 Modal 形式的模型详情视图，展示模型完整信息：

- **入口**: 点击模型卡片，或通过 URL 参数 `?detail=model_id` 直接打开
- **内容**: Model ID（可复制）、类型、状态、来源、版本、基础模型、模型路径、文件大小、训练任务链接、创建时间、描述
- **基础模型解析**: 自动将 `base_model_path` 解析为注册模型名称并显示为可点击链接，点击可在同一弹窗内查看基础模型详情。使用路径后缀匹配兼容容器内 (`/app/models/`) 与宿主机 (`/data/.../models/`) 的路径差异
- **跨页面跳转**: 部署列表和训练详情页的模型名称链接均跳转到 `/models?detail=model_id`

## 相关文档

- 数据库表: [model_registry](../database/model_registry.md), [model_versions](../database/model_versions.md)
- API 文档: [模型 API](../api/models.md)
