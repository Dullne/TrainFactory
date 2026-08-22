# 资源监控模块

## 概述

监控系统资源使用情况，包括 GPU 显存占用、CPU/内存使用、磁盘空间等，为训练任务调度提供资源信息。

## 核心功能

| 功能 | 说明 |
|------|------|
| GPU 监控 | 显存使用率、GPU 利用率、温度 |
| 系统信息 | CPU、内存、磁盘使用情况 |
| 实时刷新 | 前端定时轮询更新 |

## 核心组件

```
train_factory/api/resource_routes.py       # API 路由
train_factory/monitoring/                  # 监控核心
```

## 相关文档

- API 文档: [资源监控 API](../api/resources.md)
