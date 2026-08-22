# 向量库模块

## 概述

基于 Milvus 的向量库管理模块，支持集合创建、数据导入、向量搜索，用于 RAG 场景评估和数据检索。

## 核心功能

| 功能 | 说明 |
|------|------|
| 集合管理 | 创建、删除 Milvus 集合 |
| 数据导入 | 从数据集批量导入向量 |
| 向量搜索 | 支持语义搜索和混合搜索 |
| 集合浏览 | 浏览集合内数据 |

## 核心组件

```
train_factory/api/milvus_routes.py         # API 路由
train_factory/storage/milvus_*.py          # Entity + Service
```

## 相关文档

- 数据库表: [milvus_collections](../database/milvus_collections.md), [collection_dataset_links](../database/collection_dataset_links.md)
- API 文档: [向量库 API](../api/milvus.md)
