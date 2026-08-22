# 数据集管理 API

## 创建数据集

```http
POST /api/datasets
```

**请求体**:
```json
{
  "dataset_name": "my-embedding-dataset",
  "storage_path": "/app/data/datasets/my-dataset",
  "dataset_type": "embedding_pair",
  "usage": "train",
  "model_type": "embedding",
  "source_type": "uploaded",
  "file_format": "parquet",
  "columns": [
    {"name": "anchor", "type": "string"},
    {"name": "positive", "type": "string"}
  ],
  "num_rows": 10000,
  "tags": ["embedding", "chinese"]
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dataset_name` | string | 是 | - | 数据集名称 |
| `storage_path` | string | 是 | - | 本地存储路径 |
| `dataset_type` | string | 否 | custom | 数据集类型 |
| `usage` | string | 否 | null | 用途: raw (源文档), train (训练), eval (评估), test (测试) |
| `model_type` | string | 否 | null | 适用模型类型: embedding, rerank, llm |
| `source_type` | string | 否 | uploaded | 来源: uploaded, huggingface, modelscope, local |
| `file_format` | string | 否 | parquet | 文件格式: parquet, jsonl, csv, arrow |
| `columns` | array | 否 | null | 列结构定义 |
| `num_rows` | int | 否 | null | 数据行数 |
| `tags` | array | 否 | null | 标签列表 |

### 数据集类型

| 类型 | 说明 | 列结构 |
|------|------|--------|
| `embedding_pair` | 句子对 | anchor, positive |
| `embedding_triplet` | 三元组 | anchor, positive, negative |
| `rerank_pair` | 重排序对 | query, document, label |
| `rerank_listwise` | 列表排序 | query, documents, labels |
| `sft_instruct` | SFT 指令 | instruction, response |
| `dpo_preference` | DPO 偏好 | prompt, chosen, rejected |
| `rl_reward` | RL 奖励 | prompt, response, reward |
| `custom` | 自定义 | user-defined |

## 上传数据集

```http
POST /api/datasets/upload
```

通过文件上传创建数据集。使用 `multipart/form-data` 格式。

**表单参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 数据集文件（支持 .jsonl, .parquet, .csv, .arrow） |
| `display_name` | string | 否 | 显示名称 |
| `dataset_type` | string | 否 | 数据集类型 |
| `usage` | string | 否 | 用途: raw, train, eval, test |
| `model_type` | string | 否 | 适用模型类型（逗号分隔多选，如 `embedding,rerank`） |
| `tags` | string | 否 | 标签（逗号分隔） |

**状态流转**: `registered` → `uploading` → `ready`（上传完成自动转为 ready）

---

## 列出数据集

```http
GET /api/datasets?dataset_type=embedding_pair&usage=train&limit=10
```

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset_type` | string | 按数据集类型过滤 |
| `usage` | string | 按用途过滤: raw, train, eval, test |
| `model_type` | string | 按模型类型过滤 |
| `source_type` | string | 按来源过滤 |
| `status` | string | 按状态过滤 |
| `limit` | int | 返回数量限制 |
| `offset` | int | 偏移量 |

## 获取数据集详情

```http
GET /api/datasets/{dataset_id}
```

## 更新数据集

```http
PUT /api/datasets/{dataset_id}
```

## 删除数据集

```http
DELETE /api/datasets/{dataset_id}
```

## 获取数据集类型

```http
GET /api/datasets/types
```

返回所有支持的数据集类型及其说明。

## 预览数据集

```http
GET /api/datasets/{dataset_id}/preview?limit=10
```

**响应**:
```json
{
  "dataset_id": "xxx",
  "columns": [
    {"name": "anchor", "type": "string"},
    {"name": "positive", "type": "string"}
  ],
  "rows": [
    {"anchor": "Hello", "positive": "Hi there"},
    {"anchor": "How are you?", "positive": "How do you do?"}
  ],
  "total_rows": 10000
}
```

## 导出数据集（可读格式）

```http
POST /api/datasets/{dataset_id}/export
```

**请求体**:
```json
{
  "format": "jsonl",
  "limit": 10000,
  "expires_seconds": 3600
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `format` | string | 否 | jsonl | 导出格式：`original` / `jsonl` / `csv` |
| `limit` | int | 否 | null | 限制导出行数，不传表示导出全部 |
| `expires_seconds` | int | 否 | 3600 | 下载链接有效期（秒） |

**响应**:
```json
{
  "dataset_id": "xxx",
  "source_format": "parquet",
  "export_format": "jsonl",
  "row_count": 10000,
  "storage_uri": "s3://trainfactory/exports/u1/xxx/20260213T120000Z_abcd_dataset.jsonl",
  "download_url": "http://minio:9000/...",
  "proxy_download_url": "http://localhost:18000/api/datasets/xxx/export/download?storage_uri=s3%3A%2F%2F...",
  "expires_seconds": 3600
}
```

> 建议前端优先使用 `proxy_download_url`，避免 MinIO 内网地址在浏览器不可达的问题。

## 下载远程数据集

```http
POST /api/datasets/download
```

**请求体**:
```json
{
  "dataset_name": "all-nli",
  "remote_repo": "sentence-transformers/all-nli",
  "source_type": "huggingface",
  "hf_subset": "pair",
  "dataset_type": "embedding_pair",
  "usage": "train",
  "model_type": "embedding"
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dataset_name` | string | 是 | - | 数据集名称 |
| `remote_repo` | string | 是 | - | 远程仓库 (如 sentence-transformers/all-nli) |
| `source_type` | string | 否 | huggingface | 来源: huggingface, modelscope |
| `hf_subset` | string | 否 | null | HuggingFace 子集名称 |
| `dataset_type` | string | 否 | custom | 数据集类型 |
| `usage` | string | 否 | null | 用途: raw, train, eval, test |
| `model_type` | string | 否 | null | 适用模型类型 |

## 获取下载进度

```http
GET /api/datasets/download/{dataset_id}/progress
```

## 获取统计

```http
GET /api/datasets/stats
```

**响应**:
```json
{
  "total": 15,
  "by_type": {
    "embedding_pair": 8,
    "rerank_pair": 5,
    "sft_instruct": 2
  },
  "by_source": {
    "uploaded": 10,
    "huggingface": 5
  },
  "by_status": {
    "ready": 14,
    "processing": 1
  },
  "total_size_bytes": 1073741824,
  "total_rows": 500000
}
```
