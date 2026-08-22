# 数据生成 API

数据生成用于从文档/文本构建训练数据（QA、正负例等），支持并发 LLM 调用与可选的 Embedding/Rerank 打分。

## 生成模式

| 模式 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `doc_to_training` | 文档 → 训练数据 | 原始文档 | 训练数据 (含正负例) |
| `qa_to_training` | QA → 训练数据 | QA 数据集 | 训练数据 (含正负例) |
| `qa_extraction` | QA 提取 | 原始文档 | QA 对 |
| `doc_to_eval` | 文档 → 评估数据 | 原始文档 | 评估数据集 |
| `qa_to_eval` | QA → 评估数据 | QA 数据集 | 评估数据集 |

配置 Embedding 时自动启用相似度筛选、Milvus 入库和向量检索正负例：
- 相似度 **> threshold** 的 QA 对被过滤掉（等于 threshold 的保留）
- `similarity_threshold` 设为 **1.0** 时跳过相似度过滤，仅计算 chunk 向量入库 Milvus

| 模式 | 有 Embedding | 无 Embedding |
|------|-------------|-------------|
| `qa_to_training` | Phase 0 预索引 → 筛选+向量检索正负例 | 纯 LLM 生成正负例 |
| `doc_to_training` | Phase 0 预索引 → QA提取 ‖ 筛选+向量检索正负例 | QA提取 ‖ LLM 生成正负例 |
| `doc_to_eval` | QA提取 → 评估数据集 | QA提取 → 评估数据集 |
| `qa_to_eval` | 评估数据集 | 评估数据集 |

> **Phase 0 预索引**：配置 Embedding 时，pipeline 先执行 `pre_index_all_chunks()` 将所有文档 chunk 一次性向量化并插入 Milvus，后续流式处理直接从内存缓存 (`_chunk_emb_cache`) 读取向量，无需重复计算。

### 正负例生成方式 (`pos_neg_method`)

| 值 | 说明 | 要求 |
|----|------|------|
| `retrieval` (默认) | 向量检索候选 chunk → ContextualPrecisionMetric 评估分类正负例 → LLM 补充不足部分 | 必须配置 Embedding |
| `llm` | 纯 LLM 生成正负例 | 无额外要求 |

> `pos_neg_method` 仅对 `doc_to_training` 和 `qa_to_training` 模式有效。Embedding 配置后无论 `pos_neg_method` 为何值，都会自动启用相似度筛选和 Milvus 入库。

### 正负例检测模式 (`neg_detection_mode`)

| 值 | 正例 | 负例 | 说明 |
|----|------|------|------|
| `chunk` (默认) | 源 chunk + 检索正例 chunk | 检索负例 chunk | 粒度为 chunk |
| `statement` | answer + 从正例 chunk 提取的支持性语句 | 从负例 chunk 提取的迷惑性语句 | 粒度为 statement，需额外 LLM 调用 |

**相关参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `supplement_positives` | `true` | chunk 模式：是否将检索到的正例 chunk 加入 positives；statement 模式：是否从正例 chunk 提取支持性语句作为补充正例 |
| `confirm_positives` | `false` | statement 模式下对提取的正例语句进行二次 LLM 确认，过滤幻觉和无关语句（需 `supplement_positives=true`） |
| `confirm_negatives` | `false` | statement 模式下使用 Delta 计算 + LLM 确认筛选难负例语句（需配置 Rerank 或 Embedding） |
| `answer_rewrite` | `false` | statement 模式下将 query + answer 用 LLM 合并为完整陈述句作为首正例 |
| `rerank_score_classification` | `false` | statement 模式下给提取的语句打分，按正例分数区间筛选（需配置 Rerank 或 Embedding） |
| `evidence_removal` | `false` | 从正例 chunk 中去除证据生成硬负例变体 |
| `evidence_pruning` | `false` | 裁剪正例 chunk 中无关内容生成更干净正例 |
| `chunk_eval_mode` | `batch` | Chunk 评估模式：`batch`（一次调用返回索引，快但有位置偏差）、`individual`（每 chunk 单独调用，无偏差最准，N 倍成本） |
| `neg_chunk_scoring` | `false` | 给负例 chunk 打分，只保留分数高于最差正例的难负例（需配置 Rerank 或 Embedding） |
| `skip_easy_negatives` | `true` | 是否跳过排在所有正例之后的简单负例（模型已能区分，训练价值低） |
| `skip_perfect_ap` | `true` | 是否跳过 AP=1.0 的记录（完美排序，排序能力已足够） |
| `skip_zero_ap` | `true` | 是否跳过 AP=0.0 的记录（完全错误排序，可能数据质量问题） |
| `use_role` | `false` | 是否为每个 QA 对生成差异化角色，角色会影响正负例的风格和视角（仅 llm 模式） |
| `roles_per_doc` | `3` | 每个 QA 对生成的角色数量（1-10，需 `use_role=true`） |
| `augment` | `false` | retrieval 模式下是否用 LLM 补充不足的正负例 |

### 处理流水线

QA 去重（dedup）在 QA 生成之后、相似度过滤之前执行，避免对重复 query 做无用的 embedding 计算：

```
doc_to_training (有 Embedding):
  Phase 0: pre_index_all_chunks (chunk 向量化 + Milvus 入库)
  Phase 1: QA 提取 → QA 去重 → on_qa_complete 回调（提前注册 QA 数据集）
  Phase 2+3: 相似度过滤 + 正负例生成 (流式，与 Phase 1 并行)

doc_to_training (无 Embedding):
  Phase 1: QA 提取 → QA 去重 → on_qa_complete 回调
  Phase 2+3: LLM 正负例生成 (流式)

qa_to_training:   Phase 0 → QA 去重 → 相似度过滤 → 正负例生成
qa_extraction:    文档 → QA 提取 → QA 去重

doc_to_eval:      文档 → QA 提取 → QA 去重 → 评估数据集
qa_to_eval:       QA 数据集 → 评估数据集
```

> **on_qa_complete 回调**：Phase 1 完成时立即注册 QA 数据集，用户无需等待整个 pipeline 结束即可预览 QA 数据。

## 创建生成任务

```http
POST /api/generation/tasks
```

**请求体 (qa_to_training)**:
```json
{
  "task_name": "train-gen-001",
  "generation_mode": "qa_to_training",
  "pos_neg_method": "retrieval",
  "dataset_id": "qa-dataset-id",
  "output_format": "universal",
  "llm_config": {
    "config_id": "llm-config-id",
    "temperature": 0.7,
    "max_tokens": 2048
  },
  "eval_llm_config": {
    "config_id": "eval-llm-config-id"
  },
  "embedding_config": {
    "config_id": "embed-config-id",
    "similarity_threshold": 0.85,
    "retrieval_top_k": 10
  },
  "steps": {
    "pos_neg_extraction": {
      "enabled": true,
      "num_positive": 5,
      "num_negative": 20,
      "chunk_eval_mode": "batch",
      "neg_detection_mode": "chunk",
      "supplement_positives": true,
      "skip_easy_negatives": true,
      "skip_perfect_ap": true,
      "skip_zero_ap": true
    }
  },
  "auto_register_dataset": true
}
```

**请求体 (doc_to_training)**:
```json
{
  "task_name": "train-gen-002",
  "generation_mode": "doc_to_training",
  "pos_neg_method": "retrieval",
  "dataset_id": "raw-doc-dataset-id",
  "output_format": "universal",
  "llm_config": {
    "config_id": "llm-config-id"
  },
  "embedding_config": {
    "config_id": "embed-config-id",
    "similarity_threshold": 0.85,
    "retrieval_top_k": 10
  },
  "steps": {
    "qa_gen": { "enabled": true, "num_qa_per_doc": 3 },
    "pos_neg_extraction": {
      "enabled": true,
      "num_positive": 5,
      "num_negative": 20,
      "neg_detection_mode": "statement",
      "supplement_positives": true,
      "skip_easy_negatives": true
    }
  },
  "post_process": {
    "dedup": { "enabled": true }
  },
  "auto_register_dataset": true
}
```

**请求体 (qa_extraction)**:
```json
{
  "task_name": "qa-extract-001",
  "generation_mode": "qa_extraction",
  "dataset_id": "raw-doc-dataset-id",
  "llm_config": {
    "config_id": "llm-config-id"
  },
  "steps": {
    "doc_quality": { "enabled": true, "min_score": 0.6 },
    "qa_gen": { "enabled": true, "num_qa_per_doc": 3 }
  },
  "post_process": {
    "dedup": { "enabled": true }
  },
  "auto_register_dataset": true
}
```

**说明**:
- `llm_config` 支持 `config_id` 或 `endpoint + model` 的直连配置
- `pos_neg_method` 可选值 `retrieval`（默认）或 `llm`，`retrieval` 需要配置 `embedding_config`
- `embedding_config` 可选，配置后自动启用相似度筛选和 Milvus 入库（无论 `pos_neg_method` 为何值）
- `embedding_config.similarity_threshold` 设为 `1.0` 时跳过相似度过滤，仅入库 Milvus
- `post_process.dedup.enabled` 启用 QA 去重（基于 query 文本哈希，在相似度过滤之前执行）
- `dataset_id` 指定数据集输入（`qa_to_training` 需 qa_pair 类型，`doc_to_training` 需原始文档类型），也可用 `input_path` 直接指定路径

### 使用已有向量集合

通过 `milvus_collection_name` 参数可以选择已有的 Milvus 集合，而不是自动创建新集合：

```json
{
  "task_name": "reuse-collection-task",
  "generation_mode": "qa_to_training",
  "dataset_id": "new-dataset-id",
  "milvus_collection_name": "tf_022fa895_1d1780cb",
  "llm_config": { "config_id": "llm-config-id" },
  "auto_register_dataset": true
}
```

**行为**:
- 系统验证集合已在注册表中注册
- 自动从注册表获取绑定的 Embedding 模型配置（无需再指定 `embedding_config`）
- 新数据追加到已有集合中，支持增量构建
- 任务完成后自动关联数据集到该集合

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `milvus_collection_name` | string | 否 | 使用已有集合名称。不指定则自动创建 |

## 查询任务列表

```http
GET /api/generation/tasks
```

**响应字段补充**：

- `created_at`: 任务创建时间（ISO 8601 字符串）

## 获取任务详情

```http
GET /api/generation/tasks/{task_id}
```

## 获取任务进度

```http
GET /api/generation/tasks/{task_id}/progress
```

## 停止任务

```http
POST /api/generation/tasks/{task_id}/stop
```

## 重新运行任务

```http
POST /api/generation/tasks/{task_id}/restart?force=false
```

用于重启 `failed` / `stopped` / `completed` 任务。

- `force=false`（默认）：断点续传，优先复用检查点（如 QA 中间产物）
- `force=true`：从头重跑，忽略已有检查点

**响应**:

```json
{
  "status": "restarted",
  "task_id": "uuid"
}
```

## 删除任务

```http
DELETE /api/generation/tasks/{task_id}
```

## 获取输出格式

```http
GET /api/generation/formats
```

**响应**:
```json
{
  "output_formats": [
    { "name": "universal", "description": "通用格式，包含 query, positives, negatives" },
    { "name": "triplet", "description": "三元组格式 (query, positive, negative)" },
    { "name": "pair", "description": "对格式 (query, text, label)" }
  ],
  "source_types": ["document", "qa", "passage"],
  "length_types": ["short", "medium", "long"]
}
```
