# 数据生成模块

## 概述

数据生成模块通过 LLM 从文档/QA 数据自动生成训练数据，支持多种生成模式和完整的正负例构建流水线。

## 生成模式

| 模式 | 说明 | 输入 |
|------|------|------|
| `doc_to_training` | 文档 → 训练数据 | 文档文件 |
| `qa_to_training` | QA → 训练数据 | QA 数据集 |
| `qa_extraction` | 文档 → QA 数据 | 文档文件 |
| `doc_to_eval` | 文档 → 评估数据 | 文档文件 |
| `qa_to_eval` | QA → 评估数据 | QA 数据集 |

## 处理流水线

```
doc_to_training:
  文档 → 文档质量评估 → QA 提取 → QA 质量过滤 → QA 去重 → 相似度过滤 → 正负例生成

qa_to_training:
  QA 数据集 → QA 去重 → 相似度过滤 → 正负例生成

qa_extraction:
  文档 → 文档质量评估 → QA 提取 → QA 质量过滤 → QA 去重
```

### QA 质量过滤

QA 提取后自动过滤不适合独立检索的低质量 query：
- **文档引用检测**: 以"该文档"、"本文"、"上述"等开头的 query（脱离原文后无意义）
- **代词依赖检测**: 短 query（< 30 字符）中代词密度过高（如"它的架构是什么？"）

### 正负例生成方式

| `pos_neg_method` | 说明 | 要求 |
|-------------------|------|------|
| `retrieval`（默认） | 向量检索候选 chunk → ContextualPrecision 评估分类 | 必须配置 Embedding |
| `llm` | 纯 LLM 生成正负例 | 无额外要求 |

### 正负例检测模式 (`neg_detection_mode`)

| 值 | 正例 | 负例 | 说明 |
|----|------|------|------|
| `chunk`（默认） | 源 chunk + 检索正例 chunk | 排序错误的 chunk（按位置分 very_hard/hard/medium） | 粗粒度，直接使用完整 chunk |
| `statement` | answer + 从正例 chunk 提取的陈述句 | 从负例 chunk 提取的迷惑性语句 | 细粒度，需额外 LLM 调用 |

## 增量写入与实时预览

生成过程中结果增量追加到输出文件，支持：
- 任务运行中即可预览已生成的样本
- 断点续传（基于 checkpoint 跳过已处理文档）

## 核心组件

```
train_factory/generation/
├── pipeline.py                    # 生成流水线（步骤编排、增量写入）
└── steps/
    ├── doc_quality.py             # 文档质量评估
    ├── keypoint_gen.py            # 关键点生成
    ├── qa_gen.py                  # QA 生成（含质量过滤）
    ├── qa_dedup.py                # QA 去重
    ├── validation.py              # 数据验证
    ├── embedding_filter_step.py   # Embedding 相似度过滤 + Milvus 入库
    ├── hard_neg_extraction.py     # 难负例语句提取（statement 模式）
    └── pos_neg.py                 # 正负例生成
```

## 相关文档

- 数据库表: [generation_tasks](../database/generation_tasks.md)
- API 文档: [数据生成 API](../api/generation.md)
