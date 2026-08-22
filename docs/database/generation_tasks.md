# generation_tasks - 数据生成任务

存储 LLM 数据生成任务（从文档生成训练数据）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | VARCHAR(36) | 任务 UUID |
| `task_name` | VARCHAR(255) | 任务名称 |
| `description` | TEXT | 任务描述 |
| `input_path` | VARCHAR(1024) | 输入文件/目录路径 |
| `input_format` | VARCHAR(32) | 输入格式: auto, jsonl, json, txt |
| `content_field` | VARCHAR(64) | 内容字段名 |
| `generation_mode` | VARCHAR(32) | 生成模式: doc_to_training, qa_to_training, qa_extraction |
| `pos_neg_method` | VARCHAR(32) | 正负例生成方式: retrieval, llm |
| `output_path` | VARCHAR(1024) | 输出文件路径 |
| `output_format` | VARCHAR(32) | 输出格式: universal, triplet, pair |
| `llm_config` | JSON | LLM 配置 (端点、模型、参数) |
| `eval_llm_config` | JSON | 评估 LLM 配置 (可选，用于 chunk 评估) |
| `embedding_config` | JSON | Embedding 配置 (可选，用于相似度过滤和向量检索) |
| `rerank_config` | JSON | Rerank 配置 (可选，用于负例筛选) |
| `worker_config` | JSON | 工作配置 (并发数、超时) |
| `steps_config` | JSON | 步骤配置 (启用的生成步骤) |
| `post_process_config` | JSON | 输出配置（去重开关等） |
| `custom_prompts` | JSON | 自定义提示词 |
| `status` | VARCHAR(50) | 状态: pending, running, completed, failed, stopped |
| `progress` | FLOAT | 进度 (0-100) |
| `total_docs` | INT | 总文档数 |
| `processed_docs` | INT | 已处理文档数 |
| `output_sample_count` | INT | 输出样本数 |
| `output_dataset_id` | VARCHAR(36) | 输出数据集 ID |
| `auto_register_dataset` | BOOLEAN | 是否自动注册为数据集 |
| `source_dataset_id` | VARCHAR(36) | 源数据集 ID（记录输入数据集） |
| `embedding_config_id` | VARCHAR(36) | Embedding 配置 ID |
| `milvus_collection` | VARCHAR(255) | Milvus 集合名称 |
| `similarity_threshold` | FLOAT | 相似度过滤阈值 (默认 0.85) |
| `retrieval_top_k` | INT | 向量检索 Top-K (默认 10) |
| `filter_stats` | JSON | 过滤统计信息 |
| `qa_output_path` | VARCHAR(1024) | QA 中间产物输出路径 |
| `qa_dataset_id` | VARCHAR(36) | QA 中间产物数据集 ID |
| `qa_filtered_path` | VARCHAR(1024) | 过滤后 QA 输出路径 |
| `qa_filtered_dataset_id` | VARCHAR(36) | 过滤后 QA 数据集 ID |
| `error_message` | TEXT | 错误信息 |

## 相关文档

- 模块设计: [数据生成](../modules/generation.md)
- API 文档: [数据生成 API](../api/generation.md)
