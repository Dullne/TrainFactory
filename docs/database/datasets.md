# datasets - 数据集

存储数据集元信息。

| 字段 | 类型 | 说明 |
|------|------|------|
| `dataset_id` | VARCHAR(36) | 数据集 UUID |
| `dataset_name` | VARCHAR(255) | 数据集名称 |
| `dataset_type` | VARCHAR(50) | 类型: embedding_pair, rerank_pair, sft_instruct, dpo_preference |
| `usage` | VARCHAR(20) | 用途 (可选): raw, train, eval, test |
| `model_type` | JSON | 适用模型标签（可选，可多选）: ["embedding"], ["embedding", "rerank"]，用户自行标注 |
| `source_type` | VARCHAR(32) | 来源: uploaded, huggingface, modelscope, local, generated |
| `source_dataset_id` | VARCHAR(36) | 源数据集 ID（生成数据集的来源） |
| `storage_path` | VARCHAR(1024) | 存储路径 |
| `file_format` | VARCHAR(32) | 文件格式: parquet, jsonl, csv |
| `columns` | JSON | 列定义 |
| `num_rows` | INT | 总行数 |
| `status` | VARCHAR(50) | 状态: registered, uploading, downloading, processing, ready, archived, error |

## 状态生命周期

| 状态 | 说明 |
|------|------|
| `registered` | 元数据已注册，文件未就绪（初始状态） |
| `uploading` | 文件上传中 |
| `downloading` | 从远程仓库下载中 |
| `processing` | 数据处理中 |
| `ready` | 数据就绪，可用于训练/生成 |
| `archived` | 已归档 |
| `error` | 出错 |

> 通过 API 注册时默认为 `registered`，上传完成后自动变为 `ready`。预览操作会将 `registered` 状态的数据集自动提升为 `ready`。

## 相关文档

- 模块设计: [数据集管理](../modules/datasets.md)
- API 文档: [数据集 API](../api/datasets.md)
