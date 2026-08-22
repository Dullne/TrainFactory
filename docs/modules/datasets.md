# 数据集模块

## 数据集类型

| 类型 | 描述 | 列结构 | 适用损失 |
|------|------|--------|----------|
| `embedding_pair` | 句子对 | anchor, positive | MNR, CosineSimilarity |
| `embedding_triplet` | 三元组 | anchor, positive, negative | TripletLoss, MNR |
| `rerank_pair` | 重排序对 | query, document, label | CrossEntropy, MarginMSE |
| `rerank_listwise` | 列表排序 | query, documents, labels | LambdaLoss, ListMLE |
| `sft_instruct` | SFT 指令 | instruction, response | CrossEntropy |
| `dpo_preference` | DPO 偏好 | prompt, chosen, rejected | DPOLoss |
| `rl_reward` | RL 奖励 | prompt, response, reward | REINFORCE, GRPO |

## 存储后端抽象层

数据集文件存储通过 `StorageBackend` 抽象层管理，支持本地文件系统和 S3 两种后端：

| 后端 | 类 | 适用场景 |
|------|-----|---------|
| `local` | `LocalStorageBackend` | 单机开发、文件挂载环境 |
| `s3` | `S3StorageBackend` | 分布式部署、MinIO/AWS S3 |

通过 `get_storage_backend()` 工厂函数获取单例实例，由环境变量 `STORAGE_BACKEND` 控制切换。

核心接口：

```python
class StorageBackend(ABC):
    def exists(self, path: str) -> bool          # 文件是否存在
    def delete(self, path: str) -> None           # 删除文件
    def preview(self, path: str, ...) -> dict     # 预览数据集内容
    def export_source_path(self, path: str) -> str # 导出可访问路径
```

## 数据集资产 (Dataset Asset)

每个数据集可关联多个文件资产（`DatasetAsset`），用于管理数据集的组成文件：

| 字段 | 说明 |
|------|------|
| `dataset_id` | 所属数据集 |
| `asset_type` | 资产类型（source_file, generated 等） |
| `file_path` | 文件存储路径 |
| `file_size` | 文件大小 |

## 数据集血缘 (Dataset Lineage)

通过 `DatasetLineage` 记录数据集的来源关系，支持追溯生成链路：

```
原始文档 → QA 提取 → QA 数据集 → 正负例生成 → 训练数据集
           ↑                                      ↑
       generation_task                     generation_task
```

| 字段 | 说明 |
|------|------|
| `source_dataset_id` | 源数据集 |
| `target_dataset_id` | 目标数据集 |
| `task_id` | 生成该关系的任务 ID |
| `task_type` | 任务类型（generation, training 等） |

## 相关文档

- 数据库表: [datasets](../database/datasets.md)
- API 文档: [数据集 API](../api/datasets.md)
- 存储后端: `train_factory/storage/backends/`
