# 数据库文档

## 概述

TrainFactory 使用 MySQL 8.0 作为持久化存储，采用 SQLModel (Pydantic + SQLAlchemy) 作为 ORM。

## 配置

### 环境变量

数据库密码使用 URL 安全的十六进制字符，并为 root 与 API 用户生成不同的随机值。

```bash
# 数据库连接 URL
MYSQL_URL=mysql+pymysql://trainfactory_app:${MYSQL_APP_PASSWORD}@mysql:3306/train_factory

# 自动迁移（默认开启）
DB_AUTO_MIGRATE=true
```

### Docker Compose

```yaml
mysql:
  image: mysql:8.0
  environment:
    - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:?MYSQL_ROOT_PASSWORD is required}
    - MYSQL_DATABASE=train_factory
    - MYSQL_USER=${MYSQL_APP_USER:-trainfactory_app}
    - MYSQL_PASSWORD=${MYSQL_APP_PASSWORD:?MYSQL_APP_PASSWORD is required}
```

## 数据库表

### 表概览

| 表名 | 说明 | 主键 |
|------|------|------|
| [`training_tasks`](training_tasks.md) | 训练任务 | task_id |
| [`model_registry`](model_registry.md) | 模型注册 | model_id |
| [`model_versions`](model_versions.md) | 模型版本 | version_id |
| [`deployments`](deployments.md) | 部署记录 | deployment_id |
| [`model_configs`](model_configs.md) | 模型配置 | config_id |
| [`datasets`](datasets.md) | 数据集 | dataset_id |
| [`evaluation_tasks`](evaluation_tasks.md) | 评估任务（支持 MTEB 和 DeepEval） | task_id |
| [`generation_tasks`](generation_tasks.md) | 数据生成任务 | task_id |
| [`milvus_collections`](milvus_collections.md) | 向量集合注册表 | collection_id |
| [`collection_dataset_links`](collection_dataset_links.md) | 集合与数据集关联（多对多） | id |
| [`loaded_adapters`](loaded_adapters.md) | 已加载的 LoRA Adapter | adapter_id |
| [`external_sync_tasks`](external_sync_tasks.md) | 外部数据同步任务 | task_id |
| [`external_sync_batches`](external_sync_batches.md) | 同步数据批次 | batch_id |
| [`external_sync_generations`](external_sync_generations.md) | 同步生成任务追踪 | generation_task_id |
| [`external_sync_trainings`](external_sync_trainings.md) | 同步训练任务追踪 | training_task_id |

---

### Alembic 迁移

项目使用 Alembic 管理数据库迁移，迁移文件位于：

```
train_factory/storage/migrations/
├── env.py                    # Alembic 环境配置
└── versions/
    ├── 001_add_missing_fields.py
    ├── 002_add_lora_hotload_support.py
    ├── 003_add_evaluation_tasks.py
    ├── 004_add_dataset_model_type.py
    ├── 005_add_model_progress.py
    ├── 006_rename_deploy_mode_shared.py
    ├── 006_add_path_hashes.py
    ├── 007_dataset_model_type_to_json.py
    ├── 008_remove_legacy_fields.py
    ├── 009_task_read_models.py
    ├── 010_add_deep_evaluation_tasks.py
    ├── 011_add_model_config_container_fields.py
    ├── 012_add_generation_tasks.py        # 数据生成任务表
    ├── 012_add_deep_eval_fields.py        # DeepEval 字段扩展
    ├── 013_merge_heads.py                 # 合并分支迁移
    ├── 014_merge_evaluation_tasks.py      # 合并 deep_evaluation_tasks 到 evaluation_tasks
    ├── 015_add_eval_worker_groups.py      # 添加 worker_groups 字段
    ├── 016_drop_training_task_stats.py
    ├── 017_add_generation_content_field.py # 添加 content_field
    ├── 018_add_two_phase_generation_fields.py # 两阶段生成字段
    ├── 019_add_full_pipeline_fields.py    # 完整流水线字段
    ├── 020_simplify_generation_modes.py   # 生成模式简化 + pos_neg_method
    ├── 021_add_eval_llm_config.py         # 评估 LLM 配置
    ├── 022_add_source_dataset_id.py       # 统一 source_dataset_id
    ├── 023_add_deep_eval_fields.py        # 深度评估数据集字段
    ├── 024_add_collection_registry.py     # 向量集合注册表 + 数据集关联表
    └── 025_add_external_sync.py          # 外部数据同步 (4 张表)
```

### 自动迁移

API 启动时会自动执行迁移（可通过 `DB_AUTO_MIGRATE=false` 禁用）：

```python
# train_factory/api/server.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    run_migrations()  # 自动执行迁移
    yield
```

### 手动迁移

```bash
# 进入容器
docker exec -it train-factory-api-dev bash

# 创建新迁移
cd /app/train_factory/storage
alembic revision -m "add_new_feature"

# 执行迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

### 创建迁移示例

```python
# versions/006_add_new_column.py
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '005'

def upgrade():
    op.add_column('training_tasks',
        sa.Column('new_field', sa.String(255), nullable=True)
    )
    op.create_index('idx_new_field', 'training_tasks', ['new_field'])

def downgrade():
    op.drop_index('idx_new_field', 'training_tasks')
    op.drop_column('training_tasks', 'new_field')
```

---

## 初始化脚本

首次启动时，MySQL 容器会执行 `docker/init.sql` 创建所有表。

```bash
# 手动初始化
docker compose exec -T mysql sh -c 'exec mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE"' < docker/init.sql
```

---

## Entity 与 Service

### 目录结构

```
train_factory/storage/
├── database.py              # 数据库连接
├── entities/                # ORM 实体
│   ├── training_task_entity.py
│   ├── model_registry_entity.py
│   ├── deployment_entity.py
│   ├── loaded_adapter_entity.py        # LoRA Adapter 加载记录
│   ├── model_config_entity.py
│   ├── dataset_entity.py
│   ├── evaluation_task_entity.py     # 统一评估任务（MTEB + DeepEval）
│   ├── generation_task_entity.py     # 数据生成任务
│   ├── milvus_collection_entity.py   # 向量集合注册 + 数据集关联
│   └── external_sync_entity.py      # 外部数据同步 (4 张表)
└── services/                # 业务服务
    ├── training_task_service.py
    ├── model_registry_service.py
    ├── model_config_service.py
    ├── dataset_service.py
    ├── evaluation_task_service.py     # MTEB 评估服务
    ├── deep_evaluation_task_service.py # DeepEval 评估服务（使用统一 Entity）
    ├── generation_task_service.py     # 数据生成任务服务
    ├── milvus_collection_service.py   # 向量集合注册服务（CRUD + 关联管理）
    └── external_sync_service.py      # 外部同步服务（配置/批次/生成/训练追踪）
```

### 使用示例

```python
from train_factory.storage.services import training_task_service

# 创建任务
task = training_task_service.create_task(
    task_name="my-task",
    model_type="embedding",
    training_method="sft",
    ...
)

# 查询任务
task = training_task_service.get_task(task_id)

# 更新状态
training_task_service.update_status(task_id, "running", progress=50.0)

# 列出任务
tasks = training_task_service.list_tasks(status="running", limit=10)
```
