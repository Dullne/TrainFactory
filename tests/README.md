# Backend 测试说明

## 1) 轻量基线（无完整依赖时）

适用于离线/受限环境（缺少 `sqlmodel`、`python-jose`、`slowapi` 等）：

```bash
pytest -q \
  tests/test_sync_thresholds_unit.py \
  tests/test_sync_multitarget_unit.py \
  tests/test_generation_resume.py \
  tests/test_status_management.py \
  tests/test_pipeline_doc_id_resolution.py
```

以上用例覆盖：
- sync 多训练目标的迁移/分流核心逻辑
- sync 阈值检查（Level-1/Level-2/多目标调度）核心分支
- 生成任务重启与 checkpoint 逻辑
- 状态机转换校验
- 文档 ID 解析逻辑

## 2) 完整回归（推荐）

在安装完整依赖后执行：

```bash
pip install -e .[dev]
pytest -q tests
```

如需数据库/外部服务相关集成测试（例如 `test_sync_worker.py`、`test_sync_api.py`），请先启动对应服务并确保具备本地 socket/网络权限（MySQL、API、mock server 等）。

## 3) CI 基线（本地复现）

后端 smoke：

```bash
pytest -q \
  tests/test_sync_thresholds_unit.py \
  tests/test_sync_multitarget_unit.py \
  tests/test_generation_resume.py \
  tests/test_status_management.py \
  tests/test_pipeline_doc_id_resolution.py
```

前端质量门禁：

```bash
cd web
npm run lint
npm run build
```
