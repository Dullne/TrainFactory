# 本地后端测试

## Windows / PowerShell

使用项目的 `.venv`，避免全局 Anaconda 中的依赖参与测试。下面的安装命令使用 CPU PyTorch，不要求本机有 GPU；生产训练环境仍由 Docker 配置管理。

首次安装（在仓库根目录执行）：

```powershell
uv venv --python 3.11.13 .venv
uv pip install --python .venv/Scripts/python.exe --torch-backend cpu --constraints requirements/test-cpu.lock --overrides requirements/test-cpu-overrides.txt -e '.[dev]'
```

`test-cpu.lock` 是 Linux CI 的锁定依赖。这里将它作为 Windows 安装的版本约束，由安装器补充 Windows 专用依赖；不等同于 CI 的哈希锁定安装。

运行测试时，无须激活环境，也不依赖终端中 `python` 指向哪里：

```powershell
./scripts/run_tests.ps1 -q tests/test_loss_modules.py
./scripts/run_tests.ps1 -q tests/test_remaining_api_read_responsiveness.py
./scripts/run_tests.ps1 --collect-only -q
```

脚本会原样传递 pytest 参数，并返回 pytest 的退出码。编辑器中也应选择 `.venv/Scripts/python.exe`；若之前已选择过全局 Python，需要重新选择项目解释器。

若已有环境使用了不同的 PyTorch 版本，可重新执行上面的安装命令，然后检查：

```powershell
uv pip check --python .venv/Scripts/python.exe
.venv/Scripts/python.exe -I -c "import torch, alembic, qwen3_rerank_trainer; print(torch.__version__)"
```

## 完整验证范围

Linux 后端测试以 `.github/workflows/ci.yml` 中的 CPU 测试镜像为准。Windows 本地回归通过不代表 Linux CI、MySQL 集成测试或真实 GPU 训练已经通过。需要 Git/Docker 的 `host_tools` 测试和 MySQL 集成测试按 CI 中各自的步骤独立运行。

## 真实同步集成

这些测试会创建、更新和删除数据。准备独立 API 和可丢弃 MySQL 数据库，并为测试建立两个账户；数据库名称必须以 `tf_acceptance_` 或 `tf_sync_test_` 开头。API 与本地测试进程必须连接同一个数据库并使用相同的 `APP_TIMEZONE`。

后端测试通过真实登录验证账户身份，不再默认连接 `localhost:18000`。在仓库根目录配置：

```powershell
$env:RUN_SYNC_INTEGRATION = '1'
$env:TEST_SYNC_ISOLATED = '1'
$env:TEST_API_BASE = 'http://127.0.0.1:28026/api'
$env:TEST_SYNC_USERNAME = '<测试账户名>'
$env:TEST_SYNC_USER_ID = '<该账户的 user_id>'
$env:TEST_SYNC_OTHER_USER_ID = '<另一个测试账户的 user_id>'
$env:TEST_SYNC_PASSWORD_FILE = '<保存测试密码的 UTF-8 文本文件>'
$env:MYSQL_URL_FILE = '<保存隔离数据库 mysql+pymysql URL 的 UTF-8 文本文件>'
$env:APP_TIMEZONE = 'UTC' # 按隔离 API 的实际配置填写
$env:SYNC_DATA_DIR = '<隔离 API 同步数据目录在宿主机的实际映射路径>'
$env:TEST_SYNC_SERVER_DATA_DIR = '/app/data/sync' # 按隔离 API 的容器路径填写
$env:TEST_EXTERNAL_API_HOST = 'host.docker.internal'
$env:TEST_WORKER_API_HOST = '<本机当前可达的非 loopback 私网地址>'
$env:DISCOVER_MODELS_ALLOWED_PRIVATE_HOSTS = "host.docker.internal,$env:TEST_WORKER_API_HOST"
./scripts/run_tests.ps1 -q tests/test_sync_api.py tests/test_sync_pipeline.py tests/test_sync_worker.py
```

`MYSQL_URL` 与 `MYSQL_URL_FILE` 只能设置一个。`SYNC_DATA_DIR` 必须与容器的真实 bind mount 对应，测试会读取 API 写出的 JSONL 文件。API 若在 Docker 中，需要能访问测试进程在宿主机启动的 19876、19877 端口；API 与测试进程都应仅为该测试源配置私网访问白名单。`TEST_EXTERNAL_API_HOST` 用于容器访问宿主数据源，`TEST_WORKER_API_HOST` 用于本地 worker 访问同一数据源；两处网络解析可能不同，应分别检查可达性。安全策略会拒绝 loopback，不能用 `127.0.0.1` 替代 worker 的私网地址。密码文件只包含密码，数据库 URL 文件只包含连接 URL；不要将这些文件加入 Git。

浏览器的真实同步集成另需一个可访问的测试数据源（返回 `total`、`items`，可以是空列表），以及包含 `username`、`password` 字符串字段的私密 JSON 文件。在 `web` 目录执行：

```powershell
$env:E2E_RUN_SYNC_INTEGRATION = '1'
$env:E2E_SYNC_API_BASE = 'http://127.0.0.1:28026/api/sync'
$env:E2E_SYNC_SOURCE_URL = 'http://host.docker.internal:43244/items'
$env:E2E_SYNC_CREDENTIALS_FILE = '<测试账户 JSON 文件>'
$env:VITE_PROXY_TARGET = 'http://127.0.0.1:28026'
npm run test:e2e -- sync.integration.spec.ts --workers=1
```

测试数据源需接受测试配置使用的 `Bearer playwright-token`。上面的端口只是示例；测试服务与数据源须提前启动，且不得指向现有业务实例。
