# 前端架构

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 框架 | React 18 | 组件化 UI 框架 |
| 语言 | TypeScript | 类型安全 |
| 构建工具 | Vite 5 | 快速开发服务器和构建 |
| UI 组件库 | Ant Design 5 | 企业级 UI 组件 |
| Pro 组件 | @ant-design/pro-components | 高级业务组件 |
| 路由 | React Router 6 | SPA 路由管理 |
| HTTP 客户端 | Axios | API 请求 |
| 图表 | ECharts 6 | 训练曲线可视化 |
| 国际化 | react-i18next | 中英文双语切换 |
| 部署 | Nginx | 静态资源服务 |

## 页面模块

```
web/src/pages/
├── training/          # 训练任务（列表、创建、详情 + Loss 曲线）
├── datasets/          # 数据集（列表、创建、详情、Hub、下载、注册）
├── models/            # 模型注册（列表、卡片、详情弹窗、下载、注册）
├── evaluations/       # 评估（列表、创建、详情、深度评估在线/任务面板）
├── deployments/       # 部署（列表、创建、Adapter 管理、AdapterManager 组件）
├── generation/        # 数据生成（列表、创建、详情）
├── vectordb/          # 向量库（列表、创建集合、浏览、搜索）
├── resources/         # 资源监控（系统信息、GPU 卡片）
├── configs/           # 模型配置（列表、卡片、编辑、API 测试）
└── sync/              # 数据同步（列表、创建、详情）
```

## 公共组件

```
web/src/components/
├── Layout.tsx                # 页面布局（侧边栏 + Header + 面包屑 + LanguageToggle）
├── StatusTag.tsx             # 状态标签（动态翻译，含 adapter_load_failed 等扩展状态）
├── StatCard.tsx              # 统计卡片（支持 icon/color/suffix/tooltip）
├── GpuSelect.tsx             # GPU 选择器
├── ModelConfigSelector.tsx   # 模型配置选择器
├── ErrorBoundary.tsx         # 错误边界（class 组件，withTranslation HOC）
└── training/
    └── LossCurve.tsx         # 训练 Loss 曲线图（ECharts）
```

## i18n 国际化

### 架构

- **库**: `react-i18next` + `i18next`
- **配置**: `web/src/i18n/index.ts` — 静态导入所有 locale JSON
- **切换组件**: `web/src/i18n/LanguageToggle.tsx` — Header 右上角 toggle 按钮
- **持久化**: localStorage key `tf_language`，页面刷新后恢复
- **HTML lang**: 初始化时同步设置 `document.documentElement.lang`
- **Antd 联动**: `main.tsx` 中 `ConfigProvider locale` 根据语言动态切换 `zhCN`/`enUS`

### Namespace

11 个 namespace 与页面模块一一对应：

| Namespace | 文件 | 说明 |
|-----------|------|------|
| `common` | `common.json` | 导航、按钮、状态、分页、错误等通用文本 |
| `training` | `training.json` | 训练模块 |
| `datasets` | `datasets.json` | 数据集模块 |
| `generation` | `generation.json` | 数据生成模块 |
| `models` | `models.json` | 模型注册模块 |
| `configs` | `configs.json` | 模型配置模块 |
| `deployments` | `deployments.json` | 部署模块 |
| `evaluations` | `evaluations.json` | 评估模块 |
| `vectordb` | `vectordb.json` | 向量库模块 |
| `resources` | `resources.json` | 资源监控模块 |
| `sync` | `sync.json` | 数据同步模块 |

翻译文件位于 `web/src/i18n/locales/zh/*.json` 和 `web/src/i18n/locales/en/*.json`。

### 使用规范

```tsx
// 功能组件
import { useTranslation } from 'react-i18next'

function MyComponent() {
  const { t } = useTranslation(['myModule', 'common'])

  return <span>{t('section.key')}</span>           // 模块内 key
  return <span>{t('common:action.delete')}</span>   // 跨 namespace
  return <span>{t('key', { count: 5 })}</span>      // 变量插值
}

// Class 组件
import { withTranslation } from 'react-i18next'
class MyClass extends React.Component { ... }
export default withTranslation('common')(MyClass)

// 工具函数（非组件）
import i18n from '@/i18n'
i18n.t('common:status.running')
```

### Key 命名规范

`namespace:section.element.qualifier`

- `common:nav.training` → "训练管理" / "Training"
- `training:list.columns.taskName` → "任务名称" / "Task Name"
- `common:message.deleteSuccess` → "删除成功" / "Deleted successfully"

### 添加新翻译

1. 在 `web/src/i18n/locales/zh/<namespace>.json` 添加中文 key
2. 在 `web/src/i18n/locales/en/<namespace>.json` 添加对应英文 key
3. 确保 zh/en 的 key 结构完全一致

如需新增 namespace：
1. 创建 `zh/<name>.json` 和 `en/<name>.json`
2. 在 `web/src/i18n/index.ts` 中 import 并注册

## 关键业务组件

### AdapterManager（Adapter 管理器）

位于 `web/src/pages/deployments/AdapterManager.tsx`，嵌入到部署列表页的弹窗中。

**功能**:
- 列出部署上已加载的 adapter（状态标签：loading/loaded/unloading/failed）
- 加载 adapter：从训练任务或模型注册自动发现、手动输入路径
- 卸载 adapter：一键卸载已加载的 adapter
- 同步状态：与推理服务实际状态对齐

**入口**: 部署列表页中，`running` 状态且 `enable_lora=true` 的部署行显示闪电图标按钮（`ThunderboltOutlined`），点击打开 Modal。

**依赖 API**: `adapterApi`（`web/src/services/api.ts`）

### ApiTestModal（API 测试弹窗）

位于 `web/src/pages/configs/ApiTestModal.tsx`，从模型配置列表页打开。

**功能**:
- 向量测试：发送文本获取 embedding 向量
- 相似度测试：计算两段文本的 cosine 相似度
- Rerank 测试：测试重排序模型
- Adapter 覆盖：如果配置关联的部署启用了 LoRA，显示 adapter 选择器下拉框，选择后请求使用 adapter 名称作为 model

### ModelList / ModelCard（模型注册列表）

位于 `web/src/pages/models/ModelList.tsx` 和 `web/src/pages/models/ModelCard.tsx`。

**基础模型解析**:
- 构建 `baseModelMap`：`model_path → RegisteredModel` 的查找映射
- 支持路径后缀模糊匹配（`/app/models/bge-base-zh` 可匹配 `bge-base-zh`）
- 表格和卡片视图中，基础模型字段显示为可点击的模型名称（而非原始路径），点击跳转到对应模型详情

**表格列宽优化**: 名称列自适应宽度（`ellipsis: true`），类型/来源/状态等列固定宽度，避免列重叠。

### SyncConfigCreate（同步任务创建）

位于 `web/src/pages/sync/SyncConfigCreate.tsx`。

**Loss 函数选择器**:
- 根据 `model_type` 动态渲染对应的 loss 选择器
- `embedding`: 20 种 loss（默认 `DynamicExplicitNegativesRankingLoss`）
- `reranker`: 13 种 loss（默认 auto）
- `decoder_reranker`: 5 种 loss（infonce/bce/lambda_loss/list_mle/ranknet）+ `n_docs` 参数
- `llm`: 不显示 loss 配置

**按 model_type 条件渲染**:
- `embedding` 模型不显示 `max_length` 字段（sentence-transformers 不使用该参数）
- `keypoint_gen` 默认关闭

**训练方法限制**: 所有 model_type 在同步场景下仅支持 SFT，前端已隐藏 RL 训练方法选项。

### SyncConfigDetail（同步任务详情）

位于 `web/src/pages/sync/SyncConfigDetail.tsx`。

**主要功能**:
- 任务概览：基本信息、配置详情、统计计数
- 生成历史：生成列表、停用/启用开关、停用后自动显示重置批次数
- 训练历史：训练列表、训练状态与 adapter 状态分列显示
- Adapter 管理：当前 adapter 信息、卸载按钮、`adapter_load_failed` 状态显示重试按钮
- 生成配置视图：仅显示 `enabled=true` 的步骤标签

**状态常量**: 从 `web/src/constants/syncStatus.ts` 导入，集中管理各种 sync 状态判断逻辑。

## 构建与部署

```dockerfile
# Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build    # tsc -b && vite build → dist/

# Production stage
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

Nginx 负责：
- 静态文件服务（`/usr/share/nginx/html`）
- API 反向代理（`/api/*` → `http://train-factory-api:18000`）
- SPA 路由回退（所有路由 → `index.html`）
- 静态资源缓存（`/assets/` 1 年 immutable）
