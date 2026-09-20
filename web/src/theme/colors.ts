// CSS variables keep shared component styles in sync with the selected theme.

// 主色
export const PRIMARY = 'var(--tf-primary)'
export const PRIMARY_HOVER = 'var(--tf-primary-hover)'
export const PRIMARY_ACTIVE = 'var(--tf-primary-active)'

// 背景色
export const BG_LAYOUT = 'var(--tf-bg-layout)'
export const BG_CONTAINER = 'var(--tf-bg-container)'
export const BG_ELEVATED = 'var(--tf-bg-elevated)'
export const BG_SPOTLIGHT = 'var(--tf-bg-spotlight)'

// 文字色
export const TEXT_PRIMARY = 'var(--tf-text-primary)'
export const TEXT_SECONDARY = 'var(--tf-text-secondary)'
export const TEXT_TERTIARY = 'var(--tf-text-tertiary)'
export const TEXT_DISABLED = 'var(--tf-text-disabled)'

// 边框色
export const BORDER_PRIMARY = 'var(--tf-border-primary)'
export const BORDER_SECONDARY = 'var(--tf-border-secondary)'

// 状态色
export const STATUS_SUCCESS = '#3fb950'
export const STATUS_WARNING = '#d29922'
export const STATUS_ERROR = '#f85149'
export const STATUS_INFO = '#58a6ff'
export const STATUS_DEFAULT = '#8b949e'

// 渐变色
export const GRADIENT_PRIMARY = 'var(--tf-gradient-primary)'
export const GRADIENT_SUCCESS = 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)'
export const GRADIENT_INFO = 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
export const GRADIENT_PURPLE = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'

// 模型类型颜色
export const MODEL_TYPE_COLORS: Record<string, string> = {
  embedding: '#58a6ff',
  reranker: '#a371f7',
  decoder_reranker: '#f778ba',
  llm: '#3fb950',
}

// 训练方法颜色
export const TRAINING_METHOD_COLORS: Record<string, string> = {
  sft: '#58a6ff',
  dpo: '#a371f7',
  grpo: '#f778ba',
}

// 提供商颜色
export const PROVIDER_COLORS: Record<string, string> = {
  openai: '#10a37f',
  azure: '#0078d4',
  anthropic: '#d4a574',
  local: '#8b949e',
  huggingface: '#ffcc00',
}

// 状态映射
export const STATUS_COLOR_MAP: Record<string, string> = {
  // 训练状态
  pending: STATUS_WARNING,
  preparing: STATUS_INFO,
  running: STATUS_INFO,
  degraded: STATUS_WARNING,
  stopping: STATUS_INFO,
  recovering: STATUS_INFO,
  publishing: STATUS_INFO,
  restarting: STATUS_INFO,
  evaluating: STATUS_INFO,
  training: STATUS_INFO,
  succeeded: STATUS_SUCCESS,
  completed: STATUS_SUCCESS,
  failed: STATUS_ERROR,
  stopped: STATUS_WARNING,
  cancelled: STATUS_WARNING,

  // 部署状态
  deploying: STATUS_INFO,
  available: STATUS_SUCCESS,
  unavailable: STATUS_ERROR,

  // 通用状态
  active: STATUS_SUCCESS,
  inactive: STATUS_DEFAULT,
  archived: STATUS_DEFAULT,
  disabled: STATUS_DEFAULT,
  ready: STATUS_SUCCESS,
  error: STATUS_ERROR,
  healthy: STATUS_SUCCESS,
  unhealthy: STATUS_ERROR,
  unknown: STATUS_DEFAULT,

  // 同步状态
  idle: STATUS_DEFAULT,
  syncing: STATUS_INFO,
  generating: STATUS_INFO,
  loading_adapter: STATUS_INFO,

  // 同步批次状态
  fetched: STATUS_WARNING,
  registered: STATUS_INFO,
  uploading: STATUS_INFO,
  downloading: STATUS_INFO,
  processing: STATUS_INFO,
  generation_queued: STATUS_INFO,
  generation_done: STATUS_SUCCESS,
  adapter_loaded: STATUS_SUCCESS,
  adapter_unloaded: STATUS_DEFAULT,
  adapter_not_loaded: STATUS_DEFAULT,
  adapter_load_failed: STATUS_ERROR,
  adapter_failed: STATUS_ERROR,
}
