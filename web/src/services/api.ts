import axios, { type AxiosRequestConfig } from 'axios'
import { message } from 'antd'
import { currentReturnTo, loginPathFor } from '@/auth/returnTo'
import { toApiError } from '@/services/ApiError'
import type {
  TrainingTask,
  CreateTrainingRequest,
  Dataset,
  CreateDatasetRequest,
  UpdateDatasetRequest,
  RegisteredModel,
  Deployment,
  CreateDeploymentRequest,
  CreateContainerDeploymentRequest,
  RestartDeploymentRequest,
  UpdateDeploymentConfigRequest,
  GpuInfo,
  ModelConfig,
  CreateModelConfigRequest,
  LoadedAdapter,
  AvailableAdapter,
  EvaluationTask,
  CreateEvaluationRequest,
  AvailableDatasets,
  DeepEvaluationTask,
  TrainingTaskEvent,
  SyncTrainingTarget,
} from '@/types'

// Environment variables with defaults
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'
const API_TIMEOUT = Number(import.meta.env.VITE_API_TIMEOUT) || 30000
const AUTH_LOGOUT_TIMEOUT = Number(import.meta.env.VITE_AUTH_LOGOUT_TIMEOUT)

export type AppRequestConfig = AxiosRequestConfig & {
  silentUnauthorizedProbe?: boolean
  suppressErrorToast?: boolean
  suppressUnauthorizedRedirect?: boolean
}

export const SILENT_REQUEST_CONFIG: AppRequestConfig = {
  suppressErrorToast: true,
}

const LOGOUT_REQUEST_CONFIG: AppRequestConfig = {
  suppressErrorToast: true,
  suppressUnauthorizedRedirect: true,
  ...(Number.isFinite(AUTH_LOGOUT_TIMEOUT) && AUTH_LOGOUT_TIMEOUT > 0
    ? { timeout: AUTH_LOGOUT_TIMEOUT }
    : {}),
}

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
  // Enable credentials to send httpOnly cookies automatically
  withCredentials: true,
})

// Migrate from localStorage to httpOnly cookies
// Remove old localStorage token on first load (one-time cleanup)
if (typeof window !== 'undefined') {
  const oldToken = localStorage.getItem('auth_token')
  if (oldToken) {
    // Clear the old token - the httpOnly cookie will be used instead
    localStorage.removeItem('auth_token')
  }
}

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Auth is handled by httpOnly cookie (sent automatically with withCredentials: true)
    // No need to manually set Authorization header for browser requests
    return config
  },
  (error) => Promise.reject(toApiError(error, parseErrorMessage(error)))
)

// Parse error message from response
function parseErrorMessage(error: unknown): string {
  const response = (error as { response?: { data?: { detail?: unknown } } })?.response
  const detail = response?.data?.detail

  // FastAPI 422 validation error: detail is an array
  if (Array.isArray(detail)) {
    return detail
      .map((d: { msg?: string; loc?: string[] }) => {
        const field = d.loc?.slice(-1)[0] || 'field'
        return `${field}: ${d.msg || 'invalid'}`
      })
      .join('; ')
  }

  // String error message
  if (typeof detail === 'string') {
    return detail
  }

  // Object detail: prefer common message fields, fall back to JSON so the user
  // sees backend semantics instead of a generic axios "Request failed".
  if (detail && typeof detail === 'object') {
    const obj = detail as { message?: unknown; error?: unknown; msg?: unknown; detail?: unknown }
    const inner = obj.message ?? obj.error ?? obj.msg ?? obj.detail
    if (typeof inner === 'string') {
      return inner
    }
    try {
      return JSON.stringify(detail)
    } catch {
      return 'Request failed'
    }
  }

  // Fallback
  return (error as Error)?.message || 'Request failed'
}

type PaginationParams = {
  page?: number
  page_size?: number
  limit?: number
  offset?: number
}

function buildListParams<T extends Record<string, unknown>>(params?: T & PaginationParams) {
  if (!params) return undefined
  const { page, page_size, limit, offset, ...rest } = params
  if (limit !== undefined || offset !== undefined) {
    return {
      ...rest,
      ...(limit !== undefined ? { limit } : {}),
      ...(offset !== undefined ? { offset } : {}),
    }
  }
  const resolvedPage = page ?? 1
  const resolvedPageSize = page_size ?? 10
  return {
    ...rest,
    limit: resolvedPageSize,
    offset: Math.max(0, (resolvedPage - 1) * resolvedPageSize),
  }
}

// Response interceptor with enhanced error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const status = error.response?.status
    const errorMessage = parseErrorMessage(error)
    const requestConfig = error.config as AppRequestConfig | undefined
    const requestPath = requestConfig?.url?.split(/[?#]/, 1)[0] || ''
    const isAuthEntryRequest =
      requestPath.endsWith('/auth/login') || requestPath.endsWith('/auth/register')
    const isUnauthorizedMeProbe =
      status === 401 &&
      requestPath.endsWith('/auth/me') &&
      requestConfig?.silentUnauthorizedProbe === true

    // Entry-form errors are rendered inline, while /auth/me 401 is an expected
    // anonymous-session probe. Other failures keep the global handling below.
    const suppressErrorToast = requestConfig?.suppressErrorToast === true
    const suppressUnauthorizedRedirect =
      requestConfig?.suppressUnauthorizedRedirect === true

    if (isAuthEntryRequest || isUnauthorizedMeProbe) {
      // Intentionally silent.
    } else if (status === 401) {
      if (!suppressErrorToast) {
        message.error('Session expired, please login again')
      }
      // Redirect to login (skip if already there to avoid redirect loops),
      // carrying the original path so Login can return the user after sign-in.
      if (!suppressUnauthorizedRedirect && typeof window !== 'undefined') {
        const returnTo = currentReturnTo(window.location)
        let isLoginLocation = false
        try {
          const decodedPath = decodeURI(new URL(returnTo, window.location.origin).pathname)
          const loginComparablePath = decodedPath.replace(/\/+$/, '') || '/'
          isLoginLocation = /^\/login$/i.test(loginComparablePath)
        } catch {
          // A malformed current URL is not a valid login route.
        }

        if (!isLoginLocation) {
          window.location.href = loginPathFor(returnTo)
        }
      }
    } else if (!suppressErrorToast) {
      if (error.code === 'ECONNABORTED') {
        message.error('Request timeout, please try again')
      } else if (!error.response) {
        message.error('Network error, please check your connection')
      } else if (status === 403) {
        message.error('Permission denied')
      } else if (status === 404) {
        // Don't show message for 404, let the caller handle it
      } else if (status >= 500) {
        message.error('Server error, please try again later')
      }
    }

    return Promise.reject(toApiError(error, errorMessage))
  }
)

// Training Metrics Types
export interface TrainingMetricsResponse {
  task_id: string
  loss_history: Array<{
    step: number
    timestamp?: string
    epoch?: number
    train_loss?: number
    eval_loss?: number
    learning_rate?: number
    [key: string]: unknown
  }>
  summary: {
    task_id: string
    total_steps: number
    epochs_completed: number
    best_train_loss: number | null
    best_eval_loss: number | null
    loss_records_count: number
    start_time?: string
    end_time?: string
    status?: string
    epoch_summaries?: Array<{
      epoch: number
      timestamp: string
      [key: string]: unknown
    }>
    final_metrics?: Record<string, unknown>
  } | null
  current_metrics: {
    current_step?: number
    total_steps?: number
    current_epoch?: number
    total_epochs?: number
    train_loss?: number
    eval_loss?: number
    learning_rate?: number
    updated_at?: string
  } | null
  has_data: boolean
}

// Training API
export const trainingApi = {
  list: async (params?: { status?: string; page?: number; page_size?: number }) => {
    const res = await api.get<
      never,
      {
        tasks: TrainingTask[]
        total: number
        stats?: {
          total: number
          pending: number
          running: number
          succeeded: number
          failed: number
          stopped: number
        }
      }
    >('/train', {
      params: buildListParams(params),
    })
    return {
      items: res.tasks || [],
      total: res.total || 0,
      page: params?.page || 1,
      page_size: params?.page_size || 10,
      stats: res.stats,
    }
  },

  get: (taskId: string, config?: AppRequestConfig) =>
    api.get<never, TrainingTask>(`/train/${taskId}`, config),

  create: (data: CreateTrainingRequest) => api.post<never, TrainingTask>('/train', data),

  stop: (taskId: string) => api.post<never, void>(`/train/${taskId}/stop`),

  delete: (taskId: string) => api.delete<never, void>(`/train/${taskId}`),

  // Get training metrics including loss history for charts
  getMetrics: (taskId: string, limit?: number, config?: AppRequestConfig) =>
    api.get<never, TrainingMetricsResponse>(`/train/${taskId}/metrics`, {
      ...config,
      params: limit ? { limit } : undefined,
    }),

  getEvents: (taskId: string, limit?: number, config?: AppRequestConfig) =>
    api.get<never, { events: TrainingTaskEvent[]; total: number }>(`/train/${taskId}/events`, {
      ...config,
      params: limit ? { limit } : undefined,
    }),

  resume: (taskId: string) =>
    api.post<never, { message: string; task_id: string; checkpoint: string }>(
      `/train/${taskId}/resume`
    ),
}

// Dataset Download Types
export interface DownloadDatasetRequest {
  dataset_name: string
  remote_repo: string
  source_type: 'huggingface' | 'modelscope'
  hf_subset?: string
  dataset_type?: string
  usage?: 'train' | 'eval' | 'test'
  model_type?: ('embedding' | 'rerank' | 'llm')[]
  output_format?: 'jsonl' | 'json' | 'parquet' | 'arrow'
  clean_cache?: boolean
  description?: string
  user_id?: string
}

export interface DatasetDownloadResponse {
  dataset_id: string
  dataset_name: string
  status: string
  storage_path: string
}

export interface DatasetDownloadProgress {
  dataset_id: string
  dataset_name?: string
  source_type?: string
  remote_repo?: string
  storage_path?: string
  status: string
  progress: number
  error?: string
  created_at?: string
}

export interface ExportDatasetRequest {
  format?: 'original' | 'jsonl' | 'csv'
  limit?: number
  expires_seconds?: number
}

export interface ExportDatasetResponse {
  dataset_id: string
  source_format: string
  export_format: string
  row_count?: number
  storage_uri: string
  download_url: string
  proxy_download_url?: string
  expires_seconds: number
}

export interface DatasetPreviewResponse {
  columns?: Record<string, unknown>[] | string[]
  rows?: Record<string, unknown>[]
  total_rows?: number
}

// Dataset API
export const datasetApi = {
  list: async (params?: {
    dataset_type?: string
    usage?: string
    model_type?: string
    status?: string
    page?: number
    page_size?: number
  }, config?: AppRequestConfig) => {
    const res = await api.get<
      never,
      {
        datasets: Dataset[]
        total: number
        stats?: {
          total: number
          by_usage?: Record<string, number>
          by_status?: Record<string, number>
        }
      }
    >('/datasets', {
      ...config,
      params: buildListParams(params),
    })
    return {
      items: res.datasets || [],
      total: res.total || 0,
      page: params?.page || 1,
      page_size: params?.page_size || 10,
      stats: res.stats,
    }
  },

  get: (datasetId: string) => api.get<never, Dataset>(`/datasets/${datasetId}`),

  create: (data: CreateDatasetRequest) => api.post<never, Dataset>('/datasets', data),

  update: (datasetId: string, data: UpdateDatasetRequest) =>
    api.put<never, Dataset>(`/datasets/${datasetId}`, data),

  delete: (datasetId: string) => api.delete<never, void>(`/datasets/${datasetId}`),

  upload: (formData: FormData, onProgress?: (percent: number) => void) =>
    api.post<never, Dataset>('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 600000,
      onUploadProgress: (e: { loaded: number; total?: number }) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
      },
    }),

  preview: (datasetId: string, limit?: number) =>
    api.get<never, DatasetPreviewResponse | Record<string, unknown>[]>(
      `/datasets/${datasetId}/preview`,
      {
        params: { limit },
      }
    ),

  getTypes: () => api.get<never, { types: string[] }>('/datasets/types'),

  // Dataset Download APIs
  download: (data: DownloadDatasetRequest) =>
    api.post<never, DatasetDownloadResponse>('/datasets/download', data),

  getDownloadProgress: (datasetId: string) =>
    api.get<never, DatasetDownloadProgress>(`/datasets/download/${datasetId}/progress`),

  listDownloads: () =>
    api.get<never, { downloads: DatasetDownloadProgress[]; total: number }>('/datasets/downloads'),

  export: (datasetId: string, data?: ExportDatasetRequest) =>
    api.post<never, ExportDatasetResponse>(`/datasets/${datasetId}/export`, data || {}),
}

// Model Download Types
export interface DownloadModelRequest {
  download_source: 'modelscope' | 'huggingface'
  remote_repo: string
  model_type: 'embedding' | 'reranker' | 'decoder_reranker' | 'llm'
  display_name?: string
  description?: string
  user_id?: string
}

// Discover Models Types
export interface DiscoveredModel {
  model_uid: string
  model_name?: string
  model_type?: string
  model_path?: string
  status?: string
}

export interface DiscoverModelsResponse {
  endpoint: string
  framework: string
  models: DiscoveredModel[]
}

export interface BindExistingModelRequest {
  endpoint: string
  model_uid: string
  model_name?: string
  model_type: string
  deployment_name?: string
  inference_framework: string
  container_name?: string
  gpu_id?: number
  external_api_config_id?: string
}

export interface DownloadResponse {
  registry_id: string
  model_name: string
  display_name: string
  status: string
  model_path: string
}

export interface DownloadProgress {
  registry_id: string
  model_name?: string
  display_name?: string
  download_source?: string
  remote_repo?: string
  model_path?: string
  status: string
  progress: number
  error?: string
  created_at?: string
}

// Model Registry API
export const modelApi = {
  list: async (params?: {
    model_type?: string
    status?: string
    page?: number
    page_size?: number
  }) => {
    const res = await api.get<
      never,
      {
        models: RegisteredModel[]
        total: number
        stats?: {
          total: number
          by_status?: Record<string, number>
          by_type?: Record<string, number>
        }
      }
    >('/models', {
      params: buildListParams(params),
    })
    return {
      items: res.models || [],
      total: res.total || 0,
      page: params?.page || 1,
      page_size: params?.page_size || 10,
      stats: res.stats,
    }
  },

  get: (modelId: string) => api.get<never, RegisteredModel>(`/models/${modelId}`),

  delete: (modelId: string) => api.delete<never, void>(`/models/${modelId}`),

  register: (data: {
    model_name: string
    model_path: string
    model_type: string
    version?: string
    base_model_path?: string
    description?: string
  }) => api.post<never, RegisteredModel>('/models', data),

  registerFromTask: (
    taskId: string,
    data: { model_name: string; description?: string; version?: string }
  ) => api.post<never, RegisteredModel>(`/models/from-task/${taskId}`, data),

  // Model Download APIs
  download: (data: DownloadModelRequest) =>
    api.post<never, DownloadResponse>('/models/download', data),

  getDownloadProgress: (registryId: string) =>
    api.get<never, DownloadProgress>(`/models/download/${registryId}/progress`),

  listDownloads: () =>
    api.get<never, { downloads: DownloadProgress[]; total: number }>('/models/downloads'),
}

// Deployment API
export const deploymentApi = {
  list: async (params?: { status?: string; model_id?: string; page?: number; page_size?: number }) => {
    const res = await api.get<never, { deployments: Deployment[]; total: number }>('/deployments', {
      params: { ...buildListParams(params), sync: true },
    })
    return {
      items: res.deployments || [],
      total: res.total || 0,
      page: params?.page || 1,
      page_size: params?.page_size || 10,
    }
  },

  get: (deploymentId: string) => api.get<never, Deployment>(`/deployments/${deploymentId}`),

  create: (data: CreateDeploymentRequest) => api.post<never, Deployment>('/deployments', data),

  // Container deployment (auto-creates Docker container)
  createContainer: (data: CreateContainerDeploymentRequest) =>
    api.post<never, Deployment>('/deployments/container', data),

  start: (deploymentId: string) =>
    api.post<never, Deployment>(`/deployments/${deploymentId}/start`),

  stop: (deploymentId: string) => api.post<never, Deployment>(`/deployments/${deploymentId}/stop`),

  restart: (deploymentId: string, data: RestartDeploymentRequest = { reset_gpu: false }) =>
    api.post<never, Deployment>(`/deployments/${deploymentId}/restart`, data),

  updateConfig: (deploymentId: string, data: UpdateDeploymentConfigRequest) =>
    api.patch<never, Deployment>(`/deployments/${deploymentId}/config`, data),

  delete: (deploymentId: string, force: boolean = true) =>
    api.delete<never, void>(`/deployments/${deploymentId}`, {
      params: { force },
    }),

  // GPU info
  getGpuInfo: () => api.get<never, { gpus: GpuInfo[] }>('/gpus'),

  // Default Xinference endpoint
  getDefaultEndpoint: () => api.get<never, { endpoint: string }>('/default-endpoint'),

  // Discover models on an inference endpoint
  discoverModels: (endpoint: string, framework: string = 'xinference') =>
    api.get<never, DiscoverModelsResponse>('/deployments/discover-models', {
      params: { endpoint, framework },
    }),

  // Bind an existing model on an inference endpoint
  bindExisting: (data: BindExistingModelRequest) =>
    api.post<never, Deployment>('/deployments/bind-existing', data),
}

// Adapter API
export const adapterApi = {
  // List loaded adapters on a deployment
  listLoaded: (deploymentId: string, includeUnloaded?: boolean) =>
    api.get<never, { adapters: LoadedAdapter[] }>(`/deployments/${deploymentId}/adapters`, {
      params: { include_unloaded: includeUnloaded },
    }),

  // Load an adapter onto a deployment
  load: (
    deploymentId: string,
    data: {
      adapter_name: string
      adapter_path: string
      source_task_id?: string
      source_model_id?: string
    }
  ) => api.post<never, LoadedAdapter>(`/deployments/${deploymentId}/adapters`, data),

  // Load adapter from a training task
  loadFromTask: (deploymentId: string, taskId: string, adapterName?: string) =>
    api.post<never, LoadedAdapter>(`/deployments/${deploymentId}/adapters/from-task`, {
      task_id: taskId,
      adapter_name: adapterName,
    }),

  // Unload an adapter
  unload: (deploymentId: string, adapterName: string) =>
    api.delete<never, { message: string }>(`/deployments/${deploymentId}/adapters/${adapterName}`),

  // Sync loaded adapters with actual state
  sync: (deploymentId: string) =>
    api.post<never, { message: string }>(`/deployments/${deploymentId}/adapters/sync`),

  // List available adapters
  listAvailable: (baseModelId?: string) =>
    api.get<never, { adapters: AvailableAdapter[] }>('/adapters/available', {
      params: { base_model_id: baseModelId },
    }),

  // Get adapter by ID
  get: (adapterId: string) => api.get<never, LoadedAdapter>(`/adapters/${adapterId}`),
}

// Model Config API
export const configApi = {
  list: async (params?: { provider?: string; page?: number; page_size?: number }) => {
    const res = await api.get<never, { configs: ModelConfig[]; total: number }>('/configs', {
      params: buildListParams(params),
    })
    return {
      items: res.configs || [],
      total: res.total || 0,
      page: params?.page || 1,
      page_size: params?.page_size || 10,
    }
  },

  get: (configId: string) => api.get<never, ModelConfig>(`/configs/${configId}`),

  create: (data: CreateModelConfigRequest) => api.post<never, ModelConfig>('/configs', data),

  update: (configId: string, data: Partial<CreateModelConfigRequest>) =>
    api.put<never, ModelConfig>(`/configs/${configId}`, data),

  delete: (configId: string) => api.delete<never, void>(`/configs/${configId}`),

  validate: (data: CreateModelConfigRequest) =>
    api.post<never, { valid: boolean; message?: string }>('/configs/validate', data),

  checkConnection: (configId: string) =>
    api.post<
      never,
      {
        success: boolean
        latency_ms?: number
        error?: string
        warning?: string
        models?: string[]
        model_details?: Array<{
          id?: string | null
          parent?: string | null
          model_type?: string | null
          owned_by?: string | null
          created?: number | string | null
          context_length?: number | null
          max_tokens?: number | null
          max_completion_tokens?: number | null
          embedding_dim?: number | null
          capabilities?: unknown
        }>
      }
    >(`/configs/${configId}/check`),

  checkAll: () =>
    api.post<
      never,
      {
        total: number
        healthy: number
        results: Record<
          string,
          { success: boolean; latency_ms?: number; error?: string; config_name?: string }
        >
      }
    >('/configs/check-all'),

  // API 测试代理（后端自动处理 vLLM reranker 预格式化）
  test: (configId: string, path: string, body: Record<string, unknown>) =>
    api.post<
      never,
      {
        success: boolean
        status_code: number
        latency_ms: number
        data?: unknown // 可以是 Dict 或 List（SGLang 返回数组）
        error?: string
      }
    >(`/configs/${configId}/test`, { path, body }),

  // 列出与当前配置匹配的向量库
  listCollections: (configId: string) =>
    api.get<
      never,
      {
        collections: Array<{
          name: string
          match_type: 'exact' | 'model_match' | 'mismatch'
          embedding_model: string | null
          embedding_config_id: string | null
          dim: number | null
          status: string | null
        }>
      }
    >(`/configs/${configId}/collections`),
}

// Resource API
export const resourceApi = {
  getStatus: (config?: AppRequestConfig) =>
    api.get<
      never,
      {
        available?: boolean
        partial?: boolean
        error_code?: string | null
        errors?: Array<{ scope: string; error_code: string }>
        gpu: {
          available?: boolean
          partial?: boolean
          error_code?: string | null
          device_state?: 'detected' | 'no_devices' | 'unknown'
          total_gpus: number | null
          allocated_gpus: number | null
          free_gpus: number | null
          utilization_rate?: number | null
          gpu_memory_summary?: {
            total_gb: number
            used_gb: number
            free_gb: number
            usage_percent: number
          }
          gpu_details?: Record<
            string,
            {
              status: string
              task_id: string | null
              gpu_name: string
              memory: {
                total_gb: number
                used_gb: number
                free_gb: number
                usage_percent: number
              }
              utilization: {
                gpu_percent: number | null
                memory_percent: number | null
              }
              temperature: number | null
              power: {
                usage_w: number | null
                limit_w: number | null
              }
            }
          >
          cpu_summary?: {
            logical_cores: number
            physical_cores: number
            cpu_usage_percent: number
          }
          system_memory?: {
            total_gb: number
            used_gb: number
            free_gb: number
            usage_percent: number
          }
        }
        system: {
          available?: boolean
          partial?: boolean
          error_code?: string | null
          errors?: Array<{ scope: string; error_code: string }>
          cpu_percent: number | null
          memory_percent: number | null
          memory_used_mb: number | null
          memory_total_mb: number | null
          memory_limit_set?: boolean
          disk_usage_percent: number | null
          disk_used_gb: number | null
          disk_total_gb: number | null
          disk_mountpoint?: string | null
          disk_volumes?: Array<{
            scope: string
            scopes: string[]
            paths?: string[]
            mountpoint: string | null
            usage_percent: number
            used_gb: number
            total_gb: number
            free_gb: number
          }>
          open_files: number | null
          thread_count: number | null
        }
      }
    >('/resources/status', config),

  getGpuList: (config?: AppRequestConfig) =>
    api.get<
      never,
      {
        total: number
        gpus: Array<{
          id: number
          name: string
          memory_total_gb: number
          memory_used_gb: number
          memory_free_gb: number
          memory_usage_percent: number
          gpu_utilization: number | null
          temperature: number | null
          power_usage_w: number | null
          power_limit_w: number | null
          is_allocated: boolean
          allocated_task: string | null
        }>
      }
    >('/resources/gpus', config),

  getGpuProcesses: (config?: AppRequestConfig) =>
    api.get<
      never,
      {
        total: number
        gpus: Array<{
          gpu_index: number | null
          name: string | null
          uuid: string | null
          total_memory_mb: number | null
          processes: Array<{
            pid?: number
            name?: string | null
            cmdline?: string | null
            user?: string | null
            gpu_memory_mb: number | null
            container_id?: string | null
            container_name?: string | null
            started_at?: string | null
          }>
        }>
      }
    >('/resources/gpus/processes', config),

  cleanup: () =>
    api.post<
      never,
      {
        success: boolean
        actions: string[]
        errors: string[]
        stale_gpu_allocations_cleaned: number
      }
    >('/resources/cleanup'),
}

// Evaluation API (Independent evaluation tasks for deployed models)
export const evaluationApi = {
  createTask: (data: CreateEvaluationRequest) =>
    api.post<never, { task_id: string; message: string }>('/evaluations', data),

  listTasks: async (
    params?: { status?: string; limit?: number; offset?: number },
    config?: AppRequestConfig
  ) => {
    const res = await api.get<never, { items: EvaluationTask[]; total: number }>(
      '/evaluations/tasks',
      { ...config, params }
    )
    return res
  },

  getTask: (taskId: string) => api.get<never, EvaluationTask>(`/evaluations/tasks/${taskId}`),

  cancelTask: (taskId: string) =>
    api.post<never, { message: string; task_id: string }>(`/evaluations/tasks/${taskId}/cancel`),

  deleteTask: (taskId: string) =>
    api.delete<never, { message: string; task_id: string }>(`/evaluations/tasks/${taskId}`),

  resumeTask: (taskId: string) =>
    api.post<never, { message: string; task_id: string; skipped_evaluations: number }>(
      `/evaluations/tasks/${taskId}/resume`
    ),

  getAvailableDatasets: () => api.get<never, AvailableDatasets>('/evaluations/datasets'),
}

// Deep Evaluation API (for embedding/reranker model evaluation)
// Deep Evaluation API endpoints
export const deepEvaluationApi = {
  // Get supported eval types and metrics
  getEvalTypes: () =>
    api.get<
      never,
      Record<
        string,
        {
          metrics: string[]
          description: string
          default_metrics: string[]
        }
      >
    >('/deep-evaluation/eval-types'),

  // Get available metrics
  getMetrics: (category?: string, config?: AppRequestConfig) =>
    api.get<
      never,
      Array<{
        name: string
        description: string
        category: string
        requires_llm: boolean
        requires_expected_output: boolean
        requires_actual_output: boolean
        requires_retrieval_context: boolean
      }>
    >('/deep-evaluation/metrics', {
      ...config,
      params: category ? { category } : {},
    }),

  // Get eval-ready collections (from collection registry)
  getEvalReadyDatasets: (config?: AppRequestConfig) =>
    api.get<
      never,
      Array<{
        collection_name: string
        display_name?: string
        collection_id: string
        embedding_config_id?: string
        embedding_model?: string
        embedding_endpoint?: string
        dim?: number
        linked_datasets?: Array<{ dataset_id: string; dataset_name?: string; chunk_count?: number }>
        milvus_collection: string
        embedding_config?: { config_id?: string; endpoint?: string; model?: string }
      }>
    >('/deep-evaluation/datasets/eval-ready', config),

  // Deep evaluation tasks
  createTask: (data: Record<string, unknown>) =>
    api.post<never, { task_id: string; message: string }>('/deep-evaluation/tasks', data),

  listTasks: async (
    params?: {
      status?: string
      eval_type?: string
      limit?: number
      offset?: number
    },
    config?: AppRequestConfig
  ) => {
    const res = await api.get<never, { items: DeepEvaluationTask[]; total: number }>(
      '/deep-evaluation/tasks',
      { ...config, params }
    )
    return res
  },

  getTask: (taskId: string) =>
    api.get<never, DeepEvaluationTask>(`/deep-evaluation/tasks/${taskId}`),

  cancelTask: (taskId: string) =>
    api.post<never, { message: string; task_id: string }>(
      `/deep-evaluation/tasks/${taskId}/cancel`
    ),

  resumeTask: (taskId: string) =>
    api.post<never, { message: string; task_id: string }>(
      `/deep-evaluation/tasks/${taskId}/resume`
    ),

  deleteTask: (taskId: string) =>
    api.delete<never, { message: string; task_id: string }>(`/deep-evaluation/tasks/${taskId}`),

  // Online evaluation (deep evaluation quick test)
  evaluate: (data: {
    input: string
    expected_output?: string
    actual_output?: string
    retrieval_context?: string[]
    metrics: string[]
    llm_config: {
      config_id: string
      temperature?: number
      max_tokens?: number
      timeout?: number
    }
  }) =>
    api.post<
      never,
      {
        results: Record<
          string,
          {
            score: number
            reason: string
            details?: Record<string, unknown>
          }
        >
        overall_score: number
      }
    >('/deep-evaluation/evaluate', data),
}

export type GenerationTaskStatus =
  | 'pending'
  | 'running'
  | 'stopping'
  | 'recovering'
  | 'publishing'
  | 'restarting'
  | 'completed'
  | 'failed'
  | 'stopped'
  | 'deleting'
  | 'deleting_cascade'

export interface GenerationTaskListItem {
  task_id: string
  task_name: string
  status: GenerationTaskStatus
  generation_mode: string
  pos_neg_method?: string
  progress: number
  total_docs: number
  processed_docs: number
  output_sample_count: number
  error_message?: string
  created_at?: string
}

export interface GenerationTaskStats {
  total: number
  pending: number
  running: number
  stopping: number
  recovering: number
  publishing: number
  restarting: number
  completed: number
  failed: number
  stopped: number
}

export interface GenerationTaskListResponse {
  tasks: GenerationTaskListItem[]
  total: number
  stats: GenerationTaskStats
  limit: number
  offset: number
}

// Data Generation API
export const generationApi = {
  // Create generation task
  createTask: (data: {
    task_name: string
    dataset_id?: string
    input_path?: string
    content_field?: string
    input_format?: string
    output_format?: string
    generation_mode?: string
    llm_config: {
      config_id: string
      temperature?: number
      max_tokens?: number
      timeout?: number
      concurrency?: number
    }
    eval_llm_config?: {
      config_id: string
    }
    embedding_config?: {
      config_id: string
      batch_size?: number
      concurrency?: number
      similarity_threshold?: number
      retrieval_top_k?: number
    }
    rerank_config?: {
      config_id: string
      top_k?: number
      batch_size?: number
      concurrency?: number
      rerank_threshold?: number
    }
    worker_config?: {
      timeout_per_doc?: number
    }
    steps?: {
      doc_quality?: { enabled?: boolean; min_score?: number; use_llm?: boolean }
      keypoint_gen?: { enabled?: boolean; max_keypoints?: number }
      qa_gen?: {
        enabled?: boolean
        num_qa_per_doc?: number
        use_role?: boolean
        roles_per_doc?: number
      }
      pos_neg_extraction?: {
        enabled?: boolean
        num_positive?: number
        num_negative?: number
        use_role?: boolean
        roles_per_doc?: number
        augment?: boolean
        neg_detection_mode?: string
        supplement_positives?: boolean
        confirm_positives?: boolean
        confirm_negatives?: boolean
        answer_rewrite?: boolean
        rerank_score_classification?: boolean
        evidence_removal?: boolean
        evidence_pruning?: boolean
        neg_chunk_scoring?: boolean
        chunk_eval_mode?: string
        skip_easy_negatives?: boolean
        skip_perfect_ap?: boolean
        skip_zero_ap?: boolean
      }
      validation?: { enabled?: boolean }
    }
    post_process?: {
      dedup?: { enabled?: boolean }
      augmentation?: Record<string, unknown>
      evidence_ops?: Record<string, unknown>
      mining?: Record<string, unknown>
    }
    prompts?: Record<string, string>
    auto_register_dataset?: boolean
    // Training 模式 specific
    pos_neg_method?: string
    // 使用已有集合
    milvus_collection_name?: string
  }) =>
    api.post<
      never,
      {
        task_id: string
        task_name: string
        status: GenerationTaskStatus
        generation_mode: string
        pos_neg_method?: string
        progress: number
        total_docs: number
        processed_docs: number
        output_sample_count: number
      }
    >('/generation/tasks', data),

  // List tasks
  listTasks: (
    params?: {
      status?: GenerationTaskStatus
      generation_mode?: string
      limit?: number
      offset?: number
    },
    config?: AppRequestConfig
  ) => api.get<never, GenerationTaskListResponse>('/generation/tasks', { ...config, params }),

  // Get task detail (full config)
  getTask: (taskId: string) =>
    api.get<
      never,
      {
        task_id: string
        task_name: string
        description?: string
        status: GenerationTaskStatus
        progress: number
        total_docs: number
        processed_docs: number
        output_sample_count: number
        error_message?: string
        // 输入配置
        input_path: string
        input_format: string
        content_field?: string
        generation_mode: string
        pos_neg_method?: string
        // 输出配置
        output_path?: string
        output_format: string
        output_dataset_id?: string
        auto_register_dataset: boolean
        // 模型配置
        llm_config: Record<string, unknown>
        eval_llm_config?: Record<string, unknown>
        embedding_config?: Record<string, unknown>
        rerank_config?: Record<string, unknown>
        // 处理配置
        worker_config: Record<string, unknown>
        steps_config: Record<string, unknown>
        post_process_config?: Record<string, unknown>
        // Training 模式配置
        source_dataset_id?: string
        embedding_config_id?: string
        milvus_collection?: string
        filter_stats?: Record<string, unknown>
        // Doc-to-Training 中间产物
        qa_output_path?: string
        qa_dataset_id?: string
        qa_filtered_path?: string
        qa_filtered_dataset_id?: string
        stages?: Array<{
          stage: string
          label: string
          status: 'pending' | 'running' | 'completed' | 'skipped'
          checkpoint_path?: string
          resume_supported: boolean
          resume_ready: boolean
        }>
        // 时间戳
        created_at?: string
        started_at?: string
        completed_at?: string
      }
    >(`/generation/tasks/${taskId}`),

  // Get task progress
  getTaskProgress: (taskId: string) =>
    api.get<
      never,
      {
        task_id: string
        status: GenerationTaskStatus
        progress: number
        total_docs: number
        processed_docs: number
        output_sample_count: number
      }
    >(`/generation/tasks/${taskId}/progress`),

  // Get task artifacts grouped by stages
  getTaskArtifacts: (taskId: string, params?: { include_empty?: boolean }) =>
    api.get<
      never,
      {
        task_id: string
        total_artifacts: number
        stages: Array<{
          stage_key: string
          stage_name: string
          step_order: number
          stage_status: 'pending' | 'running' | 'completed' | 'skipped'
          artifacts: Array<{
            dataset_id: string
            dataset_name: string
            stage_key: string
            stage_name: string
            artifact_role: string
            dataset_type: string
            usage: string
            file_format?: string
            num_rows?: number
            file_size?: number
            status: string
            tags?: string[]
            created_at?: string
          }>
        }>
      }
    >(`/generation/tasks/${taskId}/artifacts`, { params }),


  // Stop task
  stopTask: (taskId: string) =>
    api.post<never, { status: string }>(`/generation/tasks/${taskId}/stop`),

  // Delete task
  deleteTask: (taskId: string, cascade = false) =>
    api.delete<never, { status: string }>(`/generation/tasks/${taskId}`, {
      params: cascade ? { cascade: true } : undefined,
    }),

  // Preview output data
  previewOutput: (taskId: string, params?: { limit?: number; offset?: number; source?: string }) =>
    api.get<
      never,
      {
        total: number
        samples: Array<Record<string, unknown>>
      }
    >(`/generation/tasks/${taskId}/preview`, { params }),

  // Restart failed/stopped/completed task
  // force=true: 从头运行（忽略检查点）; force=false: 断点续传
  restartTask: (taskId: string, force?: boolean) =>
    api.post<never, { status: string; task_id: string }>(
      `/generation/tasks/${taskId}/restart`,
      undefined,
      { params: force ? { force: true } : undefined }
    ),

  // Get supported formats
  getFormats: () =>
    api.get<
      never,
      {
        output_formats: Array<{
          name: string
          description: string
        }>
        source_types: string[]
        length_types: string[]
      }
    >('/generation/formats'),
}

// Milvus Vector Database API
export const milvusApi = {
  getStatus: () =>
    api.get<
      never,
      { connected: boolean; host: string; port: number; collection_count?: number; error?: string }
    >('/milvus/status'),

  listCollections: () =>
    api.get<never, { collections: import('@/types').MilvusCollectionSummary[]; total: number }>(
      '/milvus/collections'
    ),

  getCollection: (name: string) =>
    api.get<never, import('@/types').MilvusCollectionDetail>(
      `/milvus/collections/${encodeURIComponent(name)}`
    ),

  createCollection: (data: {
    name: string
    dim: number
    metric_type?: string
    description?: string
    embedding_config_id?: string
    display_name?: string
    enable_hybrid?: boolean
  }) =>
    api.post<never, { name: string; collection_id?: string; message: string }>(
      '/milvus/collections',
      data
    ),

  deleteCollection: (name: string) =>
    api.delete<never, { message: string }>(`/milvus/collections/${encodeURIComponent(name)}`),

  browseEntities: (
    name: string,
    params?: { offset?: number; limit?: number; include_vector?: boolean }
  ) =>
    api.get<never, { total: number; entities: Record<string, unknown>[] }>(
      `/milvus/collections/${encodeURIComponent(name)}/entities`,
      { params }
    ),

  search: (
    name: string,
    data: {
      query_text: string
      embedding_config_id?: string
      top_k?: number
      search_mode?: import('@/types').MilvusSearchMode
      ranker?: import('@/types').MilvusRankerType
      dense_weight?: number
      sparse_weight?: number
      filter_expr?: string
    }
  ) =>
    api.post<
      never,
      {
        results: import('@/types').MilvusSearchResult[]
        query_text: string
        search_mode: string
        total: number
      }
    >(`/milvus/collections/${encodeURIComponent(name)}/search`, data),

  loadCollection: (name: string) =>
    api.post<never, { message: string }>(`/milvus/collections/${encodeURIComponent(name)}/load`),

  releaseCollection: (name: string) =>
    api.post<never, { message: string }>(`/milvus/collections/${encodeURIComponent(name)}/release`),

  // Dataset link endpoints
  getLinkedDatasets: (name: string) =>
    api.get<never, { datasets: import('@/types').LinkedDatasetInfo[]; total: number }>(
      `/milvus/collections/${encodeURIComponent(name)}/datasets`
    ),

  linkDataset: (
    name: string,
    data: { dataset_id: string; dataset_name?: string; chunk_count?: number; task_id?: string }
  ) =>
    api.post<never, import('@/types').LinkedDatasetInfo>(
      `/milvus/collections/${encodeURIComponent(name)}/datasets`,
      data
    ),

  unlinkDataset: (name: string, datasetId: string) =>
    api.delete<never, { message: string }>(
      `/milvus/collections/${encodeURIComponent(name)}/datasets/${encodeURIComponent(datasetId)}`
    ),
}

// External Sync API
export const syncApi = {
  // Task CRUD
  createTask: (data: import('@/types').CreateSyncConfigRequest) =>
    api.post<never, { message: string; task: import('@/types').SyncConfig }>('/sync/tasks', data),

  listTasks: (params?: {
    limit: number
    offset: number
    external_api_config_id?: string
  }, config?: AppRequestConfig) =>
    api.get<never, { tasks: import('@/types').SyncConfig[]; total: number }>('/sync/tasks', {
      ...config,
      params: buildListParams(params),
    }),

  getTask: (taskId: string) =>
    api.get<never, { task: import('@/types').SyncConfig }>(`/sync/tasks/${taskId}`),

  updateTask: (taskId: string, data: Partial<import('@/types').CreateSyncConfigRequest>) =>
    api.patch<never, { message: string; task: import('@/types').SyncConfig }>(
      `/sync/tasks/${taskId}`,
      data
    ),

  deleteTask: (taskId: string) => api.delete<never, { message: string }>(`/sync/tasks/${taskId}`),

  // Worker control
  start: (taskId: string) =>
    api.post<never, { message: string; worker_status: string }>(`/sync/tasks/${taskId}/start`),

  stop: (taskId: string) =>
    api.post<never, { message: string; worker_status: string }>(`/sync/tasks/${taskId}/stop`),

  syncNow: (taskId: string) =>
    api.post<never, { message: string }>(`/sync/tasks/${taskId}/sync-now`),

  triggerGeneration: (taskId: string) =>
    api.post<never, { message: string }>(`/sync/tasks/${taskId}/trigger-generation`),

  triggerTraining: (taskId: string) =>
    api.post<never, { message: string }>(`/sync/tasks/${taskId}/trigger-training`),

  // History queries
  getStatus: (taskId: string) =>
    api.get<never, import('@/types').SyncStatusInfo>(`/sync/tasks/${taskId}/status`),

  listBatches: (taskId: string, params?: { status?: string; limit?: number; offset?: number }) =>
    api.get<never, { batches: import('@/types').SyncBatch[]; total: number }>(
      `/sync/tasks/${taskId}/batches`,
      { params }
    ),

  listGenerations: (
    taskId: string,
    params?: { status?: string; limit?: number; offset?: number }
  ) =>
    api.get<never, { generations: import('@/types').SyncGeneration[]; total: number }>(
      `/sync/tasks/${taskId}/generations`,
      { params }
    ),

  toggleGenerationDisabled: (taskId: string, generationTaskId: string, disabled: boolean) =>
    api.patch<
      never,
      import('@/types').SyncGeneration & { reset_batch_count?: number; reset_record_count?: number }
    >(`/sync/tasks/${taskId}/generations/${generationTaskId}/disabled`, { disabled }),

  recalculateCounters: (taskId: string) =>
    api.post<
      never,
      {
        old_total_training_samples: number
        new_total_training_samples: number
        old_pending_training_samples: number
        new_pending_training_samples: number
      }
    >(`/sync/tasks/${taskId}/recalculate-counters`),

  listTrainings: (taskId: string, params?: { status?: string; limit?: number; offset?: number }) =>
    api.get<never, { trainings: import('@/types').SyncTraining[]; total: number }>(
      `/sync/tasks/${taskId}/trainings`,
      { params }
    ),

  retryAdapterLoad: (taskId: string, trainingTaskId: string, replace = true) =>
    api.post<never, { message: string }>(
      `/sync/tasks/${taskId}/trainings/${trainingTaskId}/retry-adapter-load`,
      undefined,
      { params: { replace } }
    ),

  unloadAdapter: (taskId: string) =>
    api.post<never, { message: string }>(`/sync/tasks/${taskId}/unload-adapter`),

  // Training Targets
  listTargets: (taskId: string) =>
    api.get<never, { targets: SyncTrainingTarget[]; total: number }>(
      `/sync/tasks/${taskId}/targets`
    ),

  createTarget: (taskId: string, data: Record<string, unknown>) =>
    api.post<never, { message: string; target: SyncTrainingTarget }>(
      `/sync/tasks/${taskId}/targets`,
      data
    ),

  updateTarget: (taskId: string, targetId: string, data: Record<string, unknown>) =>
    api.patch<never, { message: string; target: SyncTrainingTarget }>(
      `/sync/tasks/${taskId}/targets/${targetId}`,
      data
    ),

  deleteTarget: (taskId: string, targetId: string) =>
    api.delete<never, { message: string }>(`/sync/tasks/${taskId}/targets/${targetId}`),

  triggerTargetTraining: (taskId: string, targetId: string) =>
    api.post<never, { message: string }>(
      `/sync/tasks/${taskId}/targets/${targetId}/trigger-training`
    ),

}

// External API Config
export const externalApiConfigApi = {
  create: (data: {
    config_name: string
    api_url: string
    auth_config: Record<string, unknown>
    description?: string
  }) =>
    api.post<never, { message: string; config: import('@/types').ExternalApiConfig }>(
      '/sync/api-configs',
      data
    ),

  list: () =>
    api.get<never, { configs: import('@/types').ExternalApiConfig[]; total: number }>(
      '/sync/api-configs'
    ),

  get: (configId: string) =>
    api.get<never, { config: import('@/types').ExternalApiConfig }>(
      `/sync/api-configs/${configId}`
    ),

  update: (
    configId: string,
    data: Partial<{
      config_name: string
      api_url: string
      auth_config: Record<string, unknown>
      description: string
      status: string
    }>
  ) =>
    api.patch<never, { message: string; config: import('@/types').ExternalApiConfig }>(
      `/sync/api-configs/${configId}`,
      data
    ),

  delete: (configId: string) =>
    api.delete<never, { message: string }>(`/sync/api-configs/${configId}`),

  testConnection: (data: { api_url: string; auth_config: Record<string, unknown> }) =>
    api.post<never, { success: boolean; message: string; status_code: number }>(
      '/sync/api-configs/test-connection',
      data
    ),

  testConnectionById: (configId: string) =>
    api.post<never, { success: boolean; message: string; status_code: number }>(
      `/sync/api-configs/${configId}/test-connection`
    ),
}

export interface AuthUser {
  user_id: string
  username: string
  email?: string | null
  is_active: boolean
  is_admin: boolean
  created_at?: string | null
  updated_at?: string | null
}

export interface AuthConfigResponse {
  self_registration_enabled: boolean
  direct_storage_registration_enabled: boolean
}

export interface AuthLoginRequest {
  username: string
  password: string
}

export interface AuthRegisterRequest {
  username: string
  password: string
  email?: string
}

export interface ChangePasswordRequest {
  old_password: string
  new_password: string
}

export interface AuthTokenResponse {
  access_token: string
  token_type: string
}

export interface AuthMessageResponse {
  message: string
}

export interface AdminUserListResponse {
  users: AuthUser[]
  total: number
  limit: number
  offset: number
}

export interface AdminUserCreateRequest {
  username: string
  password: string
  email?: string
  is_admin: boolean
}

export interface AdminUserFlagsRequest {
  is_active?: boolean
  is_admin?: boolean
}

export interface AdminPasswordResetRequest {
  new_password: string
}

export const authApi = {
  config: () => api.get<never, AuthConfigResponse>('/auth/config'),
  login: (data: AuthLoginRequest) => api.post<never, AuthTokenResponse>('/auth/login', data),
  register: (data: AuthRegisterRequest) =>
    api.post<never, AuthTokenResponse>('/auth/register', data),
  me: (config?: AppRequestConfig) => api.get<never, AuthUser>('/auth/me', config),
  logout: () =>
    api.post<never, AuthMessageResponse>('/auth/logout', undefined, LOGOUT_REQUEST_CONFIG),
  changePassword: (data: ChangePasswordRequest) =>
    api.post<never, AuthMessageResponse>('/auth/change-password', data),
}

export const adminUserApi = {
  list: (params: { limit: number; offset: number }) =>
    api.get<never, AdminUserListResponse>('/auth/admin/users', { params }),
  create: (data: AdminUserCreateRequest) => api.post<never, AuthUser>('/auth/admin/users', data),
  updateFlags: (userId: string, data: AdminUserFlagsRequest) =>
    api.patch<never, AuthUser>(`/auth/admin/users/${encodeURIComponent(userId)}`, data),
  resetPassword: (userId: string, data: AdminPasswordResetRequest) =>
    api.post<never, AuthMessageResponse>(
      `/auth/admin/users/${encodeURIComponent(userId)}/reset-password`,
      data
    ),
}

export default api
