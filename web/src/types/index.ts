// Training Task Types
export type TrainingStatus =
  | 'pending'
  | 'preparing'
  | 'running'
  | 'evaluating'
  | 'succeeded'
  | 'completed'
  | 'failed'
  | 'stopped'
  | 'cancelled'
export type ModelType = 'embedding' | 'reranker' | 'decoder_reranker' | 'llm'
export type TrainingMethod =
  | 'sft'
  | 'cpt'
  | 'dpo'
  | 'grpo'
  | 'dapo'
  | 'dr_grpo'
  | 'kto'
  | 'orpo'
  | 'ppo'
  | 'two_stage'
export type TunerType = 'lora' | 'full'

export interface TrainingTask {
  task_id: string
  task_name?: string
  description?: string
  model_type: ModelType
  training_method: TrainingMethod
  model_architecture?: 'encoder' | 'decoder'
  base_model_path: string
  train_dataset_path: string
  dataset_configs?: Array<{
    path: string
    max_samples?: number | null
    split: string
    num_rows?: number
  }>
  output_dir: string
  device?: string
  status: TrainingStatus
  progress: number
  current_step?: number
  total_steps?: number
  current_epoch?: number
  total_epochs?: number
  train_loss?: number
  eval_loss?: number
  learning_rate?: number
  gpu_ids?: number[]
  error_message?: string
  // LoRA & Checkpoints
  is_lora?: boolean
  checkpoints?: Record<string, unknown>
  best_checkpoint_path?: string
  loss_data?: string
  // Model Registry Link
  trained_model_registry_id?: string
  // RL & Two-Stage Training
  rl_config?: Record<string, unknown>
  loss_config?: Record<string, unknown>
  parent_task_id?: string
  sft_checkpoint_path?: string
  // Training Results
  training_params?: Record<string, unknown>
  final_metrics?: Record<string, unknown>
  final_model_path?: string
  // Timestamps
  created_at: string
  updated_at: string
  started_at?: string
  completed_at?: string
}

export interface TrainingTaskEvent {
  event_id: string
  task_id: string
  user_id?: string
  event_type: string
  payload?: Record<string, unknown>
  created_at?: string
}

// Training dataset configuration
export interface TrainingDatasetConfig {
  path: string
  max_samples?: number
  split: 'train' | 'eval' | 'test'
}

export interface CreateTrainingRequest {
  task_name?: string
  description?: string
  model_type: ModelType
  training_method: TrainingMethod
  model_architecture?: 'encoder' | 'decoder'
  base_model_path: string
  datasets: TrainingDatasetConfig[]
  output_dir?: string
  // Training parameters
  num_train_epochs?: number
  per_device_train_batch_size?: number
  learning_rate?: number
  warmup_ratio?: number
  gradient_accumulation_steps?: number
  gradient_checkpointing?: boolean
  logging_steps?: number
  max_length?: number
  eval_strategy?: 'no' | 'epoch' | 'steps'
  eval_steps?: number
  save_strategy?: 'no' | 'epoch' | 'steps'
  save_steps?: number
  bf16?: boolean
  fp16?: boolean
  embedding_loss_name?: string
  reranker_loss_name?: string
  // LoRA parameters
  tuner_type?: TunerType
  use_lora?: boolean
  lora_r?: number
  lora_alpha?: number
  lora_dropout?: number
  gpu_ids?: number[]
  // Decoder Reranker loss config
  loss_config?: {
    name?: string
    n_docs?: number
    n_pos?: number
    temperature?: number
    chunk_size?: number
    metric?: string
    k?: number
    num_negatives?: number
    infonce_mode?: 'single' | 'posset' | 'avgpos'
    ranknet_max_pairs_per_batch?: number
    scale?: number
    mini_batch_size?: number
    respect_input_order?: boolean
    margin?: number
    positive_margin?: number
    negative_margin?: number
    distance_metric?: 'cosine' | 'euclidean' | string
    triplet_margin?: number
    num_labels?: number
    guide_model?: string
    normalize_by_logc?: boolean
    ignore_empty?: boolean
    matryoshka_dims?: Array<number | string>
    beta?: number
    delta?: number
    sigma?: number
  }
  // RL training config
  rl_config?: {
    kl_coef?: number
    clip_range?: number
    loss_type?: string
    beta?: number
    rankings_direction?: 'auto' | 'higher_is_better' | 'lower_is_better'
    reward_model?: string
    reward_type?: 'rank_based' | 'score_based' | 'ndcg_based' | 'recall_based' | string
    reward_k?: number
    scale_rewards?: boolean
    num_iterations?: number
    chunk_size?: number
    n_docs?: number
    reference_free?: boolean
  }
  // DeepSpeed
  deepspeed?: 'zero2' | 'zero3' | 'zero2_offload' | 'zero3_offload'
  // Two-stage training
  parent_task_id?: string
  sft_checkpoint_path?: string
}

// Dataset Types
export type DatasetType =
  // Embedding types (10 types)
  | 'embedding_universal' // (query, pos[], neg[], pos_scores[], neg_scores[]) - 通用格式
  | 'embedding_pair' // (query, positive) - MNR对
  | 'embedding_triplet' // (query, positive, negative) - MNR三元组
  | 'embedding_multi_neg' // (query, pos, neg[]) - MNR多负例(数量一致)
  | 'embedding_dynamic_neg' // (query, pos, neg[]) - 动态负例(数量可变)
  | 'embedding_cosine' // (text1, text2, label) - 余弦相似度
  | 'embedding_margin' // (anchor, pos, neg, margin) - Margin三元组
  | 'embedding_margin_multi' // (anchor, pos[], neg[], margins[]) - Margin多边界
  | 'embedding_scored' // (anchor, pos[], neg[], scores[]) - Margin多分数
  | 'embedding_score_triplet' // (query, pos, neg, [pos_score, neg_score]) - Score三元组
  // Rerank types
  | 'rerank_pair'
  | 'rerank_triplet'
  | 'rerank_listwise'
  // LLM types
  | 'sft_instruct'
  | 'dpo_preference'
  | 'rl_reward'
  // Generation intermediate types
  | 'qa_pair'
  | 'custom'

export type DatasetUsage = 'raw' | 'train' | 'eval' | 'test'
export type DatasetModelType = 'embedding' | 'rerank' | 'llm'
export type DatasetSourceType = 'uploaded' | 'huggingface' | 'modelscope' | 'local'
export type DatasetStatus =
  | 'registered'
  | 'uploading'
  | 'downloading'
  | 'processing'
  | 'ready'
  | 'error'
  | 'archived'

export interface DatasetColumn {
  name: string
  type: string
}

export interface Dataset {
  dataset_id: string
  name: string // Alias for dataset_name
  dataset_name: string
  display_name?: string
  description?: string
  dataset_type: DatasetType | string
  usage?: DatasetUsage
  model_type?: DatasetModelType[]
  // Source Info
  source_type: DatasetSourceType
  source_path?: string
  source_dataset_id?: string
  remote_repo?: string
  hf_subset?: string
  // Storage
  storage_path?: string | null
  storage_backend?: 'local' | 's3'
  storage_uri?: string | null
  version?: number
  file_format: string
  // Schema
  columns?: DatasetColumn[] | string[]
  sample_data?: Record<string, unknown>[]
  content_schema?: Record<string, unknown> | null
  // Statistics
  num_rows?: number
  num_train?: number
  num_eval?: number
  num_test?: number
  file_size?: number
  size_bytes?: number // Alias for file_size
  num_samples?: number // Alias for num_rows
  // Provenance
  source_task_type?: string | null
  source_task_id?: string | null
  // Metadata
  tags?: string[]
  extra_metadata?: Record<string, unknown>
  // Status
  status: DatasetStatus
  error_message?: string
  // User
  user_id?: string
  // Timestamps
  created_at: string
  updated_at: string
}

export interface CreateDatasetRequest {
  dataset_name: string
  storage_path?: string
  storage_backend?: 'local' | 's3'
  storage_uri?: string
  dataset_type?: string
  usage?: DatasetUsage
  model_type?: DatasetModelType[]
  source_type?: string
  display_name?: string
  description?: string
  source_path?: string
  remote_repo?: string
  hf_subset?: string
  file_format?: string
  columns?: DatasetColumn[]
  num_rows?: number
  num_train?: number
  num_eval?: number
  num_test?: number
  source_task_type?: string
  source_task_id?: string
  tags?: string[]
  extra_metadata?: Record<string, unknown>
}

export interface UpdateDatasetRequest extends Partial<CreateDatasetRequest> {
  status?: DatasetStatus
}

// Model Registry Types
export type ModelStatus = 'registered' | 'available' | 'archived'

export interface RegisteredModel {
  model_id: string
  model_name: string
  display_name?: string
  version: string
  model_type: ModelType
  // Source info
  source_task_id?: string
  source_model_id?: string
  base_model_path?: string
  model_path: string
  best_checkpoint_path?: string
  checkpoints?: Record<string, unknown>
  // Model properties
  embedding_dim?: number
  source_type?: string
  is_adapter?: boolean
  // Download info
  download_source?: string
  remote_repo?: string
  download_status?: string
  download_progress?: number
  download_error?: string
  // Metadata
  description?: string
  tags?: string[]
  category?: string
  extra_metadata?: Record<string, unknown>
  // Performance
  metrics?: Record<string, number>
  file_size?: number
  // Status
  status: ModelStatus
  is_latest?: boolean
  // User
  user_id?: string
  // Timestamps
  created_at: string
  updated_at: string
}

// Deployment Types
export type DeploymentStatus =
  | 'pending'
  | 'starting'
  | 'running'
  | 'restarting'
  | 'stopping'
  | 'stopped'
  | 'degraded'
  | 'failed'
export type InferenceFramework = 'xinference' | 'vllm' | 'sglang'
export type HealthStatus = 'HEALTHY' | 'UNHEALTHY' | 'UNKNOWN'

export interface ReplicaGpuOverride {
  replica_index: number
  gpu_ids: number[]
}

export interface CommonLaunchConfig {
  tensor_parallel_size: number
  pipeline_parallel_size: number
  data_parallel_size: number
  max_context_length?: number | null
  max_concurrent_requests?: number | null
  dtype: 'auto' | 'half' | 'float16' | 'bfloat16' | 'float' | 'float32'
  quantization?: string | null
  kv_cache_dtype: string
  gpu_pool: number[]
  replica_gpu_overrides: ReplicaGpuOverride[]
  allow_gpu_reuse: boolean
}

export interface VllmLaunchConfig extends CommonLaunchConfig {
  framework: 'vllm'
  enable_expert_parallel: boolean
  enforce_eager: boolean
}

export interface SglangLaunchConfig extends CommonLaunchConfig {
  framework: 'sglang'
  expert_parallel_size: number
  attention_backend?: string | null
}

export type DeploymentLaunchConfig = VllmLaunchConfig | SglangLaunchConfig

export interface DeploymentReplica {
  replica_id: string
  deployment_id: string
  replica_index: number
  endpoint: string
  port: number
  gpu_ids: number[]
  status: DeploymentStatus
  health_status: HealthStatus
  error_message?: string | null
  created_at?: string
  updated_at?: string
  started_at?: string | null
  stopped_at?: string | null
}

export interface DeploymentStats {
  total: number
  by_status: Record<string, number>
}

export interface Deployment {
  deployment_id: string
  model_id: string
  model_uid?: string
  deployment_name?: string
  xinference_endpoint: string
  replica: number
  replica_instances?: DeploymentReplica[]
  launch_config?: DeploymentLaunchConfig | null
  gpu_memory_utilization: number // Configured limit
  gpu_memory_used_mb?: number // Real-time usage in MB
  gpu_memory_used_percent?: number // Real-time usage percentage
  deploy_mode: 'shared' | 'container'
  container_name?: string
  gpu_id?: number
  port?: number
  inference_framework: InferenceFramework
  enable_lora: boolean
  max_loras: number
  max_lora_rank: number
  external_api_config_id?: string | null
  config?: Record<string, unknown>
  runtime_info?: Record<string, unknown>
  // Health Check
  health_check_url?: string
  last_health_check?: string
  health_status?: HealthStatus
  // Status
  status: DeploymentStatus
  error_message?: string
  user_id?: string
  created_at: string
  updated_at: string
  started_at?: string
  stopped_at?: string
}

export interface CreateDeploymentRequest {
  model_id: string
  xinference_endpoint?: string
  deployment_name?: string
  gpu_id?: number
  replica?: number
  gpu_memory_utilization?: number
  inference_framework?: InferenceFramework
  enable_lora?: boolean
  max_loras?: number
  max_lora_rank?: number
  auto_start?: boolean
  external_api_config_id?: string
  config?: Record<string, unknown>
  launch_config?: DeploymentLaunchConfig
}

export interface CreateContainerDeploymentRequest {
  model_id: string
  deployment_name?: string
  gpu_id?: number
  port?: number
  replica?: number
  gpu_memory_utilization?: number
  inference_framework?: InferenceFramework
  enable_lora?: boolean
  max_loras?: number
  max_lora_rank?: number
  external_api_config_id?: string
  config?: Record<string, unknown>
  launch_config?: DeploymentLaunchConfig
  auto_start?: boolean
}

export interface RestartDeploymentRequest {
  mode?: 'auto' | 'model' | 'container'
  reset_gpu?: boolean
}

export interface UpdateDeploymentConfigRequest {
  config?: Record<string, unknown> | null
}

// Loaded Adapter Types
export interface LoadedAdapter {
  adapter_id: string
  deployment_id: string
  deployment_replica_id?: string
  adapter_name: string
  adapter_path: string
  source_task_id?: string
  source_model_id?: string
  status: 'loading' | 'loaded' | 'unloading' | 'unloaded' | 'failed'
  error_message?: string
  loaded_at: string
  unloaded_at?: string
}

export interface AvailableAdapter {
  source: 'training_task' | 'model_registry'
  source_id: string
  name: string
  path: string
  base_model_id?: string
  created_at?: string
  description?: string
}

// Model Config Types
export type ModelConfigSourceType = 'external_api' | 'local_deployed'

export interface ModelConfig {
  config_id: string
  config_name: string
  display_name?: string
  // Source type and links
  source_type?: ModelConfigSourceType
  registry_id?: string
  deployment_id?: string
  inference_framework?: InferenceFramework
  container_name?: string
  // Model info
  model_type: string
  provider: string
  api_endpoint: string
  api_key?: string
  model_name?: string
  // Model properties
  embedding_dim?: number
  max_input_length?: number
  // Config
  provider_config?: Record<string, unknown>
  default_params?: Record<string, unknown>
  // Metadata
  description?: string
  tags?: string[]
  // Status
  status: string
  is_default?: boolean
  // Health check
  last_check_status?: string
  last_check_time?: string
  last_check_error?: string
  // User
  user_id?: string
  // Timestamps
  created_at: string
  updated_at: string
}

export interface CreateModelConfigRequest {
  config_name: string
  model_type: string
  provider: string
  api_endpoint: string
  api_key?: string
  model_name?: string
  description?: string
}

// GPU Info Types
export interface GpuInfo {
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
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean
  data?: T
  message?: string
  error?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

// Evaluation Task Types
export type EvaluationTaskStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface EvaluationModelConfig {
  model_id?: string
  deployment_id?: string
  deployment_replica_id?: string
  endpoint: string
  model_name?: string
  name?: string
  gpu_id?: number
  inference_framework?: string
}

export interface EvaluationDatasetConfig {
  type: 'mteb' | 'local' | 'registered'
  name: string
  path?: string
  dataset_id?: string
  result_key?: string
}

export interface EvaluationIdentityMap {
  models: Record<string, { name: string }>
  datasets: Record<
    string,
    { name: string; type: EvaluationDatasetConfig['type'] }
  >
}

export type ModelDatasetProgressStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface ModelDatasetProgress {
  progress: number
  status: ModelDatasetProgressStatus
}

// model_progress: {model_name: {dataset_name: {progress, status}}}
export type ModelProgressMatrix = Record<string, Record<string, ModelDatasetProgress>>

export interface EvaluationTask {
  task_id: string
  task_name?: string
  description?: string
  eval_type: string
  model_configs?: EvaluationModelConfig[]
  dataset_configs?: EvaluationDatasetConfig[]
  identity_schema_version?: number
  identity_map?: EvaluationIdentityMap
  max_samples?: number
  batch_size: number
  workers: number
  model_workers: number
  status: EvaluationTaskStatus
  progress: number
  current_model?: string
  current_dataset?: string
  model_progress?: ModelProgressMatrix
  results?: Record<string, Record<string, Record<string, number> | { error: string }>>
  report_path?: string
  error_message?: string
  user_id?: string
  created_at: string
  updated_at?: string
  started_at?: string
  completed_at?: string
}

export interface CreateEvaluationRequest {
  task_name?: string
  model_configs: EvaluationModelConfig[]
  dataset_configs: EvaluationDatasetConfig[]
  max_samples?: number
  batch_size?: number
  workers?: number
  model_workers?: number
}

// Deep Evaluation Task Types (for embedding/reranker model evaluation)
export type DeepEvaluationTaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
export type DeepEvalType = 'embedding' | 'rerank' | 'llm' | 'multi'

export interface DeepEvaluationFieldMapping {
  query: string
  positives?: string
  negatives?: string
  input?: string
  expected_output?: string
  actual_output?: string
  retrieval_context?: string
}

export interface DeepEvaluationDatasetConfig {
  dataset_id: string
  dataset_name?: string
}

// 模型组内的单个模型配置
export interface DeepEvalModelInGroupConfig {
  config_id?: string
  config_name?: string
  endpoint?: string
  model_name?: string
  api_key?: string
  inference_framework?: string
  concurrency?: number
}

// 模型组内的 LLM 配置
export interface DeepEvalLLMInGroupConfig {
  config_id?: string
  config_name?: string
  endpoint?: string
  model?: string
  api_key?: string
  temperature?: number
  top_p?: number
  top_k?: number
  max_tokens?: number
  timeout?: number
  max_retries?: number
  concurrency?: number
}

// 模型组配置（一个组包含 embedding/rerank/llm 三类模型）
export interface DeepEvaluationModelGroupConfig {
  group_name: string
  embedding?: DeepEvalModelInGroupConfig
  rerank?: DeepEvalModelInGroupConfig
  llm?: DeepEvalLLMInGroupConfig
}

export interface DeepEvaluationTask {
  task_id: string
  task_name?: string
  description?: string
  eval_type: DeepEvalType
  model_configs: DeepEvaluationModelGroupConfig[]
  dataset_configs: DeepEvaluationDatasetConfig[]
  max_samples?: number
  field_mapping?: DeepEvaluationFieldMapping
  metrics: string[]
  worker_groups?: Record<string, unknown>
  status: DeepEvaluationTaskStatus
  progress: number
  total_samples: number
  processed_samples: number
  // model_progress: {model_name: {dataset_name: {progress, status}}}
  model_progress?: ModelProgressMatrix
  results_summary?: Record<string, unknown>
  results_path?: string
  error_message?: string
  user_id?: string
  created_at?: string
  started_at?: string
  completed_at?: string
}

export interface MTEBDatasetInfo {
  split: string
  lang: string
  description: string
}

export interface AvailableDatasets {
  datasets: Record<string, MTEBDatasetInfo>
  groups: Record<string, string[]>
}

// Milvus Vector Database Types
export interface LinkedDatasetInfo {
  dataset_id: string
  dataset_name: string | null
  chunk_count: number
  task_id: string | null
}

export type MilvusSearchMode = 'dense' | 'sparse' | 'hybrid'
export type MilvusRankerType = 'rrf' | 'weighted'

export interface MilvusCollectionSummary {
  name: string
  num_entities: number
  dim: number | null
  index_type: string | null
  metric_type: string | null
  load_state: 'Loaded' | 'NotLoad' | 'Loading'
  description: string | null
  hybrid_enabled: boolean
  metadata_enabled: boolean
  // Registry fields
  collection_id: string | null
  display_name: string | null
  embedding_config_id: string | null
  embedding_model: string | null
  embedding_endpoint: string | null
  linked_datasets: LinkedDatasetInfo[]
  status: string | null
  // Backward compat
  source_dataset_id: string | null
  source_dataset_name: string | null
  associated_task_id: string | null
}

export interface MilvusCollectionDetail extends MilvusCollectionSummary {
  schema_fields: Array<{
    name: string
    dtype: string
    is_primary: boolean
    max_length?: number | null
    dim?: number | null
  }>
  indexes: Array<{
    field_name: string
    index_type: string
    metric_type: string
    params: Record<string, unknown>
  }>
}

export interface MilvusSearchResult {
  chunk_id: string
  chunk_content: string
  score: number
}

// External API Config
export interface ExternalApiConfig {
  config_id: string
  config_name: string
  user_id: string
  api_url: string
  auth_config: Record<string, unknown>
  description: string
  status: string
  created_at: string
  updated_at: string
}

// External Sync Types
export type SyncStatus =
  | 'idle'
  | 'syncing'
  | 'generating'
  | 'training'
  | 'loading_adapter'
  | 'error'
export type SyncBatchStatus = 'registered' | 'fetched' | 'generation_queued' | 'generation_done'
export type SyncGenerationStatus = 'pending' | 'completed' | 'failed'
export type SyncTrainingStatus =
  | 'pending'
  | 'completed'
  | 'failed'
  | 'adapter_loaded'
  | 'adapter_unloaded'
  | 'adapter_load_failed'
  | 'adapter_failed'

export interface SyncConfig {
  task_id: string
  task_name: string
  user_id: string
  external_api_config_id: string | null
  external_api_url: string
  external_auth_config: Record<string, unknown>
  sync_interval_seconds: number
  last_sync_at: string | null
  generation_threshold: number
  generation_mode: string
  generation_config: Record<string, unknown> | null
  training_threshold: number
  training_config: Record<string, unknown> | null
  base_deployment_id: string | null
  base_deployment_replica_id: string | null
  pending_record_count: number
  pending_training_samples: number
  total_record_count: number
  total_training_samples: number
  total_trainings: number
  current_adapter_name: string | null
  current_adapter_id: string | null
  current_training_id: string | null
  is_active: boolean
  status: SyncStatus
  error_message: string | null
  training_targets?: SyncTrainingTarget[]
  created_at: string
  updated_at: string
}

export interface SyncTrainingTarget {
  target_id: string
  task_id: string
  target_name: string
  model_type: string
  data_phase: 'qa' | 'final'
  training_method: string
  training_config: Record<string, unknown>
  base_model_path: string
  base_deployment_id: string | null
  base_deployment_replica_id: string | null
  training_threshold: number
  pending_training_samples: number
  total_training_samples: number
  total_trainings: number
  current_adapter_name: string | null
  current_adapter_id: string | null
  current_training_id: string | null
  priority: number
  status: string
  is_active: boolean
  sort_order: number
  created_at: string
  updated_at: string
}

export interface CreateSyncTrainingTargetRequest {
  target_name: string
  model_type: string
  data_phase: 'qa' | 'final'
  training_method: string
  training_config: Record<string, unknown>
  base_model_path: string
  base_deployment_id?: string | null
  base_deployment_replica_id?: string | null
  training_threshold: number
  priority: number
  sort_order: number
}

export interface ReplaceSyncTrainingTargetRequest
  extends CreateSyncTrainingTargetRequest {
  target_id: string
}

export interface SyncBatch {
  batch_id: string
  task_id: string
  user_id: string
  record_count: number
  storage_path: string
  dataset_id: string | null
  since_time: string | null
  until_time: string | null
  fetched_at: string
  status: SyncBatchStatus
  generation_task_id: string | null
}

export interface SyncGeneration {
  task_id: string
  generation_task_id: string
  user_id: string
  input_batch_ids: string[]
  input_record_count: number
  output_dataset_id: string | null
  output_sample_count: number
  status: SyncGenerationStatus
  created_at: string
  completed_at: string | null
  disabled: boolean
  reset_batch_count?: number
  reset_record_count?: number
}

export interface SyncTraining {
  task_id: string
  training_task_id: string
  user_id: string
  input_dataset_ids: string[]
  total_samples: number
  training_round: number
  output_adapter_path: string | null
  loaded_adapter_name: string | null
  loaded_adapter_id: string | null
  previous_training_task_id: string | null
  target_id?: string
  status: SyncTrainingStatus
  created_at: string
  completed_at: string | null
}

export interface CreateSyncConfigRequest {
  task_name: string
  external_api_config_id?: string
  external_api_url?: string
  external_auth_config?: { token: string }
  sync_interval_seconds?: number
  generation_threshold?: number
  generation_mode?: string
  generation_config?: Record<string, unknown> | null
  training_threshold?: number
  training_config?: Record<string, unknown> | null
  base_deployment_id?: string | null
  base_deployment_replica_id?: string | null
  training_targets?: CreateSyncTrainingTargetRequest[]
  is_active?: boolean
}

export type UpdateSyncConfigRequest = Partial<
  Omit<CreateSyncConfigRequest, 'training_targets'>
> & {
  training_targets?: ReplaceSyncTrainingTargetRequest[]
}

export interface SyncStatusInfo {
  task_id: string
  status: SyncStatus
  is_active: boolean
  worker_status: string
  pending_record_count: number
  generation_threshold: number
  pending_training_samples: number
  training_threshold: number
  total_record_count: number
  total_training_samples: number
  total_trainings: number
  current_adapter_name: string | null
  last_sync_at: string | null
}
