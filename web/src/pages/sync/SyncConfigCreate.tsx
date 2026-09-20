import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Card,
  Row,
  Col,
  Space,
  message,
  Typography,
  Divider,
  Switch,
  Modal,
  Tabs,
  Collapse,
  Spin,
  Tag,
} from 'antd'
import type { FormInstance } from 'antd'
import {
  ArrowLeftOutlined,
  PlusOutlined,
  ApiOutlined,
  ImportOutlined,
  DeleteOutlined,
} from '@ant-design/icons'
import {
  syncApi,
  deploymentApi,
  modelApi,
  configApi,
  externalApiConfigApi,
  generationApi,
  trainingApi,
} from '@/services/api'
import ModelConfigSelector from '@/components/ModelConfigSelector'
import { GpuSelect } from '@/components/GpuSelect'
import TrainingTargetForm, {
  TrainingTargetFormState,
  DEFAULT_TARGET,
} from '@/components/sync/TrainingTargetForm'
import type {
  Deployment,
  ModelConfig,
  RegisteredModel,
  ExternalApiConfig,
  TrainingTask,
  SyncTrainingTarget,
  CreateSyncConfigRequest,
  CreateSyncTrainingTargetRequest,
  UpdateSyncConfigRequest,
} from '@/types'
import { TEXT_SECONDARY, BG_ELEVATED } from '@/theme'
import './SyncConfigCreate.css'
import {
  getAutomaticHealthyReplicaId,
  getHealthyDeploymentReplicas,
  isHealthyDeploymentReplicaBinding,
  isSelectableSyncDeployment,
} from './deploymentReplicaPolicy'

const { Title, Text } = Typography

// Sync 流程后端仅支持 SFT（RL 训练需要 rl_config / sft_checkpoint_path，level2_handler 不处理）
const trainingMethodOptions: Record<string, { label: string; value: string }[]> = {
  llm: [{ label: 'SFT', value: 'sft' }],
  embedding: [{ label: 'SFT', value: 'sft' }],
  reranker: [{ label: 'SFT', value: 'sft' }],
  decoder_reranker: [{ label: 'SFT', value: 'sft' }],
}

const embeddingLossOptions = [
  { label: 'DynamicExplicitNegativesRankingLoss', value: 'DynamicExplicitNegativesRankingLoss' },
  { label: 'MultipleNegativesRankingLoss', value: 'MultipleNegativesRankingLoss' },
  { label: 'CachedMultipleNegativesRankingLoss', value: 'CachedMultipleNegativesRankingLoss' },
  { label: 'ContrastiveLoss', value: 'ContrastiveLoss' },
  { label: 'OnlineContrastiveLoss', value: 'OnlineContrastiveLoss' },
  { label: 'MegaBatchMarginLoss', value: 'MegaBatchMarginLoss' },
  { label: 'CosineSimilarityLoss', value: 'CosineSimilarityLoss' },
  { label: 'CoSENTLoss', value: 'CoSENTLoss' },
  { label: 'AnglELoss', value: 'AnglELoss' },
  { label: 'TripletLoss', value: 'TripletLoss' },
  { label: 'BatchHardTripletLoss', value: 'BatchHardTripletLoss' },
  { label: 'BatchSemiHardTripletLoss', value: 'BatchSemiHardTripletLoss' },
  { label: 'BatchAllTripletLoss', value: 'BatchAllTripletLoss' },
  { label: 'SoftmaxLoss', value: 'SoftmaxLoss' },
  { label: 'DistillKLDivLoss', value: 'DistillKLDivLoss' },
  { label: 'MSELoss', value: 'MSELoss' },
  { label: 'GISTEmbedLoss', value: 'GISTEmbedLoss' },
  { label: 'DynamicExplicitNegativesRankingLoss', value: 'DynamicExplicitNegativesRankingLoss' },
  { label: 'MatryoshkaLoss', value: 'MatryoshkaLoss' },
  { label: 'Matryoshka2dLoss', value: 'Matryoshka2dLoss' },
]

const rerankerLossOptions = [
  { label: 'Auto', value: 'auto' },
  { label: 'CrossEntropyLoss', value: 'CrossEntropyLoss' },
  { label: 'BinaryCrossEntropyLoss (BCEWithLogits)', value: 'BCEWithLogitsLoss' },
  { label: 'MSELoss', value: 'MSELoss' },
  { label: 'MarginMSELoss', value: 'MarginMSELoss' },
  { label: 'MultipleNegativesRankingLoss', value: 'MultipleNegativesRankingLoss' },
  { label: 'CachedMultipleNegativesRankingLoss', value: 'CachedMultipleNegativesRankingLoss' },
  { label: 'RankNetLoss', value: 'RankNetLoss' },
  { label: 'LambdaLoss', value: 'LambdaLoss' },
  { label: 'ListMLELoss', value: 'ListMLELoss' },
  { label: 'ListNetLoss', value: 'ListNetLoss' },
  { label: 'PListMLELoss', value: 'PListMLELoss' },
]

const rerankerScaleLosses = ['MultipleNegativesRankingLoss', 'CachedMultipleNegativesRankingLoss']
const rerankerMiniBatchLosses = [
  'CachedMultipleNegativesRankingLoss',
  'RankNetLoss',
  'LambdaLoss',
  'ListMLELoss',
  'ListNetLoss',
  'PListMLELoss',
]
const rerankerNegativesLosses = [
  'MultipleNegativesRankingLoss',
  'CachedMultipleNegativesRankingLoss',
]
const rerankerTopKLosses = ['RankNetLoss', 'LambdaLoss']
const rerankerSigmaLosses = ['RankNetLoss', 'LambdaLoss']
const rerankerRespectInputOrderLosses = ['ListMLELoss', 'PListMLELoss']

const decoderRerankerLossOptions = [
  { label: 'InfoNCE', value: 'infonce' },
  { label: 'BCE', value: 'bce' },
  { label: 'LambdaLoss', value: 'lambda_loss' },
  { label: 'ListMLE', value: 'list_mle' },
  { label: 'RankNet', value: 'ranknet' },
]

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildStepsConfig(values: Record<string, any>) {
  const steps: Record<string, unknown> = {}
  if (values.doc_quality_enabled) {
    steps.doc_quality = { enabled: true, min_score: values.doc_quality_min_score ?? 0.5 }
  }
  if (values.keypoint_gen_enabled) {
    steps.keypoint_gen = { enabled: true, max_keypoints: values.keypoint_gen_max ?? 5 }
  }
  if (values.qa_gen_enabled) {
    steps.qa_gen = { enabled: true, num_qa_per_doc: values.qa_gen_count ?? 3 }
  }
  if (values.pos_neg_enabled) {
    steps.pos_neg_extraction = {
      enabled: true,
      num_positive: values.pos_neg_positive_count ?? 3,
      num_negative: values.pos_neg_negative_count ?? 7,
    }
  }
  if (values.validation_enabled) {
    steps.validation = { enabled: true }
  }
  return steps
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function populateGenerationFields(genCfg: Record<string, any>, form: FormInstance) {
  const llmCfg = genCfg.llm_config as Record<string, unknown> | undefined
  if (llmCfg) {
    form.setFieldValue(['llm_config', 'config_id'], llmCfg.config_id)
    if (llmCfg.concurrency != null) form.setFieldValue('llm_concurrency', llmCfg.concurrency)
  }
  const embCfg = genCfg.embedding_config as Record<string, unknown> | undefined
  if (embCfg) form.setFieldValue(['embedding_config', 'config_id'], embCfg.config_id)
  const rrCfg = genCfg.rerank_config as Record<string, unknown> | undefined
  if (rrCfg) form.setFieldValue(['rerank_config', 'config_id'], rrCfg.config_id)

  if (genCfg.pos_neg_method) form.setFieldValue('pos_neg_method', genCfg.pos_neg_method)
  if (genCfg.output_format) form.setFieldValue('output_format', genCfg.output_format)

  const wc = genCfg.worker_config as Record<string, unknown> | undefined
  if (wc?.timeout_per_doc != null) form.setFieldValue('timeout_per_doc', wc.timeout_per_doc)

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sc = genCfg.steps_config as Record<string, any> | undefined
  if (sc) {
    if (sc.doc_quality) {
      form.setFieldValue('doc_quality_enabled', !!sc.doc_quality.enabled)
      if (sc.doc_quality.min_score != null)
        form.setFieldValue('doc_quality_min_score', sc.doc_quality.min_score)
    }
    if (sc.keypoint_gen) {
      form.setFieldValue('keypoint_gen_enabled', !!sc.keypoint_gen.enabled)
      if (sc.keypoint_gen.max_keypoints != null)
        form.setFieldValue('keypoint_gen_max', sc.keypoint_gen.max_keypoints)
    }
    if (sc.qa_gen) {
      form.setFieldValue('qa_gen_enabled', !!sc.qa_gen.enabled)
      if (sc.qa_gen.num_qa_per_doc != null)
        form.setFieldValue('qa_gen_count', sc.qa_gen.num_qa_per_doc)
    }
    if (sc.pos_neg_extraction) {
      form.setFieldValue('pos_neg_enabled', !!sc.pos_neg_extraction.enabled)
      if (sc.pos_neg_extraction.num_positive != null)
        form.setFieldValue('pos_neg_positive_count', sc.pos_neg_extraction.num_positive)
      if (sc.pos_neg_extraction.num_negative != null)
        form.setFieldValue('pos_neg_negative_count', sc.pos_neg_extraction.num_negative)
    }
    if (sc.validation) form.setFieldValue('validation_enabled', !!sc.validation.enabled)
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pp = genCfg.post_process_config as Record<string, any> | undefined
  if (pp?.dedup) form.setFieldValue('dedup_enabled', !!pp.dedup.enabled)
}

function populateTrainingFields(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  trainCfg: Record<string, any>,
  form: FormInstance,
  registeredModels: RegisteredModel[],
  setTrainingModelId: (v: string) => void,
  setTrainingModelName: (v: string) => void,
  setTrainingModelPath: (v: string) => void,
  setTrainingModelType: (v: string) => void
) {
  // Match base_model_path to registered model
  if (trainCfg.base_model_path) {
    const match = registeredModels.find(
      (m) => (m.base_model_path || m.model_path) === trainCfg.base_model_path
    )
    if (match) {
      setTrainingModelId(match.model_id)
      setTrainingModelName(match.display_name || match.model_name || '')
      setTrainingModelPath(match.base_model_path || match.model_path || '')
    } else {
      // No registered model matched — still set path for display
      setTrainingModelId('')
      setTrainingModelName('')
      setTrainingModelPath(trainCfg.base_model_path)
    }
  }

  if (trainCfg.model_type) {
    form.setFieldValue('model_type', trainCfg.model_type)
    setTrainingModelType(trainCfg.model_type)
  }
  if (trainCfg.training_method) form.setFieldValue('training_method', trainCfg.training_method)
  if (trainCfg.lora_r != null) form.setFieldValue('lora_r', trainCfg.lora_r)
  if (trainCfg.lora_alpha != null) form.setFieldValue('lora_alpha', trainCfg.lora_alpha)
  if (trainCfg.lora_dropout != null) form.setFieldValue('lora_dropout', trainCfg.lora_dropout)
  if (trainCfg.num_train_epochs != null)
    form.setFieldValue('num_train_epochs', trainCfg.num_train_epochs)
  if (trainCfg.per_device_train_batch_size != null)
    form.setFieldValue('per_device_train_batch_size', trainCfg.per_device_train_batch_size)
  if (trainCfg.learning_rate != null) form.setFieldValue('learning_rate', trainCfg.learning_rate)
  if (trainCfg.warmup_ratio != null) form.setFieldValue('warmup_ratio', trainCfg.warmup_ratio)
  if (trainCfg.gradient_accumulation_steps != null)
    form.setFieldValue('gradient_accumulation_steps', trainCfg.gradient_accumulation_steps)
  if (trainCfg.max_length != null) form.setFieldValue('max_length', trainCfg.max_length)
  if (trainCfg.gpu_ids) form.setFieldValue('gpu_ids', trainCfg.gpu_ids)

  // Mixed precision
  if (trainCfg.bf16 === true) form.setFieldValue('mixed_precision', 'bf16')
  else if (trainCfg.fp16 === true) form.setFieldValue('mixed_precision', 'fp16')

  // Loss config
  if (trainCfg.embedding_loss_name)
    form.setFieldValue('embedding_loss_name', trainCfg.embedding_loss_name)
  if (trainCfg.reranker_loss_name)
    form.setFieldValue('reranker_loss_name', trainCfg.reranker_loss_name)
  if (trainCfg.loss_config) form.setFieldValue('loss_config', trainCfg.loss_config)
}

function buildTrainingTargetPayload(
  target: TrainingTargetFormState
): CreateSyncTrainingTargetRequest {
  const trainingConfig: Record<string, unknown> = {
    lora_r: target.lora_r,
    lora_alpha: target.lora_alpha,
    lora_dropout: target.lora_dropout,
    num_train_epochs: target.num_train_epochs,
    per_device_train_batch_size: target.per_device_train_batch_size,
    learning_rate: target.learning_rate,
    warmup_ratio: target.warmup_ratio,
    gradient_accumulation_steps: target.gradient_accumulation_steps,
  }
  if (target.mixed_precision === 'bf16') trainingConfig.bf16 = true
  if (target.mixed_precision === 'fp16') trainingConfig.fp16 = true
  if (target.max_length) trainingConfig.max_length = target.max_length
  if (target.gpu_ids?.length) trainingConfig.gpu_ids = target.gpu_ids
  if (target.embedding_loss_name) trainingConfig.embedding_loss_name = target.embedding_loss_name
  if (target.reranker_loss_name && target.reranker_loss_name !== 'auto') {
    trainingConfig.reranker_loss_name = target.reranker_loss_name
  }
  if (target.loss_config) trainingConfig.loss_config = target.loss_config
  if (target.rl_config) trainingConfig.rl_config = target.rl_config

  return {
    target_name: target.target_name.trim(),
    model_type: target.model_type,
    data_phase: target.data_phase,
    training_method: target.training_method,
    base_model_path: target.base_model_path.trim(),
    base_deployment_id: target.base_deployment_id || null,
    base_deployment_replica_id: target.base_deployment_replica_id || null,
    training_threshold: target.training_threshold,
    priority: target.priority,
    sort_order: target.sort_order,
    training_config: trainingConfig,
  }
}

export default function SyncConfigCreate() {
  const { taskId } = useParams<{ taskId: string }>()
  const isEditMode = !!taskId
  const navigate = useNavigate()
  const { t } = useTranslation(['sync', 'common'])
  const [form] = Form.useForm()
  const watchedBaseDeploymentId = Form.useWatch('base_deployment_id', form)
  const watchedBaseDeploymentReplicaId = Form.useWatch('base_deployment_replica_id', form)
  const [loading, setLoading] = useState(false)
  const submitInFlightRef = useRef(false)
  const [initialLoading, setInitialLoading] = useState(false)
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [baseModelId, setBaseModelId] = useState<string>('')
  const [baseModelName, setBaseModelName] = useState<string>('')
  const [baseModelPath, setBaseModelPath] = useState<string>('')
  const [enableGeneration, setEnableGeneration] = useState(false)
  const [enableTraining, setEnableTraining] = useState(false)
  const [enableDeployment, setEnableDeployment] = useState(false)
  const [allModelConfigs, setAllModelConfigs] = useState<ModelConfig[]>([])
  const [registeredModels, setRegisteredModels] = useState<RegisteredModel[]>([])
  const [modelByIdMap, setModelByIdMap] = useState<Map<string, RegisteredModel>>(new Map())
  // Training model selected manually (independent of deployment)
  const [trainingModelId, setTrainingModelId] = useState<string>('')
  const [trainingModelName, setTrainingModelName] = useState<string>('')
  const [trainingModelPath, setTrainingModelPath] = useState<string>('')
  // External API configs
  const [apiConfigs, setApiConfigs] = useState<ExternalApiConfig[]>([])
  const [selectedApiConfigId, setSelectedApiConfigId] = useState<string>('')
  const [apiConfigModalOpen, setApiConfigModalOpen] = useState(false)
  const [apiConfigSubmitting, setApiConfigSubmitting] = useState(false)
  const [apiConfigTesting, setApiConfigTesting] = useState(false)
  const [apiConfigForm] = Form.useForm()
  // Training model type for method linkage
  const [trainingModelType, setTrainingModelType] = useState<string>('llm')
  // Import from existing tasks
  const [generationTasks, setGenerationTasks] = useState<
    Array<{ task_id: string; task_name: string; status: string; generation_mode: string }>
  >([])
  const [trainingTaskList, setTrainingTaskList] = useState<TrainingTask[]>([])
  // Track whether base data has loaded (for edit mode)
  const [baseDataReady, setBaseDataReady] = useState(false)
  // Multi-target training
  const [trainingTargets, setTrainingTargets] = useState<TrainingTargetFormState[]>([])

  const clearDeploymentDerivedModelState = () => {
    setBaseModelId('')
    setBaseModelName('')
    setBaseModelPath('')
    setTrainingModelId('')
    setTrainingModelName('')
    setTrainingModelPath('')
  }

  const clearDeploymentSelection = () => {
    form.setFieldsValue({
      base_deployment_id: undefined,
      base_deployment_replica_id: undefined,
    })
    clearDeploymentDerivedModelState()
  }

  // Dependency chain: Generation → Training → Deployment
  const handleToggleGeneration = (checked: boolean) => {
    setEnableGeneration(checked)
    if (!checked) {
      setEnableTraining(false)
      setEnableDeployment(false)
      setTrainingTargets([])
      clearDeploymentSelection()
      form.setFieldsValue({
        llm_config: undefined,
        embedding_config: undefined,
        rerank_config: undefined,
      })
    }
  }
  const handleToggleTraining = (checked: boolean) => {
    setEnableTraining(checked)
    if (checked) {
      setEnableGeneration(true)
    } else {
      setEnableDeployment(false)
      setTrainingTargets([])
      clearDeploymentSelection()
    }
  }
  const handleToggleDeployment = (checked: boolean) => {
    setEnableDeployment(checked)
    if (checked) {
      setEnableGeneration(true)
      setEnableTraining(true)
    } else {
      clearDeploymentSelection()
    }
  }

  const addTrainingTarget = () => {
    setTrainingTargets((prev) => [
      ...prev,
      {
        ...DEFAULT_TARGET,
        key: crypto.randomUUID(),
        sort_order: prev.length,
      },
    ])
  }

  const removeTrainingTarget = (index: number) => {
    setTrainingTargets((prev) => prev.filter((_, i) => i !== index))
  }

  const updateTrainingTarget = (index: number, patch: Partial<TrainingTargetFormState>) => {
    setTrainingTargets((prev) => prev.map((t, i) => (i === index ? { ...t, ...patch } : t)))
  }

  useEffect(() => {
    const loadBaseData = async () => {
      try {
        const [depRes, cfgRes, mdlRes] = await Promise.all([
          deploymentApi.list({ page_size: 200 }),
          configApi.list({ page_size: 200 }),
          modelApi.list({ page_size: 500 }),
        ])
        const deps = depRes.items || []
        const allModels = mdlRes.items || []
        const modelById = new Map(allModels.map((m: RegisteredModel) => [m.model_id, m]))
        // Filter out deployments whose model is a LoRA adapter without N+1 API calls.
        const baseDeps = deps.filter((dep: Deployment) => {
          if (!dep.model_id) return true
          const model = modelById.get(dep.model_id)
          return model ? !model.is_adapter : true
        })

        setDeployments(baseDeps)
        setAllModelConfigs(cfgRes.items || [])
        setModelByIdMap(modelById)
        const models = allModels.filter(
          (m: RegisteredModel) => m.status === 'available' && !m.is_adapter
        )
        setRegisteredModels(models)
      } catch {
        /* handled by interceptor */
      }
      setBaseDataReady(true)
    }
    loadBaseData()
    fetchApiConfigs()

    // Fetch generation/training task lists for "import from" selectors
    generationApi
      .listTasks({ limit: 1000 })
      .then((response) => setGenerationTasks(response.tasks || []))
      .catch(() => {})
    trainingApi
      .list({ page_size: 200 })
      .then((res) => setTrainingTaskList(res.items || []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!baseDataReady || !watchedBaseDeploymentId) return
    const selected = deployments.find(
      (deployment) => deployment.deployment_id === watchedBaseDeploymentId
    )
    const currentIsHealthy = getHealthyDeploymentReplicas(selected).some(
      (replica) => replica.replica_id === watchedBaseDeploymentReplicaId
    )
    if (currentIsHealthy) return
    const nextReplicaId = getAutomaticHealthyReplicaId(selected)
    if (watchedBaseDeploymentReplicaId !== nextReplicaId) {
      form.setFieldValue('base_deployment_replica_id', nextReplicaId)
    }
  }, [baseDataReady, deployments, form, watchedBaseDeploymentId, watchedBaseDeploymentReplicaId])

  const fetchApiConfigs = () => {
    externalApiConfigApi
      .list()
      .then((res) => {
        setApiConfigs(res.configs || [])
      })
      .catch(() => {})
  }

  // Edit mode: load existing task config once base data is ready
  useEffect(() => {
    if (!isEditMode || !taskId || !baseDataReady) return
    setInitialLoading(true)
    syncApi
      .getTask(taskId)
      .then((res) => {
        const task = res.task
        // Basic fields
        form.setFieldsValue({
          task_name: task.task_name,
          sync_interval_seconds: task.sync_interval_seconds,
          generation_threshold: task.generation_threshold,
          generation_mode: task.generation_mode || 'doc_to_training',
          training_threshold: task.training_threshold,
        })
        if (task.external_api_config_id) setSelectedApiConfigId(task.external_api_config_id)

        // Generation config
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const genCfg = task.generation_config as Record<string, any> | null
        const generationEnabledByThreshold = Number(task.generation_threshold || 0) > 0
        const generationEnabledByConfig = !!(genCfg && Object.keys(genCfg).length > 0)
        if (generationEnabledByThreshold || generationEnabledByConfig) {
          setEnableGeneration(true)
        }
        if (genCfg) {
          populateGenerationFields(genCfg, form)
        }

        // Training config
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const trainCfg = task.training_config as Record<string, any> | null
        const trainingEnabledByThreshold = Number(task.training_threshold || 0) > 0
        const trainingEnabledByConfig = !!(trainCfg && Object.keys(trainCfg).length > 0)
        if (trainingEnabledByThreshold || trainingEnabledByConfig) {
          setEnableTraining(true)
        }
        if (trainCfg) {
          populateTrainingFields(
            trainCfg,
            form,
            registeredModels,
            setTrainingModelId,
            setTrainingModelName,
            setTrainingModelPath,
            setTrainingModelType
          )
        }

        // Deployment
        if (task.base_deployment_id) {
          setEnableGeneration(true)
          setEnableTraining(true)
          setEnableDeployment(true)
          form.setFieldValue('base_deployment_id', task.base_deployment_id)
          form.setFieldValue(
            'base_deployment_replica_id',
            task.base_deployment_replica_id || undefined
          )
        }

        // Multi-target training targets
        if (task.training_targets?.length) {
          setEnableTraining(true)
          setTrainingTargets(
            task.training_targets.map((tt: SyncTrainingTarget) => ({
              key: tt.target_id,
              target_name: tt.target_name,
              model_type: tt.model_type,
              data_phase: tt.data_phase as 'qa' | 'final',
              training_method: tt.training_method,
              base_model_path: tt.base_model_path || '',
              base_model_id: '',
              base_model_name: '',
              base_deployment_id: tt.base_deployment_id || '',
              base_deployment_replica_id: tt.base_deployment_replica_id || '',
              training_threshold: tt.training_threshold,
              priority: tt.priority,
              sort_order: tt.sort_order,
              lora_r: (tt.training_config?.lora_r as number) ?? 16,
              lora_alpha: (tt.training_config?.lora_alpha as number) ?? 32,
              lora_dropout: (tt.training_config?.lora_dropout as number) ?? 0,
              num_train_epochs: (tt.training_config?.num_train_epochs as number) ?? 3,
              per_device_train_batch_size:
                (tt.training_config?.per_device_train_batch_size as number) ?? 16,
              learning_rate: (tt.training_config?.learning_rate as number) ?? 2e-5,
              warmup_ratio: (tt.training_config?.warmup_ratio as number) ?? 0.1,
              gradient_accumulation_steps:
                (tt.training_config?.gradient_accumulation_steps as number) ?? 1,
              max_length: tt.training_config?.max_length as number | undefined,
              mixed_precision: tt.training_config?.bf16
                ? 'bf16'
                : tt.training_config?.fp16
                  ? 'fp16'
                  : 'none',
              gpu_ids: (tt.training_config?.gpu_ids as number[]) ?? [],
              embedding_loss_name: (tt.training_config?.embedding_loss_name as string) ?? '',
              reranker_loss_name: (tt.training_config?.reranker_loss_name as string) ?? 'auto',
              loss_config: tt.training_config?.loss_config as Record<string, unknown> | undefined,
              rl_config: tt.training_config?.rl_config as
                | {
                    beta?: number
                    rankings_direction?: 'auto' | 'higher_is_better' | 'lower_is_better'
                  }
                | undefined,
            }))
          )
        } else {
          setTrainingTargets([])
        }
      })
      .catch(() => {
        message.error(t('create.message.updateFailed'))
      })
      .finally(() => {
        setInitialLoading(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isEditMode, taskId, baseDataReady])

  const handleCreateApiConfig = async () => {
    try {
      const values = await apiConfigForm.validateFields()
      setApiConfigSubmitting(true)
      const res = await externalApiConfigApi.create({
        config_name: values.config_name,
        api_url: values.api_url,
        auth_config: { token: values.token, auth_method: values.auth_method || 'cookie' },
        description: values.description || '',
      })
      message.success(t('apiConfig.message.createSuccess'))
      setApiConfigModalOpen(false)
      apiConfigForm.resetFields()
      fetchApiConfigs()
      // Auto-select the newly created config
      setSelectedApiConfigId(res.config.config_id)
    } catch {
      message.error(t('apiConfig.message.createFailed'))
    } finally {
      setApiConfigSubmitting(false)
    }
  }

  const handleTestApiConfig = async () => {
    try {
      const values = await apiConfigForm.validateFields(['api_url', 'token', 'auth_method'])
      if (!values.api_url || !values.token) {
        message.warning(t('common:form.required'))
        return
      }
      setApiConfigTesting(true)
      const res = await externalApiConfigApi.testConnection({
        api_url: values.api_url,
        auth_config: { token: values.token, auth_method: values.auth_method || 'cookie' },
      })
      if (res.success) {
        message.success(t('apiConfig.message.testSuccess', { message: res.message }))
      } else {
        message.error(t('apiConfig.message.testFailed', { message: res.message }))
      }
    } catch {
      message.error(t('apiConfig.message.testFailed', { message: 'Request failed' }))
    } finally {
      setApiConfigTesting(false)
    }
  }

  const handleDeploymentChange = (deploymentId: string | undefined) => {
    const selected = deployments.find((d) => d.deployment_id === deploymentId)
    form.setFieldValue('base_deployment_replica_id', getAutomaticHealthyReplicaId(selected))
    if (!deploymentId) {
      clearDeploymentDerivedModelState()
      return
    }
    const dep = deployments.find((d) => d.deployment_id === deploymentId)
    if (dep?.model_id) {
      const model = modelByIdMap.get(dep.model_id)
      if (model) {
        const modelId = model.model_id || ''
        const modelName = model.display_name || model.model_name || ''
        const modelPath = model.base_model_path || model.model_path || ''
        setBaseModelId(modelId)
        setBaseModelName(modelName)
        setBaseModelPath(modelPath)
        // Auto-sync training model selector to deployment's model
        setTrainingModelId(modelId)
        setTrainingModelName(modelName)
        setTrainingModelPath(modelPath)
      } else {
        clearDeploymentDerivedModelState()
      }
    } else {
      clearDeploymentDerivedModelState()
    }

    // Auto-populate generation model configs from deployment's associated configs
    const linkedConfigs = allModelConfigs.filter((c) => c.deployment_id === deploymentId)
    for (const cfg of linkedConfigs) {
      const modelType = cfg.model_type === 'reranker' ? 'rerank' : cfg.model_type
      if (modelType === 'llm') {
        form.setFieldValue(['llm_config', 'config_id'], cfg.config_id)
      } else if (modelType === 'embedding') {
        form.setFieldValue(['embedding_config', 'config_id'], cfg.config_id)
      } else if (modelType === 'rerank') {
        form.setFieldValue(['rerank_config', 'config_id'], cfg.config_id)
      }
    }
  }

  const handleTrainingModelChange = (modelId: string | undefined) => {
    if (!modelId) {
      setTrainingModelId('')
      setTrainingModelName('')
      setTrainingModelPath('')
      return
    }
    const model = registeredModels.find((m) => m.model_id === modelId)
    if (model) {
      setTrainingModelId(model.model_id)
      setTrainingModelName(model.display_name || model.model_name || '')
      setTrainingModelPath(model.base_model_path || model.model_path || '')
    }
  }

  // Import config from existing generation task
  const handleImportGenerationConfig = useCallback(
    async (genTaskId: string | undefined) => {
      if (!genTaskId) return
      try {
        const task = await generationApi.getTask(genTaskId)
        populateGenerationFields(
          {
            llm_config: task.llm_config,
            embedding_config: task.embedding_config,
            rerank_config: task.rerank_config,
            pos_neg_method: task.pos_neg_method,
            output_format: task.output_format,
            worker_config: task.worker_config,
            steps_config: task.steps_config,
            post_process_config: task.post_process_config,
          },
          form
        )
        message.success(t('create.importSuccess'))
      } catch {
        message.error(t('create.message.updateFailed'))
      }
    },
    [form, t]
  )

  // Import config from existing training task
  const handleImportTrainingConfig = useCallback(
    async (trainTaskId: string | undefined) => {
      if (!trainTaskId) return
      try {
        const task = await trainingApi.get(trainTaskId)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const cfg: Record<string, any> = {
          base_model_path: task.base_model_path,
          model_type: task.model_type,
          training_method: task.training_method,
          gpu_ids: task.gpu_ids,
          loss_config: task.loss_config,
        }
        // Extract training params — stored in training_params JSON blob
        const tp = (task.training_params || {}) as Record<string, unknown>
        cfg.lora_r = tp.lora_r
        cfg.lora_alpha = tp.lora_alpha
        cfg.lora_dropout = tp.lora_dropout
        cfg.num_train_epochs = tp.num_train_epochs
        cfg.per_device_train_batch_size = tp.per_device_train_batch_size
        cfg.learning_rate = tp.learning_rate
        cfg.warmup_ratio = tp.warmup_ratio
        cfg.gradient_accumulation_steps = tp.gradient_accumulation_steps
        cfg.max_length = tp.max_length
        cfg.bf16 = tp.bf16
        cfg.fp16 = tp.fp16
        cfg.embedding_loss_name = tp.embedding_loss_name
        cfg.reranker_loss_name = tp.reranker_loss_name

        populateTrainingFields(
          cfg,
          form,
          registeredModels,
          setTrainingModelId,
          setTrainingModelName,
          setTrainingModelPath,
          setTrainingModelType
        )
        message.success(t('create.importSuccess'))
      } catch {
        message.error(t('create.message.updateFailed'))
      }
    },
    [form, registeredModels, t]
  )

  // Effective model: deployment takes priority over training selector
  const effectiveModelId = enableDeployment && baseModelId ? baseModelId : trainingModelId
  const effectiveModelName = enableDeployment && baseModelName ? baseModelName : trainingModelName
  const effectiveModelPath = enableDeployment && baseModelPath ? baseModelPath : trainingModelPath
  const modelFromDeployment = enableDeployment && !!baseModelId

  const handleSubmit = async (values: Record<string, unknown>) => {
    if (!selectedApiConfigId) {
      message.error(t('apiConfig.selectPlaceholder'))
      return
    }
    if (submitInFlightRef.current) return

    submitInFlightRef.current = true
    setLoading(true)
    try {
      const baseDeploymentId = (values.base_deployment_id as string) || ''
      const baseDeploymentReplicaId = (values.base_deployment_replica_id as string) || ''
      let deploymentsForValidation = deployments
      const hasDeploymentBinding =
        (enableDeployment && Boolean(baseDeploymentId)) ||
        trainingTargets.some((target) => Boolean(target.base_deployment_id))
      if (hasDeploymentBinding) {
        try {
          const response = await deploymentApi.list({ page_size: 200 })
          deploymentsForValidation = (response.items || []).filter((deployment) => {
            if (!deployment.model_id) return true
            const model = modelByIdMap.get(deployment.model_id)
            return model ? !model.is_adapter : true
          })
          setDeployments(deploymentsForValidation)
        } catch {
          message.error(t('create.fields.deploymentReplicaRefreshFailed'))
          return
        }
      }
      const usesTargetMode = enableTraining && trainingTargets.length > 0
      const targetNames = new Set<string>()
      const validateReplicaBinding = (
        deploymentId: string,
        replicaId: string,
        targetIndex?: number
      ): string | undefined => {
        if (!deploymentId) {
          return replicaId
            ? t('create.fields.deploymentReplicaInvalid', { index: targetIndex })
            : undefined
        }
        if (!replicaId) {
          return t('create.fields.deploymentReplicaRequired')
        }
        if (!isHealthyDeploymentReplicaBinding(deploymentsForValidation, deploymentId, replicaId)) {
          return t('create.fields.deploymentReplicaInvalid', { index: targetIndex })
        }
        return undefined
      }

      for (const [index, target] of trainingTargets.entries()) {
        const targetName = target.target_name.trim()
        if (!targetName) {
          message.error(t('create.fields.targetNameRequired', { index: index + 1 }))
          return
        }
        if (targetNames.has(targetName)) {
          message.error(t('create.fields.targetNameDuplicate', { name: targetName }))
          return
        }
        targetNames.add(targetName)
        if (!target.base_model_path.trim()) {
          message.error(t('create.fields.targetBaseModelRequired', { index: index + 1 }))
          return
        }
        const replicaError = validateReplicaBinding(
          target.base_deployment_id,
          target.base_deployment_replica_id,
          index + 1
        )
        if (replicaError) {
          message.error(replicaError)
          return
        }
      }

      if (enableDeployment) {
        const replicaError = validateReplicaBinding(baseDeploymentId, baseDeploymentReplicaId)
        if (replicaError) {
          message.error(replicaError)
          return
        }
      }

      const data: CreateSyncConfigRequest = {
        task_name: values.task_name as string,
        external_api_config_id: selectedApiConfigId,
        sync_interval_seconds: values.sync_interval_seconds as number,
        // 0 means auto-trigger disabled for this stage.
        generation_threshold: enableGeneration ? (values.generation_threshold as number) : 0,
        training_threshold:
          enableTraining && !usesTargetMode ? (values.training_threshold as number) : 0,
      }
      if (!isEditMode) {
        data.is_active = true
      }

      if (enableGeneration) {
        const llmConfig = {
          ...((values.llm_config as Record<string, unknown>) || {}),
          concurrency: (values.llm_concurrency as number) ?? 10,
        }
        data.generation_mode = (values.generation_mode as string) || 'doc_to_training'
        data.generation_config = {
          llm_config: llmConfig,
          embedding_config: values.embedding_config || undefined,
          rerank_config: values.rerank_config || undefined,
          pos_neg_method: (values.pos_neg_method as string) || 'retrieval',
          output_format: (values.output_format as string) || 'universal',
          worker_config: {
            timeout_per_doc: (values.timeout_per_doc as number) ?? 300,
          },
          steps_config: buildStepsConfig(values as Record<string, unknown>),
          post_process_config: {
            dedup: { enabled: Boolean(values.dedup_enabled) },
          },
        }
      } else if (isEditMode) {
        data.generation_config = null
      }

      if (enableTraining && !usesTargetMode) {
        if (!effectiveModelPath) {
          message.error(t('create.fields.baseModelRequired'))
          return
        }

        const mixedPrecision = values.mixed_precision as string
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const trainCfg: Record<string, any> = {
          base_model_path: effectiveModelPath,
          model_type: (values.model_type as string) || 'llm',
          training_method: (values.training_method as string) || 'sft',
          lora_r: values.lora_r as number,
          lora_alpha: values.lora_alpha as number,
          lora_dropout: (values.lora_dropout as number) ?? 0.0,
          num_train_epochs: (values.num_train_epochs as number) ?? 3,
          per_device_train_batch_size: (values.per_device_train_batch_size as number) ?? 16,
          learning_rate: (values.learning_rate as number) ?? 2e-5,
          warmup_ratio: (values.warmup_ratio as number) ?? 0.1,
          gradient_accumulation_steps: (values.gradient_accumulation_steps as number) ?? 1,
        }
        if (values.max_length && trainCfg.model_type !== 'embedding')
          trainCfg.max_length = values.max_length as number
        if (values.gpu_ids && (values.gpu_ids as number[]).length > 0)
          trainCfg.gpu_ids = values.gpu_ids
        if (mixedPrecision === 'bf16') trainCfg.bf16 = true
        if (mixedPrecision === 'fp16') trainCfg.fp16 = true
        // Loss 函数配置
        const mt = trainCfg.model_type as string
        if (mt === 'embedding' && values.embedding_loss_name) {
          trainCfg.embedding_loss_name = values.embedding_loss_name as string
        }
        if (mt === 'reranker' && values.reranker_loss_name) {
          trainCfg.reranker_loss_name = values.reranker_loss_name as string
          if (values.loss_config) {
            trainCfg.loss_config = values.loss_config
          }
        }
        if (mt === 'decoder_reranker' && values.loss_config) {
          trainCfg.loss_config = values.loss_config
        }
        data.training_config = trainCfg
      } else if (isEditMode) {
        data.training_config = null
      }

      if (enableDeployment) {
        data.base_deployment_id = baseDeploymentId || (isEditMode ? null : undefined)
        data.base_deployment_replica_id = baseDeploymentReplicaId || (isEditMode ? null : undefined)
      } else if (isEditMode) {
        data.base_deployment_id = null
        data.base_deployment_replica_id = null
      }

      if (!isEditMode && usesTargetMode) {
        data.training_targets = trainingTargets.map(buildTrainingTargetPayload)
      }

      if (isEditMode && taskId) {
        const updateData: UpdateSyncConfigRequest = {
          ...data,
          training_targets: enableTraining
            ? trainingTargets.map((target) => ({
                target_id: target.key,
                ...buildTrainingTargetPayload(target),
              }))
            : [],
        }
        await syncApi.updateTask(taskId, updateData)

        message.success(t('create.message.updateSuccess'))
        navigate(`/sync/${taskId}`)
      } else {
        await syncApi.createTask(data)

        message.success(t('create.message.createSuccess'))
        navigate('/sync')
      }
    } catch {
      message.error(
        isEditMode ? t('create.message.updateFailed') : t('create.message.createFailed')
      )
    } finally {
      submitInFlightRef.current = false
      setLoading(false)
    }
  }

  if (initialLoading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div className="sync-config-create-page">
      <div className="page-toolbar">
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => (isEditMode ? navigate(`/sync/${taskId}`) : navigate('/sync'))}
        >
          {t('common:action.back')}
        </Button>
        <Title level={4} style={{ margin: 0 }}>
          {isEditMode ? t('create.editTitle') : t('create.title')}
        </Title>
      </div>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        scrollToFirstError
        initialValues={{
          sync_interval_seconds: 300,
          generation_threshold: 500,
          generation_mode: 'doc_to_training',
          training_threshold: 1000,
          lora_r: 16,
          lora_alpha: 32,
          // Generation advanced defaults
          pos_neg_method: 'retrieval',
          output_format: 'universal',
          llm_concurrency: 10,
          timeout_per_doc: 300,
          doc_quality_enabled: false,
          doc_quality_min_score: 0.5,
          keypoint_gen_enabled: false,
          keypoint_gen_max: 5,
          qa_gen_enabled: true,
          qa_gen_count: 3,
          pos_neg_enabled: true,
          pos_neg_positive_count: 3,
          pos_neg_negative_count: 7,
          validation_enabled: true,
          dedup_enabled: true,
          // Training advanced defaults
          model_type: 'llm',
          training_method: 'sft',
          num_train_epochs: 3,
          per_device_train_batch_size: 16,
          learning_rate: 2e-5,
          warmup_ratio: 0.1,
          gradient_accumulation_steps: 1,
          lora_dropout: 0.0,
          mixed_precision: 'none',
        }}
      >
        <Row gutter={[16, 16]}>
          {/* Left column: Basic + External API + Generation */}
          <Col xs={24} xl={12}>
            <Card title={t('create.basicInfo')} size="small" style={{ marginBottom: 16 }}>
              <Form.Item
                name="task_name"
                label={t('create.fields.configName')}
                rules={[{ required: true, message: t('common:form.required') }]}
              >
                <Input
                  placeholder={t('create.fields.configNamePlaceholder')}
                  disabled={isEditMode}
                />
              </Form.Item>
            </Card>

            <Card title={t('create.externalApi')} size="small" style={{ marginBottom: 16 }}>
              <Form.Item label={t('apiConfig.select')} required>
                <Space.Compact style={{ width: '100%' }}>
                  <Select
                    style={{ flex: 1, minWidth: 0 }}
                    placeholder={t('apiConfig.selectPlaceholder')}
                    value={selectedApiConfigId || undefined}
                    onChange={(v) => setSelectedApiConfigId(v || '')}
                    allowClear={!isEditMode}
                    showSearch
                    optionFilterProp="label"
                    disabled={isEditMode}
                    options={apiConfigs.map((c) => ({
                      label: `${c.config_name}`,
                      value: c.config_id,
                    }))}
                  />
                  {!isEditMode && (
                    <Button
                      icon={<PlusOutlined />}
                      onClick={() => {
                        apiConfigForm.resetFields()
                        setApiConfigModalOpen(true)
                      }}
                    >
                      {t('apiConfig.createNew')}
                    </Button>
                  )}
                </Space.Compact>
              </Form.Item>
              {selectedApiConfigId &&
                (() => {
                  const selected = apiConfigs.find((c) => c.config_id === selectedApiConfigId)
                  return selected ? (
                    <div
                      style={{
                        background: BG_ELEVATED,
                        border: '1px solid var(--tf-border-secondary)',
                        borderRadius: 6,
                        padding: '8px 12px',
                        marginBottom: 16,
                        fontSize: 13,
                      }}
                    >
                      <div>
                        <span style={{ color: TEXT_SECONDARY }}>
                          {t('apiConfig.selectedUrl')}：
                        </span>
                        <span style={{ fontFamily: 'monospace' }}>{selected.api_url}</span>
                      </div>
                      {selected.description && (
                        <div>
                          <span style={{ color: TEXT_SECONDARY }}>
                            {t('apiConfig.columns.description')}：
                          </span>
                          {selected.description}
                        </div>
                      )}
                    </div>
                  ) : null
                })()}

              <Form.Item name="sync_interval_seconds" label={t('create.fields.syncInterval')}>
                <InputNumber min={10} style={{ width: '100%' }} />
              </Form.Item>
            </Card>

            {/* Generation Config (optional) */}
            <Card
              title={
                <Space>
                  {t('create.level1Config')}
                  <Switch
                    size="small"
                    checked={enableGeneration}
                    onChange={handleToggleGeneration}
                  />
                </Space>
              }
              size="small"
              style={{ marginBottom: 16 }}
            >
              {enableGeneration ? (
                <>
                  {generationTasks.length > 0 && (
                    <div style={{ marginBottom: 12 }}>
                      <Select
                        style={{ width: '100%' }}
                        placeholder={t('create.importFromGeneration')}
                        allowClear
                        showSearch
                        optionFilterProp="label"
                        suffixIcon={<ImportOutlined />}
                        options={generationTasks.map((gt) => ({
                          label: `${gt.task_name} (${gt.task_id.slice(0, 8)}) [${gt.status}]`,
                          value: gt.task_id,
                        }))}
                        onChange={handleImportGenerationConfig}
                        value={undefined}
                      />
                    </div>
                  )}
                  <Tabs
                    size="small"
                    items={[
                      {
                        key: 'gen-basic',
                        label: t('create.tabs.basic'),
                        children: (
                          <>
                            <Row gutter={16}>
                              <Col xs={24} sm={12}>
                                <Form.Item
                                  name="generation_threshold"
                                  label={t('create.fields.generationThreshold')}
                                  tooltip={t('create.fields.generationThresholdHelp')}
                                >
                                  <InputNumber min={1} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12}>
                                <Form.Item
                                  name="generation_mode"
                                  label={t('create.fields.generationMode')}
                                >
                                  <Select
                                    options={[
                                      { label: 'doc_to_training', value: 'doc_to_training' },
                                      { label: 'qa_to_training', value: 'qa_to_training' },
                                    ]}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>

                            <Divider style={{ margin: '12px 0' }}>
                              {t('create.fields.llmConfig')}
                            </Divider>
                            <Form.Item
                              name={['llm_config', 'config_id']}
                              label={t('create.fields.llmConfig')}
                            >
                              <ModelConfigSelector modelType="llm" mode="single" allowClear />
                            </Form.Item>

                            <Divider style={{ margin: '12px 0' }}>
                              {t('create.fields.embeddingConfig')}
                            </Divider>
                            <Form.Item
                              name={['embedding_config', 'config_id']}
                              label={t('create.fields.embeddingConfig')}
                            >
                              <ModelConfigSelector modelType="embedding" mode="single" allowClear />
                            </Form.Item>

                            <Divider style={{ margin: '12px 0' }}>
                              {t('create.fields.rerankConfig')}
                            </Divider>
                            <Form.Item
                              name={['rerank_config', 'config_id']}
                              label={t('create.fields.rerankConfig')}
                            >
                              <ModelConfigSelector modelType="rerank" mode="single" allowClear />
                            </Form.Item>
                          </>
                        ),
                      },
                      {
                        key: 'gen-advanced',
                        label: t('create.tabs.advanced'),
                        children: (
                          <>
                            <Row gutter={16}>
                              <Col xs={24} sm={12}>
                                <Form.Item
                                  name="pos_neg_method"
                                  label={t('create.fields.posNegMethod')}
                                >
                                  <Select
                                    options={[
                                      {
                                        label: t('create.fields.posNegMethodRetrieval'),
                                        value: 'retrieval',
                                      },
                                      { label: t('create.fields.posNegMethodLlm'), value: 'llm' },
                                    ]}
                                  />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12}>
                                <Form.Item
                                  name="output_format"
                                  label={t('create.fields.outputFormat')}
                                >
                                  <Select options={[{ label: 'universal', value: 'universal' }]} />
                                </Form.Item>
                              </Col>
                            </Row>

                            <Divider style={{ margin: '12px 0' }}>
                              {t('create.fields.workerConfig')}
                            </Divider>
                            <Row gutter={16}>
                              <Col xs={24} sm={12}>
                                <Form.Item
                                  name="llm_concurrency"
                                  label={t('create.fields.llmConcurrency')}
                                >
                                  <InputNumber min={1} max={100} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12}>
                                <Form.Item
                                  name="timeout_per_doc"
                                  label={t('create.fields.timeoutPerDoc')}
                                >
                                  <InputNumber min={30} max={3600} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                            </Row>

                            <Divider style={{ margin: '12px 0' }}>
                              {t('create.fields.stepsConfig')}
                            </Divider>
                            <Collapse
                              ghost
                              size="small"
                              items={[
                                {
                                  key: 'doc_quality',
                                  label: t('create.fields.docQuality'),
                                  children: (
                                    <Row gutter={16} align="middle">
                                      <Col xs={24} sm={8}>
                                        <Form.Item
                                          name="doc_quality_enabled"
                                          valuePropName="checked"
                                          style={{ marginBottom: 0 }}
                                        >
                                          <Switch size="small" />
                                        </Form.Item>
                                      </Col>
                                      <Col xs={24} sm={16}>
                                        <Form.Item
                                          name="doc_quality_min_score"
                                          label={t('create.fields.docQualityMinScore')}
                                          style={{ marginBottom: 0 }}
                                        >
                                          <InputNumber
                                            min={0}
                                            max={1}
                                            step={0.1}
                                            style={{ width: '100%' }}
                                          />
                                        </Form.Item>
                                      </Col>
                                    </Row>
                                  ),
                                },
                                {
                                  key: 'keypoint_gen',
                                  label: t('create.fields.keypointGen'),
                                  children: (
                                    <Row gutter={16} align="middle">
                                      <Col xs={24} sm={8}>
                                        <Form.Item
                                          name="keypoint_gen_enabled"
                                          valuePropName="checked"
                                          style={{ marginBottom: 0 }}
                                        >
                                          <Switch size="small" />
                                        </Form.Item>
                                      </Col>
                                      <Col xs={24} sm={16}>
                                        <Form.Item
                                          name="keypoint_gen_max"
                                          label={t('create.fields.keypointGenMax')}
                                          style={{ marginBottom: 0 }}
                                        >
                                          <InputNumber min={1} max={20} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                    </Row>
                                  ),
                                },
                                {
                                  key: 'qa_gen',
                                  label: t('create.fields.qaGen'),
                                  children: (
                                    <Row gutter={16} align="middle">
                                      <Col xs={24} sm={8}>
                                        <Form.Item
                                          name="qa_gen_enabled"
                                          valuePropName="checked"
                                          style={{ marginBottom: 0 }}
                                        >
                                          <Switch size="small" />
                                        </Form.Item>
                                      </Col>
                                      <Col xs={24} sm={16}>
                                        <Form.Item
                                          name="qa_gen_count"
                                          label={t('create.fields.qaGenCount')}
                                          style={{ marginBottom: 0 }}
                                        >
                                          <InputNumber min={1} max={20} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                    </Row>
                                  ),
                                },
                                {
                                  key: 'pos_neg',
                                  label: t('create.fields.posNeg'),
                                  children: (
                                    <Row gutter={16}>
                                      <Col xs={24} sm={12} xxl={8}>
                                        <Form.Item
                                          name="pos_neg_enabled"
                                          valuePropName="checked"
                                          style={{ marginBottom: 0 }}
                                        >
                                          <Switch size="small" />
                                        </Form.Item>
                                      </Col>
                                      <Col xs={24} sm={12} xxl={8}>
                                        <Form.Item
                                          name="pos_neg_positive_count"
                                          label={t('create.fields.posNegPositiveCount')}
                                          style={{ marginBottom: 0 }}
                                        >
                                          <InputNumber min={1} max={20} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                      <Col xs={24} sm={12} xxl={8}>
                                        <Form.Item
                                          name="pos_neg_negative_count"
                                          label={t('create.fields.posNegNegativeCount')}
                                          style={{ marginBottom: 0 }}
                                        >
                                          <InputNumber min={1} max={50} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                    </Row>
                                  ),
                                },
                                {
                                  key: 'validation',
                                  label: t('create.fields.validationStep'),
                                  children: (
                                    <Form.Item
                                      name="validation_enabled"
                                      valuePropName="checked"
                                      style={{ marginBottom: 0 }}
                                    >
                                      <Switch size="small" />
                                    </Form.Item>
                                  ),
                                },
                                {
                                  key: 'dedup',
                                  label: t('create.fields.dedupStep'),
                                  children: (
                                    <Form.Item
                                      name="dedup_enabled"
                                      valuePropName="checked"
                                      style={{ marginBottom: 0 }}
                                    >
                                      <Switch size="small" />
                                    </Form.Item>
                                  ),
                                },
                              ]}
                            />
                          </>
                        ),
                      },
                    ]}
                  />
                </>
              ) : (
                <div style={{ color: TEXT_SECONDARY, fontSize: 13 }}>
                  {t('create.fields.optionalHint')}
                </div>
              )}
            </Card>
          </Col>

          {/* Right column: Training + Deployment */}
          <Col xs={24} xl={12}>
            {/* Training Config (optional) */}
            <Card
              title={
                <Space>
                  {t('create.level2Config')}
                  <Switch size="small" checked={enableTraining} onChange={handleToggleTraining} />
                </Space>
              }
              size="small"
              style={{ marginBottom: 16 }}
            >
              {enableTraining ? (
                <>
                  {trainingTaskList.length > 0 && (
                    <div style={{ marginBottom: 12 }}>
                      <Select
                        style={{ width: '100%' }}
                        placeholder={t('create.importFromTraining')}
                        allowClear
                        showSearch
                        optionFilterProp="label"
                        suffixIcon={<ImportOutlined />}
                        options={trainingTaskList.map((tt) => ({
                          label: `${tt.task_name || tt.task_id.slice(0, 8)} [${tt.model_type}/${tt.training_method}] [${tt.status}]`,
                          value: tt.task_id,
                        }))}
                        onChange={handleImportTrainingConfig}
                        value={undefined}
                      />
                    </div>
                  )}
                  <Tabs
                    size="small"
                    items={[
                      {
                        key: 'train-basic',
                        label: t('create.tabs.basic'),
                        children: (
                          <>
                            <Form.Item
                              label={t('create.fields.trainingBaseModel')}
                              tooltip={t('create.fields.trainingBaseModelHelp')}
                            >
                              <Select
                                allowClear
                                showSearch
                                optionFilterProp="label"
                                placeholder={t('create.fields.trainingBaseModelPlaceholder')}
                                value={trainingModelId || undefined}
                                onChange={handleTrainingModelChange}
                                options={registeredModels.map((m) => ({
                                  label: `${m.display_name || m.model_name} (${m.model_type})`,
                                  value: m.model_id,
                                }))}
                              />
                            </Form.Item>
                            {effectiveModelName ? (
                              <div
                                style={{
                                  background: BG_ELEVATED,
                                  border: '1px solid var(--tf-border-secondary)',
                                  borderRadius: 6,
                                  padding: '8px 12px',
                                  marginBottom: 16,
                                  fontSize: 13,
                                }}
                              >
                                {modelFromDeployment && (
                                  <div
                                    style={{
                                      color: 'var(--tf-primary-text)',
                                      fontSize: 12,
                                      marginBottom: 4,
                                    }}
                                  >
                                    {t('create.fields.modelFromDeployment')}
                                  </div>
                                )}
                                <div>
                                  <span style={{ color: TEXT_SECONDARY }}>ID：</span>
                                  {effectiveModelId}
                                </div>
                                <div>
                                  <span style={{ color: TEXT_SECONDARY }}>
                                    {t('create.fields.baseModel')}：
                                  </span>
                                  {effectiveModelName}
                                </div>
                                <div style={{ wordBreak: 'break-all' }}>
                                  <span style={{ color: TEXT_SECONDARY }}>
                                    {t('create.fields.baseModelPath')}：
                                  </span>
                                  {effectiveModelPath}
                                </div>
                              </div>
                            ) : (
                              <div
                                style={{ color: TEXT_SECONDARY, fontSize: 13, marginBottom: 16 }}
                              >
                                {t('create.fields.baseModelHint')}
                              </div>
                            )}
                            <Form.Item
                              name="training_threshold"
                              label={t('create.fields.trainingThreshold')}
                              tooltip={t('create.fields.trainingThresholdHelp')}
                            >
                              <InputNumber min={1} style={{ width: '100%' }} />
                            </Form.Item>
                            <Row gutter={16}>
                              <Col xs={24} sm={12}>
                                <Form.Item name="lora_r" label={t('create.fields.loraR')}>
                                  <InputNumber min={1} max={256} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12}>
                                <Form.Item name="lora_alpha" label={t('create.fields.loraAlpha')}>
                                  <InputNumber min={1} max={512} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                            </Row>
                          </>
                        ),
                      },
                      {
                        key: 'train-advanced',
                        label: t('create.tabs.advanced'),
                        children: (
                          <>
                            <Row gutter={16}>
                              <Col xs={24} sm={12}>
                                <Form.Item name="model_type" label={t('create.fields.modelType')}>
                                  <Select
                                    options={[
                                      { label: 'LLM', value: 'llm' },
                                      { label: 'Embedding', value: 'embedding' },
                                      { label: 'Reranker', value: 'reranker' },
                                      { label: 'Decoder Reranker', value: 'decoder_reranker' },
                                    ]}
                                    onChange={(val: string) => {
                                      setTrainingModelType(val)
                                      const methods =
                                        trainingMethodOptions[val] || trainingMethodOptions.llm
                                      form.setFieldValue('training_method', methods[0]?.value)
                                    }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12}>
                                <Form.Item
                                  name="training_method"
                                  label={t('create.fields.trainingMethod')}
                                >
                                  <Select
                                    options={
                                      trainingMethodOptions[trainingModelType] ||
                                      trainingMethodOptions.llm
                                    }
                                  />
                                </Form.Item>
                              </Col>
                            </Row>

                            {/* Loss 函数选择器 - 根据 model_type 条件渲染 */}
                            <Form.Item
                              noStyle
                              shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type}
                            >
                              {({ getFieldValue }) => {
                                const mt = getFieldValue('model_type')
                                if (mt === 'embedding') {
                                  return (
                                    <Row gutter={16}>
                                      <Col span={24}>
                                        <Form.Item
                                          name="embedding_loss_name"
                                          label={t('create.fields.lossFunction')}
                                          initialValue="DynamicExplicitNegativesRankingLoss"
                                        >
                                          <Select options={embeddingLossOptions} />
                                        </Form.Item>
                                      </Col>
                                    </Row>
                                  )
                                }
                                if (mt === 'reranker') {
                                  return (
                                    <Row gutter={16}>
                                      <Col span={24}>
                                        <Form.Item
                                          name="reranker_loss_name"
                                          label={t('create.fields.lossFunction')}
                                          initialValue="auto"
                                        >
                                          <Select options={rerankerLossOptions} />
                                        </Form.Item>
                                      </Col>
                                      <Col span={24}>
                                        <Form.Item
                                          noStyle
                                          shouldUpdate={(prev, cur) =>
                                            prev.reranker_loss_name !== cur.reranker_loss_name
                                          }
                                        >
                                          {({ getFieldValue }) => {
                                            const lossName = getFieldValue('reranker_loss_name')
                                            if (!lossName || lossName === 'auto') return null

                                            const showScale = rerankerScaleLosses.includes(lossName)
                                            const showMiniBatch =
                                              rerankerMiniBatchLosses.includes(lossName)
                                            const showNegatives =
                                              rerankerNegativesLosses.includes(lossName)
                                            const showTopK = rerankerTopKLosses.includes(lossName)
                                            const showSigma = rerankerSigmaLosses.includes(lossName)
                                            const showRespectInputOrder =
                                              rerankerRespectInputOrderLosses.includes(lossName)

                                            return (
                                              <Row gutter={16}>
                                                {showScale && (
                                                  <Col xs={24} sm={12}>
                                                    <Form.Item
                                                      name={['loss_config', 'scale']}
                                                      label="Scale"
                                                      initialValue={10.0}
                                                      preserve={false}
                                                    >
                                                      <InputNumber
                                                        min={0}
                                                        max={100}
                                                        step={0.1}
                                                        style={{ width: '100%' }}
                                                      />
                                                    </Form.Item>
                                                  </Col>
                                                )}
                                                {showMiniBatch && (
                                                  <Col xs={24} sm={12}>
                                                    <Form.Item
                                                      name={['loss_config', 'mini_batch_size']}
                                                      label="Mini Batch Size"
                                                      initialValue={32}
                                                      preserve={false}
                                                    >
                                                      <InputNumber
                                                        min={1}
                                                        max={512}
                                                        step={1}
                                                        style={{ width: '100%' }}
                                                      />
                                                    </Form.Item>
                                                  </Col>
                                                )}
                                                {showNegatives && (
                                                  <Col xs={24} sm={12}>
                                                    <Form.Item
                                                      name={['loss_config', 'num_negatives']}
                                                      label="Num Negatives"
                                                      initialValue={4}
                                                      preserve={false}
                                                    >
                                                      <InputNumber
                                                        min={1}
                                                        max={128}
                                                        step={1}
                                                        style={{ width: '100%' }}
                                                      />
                                                    </Form.Item>
                                                  </Col>
                                                )}
                                                {showTopK && (
                                                  <Col xs={24} sm={12}>
                                                    <Form.Item
                                                      name={['loss_config', 'k']}
                                                      label="Top K"
                                                      preserve={false}
                                                    >
                                                      <InputNumber
                                                        min={1}
                                                        max={1024}
                                                        step={1}
                                                        style={{ width: '100%' }}
                                                      />
                                                    </Form.Item>
                                                  </Col>
                                                )}
                                                {showSigma && (
                                                  <Col xs={24} sm={12}>
                                                    <Form.Item
                                                      name={['loss_config', 'sigma']}
                                                      label="Sigma"
                                                      initialValue={1.0}
                                                      preserve={false}
                                                    >
                                                      <InputNumber
                                                        min={0}
                                                        max={100}
                                                        step={0.1}
                                                        style={{ width: '100%' }}
                                                      />
                                                    </Form.Item>
                                                  </Col>
                                                )}
                                                {showRespectInputOrder && (
                                                  <Col xs={24} sm={12}>
                                                    <Form.Item
                                                      name={['loss_config', 'respect_input_order']}
                                                      label="Respect Input Order"
                                                      initialValue={true}
                                                      preserve={false}
                                                      valuePropName="checked"
                                                    >
                                                      <Switch />
                                                    </Form.Item>
                                                  </Col>
                                                )}
                                              </Row>
                                            )
                                          }}
                                        </Form.Item>
                                      </Col>
                                    </Row>
                                  )
                                }
                                if (mt === 'decoder_reranker') {
                                  return (
                                    <Row gutter={16}>
                                      <Col xs={24} sm={12}>
                                        <Form.Item
                                          name={['loss_config', 'name']}
                                          label={t('create.fields.lossFunction')}
                                          initialValue="infonce"
                                        >
                                          <Select options={decoderRerankerLossOptions} />
                                        </Form.Item>
                                      </Col>
                                      <Col xs={24} sm={12}>
                                        <Form.Item
                                          name={['loss_config', 'n_docs']}
                                          label={t('create.fields.decoderRerankerNDocs')}
                                          initialValue={8}
                                        >
                                          <InputNumber min={2} max={32} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                    </Row>
                                  )
                                }
                                return null
                              }}
                            </Form.Item>

                            <Divider style={{ margin: '12px 0' }} />
                            <Row gutter={16}>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item
                                  name="num_train_epochs"
                                  label={t('create.fields.numTrainEpochs')}
                                >
                                  <InputNumber min={1} max={100} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item
                                  name="per_device_train_batch_size"
                                  label={t('create.fields.batchSize')}
                                >
                                  <InputNumber min={1} max={256} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item
                                  name="learning_rate"
                                  label={t('create.fields.learningRate')}
                                >
                                  <InputNumber
                                    min={0}
                                    max={1}
                                    step={1e-5}
                                    style={{ width: '100%' }}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={16}>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item
                                  name="warmup_ratio"
                                  label={t('create.fields.warmupRatio')}
                                >
                                  <InputNumber
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    style={{ width: '100%' }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item
                                  name="gradient_accumulation_steps"
                                  label={t('create.fields.gradAccumSteps')}
                                >
                                  <InputNumber min={1} max={128} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Form.Item
                                noStyle
                                shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type}
                              >
                                {({ getFieldValue }) =>
                                  getFieldValue('model_type') !== 'embedding' ? (
                                    <Col xs={24} sm={12} xxl={8}>
                                      <Form.Item
                                        name="max_length"
                                        label={t('create.fields.maxLength')}
                                      >
                                        <InputNumber
                                          min={64}
                                          max={131072}
                                          style={{ width: '100%' }}
                                        />
                                      </Form.Item>
                                    </Col>
                                  ) : null
                                }
                              </Form.Item>
                            </Row>

                            <Divider style={{ margin: '12px 0' }} />
                            <Row gutter={16}>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item
                                  name="lora_dropout"
                                  label={t('create.fields.loraDropout')}
                                >
                                  <InputNumber
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    style={{ width: '100%' }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item
                                  name="mixed_precision"
                                  label={t('create.fields.mixedPrecision')}
                                >
                                  <Select
                                    options={[
                                      { label: 'None', value: 'none' },
                                      { label: 'BF16', value: 'bf16' },
                                      { label: 'FP16', value: 'fp16' },
                                    ]}
                                  />
                                </Form.Item>
                              </Col>
                              <Col xs={24} sm={12} xxl={8}>
                                <Form.Item name="gpu_ids" label={t('create.fields.gpuIds')}>
                                  <GpuSelect allowClear />
                                </Form.Item>
                              </Col>
                            </Row>
                          </>
                        ),
                      },
                    ]}
                  />

                  {/* Multi-target training */}
                  <Divider style={{ margin: '16px 0' }}>
                    {t('create.fields.trainingTargets')}
                  </Divider>
                  <Button
                    type="dashed"
                    onClick={addTrainingTarget}
                    icon={<PlusOutlined />}
                    style={{ width: '100%', marginBottom: 16 }}
                  >
                    {t('create.fields.addTrainingTarget')}
                  </Button>
                  {trainingTargets.length > 0 ? (
                    <Collapse
                      items={trainingTargets.map((target, idx) => ({
                        key: target.key,
                        label: (
                          <Space>
                            <Tag
                              color={
                                target.model_type === 'llm'
                                  ? 'blue'
                                  : target.model_type === 'embedding'
                                    ? 'green'
                                    : 'orange'
                              }
                            >
                              {target.model_type.toUpperCase()}
                            </Tag>
                            <span>
                              {target.target_name ||
                                `${t('create.fields.trainingTargets')} ${idx + 1}`}
                            </span>
                            {target.data_phase === 'qa' ? (
                              <Tag color="purple">QA</Tag>
                            ) : (
                              <Tag color="cyan">Pos/Neg</Tag>
                            )}
                          </Space>
                        ),
                        extra: (
                          <Button
                            type="text"
                            danger
                            size="small"
                            icon={<DeleteOutlined />}
                            onClick={(e) => {
                              e.stopPropagation()
                              removeTrainingTarget(idx)
                            }}
                          />
                        ),
                        children: (
                          <TrainingTargetForm
                            target={target}
                            index={idx}
                            registeredModels={registeredModels.map((m) => ({
                              model_id: m.model_id,
                              model_name: m.model_name,
                              model_path: m.base_model_path || m.model_path,
                              display_name: m.display_name,
                            }))}
                            deployments={deployments}
                            onChange={updateTrainingTarget}
                          />
                        ),
                      }))}
                    />
                  ) : (
                    <Text type="secondary">{t('detail.noTrainingTargets')}</Text>
                  )}
                </>
              ) : (
                <div style={{ color: TEXT_SECONDARY, fontSize: 13 }}>
                  {t('create.fields.optionalHint')}
                </div>
              )}
            </Card>

            {/* Deployment Config (optional) */}
            <Card
              title={
                <Space>
                  {t('create.deploymentConfig')}
                  <Switch
                    size="small"
                    checked={enableDeployment}
                    onChange={handleToggleDeployment}
                  />
                </Space>
              }
              size="small"
            >
              {enableDeployment ? (
                <>
                  <Form.Item
                    name="base_deployment_id"
                    label={t('create.fields.baseDeployment')}
                    tooltip={t('create.fields.baseDeploymentHelp')}
                  >
                    <Select
                      allowClear
                      placeholder={t('create.fields.baseDeploymentPlaceholder')}
                      onChange={handleDeploymentChange}
                      options={deployments.filter(isSelectableSyncDeployment).map((d) => ({
                        label: `${d.deployment_name || d.deployment_id.slice(0, 8)} [${d.inference_framework}] [${d.status}]${d.enable_lora ? ' LoRA' : ''}`,
                        value: d.deployment_id,
                      }))}
                    />
                  </Form.Item>
                  {(() => {
                    const replicas =
                      deployments.find((d) => d.deployment_id === watchedBaseDeploymentId)
                        ?.replica_instances ?? []
                    if (!watchedBaseDeploymentId || replicas.length === 0) return null
                    return (
                      <Form.Item
                        name="base_deployment_replica_id"
                        label={t('create.fields.deploymentReplica')}
                        rules={[
                          {
                            required: true,
                            message: t('create.fields.deploymentReplicaRequired'),
                          },
                        ]}
                      >
                        <Select
                          placeholder={t('create.fields.deploymentReplicaPlaceholder')}
                          options={replicas.map((replica) => ({
                            label: `#${replica.replica_index} · ${replica.endpoint} · GPU ${replica.gpu_ids.join(',')}`,
                            value: replica.replica_id,
                            disabled:
                              replica.status !== 'running' || replica.health_status !== 'HEALTHY',
                          }))}
                        />
                      </Form.Item>
                    )
                  })()}
                  {baseModelPath && (
                    <Form.Item label={t('create.fields.baseModelPath')}>
                      <Input value={baseModelPath} disabled />
                    </Form.Item>
                  )}
                </>
              ) : (
                <div style={{ color: TEXT_SECONDARY, fontSize: 13 }}>
                  {t('create.fields.optionalHint')}
                </div>
              )}
            </Card>
          </Col>
        </Row>

        <div className="page-form-actions">
          <Button onClick={() => navigate('/sync')}>{t('common:action.cancel')}</Button>
          <Button type="primary" htmlType="submit" loading={loading} disabled={loading}>
            {isEditMode ? t('common:action.save') : t('create.submitButton')}
          </Button>
        </div>
      </Form>

      <Modal
        title={t('apiConfig.actions.create')}
        open={apiConfigModalOpen}
        onCancel={() => {
          setApiConfigModalOpen(false)
          apiConfigForm.resetFields()
        }}
        destroyOnClose
        footer={
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button icon={<ApiOutlined />} onClick={handleTestApiConfig} loading={apiConfigTesting}>
              {t('apiConfig.actions.testConnection')}
            </Button>
            <Space>
              <Button
                onClick={() => {
                  setApiConfigModalOpen(false)
                  apiConfigForm.resetFields()
                }}
              >
                {t('common:action.cancel')}
              </Button>
              <Button type="primary" onClick={handleCreateApiConfig} loading={apiConfigSubmitting}>
                {t('common:action.confirm')}
              </Button>
            </Space>
          </div>
        }
      >
        <Form form={apiConfigForm} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item
            name="config_name"
            label={t('apiConfig.fields.configName')}
            rules={[{ required: true, message: t('common:form.required') }]}
          >
            <Input placeholder={t('apiConfig.fields.configNamePlaceholder')} />
          </Form.Item>
          <Form.Item
            name="api_url"
            label={t('apiConfig.fields.apiUrl')}
            rules={[{ required: true, message: t('common:form.required') }]}
          >
            <Input placeholder={t('apiConfig.fields.apiUrlPlaceholder')} />
          </Form.Item>
          <Form.Item
            name="auth_method"
            label={t('apiConfig.fields.authMethod')}
            initialValue="cookie"
            tooltip={t('apiConfig.fields.authMethodHelp')}
          >
            <Select>
              <Select.Option value="cookie">{t('apiConfig.fields.authMethodCookie')}</Select.Option>
              <Select.Option value="bearer">{t('apiConfig.fields.authMethodBearer')}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="token"
            label={t('apiConfig.fields.token')}
            rules={[{ required: true, message: t('common:form.required') }]}
            tooltip={t('apiConfig.fields.tokenHelp')}
          >
            <Input.Password placeholder={t('apiConfig.fields.tokenPlaceholder')} />
          </Form.Item>
          <Form.Item name="description" label={t('apiConfig.fields.description')}>
            <Input.TextArea rows={2} placeholder={t('apiConfig.fields.descriptionPlaceholder')} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
