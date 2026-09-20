import { lazy, Suspense, useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Card,
  Descriptions,
  Tag,
  Progress,
  Button,
  Space,
  Statistic,
  Row,
  Col,
  message,
  Popconfirm,
  Tooltip,
  Typography,
  Spin,
  Alert,
} from 'antd'
import {
  ArrowLeftOutlined,
  StopOutlined,
  DeleteOutlined,
  SaveOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons'
import {
  trainingApi,
  modelApi,
  SILENT_REQUEST_CONFIG,
  type TrainingMetricsResponse,
} from '@/services/api'
import { usePolling } from '@/hooks'
import { formatDate, getStatusDescription, isActiveStatus, isSuccessStatus } from '@/utils'
import type { TrainingTask, TrainingTaskEvent, RegisteredModel } from '@/types'
import { StatusTag } from '@/components/StatusTag'
import TrainingEvents from './TrainingEvents'
import './TrainingDetail.css'

const { Title, Text } = Typography
const LossCurve = lazy(() =>
  import('@/components/training/LossCurve').then(({ LossCurve }) => ({
    default: LossCurve,
  }))
)

const TrainingDatasets = lazy(() => import('./TrainingDatasets'))
const TrainingTestResults = lazy(() => import('./TrainingTestResults'))

export default function TrainingDetail() {
  const { taskId } = useParams<{ taskId: string }>()
  const navigate = useNavigate()
  const { t } = useTranslation(['training', 'common', 'trainingDetailExtras'])
  const [loading, setLoading] = useState(true)
  const [task, setTask] = useState<TrainingTask | null>(null)
  const [metrics, setMetrics] = useState<TrainingMetricsResponse | null>(null)
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsFailed, setMetricsFailed] = useState(false)
  const [events, setEvents] = useState<TrainingTaskEvent[]>([])
  const [eventsLoading, setEventsLoading] = useState(false)
  const [eventsFailed, setEventsFailed] = useState(false)
  const [eventsLimit, setEventsLimit] = useState(100)
  const eventsLimitRef = useRef(100)
  const [registering, setRegistering] = useState(false)
  const [pollingStale, setPollingStale] = useState(false)
  const taskRequestGenerationRef = useRef(0)
  const metricsRequestGenerationRef = useRef(0)
  const eventsRequestGenerationRef = useRef(0)
  const taskLoadingOwnerRef = useRef(0)
  const metricsLoadingOwnerRef = useRef(0)
  const eventsLoadingOwnerRef = useRef(0)
  const taskScopeGenerationRef = useRef(0)
  const pollRequestGenerationRef = useRef(0)
  const mountedRef = useRef(true)
  const [resolvedBaseModel, setResolvedBaseModel] = useState<RegisteredModel | null>(null)
  const [resolvedTrainedModel, setResolvedTrainedModel] = useState<RegisteredModel | null>(null)
  const loadedTaskId = task?.task_id || ''
  const baseModelPath = task?.base_model_path || ''
  const trainedModelRegistryId = task?.trained_model_registry_id || ''
  const finalModelPath = task?.final_model_path || ''

  const fetchTask = useCallback(
    async (silent = false) => {
      if (!taskId) return false
      const requestGeneration = ++taskRequestGenerationRef.current
      const loadingOwner = silent ? null : ++taskLoadingOwnerRef.current
      if (loadingOwner !== null) setLoading(true)
      try {
        const data = await trainingApi.get(taskId, SILENT_REQUEST_CONFIG)
        if (requestGeneration !== taskRequestGenerationRef.current) return true
        setTask(data)
        return true
      } catch {
        if (requestGeneration !== taskRequestGenerationRef.current) return true
        if (!silent) {
          message.error({
            key: 'training-detail-load-failed',
            content: t('detail.message.loadFailed'),
          })
        }
        return false
      } finally {
        if (
          loadingOwner !== null &&
          loadingOwner === taskLoadingOwnerRef.current &&
          mountedRef.current
        ) {
          setLoading(false)
        }
      }
    },
    [taskId, t]
  )

  const fetchMetrics = useCallback(
    async (silent = false) => {
      if (!taskId) return false
      const requestGeneration = ++metricsRequestGenerationRef.current
      const loadingOwner = silent ? null : ++metricsLoadingOwnerRef.current
      try {
        if (loadingOwner !== null) setMetricsLoading(true)
        const data = await trainingApi.getMetrics(taskId, undefined, SILENT_REQUEST_CONFIG)
        if (requestGeneration !== metricsRequestGenerationRef.current) return true
        setMetrics(data)
        setMetricsFailed(false)
        return true
      } catch {
        if (requestGeneration !== metricsRequestGenerationRef.current) return true
        setMetricsFailed(true)
        return false
      } finally {
        if (
          loadingOwner !== null &&
          loadingOwner === metricsLoadingOwnerRef.current &&
          mountedRef.current
        ) {
          setMetricsLoading(false)
        }
      }
    },
    [taskId]
  )

  const fetchEvents = useCallback(
    async (silent = false, limit = eventsLimitRef.current) => {
      if (!taskId) return false
      // Keep background polls within the expanded scope even while its request is pending.
      eventsLimitRef.current = limit
      const requestGeneration = ++eventsRequestGenerationRef.current
      const loadingOwner = silent ? null : ++eventsLoadingOwnerRef.current
      if (loadingOwner !== null) setEventsLoading(true)
      try {
        const data = await trainingApi.getEvents(taskId, limit, SILENT_REQUEST_CONFIG)
        if (requestGeneration !== eventsRequestGenerationRef.current) return true
        setEvents(data.events || [])
        eventsLimitRef.current = limit
        setEventsLimit(limit)
        setEventsFailed(false)
        return true
      } catch {
        if (requestGeneration !== eventsRequestGenerationRef.current) return true
        setEventsFailed(true)
        return false
      } finally {
        if (
          loadingOwner !== null &&
          loadingOwner === eventsLoadingOwnerRef.current &&
          mountedRef.current
        ) {
          setEventsLoading(false)
        }
      }
    },
    [taskId]
  )

  useEffect(() => {
    taskScopeGenerationRef.current += 1
    pollRequestGenerationRef.current += 1
    taskRequestGenerationRef.current += 1
    metricsRequestGenerationRef.current += 1
    eventsRequestGenerationRef.current += 1
    taskLoadingOwnerRef.current += 1
    metricsLoadingOwnerRef.current += 1
    eventsLoadingOwnerRef.current += 1
    setTask(null)
    setMetrics(null)
    setEvents([])
    eventsLimitRef.current = 100
    setEventsLimit(100)
    setMetricsFailed(false)
    setEventsFailed(false)
    setPollingStale(false)
    setLoading(Boolean(taskId))
    setMetricsLoading(false)
    setEventsLoading(false)
    void fetchTask()
    void fetchMetrics()
    void fetchEvents()
    return () => {
      taskScopeGenerationRef.current += 1
      pollRequestGenerationRef.current += 1
      taskRequestGenerationRef.current += 1
      metricsRequestGenerationRef.current += 1
      eventsRequestGenerationRef.current += 1
      taskLoadingOwnerRef.current += 1
      metricsLoadingOwnerRef.current += 1
      eventsLoadingOwnerRef.current += 1
    }
  }, [taskId, fetchTask, fetchMetrics, fetchEvents])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      taskRequestGenerationRef.current += 1
      metricsRequestGenerationRef.current += 1
      eventsRequestGenerationRef.current += 1
      taskLoadingOwnerRef.current += 1
      metricsLoadingOwnerRef.current += 1
      eventsLoadingOwnerRef.current += 1
      taskScopeGenerationRef.current += 1
      pollRequestGenerationRef.current += 1
    }
  }, [])

  const pollTask = useCallback(async () => {
    const taskScopeGeneration = taskScopeGenerationRef.current
    const pollRequestGeneration = ++pollRequestGenerationRef.current
    const results = await Promise.all([fetchTask(true), fetchMetrics(true), fetchEvents(true)])
    if (
      mountedRef.current &&
      taskScopeGeneration === taskScopeGenerationRef.current &&
      pollRequestGeneration === pollRequestGenerationRef.current
    ) {
      setPollingStale(results.some((succeeded) => !succeeded))
    }
  }, [fetchTask, fetchMetrics, fetchEvents])

  const handleRefresh = useCallback(async () => {
    const taskScopeGeneration = taskScopeGenerationRef.current
    const refreshGeneration = ++pollRequestGenerationRef.current
    const results = await Promise.all([fetchTask(), fetchMetrics(), fetchEvents()])
    if (
      mountedRef.current &&
      taskScopeGeneration === taskScopeGenerationRef.current &&
      refreshGeneration === pollRequestGenerationRef.current
    ) {
      setPollingStale(results.some((succeeded) => !succeeded))
    }
  }, [fetchTask, fetchMetrics, fetchEvents])

  // Auto refresh for running tasks
  usePolling(pollTask, {
    interval: 5000,
    enabled: !!task && isActiveStatus(task.status),
  })

  // Resolve base model and trained model to registered models
  useEffect(() => {
    let active = true
    const taskScopeGeneration = taskScopeGenerationRef.current
    const canCommit = () =>
      active && mountedRef.current && taskScopeGeneration === taskScopeGenerationRef.current

    setResolvedBaseModel(null)
    setResolvedTrainedModel(null)
    if (
      !loadedTaskId ||
      loadedTaskId !== taskId ||
      (!baseModelPath && !trainedModelRegistryId && !finalModelPath)
    ) {
      return () => {
        active = false
      }
    }
    // Compare path suffix to handle container vs host path mismatch
    // e.g. /app/models/xxx vs /data/.../models/xxx
    const pathSuffix = (p: string) => p.split('/').filter(Boolean).pop() || p
    const matchByPath = (items: RegisteredModel[], target: string) =>
      items.find((m) => m.model_path === target) ||
      items.find((m) => pathSuffix(m.model_path) === pathSuffix(target))

    // trained model 按 ID 精确查询（避免全量列表被 limit 截断时匹配失败）
    const trainedPromise = trainedModelRegistryId
      ? modelApi.get(trainedModelRegistryId).catch(() => null)
      : Promise.resolve(null)

    modelApi
      .list({ page_size: 200 })
      .then(async ({ items }) => {
        if (!canCommit()) return
        if (baseModelPath) {
          setResolvedBaseModel(matchByPath(items, baseModelPath) || null)
        } else {
          setResolvedBaseModel(null)
        }
        if (trainedModelRegistryId) {
          const trained = await trainedPromise
          if (!canCommit()) return
          setResolvedTrainedModel(
            trained || items.find((m) => m.model_id === trainedModelRegistryId) || null
          )
        } else if (finalModelPath) {
          setResolvedTrainedModel(matchByPath(items, finalModelPath) || null)
        } else {
          setResolvedTrainedModel(null)
        }
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [loadedTaskId, taskId, baseModelPath, trainedModelRegistryId, finalModelPath])

  const handleStop = async () => {
    if (!taskId) return
    try {
      await trainingApi.stop(taskId)
      message.success(t('detail.message.stopSent'))
      fetchTask()
    } catch {
      message.error(t('detail.message.stopFailed'))
    }
  }

  const handleDelete = async () => {
    if (!taskId) return
    try {
      await trainingApi.delete(taskId)
      message.success(t('common:message.deleteSuccess'))
      navigate('/training')
    } catch {
      message.error(t('common:message.deleteFailed'))
    }
  }

  const handleRegisterModel = async () => {
    if (!taskId || !task) return
    setRegistering(true)
    try {
      await modelApi.registerFromTask(taskId, {
        model_name: task.task_name || `model-${taskId.slice(0, 8)}`,
        description: task.description,
      })
      message.success(t('detail.message.registerSuccess'))
    } catch {
      message.error(t('detail.message.registerFailed'))
    } finally {
      setRegistering(false)
    }
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    )
  }

  if (!task) {
    return (
      <Alert
        message={t('detail.taskNotFound')}
        description={t('detail.taskNotFoundDesc')}
        type="error"
        showIcon
        action={<Button onClick={() => navigate('/training')}>{t('detail.backToList')}</Button>}
      />
    )
  }

  const statusDescription = getStatusDescription(task.status)

  return (
    <div className="training-detail">
      <div className="page-toolbar">
        <div className="training-detail-heading">
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/training')}>
            {t('common:action.back')}
          </Button>
          <Title level={4} style={{ margin: 0 }}>
            {task.task_name || task.task_id.slice(0, 8)}
          </Title>
          <StatusTag status={task.status} />
        </div>
        <Space className="page-toolbar-actions" wrap>
          <Button icon={<ReloadOutlined />} onClick={() => void handleRefresh()}>
            {t('common:action.refresh')}
          </Button>
          {isSuccessStatus(task.status) &&
            (task.trained_model_registry_id ? (
              <Tooltip title={t('detail.registeredTooltip')}>
                <Button icon={<CheckCircleOutlined />} disabled>
                  {t('detail.registered')}
                </Button>
              </Tooltip>
            ) : (
              <Button
                type="primary"
                icon={<SaveOutlined />}
                onClick={handleRegisterModel}
                loading={registering}
              >
                {t('detail.registerModel')}
              </Button>
            ))}
          {isActiveStatus(task.status) && (
            <Popconfirm title={t('detail.confirmStop')} onConfirm={handleStop}>
              <Button danger icon={<StopOutlined />}>
                {t('common:action.stop')}
              </Button>
            </Popconfirm>
          )}
          {!isActiveStatus(task.status) && (
            <Popconfirm title={t('detail.confirmDelete')} onConfirm={handleDelete}>
              <Button danger icon={<DeleteOutlined />}>
                {t('common:action.delete')}
              </Button>
            </Popconfirm>
          )}
        </Space>
      </div>

      {pollingStale && (
        <Alert
          data-testid="training-stale-state"
          type="warning"
          showIcon
          message={t('detail.pollingStale')}
          style={{ marginBottom: 16 }}
        />
      )}

      {task.error_message && (
        <Card
          size="small"
          title={
            <span style={{ color: '#ff4d4f' }}>
              {'\u2715'} {t('detail.errorInfo')}
            </span>
          }
          style={{ marginBottom: 16, borderColor: '#ff4d4f' }}
        >
          <pre
            style={{
              maxHeight: 300,
              overflow: 'auto',
              margin: 0,
              padding: 12,
              background: 'var(--tf-bg-elevated)',
              color: 'var(--tf-text-primary)',
              borderRadius: 4,
              fontSize: 12,
              lineHeight: 1.6,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}
          >
            {task.error_message}
          </pre>
        </Card>
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <Card title={t('detail.trainingProgress')} size="small" style={{ marginBottom: 16 }}>
            {statusDescription && (
              <div style={{ marginBottom: 12 }}>
                <Text type="secondary">{statusDescription}</Text>
              </div>
            )}
            <Progress
              percent={task.progress}
              status={
                task.status === 'failed'
                  ? 'exception'
                  : isSuccessStatus(task.status)
                    ? 'success'
                    : 'active'
              }
              strokeWidth={20}
            />
            <Row className="training-detail-metrics" gutter={[16, 16]} style={{ marginTop: 24 }}>
              <Col xs={12} md={6}>
                <Statistic
                  title={t('detail.currentStep')}
                  value={task.current_step ?? metrics?.current_metrics?.current_step ?? 0}
                  suffix={
                    (task.total_steps ?? metrics?.current_metrics?.total_steps)
                      ? `/ ${task.total_steps ?? metrics?.current_metrics?.total_steps}`
                      : ''
                  }
                />
              </Col>
              <Col xs={12} md={6}>
                <Statistic
                  title={t('detail.currentEpoch')}
                  value={
                    task.current_epoch ??
                    metrics?.current_metrics?.current_epoch ??
                    (metrics?.summary?.epochs_completed
                      ? Math.floor(metrics.summary.epochs_completed)
                      : 0)
                  }
                  suffix={
                    (task.total_epochs ?? metrics?.current_metrics?.total_epochs)
                      ? `/ ${task.total_epochs ?? metrics?.current_metrics?.total_epochs}`
                      : ''
                  }
                />
              </Col>
              <Col xs={12} md={6}>
                <Statistic
                  title={t('detail.trainLoss')}
                  value={
                    (
                      task.train_loss ??
                      metrics?.current_metrics?.train_loss ??
                      metrics?.summary?.best_train_loss
                    )?.toFixed(4) || '-'
                  }
                />
              </Col>
              <Col xs={12} md={6}>
                <Statistic
                  title={t('detail.evalLoss')}
                  value={
                    (
                      task.eval_loss ??
                      metrics?.current_metrics?.eval_loss ??
                      metrics?.summary?.best_eval_loss
                    )?.toFixed(4) || '-'
                  }
                />
              </Col>
            </Row>
          </Card>

          {/* Loss Curve Chart */}
          <div style={{ marginBottom: 16 }}>
            <Suspense
              fallback={
                <div
                  role="status"
                  style={{
                    minHeight: 280,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 10,
                  }}
                >
                  <Spin size="small" />
                  <span>{t('detail.lossCurve.loading')}</span>
                </div>
              }
            >
              <LossCurve
                metrics={metrics}
                loading={metricsLoading}
                height={280}
                taskStatus={task.status}
                failed={metricsFailed}
                onRetry={() => void fetchMetrics()}
              />
            </Suspense>
          </div>

          <Card title={t('detail.taskConfig')} size="small" style={{ marginBottom: 16 }}>
            <Descriptions column={1} size="small">
              <Descriptions.Item label={t('detail.taskId')}>{task.task_id}</Descriptions.Item>
              <Descriptions.Item label={t('detail.modelType')}>
                <Tag>{task.model_type}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label={t('detail.trainingMethod')}>
                <Tag color="blue">{task.training_method.toUpperCase()}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label={t('detail.tunerType')}>
                {(() => {
                  const tuner = (task.training_params as Record<string, unknown> | undefined)
                    ?.tuner_type as string | undefined
                  if (tuner)
                    return (
                      <Tag color={tuner === 'full' ? 'default' : 'green'}>
                        {tuner.toUpperCase()}
                      </Tag>
                    )
                  return (
                    <Tag color={task.is_lora ? 'green' : 'default'}>
                      {task.is_lora ? 'LoRA' : t('detail.fullFinetune')}
                    </Tag>
                  )
                })()}
              </Descriptions.Item>
              <Descriptions.Item label={t('detail.gpu')}>
                {task.gpu_ids?.length ? task.gpu_ids.join(', ') : t('detail.gpuAutoAssign')}
              </Descriptions.Item>
              <Descriptions.Item label={t('detail.baseModel')}>
                {task.base_model_path ? (
                  resolvedBaseModel ? (
                    <Space>
                      <Button
                        type="link"
                        size="small"
                        style={{ padding: 0, fontSize: 12 }}
                        onClick={() => navigate(`/models?detail=${resolvedBaseModel.model_id}`)}
                      >
                        {resolvedBaseModel.model_name ||
                          resolvedBaseModel.display_name ||
                          resolvedBaseModel.model_id}
                      </Button>
                      <Text
                        copyable={{ text: task.base_model_path, tooltips: false }}
                        style={{ fontSize: 0 }}
                      />
                    </Space>
                  ) : (
                    <Text copyable style={{ fontSize: 12, wordBreak: 'break-all' }}>
                      {task.base_model_path}
                    </Text>
                  )
                ) : (
                  '-'
                )}
              </Descriptions.Item>
              <Descriptions.Item label={t('detail.outputDir')}>
                {task.output_dir || '-'}
              </Descriptions.Item>
              {task.final_model_path && (
                <Descriptions.Item label={t('detail.outputModel')}>
                  {resolvedTrainedModel ? (
                    <Space>
                      <Button
                        type="link"
                        size="small"
                        style={{ padding: 0, fontSize: 12 }}
                        onClick={() => navigate(`/models?detail=${resolvedTrainedModel.model_id}`)}
                      >
                        {resolvedTrainedModel.model_name ||
                          resolvedTrainedModel.display_name ||
                          resolvedTrainedModel.model_id}
                      </Button>
                      <Text
                        copyable={{ text: task.final_model_path, tooltips: false }}
                        style={{ fontSize: 0 }}
                      />
                    </Space>
                  ) : (
                    <Text copyable style={{ fontSize: 12, wordBreak: 'break-all' }}>
                      {task.final_model_path}
                    </Text>
                  )}
                </Descriptions.Item>
              )}
            </Descriptions>
          </Card>

          {task.dataset_configs?.length || task.train_dataset_path ? (
            <Suspense fallback={<div role="status">{t('trainingDetailExtras:loadingTable')}</div>}>
              <TrainingDatasets task={task} />
            </Suspense>
          ) : null}

          <Card title={t('detail.trainingParams')} size="small">
            {(() => {
              const params = task.training_params as Record<string, unknown> | undefined
              const loraConfig = params?.lora_config as Record<string, unknown> | undefined
              return (
                <Descriptions column={{ xs: 1, sm: 2, xxl: 3 }} size="small">
                  <Descriptions.Item label={t('detail.learningRate')}>
                    {String(task.learning_rate ?? params?.learning_rate ?? '-')}
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.epochs')}>
                    {String(params?.num_train_epochs ?? '-')}
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.batchSize')}>
                    {String(params?.per_device_train_batch_size ?? '-')}
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.warmupRatio')}>
                    {String(params?.warmup_ratio ?? '-')}
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.evalStrategy')}>
                    {String(params?.eval_strategy ?? '-')}
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.saveStrategy')}>
                    {String(params?.save_strategy ?? '-')}
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.mixedPrecision')}>
                    {params?.bf16 ? 'BF16' : params?.fp16 ? 'FP16' : 'FP32'}
                  </Descriptions.Item>
                  {(params?.gradient_accumulation_steps as number) > 1 && (
                    <Descriptions.Item label={t('detail.gradientAccumulation')}>
                      {String(params?.gradient_accumulation_steps)}
                    </Descriptions.Item>
                  )}
                  {params?.max_seq_length != null && (
                    <Descriptions.Item label={t('detail.maxSeqLength')}>
                      {String(params.max_seq_length)}
                    </Descriptions.Item>
                  )}
                  {params?.logging_steps != null && (
                    <Descriptions.Item label={t('detail.loggingSteps')}>
                      {String(params.logging_steps)}
                    </Descriptions.Item>
                  )}
                  {params?.deepspeed != null && (
                    <Descriptions.Item label={t('detail.deepspeed')}>
                      {String(params.deepspeed)}
                    </Descriptions.Item>
                  )}
                  {task.is_lora && (
                    <>
                      <Descriptions.Item label="LoRA Rank">
                        {String(params?.lora_r ?? loraConfig?.r ?? '-')}
                      </Descriptions.Item>
                      <Descriptions.Item label="LoRA Alpha">
                        {String(params?.lora_alpha ?? loraConfig?.lora_alpha ?? '-')}
                      </Descriptions.Item>
                      {Number(params?.lora_dropout ?? loraConfig?.lora_dropout ?? 0) > 0 && (
                        <Descriptions.Item label={t('detail.loraDropout')}>
                          {String(params?.lora_dropout ?? loraConfig?.lora_dropout)}
                        </Descriptions.Item>
                      )}
                    </>
                  )}
                </Descriptions>
              )
            })()}
          </Card>

          {/* Loss Config Card */}
          {(() => {
            const params = task.training_params as Record<string, unknown> | undefined
            const lossConfig = task.loss_config as Record<string, unknown> | undefined
            const lossName = (lossConfig?.name ??
              params?.embedding_loss_name ??
              params?.reranker_loss_name) as string | undefined
            if (!lossName && !lossConfig) return null
            const extraFields = lossConfig
              ? Object.entries(lossConfig).filter(([k]) => k !== 'name')
              : []
            return (
              <Card title={t('detail.lossConfig')} size="small" style={{ marginTop: 16 }}>
                <Descriptions column={{ xs: 1, sm: 2, xxl: 3 }} size="small">
                  {lossName && (
                    <Descriptions.Item label={t('detail.lossName')}>
                      <Tag color="purple">{lossName}</Tag>
                    </Descriptions.Item>
                  )}
                  {extraFields.map(([key, value]) => (
                    <Descriptions.Item key={key} label={key}>
                      {String(value ?? '-')}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </Card>
            )
          })()}

          {/* RL Config Card */}
          {(() => {
            const rlConfig = task.rl_config as Record<string, unknown> | undefined
            if (!rlConfig || Object.keys(rlConfig).length === 0) return null
            return (
              <Card title={t('detail.rlConfig')} size="small" style={{ marginTop: 16 }}>
                <Descriptions column={{ xs: 1, sm: 2, xxl: 3 }} size="small">
                  {task.sft_checkpoint_path && (
                    <Descriptions.Item
                      label={t('detail.sftCheckpointPath')}
                      span={{ xs: 1, sm: 2, xxl: 3 }}
                    >
                      <Text code style={{ fontSize: 12, wordBreak: 'break-all' }}>
                        {task.sft_checkpoint_path}
                      </Text>
                    </Descriptions.Item>
                  )}
                  {Object.entries(rlConfig).map(([key, value]) => (
                    <Descriptions.Item key={key} label={key}>
                      {String(value ?? '-')}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </Card>
            )
          })()}
        </Col>

        <Col xs={24} xl={8}>
          <Card title={t('detail.timeInfo')} size="small" style={{ marginBottom: 16 }}>
            <Descriptions column={1} size="small">
              <Descriptions.Item label={t('detail.createdAt')}>
                {formatDate(task.created_at)}
              </Descriptions.Item>
              {task.started_at && (
                <Descriptions.Item label={t('detail.startedAt')}>
                  {formatDate(task.started_at)}
                </Descriptions.Item>
              )}
              {task.completed_at && (
                <Descriptions.Item label={t('detail.completedAt')}>
                  {formatDate(task.completed_at)}
                </Descriptions.Item>
              )}
              <Descriptions.Item label={t('detail.updatedAt')}>
                {formatDate(task.updated_at)}
              </Descriptions.Item>
            </Descriptions>
          </Card>

          {/* Training result - final Loss */}
          {task.final_metrics &&
            (task.final_metrics.final_train_loss !== undefined ||
              task.final_metrics.final_eval_loss !== undefined) && (
              <Card title={t('detail.trainingResult')} size="small" style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  {task.final_metrics.final_train_loss !== undefined && (
                    <Col span={12}>
                      <Statistic
                        title={t('detail.finalTrainLoss')}
                        value={Number(task.final_metrics.final_train_loss).toFixed(4)}
                      />
                    </Col>
                  )}
                  {task.final_metrics.final_eval_loss !== undefined && (
                    <Col span={12}>
                      <Statistic
                        title={t('detail.finalEvalLoss')}
                        value={Number(task.final_metrics.final_eval_loss).toFixed(4)}
                      />
                    </Col>
                  )}
                </Row>
              </Card>
            )}
          <TrainingEvents
            key={task.task_id}
            events={events}
            loading={eventsLoading}
            failed={eventsFailed}
            limit={eventsLimit}
            onLoadOlder={() => fetchEvents(false, Math.min(eventsLimit + 100, 500))}
            onRetry={() => fetchEvents()}
          />
        </Col>
      </Row>

      {task.final_metrics?.test_before || task.final_metrics?.test_after ? (
        <Suspense fallback={<div role="status">{t('trainingDetailExtras:loadingTable')}</div>}>
          <TrainingTestResults key={task.task_id} task={task} />
        </Suspense>
      ) : null}
    </div>
  )
}
