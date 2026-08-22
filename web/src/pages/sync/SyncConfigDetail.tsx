import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Card, Descriptions, Button, Space, Tag, Table, Progress, message,
  Typography, Row, Col, Popconfirm, Tabs, Spin, Tooltip,
} from 'antd'
import {
  ArrowLeftOutlined, SyncOutlined, ThunderboltOutlined,
  PlayCircleOutlined, PauseCircleOutlined, ExperimentOutlined,
  EditOutlined, PlusOutlined, SwapOutlined, DisconnectOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { syncApi } from '@/services/api'
import { usePolling } from '@/hooks'
import { formatDate } from '@/utils'
import { StatusTag } from '@/components/StatusTag'
import { StatCard } from '@/components/StatCard'
import type { SyncConfig, SyncBatch, SyncGeneration, SyncTraining, SyncTrainingTarget } from '@/types'
import {
  ACTIVE_SYNC_STATUSES,
  SYNC_GENERATION_TOGGLE_STATUS,
  SYNC_TRAINING_CAN_LOAD_STATUSES,
  SYNC_TRAINING_SUCCEEDED_STATUSES,
} from '@/constants/syncStatus'
import {
  STATUS_INFO, STATUS_SUCCESS, STATUS_WARNING,
  TEXT_SECONDARY, BG_ELEVATED, BORDER_SECONDARY,
} from '@/theme'

const { Title, Text } = Typography

/* ---------- Generation Config View ---------- */
function GenerationConfigView({ config, t }: { config: Record<string, unknown>; t: (key: string) => string }) {
  const f = (key: string) => t(`detail.configFields.${key}`)
  const llm = config.llm_config as Record<string, unknown> | undefined
  const emb = config.embedding_config as Record<string, unknown> | undefined
  const rerank = config.rerank_config as Record<string, unknown> | undefined
  const worker = config.worker_config as Record<string, unknown> | undefined
  const steps = config.steps_config as Record<string, Record<string, unknown>> | undefined
  const postProcess = config.post_process_config as Record<string, Record<string, unknown>> | undefined

  const renderConfigId = (id: unknown) => {
    if (!id) return <span style={{ color: TEXT_SECONDARY }}>-</span>
    const s = String(id)
    return (
      <Tooltip title={s}>
        <Text copyable={{ text: s, tooltips: false }} style={{ fontSize: 12 }}>{s.slice(0, 8)}</Text>
      </Tooltip>
    )
  }

  const stepTags: { key: string; label: string; enabled: boolean; params?: string }[] = []
  if (steps?.doc_quality) {
    stepTags.push({ key: 'dq', label: f('docQuality'), enabled: !!steps.doc_quality.enabled, params: `${f('minScore')}: ${steps.doc_quality.min_score ?? '-'}` })
  }
  if (steps?.keypoint_gen) {
    stepTags.push({ key: 'kp', label: f('keypointGen'), enabled: !!steps.keypoint_gen.enabled, params: `${f('maxKeypoints')}: ${steps.keypoint_gen.max_keypoints ?? '-'}` })
  }
  if (steps?.qa_gen) {
    stepTags.push({ key: 'qa', label: f('qaGen'), enabled: !!steps.qa_gen.enabled, params: `${f('numQaPerDoc')}: ${steps.qa_gen.num_qa_per_doc ?? '-'}` })
  }
  if (steps?.pos_neg_extraction) {
    const pn = steps.pos_neg_extraction
    stepTags.push({ key: 'pn', label: f('posNegExtraction'), enabled: !!pn.enabled, params: `+${pn.num_positive ?? '?'} / -${pn.num_negative ?? '?'}` })
  }
  if (steps?.validation) {
    stepTags.push({ key: 'val', label: f('validation'), enabled: !!steps.validation.enabled })
  }
  if (postProcess?.dedup) {
    stepTags.push({ key: 'dedup', label: f('dedup'), enabled: !!postProcess.dedup.enabled })
  }

  return (
    <>
      <Descriptions size="small" column={2} style={{ marginBottom: 12 }}>
        <Descriptions.Item label={f('llmConfig')}>
          {renderConfigId(llm?.config_id)}
          {llm?.concurrency != null && <Tag style={{ marginLeft: 4 }}>{f('concurrency')}: {String(llm.concurrency)}</Tag>}
        </Descriptions.Item>
        <Descriptions.Item label={f('embeddingConfig')}>
          {renderConfigId(emb?.config_id)}
        </Descriptions.Item>
        <Descriptions.Item label={f('rerankConfig')}>
          {renderConfigId(rerank?.config_id)}
        </Descriptions.Item>
        <Descriptions.Item label={f('posNegMethod')}>
          {String(config.pos_neg_method || '-')}
        </Descriptions.Item>
        <Descriptions.Item label={f('outputFormat')}>
          {String(config.output_format || '-')}
        </Descriptions.Item>
        <Descriptions.Item label={f('timeoutPerDoc')}>
          {worker?.timeout_per_doc != null ? `${worker.timeout_per_doc}s` : '-'}
        </Descriptions.Item>
      </Descriptions>
      {stepTags.length > 0 && (
        <div>
          <Text type="secondary" style={{ fontSize: 12, marginBottom: 4, display: 'block' }}>{f('stepsConfig')}</Text>
          <Space size={[4, 4]} wrap>
            {stepTags.filter(s => s.enabled).map(s => (
              <Tooltip key={s.key} title={s.params}>
                <Tag color="green">
                  {s.label}{s.params ? ` (${s.params})` : ''}
                </Tag>
              </Tooltip>
            ))}
          </Space>
        </div>
      )}
    </>
  )
}

/* ---------- Training Config View ---------- */
function TrainingConfigView({ config, t }: { config: Record<string, unknown>; t: (key: string) => string }) {
  const f = (key: string) => t(`detail.configFields.${key}`)
  const lossConfig = config.loss_config as Record<string, unknown> | undefined

  const lossDisplay = () => {
    if (config.embedding_loss_name) {
      const name = String(config.embedding_loss_name)
      return name === 'auto' ? 'DynamicExplicitNegativesRankingLoss' : name
    }
    if (config.reranker_loss_name) return String(config.reranker_loss_name)
    if (lossConfig?.name) return `${lossConfig.name}${lossConfig.n_docs ? ` (n_docs=${lossConfig.n_docs})` : ''}`
    return '-'
  }

  const precisionDisplay = () => {
    if (config.bf16) return 'BF16'
    if (config.fp16) return 'FP16'
    return '-'
  }

  return (
    <Descriptions size="small" column={2}>
      <Descriptions.Item label={f('baseModelPath')} span={2}>
        <Text code style={{ fontSize: 11, wordBreak: 'break-all' }}>{String(config.base_model_path || '-')}</Text>
      </Descriptions.Item>
      <Descriptions.Item label={f('modelType')}>
        <Tag>{String(config.model_type || '-')}</Tag>
      </Descriptions.Item>
      <Descriptions.Item label={f('trainingMethod')}>
        <Tag>{String(config.training_method || '-')}</Tag>
      </Descriptions.Item>
      <Descriptions.Item label={`LoRA ${f('loraR')}`}>
        {String(config.lora_r ?? '-')}
      </Descriptions.Item>
      <Descriptions.Item label={`LoRA ${f('loraAlpha')}`}>
        {String(config.lora_alpha ?? '-')}
      </Descriptions.Item>
      <Descriptions.Item label={`LoRA ${f('loraDropout')}`}>
        {String(config.lora_dropout ?? '-')}
      </Descriptions.Item>
      <Descriptions.Item label={f('epochs')}>
        {String(config.num_train_epochs ?? '-')}
      </Descriptions.Item>
      <Descriptions.Item label={f('batchSize')}>
        {String(config.per_device_train_batch_size ?? '-')}
      </Descriptions.Item>
      <Descriptions.Item label={f('learningRate')}>
        {config.learning_rate != null ? String(config.learning_rate) : '-'}
      </Descriptions.Item>
      <Descriptions.Item label={f('warmupRatio')}>
        {String(config.warmup_ratio ?? '-')}
      </Descriptions.Item>
      <Descriptions.Item label={f('gradAccumSteps')}>
        {String(config.gradient_accumulation_steps ?? '-')}
      </Descriptions.Item>
      {config.model_type !== 'embedding' && (
        <Descriptions.Item label={f('maxLength')}>
          {config.max_length != null ? String(config.max_length) : '-'}
        </Descriptions.Item>
      )}
      <Descriptions.Item label={f('mixedPrecision')}>
        {precisionDisplay()}
      </Descriptions.Item>
      <Descriptions.Item label={f('gpuIds')}>
        {config.gpu_ids != null ? (Array.isArray(config.gpu_ids) ? (config.gpu_ids as number[]).join(', ') : String(config.gpu_ids)) : '-'}
      </Descriptions.Item>
      <Descriptions.Item label={f('lossFunction')}>
        {lossDisplay()}
      </Descriptions.Item>
    </Descriptions>
  )
}

export default function SyncConfigDetail() {
  const { taskId } = useParams<{ taskId: string }>()
  const navigate = useNavigate()
  const { t } = useTranslation(['sync', 'common'])

  const [config, setConfig] = useState<SyncConfig | null>(null)
  const [batches, setBatches] = useState<SyncBatch[]>([])
  const [generations, setGenerations] = useState<SyncGeneration[]>([])
  const [trainings, setTrainings] = useState<SyncTraining[]>([])
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [retryLoading, setRetryLoading] = useState<string | null>(null)

  const fetchAll = useCallback(async () => {
    if (!taskId) return
    try {
      const [configRes, batchRes, genRes, trainRes] = await Promise.all([
        syncApi.getTask(taskId),
        syncApi.listBatches(taskId, { limit: 20 }),
        syncApi.listGenerations(taskId, { limit: 20 }),
        syncApi.listTrainings(taskId, { limit: 20 }),
      ])
      setConfig(configRes.task)
      setBatches(batchRes.batches || [])
      setGenerations(genRes.generations || [])
      setTrainings(trainRes.trainings || [])
    } catch {
      // handled by interceptor
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => { fetchAll() }, [fetchAll])

  const isActive = config?.is_active && ACTIVE_SYNC_STATUSES.includes(config?.status || 'idle')
  usePolling(fetchAll, { interval: 5000, enabled: !!isActive })

  const handleAction = async (action: string) => {
    if (!taskId) return
    setActionLoading(action)
    try {
      switch (action) {
        case 'syncNow':
          await syncApi.syncNow(taskId)
          message.success(t('detail.message.syncSuccess'))
          break
        case 'triggerGeneration':
          await syncApi.triggerGeneration(taskId)
          message.success(t('detail.message.generationTriggered'))
          break
        case 'triggerTraining':
          await syncApi.triggerTraining(taskId)
          message.success(t('detail.message.trainingTriggered'))
          break
        case 'start':
          await syncApi.start(taskId)
          message.success(t('detail.message.startSuccess'))
          break
        case 'stop':
          await syncApi.stop(taskId)
          message.success(t('detail.message.stopSuccess'))
          break
      }
      fetchAll()
    } catch (err) {
      message.error(t(`detail.message.${action === 'syncNow' ? 'syncFailed' : action === 'triggerGeneration' ? 'generationFailed' : 'trainingFailed'}`))
    } finally {
      setActionLoading(null)
    }
  }

  if (loading) {
    return <div style={{ textAlign: 'center', padding: 100 }}><Spin size="large" /></div>
  }

  if (!config) {
    return <div>{t('common:message.loadFailed')}</div>
  }

  const genPercent = config.generation_threshold > 0
    ? Math.min(100, Math.round((config.pending_record_count / config.generation_threshold) * 100))
    : 0
  const trainPercent = config.training_threshold > 0
    ? Math.min(100, Math.round((config.pending_training_samples / config.training_threshold) * 100))
    : 0

  const batchColumns: ColumnsType<SyncBatch> = [
    {
      title: t('detail.batchColumns.batchId'),
      dataIndex: 'batch_id',
      key: 'batch_id',
      width: 100,
      render: (v: string) => <Text copyable={{ text: v }} style={{ fontSize: 12 }}>{v.slice(0, 8)}</Text>,
    },
    {
      title: t('detail.batchColumns.recordCount'),
      dataIndex: 'record_count',
      key: 'record_count',
      width: 100,
    },
    {
      title: t('detail.batchColumns.dataset'),
      dataIndex: 'dataset_id',
      key: 'dataset_id',
      width: 120,
      render: (v: string | null) => v
        ? <Button type="link" size="small" onClick={() => navigate(`/datasets/${v}`)}>{v.slice(0, 8)}</Button>
        : <span style={{ color: TEXT_SECONDARY }}>-</span>,
    },
    {
      title: t('detail.batchColumns.timeRange'),
      key: 'timeRange',
      render: (_, record) => (
        <span style={{ fontSize: 12 }}>
          {record.since_time ? formatDate(record.since_time) : '-'}
          {' ~ '}
          {record.until_time ? formatDate(record.until_time) : '-'}
        </span>
      ),
    },
    {
      title: t('detail.batchColumns.fetchedAt'),
      dataIndex: 'fetched_at',
      key: 'fetched_at',
      width: 160,
      render: (v: string) => <span style={{ fontSize: 12 }}>{formatDate(v)}</span>,
    },
    {
      title: t('detail.batchColumns.status'),
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => <StatusTag status={status} />,
    },
  ]

  const handleToggleDisabled = async (gen: SyncGeneration) => {
    if (!taskId) return
    try {
      const result = await syncApi.toggleGenerationDisabled(taskId, gen.generation_task_id, !gen.disabled)
      const resetCount = result.reset_batch_count
      const resetRecords = result.reset_record_count
      if (resetCount && resetRecords) {
        message.success(t('detail.generationColumns.disabledWithReset', { count: resetCount, records: resetRecords }))
      } else {
        message.success(gen.disabled ? t('detail.generationColumns.enableSuccess') : t('detail.generationColumns.disableSuccess'))
      }
      fetchAll()
    } catch {
      message.error(t('common:message.operationFailed', { error: '' }))
    }
  }

  const generationColumns: ColumnsType<SyncGeneration> = [
    {
      title: t('detail.generationColumns.taskId'),
      dataIndex: 'generation_task_id',
      key: 'taskId',
      width: 120,
      render: (v: string) => (
        <Button type="link" size="small" onClick={() => navigate(`/datasets/generation/${v}`)}>
          {v.slice(0, 8)}
        </Button>
      ),
    },
    {
      title: t('detail.generationColumns.inputRecords'),
      dataIndex: 'input_record_count',
      key: 'inputRecords',
      width: 100,
    },
    {
      title: t('detail.generationColumns.outputSamples'),
      dataIndex: 'output_sample_count',
      key: 'outputSamples',
      width: 100,
      render: (count: number, record: SyncGeneration) => record.output_dataset_id
        ? <Button type="link" size="small" onClick={() => navigate(`/datasets/${record.output_dataset_id}`)}>{count}</Button>
        : count,
    },
    {
      title: t('detail.generationColumns.status'),
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string, record: SyncGeneration) => (
        <Space size={4}>
          <StatusTag status={status} />
          {record.disabled && <Tag color="default">{t('detail.generationColumns.disabled')}</Tag>}
        </Space>
      ),
    },
    {
      title: t('detail.generationColumns.createdAt'),
      dataIndex: 'created_at',
      key: 'createdAt',
      width: 160,
      render: (v: string) => <span style={{ fontSize: 12 }}>{formatDate(v)}</span>,
    },
    {
      title: t('detail.generationColumns.action'),
      key: 'action',
      width: 100,
      render: (_: unknown, record: SyncGeneration) => {
        if (record.status !== SYNC_GENERATION_TOGGLE_STATUS) return null
        return record.disabled ? (
          <Button type="link" size="small" onClick={() => handleToggleDisabled(record)}>
            {t('detail.generationColumns.enable')}
          </Button>
        ) : (
          <Popconfirm
            title={t('detail.generationColumns.disableConfirm')}
            onConfirm={() => handleToggleDisabled(record)}
          >
            <Button type="link" size="small" danger>
              {t('detail.generationColumns.disable')}
            </Button>
          </Popconfirm>
        )
      },
    },
  ]

  const handleRetryAdapterLoad = async (trainingTaskId: string, replace = true) => {
    if (!taskId) return
    setRetryLoading(trainingTaskId)
    try {
      await syncApi.retryAdapterLoad(taskId, trainingTaskId, replace)
      message.success(t('detail.retryAdapterLoadStarted'))
      fetchAll()
    } catch {
      message.error(t('detail.retryAdapterLoadFailed'))
    } finally {
      setRetryLoading(null)
    }
  }

  const handleUnloadAdapter = async () => {
    if (!taskId) return
    setActionLoading('unloadAdapter')
    try {
      await syncApi.unloadAdapter(taskId)
      message.success(t('detail.adapterUnloaded'))
      fetchAll()
    } catch {
      message.error(t('detail.unloadAdapterFailed'))
    } finally {
      setActionLoading(null)
    }
  }

  const handleRecalculateCounters = async () => {
    if (!taskId) return
    setActionLoading('recalculate')
    try {
      const result = await syncApi.recalculateCounters(taskId)
      const { old_total_training_samples, new_total_training_samples, old_pending_training_samples, new_pending_training_samples } = result
      if (old_total_training_samples === new_total_training_samples && old_pending_training_samples === new_pending_training_samples) {
        message.info(t('detail.recalculate.noChange'))
      } else {
        message.success(t('detail.recalculate.success', {
          oldTotal: old_total_training_samples,
          newTotal: new_total_training_samples,
          oldPending: old_pending_training_samples,
          newPending: new_pending_training_samples,
        }))
      }
      fetchAll()
    } catch {
      message.error(t('detail.recalculate.failed'))
    } finally {
      setActionLoading(null)
    }
  }


  const trainingColumns: ColumnsType<SyncTraining> = [
    {
      title: t('detail.trainingColumns.taskId'),
      dataIndex: 'training_task_id',
      key: 'taskId',
      width: 120,
      render: (v: string) => (
        <Button type="link" size="small" onClick={() => navigate(`/training/${v}`)}>
          {v.slice(0, 8)}
        </Button>
      ),
    },
    {
      title: t('detail.trainingColumns.target'),
      dataIndex: 'target_id',
      key: 'target',
      width: 120,
      render: (_: unknown, record: SyncTraining) => {
        const target = config.training_targets?.find(
          (tt: SyncTrainingTarget) => tt.target_id === record.target_id
        )
        if (!target) return <span style={{ color: TEXT_SECONDARY }}>-</span>
        return (
          <Tag color={target.model_type === 'llm' ? 'blue' : target.model_type === 'embedding' ? 'green' : 'orange'}>
            {target.target_name}
          </Tag>
        )
      },
    },
    {
      title: t('detail.trainingColumns.round'),
      dataIndex: 'training_round',
      key: 'round',
      width: 80,
      render: (v: number) => <Tag>R{v}</Tag>,
    },
    {
      title: t('detail.trainingColumns.totalSamples'),
      dataIndex: 'total_samples',
      key: 'totalSamples',
      width: 100,
    },
    {
      title: t('detail.trainingColumns.trainingStatus'),
      dataIndex: 'status',
      key: 'trainingStatus',
      width: 110,
      render: (status: string) => {
        // Derive training status from the composite sync training status
        const trainingSucceeded = SYNC_TRAINING_SUCCEEDED_STATUSES.includes(status as SyncTraining['status'])
        return <StatusTag status={trainingSucceeded ? 'succeeded' : status} />
      },
    },
    {
      title: t('detail.trainingColumns.adapter'),
      dataIndex: 'loaded_adapter_name',
      key: 'adapter',
      width: 150,
      render: (name: string | null, record: SyncTraining) => name
        ? <Tooltip title={record.loaded_adapter_id}>
            <Tag color="blue">{name}</Tag>
          </Tooltip>
        : <span style={{ color: TEXT_SECONDARY }}>-</span>,
    },
    {
      title: t('detail.trainingColumns.adapterStatus'),
      dataIndex: 'status',
      key: 'adapterStatus',
      width: 140,
      render: (status: string, record: SyncTraining) => {
        if (status === 'adapter_loaded') {
          const tip = [record.loaded_adapter_name, record.loaded_adapter_id?.slice(0, 8)].filter(Boolean).join(' / ')
          return (
            <Tooltip title={tip}>
              <StatusTag status="adapter_loaded" />
            </Tooltip>
          )
        }
        if (status === 'adapter_unloaded') {
          return <StatusTag status="adapter_unloaded" />
        }
        if (status === 'adapter_load_failed' || status === 'adapter_failed') {
          return <StatusTag status="adapter_load_failed" />
        }
        if (status === 'completed') {
          // Training succeeded, no deployment was found to load adapter
          return <StatusTag status="adapter_not_loaded" />
        }
        // pending / running / failed (training itself) → no adapter status
        return <span style={{ color: TEXT_SECONDARY, fontSize: 12 }}>—</span>
      },
    },
    {
      title: t('detail.trainingColumns.actions'),
      key: 'actions',
      width: 100,
      render: (_, record: SyncTraining) => {
        const isLoading = retryLoading === record.training_task_id
        const canLoad = SYNC_TRAINING_CAN_LOAD_STATUSES.includes(record.status)
            || (record.status === 'adapter_loaded' && record.loaded_adapter_name !== config.current_adapter_name)

        if (canLoad) {
          return (
            <Space size={4}>
              <Tooltip title={t('detail.loadAdapter')}>
                <Button
                  type="text"
                  size="small"
                  icon={<PlusOutlined />}
                  loading={isLoading}
                  onClick={() => handleRetryAdapterLoad(record.training_task_id, false)}
                />
              </Tooltip>
              <Tooltip title={t('detail.replaceAdapter')}>
                <Button
                  type="text"
                  size="small"
                  icon={<SwapOutlined />}
                  loading={isLoading}
                  onClick={() => handleRetryAdapterLoad(record.training_task_id, true)}
                />
              </Tooltip>
            </Space>
          )
        }
        if (record.status === 'adapter_loaded' && record.loaded_adapter_name
            && record.loaded_adapter_name === config.current_adapter_name) {
          return (
            <Popconfirm
              title={t('detail.confirmUnloadAdapter')}
              onConfirm={handleUnloadAdapter}
            >
              <Tooltip title={t('detail.unloadAdapter')}>
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DisconnectOutlined />}
                  loading={actionLoading === 'unloadAdapter'}
                />
              </Tooltip>
            </Popconfirm>
          )
        }
        return null
      },
    },
    {
      title: t('detail.trainingColumns.createdAt'),
      dataIndex: 'created_at',
      key: 'createdAt',
      width: 160,
      render: (v: string) => <span style={{ fontSize: 12 }}>{formatDate(v)}</span>,
    },
  ]

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/sync')}>
            {t('common:action.back')}
          </Button>
          <Title level={4} style={{ margin: 0 }}>{config.task_name}</Title>
          <StatusTag status={config.status} />
          {!config.is_active && <Tag color="default">{t('common:status.disabled')}</Tag>}
        </Space>
        <Space>
          <Button
            icon={<EditOutlined />}
            onClick={() => navigate(`/sync/${taskId}/edit`)}
          >
            {t('detail.actions.edit')}
          </Button>
          <Button
            icon={<SyncOutlined />}
            onClick={() => handleAction('syncNow')}
            loading={actionLoading === 'syncNow'}
          >
            {t('detail.actions.syncNow')}
          </Button>
          <Button
            icon={<ThunderboltOutlined />}
            onClick={() => handleAction('triggerGeneration')}
            loading={actionLoading === 'triggerGeneration'}
          >
            {t('detail.actions.triggerGeneration')}
          </Button>
          <Button
            icon={<ExperimentOutlined />}
            onClick={() => handleAction('triggerTraining')}
            loading={actionLoading === 'triggerTraining'}
          >
            {t('detail.actions.triggerTraining')}
          </Button>
          {config.is_active ? (
            <Popconfirm title={t('list.actions.confirmStop')} onConfirm={() => handleAction('stop')}>
              <Button danger icon={<PauseCircleOutlined />} loading={actionLoading === 'stop'}>
                {t('detail.actions.stop')}
              </Button>
            </Popconfirm>
          ) : (
            <Button type="primary" icon={<PlayCircleOutlined />}
              onClick={() => handleAction('start')} loading={actionLoading === 'start'}>
              {t('detail.actions.start')}
            </Button>
          )}
        </Space>
      </div>

      {/* Counter Cards */}
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={6}>
          <Card size="small" style={{ background: BG_ELEVATED, borderColor: BORDER_SECONDARY }}>
            <div style={{ marginBottom: 8, color: TEXT_SECONDARY, fontSize: 12 }}>
              {t('detail.fields.pendingRecords')} / {t('detail.fields.genThreshold')}
            </div>
            <Progress
              percent={genPercent}
              strokeColor={STATUS_INFO}
              format={() => `${config.pending_record_count} / ${config.generation_threshold}`}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small" style={{ background: BG_ELEVATED, borderColor: BORDER_SECONDARY }}>
            <div style={{ marginBottom: 8, color: TEXT_SECONDARY, fontSize: 12 }}>
              {t('detail.fields.pendingTrainingSamples')} / {t('detail.fields.trainThreshold')}
            </div>
            <Progress
              percent={trainPercent}
              strokeColor={STATUS_SUCCESS}
              format={() => `${config.pending_training_samples} / ${config.training_threshold}`}
            />
          </Card>
        </Col>
        <Col span={4}>
          <StatCard
            title={t('detail.fields.totalRecords')}
            value={config.total_record_count}
            icon={<SyncOutlined />}
            color={STATUS_INFO}
          />
        </Col>
        <Col span={4}>
          <Tooltip title={t('detail.recalculate.tooltip')}>
            <div
              style={{ cursor: 'pointer' }}
              onClick={handleRecalculateCounters}
            >
              <StatCard
                title={t('detail.fields.totalTrainingSamples')}
                value={config.total_training_samples}
                icon={<ThunderboltOutlined />}
                color={STATUS_WARNING}
                loading={actionLoading === 'recalculate'}
              />
            </div>
          </Tooltip>
        </Col>
        <Col span={4}>
          <StatCard
            title={t('detail.fields.totalTrainings')}
            value={config.total_trainings}
            icon={<ExperimentOutlined />}
            color={STATUS_SUCCESS}
          />
        </Col>
      </Row>

      {/* Training Targets Status */}
      {config.training_targets && config.training_targets.length > 0 && (
        <Card title={t('detail.trainingTargets')} size="small" style={{ marginBottom: 16 }}>
          <Row gutter={[16, 16]}>
            {config.training_targets.map((target: SyncTrainingTarget) => {
              const targetTrainPercent = target.training_threshold > 0
                ? Math.min(100, Math.round((target.pending_training_samples / target.training_threshold) * 100))
                : 0
              return (
                <Col key={target.target_id} span={8}>
                  <Card size="small" bordered>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Space>
                        <Tag color={
                          target.model_type === 'llm' ? 'blue' :
                          target.model_type === 'embedding' ? 'green' :
                          'orange'
                        }>
                          {target.model_type.toUpperCase()}
                        </Tag>
                        <span>{target.target_name}</span>
                        <Tag>{target.status}</Tag>
                      </Space>
                      <Progress
                        percent={targetTrainPercent}
                        size="small"
                      />
                      <Text type="secondary">
                        {target.pending_training_samples} / {target.training_threshold}
                      </Text>
                      {target.current_adapter_name && (
                        <Text type="secondary">Adapter: {target.current_adapter_name}</Text>
                      )}
                    </Space>
                  </Card>
                </Col>
              )
            })}
          </Row>
        </Card>
      )}

      {/* Overview */}
      <Card title={t('detail.overview')} size="small" style={{ marginBottom: 16 }}>
        <Descriptions column={3} size="small">
          <Descriptions.Item label={t('detail.fields.configId')}>
            <Text copyable style={{ fontSize: 12 }}>{config.task_id}</Text>
          </Descriptions.Item>
          <Descriptions.Item label={t('detail.fields.apiUrl')}>
            {config.external_api_url}
          </Descriptions.Item>
          <Descriptions.Item label={t('detail.fields.syncInterval')}>
            {config.sync_interval_seconds}s
          </Descriptions.Item>
          <Descriptions.Item label={t('detail.fields.lastSync')}>
            {config.last_sync_at ? formatDate(config.last_sync_at) : '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('detail.fields.genThreshold')}>
            {config.generation_threshold}
          </Descriptions.Item>
          <Descriptions.Item label={t('detail.fields.trainThreshold')}>
            {config.training_threshold}
          </Descriptions.Item>
          <Descriptions.Item label={t('detail.adapterInfo')} span={config.error_message ? 1 : 3}>
            {config.current_adapter_name
              ? <Space>
                  <Tag color="blue">{config.current_adapter_name}</Tag>
                  <Popconfirm
                    title={t('detail.confirmUnloadAdapter')}
                    onConfirm={handleUnloadAdapter}
                  >
                    <Button
                      type="link"
                      size="small"
                      danger
                      loading={actionLoading === 'unloadAdapter'}
                    >
                      {t('detail.unloadAdapter')}
                    </Button>
                  </Popconfirm>
                </Space>
              : <span style={{ color: TEXT_SECONDARY }}>{t('detail.noAdapter')}</span>}
          </Descriptions.Item>
          {config.error_message && (
            <Descriptions.Item label="Error" span={2}>
              <Text type="danger" style={{ fontSize: 12 }}>{config.error_message}</Text>
            </Descriptions.Item>
          )}
        </Descriptions>
      </Card>

      {/* Config Cards */}
      {(config.generation_config || config.training_config || (config.training_targets && config.training_targets.length > 0)) && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          {config.generation_config && (
            <Col span={(config.training_config || (config.training_targets && config.training_targets.length > 0)) ? 12 : 24}>
              <Card title={t('detail.generationConfig')} size="small">
                <GenerationConfigView config={config.generation_config} t={t} />
              </Card>
            </Col>
          )}
          {config.training_targets && config.training_targets.length > 0 ? (
            <Col span={config.generation_config ? 12 : 24}>
              <Card title={t('detail.trainingConfig')} size="small">
                <Tabs
                  items={config.training_targets.map((target: SyncTrainingTarget) => ({
                    key: target.target_id,
                    label: (
                      <Space size={4}>
                        <Tag color={target.model_type === 'llm' ? 'blue' : target.model_type === 'embedding' ? 'green' : 'orange'}>
                          {target.model_type.toUpperCase()}
                        </Tag>
                        {target.target_name}
                      </Space>
                    ),
                    children: <TrainingConfigView config={{
                      ...target.training_config,
                      base_model_path: target.base_model_path,
                      model_type: target.model_type,
                      training_method: target.training_method,
                    }} t={t} />,
                  }))}
                />
              </Card>
            </Col>
          ) : config.training_config ? (
            <Col span={config.generation_config ? 12 : 24}>
              <Card title={t('detail.trainingConfig')} size="small">
                <TrainingConfigView config={config.training_config} t={t} />
              </Card>
            </Col>
          ) : null}
        </Row>
      )}

      {/* History tabs */}
      <Tabs
        defaultActiveKey="batches"
        items={[
          {
            key: 'batches',
            label: `${t('detail.batchHistory')} (${batches.length})`,
            children: (
              <Table
                rowKey="batch_id"
                columns={batchColumns}
                dataSource={batches}
                size="small"
                pagination={false}
                scroll={{ x: 800 }}
              />
            ),
          },
          {
            key: 'generations',
            label: `${t('detail.generationHistory')} (${generations.length})`,
            children: (
              <Table
                rowKey="generation_task_id"
                columns={generationColumns}
                dataSource={generations}
                size="small"
                pagination={false}
                scroll={{ x: 700 }}
                rowClassName={(record: SyncGeneration) => record.disabled ? 'ant-table-row-excluded' : ''}
              />
            ),
          },
          {
            key: 'trainings',
            label: `${t('detail.trainingHistory')} (${trainings.length})`,
            children: (
              <Table
                rowKey="training_task_id"
                columns={trainingColumns}
                dataSource={trainings}
                size="small"
                pagination={false}
                scroll={{ x: 900 }}
              />
            ),
          },
        ]}
      />
    </div>
  )
}
