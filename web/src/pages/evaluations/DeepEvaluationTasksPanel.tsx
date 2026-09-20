import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  InputNumber,
  Modal,
  Popconfirm,
  Progress,
  Radio,
  Row,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  DeleteOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
  StopOutlined,
  RedoOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { deepEvaluationApi, datasetApi, SILENT_REQUEST_CONFIG } from '@/services/api'
import { usePolling } from '@/hooks/usePolling'
import { formatDate } from '@/utils'
import type { Dataset, DeepEvaluationModelGroupConfig, DeepEvaluationTask } from '@/types'
import { StatCard } from '@/components/StatCard'
import { STATUS_ERROR, STATUS_INFO, STATUS_SUCCESS, STATUS_WARNING } from '@/theme'
import ModelConfigSelector from '@/components/ModelConfigSelector'
import type { ModelConfig } from '@/types'

const { Title, Text } = Typography
const MAX_LLM_TOKENS = 131072

const statusColorMap: Record<string, string> = {
  pending: 'default',
  running: 'processing',
  completed: 'success',
  failed: 'error',
  cancelled: 'warning',
}

// statusLabelMap is now derived from t() inside the component

// Traditional information retrieval metrics (values only; labels use i18n inside component)
const TRADITIONAL_METRICS_VALUES = [
  { value: 'mrr', label: 'MRR', descriptionKey: 'metrics.mrr', category: 'retrieval' },
  { value: 'map', label: 'MAP', descriptionKey: 'metrics.map', category: 'retrieval' },
  { value: 'ndcg@10', label: 'NDCG@10', descriptionKey: 'metrics.ndcg10', category: 'retrieval' },
  { value: 'recall@10', label: 'Recall@10', descriptionKey: 'metrics.recall10', category: 'retrieval' },
  { value: 'precision@10', label: 'Precision@10', descriptionKey: 'metrics.precision10', category: 'retrieval' },
]

// Default metrics (Traditional IR)
const DEFAULT_METRICS = ['mrr', 'ndcg@10']

interface ResultSummary {
  overall?: {
    mean?: number
    min?: number
    max?: number
  }
  metrics?: Record<string, {
    mean?: number
    min?: number
    max?: number
    count?: number
  }>
}

interface MetricInfo {
  name: string
  description: string
  category: string
  requires_llm: boolean
  requires_expected_output?: boolean
  requires_actual_output?: boolean
  requires_retrieval_context?: boolean
}

// 模型组配置
interface ModelGroup {
  group_id: string
  group_name: string
  embedding?: ModelConfig | null
  rerank?: ModelConfig | null
  llm?: ModelConfig | null
}

const formatPercent = (value: unknown) => {
  const num = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(num)) return '-'
  return `${(num * 100).toFixed(1)}%`
}

const guessField = (columns: string[], candidates: string[]) => {
  const lowerMap = new Map(columns.map((c) => [c.toLowerCase(), c]))
  for (const candidate of candidates) {
    const found = lowerMap.get(candidate.toLowerCase())
    if (found) return found
  }
  return undefined
}

const deriveColumns = (dataset?: Dataset | null) => {
  if (!dataset) return []
  if (dataset.columns?.length) {
    return dataset.columns.map((c) => (typeof c === 'string' ? c : c.name))
  }
  const sample = dataset.sample_data?.[0]
  if (sample && typeof sample === 'object') {
    return Object.keys(sample)
  }
  return []
}

export default function DeepEvaluationTasksPanel() {
  const { t } = useTranslation(['evaluations', 'common'])
  const [tasks, setTasks] = useState<DeepEvaluationTask[]>([])
  const [tasksTotal, setTasksTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [tasksStale, setTasksStale] = useState(false)
  const tasksRequestGenerationRef = useRef(0)
  const tasksLoadingOwnerRef = useRef(0)
  const datasetsRequestGenerationRef = useRef(0)
  const metricsRequestGenerationRef = useRef(0)
  const evalReadyRequestGenerationRef = useRef(0)
  const mountedRef = useRef(true)
  const [createVisible, setCreateVisible] = useState(false)
  const [detailVisible, setDetailVisible] = useState(false)
  const [selectedTask, setSelectedTask] = useState<DeepEvaluationTask | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetColumns, setDatasetColumns] = useState<string[]>([])

  const statusLabelMap: Record<string, string> = useMemo(() => ({
    pending: t('common:status.pending'),
    running: t('common:status.running'),
    completed: t('common:status.completed'),
    failed: t('common:status.failed'),
    cancelled: t('common:status.cancelled'),
  }), [t])

  const TRADITIONAL_METRICS = useMemo(() => TRADITIONAL_METRICS_VALUES.map((m) => ({
    ...m,
    description: t(m.descriptionKey),
  })), [t])

  const createGroup = (index: number): ModelGroup => ({
    group_id: `group_${Date.now()}_${index}`,
    group_name: t('deep.modelGroupDefault', { index: index + 1 }),
    embedding: null,
    rerank: null,
    llm: null,
  })
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>(() => [createGroup(0)])
  const [llmMetrics, setLlmMetrics] = useState<MetricInfo[]>([])
  const [evalReadyDatasets, setEvalReadyDatasets] = useState<Array<{
    collection_name: string
    display_name?: string
    collection_id: string
    embedding_config_id?: string
    embedding_model?: string
    linked_datasets?: Array<{ dataset_id: string; dataset_name?: string; chunk_count?: number }>
    // Backward compat
    milvus_collection: string
    embedding_config?: { config_id?: string; endpoint?: string; model?: string }
  }>>([])

  const [form] = Form.useForm()
  const selectedMetrics = Form.useWatch('metrics', form)
  const retrievalMode = Form.useWatch('retrieval_mode', form) || 'offline'
  const selectedCollectionName = Form.useWatch('milvus_collection', form)
  const selectedRetrievalEmbeddingConfigId = Form.useWatch('retrieval_embedding_config_id', form)
  const selectedMetricsRef = useMemo(() => selectedMetrics || [], [selectedMetrics])

  const datasetMap = useMemo(() => {
    return datasets.reduce<Record<string, Dataset>>((acc, dataset) => {
      acc[dataset.dataset_id] = dataset
      return acc
    }, {})
  }, [datasets])

  const fetchTasks = useCallback(async (silent = false, showError = true) => {
    const requestGeneration = ++tasksRequestGenerationRef.current
    const loadingOwner = silent ? null : ++tasksLoadingOwnerRef.current
    if (loadingOwner !== null) setLoading(true)
    try {
      const res = await deepEvaluationApi.listTasks(
        { limit: 50 },
        SILENT_REQUEST_CONFIG
      )
      if (requestGeneration !== tasksRequestGenerationRef.current) return true
      setTasks(res.items || [])
      setTasksTotal(res.total || 0)
      setSelectedTask((prev) => {
        if (!prev) return null
        return res.items?.find((t) => t.task_id === prev.task_id) || prev
      })
      setTasksStale(false)
      return true
    } catch {
      if (requestGeneration !== tasksRequestGenerationRef.current) return true
      if (silent) {
        setTasksStale(true)
      } else if (showError) {
        message.error({
          key: 'deep-evaluation-load-failed',
          content: t('deep.fetchTasksFailed'),
        })
      }
      return false
    } finally {
      if (
        loadingOwner !== null &&
        loadingOwner === tasksLoadingOwnerRef.current &&
        mountedRef.current
      ) {
        setLoading(false)
      }
    }
  }, [t])

  const fetchDatasets = useCallback(async (showError = true) => {
    const requestGeneration = ++datasetsRequestGenerationRef.current
    try {
      const res = await datasetApi.list(
        { page: 1, page_size: 200 },
        SILENT_REQUEST_CONFIG
      )
      if (requestGeneration !== datasetsRequestGenerationRef.current) return true
      setDatasets(res.items || [])
      return true
    } catch {
      if (requestGeneration !== datasetsRequestGenerationRef.current) return true
      if (showError) {
        message.error({
          key: 'deep-evaluation-load-failed',
          content: t('deep.loadDatasetsFailed'),
        })
      }
      return false
    }
  }, [t])

  const fetchMetrics = useCallback(async (showError = true) => {
    const requestGeneration = ++metricsRequestGenerationRef.current
    try {
      const res = await deepEvaluationApi.getMetrics(undefined, SILENT_REQUEST_CONFIG)
      if (requestGeneration !== metricsRequestGenerationRef.current) return true
      setLlmMetrics(res || [])
      return true
    } catch {
      if (requestGeneration !== metricsRequestGenerationRef.current) return true
      if (showError) {
        message.error({
          key: 'deep-evaluation-load-failed',
          content: t('deep.loadMetricsFailed'),
        })
      }
      return false
    }
  }, [t])

  const fetchEvalReadyDatasets = useCallback(async () => {
    const requestGeneration = ++evalReadyRequestGenerationRef.current
    try {
      const res = await deepEvaluationApi.getEvalReadyDatasets(SILENT_REQUEST_CONFIG)
      if (requestGeneration !== evalReadyRequestGenerationRef.current) return
      setEvalReadyDatasets(res || [])
    } catch {
      // 非关键功能，静默失败
    }
  }, [])

  useEffect(() => {
    let active = true
    void Promise.all([
      fetchTasks(false, false),
      fetchDatasets(false),
      fetchMetrics(false),
    ]).then((results) => {
      if (active && results.some((succeeded) => !succeeded)) {
        message.error({
          key: 'deep-evaluation-load-failed',
          content: t('deep.initialLoadFailed'),
        })
      }
    })
    void fetchEvalReadyDatasets()
    return () => {
      active = false
    }
  }, [t, fetchTasks, fetchDatasets, fetchMetrics, fetchEvalReadyDatasets])

  useEffect(
    () => {
      mountedRef.current = true
      return () => {
        mountedRef.current = false
        tasksRequestGenerationRef.current += 1
        tasksLoadingOwnerRef.current += 1
        datasetsRequestGenerationRef.current += 1
        metricsRequestGenerationRef.current += 1
        evalReadyRequestGenerationRef.current += 1
      }
    },
    []
  )

  const hasRunningTasks = useMemo(
    () => tasks.some((t) => t.status === 'running' || t.status === 'pending'),
    [tasks]
  )

  usePolling(() => fetchTasks(true).then(() => undefined), {
    interval: 3000,
    enabled: hasRunningTasks,
  })

  const stats = useMemo(() => {
    const completed = tasks.filter((t) => t.status === 'completed').length
    const failed = tasks.filter((t) => t.status === 'failed').length
    const cancelled = tasks.filter((t) => t.status === 'cancelled').length
    const running = tasks.filter((t) => t.status === 'running' || t.status === 'pending').length
    return { total: tasksTotal, completed, failed, cancelled, running }
  }, [tasks, tasksTotal])

  const datasetOptions = datasets.map((dataset) => ({
    label: dataset.dataset_name,
    value: dataset.dataset_id,
  }))

  const llmMetricMap = useMemo(() => {
    return new Map(llmMetrics.map((metric) => [metric.name, metric]))
  }, [llmMetrics])

  const selectedEvalCollection = useMemo(
    () => evalReadyDatasets.find((d) => d.collection_name === selectedCollectionName),
    [evalReadyDatasets, selectedCollectionName]
  )

  const collectionModelMismatch = useMemo(() => {
    if (!selectedEvalCollection || !selectedRetrievalEmbeddingConfigId) return false
    const collectionConfigId = selectedEvalCollection.embedding_config_id
    if (!collectionConfigId) return false
    return collectionConfigId !== selectedRetrievalEmbeddingConfigId
  }, [selectedEvalCollection, selectedRetrievalEmbeddingConfigId])

  const hasRetrievalMetrics = useMemo(() => {
    return selectedMetricsRef.some((metric: string) => TRADITIONAL_METRICS.some((m) => m.value === metric))
  }, [selectedMetricsRef, TRADITIONAL_METRICS])

  const llmMetricSelection = useMemo(() => {
    return selectedMetricsRef
      .map((metric: string) => llmMetricMap.get(metric))
      .filter(Boolean) as MetricInfo[]
  }, [selectedMetricsRef, llmMetricMap])

  const hasLLMMetrics = useMemo(() => {
    return llmMetricSelection.some((metric) => metric.requires_llm)
  }, [llmMetricSelection])

  const llmFieldRequirements = useMemo(() => {
    return {
      expected_output: llmMetricSelection.some((m) => m.requires_expected_output),
      actual_output: llmMetricSelection.some((m) => m.requires_actual_output),
      retrieval_context: llmMetricSelection.some((m) => m.requires_retrieval_context),
    }
  }, [llmMetricSelection])

  const renderTags = (labels: string[]) => {
    if (!labels.length) return '-'
    const shown = labels.slice(0, 2)
    return (
      <Space size={[0, 4]} wrap>
        {shown.map((label) => (
          <Tag key={label} color="blue">{label}</Tag>
        ))}
        {labels.length > 2 && <Tag>+{labels.length - 2}</Tag>}
      </Space>
    )
  }

  const updateFieldMappingDefaults = (dataset: Dataset) => {
    const columnNames = deriveColumns(dataset)
    setDatasetColumns(columnNames)

    const mapping = {
      query: guessField(columnNames, ['query', 'question', 'input', 'prompt']),
      positives: guessField(columnNames, ['positives', 'positive', 'pos', 'relevant']),
      negatives: guessField(columnNames, ['negatives', 'negative', 'neg', 'irrelevant']),
      input: guessField(columnNames, ['input', 'query', 'question', 'prompt']),
      expected_output: guessField(columnNames, ['expected_output', 'expected', 'answer', 'reference', 'label']),
      actual_output: guessField(columnNames, ['actual_output', 'output', 'prediction', 'response', 'answer']),
      retrieval_context: guessField(columnNames, ['retrieval_context', 'context', 'contexts', 'passages', 'documents']),
    }

    form.setFieldsValue({ field_mapping: mapping })
  }

  const handleDatasetChange = async (datasetIds: string[]) => {
    if (!datasetIds.length) return
    const datasetId = datasetIds[0]
    try {
      const dataset = await datasetApi.get(datasetId)
      updateFieldMappingDefaults(dataset)
    } catch {
      message.error(t('deep.loadFieldsFailed'))
    }
  }

  const handleCreateTask = async () => {
    try {
      // 验证模型组配置
      const validGroups = modelGroups.filter((g) => g.embedding || g.rerank || g.llm)
      if (validGroups.length === 0) {
        message.warning(t('deep.atLeastOneGroup'))
        return
      }
      if (hasLLMMetrics && !validGroups.some((g) => g.llm)) {
        message.warning(t('deep.llmMetricsNeedLLM'))
        return
      }
      const values = await form.validateFields()
      const datasetIds: string[] = values.dataset_ids || []
      if (!datasetIds.length) {
        message.warning(t('deep.atLeastOneDataset'))
        return
      }

      // 构建按组的 model_configs（新格式）
      const modelConfigs = modelGroups
        .map((group) => ({ group }))
        .filter(({ group }) => group.embedding || group.rerank || group.llm)
        .map(({ group }) => {
          const groupValues = values.groups?.[group.group_id] || {}
          const cfg: DeepEvaluationModelGroupConfig = {
            group_name: group.group_name,
          }
          if (group.embedding) {
            cfg.embedding = {
              config_id: group.embedding.config_id,
              concurrency: groupValues.embedding?.concurrency || 8,
            }
          }
          if (group.rerank) {
            cfg.rerank = {
              config_id: group.rerank.config_id,
              concurrency: groupValues.rerank?.concurrency || 4,
            }
          }
          if (group.llm) {
            cfg.llm = {
              config_id: group.llm.config_id,
              temperature: groupValues.llm?.temperature ?? 0,
              top_p: groupValues.llm?.top_p ?? 1,
              top_k: groupValues.llm?.top_k ?? undefined,
              max_tokens: groupValues.llm?.max_tokens ?? 2048,
              timeout: groupValues.llm?.timeout ?? 60,
              max_retries: groupValues.llm?.max_retries ?? 3,
              concurrency: groupValues.llm?.concurrency ?? 2,
            }
          }
          return cfg
        })

      const createPayload: Record<string, unknown> = {
        task_name: values.task_name,
        description: values.description,
        model_configs: modelConfigs,
        dataset_configs: datasetIds.map((datasetId: string) => ({ dataset_id: datasetId })),
        max_samples: values.max_samples || undefined,
        field_mapping: values.field_mapping,
        metrics: values.metrics,
        model_workers: values.model_workers || 2,
        chunk_eval_mode: values.chunk_eval_mode || 'batch',
        retrieval_mode: values.retrieval_mode || 'offline',
      }
      if (values.retrieval_mode === 'online') {
        const selectedCollection = evalReadyDatasets.find(
          (d) => d.collection_name === values.milvus_collection
        )
        const mismatch = Boolean(
          selectedCollection?.embedding_config_id &&
          values.retrieval_embedding_config_id &&
          selectedCollection.embedding_config_id !== values.retrieval_embedding_config_id
        )
        if (mismatch && !values.allow_collection_model_mismatch) {
          message.warning(t('deep.modelMismatchNeedConfirm'))
          return
        }

        createPayload.milvus_collection = values.milvus_collection
        createPayload.retrieval_top_k = values.retrieval_top_k || 20
        createPayload.allow_collection_model_mismatch = Boolean(values.allow_collection_model_mismatch)
        if (values.retrieval_embedding_config_id) {
          createPayload.retrieval_embedding_config = {
            config_id: values.retrieval_embedding_config_id,
          }
        }
      }
      await deepEvaluationApi.createTask(createPayload)
      message.success(t('deep.taskCreated'))
      setCreateVisible(false)
      form.resetFields()
      setModelGroups([createGroup(0)])
      fetchTasks()
    } catch (error) {
      const formError = error as { errorFields?: unknown } | undefined
      if (formError?.errorFields) return
      message.error(t('deep.createTaskFailed'))
    }
  }

  const handleCancelTask = async (taskId: string) => {
    try {
      await deepEvaluationApi.cancelTask(taskId)
      message.success(t('deep.taskCancelled'))
      fetchTasks()
    } catch {
      message.error(t('deep.cancelTaskFailed'))
    }
  }

  const handleResumeTask = async (taskId: string) => {
    try {
      await deepEvaluationApi.resumeTask(taskId)
      message.success(t('deep.taskResumed'))
      fetchTasks()
    } catch {
      message.error(t('deep.resumeTaskFailed'))
    }
  }

  const handleDeleteTask = async (taskId: string) => {
    try {
      await deepEvaluationApi.deleteTask(taskId)
      message.success(t('deep.taskDeleted'))
      fetchTasks()
    } catch {
      message.error(t('deep.deleteTaskFailed'))
    }
  }

  const renderSummary = (summary?: ResultSummary) => {
    if (!summary) return '-'
    const overall = summary.overall?.mean
    const metricsEntries = Object.entries(summary.metrics || {})
    return (
      <div style={{ fontSize: 12 }}>
        {overall !== undefined && <div>{t('deep.overall')}: {formatPercent(overall)}</div>}
        {metricsEntries.slice(0, 3).map(([name, stat]) => (
          <div key={name}>
            {name}: {formatPercent(stat?.mean)}
          </div>
        ))}
        {metricsEntries.length > 3 && <div>{t('deep.moreItems', { count: metricsEntries.length - 3 })}</div>}
      </div>
    )
  }

  const columns: ColumnsType<DeepEvaluationTask> = [
    {
      title: t('deep.column.taskName'),
      dataIndex: 'task_name',
      key: 'task_name',
      render: (_, record) => record.task_name || record.task_id.substring(0, 8),
    },
    {
      title: t('deep.column.modelGroup'),
      dataIndex: 'model_configs',
      key: 'model_configs',
      width: 160,
      render: (modelConfigs: DeepEvaluationTask['model_configs']) => {
        const count = modelConfigs?.length || 0
        if (!count) return '-'
        const names = modelConfigs.map((g, i) => g.group_name || t('deep.modelGroupDefault', { index: i + 1 }))
        const shown = names.slice(0, 2)
        const tipContent = modelConfigs.map((g, i) => {
          const parts: string[] = []
          if (g.embedding) parts.push(`Embedding: ${g.embedding.config_name || g.embedding.model_name || ''}`)
          if (g.rerank) parts.push(`Rerank: ${g.rerank.config_name || g.rerank.model_name || ''}`)
          if (g.llm) parts.push(`LLM: ${g.llm.config_name || g.llm.model || ''}`)
          return `${g.group_name || t('deep.modelGroupDefault', { index: i + 1 })}: ${parts.join(', ')}`
        }).join('\n')
        return (
          <Tooltip title={<pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: 12 }}>{tipContent}</pre>}>
            <div>
              <Tag color="blue">{t('deep.modelGroupCount', { count })}</Tag>
              <div style={{ fontSize: 12, color: 'var(--tf-text-secondary)', marginTop: 4 }}>
                {shown.join(', ')}{names.length > 2 ? ` +${names.length - 2}` : ''}
              </div>
            </div>
          </Tooltip>
        )
      },
    },
    {
      title: t('deep.column.dataset'),
      dataIndex: 'dataset_configs',
      key: 'dataset_configs',
      render: (datasetConfigs: DeepEvaluationTask['dataset_configs']) => {
        const names = (datasetConfigs || []).map((config) => (
          config.dataset_name || datasetMap[config.dataset_id]?.dataset_name || config.dataset_id?.substring(0, 8)
        ))
        return renderTags(names.filter(Boolean) as string[])
      },
    },
    {
      title: t('deep.column.status'),
      key: 'status',
      width: 120,
      render: (_, record) => {
        const hasIR = record.metrics?.some((m) => TRADITIONAL_METRICS.find((tm) => tm.value === m))
        const hasLLM = record.metrics?.some((m) => !TRADITIONAL_METRICS.find((tm) => tm.value === m))
        return (
          <Space direction="vertical" size={4}>
            <Tag color={statusColorMap[record.status] || 'default'}>
              {statusLabelMap[record.status] || record.status}
            </Tag>
            <Space size={2}>
              {hasIR && <Tag style={{ fontSize: 10, padding: '0 4px', lineHeight: '16px' }} color="blue">IR</Tag>}
              {hasLLM && <Tag style={{ fontSize: 10, padding: '0 4px', lineHeight: '16px' }} color="purple">LLM</Tag>}
            </Space>
          </Space>
        )
      },
    },
    {
      title: t('deep.column.progress'),
      key: 'progress',
      width: 160,
      render: (_, record) => {
        const mp = record.model_progress
        const percent = Math.round(record.progress || 0)
        let completedCount = 0
        let totalCount = 0
        if (mp) {
          for (const datasets of Object.values(mp)) {
            for (const ds of Object.values(datasets as Record<string, { status?: string }>)) {
              totalCount++
              if (ds.status === 'completed') completedCount++
            }
          }
        }
        return (
          <div>
            <Progress
              percent={percent}
              size="small"
              status={record.status === 'failed' ? 'exception' : undefined}
            />
            {totalCount > 0 && (
              <Text style={{ fontSize: 11, color: 'var(--tf-text-secondary)' }}>
                {t('deep.groupsCompleted', { completed: completedCount, total: totalCount })}
              </Text>
            )}
          </div>
        )
      },
    },
    {
      title: t('deep.column.results'),
      dataIndex: 'results_summary',
      key: 'results_summary',
      render: (summary: ResultSummary) => renderSummary(summary),
    },
    {
      title: t('deep.column.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => formatDate(time),
    },
    {
      title: t('deep.column.actions'),
      key: 'actions',
      width: 160,
      render: (_, record) => (
        <Space>
          {(record.results_summary || record.model_progress) && (
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedTask(record)
                setDetailVisible(true)
              }}
            />
          )}
          {(record.status === 'running' || record.status === 'pending') && (
            <Popconfirm
              title={t('deep.confirm.cancelTask')}
              onConfirm={() => handleCancelTask(record.task_id)}
            >
              <Button type="link" size="small" icon={<StopOutlined />} danger />
            </Popconfirm>
          )}
          {(record.status === 'failed' || record.status === 'cancelled') && (
            <Button
              type="link"
              size="small"
              icon={<RedoOutlined />}
              onClick={() => handleResumeTask(record.task_id)}
            />
          )}
          {record.status !== 'running' && record.status !== 'pending' && (
            <Popconfirm
              title={t('deep.confirm.deleteTask')}
              onConfirm={() => handleDeleteTask(record.task_id)}
            >
              <Button type="link" size="small" danger icon={<DeleteOutlined />} />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  return (
    <div>
      {tasksStale && (
        <Alert
          data-testid="deep-evaluation-stale-state"
          type="warning"
          showIcon
          message={t('deep.pollingStale')}
          style={{ marginBottom: 16 }}
        />
      )}
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={6}>
          <StatCard
            title={t('stats.totalTasks')}
            value={stats.total}
            icon={<CheckCircleOutlined />}
            color={STATUS_INFO}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('stats.running')}
            value={stats.running}
            icon={<PlayCircleOutlined />}
            color={STATUS_WARNING}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('stats.completed')}
            value={stats.completed}
            icon={<CheckCircleOutlined />}
            color={STATUS_SUCCESS}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('stats.failed')}
            value={stats.failed}
            icon={<CloseCircleOutlined />}
            color={STATUS_ERROR}
          />
        </Col>
      </Row>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>{t('deep.title')}</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void fetchTasks()} loading={loading}>{t('common:action.refresh')}</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateVisible(true)}>
            {t('deep.createTask')}
          </Button>
        </Space>
      </div>

      <Table
        rowKey="task_id"
        columns={columns}
        dataSource={tasks}
        loading={loading}
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showTotal: (total) => t('common:pagination.total', { total }),
          total: tasksTotal,
        }}
      />

      <Modal
        title={t('deep.createTitle')}
        open={createVisible}
        onCancel={() => {
          setCreateVisible(false)
          form.resetFields()
          setModelGroups([createGroup(0)])
        }}
        onOk={handleCreateTask}
        okText={t('deep.createButton')}
        destroyOnClose
        width={860}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            metrics: DEFAULT_METRICS,
            model_workers: 2,
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="task_name" label={t('deep.taskName')}>
                <Input placeholder={t('deep.taskNamePlaceholder')} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="dataset_ids" label={t('deep.dataset')} rules={[{ required: true, message: t('deep.datasetRequired') }]}>
                <Select
                  showSearch
                  placeholder={t('deep.selectRegisteredDataset')}
                  options={datasetOptions}
                  onChange={handleDatasetChange}
                  optionFilterProp="label"
                  mode="multiple"
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="max_samples" label={t('deep.sampleCount')}>
                <InputNumber min={1} placeholder={t('deep.sampleCountPlaceholder')} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={16}>
              <Form.Item name="retrieval_mode" label={t('deep.retrievalMode')} initialValue="offline">
                <Radio.Group>
                  <Radio.Button value="offline">{t('deep.retrievalModeOffline')}</Radio.Button>
                  <Radio.Button value="online">{t('deep.retrievalModeOnline')}</Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Col>
          </Row>

          {retrievalMode === 'online' && (
            <Card size="small" title={t('deep.onlineRetrievalConfig')} style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="milvus_collection"
                    label={t('deep.milvusCollection')}
                    rules={[{ required: true, message: t('deep.milvusCollectionRequired') }]}
                  >
                    <Select
                      showSearch
                      placeholder={t('deep.selectMilvusCollection')}
                      optionFilterProp="label"
                      options={evalReadyDatasets.map((d) => ({
                        label: `${d.display_name || d.collection_name} (${d.embedding_model || t('deep.unknownModel')})`,
                        value: d.collection_name,
                      }))}
                      onChange={(value) => {
                        const matched = evalReadyDatasets.find((d) => d.collection_name === value)
                        if (matched) {
                          // 自动选择第一个关联数据集
                          const firstDs = matched.linked_datasets?.[0]
                          if (firstDs) {
                            form.setFieldsValue({ dataset_ids: [firstDs.dataset_id] })
                          }
                          if (!form.getFieldValue('retrieval_embedding_config_id') && matched.embedding_config_id) {
                            form.setFieldsValue({ retrieval_embedding_config_id: matched.embedding_config_id })
                          }
                          form.setFieldsValue({ allow_collection_model_mismatch: false })
                        }
                      }}
                      notFoundContent={<Text type="secondary">{t('deep.noRegisteredCollections')}</Text>}
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="retrieval_embedding_config_id" label={t('deep.retrievalEmbeddingModel')} rules={[{ required: true, message: t('deep.selectModelRequired') }]}>
                    <ModelConfigSelector modelType="embedding" mode="single" />
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item name="retrieval_top_k" label="Top-K" initialValue={20}>
                    <InputNumber min={1} max={100} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
              {selectedEvalCollection && (
                <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
                  {t('deep.collectionIndexedBy', {
                    model: selectedEvalCollection.embedding_model || t('deep.unknownModel'),
                  })}
                </Text>
              )}
              {collectionModelMismatch && (
                <>
                  <Alert
                    type="warning"
                    showIcon
                    style={{ marginBottom: 8 }}
                    message={t('deep.modelMismatchTitle')}
                    description={t('deep.modelMismatchDisclaimer')}
                  />
                  <Form.Item
                    name="allow_collection_model_mismatch"
                    initialValue={false}
                    style={{ marginBottom: 0 }}
                  >
                    <Radio.Group>
                      <Radio value={true}>{t('deep.allowModelMismatch')}</Radio>
                      <Radio value={false}>{t('deep.requireModelMatch')}</Radio>
                    </Radio.Group>
                  </Form.Item>
                </>
              )}
            </Card>
          )}


          <Form.Item name="description" label={t('deep.taskDescription')}>
            <Input.TextArea rows={2} placeholder={t('deep.descriptionPlaceholder')} />
          </Form.Item>

          {hasRetrievalMetrics && (
            <Card size="small" title={t('deep.retrievalFieldMapping')} style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item
                    name={['field_mapping', 'query']}
                    label={t('deep.queryField')}
                  >
                    {datasetColumns.length ? (
                      <Select allowClear placeholder={t('deep.defaultQuery')} options={datasetColumns.map((c) => ({ label: c, value: c }))} />
                    ) : (
                      <Input placeholder={t('deep.defaultQuery')} />
                    )}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name={['field_mapping', 'positives']} label={t('deep.positivesField')}>
                    {datasetColumns.length ? (
                      <Select allowClear placeholder={t('deep.defaultPositives')} options={datasetColumns.map((c) => ({ label: c, value: c }))} />
                    ) : (
                      <Input placeholder={t('deep.defaultPositives')} />
                    )}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name={['field_mapping', 'negatives']} label={t('deep.negativesField')}>
                    {datasetColumns.length ? (
                      <Select allowClear placeholder={t('deep.defaultNegatives')} options={datasetColumns.map((c) => ({ label: c, value: c }))} />
                    ) : (
                      <Input placeholder={t('deep.defaultNegatives')} />
                    )}
                  </Form.Item>
                </Col>
              </Row>
            </Card>
          )}

          {hasLLMMetrics && (
            <Card size="small" title={t('deep.llmFieldMapping')} style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name={['field_mapping', 'input']}
                    label={t('deep.inputField')}
                    rules={[{ required: true, message: t('deep.inputFieldRequired') }]}
                  >
                    {datasetColumns.length ? (
                      <Select options={datasetColumns.map((c) => ({ label: c, value: c }))} />
                    ) : (
                      <Input placeholder={t('deep.inputFieldPlaceholder')} />
                    )}
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name={['field_mapping', 'actual_output']}
                    label={t('deep.actualOutputField')}
                    rules={[{ required: llmFieldRequirements.actual_output, message: t('deep.actualOutputFieldRequired') }]}
                  >
                    {datasetColumns.length ? (
                      <Select allowClear options={datasetColumns.map((c) => ({ label: c, value: c }))} />
                    ) : (
                      <Input placeholder={t('deep.actualOutputFieldPlaceholder')} />
                    )}
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name={['field_mapping', 'expected_output']}
                    label={t('deep.expectedOutputField')}
                    rules={[{ required: llmFieldRequirements.expected_output, message: t('deep.expectedOutputFieldRequired') }]}
                  >
                    {datasetColumns.length ? (
                      <Select allowClear options={datasetColumns.map((c) => ({ label: c, value: c }))} />
                    ) : (
                      <Input placeholder={t('deep.expectedOutputFieldPlaceholder')} />
                    )}
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name={['field_mapping', 'retrieval_context']}
                    label={t('deep.retrievalContextField')}
                    rules={[{ required: llmFieldRequirements.retrieval_context, message: t('deep.retrievalContextFieldRequired') }]}
                  >
                    {datasetColumns.length ? (
                      <Select allowClear options={datasetColumns.map((c) => ({ label: c, value: c }))} />
                    ) : (
                      <Input placeholder={t('deep.retrievalContextFieldPlaceholder')} />
                    )}
                  </Form.Item>
                </Col>
              </Row>
            </Card>
          )}

          <Card size="small" title={t('deep.evalMetrics')} style={{ marginBottom: 16 }}>
            <Form.Item name="metrics" rules={[{ required: true, message: t('deep.metricsRequired') }]}>
              <Select mode="multiple" placeholder={t('deep.selectMetrics')}>
                <Select.OptGroup label={t('deep.traditionalIRMetrics')}>
                  {TRADITIONAL_METRICS.map((metric) => (
                    <Select.Option key={metric.value} value={metric.value}>
                      <Tag color="blue" style={{ marginRight: 4 }}>IR</Tag>
                      {metric.label} - {metric.description}
                    </Select.Option>
                  ))}
                </Select.OptGroup>
                <Select.OptGroup label={t('deep.deepEvalMetrics')}>
                  {llmMetrics.map((metric) => (
                    <Select.Option key={metric.name} value={metric.name}>
                      <Tag color="purple" style={{ marginRight: 4 }}>LLM</Tag>
                      {metric.name} - {metric.description}
                    </Select.Option>
                  ))}
                </Select.OptGroup>
              </Select>
            </Form.Item>
          </Card>

          <Tooltip title={t('deep.chunkEvalTooltip')}>
            <Form.Item name="chunk_eval_mode" label={t('deep.chunkEvalMode')} initialValue="batch" style={{ marginBottom: 16 }}>
              <Radio.Group size="small">
                <Radio.Button value="batch">Batch</Radio.Button>
                <Radio.Button value="individual">Individual</Radio.Button>
              </Radio.Group>
            </Form.Item>
          </Tooltip>

          <Card
            size="small"
            title={
              <Space>
                <span>{t('deep.modelGroupConfig')}</span>
                <Text type="secondary" style={{ fontSize: 12, fontWeight: 'normal' }}>
                  ({t('deep.groupConcurrency')}
                  <Form.Item name="model_workers" noStyle initialValue={2}>
                    <InputNumber min={1} max={16} size="small" style={{ width: 60, marginLeft: 4, marginRight: 4 }} />
                  </Form.Item>
                  )
                </Text>
              </Space>
            }
            extra={
              <Button
                size="small"
                icon={<PlusOutlined />}
                onClick={() => setModelGroups((prev) => [
                  ...prev,
                  createGroup(prev.length),
                ])}
              >
                {t('deep.addGroup')}
              </Button>
            }
          >
            {modelGroups.map((group, index) => (
              <Card
                key={group.group_id}
                size="small"
                style={{ marginBottom: index < modelGroups.length - 1 ? 12 : 0 }}
                title={
                  <Input
                    value={group.group_name}
                    size="small"
                    style={{ width: 200 }}
                    onChange={(e) => {
                      const newGroups = [...modelGroups]
                      newGroups[index] = { ...newGroups[index], group_name: e.target.value }
                      setModelGroups(newGroups)
                    }}
                  />
                }
                extra={
                  modelGroups.length > 1 && (
                    <Button
                      type="link"
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={() => setModelGroups((prev) => prev.filter((g) => g.group_id !== group.group_id))}
                    />
                  )
                }
              >
                <Row gutter={16}>
                  <Col span={8}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>{t('deep.embeddingModel')}</Text>
                      <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>{t('deep.optional')}</Text>
                    </div>
                    <ModelConfigSelector
                      mode="single"
                      modelType="embedding"
                      onChange={(_configId, config) => {
                        const newGroups = [...modelGroups]
                        newGroups[index] = { ...newGroups[index], embedding: config || null }
                        setModelGroups(newGroups)
                      }}
                    />
                    {group.embedding && (
                      <div style={{ marginTop: 8 }}>
                        <Tag color="blue">{group.embedding.config_name || group.embedding.model_name}</Tag>
                        <div style={{ marginTop: 4 }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>{t('deep.concurrency')}</Text>
                          <Form.Item name={['groups', group.group_id, 'embedding', 'concurrency']} noStyle initialValue={8}>
                            <InputNumber min={1} max={64} size="small" style={{ width: 60 }} />
                          </Form.Item>
                        </div>
                      </div>
                    )}
                  </Col>
                  <Col span={8}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>{t('deep.rerankModel')}</Text>
                      <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>{t('deep.optional')}</Text>
                    </div>
                    <ModelConfigSelector
                      mode="single"
                      modelType="rerank"
                      onChange={(_configId, config) => {
                        const newGroups = [...modelGroups]
                        newGroups[index] = { ...newGroups[index], rerank: config || null }
                        setModelGroups(newGroups)
                      }}
                    />
                    {group.rerank && (
                      <div style={{ marginTop: 8 }}>
                        <Tag color="geekblue">{group.rerank.config_name || group.rerank.model_name}</Tag>
                        <div style={{ marginTop: 4 }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>{t('deep.concurrency')}</Text>
                          <Form.Item name={['groups', group.group_id, 'rerank', 'concurrency']} noStyle initialValue={4}>
                            <InputNumber min={1} max={64} size="small" style={{ width: 60 }} />
                          </Form.Item>
                        </div>
                      </div>
                    )}
                  </Col>
                  <Col span={8}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>{t('deep.llmModel')}</Text>
                      <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>{t('deep.optional')}</Text>
                    </div>
                    <ModelConfigSelector
                      mode="single"
                      modelType="llm"
                      onChange={(_configId, config) => {
                        const newGroups = [...modelGroups]
                        newGroups[index] = { ...newGroups[index], llm: config || null }
                        setModelGroups(newGroups)
                      }}
                    />
                    {group.llm && (
                      <div style={{ marginTop: 8 }}>
                        <Tag color="purple">{group.llm.config_name || group.llm.model_name}</Tag>
                        <div style={{ marginTop: 4 }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>{t('deep.concurrency')}</Text>
                          <Form.Item name={['groups', group.group_id, 'llm', 'concurrency']} noStyle initialValue={2}>
                            <InputNumber min={1} max={32} size="small" style={{ width: 60 }} />
                          </Form.Item>
                        </div>
                        {hasLLMMetrics && (
                          <div style={{ marginTop: 8 }}>
                            <Row gutter={8}>
                              <Col span={12}>
                                <Form.Item name={['groups', group.group_id, 'llm', 'temperature']} label="Temperature" initialValue={0}>
                                  <InputNumber min={0} max={2} step={0.1} size="small" style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item name={['groups', group.group_id, 'llm', 'top_p']} label="Top-p" initialValue={1}>
                                  <InputNumber min={0} max={1} step={0.05} size="small" style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={8}>
                              <Col span={12}>
                                <Form.Item name={['groups', group.group_id, 'llm', 'top_k']} label="Top-k">
                                  <InputNumber min={0} max={200} size="small" style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  name={['groups', group.group_id, 'llm', 'max_tokens']}
                                  label="Max Tokens"
                                  initialValue={2048}
                                  rules={[
                                    { type: 'number', max: MAX_LLM_TOKENS, message: t('deep.maxTokensRule', { max: MAX_LLM_TOKENS }) },
                                  ]}
                                  extra={t('deep.maxTokensRule', { max: MAX_LLM_TOKENS })}
                                >
                                  <InputNumber min={256} max={MAX_LLM_TOKENS} size="small" style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={8}>
                              <Col span={12}>
                                <Form.Item name={['groups', group.group_id, 'llm', 'timeout']} label="Timeout(s)" initialValue={60}>
                                  <InputNumber min={10} max={600} size="small" style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item name={['groups', group.group_id, 'llm', 'max_retries']} label="Max Retries" initialValue={3}>
                                  <InputNumber min={0} max={10} size="small" style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                            </Row>
                          </div>
                        )}
                      </div>
                    )}
                  </Col>
                </Row>
              </Card>
            ))}
          </Card>
        </Form>
      </Modal>

      <Modal
        title={selectedTask?.task_name || t('deep.detailTitle')}
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        footer={null}
        width={800}
      >
        {selectedTask?.model_progress && Object.keys(selectedTask.model_progress).length > 0 && (
          <Card size="small" title={t('deep.modelProgress')} style={{ marginBottom: 16 }}>
            {Object.entries(selectedTask.model_progress).map(([modelName, datasets]) => (
              <div key={modelName} style={{ marginBottom: 12 }}>
                <Text strong>{modelName}</Text>
                <div style={{ marginLeft: 16, marginTop: 4 }}>
                  {Object.entries(datasets).map(([datasetName, info]) => (
                    <div key={datasetName} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
                      <Text style={{ width: 150, fontSize: 12 }} ellipsis={{ tooltip: datasetName }}>
                        {datasetName}
                      </Text>
                      <Progress
                        percent={Math.round(info.progress || 0)}
                        size="small"
                        style={{ flex: 1, marginLeft: 8, marginRight: 8 }}
                        status={info.status === 'failed' ? 'exception' : info.status === 'completed' ? 'success' : 'active'}
                      />
                      <Tag
                        color={statusColorMap[info.status] || 'default'}
                        style={{ fontSize: 10, minWidth: 50, textAlign: 'center' }}
                      >
                        {statusLabelMap[info.status] || info.status}
                      </Tag>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </Card>
        )}
        {selectedTask?.results_summary ? (() => {
          type MetricValue = { mean?: number }
          type DatasetSummaryItem = {
            dataset_name?: string
            summary?: { metrics?: Record<string, MetricValue> }
          }
          type RetrievalModelData = {
            summary?: { datasets?: Record<string, DatasetSummaryItem> }
            model?: { config_name?: string; model_name?: string }
          }
          type LlmModelData = {
            datasets?: Record<string, DatasetSummaryItem>
            model?: { config_name?: string; model_name?: string; model?: string }
          }
          type GroupSummary = {
            retrieval?: Partial<Record<'embedding' | 'rerank', RetrievalModelData>>
            llm?: LlmModelData
          }
          const summary = selectedTask.results_summary as ResultSummary & { by_group?: Record<string, GroupSummary> }
          const byGroup = summary.by_group || {}
          // Build table rows: group_name, model_type, dataset_name, metric values
          const metricNames = Object.keys(summary.metrics || {})
          interface ResultRow {
            key: string
            group: string
            modelType: string
            modelName: string
            dataset: string
            [metric: string]: string | number | undefined
          }
          const rows: ResultRow[] = []
          for (const [groupName, groupData] of Object.entries(byGroup)) {
            const retrieval = groupData?.retrieval || {}
            for (const modelType of ['embedding', 'rerank'] as const) {
              const modelData = retrieval[modelType]
              if (!modelData?.summary?.datasets) continue
              const mName = modelData.model?.config_name || modelData.model?.model_name || ''
              for (const [, dsData] of Object.entries(modelData.summary.datasets as Record<string, DatasetSummaryItem>)) {
                const dsMetrics = dsData?.summary?.metrics || {}
                const row: ResultRow = {
                  key: `${groupName}-${modelType}-${dsData?.dataset_name || ''}`,
                  group: groupName,
                  modelType: modelType === 'embedding' ? 'Embedding' : 'Rerank',
                  modelName: mName,
                  dataset: dsData?.dataset_name || '',
                }
                for (const mn of metricNames) {
                  row[mn] = dsMetrics[mn]?.mean
                }
                rows.push(row)
              }
            }
            // LLM metrics
            const llmData = groupData?.llm || {}
            if (llmData.datasets) {
              const llmName = llmData.model?.config_name || llmData.model?.model_name || llmData.model?.model || ''
              for (const [, dsData] of Object.entries(llmData.datasets as Record<string, DatasetSummaryItem>)) {
                const dsMetrics = dsData?.summary?.metrics || {}
                const row: ResultRow = {
                  key: `${groupName}-llm-${dsData?.dataset_name || ''}`,
                  group: groupName,
                  modelType: 'LLM',
                  modelName: llmName,
                  dataset: dsData?.dataset_name || '',
                }
                for (const mn of metricNames) {
                  row[mn] = dsMetrics[mn]?.mean
                }
                rows.push(row)
              }
            }
          }
          const resultColumns: ColumnsType<ResultRow> = [
            { title: t('deep.column.modelGroup'), dataIndex: 'group', key: 'group', width: 140,
              onCell: (_, index) => {
                if (index === undefined || index === 0 || rows[index].group !== rows[index - 1].group) {
                  let span = 1
                  for (let i = (index || 0) + 1; i < rows.length && rows[i].group === rows[index || 0].group; i++) span++
                  return { rowSpan: span }
                }
                return { rowSpan: 0 }
              },
            },
            { title: t('deep.column.type'), dataIndex: 'modelType', key: 'modelType', width: 90,
              render: (v: string) => <Tag color={v === 'Embedding' ? 'blue' : v === 'Rerank' ? 'geekblue' : 'purple'}>{v}</Tag>,
            },
            { title: t('deep.column.model'), dataIndex: 'modelName', key: 'modelName', width: 200, ellipsis: true,
              render: (v: string) => v ? <Tooltip title={v}><span>{v}</span></Tooltip> : '-',
            },
            { title: t('deep.column.dataset'), dataIndex: 'dataset', key: 'dataset', width: 180, ellipsis: true },
            ...metricNames.map((mn) => ({
              title: mn.toUpperCase(),
              dataIndex: mn,
              key: mn,
              width: 100,
              render: (v: unknown) => typeof v === 'number' ? (v * 100).toFixed(2) + '%' : '-',
              sorter: (a: ResultRow, b: ResultRow) => ((a[mn] as number) || 0) - ((b[mn] as number) || 0),
            })),
          ]
          return (
            <Card size="small" title={t('deep.evalResults')} style={{ marginTop: 16 }}>
              {summary.overall && (
                <div style={{ marginBottom: 12 }}>
                  <Text strong>{t('deep.overallScore')}</Text>
                  <Text>{formatPercent(summary.overall.mean)}</Text>
                  {metricNames.length > 0 && (
                    <span style={{ marginLeft: 16 }}>
                      {metricNames.map((mn) => (
                        <Tag key={mn} color="blue" style={{ marginRight: 4 }}>
                          {mn}: {formatPercent(summary.metrics?.[mn]?.mean)}
                        </Tag>
                      ))}
                    </span>
                  )}
                </div>
              )}
              {rows.length > 0 ? (
                <Table
                  columns={resultColumns}
                  dataSource={rows}
                  size="small"
                  pagination={false}
                  bordered
                  scroll={{ x: 'max-content' }}
                />
              ) : (
                <pre style={{
                  maxHeight: 300,
                  overflow: 'auto',
                  background: 'var(--tf-bg-elevated)',
                  color: 'var(--tf-text-primary)',
                  border: '1px solid var(--tf-border-primary)',
                  padding: 16,
                  borderRadius: 6,
                  fontSize: 12,
                  lineHeight: 1.6,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                  margin: 0,
                }}>
                  {JSON.stringify(selectedTask.results_summary, null, 2)}
                </pre>
              )}
            </Card>
          )
        })() : (
          <Text type="secondary">{t('deep.noResultsYet')}</Text>
        )}
        {selectedTask?.results_path && (
          <div style={{ marginTop: 12 }}>
            <Text type="secondary">{t('deep.resultsFile', { path: selectedTask.results_path })}</Text>
          </div>
        )}
        {selectedTask?.error_message && (
          <div style={{ marginTop: 12 }}>
            <Text type="danger">{t('deep.errorMessage', { message: selectedTask.error_message })}</Text>
          </div>
        )}
      </Modal>
    </div>
  )
}
