import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import {
  Card,
  Button,
  Space,
  Tag,
  Typography,
  Descriptions,
  Progress,
  Table,
  Popconfirm,
  message,
  Spin,
  Row,
  Col,
  Tooltip,
  Tabs,
  Empty,
} from 'antd'
import {
  ArrowLeftOutlined,
  ReloadOutlined,
  StopOutlined,
  PlayCircleOutlined,
  DeleteOutlined,
  CloseCircleOutlined,
  FileTextOutlined,
  SettingOutlined,
  DatabaseOutlined,
  LinkOutlined,
  RedoOutlined,
} from '@ant-design/icons'
import { datasetApi, generationApi } from '@/services/api'
import { formatDate } from '@/utils'
import { usePolling } from '@/hooks/usePolling'
import { STATUS_SUCCESS } from '@/theme'
import { useTranslation } from 'react-i18next'
import { StatusTag } from '@/components/StatusTag'
import type { GenerationTaskStatus } from '@/services/api'

const { Title, Text } = Typography
const ACTIVE_GENERATION_STATUSES = new Set<GenerationTaskStatus>([
  'pending',
  'running',
  'stopping',
  'recovering',
  'publishing',
  'restarting',
])

type TaskDetail = Awaited<ReturnType<typeof generationApi.getTask>>
type TaskArtifacts = Awaited<ReturnType<typeof generationApi.getTaskArtifacts>>
type ArtifactItem = TaskArtifacts['stages'][number]['artifacts'][number]
type EmbeddingConfigLike = { similarity_threshold?: number; retrieval_top_k?: number }
type RerankConfigLike = { rerank_threshold?: number }

export default function GenerationDetail() {
  const { t } = useTranslation(['generation', 'common'])
  const { taskId } = useParams<{ taskId: string }>()
  const navigate = useNavigate()
  const [task, setTask] = useState<TaskDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [previewSource, setPreviewSource] = useState<string>('output')
  const [previewData, setPreviewData] = useState<{ total: number; samples: Record<string, unknown>[] }>({ total: 0, samples: [] })
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewPage, setPreviewPage] = useState(1)
  const previewPageSize = 10
  const [artifacts, setArtifacts] = useState<TaskArtifacts | null>(null)
  const [artifactsLoading, setArtifactsLoading] = useState(false)
  const [activeArtifactStage, setActiveArtifactStage] = useState<string>('')
  const [artifactPreviewDatasetId, setArtifactPreviewDatasetId] = useState<string>('')
  const [artifactPreviewRows, setArtifactPreviewRows] = useState<Record<string, unknown>[]>([])
  const [artifactPreviewLoading, setArtifactPreviewLoading] = useState(false)

  const generationModeLabels: Record<string, string> = {
    doc_to_training: t('modeLabel.doc_to_training'),
    qa_to_training: t('modeLabel.qa_to_training'),
    qa_extraction: t('modeLabel.qa_extraction'),
    doc_to_eval: t('modeLabel.doc_to_eval'),
    qa_to_eval: t('modeLabel.qa_to_eval'),
  }

  const posNegMethodLabels: Record<string, string> = {
    retrieval: t('posNegMethod.retrieval'),
    llm: t('posNegMethod.llm'),
  }

  const fetchTask = useCallback(async () => {
    if (!taskId) return
    try {
      const data = await generationApi.getTask(taskId)
      setTask(data)
    } catch (err) {
      message.error(t('detail.message.fetchFailed'))
    } finally {
      setLoading(false)
    }
  }, [taskId, t])

  const fetchArtifacts = useCallback(async () => {
    if (!taskId) return
    setArtifactsLoading(true)
    try {
      const data = await generationApi.getTaskArtifacts(taskId, { include_empty: true })
      setArtifacts(data)
    } catch {
      message.error(t('detail.message.fetchArtifactsFailed'))
    } finally {
      setArtifactsLoading(false)
    }
  }, [taskId, t])

  const hasPreviewableData = !!(task?.output_path || task?.qa_output_path || task?.qa_filtered_path || (task as Record<string, unknown>)?.deep_eval_path)

  const fetchPreview = useCallback(async () => {
    if (!taskId || !hasPreviewableData) return
    setPreviewLoading(true)
    try {
      const data = await generationApi.previewOutput(taskId, {
        limit: previewPageSize,
        offset: (previewPage - 1) * previewPageSize,
        source: previewSource,
      })
      setPreviewData(data)
    } catch (err) {
      console.error('Failed to fetch preview:', err)
    } finally {
      setPreviewLoading(false)
    }
  }, [taskId, hasPreviewableData, previewPage, previewSource])

  useEffect(() => {
    fetchTask()
  }, [fetchTask])

  useEffect(() => {
    fetchArtifacts()
  }, [fetchArtifacts])

  useEffect(() => {
    const stageList = artifacts?.stages || []
    if (stageList.length === 0) {
      setActiveArtifactStage('')
      return
    }
    if (!stageList.some((stage) => stage.stage_key === activeArtifactStage)) {
      setActiveArtifactStage(stageList[0].stage_key)
    }
  }, [artifacts, activeArtifactStage])

  useEffect(() => {
    const currentStage = artifacts?.stages?.find((stage) => stage.stage_key === activeArtifactStage)
    if (!currentStage) {
      setArtifactPreviewDatasetId('')
      setArtifactPreviewRows([])
      return
    }
    if (!currentStage.artifacts.some((item) => item.dataset_id === artifactPreviewDatasetId)) {
      setArtifactPreviewDatasetId('')
      setArtifactPreviewRows([])
    }
  }, [artifacts, activeArtifactStage, artifactPreviewDatasetId])

  useEffect(() => {
    if (hasPreviewableData) {
      fetchPreview()
    }
  }, [hasPreviewableData, fetchPreview])

  // 轮询运行中的任务
  const shouldPoll = task ? ACTIVE_GENERATION_STATUSES.has(task.status) : false
  usePolling(fetchTask, { interval: 3000, enabled: shouldPoll })

  const handleStop = async () => {
    if (!taskId) return
    try {
      await generationApi.stopTask(taskId)
      message.success(t('detail.message.stopSuccess'))
      fetchTask()
    } catch (err) {
      message.error(t('detail.message.stopFailed'))
    }
  }

  const handleResume = async () => {
    if (!taskId) return
    try {
      await generationApi.restartTask(taskId)
      message.success(t('detail.message.resumeSuccess'))
      fetchTask()
    } catch (err) {
      message.error(t('detail.message.resumeFailed'))
    }
  }

  const handleRerun = async () => {
    if (!taskId) return
    try {
      await generationApi.restartTask(taskId, true)
      message.success(t('detail.message.restartSuccess'))
      fetchTask()
    } catch (err) {
      message.error(t('detail.message.restartFailed'))
    }
  }

  const handleDelete = async () => {
    if (!taskId) return
    try {
      await generationApi.deleteTask(taskId, task?.status === 'deleting_cascade')
      message.success(t('detail.message.deleteSuccess'))
      navigate('/datasets?tab=generation')
    } catch (err) {
      message.error(t('detail.message.deleteFailed'))
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
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Empty description={t('detail.taskNotFound')} />
        <Button type="primary" onClick={() => navigate('/datasets?tab=generation')} style={{ marginTop: 16 }}>
          {t('detail.backToList')}
        </Button>
      </div>
    )
  }

  const canStop = task.status === 'running' || task.status === 'pending'
  const canResume = task.status === 'failed' || task.status === 'stopped'
  const canRerun = task.status === 'failed' || task.status === 'stopped' || task.status === 'completed'
  const canDelete = !ACTIVE_GENERATION_STATUSES.has(task.status)
  const embeddingConfig = (task.embedding_config ?? undefined) as EmbeddingConfigLike | undefined
  const rerankConfig = (task.rerank_config ?? undefined) as RerankConfigLike | undefined

  // 渲染配置项（隐藏 api_key）
  const renderConfig = (config: Record<string, unknown> | undefined, title: string) => {
    if (!config) return null
    const filtered = { ...config }
    if ('api_key' in filtered) {
      filtered.api_key = filtered.api_key ? '***' : undefined
    }
    return (
      <Card size="small" title={title} style={{ marginBottom: 16 }}>
        <pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          {JSON.stringify(filtered, null, 2)}
        </pre>
      </Card>
    )
  }

  // 渲染步骤配置
  const renderStepsConfig = () => {
    const steps = task.steps_config || {}
    const stepLabels: Record<string, string> = {
      doc_quality: t('detail.stepLabels.doc_quality'),
      keypoint_gen: t('detail.stepLabels.keypoint_gen'),
      role_gen: t('detail.stepLabels.role_gen'),
      qa_gen: t('detail.stepLabels.qa_gen'),
      pos_neg_extraction: t('detail.stepLabels.pos_neg_extraction'),
      validation: t('detail.stepLabels.validation'),
      embedding_scoring: t('detail.stepLabels.embedding_scoring'),
      rerank_scoring: t('detail.stepLabels.rerank_scoring'),
    }

    // 每个步骤需要展示的关键参数
    const stepParamLabels: Record<string, Record<string, string>> = {
      qa_gen: { num_qa_per_doc: t('detail.stepParams.num_qa_per_doc') },
      pos_neg_extraction: { num_positive: t('detail.stepParams.num_positive'), num_negative: t('detail.stepParams.num_negative'), use_role: t('detail.stepParams.use_role'), roles_per_doc: t('detail.stepParams.roles_per_doc'), neg_detection_mode: t('detail.stepParams.neg_detection_mode'), chunk_eval_mode: t('detail.stepParams.chunk_eval_mode'), supplement_positives: t('detail.stepParams.supplement_positives'), confirm_positives: t('detail.stepParams.confirm_positives'), confirm_negatives: t('detail.stepParams.confirm_negatives'), answer_rewrite: t('detail.stepParams.answer_rewrite'), rerank_score_classification: t('detail.stepParams.rerank_score_classification'), evidence_removal: t('detail.stepParams.evidence_removal'), evidence_pruning: t('detail.stepParams.evidence_pruning'), neg_chunk_scoring: t('detail.stepParams.neg_chunk_scoring'), skip_easy_negatives: t('detail.stepParams.skip_easy_negatives'), skip_perfect_ap: t('detail.stepParams.skip_perfect_ap'), skip_zero_ap: t('detail.stepParams.skip_zero_ap'), augment: t('detail.stepParams.augment') },
      role_gen: { roles_per_doc: t('detail.stepParams.roles_per_doc') },
    }

    const enabledSteps = Object.entries(steps).filter(([, cfg]) => (cfg as Record<string, unknown>)?.enabled)

    if (enabledSteps.length === 0) {
      return <Text type="secondary">{t('detail.config.noEnabledSteps')}</Text>
    }

    return (
      <Space direction="vertical" size={4} style={{ width: '100%' }}>
        {enabledSteps.map(([key, cfg]) => {
          const params = stepParamLabels[key]
          const cfgObj = cfg as Record<string, unknown>
          const paramText = params
            ? Object.entries(params)
                .filter(([k]) => cfgObj[k] != null)
                .map(([k, label]) => `${label}: ${cfgObj[k]}`)
                .join('，')
            : ''
          return (
            <div key={key}>
              <Tag color="blue">{stepLabels[key] || key}</Tag>
              {paramText && <Text type="secondary" style={{ fontSize: 12 }}>{paramText}</Text>}
            </div>
          )
        })}
      </Space>
    )
  }

  // Column: row index
  const colIndex = {
    title: '#',
    key: 'index',
    width: 50,
    render: (_: unknown, __: unknown, index: number) => (previewPage - 1) * previewPageSize + index + 1,
  }
  // Column: query
  const colQuery = {
    title: t('detail.preview.colQuery'),
    dataIndex: 'query',
    key: 'query',
    ellipsis: true,
    render: (val: unknown) => (
      <Tooltip title={String(val || '')}>
        <Text style={{ maxWidth: 300 }} ellipsis>{String(val || '-')}</Text>
      </Tooltip>
    ),
  }
  // Column: answer
  const colAnswer = {
    title: t('detail.preview.colAnswer'),
    dataIndex: 'answer',
    key: 'answer',
    ellipsis: true,
    render: (val: unknown) => (
      <Tooltip title={String(val || '')}>
        <Text style={{ maxWidth: 300 }} ellipsis>{String(val || '-')}</Text>
      </Tooltip>
    ),
  }
  // Column: detail JSON
  const colDetail = {
    title: t('detail.preview.detail'),
    key: 'detail',
    width: 70,
    render: (_: unknown, record: Record<string, unknown>) => (
      <Tooltip
        title={<pre style={{ margin: 0, maxHeight: 300, overflow: 'auto', fontSize: 11 }}>{JSON.stringify(record, null, 2)}</pre>}
        overlayStyle={{ maxWidth: 500 }}
      >
        <Button type="link" size="small">{t('common:action.view')}</Button>
      </Tooltip>
    ),
  }

  // Columns by source type
  const getPreviewColumns = (src: string) => {
    if (src === 'qa_full' || src === 'qa_filtered') {
      return [
        colIndex,
        colQuery,
        colAnswer,
        {
          title: t('detail.preview.colChunkId'),
          dataIndex: 'chunk_id',
          key: 'chunk_id',
          width: 120,
          ellipsis: true,
        },
        colDetail,
      ]
    }
    if (src === 'deep_eval') {
      return [
        colIndex,
        colQuery,
        colAnswer,
        {
          title: t('detail.preview.colPositives'),
          key: 'positives',
          width: 80,
          render: (_: unknown, record: Record<string, unknown>) => {
            const p = record.positives || record.positive
            return Array.isArray(p) ? p.length : (p ? 1 : 0)
          },
        },
        colDetail,
      ]
    }
    // "output" (training samples)
    return [
      colIndex,
      colQuery,
      {
        title: t('detail.preview.colPositives'),
        key: 'positives',
        width: 80,
        render: (_: unknown, record: Record<string, unknown>) => {
          const p = record.positives || record.positive
          return Array.isArray(p) ? p.length : (p ? 1 : 0)
        },
      },
      {
        title: t('detail.preview.colNegatives'),
        key: 'negatives',
        width: 80,
        render: (_: unknown, record: Record<string, unknown>) => {
          const n = record.negatives || record.negative
          return Array.isArray(n) ? n.length : (n ? 1 : 0)
        },
      },
      colDetail,
    ]
  }

  // Available preview sources based on task output paths
  const getAvailableSources = () => {
    if (!task) return []
    const sources: { key: string; label: string }[] = []
    if (task.output_path) sources.push({ key: 'output', label: t('detail.preview.tabTraining') })
    if (task.qa_filtered_path) sources.push({ key: 'qa_filtered', label: t('detail.preview.tabQaFiltered') })
    if (task.qa_output_path) sources.push({ key: 'qa_full', label: t('detail.preview.tabQaFull') })
    if ((task as Record<string, unknown>).deep_eval_path) sources.push({ key: 'deep_eval', label: t('detail.preview.tabDeepEval') })
    return sources
  }

  const getArtifactRoleColor = (role: string) => {
    if (role === 'output') return 'success'
    if (role === 'eval') return 'purple'
    if (role === 'input') return 'blue'
    if (role === 'rejects') return 'warning'
    return 'default'
  }

  const activeStageData = artifacts?.stages?.find((stage) => stage.stage_key === activeArtifactStage)

  const handlePreviewArtifactDataset = async (datasetId: string) => {
    setArtifactPreviewDatasetId(datasetId)
    setArtifactPreviewLoading(true)
    try {
      const preview = await datasetApi.preview(datasetId, 10)
      if (Array.isArray(preview)) {
        setArtifactPreviewRows(preview)
      } else {
        setArtifactPreviewRows(preview?.rows || [])
      }
    } catch {
      setArtifactPreviewRows([])
      message.error(t('detail.message.fetchArtifactPreviewFailed'))
    } finally {
      setArtifactPreviewLoading(false)
    }
  }

  const artifactColumns = [
    {
      title: t('detail.artifacts.colDataset'),
      dataIndex: 'dataset_name',
      key: 'dataset_name',
      render: (_: unknown, record: ArtifactItem) => (
        <Link to={`/datasets/${record.dataset_id}`}>{record.dataset_name}</Link>
      ),
    },
    {
      title: t('detail.artifacts.colRole'),
      dataIndex: 'artifact_role',
      key: 'artifact_role',
      width: 120,
      render: (role: string) => (
        <Tag color={getArtifactRoleColor(role)}>
          {t(`detail.artifacts.roles.${role}`, { defaultValue: role })}
        </Tag>
      ),
    },
    {
      title: t('detail.artifacts.colType'),
      key: 'dataset_type_usage',
      width: 180,
      render: (_: unknown, record: ArtifactItem) => (
        <Space size={4} wrap>
          <Tag>{record.dataset_type}</Tag>
          <Tag>{record.usage}</Tag>
        </Space>
      ),
    },
    {
      title: t('detail.artifacts.colRows'),
      dataIndex: 'num_rows',
      key: 'num_rows',
      width: 120,
      render: (value: number | undefined) => (value != null ? value.toLocaleString() : '-'),
    },
    {
      title: t('detail.artifacts.colStatus'),
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => <StatusTag status={status} />,
    },
    {
      title: t('detail.artifacts.colCreatedAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (value: string | undefined) => (value ? formatDate(value) : '-'),
    },
    {
      title: t('detail.artifacts.colActions'),
      key: 'actions',
      width: 150,
      render: (_: unknown, record: ArtifactItem) => (
        <Space size={4}>
          <Button type="link" size="small" onClick={() => handlePreviewArtifactDataset(record.dataset_id)}>
            {t('detail.artifacts.previewAction')}
          </Button>
          <Link to={`/datasets/${record.dataset_id}`}>
            <Button type="link" size="small">{t('detail.artifacts.openAction')}</Button>
          </Link>
        </Space>
      ),
    },
  ]

  const artifactPreviewColumns = (() => {
    if (!artifactPreviewRows.length) return []
    const sample = artifactPreviewRows[0] || {}
    const keys = Object.keys(sample).slice(0, 4)
    return [
      {
        title: '#',
        key: 'idx',
        width: 56,
        render: (_: unknown, __: unknown, idx: number) => idx + 1,
      },
      ...keys.map((key) => ({
        title: key,
        dataIndex: key,
        key,
        ellipsis: true,
        render: (val: unknown) => (
          <Tooltip title={String(val ?? '')}>
            <Text ellipsis style={{ maxWidth: 220 }}>{String(val ?? '-')}</Text>
          </Tooltip>
        ),
      })),
    ]
  })()

  return (
    <div style={{ padding: 24 }}>
      {/* 顶部导航 */}
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/datasets?tab=generation')}>
            {t('common:action.back')}
          </Button>
          <Title level={4} style={{ margin: 0 }}>
            {task.task_name}
          </Title>
          <StatusTag
            status={task.status}
            text={
              task.status === 'restarting'
                ? t('generationStatus.restarting')
                : undefined
            }
            description={
              task.status === 'restarting'
                ? t('generationStatusDescription.restarting')
                : undefined
            }
          />
        </Space>
        <Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => {
              fetchTask()
              fetchArtifacts()
            }}
          >
            {t('common:action.refresh')}
          </Button>
          {canStop && (
            <Popconfirm title={t('detail.confirmStop')} onConfirm={handleStop}>
              <Button danger icon={<StopOutlined />}>
                {t('common:action.stop')}
              </Button>
            </Popconfirm>
          )}
          {canResume && (
            <Popconfirm title={t('detail.confirmResume')} onConfirm={handleResume}>
              <Button icon={<PlayCircleOutlined />} style={{ color: '#52c41a', borderColor: '#52c41a' }}>
                {t('detail.resume')}
              </Button>
            </Popconfirm>
          )}
          {canRerun && (
            <Popconfirm title={t('detail.confirmRerun')} onConfirm={handleRerun}>
              <Button type="primary" icon={<RedoOutlined />}>
                {t('detail.rerun')}
              </Button>
            </Popconfirm>
          )}
          {canDelete && (
            <Popconfirm title={t('detail.confirmDelete')} onConfirm={handleDelete}>
              <Button danger icon={<DeleteOutlined />}>
                {t('common:action.delete')}
              </Button>
            </Popconfirm>
          )}
        </Space>
      </div>

      {/* 进度卡片 */}
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={24} align="middle">
          <Col span={12}>
            <div style={{ marginBottom: 8 }}>
              <Text strong>{t('detail.progress.title')}</Text>
            </div>
            <Progress
              percent={Math.round(task.progress)}
              status={task.status === 'failed' ? 'exception' : task.status === 'completed' ? 'success' : 'active'}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {task.processed_docs} / {task.total_docs}
            </Text>
          </Col>
          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 'bold', color: STATUS_SUCCESS }}>
                {task.output_sample_count.toLocaleString()}
              </div>
              <Text type="secondary">{t('detail.progress.sampleCount')}</Text>
            </div>
          </Col>
          <Col span={6}>
            {task.output_dataset_id && (
              <div style={{ textAlign: 'center' }}>
                <Link to={`/datasets/${task.output_dataset_id}`}>
                  <Button type="link" icon={<LinkOutlined />}>
                    {t('detail.viewDataset')}
                  </Button>
                </Link>
              </div>
            )}
            {task.qa_dataset_id && (
              <div style={{ textAlign: 'center' }}>
                <Link to={`/datasets/${task.qa_dataset_id}`}>
                  <Button type="link" icon={<DatabaseOutlined />} size="small">
                    {t('detail.qaFullDataset')}
                  </Button>
                </Link>
              </div>
            )}
            {task.qa_filtered_dataset_id && (
              <div style={{ textAlign: 'center' }}>
                <Link to={`/datasets/${task.qa_filtered_dataset_id}`}>
                  <Button type="link" icon={<DatabaseOutlined />} size="small">
                    {t('detail.qaFilteredDataset')}
                  </Button>
                </Link>
              </div>
            )}
          </Col>
        </Row>
        {task.error_message && (
          <div style={{ marginTop: 16, padding: 12, background: 'rgba(248, 81, 73, 0.10)', borderRadius: 4 }}>
            <Text type="danger">
              <CloseCircleOutlined style={{ marginRight: 8 }} />
              {task.error_message}
            </Text>
          </div>
        )}
        {!!task.stages?.length && (
          <div style={{ marginTop: 16 }}>
            <Text strong>{t('detail.progress.stageView')}</Text>
            <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {task.stages.map((stage) => {
                const title = stage.resume_supported
                  ? stage.resume_ready
                    ? t('detail.progress.resumeReady')
                    : t('detail.progress.resumeNotReady')
                  : t('detail.progress.resumeNotSupported')
                return (
                  <Tooltip
                    key={stage.stage}
                    title={`${title}${stage.checkpoint_path ? `\n${stage.checkpoint_path}` : ''}`}
                  >
                    <span>
                      <StatusTag
                        status={stage.status}
                        text={`${t(`stage.${stage.stage}`, { defaultValue: stage.label })}: ${t(`common:status.${stage.status}`, { defaultValue: stage.status })}`}
                        showTooltip={false}
                      />
                    </span>
                  </Tooltip>
                )
              })}
            </div>
          </div>
        )}
      </Card>

      {/* 详情标签页 */}
      <Tabs
        defaultActiveKey="info"
        items={[
          {
            key: 'info',
            label: (
              <span>
                <FileTextOutlined />
                {t('detail.tabs.info')}
              </span>
            ),
            children: (
              <Card>
                <Descriptions column={2} bordered size="small">
                  <Descriptions.Item label={t('detail.info.taskId')}>{task.task_id}</Descriptions.Item>
                  <Descriptions.Item label={t('detail.info.generationMode')}>{generationModeLabels[task.generation_mode] || task.generation_mode}</Descriptions.Item>
                  {task.pos_neg_method && ['doc_to_training', 'qa_to_training'].includes(task.generation_mode) && (
                    <Descriptions.Item label={t('detail.info.posNegMethod')}>{posNegMethodLabels[task.pos_neg_method] || task.pos_neg_method}</Descriptions.Item>
                  )}
                  <Descriptions.Item label={t('detail.info.inputPath')} span={2}>
                    <Text code copyable style={{ fontSize: 12 }}>
                      {task.input_path}
                    </Text>
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.info.inputFormat')}>{task.input_format}</Descriptions.Item>
                  <Descriptions.Item label={t('detail.info.contentField')}>{task.content_field || '-'}</Descriptions.Item>
                  <Descriptions.Item label={t('detail.info.outputFormat')}>{task.output_format}</Descriptions.Item>
                  <Descriptions.Item label={t('detail.info.autoRegister')}>{task.auto_register_dataset ? t('detail.info.autoRegisterYes') : t('detail.info.autoRegisterNo')}</Descriptions.Item>
                  {task.output_path && (
                    <Descriptions.Item label={t('detail.info.outputPath')} span={2}>
                      <Text code copyable style={{ fontSize: 12 }}>
                        {task.output_path}
                      </Text>
                    </Descriptions.Item>
                  )}
                  <Descriptions.Item label={t('detail.info.createdAt')}>{formatDate(task.created_at)}</Descriptions.Item>
                  <Descriptions.Item label={t('detail.info.startedAt')}>{task.started_at ? formatDate(task.started_at) : '-'}</Descriptions.Item>
                  <Descriptions.Item label={t('detail.info.completedAt')}>{task.completed_at ? formatDate(task.completed_at) : '-'}</Descriptions.Item>
                  {task.source_dataset_id && (
                    <Descriptions.Item label={t('detail.info.sourceDataset')}>
                      <Link to={`/datasets/${task.source_dataset_id}`}>{task.source_dataset_id}</Link>
                    </Descriptions.Item>
                  )}
                  {embeddingConfig?.similarity_threshold != null && (
                    <Descriptions.Item label={t('detail.info.similarityThreshold')}>{embeddingConfig.similarity_threshold}</Descriptions.Item>
                  )}
                  {embeddingConfig?.retrieval_top_k != null && (
                    <Descriptions.Item label={t('detail.info.retrievalTopK')}>{embeddingConfig.retrieval_top_k}</Descriptions.Item>
                  )}
                  {rerankConfig?.rerank_threshold != null && rerankConfig.rerank_threshold < 1 && (
                    <Descriptions.Item label={t('detail.info.rerankThreshold')}>{rerankConfig.rerank_threshold}</Descriptions.Item>
                  )}
                  {task.milvus_collection && (
                    <Descriptions.Item label={t('detail.info.milvusCollection')}>{task.milvus_collection}</Descriptions.Item>
                  )}
                  {task.qa_output_path && (
                    <Descriptions.Item label={t('detail.info.qaOutputPath')} span={2}>
                      <Text code copyable style={{ fontSize: 12 }}>{task.qa_output_path}</Text>
                    </Descriptions.Item>
                  )}
                  {task.qa_filtered_path && (
                    <Descriptions.Item label={t('detail.info.qaFilteredPath')} span={2}>
                      <Text code copyable style={{ fontSize: 12 }}>{task.qa_filtered_path}</Text>
                    </Descriptions.Item>
                  )}
                  {task.filter_stats && (
                    <Descriptions.Item label={t('detail.info.filterStats.label')} span={2}>
                      <Space wrap>
                        <Tag>{t('detail.info.filterStats.total', { count: (task.filter_stats as Record<string, number>).total_pairs })}</Tag>
                        <Tag color="green">{t('detail.info.filterStats.kept', { count: (task.filter_stats as Record<string, number>).kept })}</Tag>
                        <Tag color="red">{t('detail.info.filterStats.filtered', { count: (task.filter_stats as Record<string, number>).filtered_out })}</Tag>
                        {(task.filter_stats as Record<string, number>).unique_chunks != null && (
                          <Tag color="blue">{t('detail.info.filterStats.uniqueChunks', { count: (task.filter_stats as Record<string, number>).unique_chunks })}</Tag>
                        )}
                        {(task.filter_stats as Record<string, number>).milvus_inserted != null && (
                          <Tag color="purple">{t('detail.info.filterStats.milvusInserted', { count: (task.filter_stats as Record<string, number>).milvus_inserted })}</Tag>
                        )}
                      </Space>
                    </Descriptions.Item>
                  )}
                </Descriptions>
              </Card>
            ),
          },
          {
            key: 'config',
            label: (
              <span>
                <SettingOutlined />
                {t('detail.tabs.config')}
              </span>
            ),
            children: (
              <Row gutter={16}>
                <Col span={12}>
                  {renderConfig(task.llm_config as Record<string, unknown>, t('detail.config.llmConfig'))}
                  {renderConfig(task.eval_llm_config as Record<string, unknown> | undefined, t('detail.config.evalLlmConfig'))}
                  {renderConfig(task.embedding_config as Record<string, unknown> | undefined, t('detail.config.embeddingConfig'))}
                  {renderConfig(task.rerank_config as Record<string, unknown> | undefined, t('detail.config.rerankConfig'))}
                </Col>
                <Col span={12}>
                  {renderConfig(task.worker_config as Record<string, unknown>, t('detail.config.workerConfig'))}
                  <Card size="small" title={t('detail.config.enabledSteps')} style={{ marginBottom: 16 }}>
                    {renderStepsConfig()}
                  </Card>
                  {task.post_process_config && renderConfig(task.post_process_config as Record<string, unknown>, t('detail.config.postProcessConfig'))}
                </Col>
              </Row>
            ),
          },
          {
            key: 'artifacts',
            label: (
              <span>
                <DatabaseOutlined />
                {t('detail.tabs.artifacts')}
              </span>
            ),
            children: (
              <Card>
                {artifactsLoading ? (
                  <div style={{ textAlign: 'center', padding: 48 }}>
                    <Spin />
                  </div>
                ) : !artifacts?.stages?.length ? (
                  <Empty description={t('detail.artifacts.empty')} />
                ) : (
                  <>
                    <Tabs
                      activeKey={activeArtifactStage}
                      onChange={(key) => setActiveArtifactStage(key)}
                      items={artifacts.stages.map((stage) => ({
                        key: stage.stage_key,
                        label: (
                          <Space size={6}>
                            <span>{t(`stage.${stage.stage_key}`, { defaultValue: stage.stage_name })}</span>
                            <Tag>{stage.artifacts.length}</Tag>
                          </Space>
                        ),
                      }))}
                      size="small"
                      style={{ marginBottom: 16 }}
                    />

                    {!activeStageData || activeStageData.artifacts.length === 0 ? (
                      <Empty description={t('detail.artifacts.stageEmpty')} />
                    ) : (
                      <Row gutter={16}>
                        <Col span={14}>
                          <Table
                            rowKey="dataset_id"
                            columns={artifactColumns}
                            dataSource={activeStageData.artifacts}
                            pagination={false}
                            size="small"
                          />
                        </Col>
                        <Col span={10}>
                          <Card size="small" title={t('detail.artifacts.previewTitle')}>
                            {!artifactPreviewDatasetId ? (
                              <Empty description={t('detail.artifacts.previewPlaceholder')} />
                            ) : artifactPreviewLoading ? (
                              <div style={{ textAlign: 'center', padding: 24 }}>
                                <Spin />
                              </div>
                            ) : artifactPreviewRows.length === 0 ? (
                              <Empty description={t('detail.artifacts.previewEmpty')} />
                            ) : (
                              <Table
                                rowKey={(_, idx) => String(idx)}
                                columns={artifactPreviewColumns}
                                dataSource={artifactPreviewRows}
                                pagination={false}
                                size="small"
                                scroll={{ x: true }}
                              />
                            )}
                          </Card>
                        </Col>
                      </Row>
                    )}
                  </>
                )}
              </Card>
            ),
          },
          {
            key: 'preview',
            label: (
              <span>
                <DatabaseOutlined />
                {t('detail.tabs.preview')}
              </span>
            ),
            children: (() => {
              const availableSources = getAvailableSources()
              if (!hasPreviewableData || availableSources.length === 0) {
                return (
                  <Card>
                    <Empty description={
                      task.status === 'completed'
                        ? t('detail.preview.noOutput')
                        : task.status === 'running'
                          ? t('detail.preview.generating')
                          : t('detail.preview.notCompleted')
                    } />
                  </Card>
                )
              }
              // Ensure current previewSource is valid
              const activeKey = availableSources.find(s => s.key === previewSource)
                ? previewSource
                : availableSources[0].key
              return (
                <Card>
                  <Tabs
                    activeKey={activeKey}
                    onChange={(key) => {
                      setPreviewSource(key)
                      setPreviewPage(1)
                    }}
                    items={availableSources.map(src => ({
                      key: src.key,
                      label: src.label,
                    }))}
                    size="small"
                    style={{ marginBottom: 16 }}
                  />
                  {previewData.total > 0 ? (
                    <>
                      <div style={{ marginBottom: 8, color: 'var(--tf-text-secondary)', fontSize: 12 }}>
                        {t('detail.preview.records', { count: previewData.total })}
                      </div>
                      <Table
                        columns={getPreviewColumns(activeKey)}
                        dataSource={previewData.samples}
                        rowKey={(_, idx) => String(idx)}
                        loading={previewLoading}
                        pagination={{
                          current: previewPage,
                          pageSize: previewPageSize,
                          total: previewData.total,
                          showTotal: (total) => t('common:pagination.total', { total }),
                          onChange: (page) => setPreviewPage(page),
                        }}
                        size="small"
                      />
                    </>
                  ) : (
                    <Empty description={
                      task.status === 'running'
                        ? t('detail.preview.generating')
                        : t('detail.preview.noOutput')
                    } />
                  )}
                </Card>
              )
            })(),
          },
        ]}
      />
    </div>
  )
}
