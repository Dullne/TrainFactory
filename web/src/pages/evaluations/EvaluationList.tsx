import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import {
  Table,
  Typography,
  Tag,
  Space,
  Button,
  Row,
  Col,
  Tooltip,
  message,
  Tabs,
  Progress,
  Popconfirm,
  Alert,
} from 'antd'
import {
  ReloadOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  PlusOutlined,
  PlayCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  EyeOutlined,
  RedoOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import { evaluationApi, SILENT_REQUEST_CONFIG } from '@/services/api'
import { formatDate } from '@/utils'
import { StatCard } from '@/components/StatCard'
import { usePolling } from '@/hooks/usePolling'
import { CreateEvaluationModal } from './CreateEvaluationModal'
import { EvaluationDetailModal } from './EvaluationDetailModal'
import DeepEvaluationTasksPanel from './DeepEvaluationTasksPanel'
import DeepEvaluationOnline from './DeepEvaluationOnline'
import type { EvaluationTask } from '@/types'
import {
  TEXT_PRIMARY,
  TEXT_SECONDARY,
  STATUS_SUCCESS,
  STATUS_INFO,
  STATUS_ERROR,
  STATUS_WARNING,
} from '@/theme'

const { Title, Text } = Typography

export default function EvaluationList() {
  const { t } = useTranslation(['evaluations', 'common'])
  const [activeTab, setActiveTab] = useState('tasks')

  const taskStatusConfig: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
    pending: { color: 'default', icon: <ClockCircleOutlined />, label: t('common:status.pending') },
    running: { color: 'processing', icon: <PlayCircleOutlined spin />, label: t('common:status.running') },
    succeeded: { color: 'success', icon: <CheckCircleOutlined />, label: t('common:status.succeeded') },
    failed: { color: 'error', icon: <CloseCircleOutlined />, label: t('common:status.failed') },
    cancelled: { color: 'warning', icon: <StopOutlined />, label: t('common:status.cancelled') },
  }

  // Evaluation tasks state
  const [tasks, setTasks] = useState<EvaluationTask[]>([])
  const [tasksTotal, setTasksTotal] = useState(0)
  const [tasksLoading, setTasksLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [selectedTask, setSelectedTask] = useState<EvaluationTask | null>(null)
  const [tasksStale, setTasksStale] = useState(false)
  const tasksRequestGenerationRef = useRef(0)
  const tasksLoadingOwnerRef = useRef(0)
  const mountedRef = useRef(true)

  // Check if there are running tasks
  const hasRunningTasks = useMemo(
    () => tasks.some((t) => t.status === 'running' || t.status === 'pending'),
    [tasks]
  )

  const fetchEvaluationTasks = useCallback(async (silent = false) => {
    const requestGeneration = ++tasksRequestGenerationRef.current
    const loadingOwner = silent ? null : ++tasksLoadingOwnerRef.current
    // silent=true 用于后台轮询：不触发表格 loading 闪烁，也不在每 3 秒失败时弹 toast
    if (loadingOwner !== null) setTasksLoading(true)
    try {
      // 真实分页：limit/offset 服务端拉取（此前只取前 50 条却按全局 total 分页）
      const data = await evaluationApi.listTasks(
        { limit: pageSize, offset: (page - 1) * pageSize },
        SILENT_REQUEST_CONFIG
      )
      if (requestGeneration !== tasksRequestGenerationRef.current) return
      const items = data.items || []
      setTasks(items)
      setTasksTotal(data.total || 0)
      // Sync selectedTask with latest data if detail modal is open
      setSelectedTask((prev) => {
        if (!prev) return null
        const updated = items.find((t) => t.task_id === prev.task_id)
        return updated || prev
      })
      setTasksStale(false)
    } catch {
      if (requestGeneration !== tasksRequestGenerationRef.current) return
      if (silent) {
        setTasksStale(true)
      } else {
        message.error({
          key: 'evaluation-list-fetch-failed',
          content: t('list.fetchTasksFailed'),
        })
      }
    } finally {
      if (
        loadingOwner !== null &&
        loadingOwner === tasksLoadingOwnerRef.current &&
        mountedRef.current
      ) {
        setTasksLoading(false)
      }
    }
  }, [t, page, pageSize])

  // Initial fetch
  useEffect(() => {
    if (activeTab === 'tasks') {
      fetchEvaluationTasks()
    }
  }, [activeTab, fetchEvaluationTasks])

  useEffect(
    () => {
      mountedRef.current = true
      return () => {
        mountedRef.current = false
        tasksRequestGenerationRef.current += 1
        tasksLoadingOwnerRef.current += 1
      }
    },
    []
  )

  // Check if detail modal shows a running task
  const detailTaskRunning = detailModalVisible && selectedTask != null &&
    (selectedTask.status === 'running' || selectedTask.status === 'pending')

  // Polling for running tasks (list tab or detail modal)
  // 后台轮询用 silent 模式，避免每 3 秒触发一次表格 loading 闪烁
  usePolling(() => fetchEvaluationTasks(true), {
    interval: 3000,
    enabled: (hasRunningTasks && activeTab === 'tasks') || detailTaskRunning,
  })

  const handleCancelTask = async (taskId: string) => {
    try {
      await evaluationApi.cancelTask(taskId)
      message.success(t('list.taskCancelled'))
      fetchEvaluationTasks()
    } catch (err) {
      message.error(t('list.cancelTaskFailed', { error: (err as Error).message }))
    }
  }

  const handleDeleteTask = async (taskId: string) => {
    try {
      await evaluationApi.deleteTask(taskId)
      message.success(t('list.taskDeleted'))
      fetchEvaluationTasks()
    } catch (err) {
      message.error(t('list.deleteTaskFailed', { error: (err as Error).message }))
    }
  }

  const handleResumeTask = async (taskId: string) => {
    try {
      const result = await evaluationApi.resumeTask(taskId)
      if (result.skipped_evaluations > 0) {
        message.success(t('list.taskResumedWithSkipped', { count: result.skipped_evaluations }))
      } else {
        message.success(t('list.taskResumed'))
      }
      fetchEvaluationTasks()
    } catch (err) {
      message.error(t('list.resumeTaskFailed', { error: (err as Error).message }))
    }
  }

  const handleViewDetail = (task: EvaluationTask) => {
    setSelectedTask(task)
    setDetailModalVisible(true)
  }

  const modelLabels: Record<string, string> = {
    reranker: 'Reranker',
    embedding: 'Embedding',
    llm: 'LLM',
    single: t('list.typeLabel.singleModel'),
  }
  const sourceLabels: Record<string, string> = {
    mteb: 'MTEB',
    local: t('list.typeLabel.local'),
    registered: t('list.typeLabel.registered'),
    mixed: t('list.typeLabel.mixed'),
  }
  const sourceColors: Record<string, string> = {
    mteb: 'blue',
    local: 'green',
    registered: 'purple',
    mixed: 'orange',
  }

  // Evaluation tasks columns
  const tasksColumns: ColumnsType<EvaluationTask> = [
    {
      title: t('list.column.taskName'),
      dataIndex: 'task_name',
      key: 'task_name',
      render: (name, record) => (
        <div>
          <Text strong style={{ color: TEXT_PRIMARY }}>
            {name || record.task_id.substring(0, 8)}
          </Text>
          <br />
          <Tooltip title={record.task_id}>
            <Text copyable={{ text: record.task_id, tooltips: false }} style={{ color: TEXT_SECONDARY, fontSize: 11 }}>
              {record.task_id.substring(0, 8)}
            </Text>
          </Tooltip>
        </div>
      ),
    },
    {
      title: t('list.column.type'),
      dataIndex: 'eval_type',
      key: 'eval_type',
      render: (type: string) => {
        // Parse eval_type format: "model_type-dataset_source" e.g. "reranker-mteb"
        const [modelType, datasetSource] = type?.split('-') || [type, '']

        return (
          <>
            <Tag>{modelLabels[modelType] || modelType}</Tag>
            {datasetSource && (
              <Tag color={sourceColors[datasetSource] || 'default'}>
                {sourceLabels[datasetSource] || datasetSource}
              </Tag>
            )}
          </>
        )
      },
    },
    {
      title: t('list.column.modelCount'),
      key: 'models',
      render: (_, record) => {
        const count = record.model_configs?.length || 0
        return <Tag color="blue">{t('list.modelCountTag', { count })}</Tag>
      },
    },
    {
      title: t('list.column.datasetCount'),
      key: 'datasets',
      render: (_, record) => {
        const count = record.dataset_configs?.length || 0
        return <Tag color="green">{t('list.datasetCountTag', { count })}</Tag>
      },
    },
    {
      title: t('list.column.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record) => {
        const config = taskStatusConfig[status] || taskStatusConfig.pending
        return (
          <Space direction="vertical" size={4}>
            <Tag icon={config.icon} color={config.color}>
              {config.label}
            </Tag>
            {status === 'running' && (
              <div style={{ width: 100 }}>
                <Progress
                  percent={Math.round(record.progress || 0)}
                  size="small"
                  status="active"
                />
              </div>
            )}
            {status === 'running' && record.current_model && (
              <Text style={{ fontSize: 11, color: TEXT_SECONDARY }}>
                {record.current_model}
                {record.current_dataset && ` @ ${record.current_dataset}`}
              </Text>
            )}
            {status === 'failed' && record.error_message && (
              <Tooltip title={record.error_message}>
                <Text style={{ fontSize: 11, color: STATUS_ERROR, maxWidth: 150, display: 'block' }} ellipsis>
                  {record.error_message}
                </Text>
              </Tooltip>
            )}
          </Space>
        )
      },
    },
    {
      title: t('list.column.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => formatDate(time),
    },
    {
      title: t('list.column.actions'),
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title={record.status === 'succeeded' ? t('list.tooltip.viewResults') : record.status === 'failed' ? t('list.tooltip.viewDetails') : t('list.tooltip.viewConfig')}>
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetail(record)}
            />
          </Tooltip>
          {(record.status === 'pending' || record.status === 'running') && (
            <Tooltip title={t('list.tooltip.cancelTask')}>
              <Popconfirm
                title={t('list.confirm.cancelTask')}
                onConfirm={() => handleCancelTask(record.task_id)}
              >
                <Button type="text" size="small" danger icon={<StopOutlined />} />
              </Popconfirm>
            </Tooltip>
          )}
          {(record.status === 'failed' || record.status === 'cancelled') && (
            <Tooltip title={t('list.tooltip.retryTask')}>
              <Popconfirm
                title={t('list.confirm.retryTask')}
                onConfirm={() => handleResumeTask(record.task_id)}
              >
                <Button type="text" size="small" icon={<RedoOutlined />} style={{ color: '#1890ff' }} />
              </Popconfirm>
            </Tooltip>
          )}
          {record.status !== 'running' && record.status !== 'pending' && (
            <Tooltip title={t('list.tooltip.deleteTask')}>
              <Popconfirm
                title={t('list.confirm.deleteTask')}
                onConfirm={() => handleDeleteTask(record.task_id)}
              >
                <Button type="text" size="small" danger icon={<DeleteOutlined />} />
              </Popconfirm>
            </Tooltip>
          )}
        </Space>
      ),
    },
  ]

  // Task statistics
  const taskStats = useMemo(() => {
    const stats = { total: tasksTotal, running: 0, succeeded: 0, failed: 0 }
    tasks.forEach((t) => {
      if (t.status === 'running' || t.status === 'pending') stats.running++
      else if (t.status === 'succeeded') stats.succeeded++
      else if (t.status === 'failed') stats.failed++
    })
    return stats
  }, [tasks, tasksTotal])

  return (
    <div>
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        destroyInactiveTabPane
        items={[
          {
            key: 'tasks',
            label: t('tabs.tasks'),
            children: (
              <>
                {tasksStale && (
                  <Alert
                    data-testid="evaluation-stale-state"
                    type="warning"
                    showIcon
                    message={t('list.pollingStale')}
                    style={{ marginBottom: 16 }}
                  />
                )}
                {/* Task Statistics */}
                <Row gutter={16} style={{ marginBottom: 20 }}>
                  <Col span={6}>
                    <StatCard
                      title={t('stats.totalTasks')}
                      value={taskStats.total}
                      icon={<BarChartOutlined />}
                      color={STATUS_INFO}
                    />
                  </Col>
                  <Col span={6}>
                    <StatCard
                      title={t('stats.running')}
                      value={taskStats.running}
                      icon={<PlayCircleOutlined />}
                      color={STATUS_WARNING}
                    />
                  </Col>
                  <Col span={6}>
                    <StatCard
                      title={t('stats.completed')}
                      value={taskStats.succeeded}
                      icon={<CheckCircleOutlined />}
                      color={STATUS_SUCCESS}
                    />
                  </Col>
                  <Col span={6}>
                    <StatCard
                      title={t('stats.failed')}
                      value={taskStats.failed}
                      icon={<CloseCircleOutlined />}
                      color={STATUS_ERROR}
                    />
                  </Col>
                </Row>

                {/* Toolbar */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                  <Title level={4} style={{ margin: 0 }}>
                    {t('list.title')}
                  </Title>
                  <Space>
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={() => fetchEvaluationTasks()}
                      loading={tasksLoading}
                    >
                      {t('common:action.refresh')}
                    </Button>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => setCreateModalVisible(true)}
                    >
                      {t('list.createTask')}
                    </Button>
                  </Space>
                </div>

                {/* Tasks Table */}
                <Table
                  rowKey="task_id"
                  columns={tasksColumns}
                  dataSource={tasks}
                  loading={tasksLoading}
                  pagination={{
                    current: page,
                    pageSize,
                    showSizeChanger: true,
                    showTotal: (total) => t('list.totalResults', { total }),
                    total: tasksTotal,
                    onChange: (p, ps) => {
                      setPage(p)
                      setPageSize(ps)
                    },
                  }}
                />
              </>
            ),
          },
          {
            key: 'deep-eval',
            label: t('tabs.deepEval'),
            children: <DeepEvaluationTasksPanel />,
          },
          {
            key: 'online-test',
            label: t('tabs.onlineTest'),
            children: <DeepEvaluationOnline />,
          },
        ]}
      />

      {/* Create Evaluation Modal */}
      <CreateEvaluationModal
        visible={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        onSuccess={() => {
          setCreateModalVisible(false)
          fetchEvaluationTasks()
        }}
      />

      {/* Evaluation Detail Modal */}
      <EvaluationDetailModal
        visible={detailModalVisible}
        task={selectedTask}
        onClose={() => {
          setDetailModalVisible(false)
          setSelectedTask(null)
        }}
      />
    </div>
  )
}
