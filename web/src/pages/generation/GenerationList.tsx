import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Table,
  Button,
  Space,
  Tag,
  Popconfirm,
  message,
  Typography,
  Progress,
  Card,
  Row,
  Col,
  Tooltip,
} from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  DeleteOutlined,
  StopOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  RedoOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { generationApi, SILENT_REQUEST_CONFIG } from '@/services/api'
import type { GenerationTaskStats, GenerationTaskStatus } from '@/services/api'
import { useTranslation } from 'react-i18next'

const { Title, Text } = Typography
const DEFAULT_PAGE_SIZE = 10
const ACTIVE_GENERATION_STATUSES = new Set<GenerationTaskStatus>([
  'pending',
  'running',
  'stopping',
  'recovering',
  'publishing',
  'restarting',
])

interface GenerationTask {
  task_id: string
  task_name: string
  status: GenerationTaskStatus
  generation_mode: string
  progress: number
  total_docs: number
  processed_docs: number
  output_sample_count: number
  error_message?: string
  created_at?: string
}

const modeColorMap: Record<string, string> = {
  doc_to_training: 'blue',
  qa_to_training: 'purple',
  qa_extraction: 'orange',
  doc_to_eval: 'green',
  qa_to_eval: 'cyan',
}

const statusColorMap: Record<string, string> = {
  pending: 'default',
  running: 'processing',
  stopping: 'processing',
  recovering: 'processing',
  publishing: 'processing',
  restarting: 'processing',
  completed: 'success',
  failed: 'error',
  stopped: 'warning',
  deleting: 'processing',
  deleting_cascade: 'processing',
}

export default function GenerationList() {
  const { t } = useTranslation(['generation', 'common'])
  const navigate = useNavigate()
  const [tasks, setTasks] = useState<GenerationTask[]>([])
  const [loading, setLoading] = useState(false)
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: DEFAULT_PAGE_SIZE,
    total: 0,
  })
  const [stats, setStats] = useState<GenerationTaskStats>({
    total: 0,
    pending: 0,
    running: 0,
    stopping: 0,
    recovering: 0,
    publishing: 0,
    restarting: 0,
    completed: 0,
    failed: 0,
    stopped: 0,
  })
  const tasksRef = useRef<GenerationTask[]>([])
  const paginationRef = useRef(pagination)
  const requestGenerationRef = useRef(0)
  const loadingOwnerRef = useRef(0)

  // 保持 tasksRef 同步
  useEffect(() => {
    tasksRef.current = tasks
  }, [tasks])

  useEffect(() => {
    paginationRef.current = pagination
  }, [pagination])

  const fetchTasks = useCallback(
    async (requestedPage?: number, requestedPageSize?: number, silent = false) => {
      const requestGeneration = ++requestGenerationRef.current
      const loadingOwner = silent ? null : ++loadingOwnerRef.current
      const currentPage = requestedPage ?? paginationRef.current.current
      const pageSize = requestedPageSize ?? paginationRef.current.pageSize
      if (loadingOwner !== null) setLoading(true)
      try {
        const data = await generationApi.listTasks(
          {
            limit: pageSize,
            offset: (currentPage - 1) * pageSize,
          },
          silent ? SILENT_REQUEST_CONFIG : undefined
        )
        if (requestGenerationRef.current !== requestGeneration) return
        setTasks(data.tasks)
        setStats(data.stats)
        const nextPagination = {
          current: currentPage,
          pageSize: data.limit,
          total: data.total,
        }
        paginationRef.current = nextPagination
        setPagination(nextPagination)
      } catch (error) {
        if (requestGenerationRef.current !== requestGeneration) return
        if (!silent) message.error(t('list.message.fetchFailed'))
      } finally {
        if (loadingOwner !== null && loadingOwnerRef.current === loadingOwner) {
          setLoading(false)
        }
      }
    },
    [t]
  )

  useEffect(() => {
    fetchTasks(1, DEFAULT_PAGE_SIZE)
    // 定期刷新运行中的任务
    const interval = setInterval(() => {
      if (tasksRef.current.some((task) => ACTIVE_GENERATION_STATUSES.has(task.status))) {
        fetchTasks(undefined, undefined, true)
      }
    }, 5000)
    return () => {
      clearInterval(interval)
      requestGenerationRef.current += 1
      loadingOwnerRef.current += 1
    }
  }, [fetchTasks])

  const handleStop = async (taskId: string) => {
    try {
      await generationApi.stopTask(taskId)
      message.success(t('list.message.stopSuccess'))
      fetchTasks()
    } catch (error) {
      message.error(t('list.message.stopFailed'))
    }
  }

  const handleDelete = async (taskId: string, status?: string) => {
    try {
      await generationApi.deleteTask(taskId, status === 'deleting_cascade')
      message.success(t('list.message.deleteSuccess'))
      const { current, pageSize } = paginationRef.current
      const nextPage = tasksRef.current.length === 1 && current > 1 ? current - 1 : current
      fetchTasks(nextPage, pageSize)
    } catch (error) {
      message.error(t('list.message.deleteFailed'))
    }
  }

  const handleResume = async (taskId: string) => {
    try {
      await generationApi.restartTask(taskId)
      message.success(t('list.message.resumeSuccess'))
      fetchTasks()
    } catch (error) {
      message.error(t('list.message.resumeFailed'))
    }
  }

  const handleRerun = async (taskId: string) => {
    try {
      await generationApi.restartTask(taskId, true)
      message.success(t('list.message.restartSuccess'))
      fetchTasks()
    } catch (error) {
      message.error(t('list.message.restartFailed'))
    }
  }

  const columns: ColumnsType<GenerationTask> = [
    {
      title: t('list.columns.taskName'),
      dataIndex: 'task_name',
      key: 'task_name',
      width: 200,
      render: (name, record) => (
        <div>
          <a
            onClick={() => navigate(`/datasets/generation/${record.task_id}`)}
            style={{ cursor: 'pointer' }}
          >
            {name}
          </a>
          <br />
          <Tooltip title={record.task_id}>
            <Text
              copyable={{ text: record.task_id, tooltips: false }}
              style={{ color: '#888', fontSize: 11 }}
            >
              {record.task_id.slice(0, 8)}
            </Text>
          </Tooltip>
        </div>
      ),
    },
    {
      title: t('list.columns.mode'),
      dataIndex: 'generation_mode',
      key: 'generation_mode',
      width: 120,
      render: (mode: string) => (
        <Tag color={modeColorMap[mode] || 'default'}>{t(`mode.${mode}`, mode)}</Tag>
      ),
    },
    {
      title: t('list.columns.status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={statusColorMap[status] || 'default'}>
          {status === 'restarting'
            ? t('generationStatus.restarting')
            : t(`common:status.${status}`, status)}
        </Tag>
      ),
    },
    {
      title: t('list.columns.progress'),
      key: 'progress',
      width: 200,
      render: (_, record) => (
        <div>
          <Progress
            percent={Math.round(record.progress)}
            size="small"
            status={record.status === 'failed' ? 'exception' : undefined}
          />
          <span style={{ fontSize: 12, color: '#666' }}>
            {record.processed_docs} / {record.total_docs}
          </span>
        </div>
      ),
    },
    {
      title: t('list.columns.sampleCount'),
      dataIndex: 'output_sample_count',
      key: 'output_sample_count',
      width: 120,
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: t('list.columns.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (val: string | undefined) => (val ? new Date(val).toLocaleString() : '-'),
    },
    {
      title: t('list.columns.errorMessage'),
      dataIndex: 'error_message',
      key: 'error_message',
      width: 200,
      ellipsis: true,
      render: (msg: string | undefined) => msg || '-',
    },
    {
      title: t('list.columns.actions'),
      key: 'actions',
      width: 200,
      render: (_, record) => (
        <Space>
          <Tooltip title={t('list.actions.viewDetail')}>
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/datasets/generation/${record.task_id}`)}
            />
          </Tooltip>
          {(record.status === 'pending' || record.status === 'running') && (
            <Popconfirm
              title={t('list.actions.confirmStop')}
              onConfirm={() => handleStop(record.task_id)}
            >
              <Tooltip title={t('common:action.stop')}>
                <Button type="text" icon={<StopOutlined />} danger />
              </Tooltip>
            </Popconfirm>
          )}
          {(record.status === 'failed' || record.status === 'stopped') && (
            <Popconfirm
              title={t('list.actions.confirmResume')}
              onConfirm={() => handleResume(record.task_id)}
            >
              <Tooltip title={t('detail.resume')}>
                <Button type="text" icon={<PlayCircleOutlined />} style={{ color: '#52c41a' }} />
              </Tooltip>
            </Popconfirm>
          )}
          {(record.status === 'failed' ||
            record.status === 'stopped' ||
            record.status === 'completed') && (
            <Popconfirm
              title={t('list.actions.confirmRerun')}
              onConfirm={() => handleRerun(record.task_id)}
            >
              <Tooltip title={t('detail.rerun')}>
                <Button type="text" icon={<RedoOutlined />} style={{ color: '#1890ff' }} />
              </Tooltip>
            </Popconfirm>
          )}
          {!ACTIVE_GENERATION_STATUSES.has(record.status) && (
            <Popconfirm
              title={t('list.actions.confirmDelete')}
              onConfirm={() => handleDelete(record.task_id, record.status)}
            >
              <Button type="text" icon={<DeleteOutlined />} danger />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  return (
    <div>
      <div
        style={{
          marginBottom: 24,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Title level={4} style={{ margin: 0 }}>
          {t('list.title')}
        </Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => fetchTasks()} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => navigate('/datasets/generation/create')}
          >
            {t('list.createTask')}
          </Button>
        </Space>
      </div>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 'bold' }}>{stats.total}</div>
              <div style={{ color: '#666' }}>{t('list.stats.total')}</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                {stats.running +
                  (stats.stopping ?? 0) +
                  (stats.recovering ?? 0) +
                  (stats.publishing ?? 0) +
                  (stats.restarting ?? 0)}
              </div>
              <div style={{ color: '#666' }}>{t('list.stats.running')}</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
                {stats.completed}
              </div>
              <div style={{ color: '#666' }}>{t('list.stats.completed')}</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#ff4d4f' }}>
                {stats.failed}
              </div>
              <div style={{ color: '#666' }}>{t('list.stats.failed')}</div>
            </div>
          </Card>
        </Col>
      </Row>

      <Table
        columns={columns}
        dataSource={tasks}
        rowKey="task_id"
        loading={loading}
        pagination={{
          current: pagination.current,
          pageSize: pagination.pageSize,
          total: pagination.total,
          showSizeChanger: true,
          pageSizeOptions: [10, 20, 50, 100],
          onChange: (page, pageSize) => {
            const nextPagination = {
              current: page,
              pageSize,
              total: paginationRef.current.total,
            }
            paginationRef.current = nextPagination
            setPagination(nextPagination)
            fetchTasks(page, pageSize)
          },
        }}
      />
    </div>
  )
}
