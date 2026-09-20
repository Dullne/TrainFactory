import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Table,
  Alert,
  Button,
  Space,
  Tag,
  Progress,
  Popconfirm,
  message,
  Select,
  Typography,
  Tooltip,
  Row,
  Col,
  Grid,
} from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  StopOutlined,
  DeleteOutlined,
  EyeOutlined,
  ExperimentOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { trainingApi } from '@/services/api'
import { usePolling } from '@/hooks'
import { useWorkspaceList } from '@/hooks/useListWorkspace'
import { MobileList, renderListCell } from '@/components/ListWorkspace'
import { formatDate, getStatusDescription, isActiveStatus } from '@/utils'
import { StatusTag } from '@/components/StatusTag'
import { StatCard } from '@/components/StatCard'
import type { TrainingTask } from '@/types'
import './TrainingList.css'
import {
  MODEL_TYPE_COLORS,
  TRAINING_METHOD_COLORS,
  TEXT_PRIMARY,
  TEXT_SECONDARY,
  STATUS_INFO,
  STATUS_SUCCESS,
  STATUS_ERROR,
  STATUS_WARNING,
} from '@/theme'

const { Title, Text } = Typography

// Stats type from API
interface TaskStats {
  total: number
  pending: number
  running: number
  succeeded: number
  failed: number
  stopped: number
}

const MODEL_TYPE_KEYS: Record<string, string> = {
  embedding: 'common:modelType.embedding',
  reranker: 'common:modelType.reranker',
  decoder_reranker: 'common:modelType.decoderReranker',
  llm: 'common:modelType.llm',
}

const LIST_FILTERS = {
  status: [
    'pending',
    'preparing',
    'running',
    'evaluating',
    'succeeded',
    'failed',
    'stopped',
    'cancelled',
  ],
}

export default function TrainingList() {
  const navigate = useNavigate()
  const { t } = useTranslation(['training', 'common'])
  const mobile = Grid.useBreakpoint().xs === true

  const {
    data: tasks,
    loading,
    error,
    hasData,
    isStale,
    page,
    pageSize,
    total,
    extra,
    refresh,
    workspace,
  } = useWorkspaceList<TrainingTask, { stats?: TaskStats }>(trainingApi.list, {
    filters: LIST_FILTERS,
  })
  const statusFilter = workspace.values.status
  const setStatusFilter = (value: string | undefined) => workspace.setFilter('status', value)

  // Auto refresh when there are running tasks
  const hasRunningTasks = tasks.some((t) => isActiveStatus(t.status))
  usePolling(refresh, {
    interval: 5000,
    enabled: hasRunningTasks,
  })

  // Use stats from backend API (global stats, not affected by pagination)
  const stats = extra?.stats ?? {
    total: 0,
    pending: 0,
    running: 0,
    succeeded: 0,
    failed: 0,
    stopped: 0,
  }

  const handleStop = async (taskId: string) => {
    try {
      await trainingApi.stop(taskId)
      message.success(t('list.message.stopSent'))
      refresh()
    } catch {
      message.error(t('list.message.stopFailed'))
    }
  }

  const handleDelete = async (taskId: string) => {
    try {
      await trainingApi.delete(taskId)
      message.success(t('common:message.deleteSuccess'))
      refresh()
    } catch {
      message.error(t('common:message.deleteFailed'))
    }
  }

  const handleResume = async (taskId: string) => {
    try {
      const res = await trainingApi.resume(taskId)
      message.success(
        t('list.message.resumeSuccess', { checkpoint: res.checkpoint.split('/').pop() })
      )
      refresh()
    } catch (err: unknown) {
      message.error(err instanceof Error ? err.message : t('list.message.resumeFailed'))
    }
  }

  const columns: ColumnsType<TrainingTask> = [
    {
      title: t('list.columns.taskName'),
      dataIndex: 'task_name',
      key: 'task_name',
      render: (name, record) => (
        <div>
          <Text strong style={{ color: TEXT_PRIMARY }}>
            {name || record.task_id.slice(0, 8)}
          </Text>
          <br />
          <Tooltip title={record.task_id}>
            <Text
              copyable={{ text: record.task_id, tooltips: false }}
              style={{ color: TEXT_SECONDARY, fontSize: 11 }}
            >
              {record.task_id.slice(0, 8)}
            </Text>
          </Tooltip>
        </div>
      ),
    },
    {
      title: t('list.columns.baseModel'),
      dataIndex: 'base_model_path',
      key: 'base_model_path',
      ellipsis: true,
      render: (path) => {
        if (!path) return '-'
        const parts = path.split('/')
        const name = parts[parts.length - 1] || path
        return (
          <Tooltip title={path}>
            <Text style={{ color: TEXT_PRIMARY }}>{name}</Text>
          </Tooltip>
        )
      },
    },
    {
      title: t('list.columns.type'),
      key: 'type_method',
      render: (_, record) => {
        const typeColor = MODEL_TYPE_COLORS[record.model_type] || '#8b949e'
        const methodColor = TRAINING_METHOD_COLORS[record.training_method] || '#8b949e'
        return (
          <Space size={4} wrap>
            <Tag
              style={{
                backgroundColor: `${typeColor}20`,
                color: typeColor,
                border: `1px solid ${typeColor}40`,
              }}
            >
              {MODEL_TYPE_KEYS[record.model_type]
                ? t(MODEL_TYPE_KEYS[record.model_type])
                : record.model_type}
            </Tag>
            <Tag
              style={{
                backgroundColor: `${methodColor}20`,
                color: methodColor,
                border: `1px solid ${methodColor}40`,
              }}
            >
              {record.training_method?.toUpperCase()}
            </Tag>
          </Space>
        )
      },
    },
    {
      title: t('list.columns.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => <StatusTag status={status} />,
    },
    {
      title: t('list.columns.progress'),
      key: 'progress',
      width: 180,
      render: (_, record) => (
        <div>
          <Progress
            percent={record.progress}
            size="small"
            status={record.status === 'failed' ? 'exception' : undefined}
          />
          {record.current_step != null && record.total_steps != null && (
            <Text style={{ fontSize: 11, color: TEXT_SECONDARY }}>
              {t('list.step', { current: record.current_step, total: record.total_steps })}
            </Text>
          )}
          {(record.current_step == null || record.total_steps == null) &&
            getStatusDescription(record.status) && (
              <Text style={{ fontSize: 11, color: TEXT_SECONDARY }}>
                {getStatusDescription(record.status)}
              </Text>
            )}
        </div>
      ),
    },
    {
      title: t('list.columns.loss'),
      key: 'loss',
      width: 130,
      render: (_, record) => {
        if (record.train_loss == null && record.eval_loss == null) {
          return <Text style={{ color: TEXT_SECONDARY }}>-</Text>
        }
        return (
          <Space direction="vertical" size={0}>
            {typeof record.train_loss === 'number' && (
              <Text style={{ fontSize: 12, color: TEXT_PRIMARY }}>
                {t('list.trainLoss', { loss: record.train_loss.toFixed(4) })}
              </Text>
            )}
            {typeof record.eval_loss === 'number' && (
              <Text style={{ fontSize: 12, color: TEXT_PRIMARY }}>
                {t('list.evalLoss', { loss: record.eval_loss.toFixed(4) })}
              </Text>
            )}
          </Space>
        )
      },
    },
    {
      title: t('list.columns.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => (
        <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>{formatDate(time)}</Text>
      ),
    },
    {
      title: t('list.columns.actions'),
      key: 'actions',
      width: 130,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Tooltip title={t('common:action.detail')}>
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              aria-label={t('common:action.detail')}
              onClick={() => navigate(`/training/${record.task_id}`)}
            />
          </Tooltip>
          {isActiveStatus(record.status) && (
            <Tooltip title={t('common:action.stop')}>
              <Popconfirm
                title={t('list.actions.confirmStop')}
                onConfirm={() => handleStop(record.task_id)}
              >
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<StopOutlined />}
                  aria-label={t('common:action.stop')}
                />
              </Popconfirm>
            </Tooltip>
          )}
          {(record.status === 'failed' || record.status === 'stopped') && (
            <Tooltip title={t('list.actions.resumeFromCheckpoint')}>
              <Popconfirm
                title={t('list.actions.confirmResume')}
                onConfirm={() => handleResume(record.task_id)}
              >
                <Button
                  type="text"
                  size="small"
                  icon={<PlayCircleOutlined />}
                  aria-label={t('list.actions.resumeFromCheckpoint')}
                />
              </Popconfirm>
            </Tooltip>
          )}
          {!isActiveStatus(record.status) && (
            <Tooltip title={t('common:action.delete')}>
              <Popconfirm
                title={t('list.actions.confirmDelete')}
                onConfirm={() => handleDelete(record.task_id)}
              >
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  aria-label={t('common:action.delete')}
                />
              </Popconfirm>
            </Tooltip>
          )}
        </Space>
      ),
    },
  ]

  return (
    <div className="training-list-page">
      {/* Statistics */}
      <Row className="training-list-stats" gutter={[12, 12]} style={{ marginBottom: 20 }}>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stats.total')}
            value={stats.total}
            icon={<ExperimentOutlined />}
            color={STATUS_INFO}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stats.running')}
            value={stats.running}
            icon={<PlayCircleOutlined />}
            color={STATUS_WARNING}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stats.succeeded')}
            value={stats.succeeded}
            icon={<CheckCircleOutlined />}
            color={STATUS_SUCCESS}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stats.failed')}
            value={stats.failed}
            icon={<CloseCircleOutlined />}
            color={STATUS_ERROR}
          />
        </Col>
      </Row>

      {/* Toolbar */}
      <div className="training-list-toolbar">
        <Title className="training-list-toolbar-title" level={4} style={{ margin: 0 }}>
          {t('list.title')}
        </Title>
        <div className="training-list-toolbar-actions">
          <Select
            className="training-list-status-filter"
            placeholder={t('list.filter.statusPlaceholder')}
            allowClear
            value={statusFilter}
            onChange={setStatusFilter}
            options={[
              { label: t('list.filter.pending'), value: 'pending' },
              { label: t('list.filter.preparing'), value: 'preparing' },
              { label: t('list.filter.running'), value: 'running' },
              { label: t('list.filter.evaluating'), value: 'evaluating' },
              { label: t('list.filter.succeeded'), value: 'succeeded' },
              { label: t('list.filter.failed'), value: 'failed' },
              { label: t('list.filter.stopped'), value: 'stopped' },
              { label: t('list.filter.cancelled'), value: 'cancelled' },
            ]}
          />
          <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => navigate('/training/create')}
          >
            {t('list.actions.createTask')}
          </Button>
        </div>
      </div>

      {/* Table */}
      {(error || isStale) && (
        <Alert
          showIcon
          type={error ? 'warning' : 'info'}
          style={{ marginBottom: 16 }}
          message={t(
            error
              ? hasData
                ? 'common:listState.refreshFailed'
                : 'common:listState.loadFailed'
              : 'common:listState.showingPrevious'
          )}
        />
      )}
      {mobile ? (
        <MobileList
          loading={loading}
          empty={tasks.length === 0}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            showTotal: (count) => t('common:pagination.total', { total: count }),
            onChange: workspace.setPagination,
          }}
        >
          {tasks.map((task) => (
            <article
              key={task.task_id}
              className="list-workspace-card"
              aria-label={task.task_name || task.task_id}
            >
              <div className="list-workspace-card-header">
                {renderListCell(columns, 'task_name', task)}
                <StatusTag status={task.status} />
              </div>
              {renderListCell(columns, 'progress', task)}
              <dl className="list-workspace-fields">
                <div className="list-workspace-field list-workspace-field-wide">
                  <dt>{t('list.columns.baseModel')}</dt>
                  <dd>{renderListCell(columns, 'base_model_path', task)}</dd>
                </div>
                <div className="list-workspace-field">
                  <dt>{t('list.columns.type')}</dt>
                  <dd>{renderListCell(columns, 'type_method', task)}</dd>
                </div>
                <div className="list-workspace-field">
                  <dt>{t('list.columns.loss')}</dt>
                  <dd>{renderListCell(columns, 'loss', task)}</dd>
                </div>
                <div className="list-workspace-field list-workspace-field-wide">
                  <dt>{t('list.columns.createdAt')}</dt>
                  <dd>{formatDate(task.created_at)}</dd>
                </div>
              </dl>
              <div className="list-workspace-actions">
                {renderListCell(columns, 'actions', task)}
              </div>
            </article>
          ))}
        </MobileList>
      ) : (
        <Table
          rowKey="task_id"
          columns={columns}
          dataSource={tasks}
          loading={loading}
          scroll={{ x: 1000 }}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            showTotal: (total) => t('common:pagination.total', { total }),
            onChange: (p, ps) => {
              workspace.setPagination(p, ps)
            },
          }}
        />
      )}
    </div>
  )
}
