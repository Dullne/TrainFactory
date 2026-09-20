import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Table,
  Alert,
  Button,
  Space,
  Tag,
  Popconfirm,
  Popover,
  Checkbox,
  message,
  Select,
  Typography,
  Tooltip,
  Row,
  Col,
} from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  CloudDownloadOutlined,
  CloudUploadOutlined,
  EditOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  NodeIndexOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import { datasetApi } from '@/services/api'
import { useWorkspaceList } from '@/hooks/useListWorkspace'
import { formatDate, formatBytes } from '@/utils'
import type { Dataset, DatasetType, DatasetUsage, DatasetModelType } from '@/types'
import { StatCard } from '@/components/StatCard'
import { DownloadDatasetModal } from './DownloadDatasetModal'
import { UploadDatasetModal } from './UploadDatasetModal'
import { STATUS_INFO, STATUS_SUCCESS, STATUS_WARNING, STATUS_DEFAULT } from '@/theme'
import { useAuth } from '@/auth/AuthContext'

const { Title, Text } = Typography

const typeColorMap: Record<DatasetType, string> = {
  // Embedding types (10 types)
  embedding_universal: 'purple',
  embedding_pair: 'blue',
  embedding_triplet: 'cyan',
  embedding_multi_neg: 'geekblue',
  embedding_dynamic_neg: 'volcano',
  embedding_cosine: 'gold',
  embedding_margin: 'lime',
  embedding_margin_multi: 'green',
  embedding_scored: 'orange',
  embedding_score_triplet: 'magenta',
  // Rerank types
  rerank_pair: 'processing',
  rerank_triplet: 'warning',
  rerank_listwise: 'error',
  // LLM types
  sft_instruct: 'success',
  dpo_preference: 'red',
  rl_reward: 'pink',
  // Generation intermediate types
  qa_pair: 'orange',
  custom: 'default',
}

const LIST_FILTERS = {
  dataset_type: Object.keys(typeColorMap),
  usage: ['raw', 'train', 'eval', 'test'],
  model_type: ['embedding', 'rerank', 'llm'],
  status: ['ready', 'registered', 'uploading', 'downloading', 'processing', 'error', 'archived'],
}

export default function DatasetList() {
  const { t } = useTranslation(['datasets', 'common'])
  const navigate = useNavigate()
  const { config } = useAuth()
  const [downloadModalOpen, setDownloadModalOpen] = useState(false)
  const [uploadModalOpen, setUploadModalOpen] = useState(false)

  const usageOptions: { label: string; value: DatasetUsage }[] = [
    { label: t('options.usage.raw'), value: 'raw' },
    { label: t('options.usage.train'), value: 'train' },
    { label: t('options.usage.eval'), value: 'eval' },
    { label: t('options.usage.test'), value: 'test' },
  ]

  const modelTypeOptions: { label: string; value: DatasetModelType }[] = [
    { label: 'Embedding', value: 'embedding' },
    { label: 'Rerank', value: 'rerank' },
    { label: 'LLM', value: 'llm' },
  ]

  const statusOptions = [
    { label: t('options.status.ready'), value: 'ready' },
    { label: t('options.status.registered'), value: 'registered' },
    { label: t('options.status.uploading'), value: 'uploading' },
    { label: t('options.status.downloading'), value: 'downloading' },
    { label: t('options.status.processing'), value: 'processing' },
    { label: t('options.status.error'), value: 'error' },
    { label: t('options.status.archived'), value: 'archived' },
  ]

  const {
    data: datasets,
    loading,
    error,
    hasData,
    isStale,
    page,
    pageSize,
    total,
    refresh,
    extra,
    workspace,
  } = useWorkspaceList<
    Dataset,
    {
      stats?: {
        total: number
        by_usage?: Record<string, number>
        by_status?: Record<string, number>
      }
    }
  >(datasetApi.list, { filters: LIST_FILTERS })
  const typeFilter = workspace.values.dataset_type
  const usageFilter = workspace.values.usage
  const modelTypeFilter = workspace.values.model_type
  const statusFilter = workspace.values.status
  const setTypeFilter = (value: string | undefined) => workspace.setFilter('dataset_type', value)
  const setUsageFilter = (value: string | undefined) => workspace.setFilter('usage', value)
  const setModelTypeFilter = (value: string | undefined) => workspace.setFilter('model_type', value)
  const setStatusFilter = (value: string | undefined) => workspace.setFilter('status', value)

  // Global stats from the backend (filter/pagination-independent); fall back to
  // the current page only if the API did not return stats.
  const stats = useMemo(() => {
    const s = extra?.stats
    if (s) {
      const usage = s.by_usage ?? {}
      const status = s.by_status ?? {}
      return {
        total: s.total ?? total,
        train: usage.train ?? 0,
        eval: usage.eval ?? 0,
        test: usage.test ?? 0,
        ready: status.ready ?? 0,
      }
    }
    const train = datasets.filter((d) => d.usage === 'train').length
    const evalCount = datasets.filter((d) => d.usage === 'eval').length
    const testCount = datasets.filter((d) => d.usage === 'test').length
    const ready = datasets.filter((d) => d.status === 'ready').length
    return { total, train, eval: evalCount, test: testCount, ready }
  }, [datasets, total, extra])

  // Refetch when filters change

  const handleDelete = async (datasetId: string) => {
    try {
      await datasetApi.delete(datasetId)
      message.success(t('common:message.deleteSuccess'))
      refresh()
    } catch {
      message.error(t('common:message.deleteFailed'))
    }
  }

  const handleUpdateModelType = async (datasetId: string, modelTypes: DatasetModelType[]) => {
    try {
      await datasetApi.update(datasetId, { model_type: modelTypes })
      message.success(t('list.message.modelTypeUpdated'))
      refresh()
    } catch {
      message.error(t('list.message.updateFailed'))
    }
  }

  const columns: ColumnsType<Dataset> = [
    {
      title: t('list.columns.name'),
      dataIndex: 'dataset_name',
      key: 'dataset_name',
      width: 220,
      render: (_, record) => {
        const name = record.dataset_name || record.name || record.display_name
        return (
          <div style={{ minWidth: 0 }}>
            <Tooltip title={name} placement="topLeft">
              <a
                onClick={() => navigate(`/datasets/${record.dataset_id}`)}
                style={{
                  cursor: 'pointer',
                  display: 'block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {name}
              </a>
            </Tooltip>
            <Tooltip title={record.dataset_id}>
              <Text
                copyable={{ text: record.dataset_id, tooltips: false }}
                style={{ color: '#888', fontSize: 11 }}
              >
                {record.dataset_id.slice(0, 8)}
              </Text>
            </Tooltip>
          </div>
        )
      },
    },
    {
      title: t('list.columns.usage'),
      dataIndex: 'usage',
      key: 'usage',
      width: 80,
      render: (usage: DatasetUsage) => {
        const colorMap: Record<string, string> = {
          raw: 'purple',
          train: 'blue',
          eval: 'green',
          test: 'orange',
        }
        const labelMap: Record<string, string> = {
          raw: t('options.usage.raw'),
          train: t('options.usage.train'),
          eval: t('options.usage.eval'),
          test: t('options.usage.test'),
        }
        return <Tag color={colorMap[usage] || 'default'}>{labelMap[usage] || usage}</Tag>
      },
    },
    {
      title: t('list.columns.status'),
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          ready: 'success',
          registered: 'warning',
          uploading: 'processing',
          downloading: 'processing',
          processing: 'processing',
          error: 'error',
          archived: 'default',
        }
        return (
          <Tag color={colorMap[status] || 'default'}>{t(`options.status.${status}`, status)}</Tag>
        )
      },
    },
    {
      title: t('list.columns.modelType'),
      dataIndex: 'model_type',
      key: 'model_type',
      render: (modelTypes: DatasetModelType[], record: Dataset) => {
        const colorMap: Record<string, string> = {
          embedding: 'purple',
          rerank: 'orange',
          llm: 'cyan',
        }
        const tags = modelTypes?.length ? (
          <Space size={[0, 4]} wrap>
            {modelTypes.map((mt) => (
              <Tag key={mt} color={colorMap[mt] || 'default'}>
                {mt.toUpperCase()}
              </Tag>
            ))}
          </Space>
        ) : (
          <span style={{ color: '#999' }}>{t('list.notLabeled')}</span>
        )
        return (
          <Popover
            trigger="click"
            placement="bottom"
            content={
              <Checkbox.Group
                options={modelTypeOptions}
                defaultValue={modelTypes || []}
                onChange={(values) =>
                  handleUpdateModelType(record.dataset_id, values as DatasetModelType[])
                }
              />
            }
          >
            <span style={{ cursor: 'pointer' }}>
              {tags} <EditOutlined style={{ color: '#999', fontSize: 12 }} />
            </span>
          </Popover>
        )
      },
    },
    {
      title: t('list.columns.datasetType'),
      dataIndex: 'dataset_type',
      key: 'dataset_type',
      width: 130,
      render: (type: DatasetType) => (
        <Tooltip title={type}>
          <Tag
            color={typeColorMap[type]}
            style={{
              maxWidth: '100%',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {type}
          </Tag>
        </Tooltip>
      ),
    },
    {
      title: t('list.columns.format'),
      dataIndex: 'file_format',
      key: 'file_format',
      width: 70,
      render: (format) => <Tag>{format}</Tag>,
    },
    {
      title: t('list.columns.size'),
      key: 'file_size',
      width: 90,
      render: (_, record) => formatBytes(record.file_size ?? record.size_bytes ?? 0),
    },
    {
      title: t('list.columns.numRows'),
      key: 'num_rows',
      width: 90,
      render: (_, record) => {
        const total = record.num_rows ?? record.num_samples
        const hasSplits = record.num_train || record.num_eval || record.num_test

        if (!total) return '-'

        if (hasSplits) {
          const splits = []
          if (record.num_train)
            splits.push(t('list.splitTooltip.train', { count: record.num_train }))
          if (record.num_eval) splits.push(t('list.splitTooltip.eval', { count: record.num_eval }))
          if (record.num_test) splits.push(t('list.splitTooltip.test', { count: record.num_test }))

          return (
            <Tooltip
              title={
                <div>
                  {splits.map((s, i) => (
                    <div key={i}>{s}</div>
                  ))}
                </div>
              }
              placement="top"
            >
              <span
                style={{
                  borderBottom: '1px dashed #1890ff',
                  cursor: 'help',
                  color: '#1890ff',
                }}
              >
                {total.toLocaleString()}
              </span>
            </Tooltip>
          )
        }

        return total.toLocaleString()
      },
    },
    {
      title: t('list.columns.storagePath'),
      dataIndex: 'storage_path',
      key: 'storage_path',
      width: 180,
      ellipsis: { showTitle: false },
      render: (path: string | null | undefined) =>
        path ? (
          <Tooltip title={path} placement="topLeft">
            <span>{path}</span>
          </Tooltip>
        ) : (
          <span style={{ color: '#999' }}>—</span>
        ),
    },
    {
      title: t('list.columns.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => formatDate(time),
    },
    {
      title: t('list.columns.actions'),
      key: 'actions',
      width: 100,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Tooltip title={t('list.viewDetail')}>
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/datasets/${record.dataset_id}`)}
            />
          </Tooltip>
          <Popconfirm
            title={t('list.confirmDelete')}
            onConfirm={() => handleDelete(record.dataset_id)}
          >
            <Button
              type="link"
              size="small"
              danger
              icon={<DeleteOutlined />}
              title={t('common:action.delete')}
            />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col flex="1">
          <StatCard
            title={t('list.stats.totalDatasets')}
            value={stats.total}
            icon={<DatabaseOutlined />}
            color={STATUS_INFO}
          />
        </Col>
        <Col flex="1">
          <StatCard
            title={t('list.stats.trainSets')}
            value={stats.train}
            icon={<ExperimentOutlined />}
            color={STATUS_SUCCESS}
          />
        </Col>
        <Col flex="1">
          <StatCard
            title={t('list.stats.evalSets')}
            value={stats.eval}
            icon={<CheckCircleOutlined />}
            color={STATUS_WARNING}
          />
        </Col>
        <Col flex="1">
          <StatCard
            title={t('list.stats.testSets')}
            value={stats.test}
            icon={<EyeOutlined />}
            color={STATUS_WARNING}
          />
        </Col>
        <Col flex="1">
          <StatCard
            title={t('list.stats.ready')}
            value={stats.ready}
            icon={<NodeIndexOutlined />}
            color={STATUS_DEFAULT}
          />
        </Col>
      </Row>

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 16,
          gap: 16,
        }}
      >
        <Title level={4} style={{ margin: 0, whiteSpace: 'nowrap', flexShrink: 0 }}>
          {t('list.title')}
        </Title>
        <Space wrap>
          <Select
            placeholder={t('list.filter.status')}
            allowClear
            style={{ width: 100 }}
            value={statusFilter}
            onChange={setStatusFilter}
            options={statusOptions}
          />
          <Select
            placeholder={t('list.filter.usage')}
            allowClear
            style={{ width: 100 }}
            value={usageFilter}
            onChange={setUsageFilter}
            options={usageOptions}
          />
          <Select
            placeholder={t('list.filter.modelType')}
            allowClear
            style={{ width: 120 }}
            value={modelTypeFilter}
            onChange={setModelTypeFilter}
            options={modelTypeOptions}
          />
          <Select
            placeholder={t('list.filter.datasetType')}
            allowClear
            style={{ width: 160 }}
            value={typeFilter}
            onChange={setTypeFilter}
            options={Object.keys(typeColorMap).map((type) => ({
              label: type,
              value: type,
            }))}
          />
          <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          <Button icon={<CloudDownloadOutlined />} onClick={() => setDownloadModalOpen(true)}>
            {t('list.downloadDataset')}
          </Button>
          <Button icon={<CloudUploadOutlined />} onClick={() => setUploadModalOpen(true)}>
            {t('list.uploadDataset')}
          </Button>
          {config.direct_storage_registration_enabled ? (
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => navigate('/datasets/create')}
            >
              {t('list.registerDataset')}
            </Button>
          ) : null}
        </Space>
      </div>

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
      <Table
        rowKey="dataset_id"
        columns={columns}
        dataSource={datasets}
        loading={loading}
        scroll={{ x: 1400 }}
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

      <DownloadDatasetModal
        open={downloadModalOpen}
        onCancel={() => setDownloadModalOpen(false)}
        onTasksChanged={() => refresh()}
        onSuccess={() => {
          setDownloadModalOpen(false)
          refresh()
        }}
      />

      <UploadDatasetModal
        open={uploadModalOpen}
        onCancel={() => setUploadModalOpen(false)}
        onSuccess={() => {
          setUploadModalOpen(false)
          refresh()
        }}
      />
    </div>
  )
}
