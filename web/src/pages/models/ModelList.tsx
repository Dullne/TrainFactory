import { useState, useEffect, useMemo, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Table,
  Alert,
  Button,
  Space,
  Tag,
  Popconfirm,
  message,
  Select,
  Typography,
  Tooltip,
  Row,
  Col,
  Segmented,
  Empty,
  Pagination,
  Modal,
  Descriptions,
} from 'antd'
import {
  ReloadOutlined,
  DeleteOutlined,
  RocketOutlined,
  AppstoreOutlined,
  UnorderedListOutlined,
  NodeIndexOutlined,
  RobotOutlined,
  FolderOpenOutlined,
  CloudDownloadOutlined,
  CopyOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { modelApi } from '@/services/api'
import { useWorkspaceList, useListWorkspace } from '@/hooks/useListWorkspace'
import { formatDate, copyToClipboard } from '@/utils'
import type { RegisteredModel, ModelStatus, ModelType } from '@/types'
import { StatusTag } from '@/components/StatusTag'
import { StatCard } from '@/components/StatCard'
import { ModelCard, modelTypeIcons, modelTypeI18nKeys } from './ModelCard'
import { RegisterModelModal } from './RegisterModelModal'
import { DownloadModelModal } from './DownloadModelModal'
import { useAuth } from '@/auth/AuthContext'
import {
  MODEL_TYPE_COLORS,
  TEXT_SECONDARY,
  STATUS_SUCCESS,
  STATUS_INFO,
  STATUS_WARNING,
  STATUS_DEFAULT,
} from '@/theme'

const { Title, Text } = Typography

const LIST_FILTERS = {
  model_type: ['embedding', 'reranker', 'decoder_reranker', 'llm'],
  status: ['available', 'registered', 'archived'],
}
const VIEW_FILTERS = { view: ['card', 'table'] }

export default function ModelList() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { t } = useTranslation(['models', 'common'])
  const { config } = useAuth()
  const viewWorkspace = useListWorkspace({ filters: VIEW_FILTERS })
  const viewMode = viewWorkspace.values.view || 'card'
  const setViewMode = (view: string) => viewWorkspace.update({ view })
  const [registerModalOpen, setRegisterModalOpen] = useState(false)
  const [downloadModalOpen, setDownloadModalOpen] = useState(false)
  const [detailModel, setDetailModel] = useState<RegisteredModel | null>(null)
  const [resolvedBaseModel, setResolvedBaseModel] = useState<RegisteredModel | null>(null)
  const [baseModelMap, setBaseModelMap] = useState<Record<string, RegisteredModel>>({})

  const typeFilterOptions = [
    { label: t('common:modelType.embedding'), value: 'embedding' },
    { label: t('common:modelType.reranker'), value: 'reranker' },
    { label: t('common:modelType.decoderReranker'), value: 'decoder_reranker' },
    { label: t('common:modelType.llm'), value: 'llm' },
  ]

  const statusFilterOptions = [
    { label: t('common:status.available'), value: 'available' },
    { label: t('common:status.registered'), value: 'registered' },
    { label: t('common:status.archived'), value: 'archived' },
  ]

  const {
    data: models,
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
    RegisteredModel,
    {
      stats?: {
        total: number
        by_status?: Record<string, number>
        by_type?: Record<string, number>
      }
    }
  >(modelApi.list, { filters: LIST_FILTERS })
  const typeFilter = workspace.values.model_type
  const statusFilter = workspace.values.status
  const setTypeFilter = (value: string | undefined) => workspace.setFilter('model_type', value)
  const setStatusFilter = (value: string | undefined) => workspace.setFilter('status', value)

  // 统计数据：优先用后端全局 stats（不受分页影响），无则回退当前页
  const stats = useMemo(() => {
    const s = extra?.stats
    if (s) {
      const byStatus = s.by_status ?? {}
      return {
        total: s.total ?? total,
        available: byStatus.available ?? 0,
        registered: byStatus.registered ?? 0,
        archived: byStatus.archived ?? 0,
      }
    }
    const available = models.filter((m) => m.status === 'available').length
    const registered = models.filter((m) => m.status === 'registered').length
    const archived = models.filter((m) => m.status === 'archived').length
    return { total: models.length, available, registered, archived }
  }, [models, total, extra])

  // Handle ?detail=model_id URL param
  /* eslint-disable react-hooks/exhaustive-deps */
  useEffect(() => {
    const detailId = searchParams.get('detail')
    if (detailId) {
      modelApi
        .get(detailId)
        .then((m) => setDetailModel(m))
        .catch(() => {})
      searchParams.delete('detail')
      setSearchParams(searchParams, { replace: true })
    }
  }, [])
  /* eslint-enable react-hooks/exhaustive-deps */

  // Build base model lookup map: model_path → model
  const pathSuffix = useCallback((p: string) => p.split('/').filter(Boolean).pop() || p, [])
  const fetchBaseModelMap = useCallback(() => {
    modelApi
      .list({ page_size: 500 })
      .then(({ items }) => {
        const map: Record<string, RegisteredModel> = {}
        for (const m of items) {
          if (m.model_path) {
            map[m.model_path] = m
            // Also index by path suffix for fuzzy matching
            const suffix = pathSuffix(m.model_path)
            if (suffix && !map[suffix]) map[suffix] = m
          }
        }
        setBaseModelMap(map)
      })
      .catch(() => {})
  }, [pathSuffix])

  useEffect(() => {
    fetchBaseModelMap()
  }, [fetchBaseModelMap])

  // Resolve base model for detail modal from the shared map
  useEffect(() => {
    if (!detailModel?.base_model_path) {
      setResolvedBaseModel(null)
      return
    }
    const target = detailModel.base_model_path
    const match = baseModelMap[target] || baseModelMap[pathSuffix(target)] || null
    setResolvedBaseModel(match)
  }, [detailModel, baseModelMap, pathSuffix])

  const handleDetail = (model: RegisteredModel) => setDetailModel(model)

  const handleDelete = async (modelId: string) => {
    try {
      await modelApi.delete(modelId)
      message.success(t('common:message.deleteSuccess'))
      refresh()
      fetchBaseModelMap()
    } catch {
      message.error(t('common:message.deleteFailed'))
    }
  }

  const getModelTypeColor = (type: string) => {
    return MODEL_TYPE_COLORS[type] || '#8b949e'
  }

  const getSourceTypeMeta = (sourceType?: string) => {
    switch (sourceType) {
      case 'trained':
        return { label: t('source.trained'), color: 'blue' }
      case 'downloaded':
        return { label: t('source.downloaded'), color: 'green' }
      case 'uploaded':
        return { label: t('source.uploaded'), color: 'geekblue' }
      case 'external_bind':
        return { label: t('source.externalBind'), color: 'orange' }
      default:
        return { label: sourceType || t('source.unknown'), color: 'default' }
    }
  }

  const columns: ColumnsType<RegisteredModel> = [
    {
      title: t('table.modelName'),
      dataIndex: 'model_name',
      key: 'name',
      ellipsis: true,
      render: (name, record) => {
        const displayName = name || record.display_name || record.model_id
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
            <div
              style={{
                flexShrink: 0,
                width: 32,
                height: 32,
                borderRadius: 6,
                backgroundColor: `${getModelTypeColor(record.model_type)}20`,
                color: getModelTypeColor(record.model_type),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 16,
              }}
            >
              {modelTypeIcons[record.model_type] || <AppstoreOutlined />}
            </div>
            <div style={{ minWidth: 0, flex: 1 }}>
              <Tooltip title={displayName}>
                <span
                  style={{
                    fontWeight: 500,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    display: 'block',
                  }}
                >
                  {displayName}
                </span>
              </Tooltip>
              <Tooltip title={record.model_id}>
                <Text
                  copyable={{ text: record.model_id, tooltips: false }}
                  style={{ color: TEXT_SECONDARY, fontSize: 11 }}
                >
                  {record.model_id.slice(0, 8)}
                </Text>
              </Tooltip>
            </div>
            <Tooltip title={t('card.copyModelName')}>
              <CopyOutlined
                style={{ color: TEXT_SECONDARY, cursor: 'pointer', fontSize: 12, flexShrink: 0 }}
                onClick={(e) => {
                  e.stopPropagation()
                  copyToClipboard(displayName)
                  message.success(t('card.copiedModelName'))
                }}
              />
            </Tooltip>
          </div>
        )
      },
    },
    {
      title: t('table.type'),
      dataIndex: 'model_type',
      key: 'model_type',
      width: 140,
      render: (type: ModelType) => (
        <Tag
          style={{
            backgroundColor: `${getModelTypeColor(type)}20`,
            color: getModelTypeColor(type),
            border: `1px solid ${getModelTypeColor(type)}40`,
          }}
        >
          {modelTypeI18nKeys[type] ? t(modelTypeI18nKeys[type]) : type}
        </Tag>
      ),
    },
    {
      title: t('table.source'),
      dataIndex: 'source_type',
      key: 'source_type',
      width: 110,
      render: (sourceType: string | undefined, record) => {
        const meta = getSourceTypeMeta(sourceType)
        const details = record.extra_metadata ? JSON.stringify(record.extra_metadata, null, 2) : ''
        const tag = <Tag color={meta.color}>{meta.label}</Tag>
        return details ? (
          <Tooltip title={<pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{details}</pre>}>
            {tag}
          </Tooltip>
        ) : (
          tag
        )
      },
    },
    {
      title: t('table.status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: ModelStatus) => {
        const descKey = `list.${status}Desc`
        const desc = t(descKey, { defaultValue: '' })
        return desc ? (
          <Tooltip title={desc}>
            <span>
              <StatusTag status={status} />
            </span>
          </Tooltip>
        ) : (
          <StatusTag status={status} />
        )
      },
    },
    {
      title: t('table.baseModel'),
      dataIndex: 'base_model_path',
      key: 'base_model',
      ellipsis: true,
      width: 180,
      render: (basePath: string | undefined) => {
        if (!basePath) return '-'
        const resolved = baseModelMap[basePath] || baseModelMap[pathSuffix(basePath)]
        if (resolved) {
          const displayName = resolved.model_name || resolved.display_name || resolved.model_id
          return (
            <Tooltip title={basePath}>
              <Button
                type="link"
                size="small"
                style={{
                  padding: 0,
                  maxWidth: '100%',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                onClick={() => handleDetail(resolved)}
              >
                {displayName}
              </Button>
            </Tooltip>
          )
        }
        // Fallback: show path basename
        const basename = pathSuffix(basePath)
        return (
          <Tooltip title={basePath}>
            <Text style={{ fontSize: 12, color: TEXT_SECONDARY }}>{basename}</Text>
          </Tooltip>
        )
      },
    },
    {
      title: t('table.metrics'),
      dataIndex: 'metrics',
      key: 'metrics',
      render: (metrics) => {
        if (!metrics || Object.keys(metrics).length === 0) return '-'
        return (
          <Tooltip title={JSON.stringify(metrics, null, 2)}>
            <span style={{ cursor: 'pointer' }}>
              {Object.entries(metrics)
                .slice(0, 2)
                .map(([k, v]) => (
                  <Tag key={k} style={{ fontSize: 11 }}>
                    {k}: {(v as number).toFixed(3)}
                  </Tag>
                ))}
              {Object.keys(metrics).length > 2 && '...'}
            </span>
          </Tooltip>
        )
      },
    },
    {
      title: t('table.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 140,
      render: (time) => formatDate(time),
    },
    {
      title: t('table.actions'),
      key: 'actions',
      width: 160,
      render: (_, record) => (
        <Space>
          {record.status === 'available' && (
            <Button
              type="link"
              size="small"
              icon={<RocketOutlined />}
              onClick={() => navigate(`/deployments?model_id=${record.model_id}`)}
            >
              {t('list.deploy')}
            </Button>
          )}
          <Popconfirm
            title={t('list.confirmDelete')}
            onConfirm={() => handleDelete(record.model_id)}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              {t('common:action.delete')}
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      {/* 统计面板 */}
      <Row className="page-stats" gutter={[12, 12]} style={{ marginBottom: 20 }}>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.totalModels')}
            value={stats.total}
            icon={<AppstoreOutlined />}
            color={STATUS_INFO}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.available')}
            value={stats.available}
            icon={<RobotOutlined />}
            color={STATUS_SUCCESS}
            tooltip={t('list.availableDesc')}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.registered')}
            value={stats.registered}
            icon={<NodeIndexOutlined />}
            color={STATUS_WARNING}
            tooltip={t('list.registeredDesc')}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.archived')}
            value={stats.archived}
            icon={<FolderOpenOutlined />}
            color={STATUS_DEFAULT}
            tooltip={t('list.archivedDesc')}
          />
        </Col>
      </Row>

      {/* 工具栏 */}
      <div className="page-toolbar">
        <Title level={4} style={{ margin: 0 }}>
          {t('list.title')}
        </Title>
        <Space className="page-toolbar-actions" wrap>
          <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          {config.direct_storage_registration_enabled ? (
            <Button icon={<FolderOpenOutlined />} onClick={() => setRegisterModalOpen(true)}>
              {t('list.registerModel')}
            </Button>
          ) : null}
          <Button
            type="primary"
            icon={<CloudDownloadOutlined />}
            onClick={() => setDownloadModalOpen(true)}
          >
            {t('list.downloadModel')}
          </Button>
        </Space>
      </div>

      <div className="model-list-filters">
        <Segmented
          value={viewMode}
          onChange={(v) => setViewMode(v as 'card' | 'table')}
          options={[
            { value: 'card', icon: <AppstoreOutlined /> },
            { value: 'table', icon: <UnorderedListOutlined /> },
          ]}
        />
        <Select
          placeholder={t('list.typeFilter')}
          allowClear
          value={typeFilter}
          onChange={setTypeFilter}
          options={typeFilterOptions}
        />
        <Select
          placeholder={t('list.statusFilter')}
          allowClear
          value={statusFilter}
          onChange={setStatusFilter}
          options={statusFilterOptions}
        />
      </div>

      {/* 内容区域 */}
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
      {viewMode === 'card' ? (
        models.length > 0 ? (
          <>
            <Row gutter={[16, 16]}>
              {models.map((model) => (
                <ModelCard
                  key={model.model_id}
                  model={model}
                  onDelete={handleDelete}
                  onDetail={handleDetail}
                  baseModelMap={baseModelMap}
                />
              ))}
            </Row>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
              <Pagination
                current={page}
                pageSize={pageSize}
                total={total}
                showSizeChanger
                showTotal={(total) => t('common:pagination.total', { total })}
                onChange={(p, ps) => {
                  workspace.setPagination(p, ps)
                }}
              />
            </div>
          </>
        ) : (
          <Empty
            image={<AppstoreOutlined style={{ fontSize: 48, color: TEXT_SECONDARY }} />}
            description={<span style={{ color: TEXT_SECONDARY }}>{t('list.noModels')}</span>}
          />
        )
      ) : (
        <Table
          rowKey="model_id"
          columns={columns}
          dataSource={models}
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

      {/* 注册模型弹窗 */}
      {config.direct_storage_registration_enabled ? (
        <RegisterModelModal
          open={registerModalOpen}
          onCancel={() => setRegisterModalOpen(false)}
          onSuccess={() => {
            setRegisterModalOpen(false)
            refresh()
            fetchBaseModelMap()
          }}
        />
      ) : null}

      {/* 下载模型弹窗 */}
      <DownloadModelModal
        open={downloadModalOpen}
        onCancel={() => setDownloadModalOpen(false)}
        onTasksChanged={() => {
          refresh()
          fetchBaseModelMap()
        }}
        onSuccess={() => {
          setDownloadModalOpen(false)
          refresh()
          fetchBaseModelMap()
        }}
      />

      {/* 模型详情弹窗 */}
      <Modal
        open={!!detailModel}
        title={detailModel?.model_name || ''}
        onCancel={() => setDetailModel(null)}
        footer={null}
        width={640}
      >
        {detailModel && (
          <Descriptions column={2} bordered size="small">
            <Descriptions.Item label="Model ID" span={2}>
              <Typography.Text copyable code style={{ fontSize: 12 }}>
                {detailModel.model_id}
              </Typography.Text>
            </Descriptions.Item>
            <Descriptions.Item label={t('table.type')}>
              <Tag color={MODEL_TYPE_COLORS[detailModel.model_type]}>
                {t(modelTypeI18nKeys[detailModel.model_type] || 'common:modelType.embedding')}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label={t('table.status')}>
              <StatusTag status={detailModel.status} />
            </Descriptions.Item>
            <Descriptions.Item label={t('table.source')}>
              <Tag
                color={
                  detailModel.source_type === 'downloaded'
                    ? 'green'
                    : detailModel.source_type === 'trained'
                      ? 'blue'
                      : 'default'
                }
              >
                {t(`source.${detailModel.source_type}`, { defaultValue: detailModel.source_type })}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Version">{detailModel.version}</Descriptions.Item>
            <Descriptions.Item label={t('table.baseModel')} span={2}>
              {detailModel.base_model_path ? (
                resolvedBaseModel ? (
                  <Button
                    type="link"
                    size="small"
                    style={{ padding: 0, fontSize: 12 }}
                    onClick={() => setDetailModel(resolvedBaseModel)}
                  >
                    {resolvedBaseModel.model_name ||
                      resolvedBaseModel.display_name ||
                      resolvedBaseModel.model_id}
                  </Button>
                ) : (
                  <Typography.Text copyable style={{ fontSize: 12, wordBreak: 'break-all' }}>
                    {detailModel.base_model_path}
                  </Typography.Text>
                )
              ) : (
                '-'
              )}
            </Descriptions.Item>
            <Descriptions.Item label="Model Path" span={2}>
              <Typography.Text copyable style={{ fontSize: 12, wordBreak: 'break-all' }}>
                {detailModel.model_path}
              </Typography.Text>
            </Descriptions.Item>
            {detailModel.file_size && (
              <Descriptions.Item label="File Size">
                {(detailModel.file_size / 1024 / 1024 / 1024).toFixed(2)} GB
              </Descriptions.Item>
            )}
            {detailModel.source_task_id && (
              <Descriptions.Item label="Training Task">
                <Button
                  type="link"
                  size="small"
                  style={{ padding: 0 }}
                  onClick={() => {
                    setDetailModel(null)
                    navigate(`/training/${detailModel.source_task_id}`)
                  }}
                >
                  {detailModel.source_task_id.slice(0, 8)}...
                </Button>
              </Descriptions.Item>
            )}
            <Descriptions.Item label={t('table.createdAt')} span={2}>
              {formatDate(detailModel.created_at)}
            </Descriptions.Item>
            {detailModel.description && (
              <Descriptions.Item label="Description" span={2}>
                {detailModel.description}
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  )
}
