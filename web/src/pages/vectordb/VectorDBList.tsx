import { useState, useEffect, useCallback } from 'react'
import {
  Table,
  Button,
  Space,
  Popconfirm,
  message,
  Typography,
  Row,
  Col,
  Tag,
  Tooltip,
} from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  DeleteOutlined,
  SearchOutlined,
  EyeOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  CloudUploadOutlined,
  CloudDownloadOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import { milvusApi } from '@/services/api'
import type { MilvusCollectionSummary } from '@/types'
import { StatCard } from '@/components/StatCard'
import { STATUS_SUCCESS, STATUS_ERROR, STATUS_INFO, STATUS_DEFAULT } from '@/theme'
import CreateCollectionModal from './CreateCollectionModal'
import BrowseEntitiesModal from './BrowseEntitiesModal'
import SearchModal from './SearchModal'

const { Title } = Typography

export default function VectorDBList() {
  const { t } = useTranslation(['vectordb', 'common'])
  const [loading, setLoading] = useState(false)
  const [collections, setCollections] = useState<MilvusCollectionSummary[]>([])
  const [createOpen, setCreateOpen] = useState(false)
  const [browseTarget, setBrowseTarget] = useState<string>('')
  const [searchTarget, setSearchTarget] = useState<string>('')
  const [searchHybridEnabled, setSearchHybridEnabled] = useState(false)
  const [searchMetadataEnabled, setSearchMetadataEnabled] = useState(false)
  const [searchLinkedDatasets, setSearchLinkedDatasets] = useState<MilvusCollectionSummary['linked_datasets']>([])
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({})

  const fetchCollections = useCallback(async () => {
    setLoading(true)
    try {
      const res = await milvusApi.listCollections()
      setCollections(res.collections || [])
    } catch (err: unknown) {
      message.error(t('list.fetchFailed', { error: err instanceof Error ? err.message : String(err) }))
    } finally {
      setLoading(false)
    }
  }, [t])

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  const handleDelete = useCallback(async (name: string) => {
    try {
      await milvusApi.deleteCollection(name)
      message.success(t('list.collectionDeleted', { name }))
      fetchCollections()
    } catch (err: unknown) {
      message.error(t('list.deleteFailed', { error: err instanceof Error ? err.message : String(err) }))
    }
  }, [fetchCollections, t])

  const handleToggleLoad = useCallback(async (name: string, currentState: string) => {
    setActionLoading((prev) => ({ ...prev, [name]: true }))
    try {
      if (currentState === 'Loaded') {
        await milvusApi.releaseCollection(name)
        message.success(t('list.collectionReleased', { name }))
      } else {
        await milvusApi.loadCollection(name)
        message.success(t('list.collectionLoaded', { name }))
      }
      fetchCollections()
    } catch (err: unknown) {
      message.error(t('list.operationFailed', { error: err instanceof Error ? err.message : String(err) }))
    } finally {
      setActionLoading((prev) => ({ ...prev, [name]: false }))
    }
  }, [fetchCollections, t])

  // Stats
  const totalEntities = collections.reduce((sum, c) => sum + c.num_entities, 0)
  const loadedCount = collections.filter((c) => c.load_state === 'Loaded').length
  const unloadedCount = collections.length - loadedCount

  const columns: ColumnsType<MilvusCollectionSummary> = [
    {
      title: t('list.columns.collectionName'),
      dataIndex: 'name',
      key: 'name',
      width: 220,
      ellipsis: true,
      render: (name: string) => (
        <Tooltip title={name}>
          <span style={{ fontWeight: 500 }}>{name}</span>
        </Tooltip>
      ),
    },
    {
      title: t('list.columns.entityCount'),
      dataIndex: 'num_entities',
      key: 'num_entities',
      width: 100,
      sorter: (a, b) => a.num_entities - b.num_entities,
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: t('list.columns.vectorDimension'),
      dataIndex: 'dim',
      key: 'dim',
      width: 90,
      render: (v: number | null) => v ?? '-',
    },
    {
      title: t('list.columns.indexMetric'),
      key: 'index',
      width: 160,
      render: (_: unknown, record: MilvusCollectionSummary) => (
        <Space size={4}>
          {record.index_type && <Tag color="blue">{record.index_type}</Tag>}
          {record.metric_type && <Tag color="cyan">{record.metric_type}</Tag>}
          {record.hybrid_enabled && <Tag color="orange">{t('list.hybridTag')}</Tag>}
        </Space>
      ),
    },
    {
      title: t('list.columns.status'),
      dataIndex: 'load_state',
      key: 'load_state',
      width: 90,
      filters: [
        { text: t('list.filter.loaded'), value: 'Loaded' },
        { text: t('list.filter.notLoaded'), value: 'NotLoad' },
      ],
      onFilter: (value, record) => record.load_state === value,
      render: (state: string) => {
        if (state === 'Loaded') return <Tag color="success">{t('list.state.loaded')}</Tag>
        if (state === 'Loading') return <Tag color="processing">{t('list.state.loading')}</Tag>
        return <Tag color="default">{t('list.state.notLoaded')}</Tag>
      },
    },
    {
      title: t('list.columns.linkedDatasets'),
      key: 'linked_datasets',
      width: 160,
      ellipsis: true,
      render: (_: unknown, record: MilvusCollectionSummary) => {
        const linked = record.linked_datasets || []
        if (linked.length === 0) return '-'
        if (linked.length === 1) {
          const ds = linked[0]
          const label = ds.dataset_name || ds.dataset_id.slice(0, 8)
          return <Tooltip title={ds.dataset_name || ds.dataset_id}><Tag>{label}</Tag></Tooltip>
        }
        const detail = linked.map(d => d.dataset_name || d.dataset_id.slice(0, 8)).join(', ')
        return (
          <Tooltip title={detail}>
            <Tag color="blue">{t('list.datasetsCount', { count: linked.length })}</Tag>
          </Tooltip>
        )
      },
    },
    {
      title: t('list.columns.embeddingModel'),
      key: 'embedding_model',
      width: 180,
      ellipsis: true,
      render: (_: unknown, record: MilvusCollectionSummary) => {
        if (!record.embedding_model) return '-'
        return (
          <Tooltip title={`${record.embedding_model}\n${record.embedding_endpoint || ''}`}>
            <Tag color="purple">{record.embedding_model}</Tag>
          </Tooltip>
        )
      },
    },
    {
      title: t('list.columns.actions'),
      key: 'actions',
      width: 200,
      render: (_: unknown, record: MilvusCollectionSummary) => (
        <Space size={4}>
          <Tooltip title={t('list.browseData')}>
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => setBrowseTarget(record.name)}
            />
          </Tooltip>
          <Tooltip title={t('list.semanticSearch')}>
            <Button
              type="text"
              size="small"
              icon={<SearchOutlined />}
              onClick={() => { setSearchTarget(record.name); setSearchHybridEnabled(record.hybrid_enabled); setSearchMetadataEnabled(record.metadata_enabled); setSearchLinkedDatasets(record.linked_datasets || []) }}
            />
          </Tooltip>
          <Tooltip title={record.load_state === 'Loaded' ? t('list.release') : t('list.load')}>
            <Button
              type="text"
              size="small"
              loading={actionLoading[record.name]}
              icon={record.load_state === 'Loaded' ? <CloudDownloadOutlined /> : <CloudUploadOutlined />}
              onClick={() => handleToggleLoad(record.name, record.load_state)}
            />
          </Tooltip>
          <Popconfirm
            title={t('list.confirmDeleteTitle', { name: record.name })}
            description={t('list.confirmDeleteDescription')}
            onConfirm={() => handleDelete(record.name)}
            okText={t('common:action.delete')}
            okButtonProps={{ danger: true }}
          >
            <Tooltip title={t('common:action.delete')}>
              <Button type="text" size="small" danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={6}>
          <StatCard
            title={t('list.stats.totalCollections')}
            value={collections.length}
            icon={<CloudServerOutlined />}
            color={STATUS_INFO}
            loading={loading}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('list.stats.totalEntities')}
            value={totalEntities.toLocaleString()}
            icon={<DatabaseOutlined />}
            color={STATUS_INFO}
            loading={loading}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('list.stats.loaded')}
            value={loadedCount}
            icon={<CheckCircleOutlined />}
            color={STATUS_SUCCESS}
            loading={loading}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('list.stats.unloaded')}
            value={unloadedCount}
            icon={<CloseCircleOutlined />}
            color={unloadedCount > 0 ? STATUS_ERROR : STATUS_DEFAULT}
            loading={loading}
          />
        </Col>
      </Row>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>{t('list.title')}</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchCollections} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
            {t('list.createCollection')}
          </Button>
        </Space>
      </div>

      <Table
        dataSource={collections}
        columns={columns}
        rowKey="name"
        loading={loading}
        size="middle"
        pagination={{
          pageSize: 10,
          showTotal: (total) => t('common:pagination.total', { total }),
          showSizeChanger: true,
          pageSizeOptions: ['10', '20', '50'],
        }}
      />

      <CreateCollectionModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onSuccess={fetchCollections}
      />
      <BrowseEntitiesModal
        open={!!browseTarget}
        collectionName={browseTarget}
        onClose={() => setBrowseTarget('')}
      />
      <SearchModal
        open={!!searchTarget}
        collectionName={searchTarget}
        hybridEnabled={searchHybridEnabled}
        metadataEnabled={searchMetadataEnabled}
        linkedDatasets={searchLinkedDatasets}
        onClose={() => setSearchTarget('')}
      />
    </div>
  )
}
