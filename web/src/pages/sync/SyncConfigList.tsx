import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Table, Button, Space, Tag, Progress, Popconfirm, message,
  Typography, Tooltip, Row, Col, Tabs,
} from 'antd'
import {
  PlusOutlined, ReloadOutlined, DeleteOutlined, EyeOutlined,
  PlayCircleOutlined, PauseCircleOutlined, SyncOutlined, ApiOutlined, CopyOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { syncApi, SILENT_REQUEST_CONFIG } from '@/services/api'
import { usePolling } from '@/hooks'
import { formatDate, copyToClipboard } from '@/utils'
import { StatusTag } from '@/components/StatusTag'
import { StatCard } from '@/components/StatCard'
import ExternalApiConfigList from './ExternalApiConfigList'
import type { SyncConfig } from '@/types'
import { ACTIVE_SYNC_STATUSES } from '@/constants/syncStatus'
import {
  STATUS_INFO, STATUS_SUCCESS, STATUS_ERROR,
  TEXT_SECONDARY,
} from '@/theme'

const { Title } = Typography
const PAGE_SIZE = 50

export default function SyncConfigList() {
  const navigate = useNavigate()
  const { t } = useTranslation(['sync', 'common'])

  const [configs, setConfigs] = useState<SyncConfig[]>([])
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const pageSize = PAGE_SIZE
  const configsRef = useRef<SyncConfig[]>([])
  const pageRef = useRef(page)
  const displayedPageRef = useRef(page)
  const requestGenerationRef = useRef(0)
  const loadingOwnerRef = useRef(0)
  const foregroundOwnerRef = useRef<number | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      requestGenerationRef.current += 1
      loadingOwnerRef.current += 1
      foregroundOwnerRef.current = null
    }
  }, [])

  useEffect(() => {
    configsRef.current = configs
  }, [configs])

  useEffect(() => {
    pageRef.current = page
  }, [page])

  const fetchConfigs = useCallback(async (requestedPage?: number, silent = false) => {
    if (!mountedRef.current) return
    if (silent && foregroundOwnerRef.current !== null) return
    const requestGeneration = ++requestGenerationRef.current
    const loadingOwner = silent ? null : ++loadingOwnerRef.current
    if (!silent) foregroundOwnerRef.current = requestGeneration
    const currentPage = requestedPage ?? pageRef.current
    if (loadingOwner !== null) setLoading(true)
    try {
      const res = await syncApi.listTasks(
        {
          limit: pageSize,
          offset: (currentPage - 1) * pageSize,
        },
        silent ? SILENT_REQUEST_CONFIG : undefined
      )
      if (!mountedRef.current || requestGenerationRef.current !== requestGeneration) return

      const nextTotal = res.total || 0
      const lastPage = Math.max(1, Math.ceil(nextTotal / pageSize))
      setTotal(nextTotal)
      if (currentPage > lastPage) {
        configsRef.current = []
        displayedPageRef.current = lastPage
        setConfigs([])
        pageRef.current = lastPage
        if (loadingOwner !== null && loadingOwnerRef.current === loadingOwner) {
          loadingOwnerRef.current += 1
        }
        setPage(lastPage)
        return
      }

      const nextConfigs = res.tasks || []
      configsRef.current = nextConfigs
      displayedPageRef.current = currentPage
      setConfigs(nextConfigs)
    } catch {
      if (!mountedRef.current || requestGenerationRef.current !== requestGeneration) return
      const displayedPage = displayedPageRef.current
      if (currentPage !== displayedPage) {
        pageRef.current = displayedPage
        if (loadingOwner !== null && loadingOwnerRef.current === loadingOwner) {
          loadingOwnerRef.current += 1
        }
        setPage(displayedPage)
      }
      // error handled by interceptor
    } finally {
      if (!silent && foregroundOwnerRef.current === requestGeneration) {
        foregroundOwnerRef.current = null
      }
      if (
        mountedRef.current &&
        loadingOwner !== null &&
        loadingOwnerRef.current === loadingOwner
      ) {
        setLoading(false)
      }
    }
  }, [pageSize])

  useEffect(() => { fetchConfigs(page) }, [fetchConfigs, page])

  const hasActiveConfigs = configs.some(c =>
    c.is_active && ACTIVE_SYNC_STATUSES.includes(c.status)
  )
  const pollConfigs = useCallback(() => fetchConfigs(undefined, true), [fetchConfigs])
  usePolling(pollConfigs, { interval: 5000, enabled: hasActiveConfigs })

  const handleDelete = async (configId: string) => {
    try {
      await syncApi.deleteTask(configId)
      if (!mountedRef.current) return
      message.success(t('common:message.deleteSuccess'))
      const currentPage = pageRef.current
      if (configsRef.current.length === 1 && currentPage > 1) {
        const previousPage = currentPage - 1
        configsRef.current = []
        displayedPageRef.current = previousPage
        setConfigs([])
        pageRef.current = previousPage
        setPage(previousPage)
      } else {
        fetchConfigs()
      }
    } catch {
      if (!mountedRef.current) return
      message.error(t('common:message.deleteFailed'))
    }
  }

  const handleStart = async (configId: string) => {
    try {
      await syncApi.start(configId)
      if (!mountedRef.current) return
      message.success(t('detail.message.startSuccess'))
      fetchConfigs()
    } catch {
      if (!mountedRef.current) return
      message.error(t('common:message.operationFailed', { error: '' }))
    }
  }

  const handleStop = async (configId: string) => {
    try {
      await syncApi.stop(configId)
      if (!mountedRef.current) return
      message.success(t('detail.message.stopSuccess'))
      fetchConfigs()
    } catch {
      if (!mountedRef.current) return
      message.error(t('common:message.operationFailed', { error: '' }))
    }
  }

  // Stats
  const totalConfigs = total
  const activeConfigs = configs.filter(c => c.is_active).length
  const syncingConfigs = configs.filter(c => ACTIVE_SYNC_STATUSES.includes(c.status)).length
  const errorConfigs = configs.filter(c => c.status === 'error').length
  const currentPageSuffix = ` (P${page})`

  const columns: ColumnsType<SyncConfig> = [
    {
      title: t('list.columns.configId'),
      dataIndex: 'task_id',
      key: 'task_id',
      width: 100,
      render: (id: string) => (
        <Tooltip title={id}>
          <span
            style={{ fontSize: 12, fontFamily: 'monospace', color: TEXT_SECONDARY, cursor: 'pointer' }}
            onClick={() => { copyToClipboard(id); message.success(t('common:message.copied')) }}
          >
            {id.slice(0, 8)} <CopyOutlined style={{ fontSize: 11 }} />
          </span>
        </Tooltip>
      ),
    },
    {
      title: t('list.columns.configName'),
      dataIndex: 'task_name',
      key: 'task_name',
      width: 150,
      ellipsis: true,
      render: (name: string, record) => (
        <Button type="link" onClick={() => navigate(`/sync/${record.task_id}`)}>
          {name}
        </Button>
      ),
    },
    {
      title: t('list.columns.status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string, record) => (
        <Space direction="vertical" size={2}>
          <StatusTag status={status} />
          {!record.is_active && (
            <Tag color="default" style={{ fontSize: 11 }}>
              {t('common:status.disabled')}
            </Tag>
          )}
        </Space>
      ),
    },
    {
      title: t('list.columns.dataProgress'),
      key: 'dataProgress',
      width: 180,
      render: (_, record) => {
        const hasSynced = record.total_record_count > 0 || !!record.last_sync_at
        const isSyncing = record.status === 'syncing'
        const percent = isSyncing ? 50 : (hasSynced ? 100 : 0)

        const formatText = isSyncing
          ? t('list.progress.syncing')
          : hasSynced
            ? t('list.progress.syncedCount', { count: record.total_record_count })
            : t('list.progress.notSynced')

        const subtitle = record.status === 'generating'
          ? t('list.progress.generating')
          : record.status === 'training' || record.status === 'loading_adapter'
            ? t('list.progress.training')
            : record.generation_threshold > 0
              ? `${t('list.progress.pendingGeneration')}: ${record.pending_record_count}/${record.generation_threshold}`
              : ''

        return (
          <div>
            <div style={{ fontSize: 12, lineHeight: '18px', whiteSpace: 'nowrap' }}>{formatText}</div>
            <Progress
              percent={percent}
              size="small"
              strokeColor={STATUS_INFO}
              status={isSyncing ? 'active' : undefined}
              showInfo={false}
            />
            {subtitle && (
              <span style={{ fontSize: 11, color: TEXT_SECONDARY }}>
                {subtitle}
              </span>
            )}
          </div>
        )
      },
    },
    {
      title: t('list.columns.trainingProgress'),
      key: 'trainingProgress',
      width: 160,
      render: (_, record) => {
        const isTraining = record.status === 'training' || record.status === 'loading_adapter'
        const hasTrained = record.total_trainings > 0
        const percent = isTraining ? 50 : (hasTrained ? 100 : 0)

        const formatText = isTraining
          ? t('list.progress.training')
          : hasTrained
            ? t('list.progress.trainedCount', { count: record.total_training_samples })
            : t('list.progress.notTrained')

        const subtitle = isTraining
          ? `${t('list.progress.pendingTraining')}: ${record.pending_training_samples}/${record.training_threshold}`
          : hasTrained
            ? t('list.progress.trainingRound', { count: record.total_trainings })
            : ''

        return (
          <div>
            <div style={{ fontSize: 12, lineHeight: '18px', whiteSpace: 'nowrap' }}>{formatText}</div>
            <Progress
              percent={percent}
              size="small"
              strokeColor={STATUS_SUCCESS}
              status={isTraining ? 'active' : undefined}
              showInfo={false}
            />
            {subtitle && (
              <span style={{ fontSize: 11, color: TEXT_SECONDARY }}>
                {subtitle}
              </span>
            )}
          </div>
        )
      },
    },
    {
      title: t('list.columns.adapter'),
      dataIndex: 'current_adapter_name',
      key: 'adapter',
      width: 120,
      ellipsis: true,
      render: (name: string | null) => name
        ? <Tag color="blue">{name}</Tag>
        : <span style={{ color: TEXT_SECONDARY }}>-</span>,
    },
    {
      title: t('list.columns.lastSync'),
      dataIndex: 'last_sync_at',
      key: 'last_sync_at',
      width: 130,
      render: (v: string | null) => v
        ? <span style={{ fontSize: 12, whiteSpace: 'nowrap' }}>{formatDate(v)}</span>
        : <span style={{ color: TEXT_SECONDARY }}>-</span>,
    },
    {
      title: t('list.columns.actions'),
      key: 'actions',
      fixed: 'right',
      width: 120,
      render: (_, record) => (
        <Space size={4}>
          <Tooltip title={t('common:action.detail')}>
            <Button type="text" size="small" icon={<EyeOutlined />}
              onClick={() => navigate(`/sync/${record.task_id}`)} />
          </Tooltip>
          {record.is_active ? (
            <Popconfirm title={t('list.actions.confirmStop')} onConfirm={() => handleStop(record.task_id)}>
              <Tooltip title={t('common:action.stop')}>
                <Button type="text" size="small" icon={<PauseCircleOutlined />} />
              </Tooltip>
            </Popconfirm>
          ) : (
            <Popconfirm title={t('list.actions.confirmStart')} onConfirm={() => handleStart(record.task_id)}>
              <Tooltip title={t('detail.actions.start')}>
                <Button type="text" size="small" icon={<PlayCircleOutlined />} />
              </Tooltip>
            </Popconfirm>
          )}
          <Popconfirm title={t('list.actions.confirmDelete')} onConfirm={() => handleDelete(record.task_id)}>
            <Tooltip title={t('common:action.delete')}>
              <Button type="text" size="small" danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <Tabs
      defaultActiveKey="sync"
      items={[
        {
          key: 'sync',
          label: <span><SyncOutlined /> {t('apiConfig.syncTab')}</span>,
          children: (
            <div>
              <Row gutter={16} style={{ marginBottom: 20 }}>
                <Col span={6}>
                  <StatCard title={t('list.stats.total')} value={totalConfigs} icon={<SyncOutlined />} color={STATUS_INFO} />
                </Col>
                <Col span={6}>
                  <StatCard title={`${t('list.stats.active')}${currentPageSuffix}`} value={activeConfigs} icon={<PlayCircleOutlined />} color={STATUS_SUCCESS} />
                </Col>
                <Col span={6}>
                  <StatCard title={`${t('list.stats.syncing')}${currentPageSuffix}`} value={syncingConfigs} icon={<SyncOutlined spin />} color={STATUS_INFO} />
                </Col>
                <Col span={6}>
                  <StatCard title={`${t('list.stats.error')}${currentPageSuffix}`} value={errorConfigs} icon={<PauseCircleOutlined />} color={STATUS_ERROR} />
                </Col>
              </Row>

              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                <Title level={4} style={{ margin: 0 }}>{t('list.title')}</Title>
                <Space>
                  <Button icon={<ReloadOutlined />} onClick={() => fetchConfigs()} loading={loading}>
                    {t('common:action.refresh')}
                  </Button>
                  <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/sync/create')}>
                    {t('list.actions.createConfig')}
                  </Button>
                </Space>
              </div>

              <Table
                rowKey="task_id"
                columns={columns}
                dataSource={configs}
                loading={loading}
                scroll={{ x: 1060 }}
                pagination={{
                  current: page,
                  pageSize,
                  total,
                  showSizeChanger: false,
                  onChange: setPage,
                }}
              />
            </div>
          ),
        },
        {
          key: 'api',
          label: <span><ApiOutlined /> {t('apiConfig.tab')}</span>,
          children: <ExternalApiConfigList />,
        },
      ]}
    />
  )
}
