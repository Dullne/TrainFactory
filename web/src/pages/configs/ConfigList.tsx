import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useListWorkspace, useListScroll } from '@/hooks/useListWorkspace'
import { App, Button, Space, Tag, message, Typography, Collapse, Empty, Row, Col } from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  CloudServerOutlined,
  CopyOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  GlobalOutlined,
} from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { configApi } from '@/services/api'
import type { ModelConfig, CreateModelConfigRequest } from '@/types'
import { copyToClipboard } from '@/utils'
import { ConfigCard } from './ConfigCard'
import { ConfigModal } from './ConfigModal'
import { ApiTestModal } from './ApiTestModal'
import './ConfigList.css'
import { StatCard } from '@/components/StatCard'
import {
  TEXT_PRIMARY,
  TEXT_SECONDARY,
  TEXT_TERTIARY,
  BG_ELEVATED,
  BORDER_PRIMARY,
  STATUS_SUCCESS,
  STATUS_INFO,
  STATUS_WARNING,
  STATUS_DEFAULT,
} from '@/theme'

const { Title, Text } = Typography

interface ContainerGroup {
  container_name: string
  endpoint: string
  port?: number
  inference_framework?: string
  configs: ModelConfig[]
}

const formatModelValue = (value: unknown) => {
  if (value === null || value === undefined || value === '') return '-'
  return String(value)
}

const formatCreated = (value: unknown) => {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'number') {
    const ts = value > 1e12 ? value : value * 1000
    const date = new Date(ts)
    return Number.isNaN(date.getTime())
      ? String(value)
      : date.toISOString().replace('T', ' ').slice(0, 19)
  }
  return String(value)
}

const formatCapabilities = (value: unknown) => {
  if (!value) return '-'
  if (Array.isArray(value)) return value.join(', ')
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => Boolean(v))
      .map(([k]) => k)
    return entries.length ? entries.join(', ') : '-'
  }
  return String(value)
}

const groupByContainer = (items: ModelConfig[]): ContainerGroup[] => {
  const map = new Map<string, ContainerGroup>()
  items.forEach((config) => {
    // Use container_name if available, otherwise fall back to 'xinference' for xinference framework
    const containerName =
      config.container_name ||
      (config.inference_framework === 'xinference' ? 'xinference' : config.api_endpoint)
    const key = containerName
    if (!map.has(key)) {
      // Extract port from endpoint
      const portMatch = config.api_endpoint?.match(/:(\d+)/)
      const port = portMatch ? parseInt(portMatch[1], 10) : undefined
      map.set(key, {
        container_name: containerName,
        endpoint: config.api_endpoint,
        port,
        inference_framework: config.inference_framework,
        configs: [],
      })
    }
    map.get(key)!.configs.push(config)
  })
  return Array.from(map.values())
}

const WORKSPACE_FILTERS = { groups: ['all', 'internal', 'external', 'none'] }

export default function ConfigList() {
  const { modal } = App.useApp()
  const { t } = useTranslation(['configs', 'common'])
  const [loading, setLoading] = useState(false)
  const [configs, setConfigs] = useState<ModelConfig[]>([])
  const workspace = useListWorkspace({ filters: WORKSPACE_FILTERS })
  const visibleGroups = workspace.values.groups || 'all'
  useListScroll(configs.length > 0)
  const [modalVisible, setModalVisible] = useState(false)
  const [editingConfig, setEditingConfig] = useState<ModelConfig | null>(null)
  const [saving, setSaving] = useState(false)
  const [checking, setChecking] = useState<string | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<Record<string, boolean | null>>({})
  const [apiTestVisible, setApiTestVisible] = useState(false)
  const [testingConfig, setTestingConfig] = useState<ModelConfig | null>(null)
  const [checkingAll, setCheckingAll] = useState(false)
  const fetchRequestSeq = useRef(0)

  // 按来源类型分组（内部部署 / 外部API），每组内再按容器聚合
  const groupedBySource = useMemo(() => {
    const internal: ModelConfig[] = []
    const external: ModelConfig[] = []
    configs.forEach((config) => {
      if (config.source_type === 'local_deployed') {
        internal.push(config)
      } else {
        external.push(config)
      }
    })
    return {
      internal: groupByContainer(internal),
      external: groupByContainer(external),
      internalCount: internal.length,
      externalCount: external.length,
    }
  }, [configs])

  const fetchConfigs = useCallback(async () => {
    const requestId = ++fetchRequestSeq.current
    setLoading(true)
    try {
      const res = await configApi.list({ page_size: 100 })
      // Only let the latest request update state to avoid stale overwrite.
      if (requestId === fetchRequestSeq.current) {
        setConfigs(res.items)
      }
    } catch {
      if (requestId === fetchRequestSeq.current) {
        message.error(t('list.fetchFailed'))
      }
    } finally {
      if (requestId === fetchRequestSeq.current) {
        setLoading(false)
      }
    }
  }, [t])

  useEffect(() => {
    fetchConfigs()
  }, [fetchConfigs])

  const handleDelete = async (configId: string) => {
    try {
      await configApi.delete(configId)
      message.success(t('common:message.deleteSuccess'))
      fetchConfigs()
    } catch {
      message.error(t('common:message.deleteFailed'))
    }
  }

  const handleCheckConnection = async (configId: string, options?: { refreshAfter?: boolean }) => {
    const refreshAfter = options?.refreshAfter ?? true
    setChecking(configId)
    try {
      const result = await configApi.checkConnection(configId)
      setConnectionStatus((prev) => ({ ...prev, [configId]: result.success }))
      if (result.success) {
        const latency = result.latency_ms !== undefined ? `${result.latency_ms.toFixed(0)}ms` : '-'
        const models = result.models ?? []
        const modelDetails = result.model_details ?? []
        modal.success({
          title: <span style={{ color: STATUS_SUCCESS }}>{t('list.connection.successTitle')}</span>,
          width: 480,
          content: (
            <div style={{ marginTop: 12 }}>
              <div style={{ marginBottom: 12, color: TEXT_TERTIARY }}>
                {t('list.connection.latency', { latency })}
              </div>
              {result.warning && (
                <div
                  style={{
                    marginBottom: 12,
                    padding: '8px 12px',
                    background: 'rgba(250, 173, 20, 0.1)',
                    border: '1px solid rgba(250, 173, 20, 0.3)',
                    borderRadius: 6,
                    color: '#faad14',
                    fontSize: 13,
                  }}
                >
                  {result.warning}
                </div>
              )}
              <div style={{ fontWeight: 500, marginBottom: 8, color: TEXT_PRIMARY }}>
                {t('list.connection.availableModels', {
                  count: modelDetails.length || models.length,
                })}
              </div>
              {modelDetails.length > 0 ? (
                <div
                  style={{
                    background: BG_ELEVATED,
                    border: `1px solid ${BORDER_PRIMARY}`,
                    borderRadius: 6,
                    padding: '8px 12px',
                    maxHeight: 280,
                    overflow: 'auto',
                  }}
                >
                  {modelDetails.map((m, i) => (
                    <div
                      key={m.id || i}
                      style={{
                        padding: '8px 0',
                        borderBottom:
                          i < modelDetails.length - 1 ? `1px solid ${BORDER_PRIMARY}` : undefined,
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <code style={{ fontSize: 13, color: TEXT_PRIMARY }}>
                            {formatModelValue(m.id)}
                          </code>
                          {m.parent && (
                            <Tag
                              color="blue"
                              style={{ fontSize: 11, lineHeight: '18px', margin: 0 }}
                            >
                              LoRA
                            </Tag>
                          )}
                        </span>
                        <Button
                          type="text"
                          size="small"
                          icon={<CopyOutlined />}
                          onClick={() => {
                            if (m.id) {
                              copyToClipboard(m.id)
                              message.success(t('common:message.copied'))
                            }
                          }}
                        />
                      </div>
                      <div
                        style={{
                          marginTop: 6,
                          display: 'grid',
                          gridTemplateColumns: '90px 1fr',
                          rowGap: 4,
                          columnGap: 8,
                          fontSize: 12,
                        }}
                      >
                        <div style={{ color: TEXT_TERTIARY }}>{t('list.connection.modelType')}</div>
                        <div>{formatModelValue(m.model_type)}</div>
                        <div style={{ color: TEXT_TERTIARY }}>{t('list.connection.context')}</div>
                        <div>{formatModelValue(m.context_length)}</div>
                        <div style={{ color: TEXT_TERTIARY }}>max_tokens</div>
                        <div>{formatModelValue(m.max_tokens)}</div>
                        <div style={{ color: TEXT_TERTIARY }}>max_completion</div>
                        <div>{formatModelValue(m.max_completion_tokens)}</div>
                        <div style={{ color: TEXT_TERTIARY }}>embedding_dim</div>
                        <div>{formatModelValue(m.embedding_dim)}</div>
                        <div style={{ color: TEXT_TERTIARY }}>owned_by</div>
                        <div>{formatModelValue(m.owned_by)}</div>
                        <div style={{ color: TEXT_TERTIARY }}>created</div>
                        <div>{formatCreated(m.created)}</div>
                        <div style={{ color: TEXT_TERTIARY }}>{t('list.modelCapabilities')}</div>
                        <div>{formatCapabilities(m.capabilities)}</div>
                        {m.parent && (
                          <>
                            <div style={{ color: TEXT_TERTIARY }}>{t('list.modelParent')}</div>
                            <div>
                              <Tag color="blue" style={{ fontSize: 11 }}>
                                LoRA
                              </Tag>{' '}
                              {m.parent}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : models.length > 0 ? (
                <div
                  style={{
                    background: BG_ELEVATED,
                    border: `1px solid ${BORDER_PRIMARY}`,
                    borderRadius: 6,
                    padding: '8px 12px',
                    maxHeight: 240,
                    overflow: 'auto',
                  }}
                >
                  {models.map((m, i) => (
                    <div
                      key={i}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '6px 0',
                        borderBottom:
                          i < models.length - 1 ? `1px solid ${BORDER_PRIMARY}` : undefined,
                      }}
                    >
                      <code style={{ fontSize: 13, color: TEXT_PRIMARY }}>{m}</code>
                      <Button
                        type="text"
                        size="small"
                        icon={<CopyOutlined />}
                        onClick={() => {
                          copyToClipboard(m)
                          message.success(t('common:message.copied'))
                        }}
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: TEXT_TERTIARY }}>{t('list.connection.noModels')}</div>
              )}
            </div>
          ),
        })
      } else {
        message.error(
          result.error
            ? t('list.connection.failedWithError', { error: result.error })
            : t('list.connection.failed')
        )
      }
    } catch {
      setConnectionStatus((prev) => ({ ...prev, [configId]: false }))
      message.error(t('list.connection.checkFailed'))
    } finally {
      setChecking(null)
      if (refreshAfter) {
        await fetchConfigs()
      }
    }
  }

  const handleCheckAll = async () => {
    setCheckingAll(true)
    try {
      const resp = await configApi.checkAll()
      const statusMap: Record<string, boolean | null> = {}
      let successCount = 0
      let failCount = 0
      for (const [configId, result] of Object.entries(resp.results)) {
        statusMap[configId] = result.success
        if (result.success) successCount++
        else failCount++
      }
      setConnectionStatus((prev) => ({ ...prev, ...statusMap }))
      if (failCount === 0) {
        message.success(t('list.connection.allSuccess', { count: successCount }))
      } else {
        message.warning(
          t('list.connection.partialSuccess', { success: successCount, fail: failCount })
        )
      }
      await fetchConfigs()
    } catch {
      message.error(t('list.connection.checkFailed'))
    } finally {
      setCheckingAll(false)
    }
  }

  const handleEdit = (config: ModelConfig) => {
    setEditingConfig(config)
    setModalVisible(true)
  }

  const handleApiTest = (config: ModelConfig) => {
    setTestingConfig(config)
    setApiTestVisible(true)
  }

  const handleCreate = () => {
    setEditingConfig(null)
    setModalVisible(true)
  }

  const handleSave = async (values: CreateModelConfigRequest) => {
    setSaving(true)
    try {
      if (editingConfig) {
        await configApi.update(editingConfig.config_id, values)
        message.success(t('list.updateSuccess'))
      } else {
        await configApi.create(values)
        message.success(t('list.createSuccess'))
      }
      setModalVisible(false)
      fetchConfigs()
    } catch (error) {
      message.error(error instanceof Error ? error.message : t('list.saveFailed'))
    } finally {
      setSaving(false)
    }
  }

  const renderConfigCard = (config: ModelConfig, grouped?: boolean) => (
    <ConfigCard
      key={config.config_id}
      config={config}
      connectionStatus={connectionStatus[config.config_id] ?? null}
      checking={checking === config.config_id}
      grouped={grouped}
      onEdit={handleEdit}
      onDelete={handleDelete}
      onCheckConnection={handleCheckConnection}
      onApiTest={handleApiTest}
    />
  )

  const renderContainerGroups = (groups: ContainerGroup[]) => (
    <>
      {groups.map((group) => {
        // 单个配置的容器：直接显示
        if (group.configs.length === 1) {
          return renderConfigCard(group.configs[0])
        }
        // 多个配置的容器：折叠显示
        const framework = group.inference_framework
        return (
          <Collapse
            key={group.container_name}
            size="small"
            defaultActiveKey={[group.container_name]}
            style={{ marginBottom: 12, borderRadius: 8 }}
            items={[
              {
                key: group.container_name,
                label: (
                  <div className="config-group-heading">
                    <CloudServerOutlined style={{ color: TEXT_SECONDARY }} />
                    <code style={{ fontSize: 12, color: TEXT_PRIMARY }}>
                      {group.container_name}
                    </code>
                    {framework && (
                      <Tag
                        color={
                          framework === 'vllm'
                            ? 'orange'
                            : framework === 'sglang'
                              ? 'purple'
                              : 'cyan'
                        }
                        style={{ fontSize: 11 }}
                      >
                        {framework.toUpperCase()}
                      </Tag>
                    )}
                    <Tag>{t('list.configCount', { count: group.configs.length })}</Tag>
                    <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>{group.endpoint}</Text>
                    <Button
                      type="text"
                      size="small"
                      icon={<ApiOutlined />}
                      loading={group.configs.some((c) => checking === c.config_id)}
                      onClick={async (e) => {
                        e.stopPropagation()
                        for (const c of group.configs) {
                          await handleCheckConnection(c.config_id, { refreshAfter: false })
                        }
                        await fetchConfigs()
                      }}
                      style={{ marginLeft: 'auto', color: TEXT_SECONDARY }}
                    >
                      {t('list.testConnection')}
                    </Button>
                  </div>
                ),
                children: group.configs.map((c) => renderConfigCard(c, true)),
              },
            ]}
          />
        )
      })}
    </>
  )

  const collapseItems = [
    ...(groupedBySource.internalCount > 0
      ? [
          {
            key: 'internal',
            label: (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: 6,
                    backgroundColor: '#1890ff20',
                    color: '#1890ff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 600,
                    fontSize: 12,
                  }}
                >
                  {t('list.source.internalLabel')}
                </div>
                <span style={{ fontWeight: 500 }}>{t('list.source.internalTitle')}</span>
                <Tag style={{ marginLeft: 8 }}>{groupedBySource.internalCount}</Tag>
              </div>
            ),
            children: renderContainerGroups(groupedBySource.internal),
          },
        ]
      : []),
    ...(groupedBySource.externalCount > 0
      ? [
          {
            key: 'external',
            label: (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: 6,
                    backgroundColor: '#52c41a20',
                    color: '#52c41a',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 600,
                    fontSize: 12,
                  }}
                >
                  {t('list.source.externalLabel')}
                </div>
                <span style={{ fontWeight: 500 }}>{t('list.source.externalTitle')}</span>
                <Tag style={{ marginLeft: 8 }}>{groupedBySource.externalCount}</Tag>
              </div>
            ),
            children: renderContainerGroups(groupedBySource.external),
          },
        ]
      : []),
  ]

  // 统计连接正常的数量（使用数据库记录的检查状态）
  const connectedCount = useMemo(() => {
    return configs.filter((c) => c.last_check_status === 'healthy').length
  }, [configs])

  return (
    <div>
      {/* 统计面板 */}
      <Row className="page-stats" gutter={[12, 12]} style={{ marginBottom: 20 }}>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stat.totalConfigs')}
            value={configs.length}
            icon={<ApiOutlined />}
            color={STATUS_INFO}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stat.internal')}
            value={groupedBySource.internalCount}
            icon={<CloudServerOutlined />}
            color={STATUS_SUCCESS}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stat.externalApi')}
            value={groupedBySource.externalCount}
            icon={<GlobalOutlined />}
            color={STATUS_WARNING}
          />
        </Col>
        <Col xs={12} md={6}>
          <StatCard
            title={t('list.stat.connected')}
            value={connectedCount}
            icon={<CheckCircleOutlined />}
            color={STATUS_DEFAULT}
          />
        </Col>
      </Row>

      {/* 工具栏 */}
      <div className="page-toolbar">
        <Title level={4} style={{ margin: 0 }}>
          {t('list.title')}
        </Title>
        <Space className="page-toolbar-actions" wrap>
          <Button
            icon={<ApiOutlined />}
            onClick={handleCheckAll}
            loading={checkingAll}
            disabled={configs.length === 0}
          >
            {t('list.testAll')}
          </Button>
          <Button icon={<ReloadOutlined />} onClick={fetchConfigs} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
            {t('list.addConfig')}
          </Button>
        </Space>
      </div>

      {/* 分组列表 */}
      {configs.length > 0 ? (
        <Collapse
          activeKey={
            visibleGroups === 'all'
              ? ['internal', 'external']
              : visibleGroups === 'none'
                ? []
                : [visibleGroups]
          }
          onChange={(keys) =>
            workspace.update({
              groups: keys.length === 2 ? 'all' : keys.length === 0 ? 'none' : keys[0],
            })
          }
          items={collapseItems}
          style={{ background: 'transparent', border: 0 }}
          expandIconPosition="start"
        />
      ) : (
        <Empty
          image={<CloudServerOutlined style={{ fontSize: 48, color: TEXT_SECONDARY }} />}
          description={<span style={{ color: TEXT_SECONDARY }}>{t('list.emptyDescription')}</span>}
        />
      )}

      {/* 编辑/创建对话框 */}
      <ConfigModal
        visible={modalVisible}
        editingConfig={editingConfig}
        saving={saving}
        onCancel={() => setModalVisible(false)}
        onSave={handleSave}
      />

      {/* API 测试对话框 */}
      <ApiTestModal
        visible={apiTestVisible}
        config={testingConfig}
        onClose={() => setApiTestVisible(false)}
        onSuccess={fetchConfigs}
      />
    </div>
  )
}
