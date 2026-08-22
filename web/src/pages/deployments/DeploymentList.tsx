import { useState, useEffect, useMemo, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  Table,
  Button,
  Space,
  Popconfirm,
  message,
  Select,
  Input,
  Typography,
  Row,
  Col,
  Tooltip,
  Tag,
  Badge,
  Modal,
  Descriptions,
  Checkbox,
  Radio,
  Form,
  Switch,
  Alert,
} from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  RocketOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  CopyOutlined,
  CodeOutlined,
  EditOutlined,
  DownOutlined,
  RightOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import { deploymentApi, modelApi, externalApiConfigApi } from '@/services/api'
import { useList, usePolling } from '@/hooks'
import { formatDate, copyToClipboard as copyText } from '@/utils'
import type { Deployment, DeploymentStatus, RegisteredModel, CreateDeploymentRequest, ExternalApiConfig, HealthStatus } from '@/types'
import { StatusTag } from '@/components/StatusTag'
import { StatCard } from '@/components/StatCard'
import { CreateDeploymentModal } from './CreateDeploymentModal'
import { AdapterManager } from './AdapterManager'
import { STATUS_SUCCESS, STATUS_ERROR, STATUS_WARNING, STATUS_INFO } from '@/theme'

const { Title, Text } = Typography

const getRuntimeAccelerators = (deployment: Deployment): string[] => {
  const info = deployment.runtime_info as Record<string, unknown> | undefined
  const accelerators = info?.accelerators
  if (Array.isArray(accelerators)) {
    return accelerators.map((value) => String(value))
  }
  return []
}

const formatGpuLabel = (deployment: Deployment): string => {
  const runtimeAccelerators = getRuntimeAccelerators(deployment)
  if (runtimeAccelerators.length > 0) return `GPU ${runtimeAccelerators.join(',')}`
  if (deployment.gpu_id !== undefined && deployment.gpu_id !== null) return `GPU ${deployment.gpu_id}`
  return '-'
}

const normalizeHealthStatus = (healthStatus?: HealthStatus): 'healthy' | 'unhealthy' | 'unknown' => {
  switch (healthStatus) {
    case 'HEALTHY':
      return 'healthy'
    case 'UNHEALTHY':
      return 'unhealthy'
    default:
      return 'unknown'
  }
}

const canStartDeployment = (status: DeploymentStatus): boolean =>
  status === 'stopped' || status === 'failed'

const canStopDeployment = (status: DeploymentStatus): boolean =>
  status === 'running' || status === 'pending' || status === 'starting' || status === 'restarting'

const canRestartDeployment = (status: DeploymentStatus): boolean =>
  status === 'running'

const canDeleteDeployment = (status: DeploymentStatus): boolean =>
  status === 'stopped' || status === 'failed' || status === 'pending'

// Deployment group (by endpoint/container)
interface DeploymentGroup {
  key: string  // endpoint or container_name
  endpoint: string
  port: number | null
  gpu_id: number | null
  inference_framework: string
  container_name: string | null
  deploy_mode: string
  deployments: Deployment[]
  // Aggregated info
  status: DeploymentStatus
  health_status: HealthStatus
  running_count: number
  total_memory_mb: number | null
  total_memory_percent: number | null
  total_memory_utilization: number | null  // Configured limit (fallback)
}

export default function DeploymentList() {
  const { t } = useTranslation(['deployments', 'common'])
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  // 从模型列表「部署」按钮跳转而来：按 model_id 预过滤
  const modelIdFilter = searchParams.get('model_id') || undefined
  const [statusFilter, setStatusFilter] = useState<string>()
  const [createVisible, setCreateVisible] = useState(false)
  const [models, setModels] = useState<RegisteredModel[]>([])
  const [creating, setCreating] = useState(false)
  const [expandedKeys, setExpandedKeys] = useState<string[]>([])
  const [paramsVisible, setParamsVisible] = useState(false)
  const [paramsDeployment, setParamsDeployment] = useState<Deployment | null>(null)
  const [editConfigVisible, setEditConfigVisible] = useState(false)
  const [editConfigTarget, setEditConfigTarget] = useState<Deployment | null>(null)
  const [editConfigOtherJson, setEditConfigOtherJson] = useState('')
  const [editConfigRuntime, setEditConfigRuntime] = useState<Record<string, unknown> | null>(null)
  const [savingConfig, setSavingConfig] = useState(false)
  const [editConfigForm] = Form.useForm()
  const [editApiConfigs, setEditApiConfigs] = useState<ExternalApiConfig[]>([])
  const [editApiConfigsLoading, setEditApiConfigsLoading] = useState(false)
  const [restartVisible, setRestartVisible] = useState(false)
  const [restartTarget, setRestartTarget] = useState<Deployment | null>(null)
  const [restartMode, setRestartMode] = useState<'auto' | 'model' | 'container'>('auto')
  const [restartResetGpu, setRestartResetGpu] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [starting, setStarting] = useState<string | null>(null)
  const [stopping, setStopping] = useState<string | null>(null)
  const [adapterVisible, setAdapterVisible] = useState(false)
  const [adapterTarget, setAdapterTarget] = useState<Deployment | null>(null)

  const STATUS_FILTER_OPTIONS = [
    { label: t('common:status.pending'), value: 'pending' },
    { label: t('common:status.running'), value: 'running' },
    { label: t('common:status.restarting'), value: 'restarting' },
    { label: t('common:status.stopped'), value: 'stopped' },
    { label: t('common:status.failed'), value: 'failed' },
  ]

  const fetchDeployments = useCallback(
    (params: Parameters<typeof deploymentApi.list>[0]) => deploymentApi.list({ ...(params ?? {}), status: statusFilter, model_id: modelIdFilter }),
    [statusFilter, modelIdFilter]
  )

  const {
    data: deployments,
    loading,
    page,
    pageSize,
    total,
    setPage,
    setPageSize,
    fetch,
    refresh,
  } = useList<Deployment>(
    fetchDeployments,
    { defaultPageSize: 10 }
  )

  // Group deployments by endpoint
  const groupedDeployments = useMemo(() => {
    const groups = new Map<string, DeploymentGroup>()

    deployments.forEach((dep) => {
      // Group by container name, but keep endpoint in key to avoid merging different shared services
      const containerKey = dep.container_name || (dep.deploy_mode === 'shared' ? 'xinference' : dep.deployment_id)
      const endpointKey = dep.xinference_endpoint || '-'
      const key = `${containerKey}::${endpointKey}`

      if (!groups.has(key)) {
        groups.set(key, {
          key,
          endpoint: dep.xinference_endpoint,
          port: dep.port ?? null,
          gpu_id: dep.gpu_id ?? null,
          inference_framework: dep.inference_framework,
          container_name: dep.container_name ?? null,
          deploy_mode: dep.deploy_mode,
          deployments: [],
          status: dep.status,
          health_status: dep.health_status ?? 'UNKNOWN',
          running_count: 0,
          total_memory_mb: null,
          total_memory_percent: null,
          total_memory_utilization: null,
        })
      }

      const group = groups.get(key)!
      group.deployments.push(dep)

      // Update aggregated status (running > pending > stopped > failed)
      if (dep.status === 'running') {
        group.status = 'running'
        group.running_count++
      }

      if (dep.health_status === 'UNHEALTHY') {
        group.health_status = 'UNHEALTHY'
      } else if (dep.health_status === 'HEALTHY' && group.health_status !== 'UNHEALTHY') {
        group.health_status = 'HEALTHY'
      }

      // Aggregate memory usage (only for running deployments)
      if (dep.status === 'running') {
        if (dep.gpu_memory_used_mb) {
          group.total_memory_mb = (group.total_memory_mb || 0) + dep.gpu_memory_used_mb
        }
        if (dep.gpu_memory_used_percent) {
          group.total_memory_percent = (group.total_memory_percent || 0) + dep.gpu_memory_used_percent
        }

        // Aggregate configured memory utilization as fallback (only running)
        if (dep.gpu_memory_utilization) {
          group.total_memory_utilization = (group.total_memory_utilization || 0) + dep.gpu_memory_utilization
        }
      }
    })

    return Array.from(groups.values())
  }, [deployments])

  const stats = useMemo(() => {
    const running = deployments.filter((d) => d.status === 'running' || d.status === 'restarting').length
    const stopped = deployments.filter((d) => d.status === 'stopped').length
    const failed = deployments.filter((d) => d.status === 'failed').length
    const pending = deployments.filter((d) => d.status === 'pending').length
    return { total: deployments.length, running, stopped, failed, pending }
  }, [deployments])

  const fetchModels = async () => {
    try {
      const res = await modelApi.list({ page_size: 500, status: 'available' })
      setModels(res.items)
    } catch {
      // Ignore
    }
  }

  const modelMap = useMemo(() => {
    const map = new Map<string, RegisteredModel>()
    models.forEach((m) => map.set(m.model_id, m))
    return map
  }, [models])

  useEffect(() => {
    fetchModels()
  }, [])

  /* eslint-disable react-hooks/exhaustive-deps */
  useEffect(() => {
    fetch({ page: 1 })
  }, [statusFilter])
  /* eslint-enable react-hooks/exhaustive-deps */

  const hasPending = deployments.some((d) => d.status === 'pending' || d.status === 'starting' || d.status === 'restarting')
  const hasRunning = deployments.some((d) => d.status === 'running' || d.status === 'restarting')

  usePolling(refresh, {
    interval: 5000,
    enabled: hasPending,
  })

  usePolling(refresh, {
    interval: 30000,
    enabled: hasRunning && !hasPending,
  })

  const handleStart = async (deploymentId: string) => {
    setStarting(deploymentId)
    try {
      await deploymentApi.start(deploymentId)
      message.success(t('list.message.startSuccess'))
      refresh()
    } catch {
      message.error(t('list.message.startFailed'))
    } finally {
      setStarting(null)
    }
  }

  const handleStop = async (deploymentId: string) => {
    setStopping(deploymentId)
    try {
      await deploymentApi.stop(deploymentId)
      message.success(t('list.message.stopSuccess'))
      refresh()
    } catch {
      message.error(t('list.message.stopFailed'))
    } finally {
      setStopping(null)
    }
  }

  const openRestartModal = (deployment: Deployment) => {
    setRestartTarget(deployment)
    setRestartMode('auto')
    setRestartResetGpu(false)
    setRestartVisible(true)
  }

  const handleRestart = async () => {
    if (!restartTarget) return
    setRestarting(true)
    try {
      await deploymentApi.restart(restartTarget.deployment_id, { mode: restartMode, reset_gpu: restartResetGpu })
      message.success(t('list.message.restartSuccess'))
      setRestartVisible(false)
      refresh()
    } catch (error) {
      message.error(error instanceof Error ? error.message : t('list.message.restartFailed'))
    } finally {
      setRestarting(false)
    }
  }

  const handleDelete = async (deploymentId: string) => {
    try {
      await deploymentApi.delete(deploymentId)
      message.success(t('list.message.deleteSuccess'))
      refresh()
    } catch {
      message.error(t('list.message.deleteFailed'))
    }
  }

  const handleShowParams = (deployment: Deployment) => {
    setParamsDeployment(deployment)
    setParamsVisible(true)
  }

  const handleCreate = async (values: CreateDeploymentRequest & { _useContainer?: boolean }) => {
    setCreating(true)
    try {
      const { _useContainer, ...data } = values
      if (_useContainer) {
        await deploymentApi.createContainer({ ...data, auto_start: true })
      } else {
        await deploymentApi.create(data)
      }
      message.success(t('common:message.createSuccess'))
      setCreateVisible(false)
      refresh()
    } catch (error) {
      message.error(error instanceof Error ? error.message : t('common:message.createFailed'))
    } finally {
      setCreating(false)
    }
  }

  const handleBindExisting = async (values: {
    endpoint: string
    model_uid: string
    model_name?: string
    model_type: string
    deployment_name?: string
    inference_framework: string
    container_name?: string
    gpu_id?: number
  }) => {
    setCreating(true)
    try {
      await deploymentApi.bindExisting(values)
      message.success(t('list.message.bindSuccess'))
      setCreateVisible(false)
      refresh()
      fetchModels()
    } catch (error) {
      message.error(error instanceof Error ? error.message : t('list.message.bindFailed'))
    } finally {
      setCreating(false)
    }
  }

  const copyToClipboard = (text: string) => {
    copyText(text)
    message.success(t('list.message.copiedToClipboard'))
  }

  // Known config keys managed by form fields (excluded from "other config" JSON)
  const KNOWN_CONFIG_KEYS = ['external_api_config_id', 'dtype', 'enforce_eager', 'attention_backend', 'docker_cmd']

  const openEditConfigModal = (deployment: Deployment) => {
    setEditConfigTarget(deployment)
    const config = deployment.config ?? {}
    // Split config into known form fields vs other JSON
    editConfigForm.setFieldsValue({
      external_api_config_id: (config.external_api_config_id as string) || undefined,
      dtype: (config.dtype as string) || undefined,
      enforce_eager: config.enforce_eager === true,
      attention_backend: (config.attention_backend as string) || undefined,
    })
    // Collect remaining keys into "other" JSON
    const otherConfig: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(config)) {
      if (!KNOWN_CONFIG_KEYS.includes(k)) {
        otherConfig[k] = v
      }
    }
    setEditConfigOtherJson(
      Object.keys(otherConfig).length > 0 ? JSON.stringify(otherConfig, null, 2) : ''
    )
    setEditConfigRuntime((deployment.runtime_info as Record<string, unknown> | undefined) ?? null)
    setEditConfigVisible(true)
    // Load API configs for dropdown
    setEditApiConfigsLoading(true)
    externalApiConfigApi.list()
      .then((res) => setEditApiConfigs(res.configs || []))
      .catch(() => setEditApiConfigs([]))
      .finally(() => setEditApiConfigsLoading(false))
  }

  const closeEditConfigModal = () => {
    setEditConfigVisible(false)
    setEditConfigTarget(null)
    setEditConfigRuntime(null)
    editConfigForm.resetFields()
    setEditConfigOtherJson('')
  }

  const handleSaveConfig = async () => {
    if (!editConfigTarget) return

    // Parse "other" JSON
    let otherConfig: Record<string, unknown> = {}
    const trimmed = editConfigOtherJson.trim()
    if (trimmed) {
      try {
        const parsed = JSON.parse(trimmed)
        if (parsed !== null && (typeof parsed !== 'object' || Array.isArray(parsed))) {
          message.error(t('list.message.configMustBeJson'))
          return
        }
        otherConfig = (parsed ?? {}) as Record<string, unknown>
      } catch {
        message.error(t('list.message.configInvalidJson'))
        return
      }
    }

    const formValues = editConfigForm.getFieldsValue()

    // Build merged config: other fields first, then form fields (form wins on conflict)
    const config: Record<string, unknown> = { ...otherConfig }
    // Preserve docker_cmd from original config (read-only, auto-generated)
    const originalDockerCmd = (editConfigTarget.config ?? {} as Record<string, unknown>).docker_cmd
    if (originalDockerCmd !== undefined) {
      config.docker_cmd = originalDockerCmd
    }
    if (formValues.external_api_config_id) {
      config.external_api_config_id = formValues.external_api_config_id
    } else {
      // 显式置 null 通知后端解除绑定；省略该 key 会被后端当成"保留旧绑定"处理
      config.external_api_config_id = null
    }
    if (formValues.dtype) {
      config.dtype = formValues.dtype
    }
    if (formValues.enforce_eager) {
      config.enforce_eager = true
    }
    if (formValues.attention_backend) {
      config.attention_backend = formValues.attention_backend
    }

    const finalConfig = Object.keys(config).length > 0 ? config : null

    setSavingConfig(true)
    try {
      await deploymentApi.updateConfig(editConfigTarget.deployment_id, { config: finalConfig })
      message.success(t('list.message.configSaved'))
      closeEditConfigModal()
      refresh()
    } catch (error) {
      message.error(error instanceof Error ? error.message : t('list.message.configSaveFailed'))
    } finally {
      setSavingConfig(false)
    }
  }

  // Group columns (main table)
  const groupColumns: ColumnsType<DeploymentGroup> = [
    {
      title: t('list.groupColumns.containerName'),
      key: 'container_name',
      width: 240,
      ellipsis: true,
      render: (_, group) => {
        const containerName = group.container_name ||
          (group.endpoint?.includes('xinference') || group.inference_framework === 'xinference'
            ? 'xinference'
            : null)

        if (containerName) {
          return (
            <Space size={4}>
              <Tooltip title={containerName}>
                <Text code style={{ fontSize: 12 }}>{containerName}</Text>
              </Tooltip>
              <Button
                type="text"
                size="small"
                icon={<CopyOutlined />}
                onClick={(e) => {
                  e.stopPropagation()
                  copyToClipboard(containerName)
                }}
              />
            </Space>
          )
        }
        return <Text type="secondary">-</Text>
      },
    },
    {
      title: t('list.groupColumns.framework'),
      dataIndex: 'inference_framework',
      key: 'inference_framework',
      width: 90,
      render: (framework) => (
        <Tag color={framework === 'vllm' ? 'blue' : framework === 'sglang' ? 'green' : 'orange'}>
          {framework?.toUpperCase() || 'XF'}
        </Tag>
      ),
    },
    {
      title: t('list.groupColumns.port'),
      key: 'port',
      width: 90,
      render: (_, group) => group.port ? <Text code>{group.port}</Text> : <Text type="secondary">-</Text>,
    },
    {
      title: t('list.groupColumns.status'),
      key: 'status',
      width: 100,
      render: (_, group) => <StatusTag status={group.status} />,
    },
    {
      title: t('list.groupColumns.health'),
      key: 'health_status',
      width: 100,
      render: (_, group) => <StatusTag status={normalizeHealthStatus(group.health_status)} />,
    },
    {
      title: t('list.groupColumns.modelCount'),
      key: 'model_count',
      width: 80,
      render: (_, group) => (
        <Badge
          count={group.running_count}
          style={{ backgroundColor: group.running_count > 0 ? '#52c41a' : '#999' }}
          showZero
          overflowCount={999}
          title={`${group.running_count} / ${group.deployments.length}`}
        />
      ),
    },
    {
      title: 'GPU',
      key: 'gpu_id',
      width: 80,
      render: (_, group) => {
        // Shared containers don't show GPU at group level (models may be on different GPUs)
        if (group.deploy_mode === 'shared') return <Text type="secondary">-</Text>
        if (group.gpu_id !== undefined && group.gpu_id !== null) return `GPU ${group.gpu_id}`
        return '-'
      },
    },
    {
      title: t('list.groupColumns.gpuMemory'),
      key: 'gpu_memory',
      width: 120,
      render: (_, group) => {
        // Real-time usage (from container)
        if (group.total_memory_mb !== null) {
          const usedGB = (group.total_memory_mb / 1024).toFixed(1)
          const percent = group.total_memory_percent?.toFixed(1) || '?'
          return <span>{usedGB}GB ({percent}%)</span>
        }
        if (group.total_memory_percent !== null) {
          return <span>{group.total_memory_percent.toFixed(1)}%</span>
        }
        // Fallback to configured limit (for remote/shared services)
        if (group.total_memory_utilization !== null) {
          return <Text type="secondary">{Math.round(group.total_memory_utilization * 100)}%</Text>
        }
        return '-'
      },
    },
    {
      title: t('list.groupColumns.actions'),
      key: 'actions',
      width: 160,
      render: (_, group) => {
        // For single deployment groups, show actions directly
        if (group.deployments.length === 1) {
          const record = group.deployments[0]
          const canStart = canStartDeployment(record.status)
          const canStop = canStopDeployment(record.status)
          const canRestart = canRestartDeployment(record.status)
          const canDelete = canDeleteDeployment(record.status)
          return (
            <Space size={4}>
              <Tooltip title={t('list.tooltip.viewParams')}>
                <Button
                  type="text"
                  size="small"
                  icon={<CodeOutlined />}
                  aria-label={t('list.tooltip.viewParams')}
                  onClick={(e) => {
                    e.stopPropagation()
                    handleShowParams(record)
                  }}
                />
              </Tooltip>
              <Tooltip title={t('list.tooltip.editParams')}>
                <Button
                  type="text"
                  size="small"
                  icon={<EditOutlined />}
                  aria-label={t('list.tooltip.editParams')}
                  onClick={(e) => {
                    e.stopPropagation()
                    openEditConfigModal(record)
                  }}
                />
              </Tooltip>
              {canStart && (
                <Tooltip title={t('list.tooltip.start')}>
                  <Button
                    type="text"
                    size="small"
                    icon={<PlayCircleOutlined />}
                    loading={starting === record.deployment_id}
                    aria-label={t('list.tooltip.start')}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleStart(record.deployment_id)
                    }}
                  />
                </Tooltip>
              )}
              {canStop && (
                <>
                  <Popconfirm title={t('list.confirm.stop')} onConfirm={() => handleStop(record.deployment_id)}>
                    <Tooltip title={t('list.tooltip.stop')}>
                      <Button type="text" size="small" loading={stopping === record.deployment_id} icon={<PauseCircleOutlined />} aria-label={t('list.tooltip.stop')} onClick={(e) => e.stopPropagation()} />
                    </Tooltip>
                  </Popconfirm>
                </>
              )}
              {canRestart && (
                <>
                  <Tooltip title={t('list.tooltip.restart')}>
                    <Button
                      type="text"
                      size="small"
                      icon={<ReloadOutlined />}
                      aria-label={t('list.tooltip.restart')}
                      onClick={(e) => {
                        e.stopPropagation()
                        openRestartModal(record)
                      }}
                    />
                  </Tooltip>
                  {record.enable_lora && (
                    <Tooltip title={t('list.tooltip.adapter')}>
                      <Button
                        type="text"
                        size="small"
                        icon={<ThunderboltOutlined />}
                        aria-label={t('list.tooltip.adapter')}
                        onClick={(e) => {
                          e.stopPropagation()
                          setAdapterTarget(record)
                          setAdapterVisible(true)
                        }}
                      />
                    </Tooltip>
                  )}
                </>
              )}
              {canDelete && (
                <Popconfirm title={t('list.confirm.delete')} onConfirm={() => handleDelete(record.deployment_id)}>
                  <Tooltip title={t('list.tooltip.delete')}>
                    <Button type="text" size="small" danger icon={<DeleteOutlined />} aria-label={t('list.tooltip.delete')} onClick={(e) => e.stopPropagation()} />
                  </Tooltip>
                </Popconfirm>
              )}
            </Space>
          )
        }
        return null
      },
    },
  ]

  // Model columns (expanded table)
  const modelColumns: ColumnsType<Deployment> = [
    {
      title: t('list.modelColumns.deploymentName'),
      dataIndex: 'deployment_name',
      key: 'deployment_name',
      render: (name, record) => (
        <div>
          <span style={{ fontWeight: 500 }}>{name || '-'}</span>
          <br />
          <Tooltip title={record.deployment_id}>
            <Text copyable={{ text: record.deployment_id, tooltips: false }} style={{ color: '#888', fontSize: 11 }}>
              {record.deployment_id?.slice(0, 8)}
            </Text>
          </Tooltip>
        </div>
      ),
    },
    {
      title: t('list.modelColumns.model'),
      dataIndex: 'model_id',
      key: 'model',
      render: (modelId: string) => {
        const model = modelMap.get(modelId)
        if (!model) {
          return <span style={{ color: '#8b949e' }}>{t('list.modelColumns.unknownModel')}</span>
        }
        return (
          <Tooltip title={modelId}>
            <Button type="link" size="small" style={{ padding: 0 }} onClick={() => navigate(`/models?detail=${modelId}`)}>
              {model.model_name}
            </Button>
          </Tooltip>
        )
      },
    },
    {
      title: t('list.modelColumns.status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: DeploymentStatus) => <StatusTag status={status} />,
    },
    {
      title: t('list.modelColumns.health'),
      dataIndex: 'health_status',
      key: 'health_status',
      width: 100,
      render: (healthStatus?: HealthStatus) => <StatusTag status={normalizeHealthStatus(healthStatus)} />,
    },
    {
      title: 'GPU',
      dataIndex: 'gpu_id',
      key: 'gpu_id',
      width: 70,
      render: (_, record) => formatGpuLabel(record),
    },
    {
      title: t('list.modelColumns.gpuMemory'),
      key: 'gpu_memory',
      width: 120,
      render: (_, record) => {
        // Only show memory for running deployments
        if (record.status !== 'running') return '-'
        if (record.gpu_memory_used_mb !== undefined && record.gpu_memory_used_mb !== null) {
          const usedGB = (record.gpu_memory_used_mb / 1024).toFixed(1)
          const percent = record.gpu_memory_used_percent?.toFixed(1) || '?'
          return <span>{usedGB}GB ({percent}%)</span>
        }
        return record.gpu_memory_utilization ? (
          <span style={{ color: '#999' }}>{Math.round(record.gpu_memory_utilization * 100)}%</span>
        ) : '-'
      },
    },
    {
      title: t('list.modelColumns.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => formatDate(time),
    },
    {
      title: t('list.modelColumns.actions'),
      key: 'actions',
      width: 160,
      render: (_, record) => {
        const canStart = canStartDeployment(record.status)
        const canStop = canStopDeployment(record.status)
        const canRestart = canRestartDeployment(record.status)
        const canDelete = canDeleteDeployment(record.status)

        return (
          <Space size={4}>
            <Tooltip title={t('list.tooltip.viewParams')}>
              <Button
                type="text"
                size="small"
                icon={<CodeOutlined />}
                onClick={() => handleShowParams(record)}
              />
            </Tooltip>
            <Tooltip title={t('list.tooltip.editParams')}>
              <Button
                type="text"
                size="small"
                icon={<EditOutlined />}
                onClick={() => openEditConfigModal(record)}
              />
            </Tooltip>
            {canStart && (
              <Tooltip title={t('list.tooltip.start')}>
                <Button
                  type="text"
                  size="small"
                  icon={<PlayCircleOutlined />}
                  loading={starting === record.deployment_id}
                  aria-label={t('list.tooltip.start')}
                  onClick={() => handleStart(record.deployment_id)}
                />
              </Tooltip>
            )}
            {canStop && (
              <Popconfirm title={t('list.confirm.stop')} onConfirm={() => handleStop(record.deployment_id)}>
                <Tooltip title={t('list.tooltip.stop')}>
                  <Button type="text" size="small" loading={stopping === record.deployment_id} icon={<PauseCircleOutlined />} aria-label={t('list.tooltip.stop')} />
                </Tooltip>
              </Popconfirm>
            )}
            {canRestart && (
              <Tooltip title={t('list.tooltip.restart')}>
                <Button
                  type="text"
                  size="small"
                  icon={<ReloadOutlined />}
                  aria-label={t('list.tooltip.restart')}
                  onClick={() => openRestartModal(record)}
                />
              </Tooltip>
            )}
            {canRestart && record.enable_lora && (
              <Tooltip title={t('list.tooltip.adapter')}>
                <Button
                  type="text"
                  size="small"
                  icon={<ThunderboltOutlined />}
                  aria-label={t('list.tooltip.adapter')}
                  onClick={() => {
                    setAdapterTarget(record)
                    setAdapterVisible(true)
                  }}
                />
              </Tooltip>
            )}
            {canDelete && (
              <Popconfirm title={t('list.confirm.delete')} onConfirm={() => handleDelete(record.deployment_id)}>
                <Tooltip title={t('list.tooltip.delete')}>
                  <Button type="text" size="small" danger icon={<DeleteOutlined />} aria-label={t('list.tooltip.delete')} />
                </Tooltip>
              </Popconfirm>
            )}
          </Space>
        )
      },
    },
  ]

  // Expanded row render
  const expandedRowRender = (group: DeploymentGroup) => {
    if (group.deployments.length <= 1) {
      return null
    }
    return (
      <Table
        rowKey="deployment_id"
        columns={modelColumns}
        dataSource={group.deployments}
        pagination={false}
        size="small"
        style={{ margin: '0 0 0 48px' }}
      />
    )
  }

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={6}>
          <StatCard
            title={t('list.stats.total')}
            value={stats.total}
            icon={<RocketOutlined />}
            color={STATUS_INFO}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('list.stats.running')}
            value={stats.running}
            icon={<CheckCircleOutlined />}
            color={STATUS_SUCCESS}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('list.stats.stopped')}
            value={stats.stopped}
            icon={<ClockCircleOutlined />}
            color={STATUS_WARNING}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={t('list.stats.failed')}
            value={stats.failed}
            icon={<CloseCircleOutlined />}
            color={STATUS_ERROR}
          />
        </Col>
      </Row>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>
          {t('list.title')}
        </Title>
        <Space>
          <Select
            placeholder={t('list.filter.statusPlaceholder')}
            allowClear
            style={{ width: 120 }}
            value={statusFilter}
            onChange={setStatusFilter}
            options={STATUS_FILTER_OPTIONS}
          />
          <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateVisible(true)}>
            {t('list.createDeployment')}
          </Button>
        </Space>
      </div>

      <Table
        rowKey="key"
        columns={groupColumns}
        dataSource={groupedDeployments}
        loading={loading}
        expandable={{
          expandedRowRender,
          rowExpandable: (group) => group.deployments.length > 1,
          expandedRowKeys: expandedKeys,
          onExpandedRowsChange: (keys) => setExpandedKeys(keys as string[]),
          expandIcon: ({ expanded, onExpand, record }) =>
            record.deployments.length > 1 ? (
              expanded ? (
                <DownOutlined onClick={(e) => onExpand(record, e)} style={{ cursor: 'pointer', marginRight: 8 }} />
              ) : (
                <RightOutlined onClick={(e) => onExpand(record, e)} style={{ cursor: 'pointer', marginRight: 8 }} />
              )
            ) : (
              <span style={{ width: 16, marginRight: 8, display: 'inline-block' }} />
            ),
        }}
        pagination={{
          current: page,
          pageSize,
          total,
          showSizeChanger: true,
          showTotal: (count) => t('list.pagination.totalDeployments', { total: count }),
          onChange: (p, ps) => {
            setPage(p)
            setPageSize(ps)
          },
        }}
      />

      <CreateDeploymentModal
        visible={createVisible}
        models={models.filter((m) => !m.is_adapter)}
        creating={creating}
        onCancel={() => setCreateVisible(false)}
        onCreate={handleCreate}
        onBindExisting={handleBindExisting}
      />

      {/* Restart Modal */}
      <Modal
        title={t('restart.title')}
        open={restartVisible}
        onCancel={() => setRestartVisible(false)}
        onOk={handleRestart}
        okText={t('restart.okText')}
        confirmLoading={restarting}
        destroyOnHidden
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text>{t('restart.description')}</Text>
          <div style={{ marginTop: 16 }}>
            <Text strong style={{ display: 'block', marginBottom: 8 }}>{t('restart.modeLabel')}</Text>
            <Radio.Group value={restartMode} onChange={(e) => setRestartMode(e.target.value)}>
              <Space direction="vertical">
                <Radio value="auto">
                  {t('restart.modeAuto')}
                  <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                    {t('restart.modeAutoDesc')}
                  </Text>
                </Radio>
                {restartTarget?.inference_framework === 'xinference' && (
                  <Radio value="model">
                    {t('restart.modeModel')}
                    <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                      {t('restart.modeModelDesc')}
                    </Text>
                  </Radio>
                )}
                {restartTarget?.deploy_mode === 'container' && (
                  <Radio value="container">
                    {t('restart.modeContainer')}
                    <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                      {t('restart.modeContainerDesc')}
                    </Text>
                  </Radio>
                )}
              </Space>
            </Radio.Group>
          </div>
          {restartMode === 'container' && restartTarget?.deploy_mode === 'container' && restartTarget.gpu_id !== null && restartTarget.gpu_id !== undefined && (
            <div style={{ marginTop: 16 }}>
              <Checkbox
                checked={restartResetGpu}
                onChange={(e) => setRestartResetGpu(e.target.checked)}
              >
                {t('restart.resetGpu')}
              </Checkbox>
            </div>
          )}
        </Space>
      </Modal>

      {/* Edit Config Modal */}
      <Modal
        title={t('editConfig.title')}
        open={editConfigVisible}
        onCancel={closeEditConfigModal}
        onOk={handleSaveConfig}
        okText={t('editConfig.okText')}
        confirmLoading={savingConfig}
        destroyOnHidden
        width={640}
      >
        <Alert message={t('editConfig.hint')} type="info" showIcon style={{ marginBottom: 16 }} />
        <Form form={editConfigForm} layout="vertical" size="middle">
          <Form.Item
            name="external_api_config_id"
            label={t('editConfig.externalApiConfig')}
            extra={t('editConfig.externalApiConfigHint')}
          >
            <Select
              allowClear
              showSearch
              loading={editApiConfigsLoading}
              placeholder={t('editConfig.externalApiConfigPlaceholder')}
              optionFilterProp="label"
              options={editApiConfigs.map((cfg) => ({
                label: `${cfg.config_name} (${cfg.config_id.slice(0, 8)})`,
                value: cfg.config_id,
              }))}
            />
          </Form.Item>

          <Form.Item
            name="dtype"
            label={t('editConfig.dtype')}
            extra={t('editConfig.dtypeHint')}
          >
            <Select
              allowClear
              placeholder={t('editConfig.dtypePlaceholder')}
              options={[
                { label: t('editConfig.dtypeBfloat16'), value: 'bfloat16' },
                { label: 'float16', value: 'float16' },
                { label: 'float32', value: 'float32' },
                { label: 'auto', value: 'auto' },
              ]}
            />
          </Form.Item>

          {editConfigTarget?.inference_framework === 'vllm' && (
            <Form.Item
              name="enforce_eager"
              label={t('editConfig.enforceEager')}
              valuePropName="checked"
              extra={t('editConfig.enforceEagerHint')}
            >
              <Switch />
            </Form.Item>
          )}

          {editConfigTarget?.inference_framework === 'sglang' && (
            <Form.Item
              name="attention_backend"
              label={t('editConfig.attentionBackend')}
              extra={t('editConfig.attentionBackendHint')}
            >
              <Select
                allowClear
                placeholder={t('editConfig.attentionBackendPlaceholder')}
                options={[
                  { label: t('editConfig.attentionFlashinfer'), value: 'flashinfer' },
                  { label: t('editConfig.attentionTorchNative'), value: 'torch_native' },
                  { label: t('editConfig.attentionTriton'), value: 'triton' },
                  { label: t('editConfig.attentionFa3'), value: 'fa3' },
                ]}
              />
            </Form.Item>
          )}

          <Form.Item
            label={t('editConfig.otherConfig')}
            extra={t('editConfig.otherConfigHint')}
          >
            <Input.TextArea
              value={editConfigOtherJson}
              onChange={(e) => setEditConfigOtherJson(e.target.value)}
              rows={5}
              placeholder={t('editConfig.otherConfigPlaceholder')}
              style={{ fontFamily: 'monospace', fontSize: 12 }}
            />
          </Form.Item>
        </Form>

        {editConfigRuntime && Object.keys(editConfigRuntime).length > 0 && (
          <>
            <Text type="secondary">{t('editConfig.runtimeParams')}</Text>
            <pre style={{
              background: '#1a1a2e',
              color: '#e0e0e0',
              padding: 16,
              borderRadius: 8,
              overflow: 'auto',
              maxHeight: 200,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              fontSize: 12,
              lineHeight: 1.6,
              marginTop: 8,
              marginBottom: 0,
            }}>
              {JSON.stringify(editConfigRuntime, null, 2)}
            </pre>
          </>
        )}
      </Modal>

      {/* Params Detail Modal */}
      <Modal
        title={t('paramsModal.title')}
        open={paramsVisible}
        onCancel={() => setParamsVisible(false)}
        footer={<Button onClick={() => setParamsVisible(false)}>{t('common:action.close')}</Button>}
        width={720}
      >
        {paramsDeployment && (() => {
          const dep = paramsDeployment
          const config = dep.config || {}
          const dockerCmd = config.docker_cmd as string | undefined
          const model = modelMap.get(dep.model_id)
          const runtimeInfo = dep.runtime_info as Record<string, unknown> | undefined

          // Parse key parameters from docker_cmd as fallback
          const parseFromCmd = (flag: string): string | undefined => {
            if (!dockerCmd) return undefined
            // --flag value
            const re = new RegExp(`${flag}[\\s=]+([^\\s]+)`)
            const m = dockerCmd.match(re)
            return m?.[1]
          }
          const hasFlagInCmd = (flag: string): boolean => {
            return !!dockerCmd?.includes(flag)
          }

          const dtype = (config.dtype as string) || parseFromCmd('--dtype') || '-'
          const enforceEager = config.enforce_eager === true || hasFlagInCmd('--enforce-eager')
          const attentionBackend = (config.attention_backend as string) || parseFromCmd('--attention-backend') || '-'
          const maxModelLen = parseFromCmd('--max-model-len') || parseFromCmd('--max_model_len')
          const tensorParallelSize = parseFromCmd('--tensor-parallel-size') || parseFromCmd('--tp')
          const chatTemplate = parseFromCmd('--chat-template')

          const runtimeAccelerators = getRuntimeAccelerators(dep)
          const hasRuntimeInfo = runtimeInfo && Object.keys(runtimeInfo).length > 0
          const hasConfig = config && Object.keys(config).length > 0

          const renderJsonBlock = (title: string, value: Record<string, unknown>) => (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <Text strong>{title}</Text>
                <Button
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={() => {
                    copyText(JSON.stringify(value, null, 2))
                    message.success(t('list.message.copied'))
                  }}
                >
                  {t('common:action.copy')}
                </Button>
              </div>
              <pre style={{
                background: '#1a1a2e',
                color: '#e0e0e0',
                padding: 16,
                borderRadius: 8,
                overflow: 'auto',
                maxHeight: 300,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
                fontSize: 13,
                lineHeight: 1.6,
                marginBottom: 16,
              }}>
                {JSON.stringify(value, null, 2)}
              </pre>
            </>
          )

          // Framework-specific parameters
          const isVllm = dep.inference_framework === 'vllm'
          const isSglang = dep.inference_framework === 'sglang'

          return (
            <>
              <Descriptions column={2} bordered size="small" style={{ marginBottom: 16 }}>
                <Descriptions.Item label={t('paramsModal.deploymentName')}>{dep.deployment_name || '-'}</Descriptions.Item>
                <Descriptions.Item label={t('paramsModal.inferenceFramework')}>
                  <Tag color={isVllm ? 'blue' : isSglang ? 'green' : 'orange'}>
                    {dep.inference_framework?.toUpperCase()}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label={t('paramsModal.model')}>
                  {model ? (
                    <Button type="link" size="small" style={{ padding: 0 }} onClick={() => { setParamsVisible(false); navigate(`/models?detail=${dep.model_id}`) }}>
                      {model.model_name}
                    </Button>
                  ) : dep.model_id}
                </Descriptions.Item>
                <Descriptions.Item label={t('paramsModal.containerMode')}>
                  {dep.deploy_mode === 'container' ? t('paramsModal.containerModeStandalone') : t('paramsModal.containerModeShared')}
                </Descriptions.Item>
                <Descriptions.Item label={t('paramsModal.dtype')}>
                  <Tag color={dtype === 'bfloat16' ? 'blue' : dtype === 'float16' ? 'orange' : 'default'}>{dtype}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label={t('paramsModal.gpuMemoryLimit')}>{dep.gpu_memory_utilization ? `${Math.round(dep.gpu_memory_utilization * 100)}%` : '-'}</Descriptions.Item>
                {isVllm && (
                  <Descriptions.Item label={t('paramsModal.enforceEager')}>
                    <Tag color={enforceEager ? 'warning' : 'default'}>{enforceEager ? t('paramsModal.enforceEagerYes') : t('paramsModal.enforceEagerNo')}</Tag>
                  </Descriptions.Item>
                )}
                {isSglang && (
                  <Descriptions.Item label={t('paramsModal.enforceEager')}>
                    <Tag color={attentionBackend !== '-' ? 'processing' : 'default'}>{attentionBackend}</Tag>
                  </Descriptions.Item>
                )}
                <Descriptions.Item label={t('paramsModal.gpu')}>
                  {runtimeAccelerators.length > 0
                    ? `GPU ${runtimeAccelerators.join(',')}`
                    : dep.gpu_id !== undefined && dep.gpu_id !== null
                      ? `GPU ${dep.gpu_id}`
                      : dep.deploy_mode === 'shared'
                        ? t('paramsModal.gpuShared')
                        : '-'}
                </Descriptions.Item>
                <Descriptions.Item label={t('paramsModal.lora')}>{dep.enable_lora ? t('paramsModal.loraEnabled', { maxLoras: dep.max_loras, maxRank: dep.max_lora_rank }) : t('paramsModal.loraDisabled')}</Descriptions.Item>
                {maxModelLen && <Descriptions.Item label={t('paramsModal.maxLength')}>{maxModelLen}</Descriptions.Item>}
                {tensorParallelSize && <Descriptions.Item label={t('paramsModal.tensorParallel')}>{tensorParallelSize}</Descriptions.Item>}
                {chatTemplate && (
                  <Descriptions.Item label="Chat Template" span={2}>
                    <Text code style={{ fontSize: 12 }}>{chatTemplate}</Text>
                  </Descriptions.Item>
                )}
                <Descriptions.Item label={t('paramsModal.replicaCount')}>{dep.replica}</Descriptions.Item>
                <Descriptions.Item label={t('paramsModal.port')}>{dep.port || '-'}</Descriptions.Item>
              </Descriptions>
              {hasRuntimeInfo && renderJsonBlock(t('paramsModal.runtimeParams'), runtimeInfo)}
              {hasConfig && renderJsonBlock(t('paramsModal.customConfig'), config)}
              {dockerCmd && (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <Text strong>{t('paramsModal.startupCommand')}</Text>
                    <Button
                      size="small"
                      icon={<CopyOutlined />}
                      onClick={() => {
                        copyText(dockerCmd)
                        message.success(t('list.message.copied'))
                      }}
                    >
                      {t('common:action.copy')}
                    </Button>
                  </div>
                  <pre style={{
                    background: '#1a1a2e',
                    color: '#e0e0e0',
                    padding: 16,
                    borderRadius: 8,
                    overflow: 'auto',
                    maxHeight: 300,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all',
                    fontSize: 13,
                    lineHeight: 1.6,
                  }}>
                    {dockerCmd}
                  </pre>
                </>
              )}
            </>
          )
        })()}
      </Modal>

      {/* Adapter Management Modal */}
      <Modal
        title={t('adapter.title')}
        open={adapterVisible}
        onCancel={() => {
          setAdapterVisible(false)
          setAdapterTarget(null)
        }}
        footer={null}
        width={720}
        destroyOnHidden
      >
        {adapterTarget && (
          <AdapterManager deployment={adapterTarget} onRefresh={refresh} />
        )}
      </Modal>
    </div>
  )
}
