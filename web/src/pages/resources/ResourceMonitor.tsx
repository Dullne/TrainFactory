import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { Typography, Card, Row, Col, Button, Space, message, Spin, Alert } from 'antd'
import {
  ReloadOutlined,
  DesktopOutlined,
  HddOutlined,
  DatabaseOutlined,
  DashboardOutlined,
  ClearOutlined,
} from '@ant-design/icons'
import { resourceApi, SILENT_REQUEST_CONFIG } from '@/services/api'
import { useAuth } from '@/auth/AuthContext'
import { usePolling } from '@/hooks'
import { StatCard } from '@/components/StatCard'
import { GpuCard, getUsageColor } from './GpuCard'
import type { GpuInfo, GpuProcess } from './GpuCard'
import {
  BG_ELEVATED,
  BORDER_PRIMARY,
  TEXT_SECONDARY,
  STATUS_SUCCESS,
  STATUS_WARNING,
} from '@/theme'

const { Title } = Typography

interface ResourceStatus {
  available?: boolean
  partial?: boolean
  error_code?: string | null
  gpu: {
    available?: boolean
    partial?: boolean
    device_state?: 'detected' | 'no_devices' | 'unknown'
    total_gpus: number | null
    allocated_gpus: number | null
    free_gpus: number | null
    cpu_summary?: {
      logical_cores: number
      physical_cores: number
      cpu_usage_percent: number
    }
    system_memory?: {
      total_gb: number
      used_gb: number
      free_gb: number
      usage_percent: number
    }
  }
  system: {
    available?: boolean
    partial?: boolean
    cpu_percent: number | null
    memory_percent: number | null
    memory_used_mb: number | null
    memory_total_mb: number | null
    memory_limit_set?: boolean
    disk_usage_percent: number | null
    disk_used_gb: number | null
    disk_total_gb: number | null
    disk_mountpoint?: string | null
    open_files: number | null
    thread_count: number | null
  }
}

type RefreshState = 'fresh' | 'partial' | 'stale' | 'unavailable'

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export default function ResourceMonitor() {
  const { t } = useTranslation(['resources', 'common'])
  const { user } = useAuth()
  const [loading, setLoading] = useState(false)
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  const [status, setStatus] = useState<ResourceStatus | null>(null)
  const [gpuProcesses, setGpuProcesses] = useState<Record<number, GpuProcess[]>>({})
  const [cleaning, setCleaning] = useState(false)
  const [refreshState, setRefreshState] = useState<RefreshState>('fresh')
  const hasDataRef = useRef(false)
  const requestGenerationRef = useRef(0)
  const loadingOwnerRef = useRef(0)
  const mountedRef = useRef(true)

  const fetchData = useCallback(async (silent = false) => {
    const requestGeneration = ++requestGenerationRef.current
    const loadingOwner = silent ? null : ++loadingOwnerRef.current
    if (loadingOwner !== null) setLoading(true)

    try {
      const [gpuResult, statusResult, processesResult] = await Promise.allSettled([
        resourceApi.getGpuList(SILENT_REQUEST_CONFIG),
        resourceApi.getStatus(SILENT_REQUEST_CONFIG),
        resourceApi.getGpuProcesses(SILENT_REQUEST_CONFIG),
      ])

      if (requestGeneration !== requestGenerationRef.current) return

      const hadData = hasDataRef.current
      const requestFailed = [gpuResult, statusResult, processesResult].some(
        (result) => result.status === 'rejected'
      )
      const statusUnavailable =
        statusResult.status === 'fulfilled' && statusResult.value.available === false
      const preservePreviousSnapshot = hadData && (requestFailed || statusUnavailable)
      let successfulCoreRequest = false

      if (!preservePreviousSnapshot) {
        if (gpuResult.status === 'fulfilled') {
          const nextGpus = gpuResult.value.gpus || []
          setGpus(nextGpus)
          successfulCoreRequest = nextGpus.length > 0
        }
        if (statusResult.status === 'fulfilled') {
          setStatus(statusResult.value)
          successfulCoreRequest ||= !statusUnavailable
        }
        if (processesResult.status === 'fulfilled') {
          const processMap: Record<number, GpuProcess[]> = {}
          for (const item of processesResult.value.gpus || []) {
            if (typeof item.gpu_index === 'number') {
              processMap[item.gpu_index] = item.processes || []
            }
          }
          setGpuProcesses(processMap)
        }
      }

      if (successfulCoreRequest) hasDataRef.current = true
      const backendPartial =
        statusResult.status === 'fulfilled' && statusResult.value.partial === true

      if (statusUnavailable) {
        setRefreshState(hadData ? 'stale' : 'unavailable')
      } else if (requestFailed) {
        setRefreshState(hadData ? 'stale' : successfulCoreRequest ? 'partial' : 'unavailable')
        if (!silent) {
          message.error({
            key: 'resource-monitor-fetch-failed',
            content: t('monitor.message.fetchFailed'),
          })
        }
      } else {
        setRefreshState(backendPartial ? 'partial' : 'fresh')
      }
    } finally {
      if (
        loadingOwner !== null &&
        loadingOwner === loadingOwnerRef.current &&
        mountedRef.current
      ) {
        setLoading(false)
      }
    }
  }, [t])

  useEffect(() => {
    void fetchData()
  }, [fetchData])

  useEffect(
    () => {
      mountedRef.current = true
      return () => {
        mountedRef.current = false
        requestGenerationRef.current += 1
        loadingOwnerRef.current += 1
      }
    },
    []
  )

  // Auto refresh every 5 seconds
  usePolling(() => fetchData(true), {
    interval: 5000,
    enabled: true,
  })

  const handleCleanup = async () => {
    setCleaning(true)
    try {
      const result = await resourceApi.cleanup()
      if (result.success) {
        message.success(t('monitor.message.cleanupSuccess'))
        void fetchData()
      } else {
        message.error(t('monitor.message.cleanupFailed'))
      }
    } catch {
      message.error(t('monitor.message.cleanupFailed'))
    } finally {
      setCleaning(false)
    }
  }

  // Summary stats
  const stats = useMemo(() => {
    if (!status) return null
    const memoryUsedGb = isFiniteNumber(status.system.memory_used_mb)
      ? status.system.memory_used_mb / 1024
      : null
    const memoryTotalGb = isFiniteNumber(status.system.memory_total_mb)
      ? status.system.memory_total_mb / 1024
      : null
    const memoryUsage = status.system.memory_percent
    const diskUsedGb = status.system.disk_used_gb
    const diskTotalGb = status.system.disk_total_gb
    const diskUsage = status.system.disk_usage_percent
    const cpuPhysical = status.gpu.cpu_summary?.physical_cores || 0
    const cpuLogical = status.gpu.cpu_summary?.logical_cores || 0
    return {
      totalGpus: status.gpu.total_gpus,
      allocatedGpus: status.gpu.allocated_gpus,
      freeGpus: status.gpu.free_gpus,
      cpuUsage: status.gpu.cpu_summary?.cpu_usage_percent ?? status.system.cpu_percent,
      cpuPhysical: status.gpu.cpu_summary ? cpuPhysical : null,
      cpuLogical: status.gpu.cpu_summary ? cpuLogical : null,
      memoryUsage,
      memoryUsedGb,
      memoryTotalGb,
      memoryLimitSet: status.system.memory_limit_set ?? false,
      diskUsage,
      diskUsedGb,
      diskTotalGb,
    }
  }, [status])

  const memoryTitle = stats
    ? !isFiniteNumber(stats.memoryUsage)
      ? t('monitor.stat.memoryUnavailable')
      : stats.memoryLimitSet
      ? t('monitor.stat.memory', { percent: stats.memoryUsage.toFixed(0) })
      : t('monitor.stat.memoryUnlimited', { percent: stats.memoryUsage.toFixed(0) })
    : ''
  const memoryTip = stats
    ? !isFiniteNumber(stats.memoryUsage)
      ? t('monitor.tip.memoryUnavailable')
      : stats.memoryLimitSet && isFiniteNumber(stats.memoryTotalGb) && isFiniteNumber(stats.memoryUsedGb)
      ? t('monitor.tip.memoryLimited', {
          total: stats.memoryTotalGb.toFixed(0),
          used: stats.memoryUsedGb.toFixed(1),
        })
      : t('monitor.tip.memoryUnlimited')
    : undefined

  const inlineState =
    refreshState === 'stale'
      ? {
          testId: 'resource-stale-state',
          type: 'warning' as const,
          text: t('monitor.state.stale'),
        }
      : refreshState === 'unavailable'
        ? {
            testId: 'resource-unavailable-state',
            type: 'error' as const,
            text: t('monitor.state.unavailable'),
          }
        : refreshState === 'partial'
          ? {
              testId: 'resource-partial-state',
              type: 'warning' as const,
              text: t('monitor.state.partial'),
            }
          : null

  if (loading && !status) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div>
      {/* System Info Header */}
      <Title level={4} style={{ marginBottom: 16 }}>
        {t('monitor.title.system')}
      </Title>

      {inlineState && (
        <Alert
          data-testid={inlineState.testId}
          type={inlineState.type}
          message={inlineState.text}
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Stats Panel */}
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={6}>
          <StatCard
            title="GPU"
            value={
              isFiniteNumber(stats?.allocatedGpus) && isFiniteNumber(stats?.totalGpus)
                ? `${stats.allocatedGpus}/${stats.totalGpus}`
                : '--'
            }
            icon={<DesktopOutlined />}
            color={
              !isFiniteNumber(stats?.allocatedGpus)
                ? TEXT_SECONDARY
                : stats.allocatedGpus > 0
                  ? STATUS_WARNING
                  : STATUS_SUCCESS
            }
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={
              isFiniteNumber(stats?.cpuUsage)
                ? `CPU (${stats.cpuUsage.toFixed(0)}%)`
                : t('monitor.stat.cpuUnavailable')
            }
            value={
              isFiniteNumber(stats?.cpuPhysical) && isFiniteNumber(stats?.cpuLogical)
                ? `${stats.cpuPhysical}C/${stats.cpuLogical}T`
                : '--'
            }
            icon={<DashboardOutlined />}
            color={
              isFiniteNumber(stats?.cpuUsage) ? getUsageColor(stats.cpuUsage) : TEXT_SECONDARY
            }
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={memoryTitle}
            value={
              isFiniteNumber(stats?.memoryUsedGb) && isFiniteNumber(stats?.memoryTotalGb)
                ? `${stats.memoryUsedGb.toFixed(1)}/${stats.memoryTotalGb.toFixed(0)}G`
                : '--'
            }
            icon={<DatabaseOutlined />}
            color={
              isFiniteNumber(stats?.memoryUsage)
                ? getUsageColor(stats.memoryUsage)
                : TEXT_SECONDARY
            }
            tooltip={memoryTip}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title={
              isFiniteNumber(stats?.diskUsage)
                ? t('monitor.stat.disk', { percent: stats.diskUsage.toFixed(0) })
                : t('monitor.stat.diskUnavailable')
            }
            value={
              isFiniteNumber(stats?.diskUsedGb) && isFiniteNumber(stats?.diskTotalGb)
                ? `${stats.diskUsedGb.toFixed(0)}/${stats.diskTotalGb.toFixed(0)}G`
                : '--'
            }
            icon={<HddOutlined />}
            color={
              isFiniteNumber(stats?.diskUsage) ? getUsageColor(stats.diskUsage) : TEXT_SECONDARY
            }
            tooltip={
              status?.system.disk_mountpoint
                ? t('monitor.tip.disk', { mount: status.system.disk_mountpoint })
                : isFiniteNumber(stats?.diskUsage)
                  ? t('monitor.tip.diskConfigured')
                  : t('monitor.tip.diskUnavailable')
            }
          />
        </Col>
      </Row>

      {/* Toolbar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>
          {t('monitor.title.gpu')}
        </Title>
        <Space>
          {user?.is_admin && (
            <Button icon={<ClearOutlined />} onClick={handleCleanup} loading={cleaning}>
              {t('monitor.action.cleanup')}
            </Button>
          )}
          <Button icon={<ReloadOutlined />} onClick={() => void fetchData()} loading={loading}>
            {t('common:action.refresh')}
          </Button>
        </Space>
      </div>

      {/* GPU Cards */}
      {gpus.length > 0 ? (
        <Row gutter={[16, 16]}>
          {gpus.map((gpu) => (
            <GpuCard key={gpu.id} gpu={gpu} processes={gpuProcesses[gpu.id]} />
          ))}
        </Row>
      ) : (
        <Card
          style={{
            background: BG_ELEVATED,
            borderColor: BORDER_PRIMARY,
            textAlign: 'center',
            padding: 40,
          }}
        >
          <DesktopOutlined style={{ fontSize: 48, color: TEXT_SECONDARY, marginBottom: 16 }} />
          <div style={{ color: TEXT_SECONDARY }}>
            {status?.gpu.available === false
              ? t('monitor.empty.unavailable')
              : status?.gpu.total_gpus === 0
                ? t('monitor.empty.noGpu')
                : t('monitor.empty.noData')}
          </div>
        </Card>
      )}
    </div>
  )
}
