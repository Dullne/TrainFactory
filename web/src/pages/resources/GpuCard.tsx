import { Card, Progress, List, Tooltip, Tag, Typography, Col, Popover } from 'antd'
import {
  DesktopOutlined,
  ThunderboltOutlined,
  FireOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import {
  TEXT_PRIMARY,
  TEXT_SECONDARY,
  BG_ELEVATED,
  BORDER_PRIMARY,
  STATUS_SUCCESS,
  STATUS_WARNING,
  STATUS_ERROR,
} from '@/theme'
import { cardStyle, cardBodyStyle, flexBetween, flexWithGap, coloredIconContainer } from '@/styles'

const { Text } = Typography

export interface GpuInfo {
  id: number
  name: string
  memory_total_gb: number
  memory_used_gb: number
  memory_free_gb: number
  memory_usage_percent: number
  gpu_utilization: number | null
  temperature: number | null
  power_usage_w: number | null
  power_limit_w: number | null
  is_allocated: boolean
  allocated_task: string | null
}

export interface GpuProcess {
  pid?: number
  name?: string | null
  cmdline?: string | null
  user?: string | null
  gpu_memory_mb: number | null
  container_id?: string | null
  container_name?: string | null
  started_at?: string | null
}

interface GpuCardProps {
  gpu: GpuInfo
  processes?: GpuProcess[]
}

const getUsageColor = (percent: number) => {
  if (percent >= 90) return STATUS_ERROR
  if (percent >= 70) return STATUS_WARNING
  return STATUS_SUCCESS
}

export function GpuCard({ gpu, processes = [] }: GpuCardProps) {
  const { t } = useTranslation(['resources', 'common'])
  const memoryPercent = gpu.memory_usage_percent || 0
  const utilizationPercent = gpu.gpu_utilization || 0

  const statusColor = gpu.is_allocated ? STATUS_WARNING : STATUS_SUCCESS

  // 计算进程总显存占用
  const totalProcessMemory = processes.reduce((sum, p) => sum + (p.gpu_memory_mb || 0), 0)

  const processListContent = (
    <div style={{ width: 320, maxHeight: 300, overflow: 'auto' }}>
      {processes.length ? (
        <List
          size="small"
          dataSource={processes}
          split={false}
          renderItem={(process) => {
            const displayName = process.cmdline || process.name || t('gpuCard.process.name')
            const containerLabel =
              process.container_name ||
              (process.container_id ? process.container_id.slice(0, 12) : null)
            const meta = [
              process.pid ? `PID ${process.pid}` : null,
              containerLabel ? t('gpuCard.process.container', { name: containerLabel }) : null,
            ]
              .filter(Boolean)
              .join(' · ')
            return (
              <List.Item style={{ padding: '6px 0', borderBottom: `1px solid ${BORDER_PRIMARY}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <Tooltip title={displayName}>
                      <Text
                        style={{
                          color: TEXT_PRIMARY,
                          fontSize: 12,
                          display: 'block',
                          maxWidth: 220,
                        }}
                        ellipsis
                      >
                        {displayName}
                      </Text>
                    </Tooltip>
                    {meta && <Text style={{ color: TEXT_SECONDARY, fontSize: 10 }}>{meta}</Text>}
                  </div>
                  <Text style={{ color: TEXT_PRIMARY, fontSize: 12, marginLeft: 8 }}>
                    {process.gpu_memory_mb !== null ? `${process.gpu_memory_mb} MB` : '-'}
                  </Text>
                </div>
              </List.Item>
            )
          }}
        />
      ) : (
        <div style={{ color: TEXT_SECONDARY, fontSize: 12, padding: '8px 0', textAlign: 'center' }}>
          {t('gpuCard.process.none')}
        </div>
      )}
    </div>
  )

  return (
    <Col xs={24} sm={12} lg={8} xl={6}>
      <Card style={cardStyle} styles={{ body: cardBodyStyle }}>
        {/* Header */}
        <div style={{ ...flexBetween, marginBottom: 12 }}>
          <div style={flexWithGap(8)}>
            <div style={{ ...coloredIconContainer(statusColor, 36), fontSize: 18 }}>
              <DesktopOutlined />
            </div>
            <div>
              <Text strong style={{ color: TEXT_PRIMARY, fontSize: 14 }}>
                GPU {gpu.id}
              </Text>
              <br />
              <Text style={{ color: TEXT_SECONDARY, fontSize: 11 }}>{gpu.name}</Text>
            </div>
          </div>
          <Tag color={gpu.is_allocated ? 'orange' : 'green'}>
            {gpu.is_allocated ? t('gpuCard.status.allocated') : t('gpuCard.status.free')}
          </Tag>
        </div>

        {/* Memory Usage */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ ...flexBetween, marginBottom: 4 }}>
            <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>{t('gpuCard.label.vram')}</Text>
            <Text style={{ color: getUsageColor(memoryPercent), fontSize: 12 }}>
              {gpu.memory_used_gb?.toFixed(1)} / {gpu.memory_total_gb?.toFixed(1)} GB
            </Text>
          </div>
          <Progress
            percent={memoryPercent}
            showInfo={false}
            strokeColor={getUsageColor(memoryPercent)}
            trailColor={`${getUsageColor(memoryPercent)}20`}
            size="small"
          />
        </div>

        {/* GPU Utilization */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ ...flexBetween, marginBottom: 4 }}>
            <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>
              {t('gpuCard.label.utilization')}
            </Text>
            <Text style={{ color: getUsageColor(utilizationPercent), fontSize: 12 }}>
              {utilizationPercent}%
            </Text>
          </div>
          <Progress
            percent={utilizationPercent}
            showInfo={false}
            strokeColor={getUsageColor(utilizationPercent)}
            trailColor={`${getUsageColor(utilizationPercent)}20`}
            size="small"
          />
        </div>

        {/* Additional Info Row */}
        <div style={{ ...flexBetween, fontSize: 11 }}>
          <div style={{ display: 'flex', gap: 12 }}>
            {gpu.temperature !== null && (
              <Tooltip title={t('gpuCard.label.temperature')}>
                <span style={{ color: TEXT_SECONDARY }}>
                  <FireOutlined style={{ marginRight: 4 }} />
                  {gpu.temperature}°C
                </span>
              </Tooltip>
            )}
            {gpu.power_usage_w !== null && (
              <Tooltip title={t('gpuCard.label.power')}>
                <span style={{ color: TEXT_SECONDARY }}>
                  <ThunderboltOutlined style={{ marginRight: 4 }} />
                  {gpu.power_usage_w?.toFixed(0)}/{gpu.power_limit_w?.toFixed(0) || '?'}W
                </span>
              </Tooltip>
            )}
          </div>
          {gpu.allocated_task && (
            <Tooltip title={t('gpuCard.label.task', { task: gpu.allocated_task })}>
              <span style={{ color: STATUS_WARNING }}>{gpu.allocated_task.substring(0, 8)}...</span>
            </Tooltip>
          )}
        </div>

        {/* Process Summary - Click to expand */}
        <Popover
          content={processListContent}
          title={
            <span style={{ color: TEXT_PRIMARY }}>
              {t('gpuCard.process.title', { id: gpu.id })}
            </span>
          }
          trigger="click"
          placement="bottom"
          overlayStyle={{ maxWidth: 360 }}
        >
          <div
            style={{
              marginTop: 12,
              padding: '8px 10px',
              background: BG_ELEVATED,
              border: `1px solid ${BORDER_PRIMARY}`,
              borderRadius: 6,
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>
              <UnorderedListOutlined style={{ marginRight: 6 }} />
              {t('gpuCard.process.count', { count: processes.length })}
            </Text>
            <Text style={{ color: TEXT_PRIMARY, fontSize: 12 }}>
              {totalProcessMemory > 0 ? `${totalProcessMemory} MB` : '-'}
            </Text>
          </div>
        </Popover>
      </Card>
    </Col>
  )
}

export { getUsageColor }
